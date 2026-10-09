# The plugin on the 2026-10-09 vLLM nightly (8cbd5d03, 0.31.1rc1.dev173, ROCm 10.0 SDK)

Brian: "there is a new vLLM, want to look at it and test against our stack?" then "our stack is supposed to
work with any vLLM, so let's fix it so it does." Our pin was the 2026-09-23 nightly (e9757321), 320 commits
behind v0.31.0 and 360 behind this nightly, on the same ROCm 10.0 pip SDK.

## What "the new vLLM" is, and is not

- **The v0.31.0 release image** (`vllm/vllm-openai-rocm:v0.31.0`, 2026-10-04) is **ROCm 7.2.3 + torch 2.13**
  (`0.31.0+rocm723`, classic /opt/rocm layout). It is not the platform this stack targets (ROCm >= 10, the pip
  SDK nightlies): libr9k does not compile under its older clang (`r9k_attn.hip:89` / `:205`, the
  `global_load_tr` builtins' signatures differ) and the hostcall-free RCCL overlay built in the nightly does not
  match ("NCCL error: unhandled cuda error"). Not pursued.
- **No ROCm 10.1 vLLM images exist** (Docker Hub: `nightly-rocm100-*` and `base-nightly-rocm100-*` only, pip
  `rocm-sdk-core 10.0.0` inside). ROCm 10.1 waits for upstream's images.
- **The plain nightly has no plugin.** Our serving image (`docker/Dockerfile`) is the nightly plus an editable
  install of the plugin, which is what registers the entry points. The first "test" on the bare nightly ran
  stock vLLM without us; its two errors (the CUSTOM attention backend unregistered, no compressed-tensors scheme
  for the block-fp8 layers) were stock's. Build the image first: `docker build -f docker/Dockerfile --build-arg
  BASE=<nightly> -t r9700/vllm:dev-<sha> .`, then `overlay/emulated-switch/patch-hostcall.sh` for that image.

## What broke, and the fixes (all in the plugin, the kernels unchanged)

| symptom on the new nightly | cause | fix |
|---|---|---|
| `vLLM's ple_layer no longer exposes PLEVocabParallelEmbedding` at construction | upstream moved the PLE table to `models/qwen4_exp/common/ple.py` and split it into `Qwen4ExpPLEDeviceEmbedding` / `Qwen4ExpPLEPinnedHostEmbedding` (own pinned-host path, bf16 / fp8 tables, no int6) | `ple/int6.py` implements the new `allocate_embedding_weight` hook (the int6 rows in pinned host memory behind a UVA view) on the device base, keeps the meta-build path for the old base; `construction_scope` swaps whichever class names the module exposes |
| `get_supported_kernel_block_sizes() takes 0 positional arguments but 1 was given` | the attention backend API passes the KV cache spec now | the method takes `kv_cache_spec=None` |
| 27B decode 46 tok/s, drafter acceptance 1.03 tokens a step (was 4.32), outputs correct | **upstream bug**: the draft quant config now reaches DFlash2's grouped-conv kernel projections (they were built with `quant_config=None`); `_add_global_draft_layer_exclusions` aliases checkpoint-local names (`layers.0...`) to global ones (`layers.64...`) only on `exclude_modules`, and `Fp8Config` keeps `ignored_layers` (from `modules_to_not_convert`, exact match) -- so the bf16 kernel projections were loaded into fp8 parameters ("Attempted to load weight layers.0.attention_conv.kernel_projection.weight with dtype torch.bfloat16 into parameter with dtype torch.float8_e4m3fn") | `models/dflash.py alias_draft_exclusions()` adds the global aliases (both prefix forms) to the draft hf config before construction; reproduced with every plugin model hook off (`R9K_DISABLE=models`), so a stock user hits it too -- worth an upstream issue |

The MTP allowlist patch (`spec/mtp_rocm.py`) still applies and is still needed (PR #55292 closed unmerged);
`8cbd5d030` added to its tested list. The off-by-default runner patches (`moe_shared_fold`, `layer_tail_pipe`)
were not exercised on this vLLM and keep their tested lists (the gate warns if they are switched on).

## Numbers (test box, 210 W, the same probes as the v0.3.0 record)

| | old pin (e9757321) | new nightly (8cbd5d03) |
|---|---|---|
| unit gates in the image | all pass | all pass (prefill_4bit, moe_mxfp4, gemm_fp8, gdn_merge, fp8_prefill_r9k, moe_route_r9k, moe_sum_r9k) |
| 27B TP2: dec / c8 / c16 / 8k prefill, tok per step | 207.6-212.8 / 585 / 647-672 / 3,946, 4.32 | **216.0 / 524 / 639 / 3,906, 4.48**; sanity 0 bad |
| Flash-Next TP2, experts in host RAM | 118.7 / 183 / 150 / 2,608, 3.22 | 117.1 / 154 / 154 / 2,626, 3.08; sanity 0 bad; **KV 84,898 vs 133,306 tokens** (under investigation: consumed 25.13 GiB, activation 2.25, graphs 1.43) |
| Flash-Next TP4 (0.98, graphs to 256, hc fp8) | 199 / 560-613 / 888-905 / 6,850-7,070, 3.16; KV 442-475k | **209.3** / 451 / 917 / 6,854, 3.22; sanity 0 bad; **KV 411,881** (consumed 21.6 GiB, activation 2.38, graphs 1.66) |

The KV cache is sized smaller by the new vLLM on both Flash-Next configurations although its weights take
less (21.6 vs 22.0-22.6 GiB a card at TP4): with 0.98 x 31.86 - 21.6 - 2.38 = 7.2 GiB nominally free the old
formula gave ~475k tokens and the new one 411,881, i.e. vLLM's sizing now reserves more (presumably the graph
memory that used to overflow the budget -- the honest accounting we measured by hand on 2026-10-07). Not a
plugin matter; `KVMEM=` still pins the budget explicitly. Single-stream decode is 5% faster on the new nightly
at TP4 and 2-4% on the 27B; the 8-stream probe at TP4 read 451 once and 585 on the repeat (209.0 / 585 / 906 / 6,883), i.e. inside the old band.

Tooling on the test box: `~/try031b.sh LABEL 27b|fn2|fn4 [knobs]` launches one config on the new-nightly image
(`PORT` / `NAME` env for parallel runs), prints the plugin's lines, the probe, sanity at concurrency and on long
prompts; `~/try031-*.server.log` keep the container logs.
