# The plugin on the vLLM 0.33.0 release (2026-11-07)

Brian: "Ok, now pull down vllm .33 and recompile and test again". The pin moves from the 0.32-era nightly (e4a5b553,
0.32.1rc1.dev466) to the **0.33.0 release image**, `vllm/vllm-openai-rocm:v0.33.0-rocm100`
(`sha256:4fcd39b758b660c4e9c345c021e2ea8f0ccdb31e3c32507d676cc3005ae83fba`, published 2026-11-02): a tagged
release instead of a nightly, same torch 2.13.0+rocm10.0.0 (HIP 7.15.26333) as before. Measured on gpusrv1 (the
root-port box, notes/root-ports.md); **not yet on VM100** (the PLX box and its overlay).

## Build

`docker build -f docker/Dockerfile --build-arg BASE=<the pin> -t r9700/vllm:v033 .`: exllamav3 v1.6.6's extension,
libr9k and the plugin (installs as `r9700-vllm 0.4.2`) build unchanged; the `vllm.general_plugins` entry point
registers. 12 minutes on 32 cores.

## Gates (gpusrv1, `~/gate033.sh`, tree = master c7191ec + the MTP chunk fix)

- `tests/test_integration.py`: **ALL OK** (the int6 PLE scope section skips; MLA prefill raises on gfx1201 and the
  sparse layer falls back to top-k MQA, as on 0.32).
- Kernel gates (`bench/bringup-tests.sh`, single GPU): test_moe_mxfp4, test_ple_int6, test_cache_moe,
  test_gemm_fp8, test_fp8_qa, test_hc_f8, test_moe_fused: **all pass**.
- `tests/test_ar_nrank.py` (4 ranks): **PASS**; latencies within 1 us of 0.32 (5 KB one-shot 6.2 us, 20 KB two-shot
  23.6, RCCL 35-41 us to 160 KB). The launch calibration (comm/calibrate.py) picks the same cutoffs as on 0.32:
  one-shot <= 20 KiB, two-shot off, RCCL above; r9k all-gather <= 8 KiB.

## The one break: the MTP conversion chunk is now required at TP4

The first 0.33 launch of Flash-Next TP4 died in `speculator.load_model`: `HIPCachingAllocator ... OOM on device 2
while trying to allocate 2684354560 bytes (free: 452984832)`, the 2^18-row bf16 -> MXFP4 search chunk of the MTP head
(models/qwen4_exp.py). 0.32 had ~2 GB more headroom at that point, so only TP2 with offloaded experts hit it there;
on 0.33 TP4 does too. The 2^15-row chunk (same bits) is in this move.

## Serving (Flash-Next TP4, MTP-3, 210 W / -42 mV, prefix caching, `PREFIX_CACHE=1`, no hand-set cutoffs)

| | 0.32 nightly (root-ports.md gate) | **0.33.0** |
|---|---:|---:|
| probe mean (code / prose / json / math) | 207.0 (217 / 140 / 245 / 227) | **207.3** (211 / 146 / 242 / 230) |
| step (single stream) | 15.9 ms | 15.8-16.0 ms |
| 4 / 8 / 16 streams (`bench/bench.py`) | 461-465 / 703-742 / 995-1,015 | 465 / 722-724 / 928-980 |
| prefill 8k / 31k / 124k (`bench/prefill.py`) | 8.2k / 7.9k / 6.9k | 8.2k / 7.9k / 6.9k |
| session TTFT, 61-66k tokens, turns 1-3 | 0.57-0.58 s | 0.49-0.57 s |
| KV cache (util 0.97, MAXLEN 262144) | 592,236 tokens | 551,628 tokens |

Strict sanity **0 bad of 340** at 17 concurrent (`--pre "" --conc 17 --rounds 20`) and **0 bad of 80** at 16; a
10-minute 8-client soak at 18-32k-token prompts: **172 ok, 0 errors**, 7,166 prompt tok/s; 0 server errors. Claude
Code over the Anthropic `/v1/messages` endpoint (auto mode, tool calls) works unchanged.

Read with care: the first `bench.py` pass after every launch reads low at 4 streams (289 here, 298 on 0.32) and the
repeats do not; the 16-stream numbers may sit 3-7% under 0.32's but are within the run-to-run spread of two runs.
The KV pool is 7% smaller at the same utilization (0.33 holds a little more per card); still 2.1 full-length
requests.

## Not done in this move

- **VM100 / the PLX box**: the `emulated-switch` overlay is per image (`overlay/README.md`): rerun
  `rccl/build-nightly.sh` and `patch-hostcall.sh` against the new image there, then the same gates and the three
  serving configurations (Flash-Next TP2 + offload, 27B TP2, GLM / DeepSeek TP8).
- The 27B, GLM-5.3 and DeepSeek-V4.1 launchers were not run on 0.33; only Flash-Next TP4.
- MES firmware 0x93 logged one more soft `INVALIDATE_TLBS` timeout during the 0.33 load (8 on gpusrv1 in all).
