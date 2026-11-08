# The plugin on vLLM v0.33.0 (the ROCm 10 release image `v0.33.0-rocm100`)

Brian (2026-11-07): "Take a look at vllm .33 and see how it works in our stack." The production pin is the 2026-10-25 nightly
`e4a5b553` (0.32.1rc1.dev466, torch 2.13, ROCm 10.0); the current release v0.4.2 runs on it. Same method as
[vllm-0.32.md](vllm-0.32.md) and [vllm-upgrade-2026-10.md](vllm-upgrade-2026-10.md): build the plugin image on the new base,
regenerate the VM100 overlay, run the integration gate and the kernel gates, then the four serving configurations against the
v0.4.2 gate numbers ([release-v0.4.2.md](release-v0.4.2.md)).

## What "0.33" is on ROCm

- **For the first time there is an exact release on ROCm 10:** `vllm/vllm-openai-rocm:v0.33.0-rocm100` (2026-11-02; also tagged
  `v0.33.0`, the ROCm 10 image is now the default ROCm image, ROCm 7.2 moved to `-rocm72`, #58761). Version string `0.33.0`.
- **It is very close to our pin.** The GitHub release (title v0.33.0, tag `v0.33.0rc3` = `v0.33.0` = commit `4ba787ec9`; there is
  no commit 3a4d82f5) branched at `4135590ffe` on 2026-10-26, **one day and 95 main commits after `e4a5b553`**, and took 12
  cherry-picks (HiSparse, Humming, CI, a gfx950 GEMM guard, MTP in the Transformers backend). The "927 commits" of the release
  notes are counted from v0.32.0, most of which our pin already carries.
- **Same runtime as the pin:** torch `2.13.0+rocm10.0.0` (the notes' "PyTorch 2.14" is the CUDA build; the ROCm image keeps
  2.13), triton 3.8.0, aiter 0.1.26, flash_attn 2.8.3, torchvision 0.28.0, `rocm-sdk-core 10.0.0` from the same TheRock build
  (`the_rock_commit 16adc4d8`, run 32424980240), HIP 7.15.26333; `libamdhip64.so.7` and `librccl.so.1` are byte-identical to
  the pin's (md5 35e93d24 / a3963038). So the overlay's hostcall-free RCCL needs no rebuild.
- The newest ROCm 10 nightly, `nightly-rocm100-1648fefab9` (2026-11-07, `0.33.0rc1.dev992+g1648fefab`): same torch / triton / SDK /
  HIP / RCCL, **aiter 0.1.28.post1**, ~1,060 commits past the branch point.

## Release-note delta against what the plugin hooks

The code between the pin and the release is 95 main commits plus the 12 picks (`gh api .../compare/e4a5b553...4ba787ec9`; the
release notes describe all of 0.32.0 -> 0.33.0). Diffing the Python trees of the two images (`~/v033/src-old`, `src-new`, 164
files changed) against the 72 vLLM modules the plugin imports, the ones that touch our hooks:

| area | change (PR) | what it means for us |
|---|---|---|
| ROCm platform | `ROCM_OCTAVE` 3-4 bit KV backend appended to the non-MLA priority list after `TRITON_ATTN` / `TURBOQUANT` (#59774); `_maybe_default_dsv4_mxfp4_indexer` (MXFP4 indexer K cache for DeepSeek-V4, #63669) | gated on `get_cdna_version() == 4` and aiter on: not taken on gfx1201. Our `R9700Platform` (`get_device_communicator_cls`, `get_mla_prefill_backend_cls`) is unchanged and still applies. |
| platform plugins | out-of-tree detection failures now log a warning instead of debug (#63601: PIN_MEMORY eager, no platform resolution during `import vllm`) | nothing to change; our `detect()` does not raise. |
| attention | `TRITON_ATTN` sets `ptr_range()` on ROCm KV views for 32-bit buffer addressing (#60084); `attention.py` no longer re-quantizes an fp8 query; `customize_spec` runs under `set_current_vllm_config` | our CUSTOM backend pins `supports_quant_query_input = False` (0.32 fix 4) and keeps its own kernels: not affected. The integration gate's draft-metadata and kernel-block-remap checks still hold. |
| Flash-Next (Qwen4-Exp) QSA | Octave cache dtypes for the QSA main cache, a `customize_spec` that passes non-Octave specs through, `forward_qsa` signature unchanged | our `forward_qsa` rebinding (attn/qsa.py) still applies; BF16 cache as before. |
| DeepSeek-V4.1 | Engram GPU tables read from peers' HBM (#62091) and meta-init for Engram offload (#63068), both only for GPU tables with DP > 1 / meta builds; FP8 shared-expert FSE check (#63099); `krdy` ordering in index-source launches (#63576, AMD MonoKernel) | we run `cpu_offload` Engram at TP8 + EP: the CPU-offload path is unchanged apart from a `weight.is_meta` guard. |
| GLM-5.3 | FP8 indexer WK pairs kept across `load_weights` calls (#59223) | load path only. |
| speculative decoding | speculator `max_model_len` follows auto-fit (#60513); adaptive-verification cost profiling skipped in the graph memory-profiling pass (#63431); lookahead block hashes for EAGLE/MTP prefix caching on by default (#61589) | our launches run prefix caching off; no draft-metadata API change. |
| MoE | `post_gemm2_hook` / `route_major_moe_output` added to the modular kernel; hash-routing kwargs refactored | additive; our MoE methods do not use them. |
| sparse MLA | `idx_mapping` (HiSparse) variant of the sparse-utils kernels | additive. |

Not in v0.33.0 (they are on main after the branch point and in the 11-07 nightly): AITER 0.1.28, QSA top-k buffer shared across
layers (#65619), the Qwen4-Exp HyperConnection up/gate fusion (#64898), PLE shard copy via `copy_weight_` (#65692), Engram gate
sharding (#63367), the DeepSeek-V4.1 mHC gate overlap (#59802), DSpark sliding-window MLA draft layers (#63795).

## Image and overlay

- `r9700/vllm:dev-v0330` from `docker/Dockerfile` with `BASE=vllm/vllm-openai-rocm:v0.33.0-rocm100`: **683 s, 53.8 GB**.
  **exllamav3 v1.6.6 builds unchanged** (same torch), `exllamav3_ext ok`; libr9k builds unchanged; the plugin installs as
  `r9700-vllm 0.4.2` and registers under `vllm.general_plugins`.
- Overlay: the hostcall-free RCCL (`~/rccl10/.../librccl.so.1.0`) is reused (same SDK build, byte-identical `librccl.so.1` and HIP
  runtime). `patch-hostcall.sh` -> `~/p2p-patched-v0330`: `_C_stable_libtorch.abi3.so` 0 hostcall kernels after the patch;
  `_rocm_C.abi3.so` 782 still requesting hostcall, all gfx9-only (`paged_attention_ll4mi_QKV_mfma4` 768, `wvSplitKrc` 14), as on
  the pin.

## Gates (card 4 of VM100, `~/gates033.sh`, tree = branch v033)

The gates032 list plus the seven single-card tests v0.4 added (dsv4_decode, dsv4_glue, exl3_gemv, exl3_linear, exl3_moe,
exl3_moe_r9k, mhc_r9k), `BENCH=0`. `gates033.sh` now prints a per-test verdict from the full log: gates4.sh keeps each test's
last 25 lines, which hid the first FAIL below.

- First run (`~/v033/gates-v0330.log`): `tests/test_integration.py` **ALL OK** (24 checks, the int6-PLE scope section skips as on
  the pin); 40 of 42 tests pass; `test_dsv4_glue.py` SOME FAILED; `test_attn_r4d.py` no verdict.
- After the fixes below (`~/v033/gates-v0330-run2.log`): **41 of 42 pass**. `test_attn_r4d.py` needs `kernels/r4d.so`, which
  the gate image does not build (pre-existing, not a 0.33 matter).

### Fixes (branch v033): what broke, why, and the fix

1. **`test_dsv4_glue.py`: "rotated input K=5120 / K=1280 M=1 vs exl3_gemm A_had" (1 element, 1 f16 ulp).** Identical on the
   production image (not a 0.33 matter). An op-for-op f32 emulation of exllamav3 v1.6.6's `had_hf_r_128_inner` matches our
   kernel, not exllamav3's output, and every mismatch is a near-tie (the exact value just past the f16 midpoint): exllamav3 is
   compiled with HIP's default `-ffp-contract=fast`, where `__floats2half2_rn(h * r_scale)` becomes a mixed-precision multiply
   with an f16 result (one rounding); libr9k builds with `-ffp-contract=off` and rounds twice (f32, then f16). **Fix**
   (`kernels/r9k_dsv4_glue.hip` `f2h_mul`, the rotated store): the f32 x f32 product is exact in f64 and rounded once to f16.
   Now 0 ulp at M = 1, 2, 6, 8 against both of exllamav3's rotations; the test asserts bit-equality at every M (it allowed
   2 ulp above M = 1). The suh pre-pass has no serving caller (package D's interface), so no served output changes.
2. **`test_exl3_linear.py` "[graph-cold] cold capture at M=32 failed: hipErrorStreamCaptureInvalidated".** An informational
   negative control (the test reports "0 failed"; same on the production image). The first error, from a rocgdb breakpoint on
   `hipMalloc` and `AMD_LOG_LEVEL=3`: `exl3_gemm -> exl3_gemv_try_launch -> exl3_gemv_graph_param_block(device,
   allow_alloc=true) -> hipMalloc(4128) = hipErrorStreamCaptureUnsupported`. exllamav3 1.6.6 sends 1..8 rows to its multi-row
   GEMV (tag 92); only 9+ rows reach `exl3_gemv_try_launch`, whose first line pre-warms the graph-GEMV parameter block on the
   assumption that "no exllamav3 `Graph`" means "not capturing", which a torch stream capture breaks. The failed launch and
   `EndCapture` are the reported symptom. With the block pre-warmed by an eager 16-row call, the next blocker is the cooperative
   GEMM autotuner's `hipStreamSynchronize` on the capture stream on a shape's first call. **Not a plugin defect:** `apply()`
   never sends more than 8 rows to one `exl3_gemm` and `_warm` runs every row count eagerly before capture; the test's
   docstring blamed the autotuner and now states the traced chain.
3. **`kernels/build.sh`: "MISSING EXPORT: r9k_ar_ipc_open"** in the image build on the nightly 1648 (the symbol was present).
   `set -o pipefail` with `nm -D | grep -q`: grep exits on its first match, nm takes SIGPIPE (141), the pipeline fails. Latent
   since the export check was written. **Fix:** the export list is read once and each name grepped against it.

4. **`serve/ds41-exl3.sh`: "drafter self-test FAILED (Connection refused)" on a still-loading server** (found in serving,
   6cbc5a2). The WARM and SELFTEST waits were a fixed 120 x 10 s; a cold-page-cache DeepSeek launch takes 35-45 minutes.
   **Fix:** `wait_ready()` polls `/v1/models` while the container lives (`READY_WAIT`, default 3600 s), exits 2 if it goes away.
5. **`compat/gate.py`: twelve "untested on vLLM 0.33.0" warnings on the Flash-Next launch** (f820bc2): `mtp_allowlist` and
   `glm5next_rdna4`, six each (one per process that imports the plugin). A release image reports the plain
   version string (`0.33.0`, no `+g<sha>`), so the TESTED tuples need `"0.33.0"` itself; both are now listed (validated by
   Flash-Next and GLM here). A nightly reports `+g<sha>`: the 1648 run needs `"1648fefab"` added after its gate. The DeepSeek
   launch also warned for `moe_shared_fold` (tested `e97573215`); its list stays until DeepSeek passes on 0.33.

## Serving (VM100, `~/gate033.sh CONFIG`, image `r9700/vllm:dev-v0330`, overlay on, REPO = `~/r9700-build/repo-033` = branch v033)

The v0.4.2 gate for each configuration, one at a time under the lock (210 W cards; GLM and DeepSeek at the -500 MHz offset).
v0.4.2's numbers are from `notes/release-v0.4.2.md` (same scripts, the pinned image). No configuration needed a fix.

**Qwen3.8-Flash-Next TP4** (cards 0-3; `~/v033/v0330-fn4.*`)

| | v0.33.0 | v0.4.2 (e4a5b553) |
|---|---:|---:|
| probe decode mean (code / prose / json / math) | **225.4** (224.1 / 161.9 / 264.2 / 251.3) | 222.0 (218 / 160 / 262 / 249) |
| tokens per step (MTP) | 3.11 | 3.11 |
| 8 / 16 streams | 476 / 938 | 466 / 909 |
| prefill 8k | 6,953 tok/s | 6,556 |
| BetterBench combined (update p99) | **185.7** (15.0 ms) | 184.8 (15.2 ms) |
| strict sanity | 0 bad of 2,460 | 0 bad of 2,460 |
| soak (16 clients, 200-30k, 900 s) | 454 requests, 0 errors, 6,668 prompt tok/s | 424, 0 errors, 6,483 |
| KV cache | 472,439 tokens | 479,076 |

**Qwen3.8-27B-NVFP4 TP2** (cards 0,1; `~/v033/v0330-27b.*`)

| | v0.33.0 | v0.4.2 |
|---|---:|---:|
| probe decode mean (code / prose / json / math) | **218.3** (236.2 / 107.6 / 275.3 / 253.9) | 220.4 (236 / 111 / 281 / 254) |
| tokens per step (DFlash2 drafter) | 4.49 | 4.64 |
| 8 / 16 streams | 532 / 646 | 512 / 607 |
| prefill 8k | 3,968 tok/s | 3,874 |
| BetterBench combined (update p99) | **197.9** (24.3 ms) | 195.5 (24.7 ms) |
| strict sanity | 0 bad of 2,460 | 0 bad of 2,460 |
| soak | 251 requests in 948 s, 0 errors, 3,082 prompt tok/s | 199 in 961 s, 0 errors, 3,050 |

The 0.32 move's draft-metadata fix (`R9kTriton3DMetadataBuilder`) still holds: the drafter accepts (4.49 a step, not 1.0).

**GLM-5.3-Flash TP8** (32k production recipe; `~/v033/v0330-glm.*`)

| | v0.33.0 | v0.4.2 |
|---|---:|---:|
| decode-only median (512 tokens x 3) | **96.6 tok/s** | 95.1 |
| four streams, aggregate (first / again) | 178 / **229.5** | 176 / 221 |
| 12.7k needle | correct, ttft 5.0 s, prefill 2,534 tok/s, 124.2 tok/s at depth, 82 % acceptance | correct, 118.7, 83 % |
| strict sanity (thinking on) | 0 bad of 780 | 0 bad of 780 |
| soak | 183 requests in 1,019 s, 0 errors | 168 in 978 s, 0 errors |
| KV cache at 0.94 | **155,437 tokens** (2.5 GiB available) | 203,329 (2.65 GiB) |

The GLM KV cache is the one regression. The KV layout is identical (groups 32768 x 3 / 1280 / 8, lcm block 163,840 tokens, the
same mamba padding) and the CUDA-graph estimate is unchanged (0.94 = 0.9229 without it on both); the profile run's peak is
~0.15 GiB higher on 0.33, and with blocks this coarse 0.15 GiB costs a quarter of the tokens. A candidate is #57926 (the sampler
is profiled with the parameters the warm-up uses). Not a plugin matter; `KVMEM=` pins the budget explicitly (see the open items).

**DeepSeek-V4.1-Flash TP8: **not run: host CPU parked (SCMP trip during the GLM soak, 22:38-22:55 UTC); rerun after the power cycle with `bash ~/gate033.sh ds41`, then `LBL=1648 IMG=r9700/vllm:dev-1648 REPO=~/r9700-build/repo-033n`**.** Two launches (`~/v033/v0330-ds41-try1-cold.*`, `~/v033/v0330-ds41.*`):

- The first launch read the 196 GiB pack and the Engram tables from a cold page cache (the GLM gate had just run): ~21 minutes
  of Engram / pre-weight reads, then 45 shards at ~24 s each. `serve/ds41-exl3.sh`'s WARM / SELFTEST waits were a fixed 20
  minutes, so the launcher reported "drafter self-test FAILED (Connection refused)" against a server that was still loading, and
  the gate wrapper's own 25-minute wait then removed the container. **Fix (6cbc5a2):** `wait_ready()` polls `/v1/models` for as long
  as the container lives (`READY_WAIT`, default 3600 s) and exits 2 if it goes away.
- The relaunch exposed the real problem: **the .100 host's CPU was parked by the SCMP power-limit trip** (the 120 V feed; see the
  host notes): a 20M-iteration Python loop takes 62-66 s on VM100, the host and the mgmt VM (~1.5 s normally); host cores at
  400 MHz; the libr9k compile took 3.6 minutes at 20:20 and more than 12 at 23:45. GLM's decode, needle and sanity numbers
  (22:26-22:38) are normal and the build right after the GLM gate (22:55) was already slow, so the trip happened in the GLM gate's
  eight-card soak (22:38-22:55; 16 clients, prompts to 30k tokens, at the -500 MHz offset). The GLM soak counts as a stability
  pass (183 requests, 0 errors), not as a throughput number. The DeepSeek launch was stopped and nothing further was measured;
  the trip clears only with a power cycle of the host.

Nothing in the static delta touches the DeepSeek path we serve (Engram's changes are for GPU tables with DP > 1; the MXFP4 indexer
default is CDNA4-only; the MonoKernel / MegaMoE work is gfx94x / gfx950), and the integration gate and the DeepSeek-side unit
tests (dsv4_decode, dsv4_glue, exl3_gemv / linear / moe / moe_r9k, mhc) pass on the image, but the protocol's rule stands: a
configuration is validated only by the eight-card server run (validate-42.sh scope, self-test on).

## Newest nightly (`nightly-rocm100-1648fefab9`, 2026-11-07)

`r9700/vllm:dev-1648` is built (exllamav3 1.6.6 and libr9k build unchanged; **36.8 GB**, a slimmer base than the release's
53.8; the first build attempt hit the `build.sh` export-check race, fixed above), overlay pair `~/p2p-patched-1648` (38 CCOB
bundles in `_rocm_C`, 768 gfx9-only hostcall kernels, `_C_stable_libtorch` 0). Gates and the DeepSeek launch: **not run: host CPU parked (SCMP trip during the GLM soak, 22:38-22:55 UTC); rerun after the power cycle with `bash ~/gate033.sh ds41`, then `LBL=1648 IMG=r9700/vllm:dev-1648 REPO=~/r9700-build/repo-033n`**. The line has moved into code we hook since the branch point, so expect plugin work when the pin moves past the
release: aiter 0.1.28.post1, the QSA top-k buffer shared across layers (#65619; our `forward_qsa` reads `layer.topk_indices_buffer`),
the Qwen4-Exp HyperConnection up-projection / gate-mix fusion (#64898; our hc hooks), PLE shard copy through `copy_weight_`
(#65692; our int6 loader), Engram gate sharding (#63367) and the DeepSeek-V4.1 mHC gate overlap with attention / FFN (#59802; our
mHC kernels), DSpark sliding-window MLA draft layers (#63795).

## Verdict

**So far: three of four configurations at the v0.4.2 numbers with no serving fix; DeepSeek pending.**

- **v0.33.0-rocm100 is the first exact vLLM release on the platform we target**, and it is a small step from our pin: 95 main
  commits + 12 picks, the same torch / triton / aiter / SDK build / HIP runtime / RCCL. The image builds from `docker/Dockerfile`
  with only `BASE` changed, exllamav3 v1.6.6 builds unchanged (EXL3_REF stays), the overlay's RCCL is reused.
- The integration gate and 41 of 42 kernel gates pass (the 42nd needs libr4d's `r4d.so`, as on the pin). Three plugin-side fixes
  came out of the gates (build-script race, a 1-ulp rotation rounding in a non-serving pre-pass, a mis-attributed negative
  control), one out of serving (the launcher's fixed readiness wait), and the compat allowlist gained `0.33.0`. **None of the four fixes is caused by 0.33**; all reproduce on the
  pinned image. **No vLLM-0.33-specific breakage was found** -- unlike the 0.32 move (five fixes).
- **Flash-Next TP4, 27B TP2 and GLM-5.3-Flash TP8 serve at or slightly above the v0.4.2 numbers** (BetterBench 185.7 / 197.9;
  GLM decode 96.6) with 0 bad answers and 0 soak errors. The one regression: **GLM's KV cache at 0.94 shrinks 203k -> 155k tokens**
  (a ~0.15 GiB larger profile peak against coarse 163,840-token blocks).
- **DeepSeek-V4.1-Flash TP8 -- the configuration v0.4.x is about -- is not validated**: the host's CPU was parked by the power-limit
  trip before its run.

## Recommendation: wait -- move the pin to `v0.33.0-rocm100` once DeepSeek passes on it

Reasons: (1) an exact release on ROCm 10 is the pin we want (reproducible, named, documented, no nightly churn), and nothing in
three of four configurations argues against it; (2) but the protocol requires the eight-card DeepSeek validation before a pin
moves, and it has not run; (3) the GLM KV regression should be measured with `KVMEM=2.65` (the pin's budget) before the move, so
the 32k recipe either keeps its ~203k tokens or documents the new figure. Do not chase the nightly line: it is ~1,060 commits
further on and touches five of our hook sites; take the next release instead.

Steps after the host's power cycle: `ssh devops@192.168.0.123 'bash ~/gate033.sh ds41'` (launcher defaults, self-test, the
validate-42 scope via `~/v033/validate-033.sh`; ~1.5 h with a warm page cache -- run it first, or pre-read the pack); if it is
level with v0.4.2 (prose 103.5 / code 214.5 single stream, 199-221 / 422 at four, divergence 23-24 / 24, needles 3 / 3, GSM8K
97.0 %), set `docker/Dockerfile` `ARG BASE=vllm/vllm-openai-rocm:v0.33.0-rocm100`, merge v033 and gate the release as usual.

## Open items

- DeepSeek-V4.1-Flash TP8 on v0.33.0 and the nightly 1648 pass: **not run: host CPU parked (SCMP trip during the GLM soak, 22:38-22:55 UTC); rerun after the power cycle with `bash ~/gate033.sh ds41`, then `LBL=1648 IMG=r9700/vllm:dev-1648 REPO=~/r9700-build/repo-033n`**.
- **GLM-5.3-Flash KV cache, for the pin move: 155,437 tokens on 0.33 vs 203,329 on the pin** (UTIL 0.94, 32k recipe). Same
  KV layout (groups 32768 x 3 / 1280 / 8, mamba page padded 16.79 %, lcm block 163,840 tokens) and the same CUDA-graph estimate;
  "Available KV cache memory" is 2.5 GiB a rank instead of 2.65 because the profile run peaks ~0.15 GiB higher (candidate:
  #57926, the sampler profiled with the warm-up's parameters), and with blocks this coarse 0.15 GiB is a quarter of the tokens.
  **Knob: `KVMEM=2.65`** in the GLM 32k launch (the pin's measured budget, a fixed GiB a rank that bypasses the profile
  estimate; equivalently `UTIL=0.945`, +0.16 GiB of a 31.86 GiB card). I would set `KVMEM=2.65` and confirm >= 0.5 GiB free on
  the worst card through the four-stream burst and the 15-minute soak before writing it into the recipe; if it does not hold,
  keep UTIL 0.94 and document 155k.
- The serving rotations round the rotated store twice where exllamav3 rounds once: `kernels/r9k_exl3_gemv.hip` (`rotate_*`, the
  `*= RS128` before `st_h4`) and `kernels/r9k_moe_exl3.hip` (the input pre-pass and the expert-tier rotations); 1 f16 ulp on
  near-ties (~1 in 15k rotated inputs). Moving them to `f2h_mul`-style single rounding makes them bit-identical to exllamav3's
  rotation but changes DeepSeek's outputs, so it needs a DeepSeek server acceptance run (protocol: spectune MAL, divergence,
  GSM8K) before it lands.

## Reproduce (VM100)

| what | where |
|---|---|
| images | `r9700/vllm:dev-v0330` (v0.33.0-rocm100 + plugin, 53.8 GB), `r9700/vllm:dev-1648` (nightly 1648fefab9, 36.8 GB) |
| overlay pairs | `~/p2p-patched-v0330`, `~/p2p-patched-1648`; RCCL `~/rccl10/.../build-nightly/librccl.so.1.0` (unchanged) |
| trees | `~/r9700-build/repo-033` (branch v033, served), `~/r9700-build/repo-033n` (nightly build tree), `~/r9700-build/repo-033t` (card-4 test tree); the old v0.3.3-candidate `repo-033` moved to `repo-033-old-v0.3.3rc` |
| build | `~/build-release.sh TAG LABEL` (full tag, e.g. `v0.33.0-rocm100 v0330`; `REPO=` the tree) -> `~/v033/build-release-LABEL.log` |
| gates | `~/gates033.sh` (IMG / REPO / LBL; verdict table at the end) -> `~/v033/gates-v0330.log`, `gates-v0330-run2.log`, `*.full.log`; one test with full output: `~/v033/one-test.sh IMG TEST [card]` |
| serving | `~/gate033.sh fn4\|27b\|glm\|ds41` (IMG / PATCHED / REPO / LBL, takes the lock, offset for 8-card) -> `~/v033/LBL-CONFIG.{gate,launch,server,bb,validate}.log`; DeepSeek validation `~/v033/validate-033.sh` |
| diagnosis | `~/v033/colddiag*.py`, `exl3lin-gdb.log`, `exl3lin-trace.log` (the cold-capture trace); `~/v033/rotdiag*.py` (the rotation rounding); `~/v033/src-old`, `src-new` (the two images' vLLM Python trees) |
