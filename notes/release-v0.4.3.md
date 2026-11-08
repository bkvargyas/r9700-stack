# v0.4.3 -- the pin moves to vLLM 0.33.0 (2026-11-08)

Brian (2026-11-07): "Take a look at vllm .33 and see how it works in our stack." The evaluation is [vllm-0.33.md](vllm-0.33.md); this
release is its outcome. **The stock image is now the official `vllm/vllm-openai-rocm:v0.33.0-rocm100`**, the first vLLM release
shipped as a ROCm 10 image (v0.32.0 only existed for ROCm 7.2, which is why v0.3.7-v0.4.2 pinned the 2026-10-25 nightly). No
kernel or serving change; the plugin that passed the v0.4.2 gate passes the same gate on the new image. Measured on 8x R9700 in
VM100 (210 W / -42 mV, eight-card runs at the -500 MHz offset).

## Changed

- **The pin** (`docker/Dockerfile` `ARG BASE`): `vllm/vllm-openai-rocm:v0.33.0-rocm100` = vLLM **0.33.0** (tag commit `4ba787ec9`,
  branched 2026-10-26 at `4135590ffe`: our previous pin `e4a5b553` plus 95 main commits and 12 cherry-picks). The same runtime as
  the pin it replaces: torch 2.13.0+rocm10.0.0, triton 3.8.0, aiter 0.1.26, flash_attn 2.8.3, rocm-sdk-core 10.0.0 (TheRock build
  `16adc4d8`), HIP 7.15.26333; `libamdhip64` and `librccl` byte-identical, so the emulated-switch overlay's hostcall-free RCCL is
  reused and only the two vLLM extensions were re-patched. exllamav3 v1.6.6 builds unchanged (`EXL3_REF` stays). The image builds
  in 11 minutes (53.8 GB). The 0.33 release-note delta does not touch a plugin hook (ROCM_OCTAVE and the MXFP4 indexer default are
  CDNA4-only; Engram's peer-HBM tables apply to GPU tables with DP > 1), and **no serving fix was needed** (the 0.32 move took five).
- **GLM-5.3-Flash 32k recipe: `KVMEM=2.65`** (a fixed KV budget a rank, the previous pin's measured figure). On 0.33 the profile
  run peaks ~0.15 GiB higher; with the recipe's 163,840-token lcm blocks that rounds the cache from 203,329 to 155,437 tokens at
  `UTIL=0.94`. __GLMKV__ `serve/serve.sh` already takes `KVMEM=`; the README recipe and the production launcher carry it.
- **Fixes the move turned up, none caused by 0.33** (each reproduces on the previous pin): `kernels/build.sh`'s export check read
  `nm | grep -q` under `pipefail`, so present symbols were reported missing at random (it reads the list once now);
  `kernels/r9k_dsv4_glue.hip`'s rotation pre-pass rounded `h * 1/sqrt(128)` twice where exllamav3's binary rounds once (1 f16 ulp on
  near-ties; now bit-equal, and the test asserts it; the pre-pass has no serving caller); `serve/ds41-exl3.sh` waits for READY while
  the container lives (`READY_WAIT`, 3600 s) instead of a fixed 20 minutes, which had misread a cold-page-cache load (21 minutes for
  the 196 GiB pack) as a failed self-test; `r9700_vllm/compat/gate.py` lists `0.33.0` as tested for `mtp_allowlist`,
  `glm5next_rdna4` and `moe_shared_fold` (12 + 16 "untested patch" warnings a launch before; a release image reports the plain
  version string); `tests/test_exl3_linear.py`'s cold-capture negative control names its real first error (exllamav3's
  `exl3_gemv_try_launch` allocates under a torch capture at 9+ rows; the plugin never sends it more than 8).
- **Not taken: the nightly line.** `nightly-rocm100-1648fefab9` (2026-11-07, `0.33.0rc1.dev992`, aiter 0.1.28.post1) touches five
  of our hook sites and the DeepSeek launch dies on it (the router hook's target `fused_topk_bias_router._get_padding_mask` is gone).
  The next release is the next pin candidate.

## Speed and validation (the tagged tree on the new image; v0.4.2 on the previous pin in parentheses)

__GATE43__

__DS43__

__N8__

## Known limits

- Unchanged from v0.4.2: the drafter Heisenbug is guarded by the launch self-test, not fixed; the 128k recipe sits at the edge of
  the 0.5 GiB rule; the expert tier at 6 rows is 128.5 us; the hub all-reduce is opt-in.
- **The 120 V feed:** a 16-client eight-card soak reached 2,960 W at the wall on 2026-11-07 (BMC reading; the two supplies deliver
  ~2,400 W at 120 V) and parked the host CPU until a power cycle. Until the 208/240 V feed, the GLM gate soak runs 8 clients; an
  eight-card production load should be capped the same way.
- GLM's KV budget on this vLLM must be pinned with `KVMEM` (the utilization estimate loses a quarter of the tokens); the serving
  rotations in `r9k_exl3_gemv.hip` and `r9k_moe_exl3.hip` still round twice (1 f16 ulp on near-ties against exllamav3's binary); a
  change needs a DeepSeek acceptance run.
