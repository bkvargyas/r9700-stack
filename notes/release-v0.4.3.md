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
  `UTIL=0.94`. With the knob the gate measured **284,829 tokens** and a clean 15-minute soak (the figure is above the old estimate's 203k because the estimate, not the memory, was the limit). `serve/serve.sh` already takes `KVMEM=`; the README recipe and the production launcher carry it.
- **Fixes the move turned up, none caused by 0.33** (each reproduces on the previous pin): `kernels/build.sh`'s export check read
  `nm | grep -q` under `pipefail`, so present symbols were reported missing at random (it reads the list once now);
  `kernels/r9k_dsv4_glue.hip`'s rotation pre-pass rounded `h * 1/sqrt(128)` twice where exllamav3's binary rounds once (1 f16 ulp on
  near-ties; now bit-equal, and the test asserts it; the pre-pass has no serving caller); `serve/ds41-exl3.sh` waits for READY while
  the container lives (`READY_WAIT`, 3600 s) instead of a fixed 20 minutes, which had misread a cold-page-cache load (21 minutes for
  the 196 GiB pack) as a failed self-test; `r9700_vllm/compat/gate.py` lists `0.33.0` as tested for `mtp_allowlist`,
  `glm5next_rdna4` and `moe_shared_fold` (12 + 16 "untested patch" warnings a launch before; a release image reports the plain
  version string); `tests/test_exl3_linear.py`'s cold-capture negative control names its real first error (exllamav3's
  `exl3_gemv_try_launch` allocates under a torch capture at 9+ rows; the plugin never sends it more than 8).
- **Also in this release (PR #2, merged 2026-11-07): startup calibration of the TP4 collective cutoffs** (`comm/calibrate.py`;
  notes/root-ports.md). On a box where every card sits on its own CPU root port the fixed PLX-switch cutoffs put 8-16-stream
  decode on a two-shot path 4-9x slower than RCCL; at TP4 launch the communicator now graph-times the paths on every rank and
  the slowest rank decides (`R9K_ARN_CALIBRATE=0` keeps the fixed cutoffs; explicit knobs are respected). The compressed
  all-reduce keeps its original 512 KiB floor (`ar4_min`). TP2 and TP8 are untouched. Gated on the root-port box in the PR
  (Flash-Next TP4 probe 191 -> 207, 16 streams 466 -> 995-1,015) and re-gated on VM100 below.
- **Not taken: the nightly line.** `nightly-rocm100-1648fefab9` (2026-11-07, `0.33.0rc1.dev992`, aiter 0.1.28.post1) touches five
  of our hook sites and the DeepSeek launch dies on it (the router hook's target `fused_topk_bias_router._get_padding_mask` is gone).
  The next release is the next pin candidate.

## Speed and validation (the tagged tree on the new image; v0.4.2 on the previous pin in parentheses)

Gate `~/gate-043.sh` on VM100, 2026-11-08 04:57-08:09 UTC: the four configurations from the candidate tree (`git archive` of the tag)
on the plugin image built from the new `BASE`, overlay on, nothing else on the box.

- **Qwen3.8-Flash-Next TP4 (cards 0-3, MTP-3):** KV cache 472,025 tokens (479,076); probe 222.9 tok/s mean (222.0), 3.11 tokens
  a step, 469 / 912 tok/s at 8 / 16 streams (466 / 909), prefill 6,743 tok/s on 8k; BetterBench combined **185.3** (184.8), update
  p99 15.2 ms; strict sanity 0 bad of 2,460; soak (16 clients, 200-30k, 909 s) 371 requests, 0 errors (424), 6,697 prompt tok/s.
  0 server errors, 0 untested-patch warnings.
- **Qwen3.8-27B-NVFP4 TP2 (cards 0,1, DFlash2):** KV cache 367,494 tokens; probe 217.1 mean (220.4), 4.45 tokens a step, 514 / 655 at
  8 / 16 streams (512 / 607), prefill 3,822; BetterBench combined **197.7** (195.5), update p99 24.3 ms; strict sanity 0 bad of
  2,460; soak 184 requests in 950 s, 0 errors (199), 3,069 prompt tok/s. 0 server errors.
- **GLM-5.3-Flash TP8 (32k production recipe with `KVMEM=2.65`):** KV cache **284,829 tokens** (203,329 on the previous pin's
  utilization estimate; 155,437 on 0.33's estimate without the knob); decode-only **97.5 tok/s** median (95.1), four streams 173
  then **231** aggregate (176 / 221); 12.7k needle correct at 128.2 tok/s at depth (118.7) and 90 % acceptance; strict sanity
  (thinking on) 0 bad of 780; soak at **8 clients** (see Known limits), 937 s: 166 requests, 0 errors, 2,600 prompt tok/s. 0 server
  errors. The worst-card headroom was not sampled for GLM in this gate; the soak passed.

**Re-gate after PR #2 merged** (`~/gate-043b.sh`, 08:18-10:05 UTC; the first gate ran on the candidate before the TP4 collective
calibration landed on master). The PR touches the TP4 N-rank collective paths and launcher comments only, so the four-card and
two-card configurations were re-gated in full and the eight-card ones re-launched with their quick checks on the merged tree:

- **Flash-Next TP4:** the calibration picked, on this PLX box, one-shot <= 5 KiB, two-shot <= 480 KiB, RCCL above, compressed from
  512 KiB, r9k all-gather <= 128 KiB (the fixed cutoffs were 16 KiB / 512 KiB / 128 KiB). KV cache 475,758 tokens; probe 221.3
  mean, 3.08 tokens a step, 477 / 928 at 8 / 16 streams, prefill 6,777; BetterBench combined **181.9** (185.3 three hours earlier
  with the fixed cutoffs, 184.8 on v0.4.2; the run-to-run band of this benchmark is ~2 %), update p99 15.2 ms; strict sanity
  0 bad of 2,460; soak (16 clients, 903 s) 443 requests, 0 errors, 6,542 prompt tok/s. 0 server errors.
- **27B TP2:** probe 219.0 mean, 4.51 tokens a step, 523 / 636 at 8 / 16 streams; BetterBench **197.5**; strict sanity 0 bad of
  2,460; soak 191 requests in 954 s, 0 errors, 3,072 prompt tok/s. 0 server errors.
- **GLM TP8 (quick):** KV cache 284,829 tokens, decode-only 92.7 tok/s (the usual band), four streams 160 then 236, 12.7k needle
  correct at 119.1 tok/s and 81 % acceptance. 0 server errors.
- **DeepSeek TP8 (quick):** launch self-test prose 1.70 / code 3.70, KV cache 387,069 tokens. 0 server errors.

**DeepSeek-V4.1-Flash TP8, NSEQ 4, the launcher defaults** (`~/v033/v043-ds41.validate.log`, 06:58-07:46 UTC): KV cache 387,069
tokens (same). Launch self-test prose 1.75 / code 3.79. TTFT 1.28 s on 1.5k (1,205 tok/s), 3.42 s on 12.3k (3,599 tok/s); the
4 x 12k burst leaves 0.57 GiB free on the worst card (0.52). spectune per-stream decode prose **108.6-109.0 tok/s** (MAL 1.83-1.84;
v0.4.2 103.5), four streams 51.7-63.5 (138-229 aggregate; 199-221), code **219.9-221.7** (4.45-4.46; 214.5), four streams
112.5-114.2 (413-421; 422), GSM8K-style 165.7 / 194 aggregate. divergence.py 24 / 24 and 22 / 24 (the third prompt's line
was not captured by the gate's filter). Needles 3 / 3 at 8k, 16k, 28k. Strict sanity 0 bad of 25 at 5 streams and 0 of 80 at 8.
Soak (8 clients, 200-28k, 929 s): 187 requests, 0 errors, median prompt 15.1k, 2,959 prompt tok/s, 0.57 GiB free throughout.
GSM8K 200: **195 / 200 = 97.5 %** (194). 0 server errors, 0 untested-patch warnings.

**NSEQ 8 on the same tree** (`~/v033/v043-ds41-n8.log`, 07:46-08:09 UTC): launch self-test prose 1.73 / code 3.29; spectune
1 / 4 / 8 streams prose 103.7 / 51.5 x 4 (192) / 45.5 x 8 (**313**; v0.4.2 317), code 211.7 / 111.9 x 4 (421) / 73.9 x 8 (**529**;
555), GSM8K-style 155.3 / 84.7 x 4 / 66.8 x 8 (199). Strict sanity 0 bad of 45 at 9 streams and 0 of 80 at 16. Soak (8 clients,
200-28k, 603 s): 124 requests, 0 errors, 3,179 prompt tok/s. Self-test after the soak unchanged. 0 server errors.

## Known limits

- Unchanged from v0.4.2: the drafter Heisenbug is guarded by the launch self-test, not fixed; the 128k recipe sits at the edge of
  the 0.5 GiB rule; the expert tier at 6 rows is 128.5 us; the hub all-reduce is opt-in.
- **The 120 V feed:** a 16-client eight-card soak reached 2,960 W at the wall on 2026-11-07 (BMC reading; the two supplies deliver
  ~2,400 W at 120 V) and parked the host CPU until a power cycle. Until the 208/240 V feed, the GLM gate soak runs 8 clients; an
  eight-card production load should be capped the same way.
- GLM's KV budget on this vLLM must be pinned with `KVMEM` (the utilization estimate loses a quarter of the tokens); the serving
  rotations in `r9k_exl3_gemv.hip` and `r9k_moe_exl3.hip` still round twice (1 f16 ulp on near-ties against exllamav3's binary); a
  change needs a DeepSeek acceptance run.
