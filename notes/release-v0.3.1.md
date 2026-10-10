# r9700-stack v0.3.1

**Flash-Next on four cards: the decode step 16.4 -> 15.0 ms, single-stream decode 199 -> 217 tok/s, same
quality, same KV cache, same prefill.** The code of v0.3.0 plus the decode-step round of 2026-10-09
(`notes/decode-step.md`) and the 2026-10 vLLM nightly compatibility fixes (`notes/vllm-upgrade-2026-10.md`).

## What changed for someone running it

- `serve/flashnext.sh` turns on two new paths by default (the first at TP4 only):
  - **fp8 hyper-connection decode GEMMs** (`R9K_HC_FP8_DECODE=1`, `kernels/r9k_hc_f8.hip`): the down+hc_silu and
    up+gated-mean GEMMs at decode widths (up to 16 tokens) read the fp8 fragment-order copies the prefill mix path
    already holds, half the bytes of the bf16 kernels that were the step's largest stream (1.29 GB a step), with
    bf16 WMMA against the unquantized activations. 16.42 -> 15.33 ms a step. `=0` restores the bf16 kernels. At
    TP2 (experts in host RAM) it stays off: the fp8 down copy costs 0.43 GiB of KV a card there (166,818 -> 140,008
    tokens, warm cache) for +10% single-stream decode and nothing at 8 streams; `R9K_HC_FP8_DECODE=1` opts in.
  - **fused activation quant in the dense fp8 block GEMM** (`R9K_FP8_QA=1`, `r9k_gemm_fp8_block_qa`): the
    per-token-group-128 quant runs inside the GEMM's A-load with bit-identical operands; 96 launches fewer a step.
    15.33 -> 15.01 ms. `=0` splits it again.
- The plugin also runs on the 2026-10-09 vLLM ROCm 10 nightly (`nightly-rocm100-8cbd5d03`, vLLM 0.31.1rc1) with
  the same kernels; the image pin in `docker/Dockerfile` and the overlay stay on the validated September nightly.
  That nightly has an upstream bug in the DFlash draft model's fp8 exclusions ([vllm#61003](https://github.com/vllm-project/vllm/issues/61003),
  `notes/upstream-issue-dflash-exclusions.md`) which the plugin works around.
- Measured and left off: a 32 KB one-shot all-reduce cap (+0.35 ms a step), split-K across blocks for the hc
  down kernel, the fused routing + fold (-0.13 ms, two vLLM-runner patches). The decode all-reduce itself is at
  its protocol floor on this PCIe topology (fence and block-count sweeps in `notes/decode-step.md`).
- `serve/serve.sh`: `PROF=1 PROFACT=CUDA` records a GPU-only torch profile. `tests/test_hc_f8.py`,
  `tests/test_fp8_qa.py` (exact against their references, graph-timed benches); `tests/test_ar_nrank.py` takes `NBS=`.

## Numbers (test box, 4x R9700, 210 W and -42 mV, probe.py unless noted)

| Flash-Next TP4 | v0.3.0 | v0.3.1 |
|---|---:|---:|
| decode step, 512-token essay (`~/tp4tune.sh`) | 16.42-16.44 ms (five runs) | **15.01 ms** |
| single-stream decode | 199 tok/s | **217 tok/s** |
| 8 / 16 concurrent | 547-561 / 871-889 | 567 / 898 |
| 8k prefill | 6,750-6,870 | 6,743 |
| KV cache | 442-475k tokens | unchanged |

## Quality gate (2026-10-10, conc 1, no-think, both configs on this tree)

| | GSM8K 1,319 | HumanEval | strict sanity |
|---|---:|---:|---|
| v0.3.0 defaults | 95.75% (1,263) | 160 / 164 | 0 bad of 1,040 |
| v0.3.1 defaults | **95.83%** (1,264) | **161 / 164** | 0 bad of 1,040 |

Paired per question: 7 vs 8 discordant, McNemar p = 1.00; outputs identical on 25.5% of questions (the fp8 hc
decode rounds the hyper-connection GEMMs' bf16 bits differently in every layer). No detectable difference.

## Validation record (`~/validate-031.sh` on the release tree, 2026-10-10, `~/validate-031.log`)

**Flash-Next TP4 (the changed configuration).** Probe on launch: 217.2 tok/s single-stream (3.10 tokens a step),
581 / 924 tok/s at 8 / 16 streams, 6,870 prompt tok/s at 8k, KV cache 452,115 tokens. Full 20-pass BetterBench
(`r9700-fn4-v031-210w`): **combined decode 177.0 tok/s** (v0.3.0 record 160.6-162.3; radiance 1.3.0 on this box
221.9), update p99 15.6 ms, per-category decode 139.6 chat / 171.0 code / 195.4 file_edit / 222.2 json / 219.9
math / 155.0 prose / 146.8 reasoning / 201.9 summarization, TTFT p50 68-82 ms; concurrency 1 / 2 / 4 / 8:
**168.1 / 266.7 / 392.0 / 495.0** aggregate tok/s (v0.3.0: 152-155 / 234-238 / 349-352 / 487-508); prefill
5,210 / 7,176 / 7,522 / 7,243 tok/s at 2k / 8k / 16k / 32k (v0.3.0: 7,129-7,182 at 8k).
Mixed-length soak, 16 clients, 1,509 s: 664 requests ok, 0 errors, 2 refused by the server for exceeding the
32,768-token context (prompts up to 32,060 tokens plus the requested output), 6,689 prompt tok/s, VRAM flat at
31,868-31,904 of 32,624 MiB a card throughout. Strict sanity under overload: 0 bad of 1,700 / 1,440 / 1,920 at
17 / 24 / 32 clients. The long-prompt sanity in the chain died on its first calibration request (a 6,000-character
prompt, 2 output tokens) with an HTTP 400 the tool did not print; on a fresh server from the same tree it then
passed 43 rounds (3 + 40, 0 bad of 344), as did the 27B and TP2 chains' 40 rounds each. Not reproduced; the tool
now prints the server's reason on a refusal (`bench/sanity_stress.py`). Server errors 0. The two MES timeouts in dmesg are from 19:46 and 20:09 UTC the day before (the
radiance runs), none during this validation.

**27B-NVFP4 TP2.** Probe: 212.4 tok/s single-stream (4.31 tokens a step), 582 / 662 at 8 / 16 streams, 3,953
prompt tok/s at 8k, KV 367,494 tokens. Soak 16 clients, 1,530 s: 354 ok, 0 rejections, 0 errors, 3,108 prompt
tok/s, VRAM 32,573 -> 29,341 MiB. Strict sanity: 0 bad of 320 after long prompts, 0 of 900 / 720 / 960 at
9 / 12 / 16 clients. Server errors 0.

**Flash-Next TP2 (experts in host RAM).** Probe: 124.3 tok/s single-stream (3.22 tokens a step; v0.3.0 118.7),
164 / 158 at 8 / 16 streams, 2,610 prompt tok/s at 8k, KV 103,517 tokens. Soak 16 clients, 1,579 s: 251 ok,
0 rejections, 0 errors, 2,295 prompt tok/s, VRAM 30,375 -> 30,863 MiB. Strict sanity: 0 bad of 320 after long
prompts, 0 of 900 / 720 / 960 at 9 / 12 / 16 clients. Server errors 0. This chain ran with the fp8 hc decode ON
at TP2; with it off (the shipped TP2 default) the same tree gives KV 133,306 (v0.3.0's number), 113.4 tok/s
single-stream, 184 / 165 at 8 / 16 streams, 2,597 at 8k, so the default stays off there. Warm-cache launches,
on / off / on: KV 140,008 / 166,818 / 140,008 tokens (2.26 / 2.69 GiB): the fp8 decode path costs 0.43 GiB a card
at TP2, where the hyper-connection weights are not sharded four ways. (KV figures from a launch with a cold
torch.compile cache run 20-30% lower -- the chain's 103,517 was one -- because vLLM's profiling pass counts the
compile transients as activation; compare warm with warm.)

**Flash-Next TP2 on the shipped defaults** (fp8 hc decode off, fused quant on): KV 166,818 tokens; probe 120.1
tok/s single-stream, 184 / 157 at 8 / 16 streams, 2,583 at 8k. Soak 16 clients, 1,572 s: 248 ok, 0 rejections,
0 errors, 2,415 prompt tok/s. Strict sanity: 0 bad of 320 after long prompts, 0 of 540 / 720 / 960 at 9 / 12 / 16
clients. Server errors 0.

**Comm tests.** 4-rank all-gather and all-reduce (one-shot + two-shot, graph replay interleaved) bit-exact, 2-rank race test ALL OK.

**Kernel gates (single GPU, in the image):** atiled_4bit, fold_mxfp4, prefill_4bit, tuned_cfgs, moe_mxfp4, nvfp4,
cache_moe, gemm_fp8, gdn_merge ALL OK; moe_route_r9k, moe_sum_r9k, fp8_prefill_r9k PASS; the two tests added this
release on the same tree (copy, card 4): test_hc_f8 PASS, test_fp8_qa PASS, plus test_hc_mix_r9k and test_router_r9k PASS.
