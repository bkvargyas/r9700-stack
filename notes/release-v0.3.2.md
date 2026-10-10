# r9700-stack v0.3.2

**The hyper-connection weights run from their fp8 copies at every width and the bf16 copies are gone: KV cache
+14% on four cards (452k -> 516k tokens) and +33% on two (167k -> 221k), 8 streams +5.6% and 16 streams +3.4% at
TP4, same quality, single-stream and prefill unchanged within 1%.** The code of v0.3.1 plus round 3 of
`notes/decode-step.md`.

## What changed for someone running it

- `serve/flashnext.sh` at TP2 and TP4: the fp8 hyper-connection kernels (`kernels/r9k_hc_f8.hip`) now take any
  row count and are used to 255 rows (`R9K_HC_FP8_DECODE_MAXM=255`); above that the tiled fp8 GEMMs run both the
  up mix (as before) and the down GEMM (`R9K_HC_FP8_DOWN=1`, -0.8% at 8k prefill for its activation quant); with
  every width on fp8, the bf16 up / down weights are freed after quantisation (`R9K_HC_FREE_BF16=1`: 1.3 GB a
  card). The fp8 decode path is on at TP2 as well now: the freed pair outweighs the down copy that kept it off in
  v0.3.1. `R9K_HC_FREE_BF16=0` restores the v0.3.1 memory layout, `R9K_HC_FP8_DECODE_MAXM=16` its dispatch.
- The 27B is unchanged (no hyper-connections).

## Numbers (test box, 210 W and -42 mV, probe.py, warm compile cache)

| Flash-Next TP4 | v0.3.1 | v0.3.2 |
|---|---:|---:|
| KV cache | 452,115 tokens | **515,577** |
| single-stream decode | 217-220 tok/s | 219 |
| 8 / 16 streams | 550-560 / 904-907 | **582 / 935** |
| 8k prefill | 6,785-6,883 | 6,729 |
| 390-token prefill | 1,660-1,760 | 1,691 |

| Flash-Next TP2, experts in host RAM | v0.3.1 | v0.3.2 |
|---|---:|---:|
| KV cache | 166,818 tokens | **221,184** |
| single-stream decode | 120.1 tok/s | 122.5 |
| 8 / 16 streams | 185 / 153 (+-6% run to run) | 174 / 163 |
| 8k prefill | 2,589-2,603 | 2,594 |

## Quality gate (2026-10-10, conc 1, both configs on this tree)

| | GSM8K 1,319 | HumanEval | strict sanity |
|---|---:|---:|---|
| v0.3.1 defaults | 95.83% (1,264) | 159 / 164 | 0 bad of 1,040 |
| v0.3.2 defaults | 95.75% (1,263) | 162 / 164 | 0 bad of 1,040 |

Paired per question: 11 vs 10 discordant, McNemar p = 1.00; outputs identical on 18.5% of questions. No detectable
difference.

## Validation record (`~/validate-032.sh` on the release tree, 2026-10-10, `~/validate-032.log`)

**Flash-Next TP4.** Probe on launch: 218.8 tok/s single-stream (3.19 tokens a step), 589 / 933 at 8 / 16 streams,
6,753 prompt tok/s at 8k, KV cache 515,577 tokens. Full 20-pass BetterBench (`r9700-fn4-v032-210w`): combined
decode 176.8 tok/s (v0.3.1: 177.0), update p99 15.7 ms; concurrency 1 / 2 / 4 / 8: **172.3 / 261.1 / 403.1 /
514.1** aggregate tok/s (v0.3.1: 168.1 / 266.7 / 392.0 / 495.0); prefill 5,189 / 7,076 / 7,388 / 7,137 tok/s at
2k / 8k / 16k / 32k (v0.3.1: 5,210 / 7,176 / 7,522 / 7,243: the tiled fp8 down GEMM's activation quant, -1 to -2%).
Mixed-length soak, 16 clients, 1,507 s: 792 requests ok (v0.3.1: 664), 0 rejections, 0 errors, 6,414 prompt tok/s,
134.6 output tok/s, VRAM flat at 31,951-31,987 of 32,624 MiB a card throughout (v0.3.1: 31,866-31,902; the extra
is the KV cache). Strict sanity: 0 bad of 320 after long prompts (the v0.3.1 chain's one-off HTTP 400 did not
recur), 0 of 1,700 / 1,440 / 1,920 at 17 / 24 / 32 clients. Server errors 0; the two MES timeouts in dmesg are
the 2026-10-09 ones.

**27B-NVFP4 TP2** (unchanged code path). Probe: 210.6 tok/s single-stream (4.30 tokens a step), 582 / 657 at
8 / 16 streams, 3,948 prompt tok/s at 8k, KV 367,494 tokens. Soak 16 clients, 1,559 s: 292 ok, 0 errors, 10
refused by the server for exceeding the 32,768-token context (this run's prompts reached 32,333 tokens), 3,147
prompt tok/s, VRAM 32,428 -> 32,496 MiB. Strict sanity: 0 bad of 320 after long prompts, 0 of 900 at 9 clients,
0 of 720 / 960 at 12 / 16 clients. Server errors 0.

**Flash-Next TP2 (experts in host RAM; the fp8 decode path and the freed copies are new here).** Probe: 122.5
tok/s single-stream (3.13 tokens a step), 177 / 163 at 8 / 16 streams, 2,587 prompt tok/s at 8k, **KV 221,184
tokens (v0.3.1: 166,818)**. Soak 16 clients, 1,557 s: 244 ok, 0 rejections, 0 errors, 2,420 prompt tok/s, VRAM flat at 31,407-31,429 MiB. Strict sanity: 0 bad of 320 after long prompts, 0 of 900 / 720 / 960 at 9 / 12 / 16 clients. Server errors 0.

**Comm tests.** 4-rank all-gather and all-reduce (one-shot + two-shot, graph replay interleaved) bit-exact, 2-rank race test ALL OK.

**Kernel gates (single GPU, in the image):** atiled_4bit, fold_mxfp4, prefill_4bit, tuned_cfgs, moe_mxfp4, nvfp4,
cache_moe, gemm_fp8, gdn_merge ALL OK; moe_route_r9k, moe_sum_r9k, fp8_prefill_r9k PASS; on a copy of the same tree
(card 4): test_hc_f8 (exact to 300 rows), test_fp8_qa, test_hc_mix_r9k, test_router_r9k PASS. Chain finished
22:43 UTC; everything green.
