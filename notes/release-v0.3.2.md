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

TBD.
