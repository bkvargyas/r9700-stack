# v0.4.1 -- the DeepSeek-V4.1-Flash step, measured and cut down (2026-11-03)

Same pin as v0.4.0 (`nightly-rocm100-e4a5b553`, vLLM 0.32 line, torch 2.13, ROCm 10.0), same image. Brian: "We should beat it by 2x"
(the two DGX Sparks on TensorFold: 80.7 tok/s prose / 118 code single stream, 154-166 at four streams). This release is the first
pass of that program: a per-kernel ledger of the serving step (notes/exl3-ledger.md), then four kernel packages built against it
(notes/exl3-plan.md "Phase 4"). Measured on 8x R9700 in VM100 (210 W / -42 mV, eight-card runs at the -500 MHz offset).

## Changed

- **The eight-rank all-gather** (`comm/r9k_ar8.py`, `kernels/r9k_ar.hip`): the ledger's largest single item. v0.4.0 routed gathers
  whose eight-way result exceeded 128 KB but whose piece did not onto the flat eight-rank kernel, where every rank stores its whole
  piece into seven peers across the root complexes: 0.4-3.9 ms a call at the shapes the step uses (the target logits at the
  adaptive-trimmed 1-4 verify rows, Engram's two gathers) -- 4.1 ms of a 25.4 ms step. Now a one-kernel all-gather that never
  crosses a root complex (own piece to the host pool and the pair partner; the quartet's pieces pulled from host memory and
  forwarded; the partner quartet's four copied over P2P): 16-129 us, beating RCCL to six rows, RCCL above (`R9K_AG8_OS_KB` 192).
  The flat kernel is never used (`R9K_AG8_FLAT_KB` 0).
- **A one-kernel eight-rank all-reduce for messages up to 64 KiB** (the same three phases as the stage kernels, system-scope flags,
  host pools; bit-identical rounding): 14.6 / 17.3 / 23.5 / 27.2 / 33.8 us at 8 / 16 / 32 / 40 / 60 KiB against 21 / 34.5 / 46 /
  34.7 / 37.6, RCCL 104-118. `R9K_AR8_ONESHOT=0` restores the stage kernels. The MoE combine (pick weights x the expert tier's
  per-slot partials + the shared expert) is formed inside the collective's first phase (`moe/exl3_combine.py`, `R9K_EXL3_MOE_AR=0`
  to disable). GLM-5.3-Flash TP8's small collectives take the same kernels (same bits; its gate below).
- **The decode glue, one launch each** (`kernels/r9k_dsv4_glue.hip`, `dsv4_glue.py`, hooks in `models/deepseek_v41_rdna4.py`): the
  FFN front of every layer (ffn_norm, the router gate GEMV, sqrt-softplus top-k with its bias and scaling, the shared expert's
  gate_up and silu) as one kernel for up to 8 rows (40 -> 21 us a layer at one row; stock above 8); the DSpark drafter's markov
  head (five greedy draft steps 720 -> 245 us); the attention input of the compressor / indexer layers (29 -> 13 us). Within one
  bf16 ulp of stock's norms, top-k bit-exact, drafts token-identical. Knobs `R9K_DSV41_ROUTE / _ATTN_IN / _MARKOV / _QB_STACK`.
- **Our EXL3 decode GEMV for 1-48 rows** (`kernels/r9k_exl3_gemv.hip`, `quant/exl3_gemv.py`; exllamav3's trellis device functions,
  MIT): up to 16 stacked projections a launch, the input rotation in-kernel, no casts, WMMA row tiles from 9 rows -- the attention
  linears 26 -> 19 us at one row and **120 -> 44 us at 24 rows**, where v0.4.0 dequantized every matrix every step (7.6 ms of a
  four-stream step). `wo_a` is back on EXL3 through an output-projection hook (`R9K_DSV41_WOA=exl3|bf16`): -230 MiB a rank, 18 -> 12
  us at one row, slower above 16 rows. EXL3 is the default: on the release tree it is 90.8 vs 87.8-89.0 tok/s single-stream prose, 197 vs 190.5 code, and within noise at four streams (99.9-100.5 vs 101.5-102.5 code).
- The fused split-path all-reduce runs one lane with 1024-word sub-steps: 240 KiB 81.7 -> 77.2 us, 480 KiB 140 -> 122 (RCCL 124 / 161).
- `bench/ds41/spectune.py` takes `STREAMS=1,4,8`; `bench/ds41/ledger/` holds the step-ledger tools.

## Held back

- A 48-token expert-tier kernel (eight-stream steps): correct and 1.5-3x faster on one card, but in the server it leaves the DSpark
  drafter accepting nothing at the 48-row cap and costs ~6 ms a step at the 1-4-row widths the single-stream step runs
  (notes/exl3-plan.md). Reverted to v0.4.0's tier; NSEQ 8 stays unsupported on this release (exllamav3's fused tier faults in the
  warm-up above 24 tokens).

## Speed (DeepSeek-V4.1-Flash, eight cards, the v0.4.0 launcher defaults; v0.4.0 in parentheses, two DGX Sparks in brackets)

Clean tree = git archive of the tag, VM100, `~/ds41/exl3/D4/clean-REL2.log`, 2026-11-03 19:13-19:46 UTC; spectune, greedy, best of 2 reps.

| | single stream | per stream x 4 (aggregate) |
|---|---:|---:|
| prose, adaptive verification on (the default) | **82-91 tok/s** (73) [81] | **47-58** (186-230) (44-46; 167) [154] |
| prose, adaptive off | **91** (75) | 43-48 (38) |
| code, adaptive on | **185-187** (152) [118] | **99-102** (360-375) (81; 297) [166] |
| code, adaptive off | **197** (158) | 100 (368-372) (84) |
| plain decode, no drafter | 83 (52) | 53-60 (32) |

Acceptance lengths unchanged (prose 1.7-2.0, code 4.1-4.4). Against the two Sparks: single stream 1.1x prose / 1.6x code, four streams
1.2-1.5x prose / 2.2x code. The step ledger said where the rest is (notes/exl3-ledger.md); the 48-token tier and the GEMV
bandwidth are v0.4.2's.

## Validation

**DeepSeek-V4.1-Flash TP8, the launcher defaults on the tagged tree** (MAXLEN 32768, KVMEM 1.7, DSpark k = 5 with adaptive
verification, NSEQ 4; VM100 `~/ds41/exl3/S/validate-41.log`, 2026-11-03 22:02-22:34 UTC): spectune per-stream decode prose
**86.8-96.7** single / 48.1-57.2 at four (179-212 aggregate), code **201.7-202.6** / 99.7-99.9 (368-369), GSM8K-style 144.7 /
71.7-83.8. Prompt processing 3,563 tok/s on one 12k prompt, 3,333 aggregate on four. divergence.py 22 / 24 / 22 (the misses are
near-ties: |dlogprob| 0.06-0.15 between the two candidates). Needles 3 / 3 at 8k, 16k and 28k. Strict sanity 0 bad of 25 at 5
streams and 0 of 80 at 8, twice. Soak (8 clients, 200-28k-token prompts, 926 s): 180 requests, 0 errors, median prompt 14.3k
tokens, 2,901 prompt tok/s, worst card 0.86 GiB free (v0.4.0: 0.69). GSM8K 200: **193 / 200 = 96.5 %** (v0.4.0's runs: 194, 195,
196; within the noise of 200 problems). 0 server errors.

**The other three configurations on the same tree and image** (`~/gate-041.sh` -> `~/gate041.log`, 19:47-21:48 UTC; v0.4.0's gate
in parentheses):

- **Qwen3.8-27B-NVFP4 TP2:** BetterBench combined **196.4** (195.7), update p99 24.5 ms; strict sanity 0 bad of 2,460; soak 220
  requests in 945 s, 0 errors, 3,040 prompt tok/s. 0 server errors.
- **Qwen3.8-Flash-Next TP4:** KV cache 477,417 tokens; BetterBench combined **185.3** (189.7 / v0.3.8 184.6), update p99 15.1 ms;
  strict sanity 0 bad of 2,460; soak 404 requests in 904 s, 0 errors, 6,516 prompt tok/s. 0 server errors.
- **GLM-5.3-Flash TP8 (32k production recipe; its 1-row logits gather and its small all-reduces now take the one-kernel
  collectives):** decode-only **93.2 tok/s** median (v0.4.0's gate 88.1), four streams 172 then **234** tok/s aggregate (139 /
  177), 12.7k needle correct at 123.5 tok/s at depth (117) and 83 % acceptance; strict sanity (thinking on) 0 bad of 780; soak 170
  requests in 973 s, 0 errors (159). 0 server errors. KV cache 176,443 tokens (210,051: the one-kernel collectives' pools take
  ~0.2 GiB a card that vLLM's profile now leaves to the cache).

## Known limits

Unchanged from v0.4.0 (32k and 128k recipes, the capture-time KV budget, GLM above ~75k on this nightly) plus: NSEQ above 4 is not
supported for DeepSeek on this release (see Held back); the GEMV runs at 285-340 GB/s on the attention shapes (a ~9 us fixed
latency chain a launch is the next target); the split-path all-reduce at 240-480 KiB is host-read bound (~3 GB/s) and needs a data
path with fewer host reads.
