# Decode step time: fp8 hyper-connection decode, fused activation quant, and what a graph node costs (2026-10-09)

Brian: "ok, lets see if we can get our step time down then." Context: radiance 1.3.0 at TP4 Flash-Next runs a decode
step in 12.1 ms against our 16.4 (`notes/radiance-test-run.md`), at the same acceptance; the whole single-stream
gap is the step. This note is the second decode-step round after `notes/decode-nodes.md` (2026-10-07, launch
fusions: 0.15 ms). Same measurement: Flash-Next TP4, MTP-3, 210 W, the test box, `~/tp4tune.sh` (512-token essay
for the step time, `probe.py` for decode / 8 conc / 16 conc / 8k prefill); the repeatability of the step number is
0.00-0.02 ms between runs of the same tree (k1-base 16.42, k1-base2 16.42, k2-base3 16.42, k3-base4 16.44, k4-base5 16.42).

## Where the step goes (v0.3.0 graph profile, `~/tprof-v030-decode`, `~/dec-attr-v030-all.txt`)

1,766 kernels inside the step, 11.4 ms of kernel time, 16.4 ms step. The kernel time by family (eager sums):

| ms/step | launches | what |
|---:|---:|---|
| 3.48 | 194 | hyper-connection down + up GEMMs on **bf16** weights, 1.29 GB/step (`r9k_router_gemm`, `r9k_hc_up_mix`) |
| 2.65 | 192 | routed MoE gate_up + down (`r9k_moe_mxfp4a8`), near HBM bandwidth for the experts touched |
| 1.90 | 192 | dense fp8 block linears: `r9k_quant_group128_fp8` (0.23) + `r9k_gemm_fp8_block` (1.67) |
| 1.25 | 98 | 4-rank two-shot all-reduce (graph-timed; the eager trace shows it spinning on peers) |
| 1.40 | 384 | MoE glue: topk (0.43), align+sort (0.30), quant (0.12), silu+quant (0.14), sum (0.19), copy+add (0.22) |
| 0.82 | 192 | shared expert (4 launches a layer; vLLM runs it on an aux stream, overlapped with routing) |
| 0.75 | 72 | GDN spec-verify core + conv update |
| 0.29 | 95 | `hc_combine_norm` (stock Triton at decode widths) |

## What a graph node costs on this ROCm (measured, `~/graph-bubble.py`, `~/graph-fork.py`, card 4)

A replayed HIP graph of N tiny dependent kernels (one 256-element add each) runs at **2.9-3.1 us per kernel**
(N = 100..2000), against 10.7 us eager. Two independent 500-kernel branches forked inside one graph replay in
1.70 ms against 2.85 ms serial: **forked branches do run concurrently**; bandwidth-bound branches gain nothing
(2 x 50 adds on 64 MB: 22.4 vs 22.0 ms). So a tiny node is worth ~3 us of step whatever it does, and the only ways
to get it back are to remove it or to put it on a branch beside something else. The 2026-10-07 result (240 launches
removed, 0.15 ms) says most of our tiny nodes already overlap their neighbours' tails; fatter kernels and fewer
bytes are what move the step.

## What was built

1. **fp8 hyper-connection decode GEMMs** (`kernels/r9k_hc_f8.hip`, `hc.py` `down_f8` / `up_mix_f8`,
   `R9K_HC_FP8_DECODE=1`): the decode-width (M <= 8) down+hc_silu and up+gated-mean kernels read the same
   fragment-order fp8 weights the prefill mix kernels use (`quantize_fp8` makes the down copy, 3.4 MB a module), half
   the bytes of the bf16 kernels, no second weight copy. One 16-row tile per wave walking K in 256 B contiguous reads,
   fp32 products, the two K halves meet with one shuffle, the per-row scale applied once. The down GEMM spreads
   E = 336 (21 tiles) over the 64 CUs with SPLIT waves per tile and, optionally, KB blocks per tile along K
   (`R9K_HC_FP8_DECODE_KB`; partials in a scratch, summed in fixed order by the last block to arrive -- no float
   atomics, deterministic). Exact against the dequantised reference at M = 1..8 (`tests/test_hc_f8.py`); against
   the bf16 kernels the difference is the fp8 weight quantisation (rel 6e-3), the same quantisation the prefill mix
   path (`R9K_HC_FP8=r9k`, default since v0.3.0) already applies above 256 rows.
2. **Fused activation quant in the dense fp8 block GEMM** (`r9k_gemm_fp8_block_qa` in `kernels/r9k_gemm_fp8.hip`,
   `R9K_FP8_QA=1`, M <= 16): the per-token-group-128 quant that ran as its own launch before every dense fp8
   linear (96 a step) now happens in the GEMM's A-load: a lane holds its row's 128-group, the group absmax meets
   its other half with one shuffle, the hardware convert builds the fragments, the scale rides in registers to the
   fold. Operands bit-identical to `quant_group128_fp8` + `gemm_fp8_block` (`tests/test_fp8_qa.py`: rel 0 on every
   Flash-Next dense shape at M = 1, 3, 8, 16).

## Serving A/B (fn4 defaults on the same tree, `~/tp4tune/<label>/`)

| config | ms/step | probe: decode / c8 / c16 / 8k prefill |
|---|---:|---|
| base (k1-base, k1-base2) | 16.42, 16.42 | 199.2-199.4 / 551-559 / 889 / 6,654-6,867 |
| one-shot all-reduce cap 32 KB (k1-ar32) | 16.77 | 195.6 / 545 / 885 / 6,717 |
| **fp8 hc decode, split 16** (k2-f8dec) | **15.64** | 205.2 / 550 / 895 / 6,749 |
| fp8 hc decode, split 8 (k2-f8dec-s8) | 16.62 | 192.6 / 539 / 890 / 6,730 |
| fp8 hc decode + ar32 + route + fold (k2-f8dec-all) | 15.78 | 206.6 / 540 / 882 / 6,879 |
| base again (k2-base3) | 16.42 | 199.7 / 561 / 871 / 6,783 |
| **fused activation quant** (k3-qa) | **16.06** | 202.3 / 550 / 887 / 6,807 |
| **fp8 hc decode + fused quant** (k3-qa-f8dec) | **15.24** | 213.3 / 552 / 889 / 6,760 |
| base again (k3-base4) | 16.44 | 198.6 / 548 / 884 / 6,751 |
| fused routing + fold (k1-route, rerun) | 16.29 | 199.4 / 546 / 864 / 6,878 |
| fp8 hc decode (scalar v1) + routing + fold (k1-all) | 15.54 | 211.0 / 542 / 881 / 6,820 |
| **fp8 hc decode, WMMA kernels** (k4-f8dec) | **15.33** | 213.3 / 546 / 904 / 6,761 |
| **fp8 hc decode (WMMA) + fused quant** (k4-f8dec-qa) | **15.01** | 217.5 / 567 / 898 / 6,743 |

The first fp8 hc kernels were scalar (fp8 -> fp32 converts and FMAs against bf16 activations, MT tokens per
lane) and, graph-timed on a cache-resident weight, slower than the bf16 kernels they replace (12.1 vs 8.2 us at
M = 4): instruction-bound, not byte-bound. The shipped version converts the 8 fp8 weights of a lane to the bf16
WMMA A-fragment (e4m3 is exact in bf16) and feeds `v_wmma_f32_16x16x16_bf16` against the activations' B-fragment,
one tile of up to 16 tokens, loads software-pipelined in chunks of U k-steps: 10.5 us (down) / 7.5-8.5 (up+mix)
against 8.2-11.5 / 11.7-16.9 bf16 in that bench, and in serving 0.31 ms a step better than the scalar version
(the serving gain is the VRAM bytes; the bench's L3-resident weight hides it). Split-K across blocks
(`R9K_HC_FP8_DECODE_KB`) was slower at every setting (the scratch round trip costs more than the 21-block grid
loses) and stays at 1; SPLIT 16 is the best wave count (8: 16.62 ms/step in serving, 32: level).

**Net: 16.42 -> 15.01 ms/step (-8.6%), single-stream decode 199 -> 217 tok/s (+9%), 8 and 16 concurrent and 8k
prefill unchanged.** Radiance 1.3.0's 12.1 ms is still 2.9 ms away; what remains is the serial chain of ~1,700
small kernels (see above) and the all-reduce (98 x 12.8 us).

## The gate (2026-10-10, `~/gate-run.log`, `~/.r9keval/gate-{base,f8dec-qa}.json`)

Candidate = both paths on, base = the same tree with both off, Flash-Next TP4 fn4 defaults, conc 1, no-think:

| | GSM8K 1,319 | HumanEval | strict sanity (conc 17 x 60, conc 1 x 20) |
|---|---:|---:|---|
| base | 95.75% (1,263) | 160 / 164 | 0 bad of 1,040 |
| fp8 hc decode + fused quant | **95.83%** (1,264) | **161 / 164** | 0 bad of 1,040 |

Paired: base-only-right 7, candidate-only-right 8, McNemar p = 1.00; outputs identical on 25.5% of the questions
(the fp8 hc decode rounds the hyper-connection GEMMs' bf16 bits differently in every layer, as the fused sum did
in 2026-10-07's gate at 12.8%). No detectable difference; both paths are now the fn4 defaults in
`serve/flashnext.sh` (`R9K_HC_FP8_DECODE=0` / `R9K_FP8_QA=0` switch them off). The fused quant alone is
bit-identical to the two-launch path (its operands are the same bytes), so its share of the change is nil.

## Still open

- The GPU-only timeline of the step. rocprofv3 under vLLM writes nothing (the workers die by SIGKILL before the
  tool finalizes; a plain script traces fine). The torch profiler with CUDA activity only (`PROF=1 PROFACT=CUDA`,
  `~/tprof-cand-gpu`, `~/gpu-analyze.py`) still stretches the candidate's step from 15.0 to 23.1 ms (1,898
  dispatches a step incl. the drafter; 135 gaps of 50-200 us a step = launch starvation under the tracer), so it
  attributes kernel time, not idle: per step the fp8 hc kernels 1.05 + 0.97 ms (9.9 / 9.2 us each, from 3.5 ms
  bf16), fused-quant block GEMMs 1.36 (from 1.90), MoE GEMMs 2.78, two-shot all-reduce 1.27, GDN verify 0.53,
  topk 0.37, router 0.30; 12.0 ms of kernel time in all.
- The all-reduce (98 x 12.8 us) and the ~1,700-kernel serial chain are what separate 15.0 ms from radiance's 12.1.

## The all-reduce (2026-10-10, Brian: "Let's work on the all-reduce now")

101 two-shot all-reduces a step (48 layers x 2, plus the MTP layer's), 20 KB each at MTP-3 (4 tokens x 2560 bf16),
12.8 us each in the step. Measured on four ranks in HIP graphs (`tests/test_ar_nrank.py`, `~/arnsweep.log`):

| message | RCCL | one-shot | two-shot |
|---|---:|---:|---:|
| 5 KB (1 token) | 60 us | **6.8** | 11.4 |
| 20 KB (4 tokens) | 73 | 13.4 (over the 16 KB cap) | **11.4** |
| 80 KB (16 tokens) | 70 | 37 | 24.4 |

The two-shot costs the same at 5 KB and 20 KB: at decode sizes its time is protocol, not bytes. The fence sweep
(`R9K_AR_FENCE` drain,acq = 4,2 default / 4,0 / 2,2 / 2,0 / 0,0 / 3,1) moved the 20 KB two-shot by nothing
(11.3-11.5 us; 3,1 is 12.0): the release store and acquire poll are free. The block count is free too
(`~/arnnb.log`: 1-12 blocks all 11-13 us at 20 KB; the serving policy's 2 blocks are 11). What remains per call: ~3 us of graph
node (the per-kernel cost measured above), two handshakes at ~2.9 us each (a flag has to cross switch A, the root
complex and switch B: ~1.5-2 us one way, then the poll sees it), ~1.6 us of data, ~1 us of reduce + copy. The
one-shot has one handshake but pushes 60 KB a rank, 80 KB a direction across the switches at ~12 GB/s = 6.6 us,
which is why it loses above 16 KB.

Nothing cheaper is on offer on this topology without changing what is on the wire: a tag-in-data (NCCL "LL")
format removes the flag but not the latency and costs 1.33-2x the bytes; a pair-first hierarchy trades a cross
handshake for an intra-switch one and a remote read whose ordering against a third device's flag is not sound
without those tags; a lossy wire (fp8 / int8) saves under 1 us of the 1.6 us of data. The all-reduce is within
~1 us of its floor here; the lever left is the handshake COUNT (two per layer, inherent to row-parallel TP) and the
node overhead, i.e. fusing the all-reduce into its neighbours, which is the same fatter-kernel project as the rest.

## Round 3 (2026-10-10, Brian: "Go with option 2"): the fp8 hc kernels at every width, and the bf16 copies

The v0.3.1 kernels cover 16 tokens; at 8 and 16 streams (32-64 tokens a step) the step still streamed the bf16
hyper-connection weights, and the bf16 copies stayed resident for 17-255 rows. `kernels/r9k_hc_f8.hip` now takes
any M in token tiles of 16 (`R9K_HC_F8_MT` tiles per block, default 2: the converted weight fragment serves both;
more tiles per block lost to registers and the per-k-step x loads), `R9K_HC_FP8_DECODE_MAXM` sets the widest row
count on them, `R9K_HC_FREE_BF16=1` drops the bf16 up / down weights after quantisation once every row count has
an fp8 path (needs `R9K_HC_FP8_DOWN=1` for the tiled path above FP8_MIN_M). Exact at every M (`tests/test_hc_f8.py`).

Graph-timed on card 4 (L3-resident weight, so the bf16 side is flattered), down / up+mix, us:

| rows | fp8 WMMA (2 tiles) | bf16 hipBLASLt (+ hc_silu) | tiled fp8 W8A8 (quant + GEMM) |
|---:|---:|---:|---:|
| 32 | 12 / 13 | 12 / 8 | - |
| 64 | 20 / 20 | 16-22 / 9 | 81 / 12 |
| 128 | 30 / 38 | 22-27 / 15 | 107 / 41 |
| 255 | 56 / 70 | 34-37 / 27 | 110 / 25 |

Serving A/B, fn4 defaults (v0.3.1) on the same tree (`~/tp4tune/k5b-*`; a cold compile cache on every run, so the
KV figures compare with each other but not with warm records):

| config | decode | 8 streams | 16 streams | 8k prefill | 390-token prefill | KV tokens |
|---|---:|---:|---:|---:|---:|---:|
| v0.3.1 defaults (k5b-base) | 219.4 | 550 | 904 | 6,785 | 1,660 | 419,347 |
| fp8 kernels to 255 rows (k5b-maxm255) | 218.4 | **581** | **916** | 6,787 | 1,658 | 419,762 |
| + bf16 copies freed, tiled fp8 down above 255 (k5b-free) | 219.4 | 582 | **935** | 6,729 | 1,691 | **489,446** |
| v0.3.1 defaults again, warm cache (k5b-base2) | 219.8 | 560 | 907 | 6,883 | 1,759 | 452,115 |

The wider kernels alone: +5.6% at 8 streams, +1.4% at 16, nothing else moves. Freeing the bf16 copies on top:
+70k KV tokens cold-against-cold (+17%), 16 streams +3.4%, 8k prefill -0.8% (the tiled fp8 down GEMM's activation
quant above 255 rows, as `notes/prefill-fp8-pipe.md` measured), short prompts unchanged. Warm against warm
(`~/k6.log`, base / free / base / free): **452,115 / 515,577 / 452,115 / 515,577 tokens** (+63k, +14%); the log
line reads "bf16 copies freed on 97 (1310 MB)" plus the MTP layer's 3 (40 MB).

Two cards with the experts in host RAM (TP2, `~/k6.log`, warm against warm, the fp8 decode path on with it):
KV **166,818 -> 221,184 tokens (+33%)** -- here the freed bf16 pair outweighs the fp8 down copy that kept the
decode path off at TP2 in v0.3.1 -- single-stream 120.1 -> 122.5 tok/s, 8 streams 185 -> 174 and 16 streams
153 -> 163 (the TP2 probe's 8/16-stream numbers swing +-6% between runs of one config), 8k prefill unchanged.

A first pass of this A/B ran on a serving tree that had never received the v0.3.1 launcher defaults (both new
paths off: 198 tok/s single-stream, "quantised ... up, 320 MB a rank" in the log) and measured nothing about the
kernels; `feedback`: sync serve/ with the code, read the quantise log line before trusting an A/B.

### Gate (2026-10-10, `~/gate2-run.log`): candidate = both knobs on (`MAXM=255`, `FP8_DOWN=1`, `FREE_BF16=1`), base = v0.3.1 defaults, same tree, conc 1

| | GSM8K 1,319 | HumanEval | strict sanity |
|---|---:|---:|---|
| v0.3.1 defaults | 95.83% (1,264) | 159 / 164 | 0 bad of 1,040 |
| option 2 | 95.75% (1,263) | **162 / 164** | 0 bad of 1,040 |

Paired: base-only-right 11, candidate-only-right 10, McNemar p = 1.00; outputs identical on 18.5% (the fp8 weights
now reach every row count, so most answers differ in bf16 bits somewhere). No detectable difference. Both knobs and
the 255-row cap become the `serve/flashnext.sh` defaults at TP2 and TP4 (`R9K_HC_FREE_BF16=0` keeps the bf16 pair).