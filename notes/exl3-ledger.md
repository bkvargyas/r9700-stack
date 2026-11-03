# DeepSeek-V4.1-Flash EXL3 2.9 bpw -- the step ledger (package Z, Phase 4, 2026-11-03)

Measured on VM100 (eight R9700, SCLK -500 MHz, 120 V feed) from a pristine copy of the v0.4.0 tree (`git archive` of branch
v041 HEAD d79873a -> VM100 `~/r9700-exl3-Z`, libr9k.so of the same kernels), image r9700/vllm:dev, the validated launcher
defaults (serve/ds41-exl3.sh: TP8 + EP, MAXLEN 32768, KVMEM 1.7, NSEQ 4, NBT 2048, DSpark k = 5, adaptive verification on,
HIP graphs, capture sizes 1,2,4,6,12,18,24). Workload: greedy prose (bench/ds41/ledger/workload.py: "3000-word essay about
the history of X", thinking off, ignore_eos), 1 and 4 streams, measured mid-decode. Every number is rank 0 unless marked.
Numbers are per verify step ("a step" = one scheduler step: the target forward, the sampler / rejection sampler, the
drafter's propose). Card time 11:28-12:27 UTC (one hour; cards idle, lock removed, offset back to 0 after).

## Where the time goes (read your row)

1. **The step is GPU-bound, not launch-bound.** Unprofiled, the default single-stream step is **25.4 ms** (74.3 tok/s at 1.89
   accepted tokens a step); the kernels themselves sum to **23.5 ms** of it, so launch gaps + CPU gaps are **<= 1.9 ms**
   (<= 8 %). The four-stream step: 40.0 ms, kernels 37.5 ms, gaps <= 2.5 ms. (Section 3.)
2. **Launches: 1,754-1,866 a step** (T = 6: target forward 1,534, sampler 38, drafter 182; 32 a plain decoder layer), not
   ~810: the plan's 810 was 51,932 kernels over 64 *tokens*. Glue (norms, casts, adds, copies) is 563-609 of them and
   1.24-1.38 ms of GPU time; with the drafter's 91 launches / 0.35 ms. (**B**: that is the fusion pool; section 2.)
3. **Collectives are the largest row: 9.6 ms of 23.5 (41 %) at the default T = 6, 13.3 of 37.5 (36 %) at T = 24.** Of that,
   **4.1 ms a step is one slow kernel: the flat eight-rank all-gather `r9k_ag_oneshot_nrank_kernel<8>`** (Engram's two
   [T, 3200] gathers at 0.65-1.07 ms each, and the target logits gather [T, 16160] at **3.2-3.6 ms** whenever adaptive
   verification trims the verify to <= 4 rows, so that the message fits R9K_AG_MAX_KB 128 and leaves RCCL, which does the
   6-row gather in 0.20 ms). The 81 all-reduces a step cost 71-81 us each at 40-60 KB (min over the eight ranks 50 us; the
   post-MoE one also absorbs the expert-load imbalance between ranks: 99 us vs 46 us for the attention one), 106 us at
   90-160 KB, 161 us at 240 KB (min 80). (**C**, and **A** for 240/480 KB; section 1b.)
4. **Expert tier (r9k k_gu + k_dn): 2.5 ms at the default T = 6 (62 us a layer at 2-4 verify rows), 3.9 ms at a true
   6-row verify (96 us a layer), 6.8 ms at 9-16 rows (170 us), 12.6 ms at 24 rows (313 us a layer)** -- 11-25 % of the step.
   (**A**.)
5. **Attention + dense GEMVs: 6.2 ms a step at a 6-row verify** (G3: EXL3 GEMVs 2.1 ms, wo_a bf16 1.1 ms, shared experts
   1.2 ms, router gate 0.5 ms, decode attention 0.95 ms, compressor / indexer bf16 0.3 ms): the EXL3 GEMVs run at 288-292 GB/s on trellis bytes
   (fused_wqa_wkv 19.6 us, wq_b 11.4, wo_b 11.4 a layer), wo_a at 302 GB/s (27.8 us). **Above 16 rows (R9K_EXL3_RECON_M) every
   EXL3 linear dequantizes its whole matrix every step** (reconstruct + f16 hipBLAS: 7.6 ms at T = 24 vs 2.1 at T = 6).
   (**D**.)
6. **hc seams 0.7-0.8 ms** (9.7 us a seam at T = 6), **decode attention 0.86-0.95 ms** (21-23 us partial + 2.6 reduce a
   layer), router top-k 0.33 ms, indexer + KV writes + compressor 0.25 ms, sampler + rejection 0.12 ms.
7. **The drafter is 2.6-3.2 ms a step at one stream** (its three layers ~0.36 ms each; then a vocab-sliced lm_head 0.27 ms +
   RCCL gather 0.18 ms + the markov head: **five sequential [129280 x 256] f16 GEMVs + argmax = 0.67 ms**), 5.6 ms at four
   streams (20 rows: its EXL3 linears take the dequant path, 1.4 ms).
8. **Adaptive verification** (row 4): on single-stream prose it trims the GPU's verify to 4 rows (60 % of steps) or 2 rows
   (37 %), saving 1.4 ms of GPU time a step but dropping the accepted length 2.20 -> 1.84-1.89, and its small-message logits
   gather lands on the slow flat all-gather (3.2-3.6 ms): **74 vs 81 tok/s with it off**. At four streams it trims 24 -> 9-16
   rows and wins (185-202 vs 155 tok/s aggregate). It adds one drafter launch (an f32 Cijk GEMM, 0.26-0.63 ms).
9. **The eight-stream (48-token) shape does not launch on v0.4.0**: NSEQ 8 dies in the engine warm-up (right after "Warming
   up DeepSeek V4 sparse MLA attention for mixed tokens=16") with HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION, at KVMEM 1.7
   (G2) and again at KVMEM 1.0 (G2b), so it is not the KV budget. There is no T = 48 measurement; that is package A's first
   item (v0.4.0's r9k expert tier also stops at T x topk <= 256, i.e. 42 tokens).
10. **Profiling inflates the step by ~9 ms** (traced 34.5 ms vs 25.4 unprofiled, with CPU+GPU or GPU-only activity alike):
    the gaps appear between the nodes of the graph replay (5.8 us a node). Read traced kernel durations, never traced periods.

| package | its row at T = 6 (default, 1 stream) | at T = 24 (default, 4 streams) |
|---|---:|---:|
| A expert tier (main + drafter) | 2.60 ms (11 %) | 7.07 ms (19 %) |
| A/C all-reduces (81 + 7 a step) | 5.92 ms | 9.63 ms |
| C flat 8-rank all-gather (Engram x 2 + logits <= 4 rows) | 4.05 ms | 4.38 ms |
| C RCCL logits gathers (target + drafter) | 0.18 ms | 0.80 ms |
| B glue (norms / casts / adds / copies), launches | 1.60 ms, 658 launches | 1.77 ms, 653 |
| D EXL3 GEMVs + dequant path (main + drafter) | 2.01 ms | 5.02 ms |
| D bf16 GEMMs: wo_a, shared, router, compressor / indexer, lm_heads, markov head, adaptive's f32 GEMM (main + drafter) | 4.32 ms | 5.47 ms |
| decode attention (main + drafter) | 0.89 ms | 1.33 ms |
| hc seams | 0.75 ms | 0.92 ms |
| launch + CPU gaps (unprofiled step - traced kernel time) | <= 1.96 ms | <= 2.55 ms |
| **step, unprofiled** | **25.42 ms** | **40.0 ms** |

(Launch G4: defaults, GPU-only trace. Of the bf16 row, 1.61 ms at T = 6 is heads: target lm_head 0.16, drafter lm_head 0.29,
markov head 0.54, adaptive verification's f32 GEMM 0.63. Section 1 has the full table and the other launches.)

## Launches and method

| tag | launch | workloads | traces (VM100 `~/ds41/exl3/Z/`) | what it is used for |
|---|---|---|---|---|
| E1 | EAGER=1, ADAPTIVE=false (adaptive verification refuses eager mode: "requires full CUDA graphs"), NSEQ 4, torch profiler with Python stacks + shapes | 1 stream, 4 streams, 6 s each | `E1/s1/`, `E1/s4/` (8 rank traces each, 0.85 GB a rank) | kernel names -> ops / shapes / call sites, the launch list, layer classes; eager durations |
| G1 | defaults (graphs, ADAPTIVE=true), CPU+GPU profiler | 1 and 4 streams, 3 s | `G1/s1/`, `G1/s4/` | phases from the worker's execute_* annotation, cross-rank collective minima |
| G3 | graphs, **ADAPTIVE=false**, CPU+GPU profiler | 1 and 4 streams, 3 s | `G3/s1/`, `G3/s4/` | the true 6- and 24-row verify; item 4 |
| G4 | defaults, **GPU-only** profiler (activities ["CUDA"]) | 1 and 4 streams, 3 s | `G4/s1/`, `G4/s4/` | the headline table and the wall split without CPU-profiler overhead |
| G2 | defaults with NSEQ 8 (KVMEM 1.7) | -- | `G2/launch-fault.log` | did not come up (warm-up GPU fault, HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION) |
| G2b | NSEQ 8, KVMEM 1.0 | -- | `G2b/launch-fault.log` | did not come up either (same fault) |

Each launch: `bench/ds41/ledger/zrun.sh` (relaunch from `~/r9700-exl3-Z` with VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 and a
`--profiler-config` into a mounted `/prof`, warm-up, then per workload `workload.py`: a 3 s **unprofiled timing window**
(tok/s from continuous usage stats, accepted / drafts from /metrics) followed by a 3 s **profile window** between
POST /start_profile and /stop_profile). Raw logs: `<tag>/run.log`, `<tag>/launch.log`, `<tag>/s<N>.json` (both windows).
Analysis: `ledger.py` (steps = the worker's `execute_context_C(c)_generation_G(g)` annotations; forward = kernels launched
inside it, i.e. the target's full-graph replay; the rest of the step = sample_tokens: sampler / rejection sampler and the
drafter's propose -- this pin's DSpark runs on model runner v2, which has no profiler scopes, so the drafter is told apart by
its own graph replay plus the kernel names E1 saw only under `propose`, `E1/draft-names.json`), `gputl.py` (GPU-only traces:
a step = one target-forward graph replay to the next, phases by correlation group), `layers.py` / `classavg.py` (layer
segmentation at the hyper-connection seams, per-position means over a layer class), `widths.py` (the verify width each step
actually ran), `mklist.py` (the launch list). Outputs next to the traces: `<tag>/s<N>-ledger.txt|json`, `-list.txt`,
`-layers.txt`, `-classavg.txt`, `G4/s<N>-gputl.txt`.

Why the torch profiler and not rocprofv3: on this pin the kineto/roctracer GPU activity carries every graph node's kernel
name, grid and hardware timestamps (G4: 1,865 kernels a step, the eager launch's set), and the profiler's annotations are the
only way to tell the target forward, the sampler and the drafter apart; rocprofv3 would need its own launch for the same
kernel list. Caveats, both measured: (a) tracing adds ~9 ms of idle between graph nodes (G1 36.2 ms traced vs 25.9
unprofiled; G4 GPU-only 34.5 vs 25.4), so periods come from the unprofiled windows; (b) a collective kernel's duration
includes waiting for the slowest rank; `ledger.py --ranks` matches each call across the eight rank traces and reports the
minimum over ranks next to rank 0 (where the ranks' call lists line up: the ar8 fused path; the pair / quartet stage
kernels differ between even and odd ranks).

**The verify width the GPU ran** (`widths.py`, from the shared-expert GEMM hipBLAS picks per step): with adaptive
verification the scheduler still books k + 1 = 6 tokens a stream, but the device trims them.

| launch | steps | rows verified on the GPU |
|---|---:|---|
| G1 1 stream, adaptive | 85 | 4 rows 60.0 %, 2 rows 36.5 %, 5-8 rows 3.5 % |
| G1 4 streams, adaptive | 63 | 9-16 rows 79.4 %, 5-8 rows 20.6 % |
| G3 1 stream, ADAPTIVE=false | 87 | 6 rows 100 % |
| G3 4 streams, ADAPTIVE=false | 49 | 24 rows 100 % (EXL3 dequant path: 170 reconstructs a step) |

## 1. GPU time a step by category, us

Rank 0, graph replay, sum of kernel durations a step (traced; see caveat (b) for the collectives). "T" is the scheduled
verify width (streams x 6); the row counts the GPU actually ran are in the table above. Launches a step in brackets.

| category | T=6 default (G4) | T=6 ADAPTIVE=false (G3) | T=24 default (G4) | T=24 ADAPTIVE=false (G3) | T=48 (NSEQ 8) |
|---|---:|---:|---:|---:|---:|
| expert tier r9k k_gu / k_dn | 2,503 [80] | 3,857 [80] | 6,814 [80] | 12,636 [80] | -- |
| EXL3 GEMV (exl3_gemv dot / had_in, reconstruct, had_r_128) | 1,677 [260] | 2,068 [260] | 3,593 [490] | 3,059 [510] | -- |
| EXL3 dequant path: f16 hipBLAS | 0 | 0 | 0 | 4,547 [178] | -- |
| hipBLAS / hipBLASLt bf16 (wo_a, shared w13 / w2, router, compressor / indexer bf16, target lm_head) | 2,709 [177] | 3,456 [177] | 3,706 [177] | 3,605 [169] | -- |
| decode attention: split-K partial + reduce | 860 [80] | 952 [80] | 1,276 [80] | 1,781 [80] | -- |
| attention: indexer | 124 [46] | 148 [54] | 100 [46] | 144 [54] | |
| attention: KV writes / RoPE / compressor | 109 [48] | 131 [53] | 118 [48] | 140 [54] | |
| collectives (all, incl. Engram + logits gathers) | 9,568 [149] | 9,007 [90] | 13,318 [87] | 15,537 [86] | -- |
| hc seams (r9k_mhc seam / post, hc head) | 695 [87] | 794 [87] | 846 [87] | 1,144 [87] | |
| router top-k (topkGatingSoftplusSqrt) | 332 [40] | 335 [40] | 341 [40] | 340 [40] | |
| norms / casts / elementwise glue | 1,240 [564] | 1,381 [609] | 1,302 [561] | 2,190 [611] | -- |
| Engram (hash, lookup, post-wkv) | 21 [5] | 26 [5] | 26 [5] | 29 [5] | |
| sampler / rejection sampler (G4: + eager input prep) | 394 [141] | 121 [36] | 415 [145] | 144 [39] | |
| everything else | 3 | 4 | 5 | 0 | |
| **drafter**: expert tier | 101 [6] | 100 [6] | 258 [6] | 252 [6] | |
| drafter: EXL3 GEMV (+ dequant f16 GEMMs) | 329 [26] | 325 [26] | 601 + 823 [76] | 601 + 824 [76] | |
| drafter: hipBLAS (heads: lm_head slice, 5 x markov head; adaptive's f32 GEMM) | 1,608 [23] | 985 [22] | 1,763 [23] | 1,138 [22] | |
| drafter: collectives (7 all-reduces + logits RCCL gather) | 690 [8] | 685 [8] | 1,497 [8] | 1,434 [8] | |
| drafter: glue / attention / hc / router / other | 497 [131] | 489 [127] | 645 [128] | 641 [121] | |
| **GPU kernel time a step** | **23,460** | **24,866** | **37,450** | **50,180** | -- |
| launches a step | 1,865 | 1,754 | 2,082 | 2,221 | -- |
| **step, unprofiled** (timing window) | **25,420** | **27,270** | **40,000** | **52,630** | -- |

T = 48: no launch came up (G2, G2b; summary item 9). G3 columns: `ledger.py` (phases from the CPU annotation); G4 columns: `gputl.py` (phases by graph replay; its "sampler" row
also holds the forward's eager input-prep kernels, ~0.27 ms). G1 (the same defaults with the CPU profiler on) agrees with G4
within 1-4 % on every row (`G1/s1-ledger.txt`: 23,755 us, 1,866 launches; `G1/s4-ledger.txt`: 36,445 us, 2,062).

### 1a. Per call / per layer (graph replay, G3 unless noted; class means over the 26 plain layers, `G3/s1-classavg.txt`)

| kernel | T≈4 (G1/G4) | T=6 (G3) | T=9-16 (G4) | T=24 (G3) | bytes read a rank | GB/s at T=6 |
|---|---:|---:|---:|---:|---:|---:|
| k_gu + k_dn a K=3 layer (35 layers) | 41.0 + 20.9 | 65.0 + 31.1 | 119.1 + 50.6 | 227.9 + 85.1 | distinct local experts x 13.27 MB | -- |
| k_gu + k_dn a K=2 layer (layers 18-22) | 37.8 + 19.8 | 66.8 + 32.2 | 124.3 + 50.8 | 248.1 + 88.5 | distinct local experts x 8.85 MB | -- |
| fused_wqa_wkv exl3_gemv dot_multi (K=5, 5120 -> 1792) | 16.1 | 19.6 | 20.6 + 15.0 (two row chunks) | dequant, per part: wq_a reconstruct 19.8 + f16 GEMM 32.6, wkv 7.3 + 31.9 | 5.73 MB | 292 |
| wq_b dot_single (1280 -> 4096) | 9.5 | 11.4 | 11.9 + 8.8 | 16.1 + 20.0 | 3.28 MB | 288 |
| wo_b dot_single (1024 -> 5120, row split) | 9.3 | 11.4 | 11.9 + 8.8 | 14.9 + 18.6 | 3.28 MB | 288 |
| had_in (one a GEMV group) | 1.4-2.0 | 1.4-2.0 | 1.5-2.0 | had_r_128 in + out 2.4-3.0 each | | |
| wo_a bf16 bmm (4096 -> 1024 slice) | 27.4 | 27.8 | 28.1 | 29.2 | 8.39 MB | 302 |
| shared w13 / w2 bf16 | 11.9 / 7.5 (wvSplitK) | 17.5 / 12.9 | 18.1 / 13.8 | 20.1 / 15.5 | 5.90 / 2.95 MB | 337 / 229 |
| router gate bf16 mm [5120 -> 384] | 11.8 | 13.0 | 14.1 | 15.2 | 3.93 MB | 302 |
| decode attention partial + reduce (plain layer) | 18.4 + 4.2 | 22.8 + 2.6 | 21.1 + 3.1 | 42.8 + 2.4 | | |
| all-reduce after attention / after the MoE | 42.2 / 94.4 | 46.3 / 98.8 | 48.0 / 99.9 | 88.5 / 250.5 | | |
| r9k_mhc seam | 8.3-8.7 | 9.7-9.9 | 10.0 | 13.7-14.6 | hc_fn 1.97 MB f32 | 200 |
| topkGatingSoftplusSqrt | 8.3 | 8.3 | 8.4 | 8.5 | | |
| fused q-norm / RoPE / KV insert | 2.3 | 2.3 | 2.4 | 2.6 | | |
| **a plain layer, all launches** | 383 (32) | 449 (32) | 468 (32) | 1,028 (42: the dequant path) | | |

The expert-tier GB/s needs the number of distinct local experts a layer, which the trace does not carry; the plan's
estimate (4.3 distinct local experts at T = 6, 57 MB) would put 96 us at 594 GB/s. The per-layer expert time varies with the
routing: 6-127 us for k_gu at T = 6 across the plain layers (min / max in `G3/s1-classavg.txt`).

### 1b. Collectives by message size (per call; rank 0 mean, and the minimum over the eight ranks where measured)

| call (a step) | message a rank | path | rank 0 us | min over ranks | calls a step | source |
|---|---|---|---:|---:|---:|---|
| all-reduce [6, 5120] bf16 | 60 KB | ar8 fused (split path) | 81.4 | 50.1 | 81 | G3 s1 |
| -- of which after attention / after the MoE | | | 46.3 / 98.8 | | 40 / 40 | G3 classavg |
| all-reduce [4, 5120] | 40 KB | ar8 fused | 71.2 | -- | 52.5 | G4 s1 |
| all-reduce [2, 5120] | 20 KB (< SPLIT_KB 32) | three stage kernels: pair one-shot x 2 + quartet one-shot | 25.4 + 33.4 = 58.8 | -- | 28.5 | G4 s1 |
| all-reduce [9-16, 5120] | 90-160 KB | ar8 fused | 106.4 | 52.1 (G1) | 81 | G4 s4 |
| all-reduce [24, 5120] | 240 KB | ar8 fused | 160.6 | 79.6 | 81 | G3 s4 |
| drafter all-reduce [5, 5120] / [20, 5120] | 50 / 200 KB | ar8 fused | 72.5 / 135.6 | 39.4 / 70.3 | 7 | G3 s1 / s4 |
| **Engram all-gather [T, 3200] -> [T, 25600]** | 38 KB at T=6 | **flat 8-rank one-shot `r9k_ag_oneshot_nrank_kernel<8>`** | **1,069** (T=6), 653 (T≈4), 805 (T=24) | 1,067 | 2 | G3 s1, G4 s1, G3 s4 |
| Engram pair / quartet all-gathers | | ag<2> x 2, ag<4> | 10-12, 14-17 | | 4 + 2 | G3 classavg |
| **target logits all-gather [T, 16160] at <= 4 rows** | <= 126 KiB | **flat 8-rank one-shot** | **3,205** (G4), 3,636 (G1) | 3,636 | 1 | G4 s1, G1 s1 |
| target logits all-gather at 6 / 24 rows | 194 / 776 KB | RCCL (ncclDevKernel_Generic_4) | 200 / 574 | 196 / 565 | 1 | G3 s1 / s4 |
| drafter logits all-gather [5, 16160] / [20, 16160] | 162 / 646 KB | RCCL | 178 / 485 | 174 / 480 | 1 | G3 s1 / s4 |

The flat eight-rank all-gather is what `R9kAllReduce8.all_gather` falls back to (v0.4.0 `comm/r9k_ar8.py` `_ag_takes`) when
the gathered size exceeds the pair path's R9K_AG_MAX_KB (128) but the input alone does not: every rank reads all seven
peers, three of them through the far switch. Its time is the same on every rank (min over ranks = rank 0), so it is the
kernel, not waiting. At T <= 4 the logits message is 126 KiB and takes it; at 6 rows (194 KB) it goes to RCCL at 0.2 ms.

## 2. The launch list (T = 6)

Full list, every kernel of one median T = 6 step with grid / block, eager duration, weight bytes and GB/s, cpu op and input
shapes: **`bench/ds41/ledger/launch-list-T6.txt`** (from E1, rank 0, sections: step preamble, layers 0, 1, 2, 3, 24, 37, 39,
the sampler, the drafter's preamble, layer 0 and last layer + heads). The whole step (1,672 eager kernels) with Python call
sites: VM100 `~/ds41/exl3/Z/E1/s1-list.txt`; the graph-replay order and durations: `G3/s1-list.txt`, `G3/s1-classavg.txt`.

Layer classes (identical kernel sequences; graph replay counts, G3): plain layers 3-7, 9-12, 15-19, 21-23, 25-27, 29-31,
33-35 (26 layers): **32 launches**; layers 2, 8, 14 (kv source: compressor + indexer wk): 50; layer 20 (kv source, last
compressor layer): 56; layers 24, 28, 32, 36 (indexer, no compressor): 49 / 51; layers 37, 38 (aux hidden state for the
drafter): 34; layer 39 (+ final hc head): 40; layer 0 (+ Engram for layer 1): 44; layer 13 (+ Engram for layer 14): 43.
The forward also has 70 preamble launches (input prep, embedding all-reduce, Engram hashing / lookups).

**A plain decoder layer at T = 6** (graph replay means over the 26 plain layers, G3; bytes per rank):

| # | kernel | grid (eager) | us | category | what |
|---:|---|---|---:|---|---|
| 0 | r9k_mhc_seam_kernel<4, 8> | 80 x 256 | 9.9 | hc seam | post of the previous FFN + pre of attention, hc_fn 1.97 MB |
| 1 | vllm::rms_norm_kernel | 6 x 640 | 1.9 | glue | attn_norm |
| 2 | elementwise copy (bf16 -> f16 [6, 5120]) | 60 x 128 | 3.6 | glue | EXL3 input cast |
| 3 | exl3_gemv_mr_had_in_multi | 60 x 256 | 2.0 | EXL3 | input Hadamard + suh |
| 4 | exl3_gemv_mr_dot_multi<5, true, 2, 8, 8> | 80 x 2 x 256 | 19.6 | EXL3 | fused_wqa_wkv 5120 -> 1280 + 512, 5.73 MB, 292 GB/s |
| 5-6 | elementwise copy x 2 (f32 -> bf16 [6,1280], [6,512]) | 8, 3 x 128 | 4.3 | glue | output casts |
| 7 | Triton `kernel` | | 1.5 | glue | q_norm (Triton) |
| 8 | elementwise copy (bf16 -> f16 [6, 1280]) | 15 x 128 | 3.6 | glue | |
| 9 | exl3_gemv_mr_had_in_single | 8 x 256 | 1.4 | EXL3 | |
| 10 | exl3_gemv_mr_dot_single<5, true, 2, 4, 8> | 256 x 128 | 11.4 | EXL3 | wq_b 1280 -> 4096, 3.28 MB, 288 GB/s |
| 11 | vectorized copy (f32 -> bf16 [6, 4096]) | 12 x 256 | 1.4 | glue | |
| 12 | fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel | 7 x 256 | 2.3 | attention | q norm + RoPE, KV RoPE + fp8 quant + insert |
| 13 | _sparse_attn_decode_partial_kernel (Triton) | | 22.8 | attention | split-K partial |
| 14 | _sparse_attn_decode_reduce_kernel (Triton) | | 2.6 | attention | reduce + fused inverse RoPE |
| 15 | Cijk_Alik_Bljk_BBS MT16x64x128 (aten::bmm) | 8192 x 128 | 27.8 | hipBLAS | wo_a slice [6,4096] x [4096,1024] bf16, 8.39 MB, 302 GB/s |
| 16 | elementwise copy (bf16 -> f16 [6, 1024]) | 12 x 128 | 3.6 | glue | |
| 17 | exl3_gemv_mr_had_in_single | 6 x 256 | 1.5 | EXL3 | |
| 18 | exl3_gemv_mr_dot_single<5, true, 2, 4, 8> | 320 x 128 | 11.4 | EXL3 | wo_b 1024 -> 5120, 3.28 MB, 288 GB/s |
| 19 | vectorized copy (f32 -> bf16 [6, 5120]) | 15 x 256 | 1.4 | glue | |
| 20 | r9k_ar8_fused_kernel | 80 x 480 | 46.3 | collective | all-reduce [6, 5120], 60 KB |
| 21 | r9k_mhc_seam_kernel<4, 8> | 80 x 256 | 9.7 | hc seam | post of attention + pre of the FFN |
| 22 | vllm::rms_norm_kernel | 6 x 640 | 1.9 | glue | ffn_norm |
| 23 | Cijk_Alik_Bljk_BBS MT32x32x256 (aten::mm) | 2304 x 128 | 17.5 | hipBLAS | shared w13 [5120, 576] bf16, 5.90 MB, 337 GB/s |
| 24 | act_and_mul (silu_and_mul_with_clamp) | 6 x 36 | 1.4 | glue | |
| 25 | Cijk_Alik_Bljk_BBS MT128x32x64 (aten::mm) | 5120 x 128 | 12.9 | hipBLAS | shared w2 [288, 5120] bf16, 2.95 MB, 229 GB/s |
| 26 | Cijk_Alik_Bljk_BSS MT16x32x256 (aten::mm) | 1536 x 64 | 13.0 | hipBLAS | router gate [5120, 384], 3.93 MB, 302 GB/s |
| 27 | topkGatingSoftplusSqrt<12, 384, ...> | 2 x 32x4 | 8.3 | router top-k | sqrtsoftplus + bias + top-6 + weights |
| 28 | k_gu<3, 2, 8> | 18 x 8 x 16 | 71.0 (6-127) | expert tier | gate + up of the local picks |
| 29 | k_dn<3, 2, 8> | 40 x 3 x 16 | 32.4 (4-64) | expert tier | down + per-slot outputs |
| 30 | vectorized add (bf16 [6, 5120]) | 15 x 256 | 1.4 | glue | routed + shared |
| 31 | r9k_ar8_fused_kernel | 80 x 480 | 98.8 | collective | all-reduce [6, 5120] (absorbs the ranks' expert imbalance) |
| | **total** | | **448.5** | | 12 glue, 6 EXL3, 4 hipBLAS, 3 attention, 2 seams, 2 experts, 2 collectives, 1 top-k |

Layers 2, 8, 14 add (beyond the plain list): the compressor's wkv + wgate as one bf16 mm [5120 -> 1024] (23.8 us, BSS), the
indexer weights_proj [5120 -> 32] (20.1 us, BBS MT16x16x192), indexer wk [512 -> 128] (9.1 us), _fused_save_compress_norm,
_indexer_k_norm_rope_quant_store, _rope_quant_insert, _paged_mqa_logits_tiled, topKPerRowDecode<512>, _compute_topk_lens,
cumsum, _pack_global_topk_ragged (2-4 us each): 547 us a layer. Indexer-only layers (24, 28, 32, 36) add the logits /
top-k part. The Engram layers add pair + quartet all-gathers (10-17 us), the engram.wkv GEMV (6144 -> 3200, K=5, 33 us,
12.3 MB, 372 GB/s) and the flat all-gather (919-1,069 us).

**A drafter layer at T = 6** (mtp.0 / mtp.1; 34 launches, 358 us): the same sequence at K = 6 (fused_wqa_wkv 19.5 us, wq_b
11.8, wo_b 11.5; attention over 5 queries 8.5 + 2.9; wo_a 27.1), shared expert through wvSplitK (12.4 + 8.0), router [5120 ->
128] (8.2 + a 1.8 us bias GEMM), topkGatingSoftplusSqrt<4, 128>, k_gu<4> 32.2 + k_dn<4> 20.1, two all-reduces [5, 5120]
44 / 92 us. The drafter's preamble (64 launches): three _prepare_dflash_inputs, cat of the three aux hidden states,
**main_proj 15360 -> 5120 (K=6, replicated: 59 MB, 116 us in the replay, ~510 GB/s)**, the three KV writes for the block, input prep
for its attention, embedding all-reduce. Its last layer adds the hc head, final norm, the drafter lm_head slice
(wvSplitK [16160, 5120] bf16 165 MB, 265 us, 623 GB/s), its RCCL logits gather (177 us), and the **markov head: five
sequential (wvSplitK [129280, 256] f16 66 MB 107 us + argmax 27.5 us)** = 673 us.

**Launch counts a step** (G3, T = 6, 1,754 in all; G4 defaults 1,865):

| category | target forward + sampler | drafter |
|---|---:|---:|
| norms / casts / elementwise glue | 609 | 91 |
| EXL3 GEMV (dot + had_in) | 260 | 26 |
| hipBLAS bf16 | 177 | 22 |
| collectives | 90 (81 all-reduces, 6 Engram + 1 logits gathers, 2 more) | 8 |
| hc seams | 87 | 7 |
| expert tier | 80 | 6 |
| decode attention | 80 | 6 |
| indexer / KV writes / compressor | 107 | 6 |
| router top-k | 40 | 3 |
| sampler / rejection sampler | 36 | -- |
| Engram, other | 6 | 7 |
| **total** | **1,572** | **182** |

Graph launches a step: 2 hipGraphLaunch (the target forward: 1,475 nodes; the drafter: 175 nodes) + ~104 eager launches
(input prep 59, sampler / rejection 38, drafter input prep 7). At 4 streams with adaptive verification (9-16 rows) the
EXL3 GEMVs run two 8-row chunks (490 launches) and the drafter's EXL3 linears take the dequant path (+50): 2,082.

## 3. Wall-clock split and the tok/s arithmetic

Unprofiled timing windows (3 s, mid-decode; `<tag>/s<N>.json`, window "timing"). drafts = verify steps x streams (the
/metrics spec-decode counter counts one draft a request a step), tokens = drafts + accepted.

| launch | streams | tokens a stream | drafts | accepted | MAL | steps | step wall | (drafts + accepted) / streams / 3 s | observed tok/s a stream (aggregate) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G4 defaults | 1 | 223 | 118 | 105 | 1.89 | 118 | 3000 / 118 = **25.42 ms** | 74.3 | **74.3** |
| G1 defaults | 1 | 213 | 116 | 97 | 1.84 | 116 | 25.86 ms | 71.0 | 71.0 |
| G3 ADAPTIVE=false | 1 | 242 | 110 | 132 | 2.20 | 110 | 27.27 ms | 80.7 | 80.7 |
| G4 defaults | 4 | 154 / 159 / 148 / 146 | 300 | 307 | 2.02 | 75 | **40.0 ms** | 50.6 | 50.6 (202.3) |
| G1 defaults | 4 | 138 / 143 / 140 / 135 | 308 | 248 | 1.81 | 77 | 38.96 ms | 46.3 | 46.4 (185.3) |
| G3 ADAPTIVE=false | 4 | 111 / 116 / 115 / 122 | 228 | 236 | 2.04 | 57 | 52.63 ms | 38.7 | 38.7 (154.7) |
| E1 eager, ADAPTIVE=false | 1 | 107 | 50 | 57 | 2.14 | 50 | 120.0 ms | 17.8 | 17.8 |

So tok/s = MAL / step wall: 1.89 / 25.42 ms = 74.3 (single stream, defaults); 4 x 2.02 / 40.0 ms = 202 aggregate. The
same windows with the profiler on run 34-50 ms a step (MAL unchanged within noise).

**Split of the unprofiled step** (G4 GPU-only trace for the kernel time; the rest by difference):

| | T = 6 default | T = 24 default | T = 6 ADAPTIVE=false (G3) | T = 24 ADAPTIVE=false (G3) |
|---|---:|---:|---:|---:|
| step wall, unprofiled | 25.42 ms | 40.00 ms | 27.27 ms | 52.63 ms |
| target forward graph replay, kernel time | 16.37 | 30.84 | 21.50 | 44.12 |
| drafter graph replay, kernel time | 3.17 | 5.55 | 2.54 | 4.84 |
| eager kernels (input prep, lm_head, logits gather, sampler / rejection) | 3.92 (3.2 of it the flat logits all-gather) | 1.06 | 0.83 | 1.22 |
| **all kernels** | **23.46** | **37.45** | **24.87** | **50.18** |
| **launch gaps + CPU (scheduler, sampler / rejection on the host, async scheduling, drafter input prep)** | **<= 1.96 (7.7 %)** | **<= 2.55 (6.4 %)** | **<= 2.40 (8.8 %)** | **<= 2.45 (4.7 %)** |

"<=": the traced kernel durations of the collectives include rank skew, which is larger with tracing on; the true kernel
time is a little lower and the gaps a little higher, but the ceiling holds. With tracing the same steps show 8.7-9.9 ms of
idle **inside** the forward replay (5.8 us between consecutive graph nodes) and ~1.0 ms between the replays (G4 s1: 1.03 ms
between the eager kernels after the forward, 1.04 ms inside the drafter replay); that idle exists only under tracing (the
traced period is 34.5 ms against 25.4 unprofiled). Graph replay vs eager: the eager launch E1 takes 120 ms a step for
17.7 ms of non-collective kernel time (CPU-launch-bound at ~70 us a launch); the graph replay of the same work is 27.3 ms.

## 4. Adaptive verification: true vs false (T = 6 sched, one stream)

| | ADAPTIVE=true (default; G4 / G1) | ADAPTIVE=false (G3) | delta |
|---|---:|---:|---:|
| rows verified on the GPU | 4 (60 %), 2 (37 %) | 6 | |
| GPU kernel time a step | 23.46 / 23.76 ms | 24.87 ms | -1.4 ms |
| -- expert tier | 2.50 | 3.86 | -1.35 |
| -- EXL3 GEMV + hipBLAS + attention | 5.25 | 6.48 | -1.23 |
| -- target logits all-gather | 3.21 (flat 8-rank ag, <= 126 KiB) | 0.20 (RCCL, 194 KB) | **+3.0** |
| -- Engram all-gathers | 0.85 (1.3 flat calls) | 2.14 (2 flat calls) | -1.3 |
| -- drafter hipBLAS (adaptive adds one f32 Cijk GEMM, MT256x128x8 / MT128x64x16) | 1.61 | 0.99 | +0.62 |
| launches | 1,865 | 1,754 | +111 (three-kernel all-reduces at 2 rows, the extra GEMM) |
| step wall, unprofiled | 25.42 / 25.86 ms | 27.27 ms | -1.4 to -1.9 |
| MAL | 1.89 / 1.84 | 2.20 | -0.31 to -0.36 |
| tok/s | 74.3 / 71.0 | 80.7 | **-8 to -12 %** |
| four streams, aggregate tok/s | 202.3 / 185.3 (9-16 rows) | 154.7 (24 rows) | +20 to +31 % |

Adaptive verification's single-stream loss is mostly its own message size: at <= 4 rows the logits gather leaves RCCL for
the flat eight-rank all-gather (+3.0 ms). Arithmetic, not a measurement: with that gather at the RCCL
speed measured for 6 rows (0.2 ms) the adaptive step would be 25.42 - 3.0 = 22.4 ms, i.e. 1.89 / 22.4 ms = 84 tok/s, ahead of
ADAPTIVE=false.

## 5. Notes for the packages

- **A (eight-stream expert tier + 240/480 KB collectives).** Expert tier per layer 62 / 96 / 170 / 313 us at 2-4 / 6 / 9-16 /
  24 rows (P1's targets T=6 <= 120 and T=24 <= 400 hold). At 24 rows the all-reduce is 161 us a call rank 0, 80 us min over
  ranks; x 81 = 13.0 ms of the 50.2. v0.4.0's r9k tier gates T x topk <= 256 (T <= 42), so a 48-token step takes the fused
  exllamav3 tier; NSEQ 8 does not come up at all on v0.4.0 (G2 at KVMEM 1.7 and G2b at 1.0: GPU memory-aperture
  violation in the engine warm-up after the sparse-MLA mixed-tokens=16 warm-up; logs `G2/launch-fault.log`,
  `G2b/launch-fault.log`), so there is no measured 48-token step yet. The 24-row numbers above are the nearest measured shape.
- **B (launch fusion).** Unprofiled gaps are <= 1.9-2.5 ms a step, so fusion pays mainly in the glue kernels' own time:
  609 + 91 glue launches = 1.6-1.7 ms a step at 1.4-4.9 us each (the casts around every EXL3 GEMV alone: 7 a layer, ~19 us),
  plus the per-kernel floor of the short ones (had_in 1.4-2.0, rms_norm 1.9, add 1.4). The router is 2 launches a layer
  (gate mm 13.0 + top-k 8.3 = 21 us); the MoE combine is 1 add (1.4 us).
- **C (collectives).** Before the single-kernel all-reduce: **the flat eight-rank all-gather is 4.1 ms a default step**
  (Engram 2 x 0.65-1.07 ms, logits 3.2-3.6 ms at <= 4 rows) -- more than all 81 all-reduces' minimum time. All-reduce per
  call: 59 us (20 KB, three stage kernels), 71 (40 KB), 81 (60 KB; min 50), 106 (90-160 KB), 161 (240 KB; min 80); the
  post-MoE call carries the expert imbalance (99 vs 46 us after attention). A fused MoE reduction would also absorb that
  wait into a kernel that is doing work.
- **D (attention GEMV bandwidth).** EXL3 GEMVs 288-292 GB/s at 6 rows, 345-356 at 4 (fused_wqa_wkv 19.6, wq_b 11.4, wo_b
  11.4 us a layer = 42 us + 5 us of had_in + ~19 us of casts); wo_a bf16 27.8 us at 302 GB/s (8.39 MB); shared experts 30 us
  (8.85 MB); router 13 us. Above 16 rows every EXL3 linear dequantizes its matrix each step (reconstruct + f16 GEMM: wq_a 19.8 + 32.6 us,
  wkv 7.3 + 31.9, wq_b 16.1 + 20.0, wo_b 14.9 + 18.6: 7.6 ms a step at 24 rows and the drafter's 1.4 ms at 20): a multi-row GEMV to 24-32 rows removes
  it.
- **Not in A-D**: the drafter's markov head (5 sequential 66 MB GEMVs + argmax, 0.67 ms a step, replicated on every rank),
  its main_proj (59 MB replicated, 0.12 ms), the target lm_head (165 MB slice: 0.16 ms via wvSplitK at <= 4 rows, 0.29-0.30
  ms via Cijk at 6+), and the expert-load imbalance between ranks.

## Files

Repo: `bench/ds41/ledger/` -- `workload.py` (the fixed workload + timing / profile windows), `zrun.sh` / `zrun2.sh` (one
ledger launch; zrun2 adds PROFCFG=cuda), `zall.sh`, `zrest.sh` (the launch sequences run), `warm.py`, `ledger.py`,
`gputl.py`, `layers.py`, `classavg.py`, `widths.py`, `mklist.py`, `launch-list-T6.txt`.
VM100 `~/ds41/exl3/Z/`: `E1/ G1/ G3/ G4/ G2/ G2b/` (rank traces under `s<N>/`, analyses next to them), `zall.log`,
`zrest.log`, `tools/` (the copies that ran). The launches served `~/r9700-exl3-Z` (v0.4.0, untouched by the other packages).
