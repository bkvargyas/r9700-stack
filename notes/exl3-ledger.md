# DeepSeek-V4.1-Flash EXL3 2.9 bpw -- the step ledger (package Z, Phase 4, 2026-11-03)

Measured on VM100 (eight R9700, SCLK -500 MHz, 120 V feed) from a pristine copy of the v0.4.0 tree (`git archive` of branch
v041 HEAD d79873a -> VM100 `~/r9700-exl3-Z`, libr9k.so of the same kernels), image r9700/vllm:dev, the validated launcher
defaults (serve/ds41-exl3.sh: TP8 + EP, MAXLEN 32768, KVMEM 1.7, NSEQ 4, NBT 2048, DSpark k = 5, adaptive verification on,
HIP graphs, capture sizes 1,2,4,6,12,18,24). Workload: greedy prose (bench/ds41/ledger/workload.py: "3000-word essay about
the history of X", thinking off, ignore_eos), 1 and 4 streams, measured mid-decode. Every number is rank 0 unless marked.
Numbers are per verify step ("a step" = one scheduler step: the target forward, the sampler / rejection sampler, the
drafter's propose). Card time 11:28-12:27 UTC (one hour; cards idle, lock removed, offset back to 0 after).

**The same ledger on v0.4.1 (package Z2) is the last section, "v0.4.1 (2026-11-03)".** Sections 1-5 below are v0.4.0's.

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

## v0.4.1 (2026-11-03)

Package Z2: the same ledger on the v0.4.1 tag. Measured on VM100 (eight R9700, SCLK -500 MHz, 120 V feed) from a pristine copy
of the tag (`git archive v0.4.1` = 9be646b -> VM100 `~/r9700-exl3-Z2`, libr9k.so built from it in r9700/vllm:dev: md5
dea113e9..., the same bytes as the release-validation tree `~/r9700-exl3-REL`), image r9700/vllm:dev, the launcher defaults
(serve/ds41-exl3.sh, unchanged from v0.4.0: TP8 + EP, MAXLEN 32768, KVMEM 1.7, NSEQ 4, NBT 2048, DSpark k = 5, adaptive
verification on, HIP graphs, capture sizes 1,2,4,6,12,18,24). Same workload (workload.py, greedy prose, 1 and 4 streams,
3 s timing window + 3 s profile window mid-decode), same tools (`bench/ds41/ledger/`, now with the v0.4.1 kernel categories;
sequence `zv041.sh`, analyses `zana.sh`). Rank 0 unless marked; "a step" = one scheduler step (target forward, sampler /
rejection sampler, drafter propose). Card time 2026-11-04 00:03-00:37 UTC (34 min; cards idle, lock removed, offset back to 0
after; package G took the box at 00:36).

### Z2.0 What changed since v0.4.0 and what is left (read your row)

1. **The default single-stream step is 19.1 ms (was 25.4): 105.3 tok/s at 2.01 accepted tokens a step (was 74.3 at 1.89).**
   GPU kernels 17.75 ms (was 23.46), launch + CPU gaps <= 1.36 ms (7 %). Four streams: **32.3 ms (was 40.0), 248 tok/s
   aggregate (was 202)**, kernels 31.1 ms, gaps <= 1.1 ms. Launches a step 1,132 / 1,318 (was 1,865 / 2,082).
2. **C's flat-all-gather fix is the biggest single gain: 4.05 -> 0.19 ms a step** (Engram 4 x 22 us + logits 101 us on
   `r9k_ag8_os_kernel`). That is also why adaptive verification now wins single-stream (105 vs 97 tok/s with it off; v0.4.0:
   74 vs 81).
3. **The 81 all-reduces did not get cheaper at rank 0: 5.24 ms a step at T = 6 (64.6 us a call; v0.4.0 5.9).** The one-kernel
   all-reduce itself takes 23 us a call (min over the eight ranks, 20-40 KiB); **the other ~3.4 ms a step is rank 0 waiting
   for the slowest rank** -- the post-MoE call averages 68 us against 25 after attention (expert-load imbalance across the EP
   ranks plus arrival skew). At 24 rows: 152.8 us a call rank 0, 73 min (12.4 ms vs 5.9 a step). **Nobody owns this row.**
4. **C's fused MoE reduction never runs in the server**: every launch logs "exl3 MoE combine fusion skipped ... (runner shape
   changed)" for all 43 runners (also in the release-validation logs `D4/clean-REL2.log`, `C4/bisect-C.log`). The likely
   cause is `moe/exl3_combine.py`'s gate `cfg.tp_size == 8`: with `--enable-expert-parallel` the pin's FusedMoEParallelConfig
   has tp_size 1, ep_size 8. Trace: `r9k_ar8_os_kernel<bf16, 0>` only (no MoE variant), plus one add a layer.
5. **D's GEMV removed the dequant path and the casts** (EXL3 GEMVs + wo_a + casts: ~3.3 -> 2.27 ms at T = 6, 7.6 + 1.2 -> 4.63
   ms at 24 rows), but runs at 250-270 GB/s at 6 rows (wo_a 166) -- below exllamav3's 288-292 per kernel; the gain is the
   launches it removed. **F's row.**
6. **B's FFN front works (48 launches, 1.43 ms; plain layer 32 -> 18 launches), B's markov head does not under the default
   adaptive verification**: with ADAPTIVE=false the drafter runs `r9k_dsv4_markov_kernel` (0.37 ms), with the defaults it runs
   stock's five [129280 x 256] GEMVs + argmax (0.67 ms) plus adaptive's f32 Cijk GEMM (0.62 ms) -- the drafter tail is 2.07 ms
   of a 19.1 ms step. (The hook falls back somewhere in `dsv4_glue._markov_why_not`; "probabilistic drafts" is the suspect,
   not confirmed.) Remaining glue: 325 + 272 (drafter) us, 173 + 61 launches.
7. **Expert tier: 2.38 ms at T = 6 default (2-4 rows), 3.80 at 6 rows, 7.18 at 9-16 rows, 12.52 at 24 rows** -- unchanged
   from v0.4.0 (A's tier was held back), now 13-30 % of the step. **A2's row.**
8. Above 8 rows B's kernels and the 6-row attention-input fusion fall back to stock (shared w13 / w2, router gate, top-k:
   62 us a layer at 9-24 rows vs 52 at 6); above 192 KiB the logits gathers go to RCCL (target 334-574 us, drafter 479-492):
   0.8-1.1 ms of the four-stream step.
9. Unchanged: decode attention 0.87-2.1 ms, hc seams 0.69-1.14 ms, KV 329,008 tokens at KVMEM 1.7 (model 26.32 GiB a rank,
   graph pools 1.22 GiB, was 1.41). NSEQ 8 is still unsupported (no T = 48 column).
10. Profiling still inflates the step (+5.5 ms at T = 6: traced period 24.6 vs 19.1; ~6 us between graph nodes): read traced
    kernel durations, never traced periods.

| v0.4.2 package | its row at T = 6 (default, 1 stream) | at T = 24 (default, 4 streams; 9-16 rows run) | at a true 24-row verify | the ceiling it can reach |
|---|---:|---:|---:|---|
| A2 expert tier to 48 tokens (main + drafter) | 2.47 ms (14 %) | 7.44 ms (24 %) | 12.77 ms | no gain owed at 1-6 rows (must hold 50 / 114 us a layer at 2-4 / 6 rows); at 24 rows A's card-5 ratio (518 -> 414-435, -17-20 %) is ~-2.3 ms; its real prize is the 8-stream (48-token) shape that does not launch today |
| F EXL3 GEMV latency (r9k_exl3_gemv, main + drafter) | 2.66 ms (15 %) | 4.22 ms (14 %) | 5.34 ms | the four attention GEMVs of a layer read 14.9 MB: 63 us at 6 rows (235 GB/s) -> 33 us at 450 GB/s = **-1.2 ms a step** at T = 6 (+ -0.1 drafter); 112 us at 24 rows -> 33 = -3.2 ms |
| G split-path all-reduce at 240-480 KiB (+ the > 192 KiB gathers) | 0 (all <= 64 KiB, one-kernel) | 9.93 ms rank 0, 4.87 ms min over ranks (88 calls at 90-200 KiB) + RCCL gathers 0.82 ms | 13.30 ms rank 0, 6.38 min | min 54 -> ~40 us at 90-160 KiB = -1.1 ms; at 240 KiB 73 -> 50 us = -1.9 ms; gathers on a host-pool path ~-0.5 ms. Rank skew (rank 0 minus min: 5.1-6.9 ms at four streams) is not G's to win |
| E KV headroom | no step time | -- | -- | capacity only: 329,008 KV tokens at KVMEM 1.7 (unchanged), worst card ~0.86 GiB free in the release soak |
| (unowned) rank skew in the 81 target all-reduces (rank 0 minus min over ranks) | ~3.4 ms (64.6 vs 23.2 us a call) | ~4.6 ms (G1) | ~6.5 ms | upper bound if every rank arrived with rank 0's work: T = 6 step ~15.7 ms (~128 tok/s at 2.01); really bounded by how evenly the routed experts spread over the EP ranks |
| (unowned / B follow-up) drafter tail under adaptive | 1.7 ms (stock markov 0.67 + f32 GEMM 0.62 + lm_head 0.27 + gather 0.15) | 2.1 ms | -- | B's markov kernel under adaptive: -0.3 ms; the adaptive f32 GEMM on a GEMV: ~-0.5 ms |
| (C follow-up) MoE combine inside the all-reduce | not running (gate fails under EP) | -- | -- | C measured 40.9 -> 18.5 us at T = 1: up to ~-0.9 ms a step at 1-4 rows once it installs |
| **step, unprofiled** | **19.11 ms** (25.42) | **32.26 ms** (40.00) | **43.48 ms** (52.63) | |

### Z2.1 Launches and method

| tag | launch | workloads | traces (VM100 `~/ds41/exl3/Z2/`) | what it is used for |
|---|---|---|---|---|
| G4 | defaults (ADAPTIVE=true), **GPU-only** profiler | 1 and 4 streams, 3 s | `G4/s1/`, `G4/s4/` (8 rank traces each) | the headline table, the wall split |
| G3 | **ADAPTIVE=false**, CPU+GPU profiler | 1 and 4 streams, 3 s | `G3/s1/`, `G3/s4/` | the true 6- and 24-row verify, per-layer classes, cross-rank minima |
| G1 | defaults, CPU+GPU profiler | 1 and 4 streams, 3 s | `G1/s1/`, `G1/s4/` | verify widths, per-layer at 2-4 and 9-16 rows, cross-rank minima at the default widths |
| E1 | EAGER=1, ADAPTIVE=false, stacks + shapes | 1 stream 6 s (the 4-stream profile window failed: stop_profile errored while the traces were written, `E1/s4/` holds 7 partial traces, unused) | `E1/s1/` | kernel names -> call sites, draft-names.json, the launch list |

Logs: `zv041.log` (the sequence), `<tag>/run.log`, `<tag>/launch.log`, `<tag>/s<N>.json` (both windows), `build.log`.
Analyses next to the traces: `<tag>/s<N>-ledger.txt|json`, `-list.txt`, `-layers.txt`, `-classavg.txt`, `-widths.txt`,
`G4/s<N>-gputl.txt`, `E1/launch-list-T6.txt` (in the repo as `bench/ds41/ledger/launch-list-T6-v041.txt`).
Tool changes (bench/ds41/ledger/): `ledger.py` categories for D's `k_gemv` / `k_gemv_mma` (anonymous namespace) and B's
`r9k_dsv4_sk` / `r9k_dsv4_norm` / `r9k_dsv4_markov`, a collectives-by-kernel listing in `ledger.py` and `gputl.py`;
`widths.py` reads the verify width off `k_gemv<bits, cb, M>` (M = rows, 1-8) / `k_gemv_mma<.., RT, ..>` (9-48 rows);
`mklist.py` labels the four attention GEMVs of a layer and their bytes; `zrun2.sh` takes ZTREE / ZOUT / ZLOCK (defaults =
package Z's); new `zv041.sh`, `zana.sh`. Build note: the tag's `kernels/build.sh` export check failed once on a SIGPIPE race
(`set -o pipefail` + `nm -D | grep -q`, "MISSING EXPORT: r9k_ag_oneshot_nrank_g" for a symbol that is there); all 129
exports verified into a file, the .so installed by hand.

**The verify width the GPU ran** (`widths.py`):

| launch | steps | rows verified on the GPU |
|---|---:|---|
| G1 1 stream, adaptive | 116 | 4 rows 50.9 %, 2 rows 46.6 %, 6 rows 2.6 % |
| G1 4 streams, adaptive | 78 | 9-16 rows 92.3 %, 17-24 rows 7.7 % |
| G3 1 stream, ADAPTIVE=false | 115 | 6 rows 100 % |
| G3 4 streams, ADAPTIVE=false | 63 | 24 rows 100 % (no dequant path: 0 reconstructs; engram wkv 6 exllamav3 launches) |

### Z2.2 GPU time a step by category, us [launches a step]

Rank 0, graph replay, sum of kernel durations a step (traced; collectives include waiting for the slowest rank, Z2.3).
G4 columns: `gputl.py` (phases by graph replay; its "sampler" row also holds the forward's eager input prep); G3: `ledger.py`.
G1 agrees with G4 within 1-3 % on every row (`G1/s1-ledger.txt` 17,691 us, 1,132 launches; `G1/s4-ledger.txt` 31,296, 1,319).
v0.4.0 in parentheses (section 1).

| category | T=6 default (G4) | T=6 ADAPTIVE=false (G3) | T=24 default (G4) | T=24 ADAPTIVE=false (G3) |
|---|---:|---:|---:|---:|
| expert tier r9k k_gu / k_dn | 2,383 [80] (2,503) | 3,798 [80] (3,857) | 7,184 [80] (6,814) | 12,524 [80] (12,636) |
| EXL3 GEMV r9k_exl3_gemv (k_gemv / k_gemv_mma; incl. wo_a, no casts) | 2,273 [162] | 2,691 [162] | 3,513 [162] | 4,634 [160] |
| EXL3 exllamav3 path (+ dequant f16 GEMMs) | 0 (1,677 [260]) | 0 (2,068 [260]) | 8 [0.3] (3,593 [490]) | 122 + 85 [8] (3,059 + 4,547 [688]) |
| B r9k_dsv4_sk (FFN front x 40 + attention input x 8; <= 8 rows) | 1,429 [48] | 1,725 [48] | 0 (stock above 8 rows) | 0 |
| hipBLAS bf16 (shared w2; above 8 rows also shared w13, router gate, compressor / indexer; target lm_head) | 561 [45] (2,709 [177]) | 878 [45] (3,456) | 2,560 [137] (3,706) | 2,626 [137] (3,605) |
| router top-k (stock, above 8 rows) | 0 (332 [40]) | 0 (335) | 340 [40] | 342 [40] |
| decode attention: split-K partial + reduce | 869 [80] (860) | 1,011 [80] (952) | 1,383 [80] (1,276) | 2,107 [80] (1,781) |
| attention: indexer | 231 [46] | 183 [54] | 112 [46] | 148 [54] |
| attention: KV writes / RoPE / compressor | 109 [48] | 130 [53] | 116 [48] | 141 [53] |
| collectives (all-reduces, Engram + logits gathers) | 5,425 [86] (9,568 [149]) | 6,302 [86] (9,007) | 9,558 [86] (13,318) | 13,334 [86] (15,537) |
| hc seams | 691 [87] (695) | 800 [87] (794) | 861 [87] (846) | 1,142 [87] (1,144) |
| norms / casts / elementwise glue | 325 [173] (1,240 [564]) | 453 [219] (1,381 [609]) | 489 [261] (1,302) | 659 [312] (2,190) |
| Engram (hash, lookup) | 21 [5] | 25 [5] | 26 [5] | 29 [5] |
| sampler / rejection sampler (G4: + eager input prep) | 386 [137] (394) | 111 [35] (121) | 405 [141] (415) | 144 [39] (144) |
| everything else | 3 [1] | 4 [1] | 6 [1] | 0 |
| **drafter**: expert tier | 92 [6] (101) | 102 [6] (100) | 260 [6] (258) | 244 [6] (252) |
| drafter: EXL3 GEMV (r9k_exl3_gemv + exllamav3's main_proj at 5 rows) | 262 + 123 [17] (329) | 263 + 118 [17] (325) | 711 [16] (601 + 823) | 711 [16] (601 + 824) |
| drafter: hipBLAS (lm_head slice; stock markov head x 5 and adaptive's f32 GEMM when adaptive; shared w2) | 1,464 [11] (1,608) | 308 [5] (985) | 1,675 [20] (1,763) | 500 [14] (1,138) |
| drafter: B markov head r9k_dsv4_markov | 0 (falls back under adaptive) | 365 [6] | 0 | 404 [6] |
| drafter: B r9k_dsv4_sk | 83 [3] | 83 [3] | 0 | 0 |
| drafter: collectives (7 all-reduces + logits gather) | 622 [8] (690) | 626 [8] (685) | 1,408 [8] (1,497) | 1,419 [8] (1,434) |
| drafter: glue / attention / hc / router / other | 399 [89] (497 [131]) | 215 [59] (489) | 506 [94] (645) | 295 [64] (641) |
| **GPU kernel time a step** | **17,750** (23,460) | **20,189** (24,866) | **31,120** (37,450) | **41,609** (50,180) |
| launches a step | **1,132** (1,865) | **1,059** (1,754) | **1,318** (2,082) | **1,256** (2,221) |
| **step, unprofiled** (timing window) | **19,110** (25,420) | **21,740** (27,270) | **32,260** (40,000) | **43,480** (52,630) |

### Z2.3 Collectives by kernel and size (per call; rank 0, and the minimum over the eight ranks)

| call (a step) | message a rank | kernel / path | rank 0 us | min over ranks | calls | source |
|---|---|---|---:|---:|---:|---|
| all-reduce [2-4, 5120] (default, 1 stream) | 20-40 KiB | **one-kernel `r9k_ar8_os_kernel`** (C) | 64.6 (after attention 25.2, after the MoE 68.1) | **23.2** | 81 | G4 s1, G1 s1 classavg / --ranks |
| all-reduce [6, 5120] | 60 KiB | one-kernel (R9K_AR8_OS_KB 64) | 73.9 (40.7 / 90.8) | 32.8 | 81 | G3 s1 |
| all-reduce [9-16, 5120] | 90-160 KiB | split path `r9k_ar8_fused_kernel` (one lane, 1024-word sub-steps) | 110.8 (61.7 / 134.8) | 54.4 | 81 | G1 s4 |
| all-reduce [24, 5120] | 240 KiB | split path | 152.8 (84.8 / 219.4) | 73.1 | 81 | G3 s4 |
| drafter all-reduce [5, 5120] / [20, 5120] | 50 / 200 KiB | one-kernel / split path | 67.4 / 132.0 | 30.5 / 65.3 | 7 | G4 s1, G3 s1 / G4 s4, G1 s4 |
| Engram all-gathers [T, 3200] | 6-150 KiB gathered | **one-kernel `r9k_ag8_os_kernel`** (was flat 8-rank 653-1,069 + pair / quartet stages) | 21.6 (2-4 rows), 32.6 (6), 55 (9-16), 96 (24) | 18.2 / 28.9 / 51.6 / 89.5 | 4 | G4 s1, G3 s1, G1 s4, G3 s4 |
| target logits all-gather [T, 16160] at <= 6 rows | 63-189 KiB | one-kernel ag8 (was flat 8-rank 3,205 at <= 4 rows) | 101 (2-4 rows), 185 (6) | 93 / 175 | 1 | G4 s1, G1 s1, G3 s1 |
| target logits all-gather at 9-24 rows | 284-758 KiB | RCCL `ncclDevKernel_Generic_4` | 334-337 (9-16), 574 (24) | 329 / 565 | 1 | G4 s4, G1 s4, G3 s4 |
| drafter logits all-gather [5, 16160] / [20, 16160] | 158 / 631 KiB | one-kernel ag8 / RCCL | 141-151 / 479-492 | 141 / 478 | 1 | G1 s1, G3 s1 / G1 s4, G3 s4 |
| the flat eight-rank all-gather `r9k_ag_oneshot_nrank_kernel<8>` | | | **0 calls** (v0.4.0: 4,050 us a step) | | 0 | |

Sum of the minima a step (the collective cost if the ranks arrived together): 1.88 ms at T = 6 default (rank 0 5.24 + 0.47
drafter), 2.65 at 6 rows, 4.41 at 9-16 rows, 5.92 at 24 rows. The rest of the rank-0 time is waiting, and it sits on the
post-MoE call (classavg: 68 vs 25 us at 2-4 rows, 219 vs 85 at 24 rows; per-layer k_gu on rank 0 ranges 5-64 us at 2-4 rows):
expert-load imbalance between the EP ranks plus arrival skew from the preceding compute.

### Z2.4 Per call / per layer (graph replay; class means over the 26 plain layers)

| kernel | T=2-4 (G1 s1) | T=6 (G3 s1) | T=9-16 (G1 s4) | T=24 (G3 s4) | bytes a rank | GB/s at T=6 | v0.4.0 at T=6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| k_gu + k_dn a K=3 layer | 32.4 + 17.7 | 76.3 + 37.5 (6-176 / 4-77 by layer) | 134.4 + 58.0 | 233.5 + 87.2 | distinct local experts x 13.27 MB | -- | 65.0 + 31.1 |
| fused_wqa_wkv k_gemv (5120 -> 1792, K=5) | 17.4 | 21.5 | 28.7 (k_gemv_mma RT 1) | 41.2 (RT 2) | 5.73 MB | 267 | 19.6 + had_in 2.0 + casts 7.9 |
| wq_b k_gemv (1280 -> 4096) | 10.5 | 13.2 | 17.5 | 23.1 | 3.28 MB | 248 | 11.4 + 1.4 + 5.0 |
| **wo_a on EXL3** k_gemv (4096 -> 1024 slice) | 13.3 | 15.8 | 17.9 | 25.0 | 2.62 MB | **166** | 27.8 (bf16, 8.39 MB) |
| wo_b k_gemv (1024 -> 5120) | 11.1 | 12.9 | 16.6 | 23.0 | 3.28 MB | 254 | 11.4 + 1.5 + 5.0 |
| **FFN front** r9k_dsv4_sk (ffn_norm + router gate + top-k + shared w13 + silu) + shared w2 | 30.8 + 6.5 | 38.2 + 14.0 | stock: 1.9 + 19.8 + 1.5 + 15.4 + 14.7 + 8.5 = 61.8 | stock 62.4 | 3.93 + 5.90 + 2.95 MB | -- | 55.0 (6 launches) |
| decode attention partial + reduce | 15.4 + 5.1 | 23.0 + 3.3 | 30.9 + 3.5 | 51.0 + 4.2 | | | 22.8 + 2.6 |
| all-reduce after attention / after the MoE | 25.2 / 68.1 | 40.7 / 90.8 | 61.7 / 134.8 | 84.8 / 219.4 | | | 46.3 / 98.8 |
| r9k_mhc seam | 8.2 | 9.7-10.1 | 10.2-11.1 | 13.4-14.6 | 1.97 MB | 200 | 9.7-9.9 |
| **a plain layer, all launches** | **277 (18)** | **414 (18)** | **595 (22)** | **891 (22)** | | | 449 (32) |

A plain layer at T = 6 is now 18 launches (v0.4.0: 32): seam, rms_norm, wqa_wkv, q_norm (Triton), wq_b, q-norm/RoPE/KV insert,
attention partial + reduce, wo_a, wo_b, all-reduce, seam, FFN front (sk), shared w2, k_gu, k_dn, add (routed + shared),
all-reduce. The other classes (G3 s1): layer 0 24 launches, Engram layer 1 19, kv-source layers 2 / 8 / 14 30, layer 13 23,
layer 20 36, indexer layers 24 / 28 / 32 30, layer 36 32, aux layers 37 / 38 20, last layer 39 26; drafter layers 19 each, its
last layer + heads 36 (ADAPTIVE=false, B's markov kernel: 5 x 52-131 us + 4) or 62 (defaults: stock markov 5 x (107 + 27.5
argmax) + adaptive's f32 Cijk MT256x128x8 625 us). The eager launch list of one T = 6 step (E1, 1,013 launches, grid / block,
bytes, GB/s, cpu op, shapes): `bench/ds41/ledger/launch-list-T6-v041.txt`.

The GEMVs: at 2-4 rows they reach 296-329 GB/s (wo_a 197), at 6 rows 248-267 (wo_a 166), at 24 rows 139-143 -- every one
carries D's ~9 us fixed chain (rotation into LDS, then the stream); wo_a, the smallest matrix with the longest K, suffers most.

**Launch counts a step by category** (G3 = ADAPTIVE=false, G4 = defaults; T = 6):

| category | target forward + sampler, G3 | drafter, G3 | target forward + sampler, G4 | drafter, G4 | v0.4.0 G3 (target / drafter) |
|---|---:|---:|---:|---:|---:|
| norms / casts / elementwise glue | 219 | 38 | 173 | 61 | 609 / 91 |
| EXL3 GEMV (r9k_exl3_gemv; + exllamav3) | 162 | 15 + 2 | 162 | 15 + 2 | 260 / 26 |
| B r9k_dsv4_sk | 48 | 3 | 48 | 3 | -- |
| B markov head | -- | 6 | -- | 0 | -- |
| hipBLAS bf16 | 45 | 5 | 45 | 11 | 177 / 22 |
| collectives | 86 (81 all-reduces, 4 Engram + 1 logits gathers) | 8 | 86 | 8 | 90 / 8 |
| hc seams | 87 | 7 | 87 | 7 | 87 / 7 |
| expert tier | 80 | 6 | 80 | 6 | 80 / 6 |
| decode attention | 80 | 6 | 80 | 6 | 80 / 6 |
| indexer / KV writes / compressor | 107 | 6 | 94 | 6 | 107 / 6 |
| router top-k | 0 | 0 | 0 | 0 | 40 / 3 |
| sampler / rejection sampler (G4: + eager input prep) | 35 | -- | 137 | -- | 36 / -- |
| Engram, other | 6 | 2 | 6 | 9 | 6 / 7 |
| **total** | **955** | **104** | **998** | **134** | 1,572 / 182 |

### Z2.5 Wall-clock split and the tok/s arithmetic

Unprofiled timing windows (3 s, mid-decode; `<tag>/s<N>.json`, window "timing"). drafts = verify steps x streams.

| launch | streams | tokens a stream | drafts | accepted | MAL | steps | step wall | (drafts + accepted) / streams / 3 s | observed tok/s a stream (aggregate) | v0.4.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G4 defaults | 1 | 316 | 157 | 159 | 2.01 | 157 | 3000 / 157 = **19.11 ms** | 105.3 | **105.3** | 74.3 |
| G1 defaults | 1 | 306 | 157 | 149 | 1.95 | 157 | 19.11 ms | 102.0 | 102.0 | 71.0 |
| G3 ADAPTIVE=false | 1 | 291 | 138 | 153 | 2.11 | 138 | 21.74 ms | 97.0 | 97.0 | 80.7 |
| G4 defaults | 4 | 175 / 194 / 193 / 181 | 372 | 371 | 2.00 | 93 | **32.26 ms** | 61.9 | 61.9 (**247.7**) | 50.6 (202.3) |
| G1 defaults | 4 | 179 / 175 / 172 / 190 | 376 | 340 | 1.90 | 94 | 31.92 ms | 59.7 | 59.7 (238.7) | 46.4 (185.3) |
| G3 ADAPTIVE=false | 4 | 149 / 152 / 130 / 128 | 276 | 283 | 2.03 | 69 | 43.48 ms | 46.6 | 46.6 (186.3) | 38.7 (154.7) |
| E1 eager, ADAPTIVE=false | 1 | 137 | 60 | 77 | 2.28 | 60 | 100.0 ms | 22.8 | 22.8 | 17.8 |

So tok/s = MAL / step wall: 2.01 / 19.11 ms = 105.3 single stream; 4 x 2.00 / 32.26 ms = 247.7 aggregate. (spectune's release
numbers, 87-97 prose single / 179-212 at four, run a different prompt set and include prefill and the request tail; this
workload measures mid-decode only.)

**Split of the unprofiled step** (G4: GPU-only trace by graph replay; G3: by the CPU annotation phases):

| | T = 6 default (G4) | T = 24 default (G4) | T = 6 ADAPTIVE=false (G3) | T = 24 ADAPTIVE=false (G3) |
|---|---:|---:|---:|---:|
| step wall, unprofiled | 19.11 ms (25.42) | 32.26 (40.00) | 21.74 (27.27) | 43.48 (52.63) |
| target forward graph replay, kernel time | 13.95 (16.37) | 25.52 (30.84) | 17.51 incl. eager input prep | 37.01 |
| drafter graph replay | 2.99 (3.17) | 4.51 (5.55) | 2.08 (draft phase) | 3.57 |
| eager kernels (input prep, lm_head, logits gather, sampler / rejection) | 0.81 (3.92) | 1.09 (1.06) | 0.60 (sample phase) | 1.02 |
| **all kernels** | **17.75** (23.46) | **31.12** (37.45) | **20.19** (24.87) | **41.61** (50.18) |
| **launch gaps + CPU** | **<= 1.36 (7.1 %)** | **<= 1.14 (3.5 %)** | **<= 1.55 (7.1 %)** | **<= 1.87 (4.3 %)** |

Traced, the same steps run 24.65 / 38.99 ms (G4) with 5.2 / 6.1 ms of idle inside the forward replay (~6 us a node), as on
v0.4.0: periods come from the unprofiled windows.

**Adaptive verification, true vs false (one stream)**: GPU time 17.75 vs 20.19 ms (-2.4: expert tier -1.4, GEMVs -0.4, FFN
front -0.3, logits gather 101 vs 185 us), drafter +0.9 ms (stock markov + f32 GEMM), MAL 2.01 vs 2.11, **105.3 vs 97.0 tok/s
(+9 %)**; v0.4.0 had it the other way (-8 to -12 %) because of the flat all-gather. Four streams: 247.7 vs 186.3 aggregate (+33 %).
