# DeepSeek-V4.1-Flash EXL3: expert-parallel rank skew (package H3, v0.4.2, 2026-11-04)

The row: Z2's ledger (notes/exl3-ledger.md, v0.4.1, Z2.3) found rank 0 waiting ~3.4 ms a step (T = 6 default) / ~6.5 ms (24
rows) in the 81 all-reduces, mostly on the post-MoE call. This note measures where that wait comes from, costs the remedies,
and ships the cheapest one that pays: a load-time expert placement map (`R9K_EXL3_EP_MAP`).

Card time on VM100 (eight R9700, SCLK -500 MHz), image r9700/vllm:dev, trees from `git archive v0.4.1` plus only H3's files
(`~/ds41/exl3/H3/t` for the measurement, `t2` = + the placement module and its hook, for the A/B; libr9k.so = Z2's build,
md5 dea113e9...). Launcher defaults throughout (TP8 + EP, DSpark k = 5, adaptive verification, NSEQ 4, KVMEM 1.7, HIP graphs).
Launches: M1 02:50-03:04 UTC (recorder + profiles + histograms), AB 03:43-04:04, AB2 06:1x-06:34. Logs and data:
VM100 `~/ds41/exl3/H3/{M1,AB-A,AB-B,AB2-A,AB2-B}/`.

## 1. Where the wait comes from

### 1a. It is the expert tier, call by call (Z2's traces, re-read per rank)

`bench/ds41/epskew/skew_traces.py` matches every MoE call across the eight rank traces (by the post-MoE all-reduce's end
time; the eight ranks leave it within ~6 us) and splits each rank's time before the all-reduce. Per target MoE call:

| window (Z2) | tier k_gu + k_dn, mean over ranks / max over ranks (us) | segment before the post-MoE all-reduce, mean / max | post-MoE all-reduce rank 0 / min | late rank = rank with the slowest tier | a step: tier skew (max - mean) | of which persistent within the window |
|---|---:|---:|---:|---:|---:|---:|
| G4 s1 (default, 2-4 rows) | 58.3 / 114.0 | 155.2 / 211.3 | 79.2 / 23.3 | 93 % of calls | **2.20 ms** | 1.14 |
| G3 s1 (6 rows) | 95.1 / 169.5 | 205.4 / 280.0 | 107.2 / 32.1 | 96 % | 2.97 ms | 1.77 |
| G1 s4 (9-16 rows) | 184.1 / 290.4 | 326.6 / 433.0 | 159.1 / 52.5 | 97 % | 4.25 ms | 2.85 |
| G3 s4 (24 rows) | 315.2 / 464.0 | 462.4 / 610.7 | 221.7 / 68.5 | 98 % | 6.22 ms | 4.87 |

* The segment skew (what every rank waits) equals the tier skew to 0.01-0.02 ms: **the wait is the expert tier being
  uneven across ranks, nothing else** (attention, hc seams, FFN front are balanced: the post-attention all-reduce waits only
  0.2-0.3 ms a step).
* The slow rank moves. Every rank is the hottest in 9-16 % of calls; for a given layer the most frequent hot rank holds it in
  20-65 % of steps. The drafter is the exception: rank 6 is the hottest in 52-60 % of drafter calls (tier 84-86 us against
  30-50 on the others; its post-MoE all-reduce is 51 us on rank 6 and 90-108 elsewhere).
* "Persistent within the window" = max - mean of the per-layer rank means over the 3 s window. It is an upper bound on what a
  static placement can win, because one window holds one prompt.

### 1b. The tier's cost on a rank (fit on recorded routing + the same window's traces)

Recorder: `bench/ds41/epskew/epskew_ext.py`, a worker extension that wraps Exl3MoEMethod.apply and writes each decode call's
top-k ids into a device ring buffer (`index_copy_` at a device pointer, so the HIP graphs replay it). Every rank records
(a recorder on rank 0 only would make rank 0 late). Its own launch reproduces Z2's skew (2.36 vs 2.21 ms at one stream,
4.21 vs 4.26 at four), so it does not distort what it measures. `skew_model.py fit` aligns the ring with the traces call by
call. The correlation between distinct local experts and k_gu + k_dn time is 0.96-0.995 by layer. Per rank and call:

    tier = a + b * n + c * (s - n)      n = distinct local experts, s = local slots (token, pick)

| K (bits) | verify slots | a us | b us per distinct expert | c us per extra slot on a shared expert | R2 |
|---|---:|---:|---:|---:|---:|
| 3 | 12 (2 tokens) | 8.4 | 28.6 | 2.2 | 0.998 |
| 3 | 24 (4) | 11.9 | 27.2 | 5.7 | 0.990 |
| 3 | 72 (12) | 20.7 | 27.0 | 6.3 | 0.989 |
| 3 | 108 (18) | 31.3 | 26.6 | 6.5 | 0.986 |
| 2 (layers 18-22) | 24 / 72 | 13.8 / 16.4 | 22.5 / 24.3 | 7.7 / 7.6 | 0.99 |
| 4 (drafter) | 15 / 60 | 10.1 / 16.7 | 35.2 / 32.9 | 4.6 / 6.2 | 0.98-0.99 |

So **one more distinct expert on a rank costs ~27 us** (13.27 MB at ~490 GB/s), and an extra token on an expert it already
reads costs only 2-7 us. The model reproduces the measured sum over calls of the max over ranks within 1 % (566.9 vs 572.7 ms
over 4,733 calls at one stream, 894.5 vs 900.2 over 3,271 at four).

### 1c. The routing distribution (12 prose + 12 code prompts, 1 and 4 streams, 619-4,096 steps each)

`skew_model.py hist` (M1 `hist.json`). Target layers, mean over the 40:

| workload | steps | slots a step | distinct local experts a rank a step: mean / max over ranks | picks a rank a step: max / mean | top-1 / 8 / 24 / 48 experts' share of picks (uniform: 0.3 / 2.1 / 6.3 / 12.5 %) | worst rank's expected load, x mean |
|---|---:|---:|---:|---:|---:|---:|
| prose, 1 stream | 4,096 | 19.3 | 1.72 / 3.72 | 2.35 | 7.1 / 28.7 / 51.4 / 69.0 % | 1.57 |
| code, 1 stream | 2,346 | 34.1 | 2.68 / 5.08 | 2.16 | 4.5 / 21.1 / 41.6 / 59.9 % | 1.40 |
| prose, 4 streams | 1,433 | 63.7 | 4.62 / 7.71 | 1.87 | 7.3 / 29.7 / 52.6 / 70.3 % | 1.57 |
| code, 4 streams | 619 | 126.6 | 7.86 / 11.55 | 1.69 | 4.5 / 21.2 / 41.8 / 60.3 % | 1.40 |
| mixed, 4 streams | 1,058 | 83.3 | 5.93 / 9.27 | 1.76 | 4.4 / 20.6 / 39.0 / 55.6 % | 1.34 |

* Routing is very concentrated: 8 of 384 experts take 21-30 % of the picks and 48 take 56-70 %. **But which experts are hot
  depends on the domain.** The per-expert frequencies correlate 0.99 between prose runs and 0.80-0.91 between the prose sets
  and the held-out 3000-word essay windows, but -0.06 between prose and code.
* **Structural vs noise.** Under stock's linear placement, the real routing has the same tier skew as uniform random routing
  with the same verify widths (`skew_noise.py`: 2.58 vs 2.70 ms a step at one stream, 4.61 vs 4.47 at four; real routing is
  cheaper overall because hot experts repeat). Most of the skew is per-step multinomial noise: 12-24 picks a step dealt over
  8 ranks puts 3-4 distinct experts on the worst rank against a mean of 1.7. That noise is bounded below by the per-step
  routing itself and no static placement removes it. The structural part, that is hot experts that happen to share a rank,
  is what a placement can fix. Measured below: 0.5 ms a step at one stream, 1.3 ms at four.

## 2. The remedies, costed

Model numbers: `skew_place.py` evaluates each policy on recorded routing with the 1b cost (tier ms a step = sum over the 43 MoE
layers of the max over ranks). Since the late rank is the slow-tier rank in 93-98 % of calls, the change in that sum is the
change in the step. Maps were trained on prose + code (1 stream, 6,442 steps). "Held out" = the 3000-word essay windows
(prof-s1 / prof-s4), which were not in the training set. T = 6 column = one stream at the default adaptive widths (2-4 rows);
T = 24 = four streams (9-24 rows).

| remedy | ms a step, T = 6 (1 stream) | ms a step, T = 24 (4 streams) | memory | runtime cost | work | verdict |
|---|---:|---:|---|---|---|---|
| (a) **load-time placement map** from the routing histogram | model -0.62 held-out (-0.83 prose / -1.00 code in-sample); **measured -0.59 wall, -0.51 tier skew** | model -1.60 held-out (-1.48 prose / -1.94 code); **measured -1.02 wall, -1.31 tier skew** | 0 | 0 (load time only) | built (section 3) | **ships behind the knob** |
| (a) map from one domain, served on the other | prose map on code ~0 (+0.04); code map on prose -0.1 | prose map on code +0.17; code map on prose -0.19 | 0 | 0 | -- | train on the served mix |
| (b) + replicas, device-side least-loaded copy a step, 8 extra copies a layer (1 a rank a layer) | -0.24 more (model) | -0.43 more | 40 x 12.66 MiB = **0.49 GiB a rank** (the worst card has ~0.86 GiB free; or KVMEM -0.5 = -96k KV tokens) | one per-step "which copy" pass a layer: all ranks compute the same greedy assignment from the replicated routing (no communication). As a separate kernel ~3-5 us x 43 = 0.13-0.2 ms, which eats most of the gain; free only inside the tier's slot-grouping prologue (A2's kernel) | loader with 48 + r local slots per layer (the stash maps one global id to two locals), a dynamic g2l table per step, the prologue change | not now: <= 0.25 / 0.45 net for 0.5 GiB |
| (b) 16 / 32 extra copies a layer | -0.37 / -0.55 more | -0.77 / -1.19 more | 0.99 / 1.98 GiB a rank | as above | as above | does not fit at KVMEM 1.7 |
| (c) stock EPLB (`--enable-eplb`, redundant experts, token-hash replica choice, periodic rearrangement) | with 8 redundant copies a layer: -0.37 (vs -0.62 for the map alone) | -1.02 (vs -1.60) | as (b) | rearrangement every `step_interval` (3,000) steps: up to all 40 x 48 x 12.66 MiB = 24 GB a rank moved over RCCL | **not supported for us today**: RoutedExperts raises NotImplementedError ("EPLB is not supported Exl3MoEMethod", supports_eplb False). It would need supports_eplb + get_expert_weights over the nine [E_local, ...] buffers (the pointer tables stay valid for in-place moves); redundant slots in the loader; and the logical -> physical map + load recording inside B's fused top-k front (r9k_dsv4_sk bypasses stock's router at <= 8 rows, the decode path). The pinned model (DeepseekV4MixtureOfExperts) itself supports it | **worse than the static map**: the token hash is load-blind, so a hot expert lands on several hosts in one step and each pays a full weight read. Its only advantage is following workload drift, which a re-run of skew_place.py plus a restart also gives |
| (d) accept and hide: overlap the post-MoE all-reduce with the next layer | ~0 | ~0 | 0 | -- | -- | the chain is strictly serial: the all-reduced MoE output feeds the hc seam, which mixes the four residual streams through hc_fn of that output and produces the next layer's attention input. Nothing of size in the next layer is independent of it (Engram's hash + host lookup is, but costs 21 us a step). Moving shared-expert columns to the waiting ranks per step would need the full shared expert on every rank (bf16 70.8 MB a layer = 2.8 GB a rank) for at most the shared expert's ~20 us a layer |
| floor: every rank at the mean (perfect balance of the same distinct experts) | -2.58 | -4.61 | -- | -- | -- | unreachable: per-step noise |

## 3. What was built: R9K_EXL3_EP_MAP (load-time placement)

* `r9700_vllm/moe/exl3_placement.py` (new) wraps `Exl3MoEMethod.create_weights`. Before the layer's weights are created it
  replaces the layer's stock expert map (`RoutedExperts._expert_map`, plus `expert_map_manager._expert_map` for consistency)
  with the map's global -> local table for this EP rank. quant/exl3_moe.py then needs no change: its weight loader already
  routes every checkpoint expert through the layer's expert map, and process_weights_after_loading derives `exl3_g2l` from
  the same map. So the r9k tier, the mgemv / coop / fused tiers and the prefill path all follow automatically. Pack tensors
  are untouched (a load-time transform), and so is VRAM (48 experts a rank either way). A map that does not partition G into
  equal shares for this EP size is refused for that layer (it stays linear). Layers absent from the file stay linear. The
  drafter's layers are keyed 40-42 (DSpark names them `layers.{num_hidden_layers + i}`, G = 128, 16 a rank).
* Hook: `r9700_vllm/__init__.py`, three lines inside the exl3 registration block (`exl3_placement.install()`, a no-op when the
  knob is unset; the registered line gains `exl3:ep_map`). No edit to exl3_moe.py, exl3_decode.py or the kernel. A2's tier
  rewrite and E's dq-path edits are unaffected (they read `exl3_g2l` / the stash as before).
* The map: `bench/ds41/epskew/map-ds41-v041-prose-code.json`, from `skew_place.py` on M1's prose + code one-stream rings.
  Format `{"version": 1, "ranks": 8, "layers": {"<layer>": {"G": 384, "owner": [rank of expert 0, ...]}}}`. It is built by
  frequency-greedy assignment followed by 20,000 swap trials a layer that minimize the sum over training steps of the max over
  ranks (1b cost).
* Correctness: the same experts compute the same slots with the same weights; only which rank holds them changes. Checked in
  the server through the extension's `r9k_ep_maps` RPC: all 344 (rank, layer) local expert lists equal the map's. **Not
  bit-exact against linear placement, by construction**: each rank's bf16 partial sum before the all-reduce groups the picks
  differently, so the last bit can differ, as with any change of EP layout. Greedy decodes then take a different path
  sometimes (see the MAL and divergence rows). The acceptance gates are the server metrics below.

### A/B (same tree t2, same session; A = no knob, B = `R9K_EXL3_EP_MAP=map-ds41-v041-prose-code.json`)

Step wall (workload.py, greedy 3000-word essays, mid-decode 6 s timing windows, three windows an arm over two launch pairs):

| | A: step ms (MAL) | B: step ms (MAL) | delta |
|---|---|---|---:|
| 1 stream | 19.23 (1.97), 18.99 (1.79), 19.11 (1.83): **19.11** | 18.41 (1.84), 18.58 (1.94), 18.58 (1.89): **18.52** | **-0.59 ms (-3.1 %)** |
| 4 streams | 32.43 (1.89), 32.43 (1.90), 32.43 (1.89): **32.43** | 31.58 (1.93), 31.41 (1.93), 31.25 (1.86): **31.41** | **-1.02 ms (-3.1 %)** |

Per-rank traces of the same launches (AB2, 3 s profile windows, `skew_traces.py`):

| | 1 stream A -> B | 4 streams A -> B |
|---|---|---|
| post-MoE all-reduce rank 0 (min over ranks) | 85.8 (24.0) -> **69.9** (24.1) us | 154.7 (50.8) -> **122.7** (52.4) us |
| tier skew a step (persistent part) | 2.37 (1.31) -> **1.86** (0.64) ms | 4.18 (2.99) -> **2.86** (1.13) ms |
| tier max over ranks, summed a step | 4.88 -> 4.39 ms | 10.99 -> 9.94 ms |
| drafter: post-MoE all-reduce rank 0; rank 6's tier | 104.8 -> 59.0 us; 82.7 -> 49.7 us | 205.7 -> 118.9 us; 191.8 -> 116.0 us |

Server acceptance (AB pair; spectune reps 1, natural stop, 700 tokens; release bands MAL 1.7-2.0 prose / 4.1-4.5 code):

| | A | B |
|---|---|---|
| spectune prose x1: tok/s a stream (MAL) | 94.4 (1.78) | 100.5 (1.87) |
| prose x4: aggregate tok/s (MAL) | 214.8 (1.84) | 222.0 (1.82) |
| code x1 | 196.7 (4.43) | 191.5 (4.14) |
| code x4: aggregate | 372.5 (4.26) | **391.9** (4.26) |
| divergence.py top-1 (lighthouses / fibonacci / 17 x 23) | 21 / 24 / 24 of 24 | 24 / 23 / 23 of 24 |
| GSM8K 100 (thinking, 8 concurrent) | 98 / 100 | 98 / 100 |

MAL stays in its bands. The single 700-token code x1 run differs by its greedy path (MAL 4.43 vs 4.14), not by step time.

Reading: the wall gain at four streams (-1.02 ms) is below the model's -1.6 because the trace skew fell 1.31 ms, not 1.6.
The held-out essays at 9-24 rows share less with the training histogram than the one-stream windows do. The gain is
workload-dependent by nature: on traffic unlike the training mix (pure code served with a prose map) it falls to ~0, and it
never costs anything beyond that, because a placement cannot add tier work (the sum over ranks of distinct experts is
placement-invariant).

## 4. Recommendations for the owner

1. **Turn the knob on by default for the DeepSeek launcher** (serve/ds41-exl3.sh is not H3's file): mount the map and pass
   `-e R9K_EXL3_EP_MAP=...`. That is -0.6 ms at one stream and -1.0 ms at four, for zero memory and zero kernels. Regenerate
   the map from a histogram of the traffic actually served (`run_skew.sh TAG hist:...` + `skew_place.py`; the M1 recipe is
   ~15 minutes of card time). A mix weighted toward the served domains is the right training set. The prose + code map gives
   most of each domain's in-domain gain (-0.83 prose, -1.00 code at one stream).
2. Replicas (b) are worth revisiting only if A2's tier prologue can choose the copy for free and the KV budget can give
   0.5-1 GiB a rank. Even then it is +0.25-0.45 ms at 8 copies a layer.
3. EPLB (c): no. It is unsupported for Exl3MoEMethod, its replica choice is load-blind (simulated worse than the static map),
   and it needs B's fused top-k front to apply the logical -> physical map.
4. What is left after the map (~1.9 ms at one stream, ~2.9 at four) is per-step noise of the routing itself, not a placement
   problem. It shrinks only with fewer bytes a distinct expert (A2's / F's bandwidth) or with dynamic per-step balancing
   (replicas).

## Files

New: `notes/exl3-ep-skew.md`; `r9700_vllm/moe/exl3_placement.py`; `bench/ds41/epskew/` (`epskew_ext.py` the routing recorder,
`run_skew.py` its client, `run_skew.sh` the VM100 launch runner, `skew_traces.py` the per-rank trace skew,
`skew_model.py` the cost fit + histogram, `skew_noise.py` the uniform-routing reference, `skew_place.py` the policy
simulator / map writer, `map-ds41-v041-prose-code.json` the map). Edited: `r9700_vllm/__init__.py` (3-line hook).
