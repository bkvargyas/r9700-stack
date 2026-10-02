# Changelog

All notable changes to r9700-stack. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project uses [semantic versioning](https://semver.org/). Every number below was measured on 4× (or 2×)
Radeon AI PRO R9700 at a 225 W cap; [PROGRESS.md](PROGRESS.md) has the method behind each one.

## [Unreleased]

## [0.2.2] - 2026-10-02

A fix release. Three bugs, all present since 0.1.0, all invisible to fixed-depth benchmarks and to evals at
concurrency 1; each was found by loading the server the way real traffic does. **Upgrade.**
On two cards (Flash-Next with offloaded experts, and the 27B): wrong output for some requests under concurrency, and
(Flash-Next) a memory leak that ends in out-of-memory. On any Flash-Next configuration, four cards first: out of
memory when one step holds prompts of very different lengths. Speed and (correct) outputs are unchanged.

### Fixed
- **Garbled answers on two cards when a request joins a batch in flight.** A race in our 2-rank P2P all-reduce
  (`kernels/r9k_ar.hip`): the half of its double buffer was chosen by a per-block counter while the number of blocks
  followed the message size, so after a small message the next larger one could overwrite the previous message's
  last rows before the other rank had summed them. The last sequence of the batch then decoded garbage (the right
  beginning, then one token repeated). With more requests than `max_num_seqs`: 34-41% of rounds had a bad answer on
  Flash-Next TP2 (5-8% of answers), 8 of 900 answers on the 27B; after the fix 0 of 10,440. Every call now advances
  every block counter (fixed launch grid), and the compressed path shares the exact path's counters. The 4-rank
  kernels had the same construction and are fixed the same way. `tests/test_ar_race.py` reproduces it with the old
  grid and passes with the new one. On 0.1.0-0.2.1, `R9K_R4D_AR=0` (RCCL all-reduce) avoids it.
- **Out of memory in the prefill of Flash-Next's short convolution when a step holds prompts of very different
  lengths.** The stock method (and our drop-in for it) packs all of a step's prefills into `[prefills, longest,
  4*hidden]` and holds about six such tensors: 3,000 + 15 x 70 tokens peak at 4.8 GiB. On four cards a mixed-length
  soak at 16 running sequences went from 29.4 to 32.2 GiB in 40 seconds and died. Now packed by length (groups of
  at most 1.25x the step's tokens), the same bits, 0.4-0.5 GiB. `tests/test_ple_conv.py`.
- **Memory leak in the expert cache: Flash-Next with offloaded experts ran out of VRAM under mixed-length prompts.**
  The fused LRU path kept one set of align buffers per batch shape per layer for good (28 MiB per rank at a
  4,096-token chunk), and under real traffic nearly every prefill step has a new shape. At the 0.2.1 defaults a
  15-minute soak of 18k-30k-token prompts from 8 clients took a card from 30.5 to 32.6 GiB and killed the engine
  (77 requests served, 369 failed); fixed, the same soak serves 111 with none failed and VRAM flat at 31.0 GiB,
  and a 31-minute soak from 16 clients (270 to 28.5k tokens) serves 291 with none failed. Now one buffer set per
  layer, reused by every step. `tests/test_cache_shapes.py`.
- The 0.2.1 notes give Flash-Next TP2-offload prefill as 2,208-3,839 tok/s, 5-14% above 0.2.0. Both are BetterBench
  figures on its shuffled-paragraph filler, and the gain exists only for such narrow prompts (the cache can now
  follow their experts). Real text prefills at about 2,300-2,600 tok/s on both versions. README and PROGRESS say so.

### Added
- `bench/sanity_stress.py`: the 8-way sanity check, strict (the answer and nothing else), in rounds, at any
  concurrency and optionally after long prefills. `bench/soak.py`: mixed-length long-context soak.
  `bench/prefill_kinds.py`: prefill by kind of prompt.
- Release checklist (notes/picking-up.md): stress at more requests than `max_num_seqs`, soak with mixed lengths,
  on every configuration, before a tag.
- A same-day baseline of the 27B against the reference stack on the same two cards (PROGRESS.md): decode 196.6 vs
  197.5 tok/s, prefill 81-87%, TTFT 116 vs 65 ms, concurrency 94-97%.

### Rejected, with numbers
- `UTIL=0.96` for TP2 with offload: ran at 69 MiB free and died when the queue drained (PROGRESS.md 2026-10-01).

### Validation
- On the release code, every configuration: 26 unit gates, the all-reduce suites, full BetterBench (speed unchanged:
  97.4 / 157.3 / 197.5 tok/s for Flash-Next TP2 + offload / Flash-Next TP4 / 27B TP2), strict stress over
  `max_num_seqs` (0 bad of 9,500), mixed-length soaks (1,077 requests served, none failed; VRAM peaks 30.9 / 30.2 /
  32.5 GiB of 32.6).

## [0.2.1] - 2026-09-30

Flash-Next on **two** cards with the experts in host RAM: new defaults, measured end to end, and the tools that
found them. Full BetterBench against the 0.2.0 defaults: single-stream decode 97.4 tok/s (93.8), TTFT 484 ms (555),
prefill 2,208 / 3,558 / 3,839 / 3,793 (2,099 / 3,163 / 3,434 / 3,329), concurrency 87 / 109 / 116 / 118 / 113
(84 / 104 / 108 / 111 / 95), time to first token at 8 concurrent 0.95 s (7.1 s). Four-card numbers and all
numerics are unchanged. And a correction: 0.2.0's decode fusions cost no accuracy.

### Changed
- `serve/flashnext.sh` defaults `NSEQ=8` (the launcher's default was 4; benchmarks passed 16). Each request holds
  18 KV blocks whatever its length, and at TP2 with offloaded experts the graphs and activations for 16 sequences
  took the room of the requests they were for: four to six ran, the rest queued. Pass `NSEQ=16` for TP4.
- `R9K_LRU_THRESH` defaults to `0.99` (was `0.5`). Eight different requests route to ~150 distinct experts a
  layer; over 0.5 x 270 slots the cache manager inserted nothing, on 72% of steps, and every step read 31 experts
  a layer from host (2 GB, 207 ms per forward pass). BetterBench conc-8 with all eight running: 77 -> 118 tok/s.
- `R9K_LRU_GATHER` defaults to `64,16` (was `8,16`): the expert insert copy is ~15% faster at 8+ inserts and equal
  below. Outputs cannot change.
- Quality evals for a change of default use the full 1,319-question test set, not the first 800.

### Added
- `R9K_EXPERT_CACHE_STATS=1`: opt-in counters in the expert cache (distinct routed experts, inserts, experts read
  through from host, steps over the insert threshold), logged every `R9K_EXPERT_CACHE_STATS_SEC` seconds per rank;
  `tests/test_cache_stats.py`. `bench/mix.py` (conc-8 by workload mix) and `tuning/lru_gather_bench.py` (host -> VRAM
  copy rate of the insert kernel).
- What bounds Flash-Next at TP2 with offloaded experts, measured (PROGRESS.md 2026-09-29): the PCIe link. A miss is
  1.245 MiB per rank and the copy runs at the link rate (11.4-13.5 GB/s on PCIe 3); one prompt type misses 4.5%
  of its routed experts per step, four types 14%, which is what the checkpoint's routing profile predicts for 270
  slots (13.3%). The cards draw 206 W with one request and 170 W with eight.
- The card-placement rule, measured: tensor parallel wants both cards on one PLX switch, offloaded experts want
  one card per switch (same-switch pair: 81 / 54 / 1,379 single / conc-8 / prefill-8k vs 114 / 132 / 2,464).
- Operational note in the launcher and README: the first launch of a configuration gets a smaller KV pool than
  every later one (94k against 121k tokens at `NSEQ=8`); restart once.

### Rejected, with numbers (PROGRESS.md)
- 240 expert slots (more KV; -28% on a mixed batch, -6% single stream), `NBT=2048` as a default (-24% prefill),
  re-splitting the slots across layers by routing mass (13.3% -> 12.8% misses). `UTIL=0.96` ran clean with 28% more
  KV but has not held a 32k context under load: not adopted.

### Fixed
- **The quality statement of 0.2.0 was wrong in the project's disfavour.** It said the round-3 decode fusions
  together sit about half a point below the round-2 numerics. On the full 1,319-question test set they do not: the
  default scores 97.04% against 96.82% (5 / 8 discordant, p = 0.58), each of the five fusions alone is equally
  indistinguishable (6 / 6, 0 / 2, 5 / 8, 5 / 7, 6 / 5), and the reference reproduces itself to 1,318 of 1,319
  outputs. The half point came from evaluating on the first 800 questions, where the round-2 numerics score 98.0%
  (95.0% on the other 519).
- PROGRESS.md reported a concurrency regression for Flash-Next TP2 with offloaded experts (conc-8 132 vs 206 on
  2026-09-20). There is none: the two figures came from different harnesses (eight copies of one prompt against
  four different prompts).

## [0.2.0] - 2026-09-29

Qwen3.8-Flash-Next on four R9700s (TP4) goes ahead of the fastest known alternative stack on every metric:
single-stream decode, time to first token, prompt processing at every depth, and aggregate throughput at every
concurrency level -- measured on a PCIe 3 host; PCIe 5 would be faster still. Full BetterBench, TP4, 225 W:
decode 159.2 tok/s vs 134.1 (+19%), step p50 16.8 vs 20.4 ms, TTFT 94 vs 145 ms, prefill 6,408 / 7,365 /
7,451 / 7,182 vs 5,279 / 5,711 / 5,977 / 6,106 tok/s at 2k / 8k / 16k / 32k, concurrency 151 / 231 / 350 / 478 /
635 vs 126 / 197 / 303 / 427 / 542 at 1 / 2 / 4 / 8 / 16. v0.1.0 had Flash-Next at ~85 tok/s on two cards.

### Added
- **Four-card support.** `GPUS` / `TP` / `PORT` / `NAME` launcher knobs, TP=4 MoE and dense configs, and the
  host work that makes four cards behind two PLX switches usable: 32 GB BARs for both cards on each switch
  (`host/`, DKMS `chainfix` 1.2), switch-local P2P on both PEX 8747s via a guest placement module (2× peer
  bandwidth, worth 9-10% of prefill).
- **N-rank P2P all-reduce** (one-shot and two-shot) for TP>2, and a **compressed hierarchical 4-rank
  all-reduce** for prefill-sized messages (on by default).
- **Sparse attention (QSA) for Qwen4Exp**: our own bitmap / attention / merge kernels, and a **WMMA indexer
  scorer** (`r9k_qsa_score`) that scores only the visible columns instead of all 8,192 (prefill +10-12%).
- **Decode fusions**, each replacing a run of tiny per-step launches with one kernel (every HIP-graph node costs
  ~1.5 µs of dispatch on this ROCm): a bf16 skinny **router GEMM** for the MoE gate; fused **Gemma norm + rope**
  for the sparse-attention indexer (1-D or MRoPE positions, any stride); the **hyper-connection** gate mix,
  combine-norm and decode input mix; the **shared expert in four launches** with its sigmoid gate folded into
  the down GEMM epilogue; and the **Gated DeltaNet speculative-decode core** (`r9k_gdn_decode_mtp`: gating,
  delta-rule recurrence over the draft tokens and the gated RMS norm in one launch per sequence and head,
  replacing ~11 graph nodes per layer).
- **Prefill kernels**: per-row output scales from LDS and LDS-staged 16 B row stores in both MoE prefill
  epilogues (TP=4 down GEMM 1,191 → 690 µs); tiles that accept K % BK for the TP=4 down GEMM; the PLE prefill
  short conv on a tiled kernel (26.5 → 4.1 ms per 4k chunk).
- **Hybrid block-fp8 dispatch** and MXFP4 LM heads for decode; TP=4 shapes in the tuning table.
- **Prefill chunks up to 2,048 tokens captured in HIP graphs** (`CGSIZES`), removing ~285 ms of CPU-bound
  eager forward from short prompts (TTFT 137 → 88 ms).
- `R9K_*=stock` knobs for every fusion, `R9K_GDN_DEBUG` / `R9K_GDN_FUSED` diagnosis knobs, `tools/`
  benchmark harnesses (`bench/tp4vs2.sh`, probes, paired eval).
- This changelog; `notes/independence.md` now lists every runtime hook beyond vLLM's extension points.

### Changed
- **Base image: vLLM 0.30 nightly (rocm100, torch 2.12).** The whole stack runs unchanged on it.
- **Prefix caching off by default for Flash-Next** (`PREFIX_CACHE=1` restores it): the short-prefill root cause.
- MTP-3 stays the default after measuring MTP-4 (+4.4% single-stream, -14% at 16 concurrent; `MTP=4` is a knob).
- Both routed MoE GEMMs default to the 64×128 BK=32 prefill tile at TP=2 and TP=4.
- Compressed all-reduce on by default; `ar4` at 128 blocks.

### Fixed
- The fused indexer rope silently fell back to stock in serving: positions arrive as a stride-3 column view of a
  `[T, 3]` buffer, which the kernel now accepts.
- The Gated DeltaNet core must be a piecewise-graph splitting op like the vLLM op it replaces; without that a
  prefill piece captured it with stale kernels and the first request faulted. Registered at plugin load.
- Flash-Next's output gate is `sigmoid`, not silu; the GDN core supports both and its fit check logs its reason.
- Custom-op signatures carry type annotations (schema inference refused the untyped tensors at engine start).
- A staged-store epilogue overwrote its own row tables at cfg 17 (memory fault); the tables now sit past the stage.

### Quality
- Every change to a shipped default is checked by an 800-question chain-of-thought paired eval at concurrency 1
  (paired McNemar), plus the 8-prompt sanity set. The v0.2.0 default scores 97.50% on that eval (no detectable
  difference from the previous default alone, p = 0.22); across runs the round-3 decode fusions together sit
  about half a point below the round-2 build's 97.9-98.0%. `R9K_ROUTER=stock R9K_QSA_GLUE=stock R9K_HC_MIX=stock`
  restores round-2 numerics at ~10% of decode. Details in PROGRESS.md.

## [0.1.0] - 2026-09-22

First release: tuned gfx1201 kernels and a vLLM plugin running Qwen3.8-27B-NVFP4 and Qwen3.8-Flash-Next on 2×
R9700 with stock vLLM and stock ROCm 10.

### Added
- MXFP4 × FP8 MoE GEMM in three shapes (decode with in-block split-K, LDS-tiled prefill with a 17-entry tile
  table, fragment-tiled-activation prefill).
- Paged attention for head_dim 256 (GQA 1-8, block 16, bf16 and fp8 KV).
- 2-rank P2P all-reduce with a compressed path (Walsh-Hadamard rotation + 4-bit groups) for prefill messages.
- Expert LRU cache streaming from pinned host memory, int6 PLE gather, dense fp8 GEMM for LM heads, load-time
  NVFP4 → MXFP4 conversion.
- Registration entirely through vLLM's extension points; Apache-2.0 with one carve-out (see `NOTICE`).

[Unreleased]: https://github.com/bkvargyas/r9700-stack/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/bkvargyas/r9700-stack/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/bkvargyas/r9700-stack/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/bkvargyas/r9700-stack/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/bkvargyas/r9700-stack/releases/tag/v0.1.0
