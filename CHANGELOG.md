# Changelog

All notable changes to r9700-stack. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project uses [semantic versioning](https://semver.org/). Every number below was measured on 4× (or 2×)
Radeon AI PRO R9700 at a 225 W cap; [PROGRESS.md](PROGRESS.md) has the method behind each one.

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

[0.2.0]: https://github.com/bkvargyas/r9700-stack/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/bkvargyas/r9700-stack/releases/tag/v0.1.0
