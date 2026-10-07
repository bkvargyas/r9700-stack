# r9700-stack

Tuned GPU kernels and a vLLM plugin that make **Qwen3.8** run fast on **AMD Radeon AI PRO R9700** cards
(gfx1201 / RDNA4) -- on **stock vLLM and stock ROCm**. No fork, no patched source, no vendored binaries:
everything loads at runtime through vLLM's own extension points, so vLLM and ROCm update underneath it.

## Highlights

- **Qwen3.8-Flash-Next on four R9700s beats the best known alternative stack on every metric we measure**:
  single-stream decode +19%, first token 1.5x faster, prompt processing +18-29% at every depth, aggregate
  throughput +12-20% at every concurrency. On a PCIe 3 host where the cards talk at ~13.7 GB/s.
- **Qwen3.8-27B-NVFP4 on two cards** decodes at parity with that stack (197 tok/s single-stream) with nothing
  third-party loaded at runtime; the two-card Flash-Next runs with its experts streamed from host RAM.
- **KV cache**: one state page per request for the Gated DeltaNet layers (+64% on the 27B), and on four cards
  the profiling and cudagraph accounting fixed: Flash-Next TP4 went from 279k to 475k tokens this week.
- **Quality is gated, not assumed**: every change to a default ships only after the full 1,319-question GSM8K
  set, paired per question against the previous numerics (McNemar), plus HumanEval, strict sanity under
  overload and a mixed-length soak. No default has moved quality by a detectable amount.
- Built for **NVFP4** and **MXFP4** checkpoints; fp8 layers inside a checkpoint are converted at load.

## Stats

Flash-Next, 4x R9700, TP4, MTP-3 speculative decoding. The reference is the fastest known alternative stack on
the same box, checkpoint and power cap (full 20-pass BetterBench, v0.2.0 at 225 W; current code, 210 W, in the
right-hand column from the 2026-10-07 runs):

| | this stack vs reference (BetterBench, v0.2.0, 225 W) | now (2026-10-07, 210 W, probe) |
|---|--:|--:|
| single-stream decode | **159 tok/s** vs 134 (+19%) | 199 tok/s |
| decode step p50 | **16.8 ms** vs 20.4 | 16.4 ms |
| time to first token p50 | **94 ms** vs 145 | |
| prefill 2k / 8k / 16k / 32k tok/s | **6,408 / 7,365 / 7,451 / 7,182** vs 5,279 / 5,711 / 5,977 / 6,106 | 8k: 6,850-7,070 |
| aggregate tok/s at 1 / 2 / 4 / 8 / 16 | **151 / 231 / 350 / 478 / 635** vs 126 / 197 / 303 / 427 / 542 | 8 streams: 560-610, 16: 890-905 |
| KV cache | | **475k tokens** (was 279k) |
| GSM8K (1,319 questions, paired vs the previous numerics) | 97.0% vs 96.8% with the fusions off, p = 0.58 | 95.5% vs 95.6%, p = 0.86 |
| HumanEval | | 160 / 164 |

Two cards: 27B-NVFP4 197.5 tok/s single-stream, 23.5 ms step, 174 / 280 / 413 / 519 tok/s at 1 / 2 / 4 / 8,
first token 47 ms; Flash-Next with experts in host RAM 97 tok/s single-stream, link-bound near 115 tok/s
aggregate. Every number, how it was produced, and what was tried and rejected: [PROGRESS.md](PROGRESS.md),
[CHANGELOG.md](CHANGELOG.md), `notes/`.

## Running it

Requirements: 2x or 4x Radeon AI PRO R9700, ROCm >= 10, a released vLLM build, the model weights.

```
bash serve/27b.sh                                   # Qwen3.8-27B-NVFP4, two cards
GPUS=0,1,2,3 TP=4 OFFLOAD_GB=0 NSEQ=16 bash serve/flashnext.sh    # Flash-Next, four cards
bash serve/flashnext.sh                             # Flash-Next, two cards, experts in host RAM
```

The serve scripts carry the measured defaults and document each knob next to it; `R9K_*=stock` switches any
kernel back to vLLM's. Tests are in `tests/`, the benchmark and gate tools in `bench/`, the release checklist
in [notes/picking-up.md](notes/picking-up.md).

**Current release: [v0.2.5](notes/release-v0.2.5.md).** The 2026-10-07 defaults above are on `master` and
validated as described in the changelog; the next tag carries them.

## Notes for operators

- **Power**: measured at a 225 W cap through v0.2.3 and at 210 W with a -42 mV offset since; the cap is a
  thermal and acoustic choice, not a limit of the code.
- **MES timeouts under KVM passthrough** (`MES(1) failed to respond to msg=INVALIDATE_TLBS`): add
  `amdgpu.mes_log_enable=1 amdgpu.gpu_recovery=0` to the guest kernel. [notes/mes-timeouts.md](notes/mes-timeouts.md),
  reported as [drm/amd #5759](https://gitlab.freedesktop.org/drm/amd/-/issues/5759).
- **Card placement**: tensor-parallel traffic wants cards on the same PCIe switch; offloaded experts want one
  card per switch. The serve scripts say which.
