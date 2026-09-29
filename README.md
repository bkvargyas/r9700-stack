# r9700-stack

Tuned GPU kernels and a vLLM plugin that make **Qwen3.8** run fast on **AMD Radeon AI PRO R9700** cards
(gfx1201 / RDNA4).

**Headline (September 2026): Qwen3.8-Flash-Next on four R9700s is now faster than the best known alternative
stack on every metric we measure** -- 159 tok/s single-stream decode against 134 (+19%), first token in 94 ms
against 145, prompt processing 18-29% ahead at every depth from 2k to 32k tokens, and 12-20% more aggregate
throughput at every concurrency level. All of it on stock vLLM and stock ROCm, from a plugin -- and on a **PCIe 3** host, where
the cards talk to each other at ~13.7 GB/s; the same code on a PCIe 5 box would move the multi-card numbers up
again. The numbers are in [Benchmarks](#benchmarks); the story of how each one moved is in
[PROGRESS.md](PROGRESS.md).

**Built for stock upstream releases.** It targets **released vLLM** and **ROCm 10 or newer**, unmodified — no
fork, no patched source, no vendored binaries. Everything loads as a plugin at runtime through vLLM's own
extension points (quantization config, model registry, platform plugin, attention backend, custom ops), so you
can update vLLM or ROCm without re-porting anything. That constraint was the point of the project: hand-tuned
kernels normally mean a fork you then maintain forever.

Requirements: 2× or 4× Radeon AI PRO R9700 (gfx1201), ROCm ≥ 10, a released vLLM build, and the model weights.

### Checkpoint formats

Built primarily for **NVFP4** checkpoints — e2m1 codes with an e4m3 scale per 16 elements and an fp32 per-row
global, as published by `unsloth/Qwen3.8-27B-NVFP4` and similar. Two paths are supported and both are tested:

| | what happens | when to use it |
|---|---|---|
| `R9K_NVFP4=mxfp4` *(default in `serve/27b.sh`)* | NVFP4 is converted to MXFP4 once at load | **+6% everywhere.** The MXFP4 kernels are the faster ones, and the conversion is measured to cost no detectable quality |
| `R9K_NVFP4=native` | the checkpoint's own NVFP4 bits are kept, and a native NVFP4 kernel variant runs on them | when you want the checkpoint's numerics preserved exactly |

**MXFP4** checkpoints work directly — compressed-tensors `mxfp4-pack-quantized`, as used by the Flash-Next GPTQ
build. Any **fp8** layers inside a checkpoint can also be converted to MXFP4 at load (`R9K_FP8_TO_MXFP4=1`,
measured worth ~21% of prefill on the 27B).

So: bring an NVFP4 image and it will run; bring an MXFP4 one and it will run; mixed fp8/MXFP4 checkpoints are
handled by converting the fp8 parts.

## Benchmarks

### Qwen3.8-Flash-Next, 4× R9700 (TP4)

Flash-Next (the MoE + Gated DeltaNet model, MXFP4/fp8 GPTQ checkpoint), tensor-parallel over four cards at a
225 W cap, full BetterBench (20 passes), MTP-3 speculative decoding on both stacks. The reference column is the
fastest known alternative stack for this model on **the same box, the same checkpoint and the same power cap**.

| | this stack (v0.2.0) | reference stack | |
|---|--:|--:|--:|
| single-stream decode | **159.2 tok/s** | 134.1 | **+19%** |
| decode step p50 | **16.8 ms** | 20.4 ms | **-18%** |
| time to first token p50 | **94 ms** | 145 ms | **1.5× faster** |
| prefill 2k / 8k / 16k / 32k (tok/s) | **6,408 / 7,365 / 7,451 / 7,182** | 5,279 / 5,711 / 5,977 / 6,106 | **+21% / +29% / +25% / +18%** |
| concurrency 1 / 2 / 4 / 8 / 16 (aggregate tok/s) | **151 / 231 / 350 / 478 / 635** | 126 / 197 / 303 / 427 / 542 | **+20% / +17% / +16% / +12% / +17%** |

Where it came from, in one line each: prefill from an MXFP4×FP8 MoE GEMM with the per-row scales in LDS, a
WMMA scorer for the sparse-attention indexer that only touches the visible columns, and a compressed 4-rank
all-reduce; decode from replacing hundreds of tiny per-step launches (norm + rope glue, router GEMM,
hyper-connection mix, the shared expert, the Gated DeltaNet speculative-decode core) with one kernel each --
on this ROCm every HIP-graph node costs about 1.5 µs of dispatch, so the launches were the cost.

The last two fusions (the Gated DeltaNet speculative-decode core and the four-launch shared expert) alone took
single-stream decode from 139 to 159 tok/s and the step from 19.0 to 16.8 ms, in one day.

Quality: 800 chain-of-thought questions at concurrency 1, paired against the previous default before any change
to the default ships (paired McNemar). v0.2.0 scores 97.5%, no detectable difference from the previous default;
the decode fusions can be switched off individually (`R9K_*=stock`) if you want the earlier numerics back.

### Qwen3.8-27B-NVFP4, 2× R9700 (TP2)

Measured with
BetterBench 0.6.0 (29 prompts across 8 categories, 20 passes each) on the shipped default
configuration — our own attention and all-reduce kernels, nothing third-party loaded.

The right-hand column is the fastest known alternative stack for this model, measured on **the same box, the
same checkpoint and the same power cap**, so the comparison is like-for-like.

**Decode**

| | this stack | reference stack | |
|---|--:|--:|--:|
| combined (weighted across categories) | **184.8 tok/s** | 196.5 | 94% |
| update p99 | 26.2 ms | 24.2 ms | |
| TTFT p50 | 103 ms | 65 ms | |

**Concurrency** (aggregate tok/s)

| level | this stack | reference stack | |
|--:|--:|--:|--:|
| 1 | **166.1** | 176.8 | 94% |
| 2 | **269.5** | 294.4 | 92% |
| 4 | **393.7** | 427.9 | 92% |
| 8 | **520.9** | 549.1 | 95% |

**Prompt processing** (prefill, tok/s median, cold prefix cache)

| depth | this stack | reference stack | |
|--:|--:|--:|--:|
| 2k | **4,099** | 4,776 | 86% |
| 8k | **4,143** | 4,950 | 84% |
| 16k | **4,043** | 4,906 | 82% |
| 32k | **3,809** | 4,745 | 80% |

Quality: GSM8K, full 1,319-question test set, greedy, concurrency 1 — **94.4–95.5%** depending on configuration,
with no statistically detectable difference between them (paired McNemar).

**Prefill is where the gap lives, and it is a deliberate trade.** The default configuration uses our own
all-reduce so that nothing unlicensed is loaded at runtime; that costs about 6% of prefill against the
third-party one. Setting `R9K_AR_IMPL=r4d R9K_PAGED_ATTN=r4d` recovers it (~4,400 tok/s at 8k) if you have that
library and would rather have the speed. Beyond that, large-message all-reduce on this host is bandwidth-bound
on a PCIe 3 link at ~13.7 GB/s, which is the practical ceiling.

[PROGRESS.md](PROGRESS.md) has every number, how it was produced, and what was tried and rejected.

### About the 225 W power cap

**Every number in this repository was measured with the cards capped at 225 W, and that is deliberate.**

The R9700 will draw more if you let it, and prefill in particular is clock-limited — the card sustains about
2.40 GHz during prefill against 2.82 GHz on decode, so raising the cap would improve prefill by some margin we
have never measured. We have not measured it on purpose, for two reasons:

1. **Comparability.** Every reference measurement we hold for other stacks was taken at 225 W on this same box.
   Raising ours would make the comparison apples-to-oranges — the exact mistake that produced one bad baseline
   earlier in this project.
2. **It matches production.** These cards run capped here permanently. A number measured uncapped is a number
   nobody would ever see in service, and optimising against it would mean tuning for the wrong operating point.

So treat 225 W as a fixed condition of the benchmarks rather than a tuning knob. If you run uncapped, your
numbers should be better than these, and they will not be comparable to them.

## Quick start

You need the models on disk and a ROCm 10 container. Then:

```bash
serve/27b.sh                      # Qwen3.8-27B-NVFP4 on 2 GPUs
serve/flashnext.sh                # Qwen3.8-Flash-Next on 2 GPUs
GPUS=0,1,2,3 TP=4 serve/flashnext.sh   # Flash-Next on 4 GPUs (the headline configuration)
```

Both are thin wrappers over `serve/serve.sh` and every tuning knob in them is commented with what it was measured
to be worth. `DRYRUN=1` prints the docker command without starting anything.

An OpenAI-compatible endpoint comes up on `:8080`.

## What's in it

- **`kernels/`** — HIP kernels for gfx1201: MXFP4×FP8 MoE GEMM (decode, prefill and fragment-tiled prefill
  variants), paged attention, sparse-attention (QSA) scoring and attention, 2-rank and compressed 4-rank
  peer-to-peer all-reduce, fp8 GEMM, int6 embedding gather, and the decode fusions: Gated DeltaNet
  speculative-decode core, router GEMM, indexer norm+rope, hyper-connection mix, shared expert.
- **`r9700_vllm/`** — the plugin. Registers through vLLM's official extension points (quantization config,
  model registry, platform plugin, attention backend, custom ops, pluggable layers); the few runtime hooks
  beyond those are version-gated and listed in [notes/independence.md](notes/independence.md).
- **`serve/`** — launchers, with measured knobs.
- **`tests/`** — correctness gates. Each kernel is checked against a reference implementation, and several are
  checked to be *bit-identical* to the path they replace.
- **`tuning/`** — the benchmark harnesses used to pick tile configurations.
- **`host/`** — Proxmox host setup: 32 GB BARs for cards behind the PLX switches, including a second card on the
  same switch (DKMS kernel module + barfix script + VM hookscript).
- **`notes/`** — design notes and investigation write-ups.

## Docs

| file | what it's for |
|---|---|
| [PROGRESS.md](PROGRESS.md) | The full engineering log: every change, what it measured, and what was tried and rejected. |
| [notes/picking-up.md](notes/picking-up.md) | **Start here if you're returning to this after a break.** Current state, open threads, how to run things. |
| [host/README.md](host/README.md) | Host PCIe setup: why a second card on one PLX switch gets no BAR, and the fix. |
| [notes/independence.md](notes/independence.md) | Which third-party pieces were replaced with our own, and why. |
| [notes/replacement-plan.md](notes/replacement-plan.md) | The plan for replacing what is left, so the repo is cleanly Apache-2.0. |
| [CREDITS.md](CREDITS.md) | The two third-party components, and the licence position. |

## Licence

[Apache-2.0](LICENSE), with one carve-out listed in [NOTICE](NOTICE) — read that file before reusing anything.

The short version: the vendored LRU cache kernels are Apache-2.0 too and pass on normally. Nothing loads libr4d
at runtime any more, and the GEMM constants that were once listed as derived from it turn out to be generated
from the OCP format specs — `tools/gen_kmag.py --check` proves it. The one remaining carve-out is the chat
template, copied from [vllm-mxfp4](https://github.com/GGZ14/vllm-mxfp4) with its author's permission. Both authors gave permission for **this** project; neither
upstream has a licence file, so that permission is not ours to pass on. Those parts are not under Apache-2.0 —
if you want to reuse them, ask their authors.
