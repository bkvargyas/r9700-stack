# r9700-stack

Tuned GPU kernels and a vLLM plugin that make **Qwen3.8** run fast on **AMD Radeon AI PRO R9700** cards
(gfx1201 / RDNA4) — using **stock vLLM and stock ROCm**, with no forks and no patched source.

Everything here loads as a plugin at runtime. You can update vLLM or ROCm without re-porting anything, which was
the whole point: hand-tuned kernels usually mean a fork you then have to maintain forever.

## What it runs

| model | single stream | 8 concurrent | prefill |
|---|---|---|---|
| Qwen3.8-27B-NVFP4 (TP2) | ~183 tok/s | ~522 tok/s | ~4,400 tok/s @8k |
| Qwen3.8-Flash-Next (TP2) | ~85 tok/s | ~206 tok/s | ~2,150 tok/s |

Measured on 2× R9700 at a 225 W cap. GSM8K-500 ≈ 97%. For reference, the fastest known alternative stack on the
same box and model scores ~196 tok/s single stream — see [PROGRESS.md](PROGRESS.md) for the full comparison and
how each number was produced.

## Quick start

You need the models on disk and a ROCm 10 container. Then:

```bash
serve/27b.sh          # Qwen3.8-27B-NVFP4 on 2 GPUs
serve/flashnext.sh    # Qwen3.8-Flash-Next on 2 GPUs
```

Both are thin wrappers over `serve/serve.sh` and every tuning knob in them is commented with what it was measured
to be worth. `DRYRUN=1` prints the docker command without starting anything.

An OpenAI-compatible endpoint comes up on `:8080`.

## What's in it

- **`kernels/`** — HIP kernels for gfx1201: MXFP4×FP8 MoE GEMM (decode, prefill and fragment-tiled prefill
  variants), paged attention, 2-rank peer-to-peer all-reduce, fp8 GEMM, int6 embedding gather.
- **`r9700_vllm/`** — the plugin. Registers through vLLM's official extension points (quantization config,
  model registry, platform plugin, attention backend, custom ops) — no monkey-patching of vLLM internals.
- **`serve/`** — launchers, with measured knobs.
- **`tests/`** — correctness gates. Each kernel is checked against a reference implementation, and several are
  checked to be *bit-identical* to the path they replace.
- **`tuning/`** — the benchmark harnesses used to pick tile configurations.
- **`notes/`** — design notes and investigation write-ups.

## Docs

| file | what it's for |
|---|---|
| [PROGRESS.md](PROGRESS.md) | The full engineering log: every change, what it measured, and what was tried and rejected. |
| [notes/picking-up.md](notes/picking-up.md) | **Start here if you're returning to this after a break.** Current state, open threads, how to run things. |
| [notes/independence.md](notes/independence.md) | Which third-party pieces were replaced with our own, and why. |
| [CREDITS.md](CREDITS.md) | Other people's work this builds on, and its licence status. |

## Licence

[Apache-2.0](LICENSE), with two carve-outs listed in [NOTICE](NOTICE) — read that file before reusing anything.

The short version: the vendored LRU cache kernels are Apache-2.0 too and pass on normally. But parts of the MoE
GEMM derive from [libr4d](https://codeberg.org/StillDeadcode/libr4d), and the chat template is copied from
[vllm-mxfp4](https://github.com/GGZ14/vllm-mxfp4). Both authors gave permission for **this** project; neither
upstream has a licence file, so that permission is not ours to pass on. Those parts are not under Apache-2.0 —
if you want to reuse them, ask their authors.
