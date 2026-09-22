# r9700-stack v0.1.0

Tuned gfx1201 kernels and a vLLM plugin that run **Qwen3.8-27B-NVFP4** and **Qwen3.8-Flash-Next** on 2× AMD
Radeon AI PRO R9700, using **stock vLLM and stock ROCm 10** — no forks, no patched source, nothing to re-port
when either moves.

## Performance

Qwen3.8-27B-NVFP4, TP2, 2× R9700 at a 225 W cap, BetterBench 0.6.0. The reference column is the fastest known
alternative stack, measured on the same box, checkpoint and cap.

| | this stack | reference | |
|---|--:|--:|--:|
| combined decode | **184.8 tok/s** | 196.5 | 94% |
| concurrency 1 / 2 / 4 / 8 | **166 / 270 / 394 / 521** | 177 / 294 / 428 / 549 | 92-95% |
| prefill 2k / 8k / 16k / 32k | **4,099 / 4,143 / 4,043 / 3,809** | 4,776 / 4,950 / 4,906 / 4,745 | 80-86% |
| GSM8K (1,319, greedy, conc 1) | **94.4-95.5%** | — | |

Prefill carries the gap, and it is a deliberate trade: the default runs our own all-reduce so nothing unlicensed
loads at runtime, which costs ~6% of prefill. `R9K_AR_IMPL=r4d R9K_PAGED_ATTN=r4d` recovers it if you have that
library. The rest is a PCIe 3 bandwidth ceiling on this host.

## What's in it

- **MXFP4 × FP8 MoE GEMM** for gfx1201, in three shapes: a decode kernel with in-block split-K, an LDS-tiled
  prefill kernel with a 17-entry tile table, and a fragment-tiled-activation prefill kernel that loads each A
  fragment straight from global into the WMMA register (worth +11.9% of prefill on its own).
- **Paged attention** — head_dim 256, GQA 1–8, block 16, causal, bf16 and fp8 KV. S is computed transposed so
  each lane owns one query row, which makes the online-softmax max/sum/rescale per-lane scalars.
- **2-rank P2P all-reduce** — one-shot over IPC-shared fine-grained scratch, with a compressed path for
  prefill-sized messages (Walsh-Hadamard rotation + 4-bit group quantisation, 4.25 bits/element).
- **Expert LRU cache** streaming experts from pinned host memory, an int6 PLE gather, a dense fp8 GEMM for LM
  heads, and load-time NVFP4→MXFP4 conversion.
- Registered entirely through vLLM's **official extension points** — quantization config, model registry,
  platform plugin, attention backend, custom ops. No monkey-patching.

## Licence

**Apache-2.0**, with one carve-out: `serve/templates/qwen-fixed-v22.3.jinja` is GGZ14's, used with permission
and credit. `kernels/third_party/davetha/` is Apache-2.0 upstream and passes on normally. See `NOTICE`.

Nothing in the default configuration loads any unlicensed binary — verified by moving `r4d.so` aside and serving
without it. `tools/gen_kmag.py --check` regenerates the folded-unpack constants from the OCP e2m1/e4m3
specifications and verifies them against the checked-in table, so the one part of the GEMM that used to be
described as third-party is demonstrably computed rather than copied.

## Known limits

- **PCIe 3 on the test host.** The compressed all-reduce runs at ~13.7 GB/s, which is the practical ceiling of
  that link; large-message all-reduce is bandwidth-bound and the remaining gap to the fastest known alternative
  stack lives there.
- **No decode-band attention kernel.** Short-query groups inside a mixed prefill+decode batch run on the prefill
  kernel. Correct, and costs nothing measurable end to end, but it is the obvious next kernel.
- **TTFT carries ~80 ms of fixed cost** because prefill runs eager — ~1,950 kernel launches per forward, of which
  roughly a third are ours. This is stock vLLM behaviour; `cudagraph_mode` already defaults to the most
  aggressive useful setting and forcing `FULL` is worse. Investigated and documented rather than patched.
- **The stack is not reproducible at concurrency > 1.** Dynamic batching changes reduction order, so ~68% of
  answers differ between two runs of identical code at concurrency 8. Evaluate at concurrency 1, where it is
  bit-reproducible. `bench/eval.py selftest` measures this directly.

## Read next

`notes/picking-up.md` is the orientation document — hardware, how to run things, the measurement rules that cost
us time to learn, what is left, and a list of things already tried and rejected so they are not retried.
