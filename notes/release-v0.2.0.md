# r9700-stack v0.2.0

Tuned gfx1201 kernels and a vLLM plugin that run **Qwen3.8-Flash-Next** on 4× and **Qwen3.8-27B-NVFP4** on 2×
AMD Radeon AI PRO R9700, using **stock vLLM (0.30 nightly) and stock ROCm 10** -- no forks, no patched source,
nothing to re-port when either moves.

**The headline: on four cards, Flash-Next is now ahead of the fastest known alternative stack on every metric
we measure.** v0.1.0 trailed it by 5-20%. This is on a **PCIe 3** host (cards behind two PLX switches on Gen3
uplinks, ~13.7 GB/s between them); a PCIe 5 box would lift the multi-card numbers further with no code change.

## Performance

Qwen3.8-Flash-Next (MXFP4 / fp8 GPTQ), TP4, 4× R9700 at a 225 W cap, full BetterBench (20 passes), MTP-3
speculative decoding on both stacks. The reference column is the fastest known alternative stack, measured on
the same box, checkpoint and cap.

| | v0.2.0 | reference | |
|---|--:|--:|--:|
| single-stream decode | **159.2 tok/s** | 134.1 | **+19%** |
| decode step p50 | **16.8 ms** | 20.4 ms | **-18%** |
| time to first token p50 | **94 ms** | 145 ms | **1.5× faster** |
| prefill 2k / 8k / 16k / 32k | **6,408 / 7,365 / 7,451 / 7,182** | 5,279 / 5,711 / 5,977 / 6,106 | **+18-29%** |
| concurrency 1 / 2 / 4 / 8 / 16 | **151 / 231 / 350 / 478 / 635** | 126 / 197 / 303 / 427 / 542 | **+12-20%** |
| sanity set | 8 / 8 | | |

Qwen3.8-27B-NVFP4 on 2× R9700 (TP2) is unchanged from v0.1.0: 94% of the reference stack on decode, 80-86% on
prefill (the PCIe 3 all-reduce ceiling on this host).

## What moved it

- **Prefill (+20-31% over the reference at every depth):** a WMMA scorer for the sparse-attention indexer that
  touches only the visible columns; per-row scales and staged 16 B stores in the MoE prefill epilogues; the
  compressed hierarchical 4-rank all-reduce; switch-local P2P on both PLX switches (2× peer bandwidth); prefill
  chunks up to 2k tokens captured as HIP graphs (TTFT 137 → 88 ms).
- **Decode (+19% single-stream, +12-20% at concurrency):** on this ROCm each HIP-graph node costs ~1.5 µs of
  dispatch, and a Flash-Next step had ~3,250 of them against the reference stack's ~1,640. Five fusions replaced
  the runs of tiny launches with one kernel each: router GEMM, indexer norm + rope, hyper-connection mix,
  the shared expert (8 → 4 launches), and the Gated DeltaNet speculative-decode core (11 graph nodes → 1 per
  layer). The last two alone took the step from 19.0 to 16.8 ms.

## What's in it (new since v0.1.0)

- **Four-card host work** (`host/`): 32 GB BARs for two cards per PLX switch, guest-side BAR placement so every
  card sits at its host address, switch-local P2P on both switches. Documented, scripted, DKMS-packaged.
- **N-rank all-reduce** (one-shot / two-shot) and the **compressed 4-rank hierarchical** variant.
- **Sparse attention** kernels for Qwen4Exp and the **indexer scorer**.
- **Decode fusions** listed above, each with a `R9K_*=stock` knob and a bit-level or paired-eval gate.
- **vLLM 0.30** base; TP=4 tuning table; hybrid block-fp8 dispatch; MXFP4 LM heads.
- A `CHANGELOG.md`, and `notes/independence.md` lists every runtime hook beyond vLLM's extension points.

## Quality

Every change to a shipped default runs an 800-question chain-of-thought paired eval at concurrency 1 (paired
McNemar) plus the 8-prompt sanity set before it ships. RESULT_RELEASE_QUALITY

## Licence

**Apache-2.0**, with one carve-out: `serve/templates/qwen-fixed-v22.3.jinja` is GGZ14's, used with permission
and credit. `kernels/third_party/davetha/` is Apache-2.0 upstream and passes on normally. See `NOTICE`.

## Known limits

- **Two P2P chains only.** Switch-local P2P covers the pairs on each PLX switch; traffic between the switches
  still crosses the host's Gen3 uplinks, which is why the 4-rank all-reduce is compressed.
- **The stack is not reproducible at concurrency > 1** (dynamic batching changes reduction order); evaluate at
  concurrency 1, where it is near bit-reproducible (`bench/eval.py selftest`).
- **Numerics move a little with every fusion.** Each is gated, but the accumulated drift from a series of them is
  only visible in the paired evals, which is why they are run against the previous default rather than the
  original.

## Read next

`notes/picking-up.md` for orientation; `PROGRESS.md` for every number and every rejected idea; `CHANGELOG.md`
for the list.
