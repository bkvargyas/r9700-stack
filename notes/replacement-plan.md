# Plan: replace everything not under Apache-2.0 or MIT

**Goal (Brian, 2026-09-21):** every component this repo ships is either ours, or carries a permissive licence
(Apache-2.0 / MIT). When this is finished the two carve-outs in `NOTICE` disappear and the whole repo is cleanly
Apache-2.0 — no "used with permission" asterisks, and forks get the same rights we have.

**Current decision: keep the defaults as they are.** Our attention is the default (it is 99.8% of libr4d, so it
costs nothing). libr4d's all-reduce is still the default because ours is ~11% slower on prefill. Nothing in this
plan is urgent; it is a path to licence cleanliness, not a performance fix.

## Scope

Everything else is already fine and needs no work:

| component | licence | action |
|---|---|---|
| vLLM, PyTorch, Triton, ROCm | Apache-2.0 / BSD / MIT | none — runtime deps, not vendored |
| davetha LRU expert-cache kernels | Apache-2.0 | none — vendored with LICENSE + NOTICE intact |
| our attention, all-reduce, PLE, quantisers, prefill GEMMs | ours | none |
| model checkpoints (tcclaviger, unsloth) | per model card | out of scope — weights, not code in this repo |

Three things actually need replacing:

| # | item | files | cost to replace | risk |
|---|---|---|---|---|
| 1 | libr4d all-reduce (still the default) | `r9700_vllm/comm/r4d_ar.py` | **~11% prefill** unless we close the gap | low |
| 2 | libr4d-derived GEMM | `kernels/r9k_moe_mxfp4a8.hip`, `kernels/r9k_gemm_fp8.hip`, `r9700_vllm/kernels/moe.py:100-109` | **~5%** | medium |
| 3 | GGZ14 chat template | `serve/templates/qwen-fixed-v22.3.jinja` | **+14% acceptance at risk** | **high** |

## Sequencing, and why

Phase 1 first because it is the only one that *wins back* performance rather than spending it — and until it is
done, "go fully independent" means accepting an 11% prefill cut, which nobody will want to do. Phase 3 last
because it is the only one that can make the model measurably worse and the hardest to verify.

---

## Phase 1 — close the all-reduce gap, then flip the default

Our exact path already matches libr4d (decode 25.4 vs 25.1 ms/step). The whole 11% is the **compressed**
large-message path: 179 us at 2 MB, running at **~0.22 us/KB against 0.115 us/KB for our own exact kernel**. It is
overhead-bound, not link-bound — the compressed payload is 2.56x smaller yet takes longer per byte.

**Step 1.1 — profile before writing anything.** `rocprof` the three-kernel split path (pack → push → reduce) at
2 MB and account for all ~87 us of non-link time: per-kernel duration, gaps between launches, and fence cost.
*This step is mandatory.* One attempt at this was already wasted guessing — fusing all three into one kernel
measured **314 us vs 179** (see `notes/independence.md`), because it serialises the grid around the handshake.

**Step 1.2 — act on what the profile says.** Candidates, in the order they are cheap to test:
- **Fence strength.** `drain`/`acq` are already parameters (`3,0` for fine-grained). Test `drain=1` and `2`.
  A system-scope `__threadfence_system()` per block is not free, and we never measured whether we need it.
- **Two kernels instead of three.** Fuse pack→push (keeps the parallel reduce, avoids one launch, one fence and
  one 800 KB DRAM round trip) without fusing the reduce, which is what killed the all-in-one attempt.
- **Overlap.** The pack has no dependency on the peer; it could run while the previous layer's work drains.

**Acceptance:** `R9K_AR_IMPL=r9k` within **3%** of libr4d on 9k prefill and conc-8, decode unchanged, GSM8K flat.
Then flip the default, delete `r4d_ar.py`, and drop the two libr4d all-reduce rows from `NOTICE`.

**Fallback if it will not close:** ship it anyway with the default flipped and document the cost. 11% of prefill
to remove a licence dependency is a legitimate trade; it should just be Brian's explicit choice, not a default
someone inherits by accident.

### Attempt 1 (2026-09-21): decode gaps closed, prefill unchanged. Default NOT flipped.

Step 1.1 paid for itself immediately by **refuting the premise of this phase**. There was no per-call overhead to
remove: profiled, the push kernel is within 1-2 us of our exact kernel at the same byte count and the three-launch
gap is under 10 us. The "0.22 vs 0.115 us/KB" figure that motivated Phase 1 was an arithmetic error -- compressed
time divided by *wire* bytes against exact time divided by *raw* bytes. Per raw byte the compressed path was
already ahead (0.085 vs 0.114).

What the profile did find: a **~54 us/call system-scope fence cost**. The sweep showed `drain=2` was fast, but
`drain<3` is `__threadfence()` -- *agent* scope, which orders nothing for a peer GPU. Fast, correct-looking in a
lockstep test, and a real race. Replaced instead with a **release store / acquire load** at system scope
(`drain=4, acq=2`, now the default), which states exactly the ordering the handshake needs and lets the hardware
emit the minimum barrier. 2 MB compressed 174.7 -> 128.8 us; the exact kernel 233.3 -> 206.7 us as a side-effect.

End to end that bought the **decode** side and nothing else:

| | ours before | ours after | libr4d | gap now |
|---|---|---|---|---|
| ms/step | 25.7 | **25.1** | 24.9 | +0.8% |
| conc-8 | 422.2 | **460.6** | 484.8 | -5.0% |
| single decode | 114.5 | 115.6 | 117.1 | -1.3% |
| prefill 9k | 3,909 | 3,920 | 4,403 | **-11.0%** |

**Why prefill did not move, and the lesson:** a 26% win at 2 MB is irrelevant because prefill all-reduce messages
are **not** 2 MB. At `NBT=4096` and hidden 5120 a prefill message is ~40 MB, where a 54 us fixed saving sits
against a ~1.3 ms transfer. I optimised the size that was convenient to benchmark rather than the size that
dominates the workload. **Profile at the size the workload actually uses.**

### Attempt 2 (2026-09-21): pipelining ruled out by measurement. We are link-bound.

Measured the real sizes first (`R9K_AR_HIST=1`, now built in): prefill drives **16-32+ MiB compressed** messages
(the 32 MiB bucket went 387 -> 1,104 over three 9k prefills); decode lives in 8-512 KiB. So the earlier ~40 MB
arithmetic was right, and attempt 1 tuned the wrong size.

Then profiled a 9k prefill on both backends:

| | ours | libr4d |
|---|---|---|
| GPU busy | 2,267 ms | 1,991 ms |
| all-reduce total (387 calls) | **693 ms** = push 442 + pack 145 + reduce 106 | **375 ms**, one fused kernel |
| per call | **1.79 ms** | **0.97 ms** |

**This rules out the pipelining plan arithmetically.** libr4d's entire fused all-reduce (0.97 ms) is faster than
our `push` phase alone (1.14 ms), so even perfectly hiding *all* of pack and reduce behind the transfer lands at
1.14 ms/call -- still 18% behind. Pipelining cannot close this.

And our push is not inefficient: 15.6 MB in 1.14 ms = **13.7 GB/s, at PCIe 3's practical ceiling**. We are
link-bound, not overhead-bound. Nothing about scheduling, fusion or overlap gets past a saturated wire.

So libr4d is either moving **fewer bytes** than we are, or streaming transfer and compute together inside one
kernel so that "exchange" is never a distinct phase. **We do not know which, and should not guess** -- this phase
has already produced three wrong causes reasoned past the evidence.

### What is actually left for Phase 1

The only lever consistent with being link-bound is **sending fewer bytes**:

1. **Fewer bits per element.** 4-bit instead of 6-bit cuts the payload ~1.5x -> push ~0.76 ms/call, which would
   put the total near libr4d even without overlap. Cost: relative error goes from ~0.025 to ~0.1 per call, over
   ~387 calls per prefill. **A quality trade, so it is Brian's decision, and it must be gated on a real eval --
   GSM8K-500 provably cannot resolve it (see the measurement rules in notes/picking-up.md).**
2. **A smaller scale field.** fp8 rather than bf16 per group, or a 128-element group: both ~6.125 bits/elem
   against our 6.25. Marginal (~2%), cheap, no quality cost worth mentioning. Not enough alone.
3. **Understand what libr4d actually does differently** before building anything else. The honest statement is
   that a ~15% byte or bandwidth advantage is unaccounted for.

### Attempt 3 (2026-09-22): 4-bit built and measured. +5.3% prefill, no detectable quality cost.

`R9K_AR_QUANT_BITS=4` (opt-in; 6 is the default) ships 4.25 bits/elem instead of 6.25 -- 34 bytes per 64-element
group instead of 50, 1.47x fewer bytes, which is the only lever that exists once the link is saturated.

| our AR | 6-bit | 4-bit | libr4d |
|---|---|---|---|
| prefill 9k | 3,913 | **4,120** (+5.3%) | 4,403 |
| prefill 20.7k | 3,717 | **3,915** (+5.3%) | 4,153 |
| ms/step | 25.2 | 25.1 | 24.9 |
| GSM8K (1,319, conc=1) | 94.77% | 94.39% | 95.45% |

**Prefill gap 11% -> 6.4%.** Quality, controlled comparison (only the width changes): McNemar **p=0.551, no
detectable difference**. A confounded base-vs-4bit comparison reads p=0.034, which should not be believed: it
changes the AR backend AND the width together, it is the fifth pairwise test run (fails Bonferroni at 0.05/5),
and the ordering is incoherent as quality -- libr4d-*compressed* scored 95.45% while libr4d-*exact* scored
94.77%, and compression cannot improve accuracy. Our 6-bit ties libr4d-exact exactly (94.77%, 1250/1319). These
are ~9-question numerical differences, not quality.

Honest limits: the eval resolves ~1%, so a smaller effect could hide; and the per-call perturbation really is
4.4x larger (rel 0.108 vs 0.024, and 85.6% of outputs differ) even though accuracy does not move.

**Enabled 2026-09-22 (Brian: enable it, but do not sacrifice accuracy).** Default is now 4-bit, after a second,
harder eval. The first test used short non-thinking answers -- the condition where a per-call perturbation has
the LEAST chance to compound -- so a null there was the weakest possible evidence. The long-chain test was run
precisely because it should be the most sensitive:

| | 6-bit | 4-bit | discordant | McNemar |
|---|---|---|---|---|
| GSM8K 1319, short answers | 94.77% | 94.39% | 25 / 20 | p=0.551 |
| GSM8K 800, full chain-of-thought | 96.88% | **97.00%** | 4 / 5 | **p=1.000** |

**The prediction was wrong in an informative way.** Long chains do not compound the error, they absorb it: only
**1.1%** of answers changed outcome under thinking, against **3.4%** on short answers, because later reasoning
catches and corrects a perturbed intermediate step. Two independent nulls with point estimates in opposite
directions (-0.38pp, +0.12pp) is what a genuinely zero effect looks like.

Bound honestly: both evals resolve ~1%, so this is "smaller than we can measure", not "exactly zero".

**Recommendation: stop here either way.** The decode half of Phase 1 is done; the prefill half is link-bound and
4-bit is the last byte-count lever short of understanding what libr4d does differently, which is unknown.

---

## Phase 2 — re-derive the GEMM

From the provenance audit (`notes/independence.md`): the file is 1,268 lines, ~9% still plausibly libr4d-derived.
fp8 WMMA and its fragment layout are fixed by the ISA, split-K through LDS is textbook, and neither is libr4d's
to own. **The one genuinely inherited idea is the folded-exponent MXFP4 scheme**, and it has grown more
load-bearing since the audit because the A-tiled prefill kernel is folded-only.

**Step 2.1 — decide the strategy, because there are two and they are very different.**

- **(a) Re-derive the fold.** Regenerate `kMag` from the OCP e2m1/e4m3 definitions with a checked-in generator
  script; replace the 4x`v_perm_b32` unpack with a shift/LUT version (priced at **~5%** in
  `notes/review-fable.md:270`); re-derive the `Wref` = row-max-E8M0 / `2^(Wref-127)` ABI.
- **(b) Drop the fold entirely.** The fp32 per-group scaling path **is already ours** and is the code default
  (`R9K_FOLD` defaults to `0`). The blocker is that the A-tiled prefill kernel requires fold. Extend A-tiled to
  the non-folded path and the whole inherited scheme can be deleted rather than re-derived.

**(b) is probably better and nobody has costed it.** It trades "rewrite a clever thing without looking at it" for
"extend a kernel we wrote last week", and it removes the lineage instead of paraphrasing it. Measure what fold is
actually worth on the *current* stack first — the +5% prefill / +4% conc-8 figure predates the A-tiled kernel and
may no longer hold.

**Step 2.2 — the layout convention.** `r9700_vllm/kernels/moe.py:100-109` says it is "identical to libr4d
`mxfp4_layout.permute_w`". Re-derive it from what the gfx12 WMMA builtin requires and document that derivation.
The resulting bytes will very likely be identical — that is fine and expected, because the layout is forced by
hardware. What changes is that we can show how we got there.

**Step 2.3 — `kernels/r9k_gemm_fp8.hip`** carries the same lineage (`notes/review-fable.md:267`) and needs the
same treatment. Easy to forget; it is not the file anyone thinks of.

**Acceptance:** no derivation notice left in any header; `git grep -i libr4d kernels/` returns only comments
explaining what we did *not* take; all gates pass; the perf delta measured and recorded, not estimated.

---

## Phase 3 — replace the chat template

**The riskiest item, and the one most likely to quietly cost quality.** `qwen-fixed-v22.3.jinja` is worth about
**+14% speculative acceptance** (code category 3.34 → 4.49 tok/step). That gain came from specific corrections we
did not derive ourselves, so "write our own template" is not a rewrite — it is a re-derivation plus a measurement.

**Step 3.1 — diff it against the checkpoint's own template** and enumerate exactly what it fixes. Expect these to
be *facts about the model's expected prompt format* (tool-call framing, thinking-block handling, whitespace and
BOS/EOS placement) rather than creative expression. Facts are re-derivable from the model card and the
checkpoint's tokeniser config; that is the legitimate path.

**Step 3.2 — write ours from the checkpoint template plus those corrections**, documenting the reason for each.

**Step 3.3 — measure acceptance, not just correctness.** A template can be perfectly valid and still cost 14% of
decode. Compare `tok/update` per BetterBench category against the current template, on the same build.

**Acceptance:** acceptance within noise of `qwen-fixed` across categories, GSM8K flat. If we cannot reproduce it,
**stop and keep the credited copy** — GGZ14 gave permission, so shipping it is a legitimate outcome, and a 14%
decode regression to remove a carve-out on a *template file* is a bad trade.

---

## Definition of done

- `NOTICE` lists only Apache-2.0 third-party code (davetha), with no "used with permission" entries.
- `git grep -riE "libr4d|GGZ14|vllm-mxfp4" -- kernels/ r9700_vllm/ serve/` returns only historical notes.
- `CREDITS.md` keeps every acknowledgement — removing the dependency is not a reason to stop crediting the people
  whose work we learned from. Ideas stay credited even though ideas need no licence.
- Gates green, and a full BetterBench run recorded against the pre-replacement numbers so the total cost of
  licence cleanliness is a measured number rather than a guess.

## What this is not

This is not a performance project. Phase 1 wins back ~11%; phases 2 and 3 cost up to ~5% and risk 14% of decode
acceptance respectively. The deliverable is **a repo anyone can use under Apache-2.0 without asking anyone's
permission** — worth doing if this is ever published properly, shared, or built on. If it stays a private
project on one box, the honest answer is that the current state (permission granted, carve-outs documented) is
already sufficient and this plan can sit here unexecuted.
