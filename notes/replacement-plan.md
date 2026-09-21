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
