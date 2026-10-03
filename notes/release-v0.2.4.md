# r9700-stack v0.2.4

One change: the Gated DeltaNet layers keep **one state page per request** under speculative decoding instead of
vLLM's one page per candidate token. KV cache on the 27B on two cards goes from 223,329 to 367,494 tokens (+64%);
Flash-Next gains 18% on two cards and 9% on four; a running request pins a quarter to two thirds less of the pool;
and four concurrent 27B requests fit on one card, where stock pages admit two. Decode and concurrency are equal or
better, long prefill is 1-6% slower (a larger attention block). Output is bit-identical to the slot design.

**Checked, the same day as the tag (2026-10-03, 210 W and -42 mV on every card, prefix caching off so one-page
is active on the 27B):** the unit gates; strict sanity under overload on all three configurations (0 bad of 8,340
across the day); GSM8K on the full 1,319-question test set at concurrency 1 and HumanEval on all 164 problems on
all three configurations; the paired thinking-mode GSM8K against the September baselines on the 27B; a full
BetterBench on all three configurations; a 25-minute mixed-length soak (16 clients, 200-30k-token prompts) on all
three with VRAM sampled. Numbers in the tables below. The tag itself went out a few hours before these finished,
by decision; nothing in them changed the code.

## What it does

vLLM verifies the speculative candidates of a step from a checkpoint and must be able to resume from whichever
candidate turns out to be the last accepted one, so it keeps a state page per candidate: 1 + 7 pages per request
per GDN layer group on the 27B. The plugin's verify kernel (`r9k_gdn_spec_verify`) carries, inside the single
page, a record of the previous step's candidate rows (their conv'd q/k/v and the a/b gates, two slots so a step can
read one and write the other, and a flag word for the live slot, its parity and a done counter). At the start of a
step it replays the accepted rows from the record into the state, stores that as the new checkpoint, then runs the
new candidates. The replay recomputes exactly what the slot design stored, so the result is bit-identical to it
(`tests/test_gdn_onepage_r9k.py`: multi-step simulation with random acceptance, both state dtypes, the sigmoid
gate, row maps, several layers, and identical requests in lockstep against the same request alone).

Stage 1 covers `mamba_cache_mode` "none", i.e. prefix caching off: `PREFIX_CACHE=0`, which is already the default
for Flash-Next and a knob for the 27B (`serve/27b.sh ... PREFIX_CACHE=0`). With prefix caching on, vLLM's pages are
used unchanged. `R9K_GDN_STATE=stock` forces the old layout.

| 210 W, -42 mV, prefix caching off | stock pages | one page | |
|---|--:|--:|---|
| 27B, two cards: KV cache | 223,329 | 367,494 | +64% |
| 27B, two cards: pool pinned by 8 running requests | 59.4% | 11.5% | |
| 27B, two cards: decode / c8 / c16 / 8k prefill, tok/s | 206 / 534 / 620 / 4,055 | 212 / 579 / 673 / 3,940 | |
| 27B, one card: KV cache | 37,732 | 62,295 | +65% |
| Flash-Next, two cards (restarted launches): KV cache | 154,641 | 183,202 | +18% |
| Flash-Next, two cards: pinned per running request | 10,386 | 6,736 | -35% |
| Flash-Next, four cards: KV cache | 256,682 | 279,564 | +9% |
| Flash-Next, four cards: pinned per running request / 8 running | 6,152 / 19.2% | 3,739 / 10.7% | |
| Flash-Next, four cards: decode / c16 / 8k prefill | 194 / 893 / 7,230 | 195 / 898 / 6,790 | prefill -6% |

## A defect found and fixed before the tag

The stock conv-window update takes `max_query_len`, the number of candidate columns the window rolls by, and the
plugin passed the width of the speculative state-index tensor. With one page per request that KV group's block
table has a single column, so on every step vLLM builds eagerly -- mixed prefill-plus-decode steps, non-uniform
batches -- the width was 1 and the window rolled as if each step had one candidate. Full-cudagraph decode steps
were right only because vLLM fills its graph buffer with a broadcasting copy. The symptom was subtle: a lone
request matched stock exactly, identical copies in a batch agreed with each other but not with the lone run, and
under `--enforce-eager` the output was garbage. A per-step checksum trace of the first GDN layer found it in the
second step after a prefill. The width now comes from the record. PROGRESS.md 2026-10-03 has the whole trail,
including what remains of batch dependence afterwards and why none of it is in the GDN path (the compressed
all-reduce on two cards, and the drafter's choices at three or more identical prompts in one step).

## Checked

| 210 W, -42 mV, one-page | 27B two cards | Flash-Next four cards | Flash-Next two cards (offload) |
|---|--:|--:|--:|
| GSM8K full 1,319, no thinking, conc 1 | 94.69% | 95.68% | 95.60% |
| HumanEval 164 | 97.56% (160) | 96.34% (158) | 98.17% (161) |
| BetterBench combined decode, tok/s (README 225 W ref.) | 200.3 (197.5) | 160.9 (159.2) | 97.3 (97.4) |
| BetterBench concurrency 1 / 2 / 4 / 8, tok/s | 185 / 314 / 450 / 558 | 151 / 236 / 353 / 505 | 88 / 109 / 117 / 114 |
| BetterBench prefill 2k / 8k / 16k / 32k, tok/s | 3,860 / 4,076 / 4,035 / 3,825 | 5,689 / 6,930 / 7,169 / 6,929 | 2,149 / 3,511 / 3,834 / 3,784 |
| time to first token p50 | 71 ms | 77 ms | 463 ms |
| mixed soak, 16 clients, 25 min | 383 served, 0 errors, VRAM flat | 661 served, 0 errors, VRAM plateau | 239 served, 0 errors, 14 over-length prompts rejected by design |
| strict sanity under overload | 0 bad of 900 and 960 | 0 bad of 1,700 and 1,920 | 0 bad of 900 and 960 |

Against the README's 225 W references: decode and step time level or slightly better (the undervolt), concurrency
up 6-12% on the 27B and 6% at 8 clients on four cards (the KV room), first token keeps the v0.2.3 gain, prefill
4-11% lower on four cards and 0-8% on the 27B at the short depths (the lower cap plus the larger attention block).

**Paired GSM8K, 27B two cards, thinking on, first 800 questions, concurrency 1** (the comparable form of the
September baselines): one-page 96.88% and stock pages 96.88% on the same code, 0 discordant questions, 782 of 800
outputs byte-identical; against our v0.2.0-era baseline (97.50%) 7 vs 12 discordant, p = 0.36; against the reference
stack (97.62%) 5 vs 11, p = 0.21: no detectable difference either way. The 99% output divergence from the September
runs is the v0.2.3 short-prefill core taking a more exact numeric path.

**Determinism.** Replaying the 18 differing thinking answers alone, twice per server: one-page gave identical
text and identical step counts every time, on both card pairs; stock pages did not (one question changed length
between two runs on the same server), and stock's step counts moved with the attention block size too. The slot
design has a rare run-to-run wobble that one-page does not; accuracy is unaffected. Low-priority follow-up now that
one-page replaces it.

- Reviewer passes by the 27B itself on the three diffs: no confirmed defect.
- The deliberate five-card reproduction of the night's host crash (three servers launched at once): clean.

## Not checked

- Nothing from the v0.2.2 checklist is outstanding for this release.

## Operational notes from the same day (PROGRESS.md 2026-10-03)

- The test host crashed once when a chronic MES firmware timeout on one card escalated into a GPU reset; under
  passthrough the reset took the card off the bus and the host sync-flooded. The timeout has occurred ~130 times
  since September on every card and self-recovers; `amdgpu.gpu_recovery=0` in the guest is the proposed
  containment. Not caused by this code or by load.
- All cards now run at 210 W with a -42 mV offset (LACT); the guest's power-cap service had a bug that left cards at
  225 W after a reboot, fixed.

## Upgrading

Rebuild `libr9k.so` (`kernels/build.sh`; `serve/serve.sh` does it when the kernels are newer than the library).
To get the memory, run with prefix caching off (`PREFIX_CACHE=0` for the 27B); nothing else changes. First launch
of a configuration compiles and keeps ~0.45 GiB; restart once, as before.

## Files

- `kernels/r9k_gdn.hip`: `r9k_gdn_spec_verify`, `r9k_gdn_spec_commit`, flags in `r9k_gdn_seq`; `kernels/build.sh`.
- `r9700_vllm/models/gdn.py`: `spec_verify`, `_spec_width`, the one-page KV spec and binding, the mixed-step and
  non-spec-decode paths, `R9K_GDN_STATE`, diagnosis knobs `R9K_GDN_TRACE` / `R9K_GDN_PAGE_PAD`.
- `r9700_vllm/models/qwen3_5.py` (new), `r9700_vllm/models/qwen4_exp.py`, `r9700_vllm/__init__.py`: model classes
  that size the mamba page for the record.
- `tests/test_gdn_onepage_r9k.py` (new).
- `PROGRESS.md` 2026-10-03: design, measurements, the defect and its trace, the host crash, the undervolt.
