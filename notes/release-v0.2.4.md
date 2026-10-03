# r9700-stack v0.2.4

One change: the Gated DeltaNet layers keep **one state page per request** under speculative decoding instead of
vLLM's one page per candidate token. KV cache on the 27B on two cards goes from 223,329 to 367,494 tokens (+64%);
Flash-Next gains 18% on two cards and 9% on four; a running request pins a quarter to two thirds less of the pool;
and four concurrent 27B requests fit on one card, where stock pages admit two. Decode and concurrency are equal or
better, long prefill is 1-6% slower (a larger attention block). Output is bit-identical to the slot design.

**What this release was checked with, and what it was not.** Tagged, by decision, on the unit gates and strict
sanity under overload on every configuration (0 bad answers of 5,680), single-request texts identical to stock
pages, and sixteen-way concurrency agreement at least as good as stock's. **Not run:** a quality eval (GSM8K,
HumanEval), a full BetterBench, a mixed-length soak. Those were owed with v0.2.3 and are still owed. The test box
moved to 210 W and a -42 mV voltage offset the same day; probe numbers here are at that setting.

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

- Unit gates: `test_gdn_onepage_r9k.py` (new, 13 cases), `test_gdn_prefill_r9k.py`, the rest of the suite
  unchanged.
- Strict sanity under overload (more requests than `max_num_seqs`): 27B one card 0 bad of 360 and 0 of 2,040 at
  210 W with 17 clients; 27B two cards 0 of 990 twice; Flash-Next two cards 0 of 990 twice; Flash-Next four cards
  0 of 1,870. Reviewer passes by the 27B itself on the three diffs: no confirmed defect.
- Texts: six single-request prompts identical to stock pages at temperature 0; sixteen fixed prompts at 16
  concurrent, six runs, every run completing all 4,096 tokens, run-to-run agreement 7-16 of 16 (stock 5-16).
- Probes vs stock on every configuration (table above): decode and concurrency equal or better.
- The deliberate five-card reproduction of the night's host crash (three servers launched at once): clean.

## Not checked

- No quality eval (GSM8K or HumanEval) on this code or on v0.2.3's.
- No full BetterBench; the probe numbers are per-request client medians.
- No mixed-length soak.

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
