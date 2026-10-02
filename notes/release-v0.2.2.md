# r9700-stack v0.2.2

A fix release. **Upgrade, whatever you run.** Every release so far (v0.1.0, v0.2.0, v0.2.1) has three bugs that
no fixed-depth benchmark and no evaluation at concurrency 1 can see; each was found by loading the server the way
real traffic does.

| | affects | symptom |
|---|---|---|
| 1 | two cards: Flash-Next with offloaded experts, and the 27B | **wrong output**: when a request joins a batch that is already decoding, one sequence can turn to garbage (the right beginning, then one token repeated) |
| 2 | Flash-Next with offloaded experts | **memory leak**: out of memory after minutes to hours of mixed-length prompts |
| 3 | Flash-Next, any configuration; four cards at 16 concurrent first | **out of memory** when one step holds prompts of very different lengths |

Speed and correct outputs are unchanged from v0.2.1.

**If you cannot upgrade:** `R9K_R4D_AR=0` (all-reduce on RCCL) avoids the wrong output on two cards. The other two
have no setting that avoids them; fewer concurrent sequences (`NSEQ`) makes the third less likely, and restarting
the server returns leaked memory.

## 1. Garbled answers: a race in the 2-rank all-reduce

The two cards exchange and sum their halves of every layer's output through a double-buffered scratch, so that a
card already on the next message cannot overwrite what the other is still reading. Which half to use was decided
per block of the kernel, while the number of blocks followed the message size. After a small message the next
larger one could therefore land in the same half as the message before and overwrite its last rows. The two cards
then disagreed about the last sequence of the batch, and it decoded garbage from that step on.

It needs two back-to-back messages of different sizes with the cards slightly out of step, which is what happens
when a prompt joins a batch of running decodes. Measured with a strict version of the sanity check
(`bench/sanity_stress.py`), sending more requests than `max_num_seqs`:

| | v0.2.1 | v0.2.2 |
|---|--:|--:|
| Flash-Next TP2, 9 / 12 / 16 concurrent (limit 8): bad answers | 74 of 900 / 57 of 1,200 / 64 of 1,600 | 0 of 4,600 |
| Flash-Next TP2, sparse graph sizes: rounds with a bad answer | 99 of 100 | 0 of 100 |
| 27B TP2, 9 concurrent (limit 8): bad answers | 8 of 900 | 0 of 2,700 |
| Flash-Next TP4, 17 / 24 / 32 concurrent (limit 16): bad answers | 0 of 6,020 | 0 of 6,020 |
| the same server on RCCL or libr4d all-reduce | 0 of 1,620 each | |

The fix: every call advances every block's counter (a fixed launch grid; blocks without data do nothing else), so
all blocks always agree on the half. The 4-rank kernels were built the same way and are fixed the same way.
`tests/test_ar_race.py` reproduces the fault with the old grid (6,008 wrong outputs in 1,500 graph replays) and
passes with the new one.

## 2. Memory leak in the expert cache

The cache kept one set of alignment buffers per batch shape, per layer, for good. A prefill chunk shares each
step's token budget with whatever is decoding, so under real traffic nearly every step has a new shape, and each
left up to 28 MiB per card behind.

With a new soak test (`bench/soak.py`: 8 clients for 15 minutes, each sending an 18k-30k-token slice of real text):

| | v0.2.1 | v0.2.2 |
|---|--:|--:|
| requests served / failed | 77 / **369** | 111 / 0 |
| VRAM used per card | 30.5 -> 32.6 GiB in 13 minutes, then out of memory, engine dead | 31.0 GiB flat, one step to 31.6 |

Now there is one buffer set per layer, reused by every step. `tests/test_cache_shapes.py`.

## 3. Out of memory when a step holds prompts of very different lengths

Flash-Next's short convolution packs all the prompts being prefilled in one step into a batch of *prompts x
longest*, and holds about six tensors of that size. One long chunk next to many short prompts is many times the
step's real tokens: 3,000 + 15 x 70 tokens peak at 4.8 GiB in that one function, where vLLM's memory profile (one
sequence) budgeted for 84 MiB per tensor. This is vLLM's own algorithm; our kernel override mirrored it.

On four cards, a soak of 24 clients with prompts of 200 to 30k tokens took the cards from 29.4 to 32.2 GiB in 40
seconds and killed the engine. On two cards it showed as a one-time 640 MiB step that happened to fit.

The fix packs the step's prompts by length, in groups of at most 1.25x the step's own token count. Each sequence's
convolution reads only its own tokens and state, so the results are bit-identical (`tests/test_ple_conv.py`
compares with stock); the peak for the cases above is 0.4-0.5 GiB.

## Also in this release

- **Two-card prefill depends on the prompt.** v0.2.1 reported prefill of 2,208-3,839 tok/s, 5-14% above v0.2.0.
  Those are BetterBench figures on its filler (one paragraph's words, shuffled), and the gain exists only for such
  narrow prompts. Real text prefills at about **2,300-2,600 tok/s** on every version (`bench/prefill_kinds.py`).
  The four-card numbers do not have this dependence.
- **`UTIL=0.96` is rejected for two cards**: it ran at 69 MiB free and died when the request queue drained (bug
  3, as it turned out). The default stays 0.94.
- **A same-day baseline for the 27B** against the reference stack on the same two cards: decode 196.6 vs 197.5
  tok/s, prefill 81-87%, time to first token 116 vs 65 ms, concurrency 94-97%.
- **Release checklist.** A release is now stressed at more requests than `max_num_seqs` and soaked with
  mixed-length prompts, on every configuration, before it is tagged; the sanity check is strict.

## Upgrading

No configuration changes. `libr9k.so` must be rebuilt (`kernels/build.sh`): the plugin refuses an older one.

## Read next

`PROGRESS.md` (2026-10-01 and 2026-10-02 sections) for the measurements and the bisect; `CHANGELOG.md` for the list.
