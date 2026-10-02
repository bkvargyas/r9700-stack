# r9700-stack v0.2.2

A fix release for **two cards** (tensor parallel 2): Qwen3.8-Flash-Next with offloaded experts, and Qwen3.8-27B.

**If you run two cards, upgrade.** Every release so far (v0.1.0, v0.2.0, v0.2.1) has two bugs there:

1. **Wrong output for some requests under concurrency.** When a request joins a batch that is already decoding,
   one sequence of the batch can turn to garbage: the right beginning, then one token repeated to the end.
2. **A memory leak** (Flash-Next with offloaded experts only) that ends in out-of-memory after minutes to hours of
   mixed-length prompts.

Neither shows in a fixed-depth benchmark or in an eval at concurrency 1, which is how both got through. Four
cards with everything in VRAM were not affected by the leak and produced no wrong answer in 15,000; they receive
the same all-reduce fix as a precaution. Speed and correct outputs are unchanged from v0.2.1.

**If you cannot upgrade:** `R9K_R4D_AR=0` (all-reduce on RCCL) avoids the wrong output. The leak has no setting
that avoids it; restarting the server returns the memory.

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

## Also in this release

- **Two-card prefill depends on the prompt.** v0.2.1 reported prefill of 2,208-3,839 tok/s, 5-14% above v0.2.0.
  Those are BetterBench figures on its filler (one paragraph's words, shuffled), and the gain exists only for such
  narrow prompts. Real text prefills at about **2,300-2,600 tok/s** on every version (`bench/prefill_kinds.py`).
  The four-card numbers do not have this dependence.
- **`UTIL=0.96` is rejected for two cards**: it runs at 69 MiB free and dies when the request queue drains.
- **A same-day baseline for the 27B** against the reference stack on the same two cards: decode 196.6 vs 197.5
  tok/s, prefill 81-87%, time to first token 116 vs 65 ms, concurrency 94-97%.
- **Release checklist.** A release is now stressed at more requests than `max_num_seqs` and soaked with
  mixed-length prompts, on every configuration, before it is tagged; the sanity check is strict.

## Upgrading

No configuration changes. `libr9k.so` must be rebuilt (`kernels/build.sh`): the plugin refuses an older one.

## Read next

`PROGRESS.md` (2026-10-01 and 2026-10-02 sections) for the measurements and the bisect; `CHANGELOG.md` for the list.
