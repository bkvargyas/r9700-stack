# r9700-stack v0.2.2

A fix release for **Qwen3.8-Flash-Next with offloaded experts (two cards, TP2)**.

**If you run that configuration, upgrade.** v0.2.1 and every earlier version leak VRAM under mixed-length prompts
and run out of memory within minutes to hours of real traffic. Four cards with everything in VRAM were not
affected. Speed and model outputs are unchanged from v0.2.1.

## The bug

The expert cache kept one set of alignment buffers per batch shape, per layer, for good. A prefill chunk shares
each step's token budget with whatever is decoding, so under real traffic nearly every step has a shape nobody
has seen before, and each new one left up to 28 MiB per card behind. Benchmarks use fixed prompt depths and a
handful of shapes, which is why none of them showed it.

Measured at the v0.2.1 defaults with a new soak test (`bench/soak.py`: 8 clients for 15 minutes, each sending an
18k-30k-token slice of real text):

| | v0.2.1 | v0.2.2 |
|---|--:|--:|
| requests served / failed | 77 / **369** | 111 / 0 |
| VRAM used per card | 30.5 → 32.6 GiB in 13 minutes, then out of memory, engine dead | 31.0 GiB flat, one step to 31.6 |
| sanity set afterwards | server gone | 8 / 8 |

A rougher 31-minute soak of the fix (16 clients, prompts from 270 to 28.5k tokens): 291 served, 0 failed, the same
31.6 GiB plateau throughout.

**Workaround if you cannot upgrade:** none that has been tested. Restarting the server returns the memory.

## The fix

The buffers are kept only while a HIP graph is being captured, where replays need them at a stable address and
the capture sizes bound their number. Every other step uses ordinary temporaries.
`tests/test_cache_shapes.py`: 900 different eager batch shapes leave no buffer set and no allocated memory behind,
and a captured graph still replays new routings exactly.

## Also in this release

- **Four cards soaked for the first time** (TP4, same test): 236 served, 0 failed, VRAM flat at 31.96 GiB. Nothing
  found.
- **Two-card prefill depends on the prompt.** v0.2.1 reported prefill of 2,208-3,839 tok/s, 5-14% above v0.2.0.
  Those are BetterBench figures on its filler (one paragraph's words, shuffled), and the gain exists only for such
  narrow prompts. Real text prefills at about **2,300-2,600 tok/s** on v0.2.0, v0.2.1 and v0.2.2 alike
  (`bench/prefill_kinds.py`). The four-card numbers do not have this dependence.
- **`UTIL=0.96` is rejected for two cards.** It runs at 69 MiB free and dies when the request queue drains. The
  default stays 0.94, which peaks with about 980 MiB free.
- Releases are now soaked with mixed-length prompts on both configurations before they are tagged.

## Upgrading

No configuration changes. Restart once more after the first launch (as before, the launch that compiles gets a
smaller KV pool).

## Read next

`PROGRESS.md` (2026-10-01 section) for the measurements; `CHANGELOG.md` for the list.
