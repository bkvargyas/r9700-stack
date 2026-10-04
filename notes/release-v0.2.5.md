# r9700-stack v0.2.5

The v0.2.4 code, re-tagged with its validation record and the version string fixed. No behaviour change: against
v0.2.4 the plugin differs by the `__version__` string and three lines of off-by-default diagnostics
(`R9K_GDN_TRACE=1` now also covers the draft-less decode path).

**Why a tag.** v0.2.4 went out, by decision, on the unit gates and strict sanity under overload; the rest of the
release checklist ran the same day and was green on every configuration, and the tagged build announced itself as
0.2.3 in its startup log. This tag puts the record and the right version on the artifact people download.

## What was checked (on this code, 2026-10-03, 210 W and -42 mV, prefix caching off)

See `notes/release-v0.2.4.md`, section "Checked", for the tables. In one line each:

- GSM8K, full 1,319-question set at concurrency 1: 27B two cards 94.69%, Flash-Next four cards 95.68%, two cards
  95.60%. HumanEval 164: 97.56% / 96.34% / 98.17%.
- Paired thinking-mode GSM8K on the 27B (first 800, concurrency 1): one-page and stock pages both 96.88%, 0
  discordant; no detectable difference against the September baseline (p = 0.36) or the reference stack (p = 0.21).
- BetterBench: decode level with or above the 225 W references on all three configurations, concurrency up 6-12% on
  the 27B, prefill 0-11% lower (the lower cap and the larger attention block).
- Mixed-length soaks, 16 clients, 25 minutes each: 1,283 requests, 0 errors, VRAM flat or plateaued.
- Strict sanity under overload: 0 bad answers of 8,340.
- One-page is deterministic run to run where the slot design was not (PROGRESS.md 2026-10-03, evening).

## Also in this tag

- `notes/bug-mes-invalidate-tlbs.md`: the gfx1201 MES `INVALIDATE_TLBS` timeouts, the one escalation that reset
  the host, the per-card history, and the finding that linux-firmware's 2026-09-11 MES 0x93 does not reduce them --
  written as a comment for drm/amd issue 5759.

## Upgrading

Nothing to do beyond v0.2.4. `libr9k.so` is unchanged.
