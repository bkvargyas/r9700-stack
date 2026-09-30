# r9700-stack v0.2.1

A tuning-and-corrections release for **Qwen3.8-Flash-Next on two R9700s** (TP2, experts in host RAM), plus a
correction to v0.2.0's quality statement. No new kernels. The four-card numbers and all numerics are those of
v0.2.0; stock vLLM (0.30 nightly) and stock ROCm 10, as before.

## Performance: Flash-Next on 2× R9700 with offloaded experts

Full BetterBench (20 passes), one card per PLX switch, 225 W cap, 34 GB of experts per rank in host RAM, 270
expert slots per layer in VRAM, MTP-3.

| | v0.2.1 defaults | v0.2.0 defaults | |
|---|--:|--:|--:|
| single-stream decode | **97.4 tok/s** | 93.8 | +4% |
| decode step p50 | 25.6 ms | 25.7 ms | = |
| time to first token p50 | **484 ms** | 555 ms | -13% |
| prefill 2k / 8k / 16k / 32k | **2,208 / 3,558 / 3,839 / 3,793** | 2,099 / 3,163 / 3,434 / 3,329 | +5% / +12% / +12% / +14% |
| concurrency 1 / 2 / 4 / 8 / 16 | **87 / 109 / 116 / 118 / 113** | 84 / 104 / 108 / 111 / 95 | +4% / +5% / +7% / +6% / +19% |
| time to first token at 8 concurrent | **0.95 s** | 7.1 s | 7× faster |
| 8 copies of one prompt (code / json) | 434 / 577 tok/s | 283 / 347 | +53% / +66% |
| sanity set | 8 / 8 | 8 / 8 | |

## What changed

Three defaults, each found by measurement and confirmed by the BetterBench above:

- **`NSEQ=8`** in `serve/flashnext.sh`. Every request holds 18 KV blocks whatever its length (vLLM: four
  recurrent-state groups × (1 + 3 MTP blocks) + 2 attention blocks). With graphs and activations sized for 16
  sequences there was KV room for four to six; the rest queued. Pass `NSEQ=16` on four cards.
- **`R9K_LRU_THRESH=0.99`** (was 0.5). Eight different requests route to about 150 distinct experts per layer.
  Above half the slots the cache manager inserted nothing -- on 72% of steps -- so the cache stopped following the
  traffic and every step read 31 experts per layer from host RAM. With only `NSEQ=8`, conc-8 *fell* to 77 tok/s;
  with the threshold raised it is 118.
- **`R9K_LRU_GATHER=64,16`** (was 8,16): the insert copy is ~15% faster at 8 or more inserts, equal below.

None of them can change outputs: the expert cache is bit-identical to reading every expert from the backing store
(`tests/test_cache_moe.py`).

## What bounds this configuration

The PCIe link, not the GPUs. Every routed expert that is not resident is 1.245 MiB per card, and the copy kernel
already runs at the link rate (11.4-13.5 GB/s on PCIe 3). Flash-Next routes diffusely -- 90% of a layer's routing
needs 216-365 of its 512 experts -- so real, mixed traffic misses 12-16% of its routed experts per step and total
throughput levels off near 115 tok/s from four requests up. More users share that total; they do not add to it
(per-request decode 96 / 64 / 34 / 17 tok/s at 1 / 2 / 4 / 8).

The cards agree: a kernel is always running, but they draw 206 W with one request and 170 W with eight, against
219 W in prefill. Under concurrent load most of the compute is waiting for experts.

Near-identical requests are the best case, because they share their experts: eight copies of one prompt run at
434-577 tok/s. Four cards with everything in VRAM are 1.6× faster single-stream and 3-6× at concurrency, and a
PCIe 4 or 5 host would lift the two-card numbers with no code change.

## Quality: a correction to v0.2.0

v0.2.0's notes said the round-3 decode fusions together sit about half a point below the earlier numerics. **They
do not.** On the full GSM8K test set (1,319 questions, chain-of-thought, concurrency 1, paired):

| fusions on | accuracy | reference-only right | leg-only right | p |
|---|--:|--:|--:|--:|
| none (reference) | 96.82% | | | |
| none, six hours later | 96.82% | 0 | 0 | 1.00 |
| router GEMM | 96.82% | 6 | 6 | 1.00 |
| indexer norm + rope glue | 96.97% | 0 | 2 | 0.50 |
| hyper-connection mix | 97.04% | 5 | 8 | 0.58 |
| GDN speculative-decode core | 96.97% | 5 | 7 | 0.77 |
| shared expert | 96.74% | 6 | 5 | 1.00 |
| **all five (the default)** | **97.04%** | 5 | 8 | 0.58 |

The half point came from evaluating on the first 800 questions, where the reference happens to score 98.0% (95.0%
on the other 519): anything that reshuffles marginal questions looks like a loss on that slice. Evals for a change
of default now use the full set. Nothing needs switching off.

## New tools

- `R9K_EXPERT_CACHE_STATS=1`: per-rank counters from the expert cache, logged every
  `R9K_EXPERT_CACHE_STATS_SEC` seconds (distinct routed experts, inserts, experts read through from host, steps
  over the insert threshold). Opt-in; it adds a few small launches per layer.
- `bench/mix.py`: conc-8 throughput by workload mix (one prompt type against several).
- `tuning/lru_gather_bench.py`: host → VRAM copy rate of the insert kernel by insert count and launch grid.

## Operational notes

- **Restart once after the first launch of a new configuration.** The launch that compiles leaves ~0.45 GiB less
  for the KV pool (94k against 121k tokens at `NSEQ=8`).
- On a box with two PLX switches put **one card on each** for offloaded experts (`GPUS=0,2`), and both on one
  switch for plain tensor parallel.

## Tried and rejected

240 expert slots (more KV room; -28% on a mixed batch, -6% single stream), `NBT=2048` as a default (-24%
prefill), re-splitting the slots across layers by routing mass (13.3% → 12.8% misses). `UTIL=0.96` ran clean with
28% more KV room but has not held a 32k context under load, so it is not a default.

## Read next

`PROGRESS.md` (2026-09-29 sections) for every number above and the method behind it; `CHANGELOG.md` for the list.
