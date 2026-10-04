# r9700-stack

Tuned GPU kernels and a vLLM plugin that make **Qwen3.8** run fast on **AMD Radeon AI PRO R9700** cards
(gfx1201 / RDNA4).

**Headline (September 2026): Qwen3.8-Flash-Next on four R9700s is now faster than the best known alternative
stack on every metric we measure** -- 159 tok/s single-stream decode against 134 (+19%), first token in 94 ms
against 145, prompt processing 18-29% ahead at every depth from 2k to 32k tokens, and 12-20% more aggregate
throughput at every concurrency level. All of it on stock vLLM and stock ROCm, from a plugin -- and on a **PCIe 3** host, where
the cards talk to each other at ~13.7 GB/s; the same code on a PCIe 5 box would move the multi-card numbers up
again. The numbers are in [Benchmarks](#benchmarks); the story of how each one moved is in
[PROGRESS.md](PROGRESS.md).

**Current release: [v0.2.5](notes/release-v0.2.5.md) (2026-10-04)**, the v0.2.4 code with its full validation record and the version string fixed. **v0.2.4 (2026-10-03):** One state page per request for the GDN layers
under speculative decoding: 64% more KV cache for the 27B on two cards (223k to 367k tokens), 18% for Flash-Next on
two cards, 9% on four, and four concurrent 27B requests on one card where stock pages admit two; decode and
concurrency equal or better, long prefill 1-6% slower. Needs prefix caching off (`PREFIX_CACHE=0`; the default for
Flash-Next, a knob for the 27B). Tagged on the unit gates and strict sanity under overload (0 bad of 5,680), not on
the full [release checks](#stability-what-a-release-is-checked-against); the notes say what was run.
**If you run anything before v0.2.2, upgrade:** v0.2.2 fixed three bugs that only show under real traffic (wrong
output on two cards when a request joined a running batch, a VRAM leak, and an out-of-memory under mixed-length
prompts).

**Built for stock upstream releases.** It targets **released vLLM** and **ROCm 10 or newer**, unmodified — no
fork, no patched source, no vendored binaries. Everything loads as a plugin at runtime through vLLM's own
extension points (quantization config, model registry, platform plugin, attention backend, custom ops), so you
can update vLLM or ROCm without re-porting anything. That constraint was the point of the project: hand-tuned
kernels normally mean a fork you then maintain forever.

Requirements: 2× or 4× Radeon AI PRO R9700 (gfx1201), ROCm ≥ 10, a released vLLM build, and the model weights.

### Checkpoint formats

Built primarily for **NVFP4** checkpoints — e2m1 codes with an e4m3 scale per 16 elements and an fp32 per-row
global, as published by `unsloth/Qwen3.8-27B-NVFP4` and similar. Two paths are supported and both are tested:

| | what happens | when to use it |
|---|---|---|
| `R9K_NVFP4=mxfp4` *(default in `serve/27b.sh`)* | NVFP4 is converted to MXFP4 once at load | **+6% everywhere.** The MXFP4 kernels are the faster ones, and the conversion is measured to cost no detectable quality |
| `R9K_NVFP4=native` | the checkpoint's own NVFP4 bits are kept, and a native NVFP4 kernel variant runs on them | when you want the checkpoint's numerics preserved exactly |

**MXFP4** checkpoints work directly — compressed-tensors `mxfp4-pack-quantized`, as used by the Flash-Next GPTQ
build. Any **fp8** layers inside a checkpoint can also be converted to MXFP4 at load (`R9K_FP8_TO_MXFP4=1`,
measured worth ~21% of prefill on the 27B).

So: bring an NVFP4 image and it will run; bring an MXFP4 one and it will run; mixed fp8/MXFP4 checkpoints are
handled by converting the fp8 parts.

## Benchmarks

### Qwen3.8-Flash-Next, 4× R9700 (TP4)

Flash-Next (the MoE + Gated DeltaNet model, MXFP4/fp8 GPTQ checkpoint), tensor-parallel over four cards at a
225 W cap, full BetterBench (20 passes), MTP-3 speculative decoding on both stacks. The reference column is the
fastest known alternative stack for this model on **the same box, the same checkpoint and the same power cap**.

| | this stack | reference stack | |
|---|--:|--:|--:|
| single-stream decode | **159.2 tok/s** | 134.1 | **+19%** |
| decode step p50 | **16.8 ms** | 20.4 ms | **-18%** |
| time to first token p50 | **94 ms** | 145 ms | **1.5× faster** |
| prefill 2k / 8k / 16k / 32k (tok/s) | **6,408 / 7,365 / 7,451 / 7,182** | 5,279 / 5,711 / 5,977 / 6,106 | **+21% / +29% / +25% / +18%** |
| concurrency 1 / 2 / 4 / 8 / 16 (aggregate tok/s) | **151 / 231 / 350 / 478 / 635** | 126 / 197 / 303 / 427 / 542 | **+20% / +17% / +16% / +12% / +17%** |

The table is the v0.2.0 measurement against the reference. v0.2.2 on the same box, same settings: 157.3-159.2
tok/s, step 16.8 ms, first token 97-100 ms, prefill 6,476 / 7,348 / 7,445 / 7,180, concurrency 151 / 239 / 355 /
473 / 624 -- the same numbers to within the run-to-run band, and unlike v0.2.0 it survives 24 clients sending
prompts of 200 to 30k tokens (450 requests in 15 minutes, none failed, 30.2 of 32.6 GiB at the peak; the code
before v0.2.2 ran out of memory about a minute into that).

Where it came from, in one line each: prefill from an MXFP4×FP8 MoE GEMM with the per-row scales in LDS, a
WMMA scorer for the sparse-attention indexer that only touches the visible columns, and a compressed 4-rank
all-reduce; decode from replacing hundreds of tiny per-step launches (norm + rope glue, router GEMM,
hyper-connection mix, the shared expert, the Gated DeltaNet speculative-decode core) with one kernel each --
on this ROCm every HIP-graph node costs about 1.5 µs of dispatch, so the launches were the cost.

The last two fusions (the Gated DeltaNet speculative-decode core and the four-launch shared expert) alone took
single-stream decode from 139 to 159 tok/s and the step from 19.0 to 16.8 ms, in one day.

Quality: the full GSM8K test set (1,319 questions) with chain-of-thought at concurrency 1, paired against the
previous numerics before any change to a default ships (exact McNemar). The default scores **97.04%** against
96.82% with all five decode fusions switched off (5 / 8 discordant, p = 0.58), and each fusion alone is equally
indistinguishable; the reference reproduces itself to 1,318 of 1,319 outputs. The fusions can still be switched off
individually (`R9K_*=stock`).

### Qwen3.8-Flash-Next, 2× R9700 (TP2) with experts in host RAM

The two-card way to run Flash-Next: the routed experts stream from pinned host memory through the plugin's LRU
expert cache (34 GB per rank offloaded, 270 expert slots per layer resident), so the model fits two cards with
room for a 120k-token KV cache. Full BetterBench (20 passes), measured on one card per PLX switch; v0.2.2 has
v0.2.1's defaults and re-measured at the same numbers (97.4 tok/s, 25.6 ms, 483 ms, 87 / 109 / 116 / 116 / 116):

| | v0.2.1 and v0.2.2 | v0.2.0 |
|---|--:|--:|
| single-stream decode | **97.4 tok/s** | 93.8 |
| decode step p50 | 25.6 ms | 25.7 ms |
| time to first token p50 | **484 ms** | 555 ms |
| prefill 2k / 8k / 16k / 32k (tok/s) | **2,208 / 3,558 / 3,839 / 3,793** | 2,099 / 3,163 / 3,434 / 3,329 |
| concurrency 1 / 2 / 4 / 8 / 16 (aggregate tok/s) | **87 / 109 / 116 / 118 / 113** | 84 / 104 / 108 / 111 / 95 |
| time to first token at 8 concurrent | **0.95 s** | 7.1 s |

Read it as a **link-bound** configuration for one to a few users. Every routed expert that is not resident is
1.245 MiB per card over PCIe, and the copy already runs at the link rate, so total throughput levels off near
115 tok/s from four requests up: more users share it, they do not add to it (per-request decode 96 / 64 / 34 /
17 tok/s at 1 / 2 / 4 / 8). Eight requests now run at once instead of four to six, which is what took the
time to first token at 8 concurrent from 7 s to under one. The cards say the same thing: always busy, and
drawing 206 W with one request but 170 W with eight, because they are waiting for experts. The best case is a
batch of near-identical requests, which share their experts: eight copies of one prompt run at 434-577 tok/s.
Four cards with everything in VRAM are 1.6× faster single-stream and 3-6× at concurrency; a PCIe 5 host would
narrow that gap without any code change.

**Prefill here depends on the prompt.** BetterBench's prefill filler is one paragraph's words shuffled, which routes
to few enough experts for the cache to follow; real text does not. Measured on the same server, real documents
prefill at about **2,300-2,600 tok/s** (8k-26k tokens) against 3,650 for the filler, and that figure is the same on
every version: the prefill gain in the table is a gain on narrow prompts (`bench/prefill_kinds.py`).

Two operational notes: **restart once after the first launch of a new configuration** (the launch that compiles
leaves ~0.45 GiB less for the KV pool: 94k against 121k tokens), and pass `NSEQ=16` for four cards
(`serve/flashnext.sh` defaults to 8, the right value for two).

**Card placement matters, in opposite directions.** Tensor-parallel traffic wants both cards on the same PLX
switch (switch-local P2P: the 4-card numbers above). Offloaded experts want one card per switch, because the
expert stream comes down each switch's single Gen3 uplink from the host: on the same-switch pair this exact
configuration measured 81 tok/s single-stream, 54 at eight streams and 1,379 tok/s prefill against 114 / 132 /
2,464 on the split pair. On the 4-card box that is `GPUS=0,2`.

### Qwen3.8-27B-NVFP4, 2× R9700 (TP2)

The shipped default configuration (`serve/27b.sh`): our own attention and all-reduce kernels, nothing third-party
loaded. Both columns were measured **on the same day (2026-10-02), on the same two cards, checkpoint and power
cap**, each with a full BetterBench (29 prompts across 8 categories, 20 passes); the right-hand one is the fastest
known alternative stack for this model.

| | this stack (v0.2.2) | reference stack | |
|---|--:|--:|--:|
| single-stream decode | **197.5 tok/s** | 197.5 | 100% |
| decode step p50 | 23.5 ms | 23.2 ms | |
| time to first token p50 | 114 ms | 65 ms | |
| prefill 2k / 8k / 16k / 32k (tok/s) | 4,190 / 4,191 / 4,072 / 3,837 | 4,780 / 4,947 / 4,903 / 4,746 | 88% / 85% / 83% / 81% |
| concurrency 1 / 2 / 4 / 8 (aggregate tok/s) | 174 / 280 / 413 / 519 | 180 / 303 / 428 / 558 | 97% / 92% / 96% / 93% |
| KV cache | 211k tokens | 799k tokens | |

Decode is at parity; prefill and KV capacity are where the work is (the reference prefills in 8,192-token chunks
and keeps an 8-bit KV cache).

**Time to first token is fixed in v0.2.3**: a short prompt's first token went from 97 to 47 ms, against the
reference's 46 -- and from 87 to 44 ms for Flash-Next on four cards. The cause was vLLM's chunked GDN prefill core
running eagerly in every GDN layer; short prefills now take one launch per layer. Measured per request with
`bench/ttft_breakdown.py`, not yet with a full BetterBench; see the [changelog](CHANGELOG.md).

Quality: GSM8K, full 1,319-question test set, greedy, concurrency 1 — **94.4–95.5%** depending on configuration,
with no statistically detectable difference between them (paired McNemar).

**Part of the prefill gap is a deliberate trade.** The default configuration uses our own all-reduce so that
nothing unlicensed is loaded at runtime; that costs about 6% of prefill against the third-party one. Setting
`R9K_AR_IMPL=r4d R9K_PAGED_ATTN=r4d` recovers it if you have that library and would rather have the speed. Beyond
that, large-message all-reduce on this host is bandwidth-bound on a PCIe 3 link at ~13.7 GB/s, which is the
practical ceiling.

[PROGRESS.md](PROGRESS.md) has every number, how it was produced, and what was tried and rejected.

### Stability: what a release is checked against

Benchmarks measure speed. They did not notice that three releases produced wrong output for some requests and ran
out of memory under real traffic, because every check ran at fixed prompt lengths and at or below the sequence
limit. Since v0.2.2 a release is meant to be tagged only after, on **every** configuration above and on the exact
release code (v0.2.3 was tagged on the first two rows plus the soaks for its memory change, by decision; v0.2.4
was tagged on the first two rows and the rest ran the same day -- all green, numbers in its notes):

| check | passes when | v0.2.2 |
|---|---|---|
| unit gates and all-reduce suites (`tests/`) | all pass | 26 + 5 |
| strict sanity under overload (`bench/sanity_stress.py`, more requests than `max_num_seqs`) | no bad answer | 0 of 9,500 |
| mixed-length soak (`bench/soak.py`, 12-24 clients, prompts of 200 to 30k tokens, 15-26 minutes) | nothing fails, VRAM levels off | 1,077 served, 0 failed; peaks 30.9 / 30.2 / 32.5 of 32.6 GiB |
| strict sanity right after long prompts | no bad answer | 0 of 480 |
| full BetterBench | within noise of the release before | yes, all three |

The commands are in [notes/picking-up.md](notes/picking-up.md).

### If your R9700s log MES timeouts (RDNA4 under KVM passthrough)

`amdgpu ...: MES(1) failed to respond to msg=INVALIDATE_TLBS` in the guest kernel log, during model loads and graph
capture, is a firmware behaviour we hit about once per heavy launch on three MES firmware versions, and once it
escalated into a GPU reset that took the card off the bus and the host down. Two guest kernel parameters settle it:

```
amdgpu.mes_log_enable=1 amdgpu.gpu_recovery=0
```

The first removed the timeouts in 60 of 60 launches that produced them otherwise, at no measurable cost; the second
keeps a future hang from becoming a host reset. The evidence, the mechanism as far as we can read it, and how to
apply it are in [notes/mes-timeouts.md](notes/mes-timeouts.md); the report to AMD is
[drm/amd #5759](https://gitlab.freedesktop.org/drm/amd/-/issues/5759#note_3693873).

### About the 225 W power cap

**Every number in this repository was measured with the cards capped at 225 W, and that is deliberate.**

The R9700 will draw more if you let it, and prefill in particular is clock-limited — the card sustains about
2.40 GHz during prefill against 2.82 GHz on decode, so raising the cap would improve prefill by some margin we
have never measured. We have not measured it on purpose, for two reasons:

1. **Comparability.** Every reference measurement we hold for other stacks was taken at 225 W on this same box.
   Raising ours would make the comparison apples-to-oranges — the exact mistake that produced one bad baseline
   earlier in this project.
2. **It matches production.** These cards run capped here permanently. A number measured uncapped is a number
   nobody would ever see in service, and optimising against it would mean tuning for the wrong operating point.

So treat 225 W as a fixed condition of the benchmarks rather than a tuning knob. If you run uncapped, your
numbers should be better than these, and they will not be comparable to them.

**From 2026-10-03 (v0.2.4) the cards run at 210 W with a -42 mV GPU voltage offset**, set through LACT. The offset
alone gained about 5% decode at the same cap (the card sustains higher clocks under it); the lower cap costs 0.7%
decode, 1.4% at 8 concurrent and 2.2% on an 8k prefill against 225 W. The benchmark tables above are the 225 W
numbers and stay the reference; PROGRESS.md marks which later measurements are at the new setting.

## Quick start

You need the models on disk and a ROCm 10 container. Then:

```bash
serve/27b.sh                      # Qwen3.8-27B-NVFP4 on 2 GPUs
serve/flashnext.sh                # Qwen3.8-Flash-Next on 2 GPUs, experts in host RAM (GPUS=0,2 on a 4-card box)
GPUS=0,1,2,3 TP=4 OFFLOAD_GB=0 serve/flashnext.sh   # Flash-Next on 4 GPUs, everything in VRAM (the headline)
```

Both are thin wrappers over `serve/serve.sh` and every tuning knob in them is commented with what it was measured
to be worth. `DRYRUN=1` prints the docker command without starting anything.

An OpenAI-compatible endpoint comes up on `:8080`. After upgrading, rebuild the kernel library
(`kernels/build.sh`): the plugin refuses a `libr9k.so` from before v0.2.2. On a first launch of a new configuration,
restart once (the launch that compiles gets a smaller KV pool).

## What's in it

- **`kernels/`** — HIP kernels for gfx1201: MXFP4×FP8 MoE GEMM (decode, prefill and fragment-tiled prefill
  variants), paged attention, sparse-attention (QSA) scoring and attention, 2-rank and compressed 4-rank
  peer-to-peer all-reduce, fp8 GEMM, int6 embedding gather, and the decode fusions: Gated DeltaNet
  speculative-decode core, router GEMM, indexer norm+rope, hyper-connection mix, shared expert.
- **`r9700_vllm/`** — the plugin. Registers through vLLM's official extension points (quantization config,
  model registry, platform plugin, attention backend, custom ops, pluggable layers); the few runtime hooks
  beyond those are version-gated and listed in [notes/independence.md](notes/independence.md).
- **`serve/`** — launchers, with measured knobs.
- **`tests/`** — correctness gates. Each kernel is checked against a reference implementation, and several are
  checked to be *bit-identical* to the path they replace.
- **`bench/`** — the serving-level checks: quality evals (`eval.py`), the strict overload sanity
  (`sanity_stress.py`), the mixed-length soak (`soak.py`), throughput by workload mix and by kind of prompt.
- **`tuning/`** — the benchmark harnesses used to pick tile configurations.
- **`host/`** — Proxmox host setup: 32 GB BARs for cards behind the PLX switches, including a second card on the
  same switch (DKMS kernel module + barfix script + VM hookscript).
- **`notes/`** — design notes and investigation write-ups.

## Docs

| file | what it's for |
|---|---|
| [CHANGELOG.md](CHANGELOG.md) | What changed in each release. |
| [notes/release-v0.2.5.md](notes/release-v0.2.5.md) | The current release: v0.2.4 plus its validation record and the version string. |
| [notes/release-v0.2.4.md](notes/release-v0.2.4.md) | v0.2.4: one GDN state page per request, the defect found on the way, and exactly what was checked. |
| [notes/release-v0.2.3.md](notes/release-v0.2.3.md) | v0.2.3: the first-token fix, the memory change, and exactly what was checked. |
| [notes/release-v0.2.2.md](notes/release-v0.2.2.md) | v0.2.2: the three real-traffic bugs it fixes, how they were found, and its validation. |
| [PROGRESS.md](PROGRESS.md) | The full engineering log: every change, what it measured, and what was tried and rejected. |
| [notes/picking-up.md](notes/picking-up.md) | **Start here if you're returning to this after a break.** Current state, open threads, how to run things. |
| [notes/radiance-test-run.md](notes/radiance-test-run.md) | A test run of radiance (a C++/HIP LLM server with its own kernels) on two R9700 through our release checklist, with our numbers beside it. |
| [notes/mes-timeouts.md](notes/mes-timeouts.md) | RDNA4 MES timeouts under KVM passthrough: the symptom, the host crash it can cause, what we ruled out, and the two kernel parameters that stop it. |
| [host/README.md](host/README.md) | Host PCIe setup: why a second card on one PLX switch gets no BAR, and the fix. |
| [notes/independence.md](notes/independence.md) | Which third-party pieces were replaced with our own, and why. |
| [notes/replacement-plan.md](notes/replacement-plan.md) | The plan for replacing what is left, so the repo is cleanly Apache-2.0. |
| [CREDITS.md](CREDITS.md) | The two third-party components, and the licence position. |

## Licence

[Apache-2.0](LICENSE), with one carve-out listed in [NOTICE](NOTICE) — read that file before reusing anything.

The short version: the vendored LRU cache kernels are Apache-2.0 too and pass on normally. Nothing loads libr4d
at runtime any more, and the GEMM constants that were once listed as derived from it turn out to be generated
from the OCP format specs — `tools/gen_kmag.py --check` proves it. The one remaining carve-out is the chat
template, copied from [vllm-mxfp4](https://github.com/GGZ14/vllm-mxfp4) with its author's permission. Both authors gave permission for **this** project; neither
upstream has a licence file, so that permission is not ours to pass on. Those parts are not under Apache-2.0 —
if you want to reuse them, ask their authors.
