# Picking this up again

Written 2026-09-21 for whoever comes back to this cold — probably us in a few months. It answers: what state is
it in, where does the hardware live, how do I run things, what is actually left to do, and what has already been
tried and failed so you don't repeat it.

## The one-paragraph summary

A kernel library plus vLLM plugin that runs Qwen3.8-27B-NVFP4 and Qwen3.8-Flash-Next on 2× R9700 using **stock**
vLLM + ROCm 10. It reaches roughly 93% of the fastest known alternative stack on decode and 85-91% on prefill, on
the same box and checkpoint, with equal quality. All the hard parts are HIP kernels in `kernels/` and the
registration glue in `r9700_vllm/`. Everything that mattered is in `PROGRESS.md`; this file is the orientation.

## Hardware and where things live

- **Dev/control box (mgmt VM)** — where this repo lives and where builds are driven from. No GPUs.
- **Test box: VM100, `192.168.0.123`** — 2× Radeon AI PRO R9700, 225 W cap. A copy of the repo is at
  `~/r9700-build/repo`, synced by `tar` over ssh (it is **not** a git repo there). Models in `~/models`.
- **VM100 needs `OVERLAYS=emulated-switch`.** The host sits behind a PLX PCIe switch that rejects hostcall
  kernels, so serving requires a hostcall-free RCCL and patched vLLM extensions, which that overlay mounts.
  **Without it every RCCL collective fails at launch** and it looks exactly like broken hardware. See
  `notes/independence.md` and the measurement-discipline section of `PROGRESS.md`. `serve/27b.sh` and
  `serve/flashnext.sh` now default it on.

## Running things

```bash
# serve (from the repo, on the test box)
serve/27b.sh                 # or flashnext.sh; DRYRUN=1 to print the docker command only
OVERLAYS= serve/27b.sh       # on a host that does NOT need the overlay

# correctness gates — run these before trusting any kernel change
TESTS="test_moe_mxfp4.py test_attn_r9k.py ..." ~/gates.sh     # on VM100; single GPU
~/artest.sh                                                    # the 2-rank all-reduce tests, needs both GPUs
```

Helper scripts on VM100 (`~/gates.sh`, `~/artest.sh`, `~/steptime.sh`, `~/pfprof.sh`, `~/batch.sh`) are outside
the repo. They are small; recreate from the invocations in `PROGRESS.md` if the box is ever rebuilt.

## Measuring anything — read this before you believe a number

These cost real time to learn:

- **Check `results.json` `env.endpoint`** before quoting a baseline. One box is production and one is the test
  box; they are not comparable.
- **Never run two benchmarks at once.** Two servers on the same GPUs produced 30% CVs and nonsense.
- **GSM8K-500 cannot separate these configs.** Six runs scatter 95.6-97.4% with no consistent ordering. It can
  catch a *broken* build, not a 1-2% regression. This is an open weakness — several lossy changes (6-bit
  all-reduce compression, fp8 KV, MXFP4 conversion, folded MXFP4) are each individually "gated" by an eval that
  cannot see them, and their *combined* effect has never been measured.
- **BetterBench's combined decode number moves with speculative acceptance**, which is sampling-noisy (per
  category CV 15-33%). Two unpaired single runs cannot resolve 3%. Compare `update p50` (step time) if you want
  to know whether a *kernel* changed. Use its paired `ab` mode for small differences.
- **Test numeric changes on LONG generations, not short answers.** Intuition says a short answer is a cleaner
  test; it is the opposite. A per-call perturbation needs room to compound, so short answers are the most
  forgiving condition and a null there proves little. Measured 2026-09-22 on 4-bit all-reduce: 3.4% of short
  answers changed outcome vs 1.1% under chain-of-thought -- long reasoning *absorbs* numeric error because later
  steps correct earlier ones. Use `EVAL_THINK=1`.
- **Cold single-shot kernel timings overstate by ~15%** — the card sits at 2.2-2.4 GHz under sustained load.
  Use `tuning/prefill_ab.py` (warmed, interleaved).
- **Gates that call the inner op miss served-path breakage.** Exercise the public wrappers in `r9700_vllm/ops.py`.

## Current state

Working and default: our own paged attention; A-tiled (fragment-tiled activation) prefill GEMM; folded-exponent
MXFP4; DFlash2 speculative decoding; NVFP4→MXFP4 conversion at load; GDN `in_proj` merge; expert LRU cache.

**The default configuration no longer uses libr4d at all** (2026-09-22): attention and the 2-rank all-reduce are
both ours. Verified by moving `r4d.so` aside and serving without it. Costs ~6.5% prefill against libr4d
(4,124 vs 4,413 @9k); decode is slightly *better* (117.2 vs 114.7 single, 25.4 vs 25.0 ms/step) and conc-8 is
~3% behind. `R9K_PAGED_ATTN=r4d` / `R9K_AR_IMPL=r4d` still select libr4d's for A/B. The only libr4d exposure
left anywhere is the **derived GEMM source** -- licensing, not runtime. See `notes/replacement-plan.md` Phase 2.

Superseded note: our own all-reduce (`R9K_AR_IMPL=r9k`) — matches libr4d on decode but is **~11% WORSE on prefill**
(3,909 vs 4,413 tok/s at 9k) and ~11% worse at conc-8, so libr4d's is still the default. Turning it on makes the
build fully independent of libr4d at runtime, at that cost. **Decided 2026-09-21: keep the defaults as they are**
— our attention (free, 99.8%) on, our all-reduce off. For context, dropping libr4d with no replacement at all
costs −55% prefill, so the work took "unusable without it" down to "11% behind", not to parity.

`notes/replacement-plan.md` is the plan for removing the remaining non-permissive dependencies entirely.

## What's actually left, roughly in order

1. **An eval that can detect regressions.** See the GSM8K point above. Until this exists, quality claims here are
   weaker than they look.
2. **All-reduce per-call overhead.** Worth the ~11% that full independence costs. Our compressed path runs at
   ~0.22 µs/KB against 0.115 for our exact path, so it is per-call overhead, not the link. **Profile where the
   ~87 µs go at 2 MB before writing kernels** — one attempt was already wasted guessing.
3. ~~**Power cap 225 W → 300 W.**~~ **Closed 2026-09-22 — do not re-propose.** Brian's reasoning, which is
   better than the perf argument for it: every libr4d/reference measurement we have was taken at the 225 W cap,
   so raising ours would make the whole comparison apples-to-oranges — the same mistake as the `bb-prod.log`
   baseline mix-up. He also intends to run capped long term, so a number measured uncapped is one he would never
   see in production. The cap is a fixed condition of this project, not a tuning knob.
4. **Decode-band attention kernel.** Short-q groups in mixed batches currently use the prefill kernel. Costs
   nothing measurable today; would matter at higher concurrency.
5. **GEMM independence** (~5%) — only if the licence situation changes. See `notes/independence.md`.
6. **Fuse our own prefill kernel launches** (~650 of ~1,950; maybe 8-15 ms of TTFT). See the TTFT section below
   for why that is the only lever left there, and why the rest is not ours to fix.

## TTFT: investigated 2026-09-21, localised, and closed

We were 101 ms vs the reference stack's 65 ms on time-to-first-token, consistent across every BetterBench
category. Chased it properly; here is the whole answer so nobody starts over.

**It decomposes as ~80 ms fixed + 0-25 ms waiting for the next engine step boundary.** Back-to-back requests
reliably land just after a step starts and pay nearly a full step (p50 103.7 ms, and pathologically tight --
194/200 samples inside one 5 ms bucket). Insert a random idle gap before each request and it drops to ~82 ms with
a 80-105 ms spread, which is exactly our 25 ms step time. Sequential benchmarks therefore see the worst case;
concurrent users see ~82 ms.

**The ~80 ms is prefill, and prefill is eager.** vLLM's own histograms: queue time **0.0 ms**, prefill time
**~78 ms** for an *8-token* prompt, and its TTFT matches a client stopwatch to within 1-2 ms (so it is not HTTP,
streaming or detokenisation). Profiling a 35-token prefill: **GPU busy 32.2 ms against ~78 ms wall clock**, with
about **1,950 kernel launches** in one forward -- ~24 us of dead time each. Over half the GPU time is our MoE
kernel running in *decode* mode (`r9k_moe_mxfp4a8_kernel`, correct: M is far below the `PREFILL_MIN_M=128` gate),
i.e. mostly irreducible expert-weight traffic that does not shrink just because the prompt did.

**Nothing at the configuration level fixes it.** Ruled out by measurement: speculative decoding (it *helps* TTFT
by 37 ms -- running without a drafter is worse), `qwen-fixed` chat-template rendering, `HSA_ENABLE_MWAITX`,
`GPU_MAX_HW_QUEUES`, and the API-server layer. `cudagraph_mode` already defaults to `FULL_AND_PIECEWISE`, so the
1,950 launches are what remains *after* piecewise capture -- capture only covers shapes in the capture list,
which vLLM populates for decode, not for arbitrary prefill lengths. **Forcing `CGMODE=FULL` makes TTFT 55 ms
worse** (159 ms) because our custom attention backend cannot be fully captured and it falls back.

**Left open deliberately.** Eager prefill is stock vLLM behaviour and fixing it upstream means patching vLLM,
which is the fork-maintenance burden this project exists to avoid. The part that *is* ours: ~650 of the 1,950
launches are our kernels (256 MoE + 258 quantiser + 140 all-reduce), worth maybe 8-15 ms if fused. Modest, real,
and the only honest lever left.

## Already tried, measured, and rejected — don't redo these

- `CGMODE=FULL`: TTFT 159 ms vs 104. Custom attention backend cannot be fully captured.
- `HSA_ENABLE_MWAITX=0`, `HWQ=4`, dropping the chat template: none moved TTFT by more than ~1 ms.
- `NBT=8192` (bigger prefill chunks): no gain at 9k, engine crash at 20.7k.
- `R9K_FP8_TO_MXFP4=0` (keep MLP layers 56-63 on fp8): −21% prefill, −18% decode.
- Fusing the compressed all-reduce into one kernel: 314 µs vs 179 for the split version, at every block count.
- Reduce-scatter + all-gather for large messages: no byte win at **two** ranks, by analysis.
- bf16 LM head: −8% decode, no quality gain.
- Drafter W4: −7% acceptance. LDS-A staging on the decode kernel. NT loads at MT>1 (−24% QFN prefill).
- fp8→MXFP4 conversion on Flash-Next (helps the 27B, not QFN).

`PROGRESS.md` has the numbers behind each.

## Gotchas that will bite you

- Our ctypes kernels must be **torch custom ops** (`r9700_vllm/ops.py`) to survive inside compiled graphs.
- The **torch.compile cache is keyed by a hash of the plugin source** (`CKEY` in `serve/serve.sh`). Without that,
  a stale graph silently keeps calling the old kernel path and you measure nothing.
- vLLM's memory profile **underestimates when load-time requantisation is on** — hence the fixed `KVMEM=` budget.
- `DRYRUN=1` does **not** source overlays, so a launcher can pass a dry run and still fail to serve.
- Kill background benchmark scripts with care: several have `trap ... EXIT` handlers that remove the running
  container, so killing the script also destroys the logs you wanted.

## Deferred / next up (2026-09-22)

### vLLM upgrade — deferred, two blockers to clear first

Brian asked to move to a newer vLLM and re-verify the plugin (the project's whole claim is that this is a
no-op). Held off because two things need deciding first, and both are worse to discover mid-pull:

1. **Disk.** `/dev/sda1` is 93% full — 42 GB free, and the ROCm vLLM images are ~52 GB each. Docker reports
   26 GB reclaimable plus 16 GB of build cache, which is enough if freed, but beyond that the only candidates
   are reference images we should keep: `stilldeadcode/vllm-radiance:0.9.3` (13.7 GB) is the stack every
   benchmark in the README is measured against, and deleting it costs us the ability to re-measure the baseline
   on the same box.
2. **The overlay is pinned to the current vLLM build.** `OVERLAYS=emulated-switch` mounts hostcall-patched
   copies of vLLM's own extensions -- `~/p2p-patched-nightly/_rocm_C.abi3.so` (998 MB) and
   `_C_stable_libtorch.abi3.so` (350 MB), built 2026-09-18 against *this* vLLM. A new vLLM ships new
   extensions and the old patched pair will not match. `overlay/emulated-switch/patch-hostcall.sh` regenerates
   them, so it is a documented step -- but if it fails on a newer vLLM, TP serving on this box will not start
   at all, and that failure looks exactly like dead hardware (see the OVERLAYS note above).

Sequence when it happens: reclaim cache + dangling layers, pull without deleting any reference image, build the
plugin, run the 12-gate suite, regenerate the overlay, then serve and re-benchmark.

### Three GPUs: TP=3 will NOT work for this model -- read before planning around it

Brian is adding a third R9700 behind the PLX switch. From `Qwen3.8-27B-NVFP4/config.json`:

    hidden_size 5120   num_attention_heads 24   num_key_value_heads 4
    intermediate_size 17408   num_hidden_layers 64   head_dim 256

TP=3 divisibility: attention heads 24/3 = 8 fine, but **KV heads 4/3, hidden 5120/3 and intermediate 17408/3 are
all non-integer**. vLLM requires the KV-head count to divide the TP size (or vice versa for replication); 4 and 3
satisfy neither. **TP=3 will be rejected.** TP=4 divides cleanly on all four (6 / 1 / 1280 / 4352), so the useful
progression for this model is 2 -> 4, not 2 -> 3.

So a third card is worth having as a **topology and bandwidth experiment** -- does a third PLX chain get full
H2D bandwidth, does its ReBAR hold -- which is exactly the groundwork for going to four. It is not a serving
configuration for the 27B. Options with three: TP=2 plus a spare card for a second model or dev work, or
pipeline parallel (PP=3 over 64 layers) which works but adds latency and does not help single-stream.

Also note: **our all-reduce is 2-rank only** (`R9kAllReduce` disables itself when world_size != 2, as does
libr4d's). Anything above TP=2 falls back to RCCL for collectives, which measured 30.3 ms/step against our
25.1 at TP=2. A TP=4 build would need an N-rank all-reduce written before it is competitive.

### Expect the third card to look faulty at first. It probably is not.

History on this box, from memory `reference_r9700_host_100.md`: a card was diagnosed **"FAULTY for passthrough
(key finding)"** after its Resizable BAR collapsed to 1 MB on every FLR. That conclusion was **corrected later
the same day**: the root cause was the **per-PLX-chain prefetch window**, not the card. The window trick had
only ever been applied to the first card's chain, so the second card's chain had a 258 MB window and its BAR was
squeezed on every realloc. Enlarging that chain's own window fixed it, and `/usr/local/sbin/r9700-barfix.sh`
now does both chains.

A third card on a third chain will hit the same thing: **its chain's prefetch window will not have been
enlarged, so its 32 GB BAR will not hold, and it will present as a broken card.** Extend `r9700-barfix.sh` to
the new chain before concluding anything about the hardware.

This is worth stating plainly because it is now a pattern rather than an anecdote. Twice on this box an apparent
hardware fault has turned out to be PLX/topology configuration -- the BAR case above, and on 2026-09-21 an RCCL
failure that looked exactly like a dead GPU (single-GPU compute fine, comm init fine, every collective failing)
which was a missing `OVERLAYS=emulated-switch`. On this machine, suspect the topology before the silicon.
