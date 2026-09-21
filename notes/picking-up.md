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
- **Cold single-shot kernel timings overstate by ~15%** — the card sits at 2.2-2.4 GHz under sustained load.
  Use `tuning/prefill_ab.py` (warmed, interleaved).
- **Gates that call the inner op miss served-path breakage.** Exercise the public wrappers in `r9700_vllm/ops.py`.

## Current state

Working and default: our own paged attention; A-tiled (fragment-tiled activation) prefill GEMM; folded-exponent
MXFP4; DFlash2 speculative decoding; NVFP4→MXFP4 conversion at load; GDN `in_proj` merge; expert LRU cache.

Optional: our own all-reduce (`R9K_AR_IMPL=r9k`) — matches libr4d on decode, ~11% behind on prefill, so libr4d's
is still the default. Turning it on makes the build fully independent of libr4d at runtime.

## What's actually left, roughly in order

1. **An eval that can detect regressions.** See the GSM8K point above. Until this exists, quality claims here are
   weaker than they look.
3. **All-reduce per-call overhead.** Worth the ~11% that full independence costs. Our compressed path runs at
   ~0.22 µs/KB against 0.115 for our exact path, so it is per-call overhead, not the link. **Profile where the
   ~87 µs go at 2 MB before writing kernels** — one attempt was already wasted guessing.
4. **Power cap 225 W → 300 W.** Never tested; prefill is clock-limited (2.40 GHz vs 2.82 on decode) so it could be
   the biggest single lever left. Needs Brian's say-so, it is his hardware.
5. **Decode-band attention kernel.** Short-q groups in mixed batches currently use the prefill kernel. Costs
   nothing measurable today; would matter at higher concurrency.
6. **GEMM independence** (~5%) — only if the licence situation changes. See `notes/independence.md`.

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
