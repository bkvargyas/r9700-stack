# r9700-stack v0.2.3

Two changes. A short prompt's first token arrives in about half the time -- the 27B on two cards is now level with
the reference stack -- and Flash-Next on two cards gets 28% more KV cache. Decode, long-prompt prefill and memory
use are otherwise unchanged.

**What this release was checked with, and what it was not.** v0.2.2 set a checklist for every release (unit gates,
all-reduce suites, strict sanity under overload, mixed-length soaks, a full BetterBench, on every configuration).
v0.2.3 is tagged on part of it, by decision: the unit gates, strict sanity under overload on all three
configurations (0 bad answers of 3,850), and the two long soaks for the memory change. **Not run:** a quality
eval, a full BetterBench, the mixed soak on a restarted two-card launch. Short prompts now go through a numerically
different (more exact) path, so an eval is owed with the next release.

## 1. Time to first token: the GDN prefill core

| first token, ms (client median, one request at a time) | 45 tokens | 83 | 159 | 236 | 320 | 404 | 656 |
|---|--:|--:|--:|--:|--:|--:|--:|
| 27B, two cards: v0.2.2 | 97 | 96 | 147 | ~147 | 148 | 148 | 172 |
| 27B, two cards: **v0.2.3** | **47** | **60** | **81** | **92** | **128** | 142 | 172 |
| reference stack, same cards, same day | 46 | 68 | 93 | | 125 | | 158 |
| Flash-Next, four cards: v0.2.2 | 87 | 87 | 88 | 88 | 92 | 107 | 131 |
| Flash-Next, four cards: **v0.2.3** | **44** | **58** | **73** | 87 | 91 | 107 | 132 |

The 27B answered a 45-token prompt in 97 ms and the reference in 46, and the 2026-09-21 investigation had closed
that as upstream behaviour. The server's own counters said otherwise: zero queue time, one engine step per request,
and that step ~90 ms where a decode step is 23. A profile of the step with Python stacks found it: a prefill step
runs vLLM's chunked GDN core eagerly in every GDN layer -- six Triton kernels plus glue, about 30 launches and 1.2
ms of Python per layer *whatever the prompt length*, 48 layers on the 27B. That is the fixed ~60 ms, and it is also
what every running request waits through when a new one joins the batch.

Now, for a step with few prefill tokens, the recurrence runs token by token in one launch per layer
(`r9k_gdn_seq`: the arithmetic decode already uses, the state in registers, any number of tokens) after the causal
conv in one launch (`r9k_gdn_conv`, bit-identical to vLLM's Triton kernel -- which, it turned out, truncates its
bf16 tap products on this backend; found by enumerating the rounding choices against its output). Both kernels
address the batch's own rows, so a step that mixes prefills with running spec-decode sequences needs no gather.

Token by token is slower on the GPU than the chunked form (~1.8 us per token and layer), so it only pays while the
step is bound by launches rather than compute. `R9K_GDN_PREFILL_MAX` (prefill tokens per step) defaults to 256,
where Flash-Next on four cards breaks even; `serve/27b.sh` sets 400, the 27B breaking even near 440. Above the
threshold nothing changes. `R9K_GDN_PREFILL_MAX=0` restores vLLM's core; `R9K_GDN_PREFILL_CONV=stock` keeps its
conv. An older `libr9k.so` without the two new symbols keeps the chunked core (and the old first-token time).

Numerics: against an fp64 token-by-token reference the new core differs only by bf16 rounding flips (relative
error 1-6e-5); vLLM's chunked form, which does its solves on bf16 q/k, is at 5e-3 from the same reference. So the
outputs for short prompts are not bit-identical to v0.2.2 -- they are closer to the exact recurrence, and now and
then a token differs. `tests/test_gdn_prefill_r9k.py` holds all of this, including spec-decode sequences through
the new kernel being bit-identical to the decode kernel.

The investigation also measured and rejected two other candidates: prefill graphs up to 2,048 tokens (moved only the
158-319-token range on the 27B and cost 0.8 GiB of VRAM) and prefix caching off (no change in first-token time;
2.1 GiB less VRAM and 6% more KV on the 27B, noted for later). Flash-Next on two cards with offloaded experts:
52 ms at 44 tokens; above its graph limit it runs eagerly as before (177 ms at 82 tokens).

## 2. Flash-Next on two cards: `--gpu-memory-utilization` 0.94 -> 0.96

| | 0.94 | 0.96 |
|---|--:|--:|
| KV cache, first launch of a configuration | 95,429 tokens | 128,772 |
| KV cache, restarted launch (the normal way to run it) | 121,299 | 155,791 (+28%) |

v0.2.2 rejected 0.96 because it died with 69 MiB free when the request queue drained. That allocation was the
short-conv prefill batch fixed in the same release, and it had not been retested since. At 0.96 on this code: the
soak that killed it (8 clients, prompts of 18k-30k tokens, 16 minutes) served 86 with none failed; the mixed soak
(16 clients, 200-30k tokens, 26 minutes) served 227 with none failed; VRAM flat at 30,396 of 32,624 MiB; strict
sanity after 24k-token prefills 0 bad of 180. Those ran on a first launch, which has the smaller pool. On a
restarted launch the first 5.7 minutes of the mixed soak peaked at 31,350 MiB with no error, and the run was
stopped there, so that case has had a burst, not a soak. `serve/serve.sh` picks 0.96 only for TP2 with offloaded
experts; four cards and the 27B keep 0.94; `UTIL=0.94` restores the old value.

## Checked

- Unit gates: `test_gdn_prefill_r9k.py` (new), `test_gdn_decode_r9k.py`, `test_gdn_merge.py`, `test_moe_mxfp4.py`,
  `test_cache_moe.py`, `test_gemm_fp8.py`, `test_ple_conv.py`.
- Strict sanity with more requests than `max_num_seqs`, after short and after long prefills: 27B two cards 0 bad
  of 990; Flash-Next four cards 0 of 1,870; Flash-Next two cards 0 of 990. (Run at a 512-token threshold, which
  sends more steps down the new path than the shipped 256 / 400.)
- Concurrency probes, three per launch, Flash-Next four cards with and without the change: 884-887 against
  866-880 tok/s at 16 concurrent, i.e. unchanged.
- The two soaks and the sanity checks of section 2.
- The smoke test caught one bug before the tag: with speculative decoding the conv state is wider than the conv's
  three columns, the wrapper asserted exactly three, and the server died at warm-up. Fixed; the unit test has the
  case.

## Not checked

- No quality eval (GSM8K or HumanEval) on this code.
- No full BetterBench; the first-token numbers above are per-request client medians (`bench/ttft_breakdown.py`),
  not BetterBench's streaming p50.
- No full-length soak of a restarted two-card launch at 0.96.

## Upgrading

Rebuild `libr9k.so` (`kernels/build.sh`; `serve/serve.sh` does it when the kernels are newer than the library).
Nothing else changes: same models, same launch scripts, same knobs plus the two new ones above. On a first launch
of a configuration the plugin source hash changes the compile cache key, so the first launch compiles and keeps
~0.45 GiB; restart once, as before.

## Files

- `kernels/r9k_gdn.hip`: `r9k_gdn_seq`, `r9k_gdn_conv`; `r9700_vllm/models/gdn.py`: `seq_core`, `conv_prefill`,
  `_prefill`, the knobs.
- `tests/test_gdn_prefill_r9k.py`, `bench/ttft_breakdown.py`.
- `serve/serve.sh` (UTIL), `serve/27b.sh` (threshold 400), `serve/flashnext.sh` (notes).
- `PROGRESS.md` 2026-10-02: the investigation, the kernel, the threshold measurements, the 0.96 soaks.
