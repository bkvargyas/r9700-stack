# radiance, test run on two R9700 (2026-10-04)

*What: [radiance](https://codeberg.org/StillDeadcode/radiance), an LLM server in C++ and HIP with its own
engine and kernel library (libr4d), run from its published Docker image on the two R9700 of our production host,
through the same checklist a r9700-stack release gets. Why: Brian asked for a test run; the numbers below are the
answer, with ours beside them where the configuration is comparable. We tested it as a black box: image, containers
and flags as published, our own clients, no reading of its kernel source.*

## Setup

- **Box**: a new VM (VM 102) on the production Proxmox host, with the two production cards passed through on the same
  emulated switch as the production VM, 32 cores, 96 GB. Debian 13 with the test box's kernel and firmware (XanMod
  7.2.6, firmware-amd-graphics 20260810, MES 0x91), `amdgpu.mes_log_enable=1 gpu_recovery=0`, both cards at 210 W and
  -42 mV through LACT, as our v0.2.4/v0.2.5 numbers were taken. The two R9700 sit on separate root ports here; our own
  numbers come from the test box, where card pairs share a PLX switch.
- **radiance**: image `stilldeadcode/radiance:latest` as of 2026-10-04 16:22 UTC, which is the **1.0.4** build (415 MB
  unpacked; the project cut 1.0.0 through 1.0.8 that same day), the compose files from the 1.0.3 checkout (identical to
  1.0.8's for two cards) with their published
  flags, the three containers from Hugging Face (`qwen3.8-27b-fp8.rad` 29.4 GB, `qwen3.8-next-flash-fp8-iq4r-moe.rad`
  113.6 GB, `minicpm5-2b-fp8.rad` 3.0 GB) on an ext4 disk. Served without `--api-key` so our clients work unchanged;
  radiance's prefix-cache disk tier for Flash-Next cut from 128 to 64 GiB to fit the disk.
- **Clients**: our bench tools (`bench/`: probe, sanity_stress, eval GSM8K, humaneval, soak, BetterBench) from the test
  box over the 10 GbE LAN. Every request field they send is accepted by radiance (it refuses unknown fields rather than
  ignoring them, and documents the list).
- **Ours**: r9700-stack v0.2.4/v0.2.5 on stock vLLM, from `notes/release-v0.2.4.md` (same day's power settings, same
  tests, same question sets).

## Qwen3.8-27B, two cards

radiance serves the published FP8 checkpoint with its DFlash2 drafter at depth 7; we serve the MXFP4 (4-bit) weights
with the same drafter at depth 7. Half the weight bytes per token is a real advantage in a memory-bound decode, so the
decode rows are not a like-for-like contest; the quality rows are the same questions on different quantisations.

| 27B, two cards, 210 W / -42 mV | radiance FP8 | ours v0.2.4 MXFP4 |
|---|--:|--:|
| time from start to healthy | 38 s | ~10 min (compile and graph capture) |
| KV cache, tokens | 548,816 | 367,494 |
| GSM8K full 1,319, no thinking, conc 1 | **95.60%** | 94.69% |
| HumanEval 164 | 97.56% (160) | 97.56% (160) |
| probe: decode mixed prompts / 8 conc / 16 conc / 8k prefill, tok/s | 135 / 405 / 622 / 3,861 | 212 / 579 / 673 / 3,940 |
| BetterBench combined decode, tok/s | 112.0 | 200.3 |
| BetterBench time to first token p50 | 73 ms | 71 ms |
| BetterBench concurrency 1 / 2 / 4 / 8, aggregate tok/s | 94.5 / 173.6 / 291.7 / 421.6 | 185 / 314 / 450 / 558 |
| BetterBench prefill 2k / 8k / 16k / 32k, tok/s | 3,341 / 3,772 / 3,794 / 3,696 | 3,860 / 4,076 / 4,035 / 3,825 |
| mixed soak, 16 clients, 200-30k tokens, 25 min | 308 served, 0 errors, 2,804 prompt tok/s, 52 output tok/s, median latency 80 s | 383, 0 errors, 3,093, 64, 64 s |
| strict sanity under overload | 0 bad of 3,300 (conc 33) and 0 of 3,840 (conc 64) | 0 of 900 and 0 of 960 |

Paired GSM8K (same questions): radiance right where ours was wrong on 23, the reverse on 11, McNemar p = 0.058, no
detectable difference; 0.8% of the outputs are byte-identical, as expected across quantisations. Drafter acceptance
reported by radiance: 0.61 to 0.65. VRAM stayed flat through the soak; no MES timeout in the guest.

Reading: correct, stable, a larger KV pool and a much faster start; decode a third slower at the same drafter, prefill
the same. Its single-stream rate on prose (65 tok/s in probe, 81 in BetterBench) is where the drafter accepts least.

## Qwen3.8-Flash-Next, two cards

radiance's container holds 4-bit codebook experts and an int8 trunk, placed `expert_tiered`: 21.2 GiB of experts per
card in VRAM and the rest streamed from a 12 GiB pinned host pool, MTP depth 3, and a lossy 6-bit all-reduce wire
(`--tp-wire wht6`) above 128 KiB a message. Ours streams 20 GB of experts per rank from host memory, MTP depth 3, with
our compressed all-reduce above 128 KB. Similar designs, so this comparison is closer.

| Flash-Next, two cards, 210 W / -42 mV | radiance | ours v0.2.4 (offload) |
|---|--:|--:|
| time from start to healthy | 82 s (container in the host page cache; a cold load reads 66 GiB at 1.0 GB/s) | ~8.5 min |
| VRAM per card | 4.2 GiB trunk + 21.2 GiB experts + 4.7 GiB KV | |
| KV cache, tokens | 489,544 | 183,202 |
| GSM8K full 1,319, no thinking, conc 1 | **95.83%** | 95.60% |
| HumanEval 164 | 98.17% (161) | 98.17% (161) |
| probe on a fresh server: decode / 8 conc / 16 conc / 8k prefill, tok/s | 54 / 167 / 236 / 5,653 | |
| BetterBench combined decode, tok/s | 72.6 | 97.3 |
| BetterBench time to first token p50 | **141 ms** | 463 ms |
| BetterBench concurrency 1 / 2 / 4 / 8, aggregate tok/s | 48.7 / 76.0 / 106.1 / **143.2** | 87.8 / 108.8 / 116.9 / 113.8 |
| BetterBench prefill 2k / 8k / 16k / 32k / 64k, tok/s | 468 / 838 / 1,251 / 1,426 / 1,534 | 2,149 / 3,511 / 3,834 / 3,784 / - |
| mixed soak, 16 clients, 200-30k tokens, 25 min | **72 served**, 0 errors, 565 prompt tok/s, 10.6 output tok/s, median latency 359 s | 239 served (14 over-length rejected by design), 0 errors, 2,394, 39, 104 s |
| strict sanity under overload | **18 bad of 900 (conc 9), 21 of 960 (conc 16)**; 0 of 40 at conc 1 | 0 of 900 and 0 of 960 |

Paired GSM8K: 16 vs 13 discordant, p = 0.71, no detectable difference. VRAM flat through the soak, no MES timeout.

Two problems, both only under concurrency, and we looked at each a little further (`radtest2-4` logs):

**1. Wrong answers in a batch.** The strict sanity check asks "What is N times 3? Reply with just the number" at
temperature 0 with thinking off. Alone, radiance answers every one of 40 correctly. With 9 or 16 in flight, 2-6% of
the replies are either a step-by-step explanation the single-stream server never produces ("1. **Identify the numbers
involved**...") or a wrong number -- sometimes another row's answer (153 for a row that asked 51 x 3's neighbour),
sometimes nothing in the set (10, 999, 1320, 12540). Speculation off makes it worse (11 of 180 and 9 of 160); the
exact all-reduce wire makes it better but not clean (1 of 180, 5 of 160, including a wrong number); a 24k-token prefill
before each round does not change it (3 of 90). So the lossy wire explains part of the batch-dependence and something
in the batched path explains the rest. Our bar for a release is zero of these; the one time our stack produced them
(2026-10-02) it was a kernel bug and we fixed it before tagging.

**2. Prefill collapses under concurrent long prompts.** A single 8k prompt prefills at 5.6k tok/s on a fresh server.
Four clients sending 6-10k-token prompts get 400-430 prompt tok/s in aggregate (13-14 requests served in four
minutes), with the prefix-cache tiers on or off, so it is not the disk tier (which, in the main run, had in fact failed
to start: `E cannot create /data/kvcache/flashnext`, because radiance does not create the parent directory; we created
it by hand for the follow-up and it then ran). BetterBench's sequential prefill sweep after hours of sessions reads
470-1,530 tok/s, and the 16-client soak served 72 requests where ours served 239. Something in how concurrent prefills
are scheduled against the expert stream is the likely place; we did not look further. A probe repeated after the
four-minute run reports 45,740 tok/s "prefill" with the tiers on -- that is a prefix-cache hit on probe's fixed prompt,
not prefill.

## Things worth copying

- The startup log states every budget before allocating (weights, experts, KV, host pool, headroom, what the elastic
  share divides) and the resulting KV pool in sessions. Our launcher prints the KV size after the fact.
- Model containers: tokeniser, template, drafter and weights in one file, and a 38-second start for the 27B against
  our ten minutes of compile and capture. Our first-launch compile cache is the same idea in a worse place.
- Unknown request fields are refused with the field named (`docs/GUIDE.md` §5.8); `reasoning_content` is separate from
  `content`; `/stats` exposes KV use, acceptance, step counts per request.
- A KV pool of 549k (27B) and 490k (Flash-Next) tokens on two cards with fp8 KV, against our 367k and 183k.

## Not done

- Nothing with thinking on except what BetterBench's defaults send; HumanEval runs concurrently, so radiance's
  Flash-Next number (161/164) includes whatever the concurrency problem does to code.
- MiniCPM5-2B only as a smoke test (answers correct, 641 tok/s single-stream, 15.5k tok/s 8k prefill).
- Vision, tools, constrained output, the dashboard, long context past 32k, `rad-convert`.

## Where things are

VM 102 "radiance" on the production host (stopped after the run; the production VM has its cards back): the compose
directory `~/radiance-compose` with `*-nokey.yaml`, `flashnext-nospec.yaml`, `flashnext-exact.yaml`,
`flashnext-notier.yaml`; the containers under `/srv/models/rad`; the repo clone `~/radiance`. Results: GSM8K in
`~/.r9keval/radiance-{27b,fn}.json` on the test box (compare with `bench/eval.py compare`), BetterBench logs
`~/radtest-{27b,fn}.bb.log` there, the run logs `radtest*.log` on the mgmt VM.
