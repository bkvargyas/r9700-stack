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

## Re-test on 1.0.8 (2026-10-05)

Brian asked for the 1.0.8 image (built 2026-10-04 20:03 UTC; the diff from 1.0.3 is four-card Flash-Next support, a
batch-size-4 paged-attention kernel, tests and docs; the two-card flags are unchanged). Same VM, same cards at 210 W and
-42 mV, same clients. Raw numbers only on his instruction: the 27B's BetterBench and soak were cut short and
Flash-Next skipped the evals.

| 27B, two cards | 1.0.8 | 1.0.4 |
|---|--:|--:|
| start to healthy | 42 s | 38 s |
| GSM8K 1,319, no thinking, conc 1 | 95.60% (1,261) | 95.60% (1,261) |
| HumanEval 164 (runs concurrently, thinking on) | 96.34% (158) | 97.56% (160) |
| probe: decode / 8 conc / 16 conc / 8k prefill, tok/s | 133 / 412 / 609 / 3,884 | 135 / 405 / 622 / 3,861 |
| strict sanity, conc 33 x 100 and conc 64 x 60 | 0 bad of 7,140 | 0 bad of 7,140 |

| Flash-Next, two cards | 1.0.8 | 1.0.4 |
|---|--:|--:|
| probe on a fresh server: decode / 8 conc / 16 conc / 8k prefill, tok/s | 56 / 191 / 258 / 5,670 | 54 / 167 / 236 / 5,653 |
| strict sanity, conc 1 | 0 bad of 60 | 0 bad of 40 |
| strict sanity, conc 9 | 6 bad of 270 | 18 bad of 900 |
| strict sanity, conc 16 | 2 bad of 320, and one run of 0 bad of 960 | 21 bad of 960 |
| 4 clients, 6-10k-token prompts, 4 min | 12 served, 418 prompt tok/s | 13-14 served, 400-430 prompt tok/s |
| 16-client soak, 200-30k tokens | 26 served in 759 s, 577 prompt tok/s, median latency 427 s | 72 served in 1,500 s, 565 prompt tok/s, 359 s |

So 1.0.8 changes nothing we measured on two cards. The 27B repeats to the question (HumanEval moved two problems, which
is what that concurrent, thinking-on test does run to run). Flash-Next still answers wrongly in a batch at the same
2% rate and in the same two shapes (an explanation at temperature 0 with thinking off; another row's or a nonsense
number: 64, 155, 12), with one clean run of 960 among the batched checks, and its prefill under concurrent long prompts
is the same ~420 tok/s. Two cautions on the raw run's own numbers: its first conc-9 sanity check produced an unreadable
log and no summary line, so it is not counted (the rerun with logs kept is what the table shows), and a probe repeated
against a server that has seen the same prompts reports 52,568 tok/s "prefill" and 189 tok/s decode, both prefix-cache
effects, not performance.

## 1.1.1 on two, three and four cards (2026-10-06)

Brian asked for the latest release, tested for accuracy, and whether TP=3 could run. 1.1.1 was published 2026-10-06
12:50 UTC, five releases past 1.0.8. The 1.0.8 -> 1.1.1 diff, by its commit titles: three-rank serving (`docs/TP3.md`:
"any world size", attention served on the largest multiple of the KV head count under the world and the ranks past it
attention-zero; the delta-net heads split unevenly, 5/5/6, nothing padded), three- and four-rank all-reduce and
all-gather kernels, a fused router GEMM + top-k + scatter launch, a second prefill form for `hc_read`, a decode form of
the quantised grouped GEMM with two K blocks a pass, `--embedding-placement` and `--ngram-placement`, a request log,
a Grafana dashboard, and `--expert-vs-cache-ratio` removed (a compose file that still passes it does not start).
Only Flash-Next serves at three ranks: the qwen4exp plugin asks for the attention world; the 27B's plugin does not,
and `docs/ARCHITECTURES.md` says a world that neither divides nor is a multiple of the KV head count is refused.

Two cards: VM 102 as before (210 W, -42 mV, the same compose files less the removed flag). Three and four cards: the
production host has exactly two R9700s, so these ran on the five-card test box (VM 100 on .100, same caps), HIP
devices 0,1,2 for three (the switch-local pair 03/04:00.0 plus 07:00.0 across the root complex) and 0-3 for four
(both switch-local pairs); the passively cooled fifth card stayed out. Flags from the README's three- and four-rank
rows: exact wire, `--host-pool-mib` 8192 / 12288, otherwise the two-card recipe (which runs the lossy wht6 wire); the
disk prefix tier was 16 GiB there for lack of disk. Clients on the test box in every case. Accuracy was the ask, so
no BetterBench and no long soak: strict sanity alone and under overload with every log kept, GSM8K 1,319 at conc 1
without thinking, HumanEval 164, then for Flash-Next the 4-client long-prompt prefill that collapsed on 1.0.x.

| 27B, two cards | 1.1.1 | 1.0.8 | 1.0.4 |
|---|--:|--:|--:|
| start to healthy | 58 s | 42 s | 38 s |
| KV pool, fp8 | 584,400 tokens | 548,816 | 548,816 |
| GSM8K 1,319, no thinking, conc 1 | 95.60% (1,261), output identical to 1.0.8 and 1.0.4 on 100% of questions | 95.60% | 95.60% |
| HumanEval 164 | 97.56% (160) | 96.34% (158) | 97.56% (160) |
| probe, fresh: decode / 8 conc / 16 conc / 8k prefill, tok/s | 156 / 466 / 725 / 3,933 | 133 / 412 / 609 / 3,884 | 135 / 405 / 622 / 3,861 |
| strict sanity, conc 1 x 40, 33 x 100, 64 x 60, then 33 x 40 after the evals | 0 bad of 8,540 | 0 of 7,140 | 0 of 7,140 |

| Flash-Next, 1.1.1 | two cards (VM 102) | three cards (test box) | four cards (test box) |
|---|--:|--:|--:|
| start to healthy | 222 s | 85 s | 120 s |
| VRAM a card at rest | 32.5 / 32.5 GB | 32.6 / 32.6 / 23.4 GB | 31.6 GB x 4 |
| experts resident | elastic 26.2 GiB, host pool 12 GiB | 15.8 GiB a card, host pool 8 GiB | all of them ("the whole expert plane is resident"), host pool 12 GiB unused |
| GSM8K 1,319, no thinking, conc 1 | 95.83% (1,264) | 95.91% (1,265) | 95.83% (1,264) |
| paired vs two cards (McNemar) | - | p = 1.00, 72% of outputs differ | p = 1.00, 73% of outputs differ |
| HumanEval 164 | 97.56% (160) | 96.34% (158) | 96.95% (159) |
| strict sanity, conc 1 x 40, 9 x 100, 16 x 60, 9 x 40 after | 0 bad of 2,260 | 0 bad of 2,260 | 0 bad of 2,260 |
| probe, fresh: decode / 8 conc / 16 conc / 8k prefill, tok/s | 193 / 625 / 652 / 5,990 | 205 / 568 / 600 / 5,053 | 263 / 710 / 722 / 5,191 |
| 4 clients, 6-10k-token prompts, 4 min | 112 served, 3,836 prompt tok/s, median latency 8.7 s | 109 served, 3,153 tok/s, 8.8 s | 118 served, 3,365 tok/s, 8.3 s |
| MES timeouts, E-lines | 0, 0 | 0, 0 | 0, 0 |

For reference, two cards on 1.0.8: probe 56 / 191 / 258 / 5,670, strict sanity 6 of 270 at conc 9 and 2 of 320 at
conc 16, the long-prompt test 12 served at 418 prompt tok/s.

So 1.1.1 is the release that fixed Flash-Next on two cards. The batched wrong answers are gone (0 of 4,520 across
the three Flash-Next configurations, where 1.0.4 and 1.0.8 failed about 2% of every batch), the concurrent
long-prompt prefill went from ~420 to 3,836 prompt tok/s, and the fresh-server decode from 56 to 193 tok/s, with
GSM8K unchanged and the two-card output identical to the 1.0.4 run on 99.8% of questions. The 27B repeats to the
question and gained 10-20% on the probe. Nothing in the 1.0.8 -> 1.1.1 commit titles names the batch bug; the
candidates are the router fusion, the new decode form of the grouped GEMM and the KV changes that came with the
three-rank work, and we test radiance as a black box, so this stays an observation.

On cards: the third buys about 6% single-stream decode and nothing else (lower aggregate, lower prefill, the exact
wire against wht6 and a root-complex hop against the pair are both in that number); the fourth makes every expert
resident and is the fastest configuration we have measured on this model, 263 tok/s single-stream and 722 at 16 streams, with the long-prompt test between the other two (118 served at 3,365 prompt tok/s), at the same accuracy (GSM8K 95.83%, p = 1.00 against two and three ranks). Three ranks change the
summation order, so 72% of GSM8K outputs differ from two ranks, with no accuracy effect.

Six cards, later the same day: `--tp 6` on this container is refused at declare ("an expert is 5 blocks of 128, which
6 ranks cannot each take a whole block of in both parities; lower the rank count, or serve whole experts"), so two,
three, four and (per the docs) eight ranks are the widths this container serves.

Mistakes this run: stripping the removed flag with a `sed` that also ate the line's indentation broke the three
Flash-Next compose files on VM 102 (go-yaml "did not find expected key"); the two-card Flash-Next phase was
relaunched after the fix, which is why its results are in `radtest12.log` rather than `radtest10.log`.

Where things are: the Flash-Next container (122 GB, sha256-verified copy of the host's) is at `~/rad` on the test
box with `~/radiance-compose-tp3/flashnext-tp{3,4}.yaml` (port 8010, state `~/radiance-state`); the 1.1.1 image is
on both VMs; GSM8K outputs `~/.r9keval/radiance111-{27b,fn,fn-tp3,fn-tp4}.json` and sanity logs `~/san111-*.log` on
the test box; run logs `radtest1{0,1,2,3}.log` in the job directory on the mgmt VM. VM 102 is still up with the cards.

## 1.2.0: Flash-Next TP4 and the 27B TP2 side by side on six cards (2026-10-06)

Brian: "run flash next TP4 and the 27B side by side; pull the latest update". 1.2.0 (published 19:42 UTC, one squashed
commit) adds a Qwen3.8-27B container in AMD's Quark AWQ-MXFP4 with vision and a `--p2p auto|on|off` flag; nothing our
compose files pass was removed. The test box now has six cards on three switch-local PLX pairs (`host/README.md`), so
Flash-Next ran at `--tp 4` on HIP 0-3 (chains A and B) and the 27B FP8 at `--tp 2` on HIP 4,5 (chain C, the new pair)
at the same time, two compose projects on ports 8010 and 8011, both at 210 W / -42 mV, clients on the same VM. The 27B
container was copied over for this (sha256-verified).

| | Flash-Next TP4 (HIP 0-3) | 27B TP2 (HIP 4,5) |
|---|--:|--:|
| start to healthy, both loading from one disk at once | 527 s | 527 s |
| VRAM a card | 31.5 GB | 32.5 GB |
| probe ALONE: decode / 8 conc / 16 conc / 8k prefill, tok/s | 264 / 740 / 786 / 5,197 | 147 / 430 / 657 / 3,405 |
| probe TOGETHER (other server probed at the same moment) | 263 / 742 / 725 / cache hit | 152 / 432 / 656 / cache hit |
| strict sanity, both servers loaded at once | 0 bad of 1,900 (c1/9/16) + 0 of 360 after | 0 bad of 7,180 (c1/33/64) + 0 of 1,320 after |
| GSM8K 1,319, no thinking, conc 1, both at once | 95.83% (1,264), output identical to 1.1.1 TP4 on 100% | 95.60% (1,261), identical to 1.1.1 on VM 102 on 100% |
| HumanEval 164, both at once | 96.34% (158) | 97.56% (160) |
| mixed load, 4 min, both at once | 4 clients 6-10k prompts: 106 served, 3,579 prompt tok/s, 9.1 s median | 16 clients 200-30k: 47 served, 2,558 prompt tok/s, 91 s median |
| hottest card during the mixed load | 61 C (fans ~1.1-1.3k rpm) | 67 C (fans 2.5-2.8k rpm) |
| MES timeouts / E-lines / IOMMU+AER faults | 0 / 0 / 0 | 0 / 0 / 0 |

Host draw by the BMC during the mixed load: 783 to 1,804 W (idle floor ~400 W), so six cards at the 210 W cap want
1.8 kW at the wall, not the 1.5 kW I had estimated.

The two servers do not see each other: decode and aggregate numbers alone and together agree within noise (the only
move is Flash-Next's 16-stream aggregate, 786 -> 725, while the 27B's probe ran), and every accuracy number repeats
to the question. The 27B on the PLX pair is a few percent under the same model on VM 102 (147 / 430 / 657 / 3,405
against 156 / 466 / 725 / 3,933): the PEX 8747 links each card at Gen3 x16, the production box has Gen5 root ports.
The chain C pair runs 15 C hotter than the other four under the same cap, with its fans at twice the speed; the
hotter of the two is the card that was passively cooled until today. Total 57 minutes. Logs `radtest14.log` and the
samplers `sbs-{temps,power}.log` in the job directory; evals `radiance120-{fn-tp4,27b-tp2}` on the test box.

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

## 1.3.0: Flash-Next TP4 and the 27B TP2 through the full checklist, against our v0.3.0 (2026-10-09)

Brian: "pull the latest radiance and test against our current build". 1.3.0 (published 2026-10-08, five releases
past 1.2.0: an adaptive draft window, a new MoE decode GEMM form, opt-in MoE prefill forms, static YaRN, a
three-rank wht6 wire, HF repo ids) on the test box, same compose files (`~/radiance-compose-tp3`, image in `.env`),
sequentially: Flash-Next TP4 on HIP 0-3 (:8010), then the 27B TP2 on HIP 4,5 (:8011), 210 W / -42 mV, our
clients from the v0.3.0 tree (`~/radtest130.sh` -> `~/radtest130.log`). Our column is the v0.3.0 record (BetterBench
on the same box and prompts the night before).

| | Flash-Next TP4: radiance 1.3.0 | ours v0.3.0 | 27B TP2: radiance 1.3.0 | ours v0.3.0 |
|---|--:|--:|--:|--:|
| probe: decode / 8 conc / 16 conc / 8k prefill | 265 / 798 / 836 / 5,367 (1.2.0: 264 / 740 / 786 / 5,197) | 199 / 560-613 / 888-905 / 6,850-7,070 | 143 / 454 / 668 / 3,394 (1.2.0: 147 / 430 / 657 / 3,405) | 213 / 578 / 672 / 3,951 |
| strict sanity (long prompts; conc N+1; conc 2N) | 0 bad of 160 / 900 / 960, 0 of 360 after the soak | 0 of 5,380 | 0 of 160 / 1,700 / 1,920, 0 of 680 after | 0 of 2,900 |
| GSM8K 1,319, no thinking, conc 1 | 95.83% (1,264) = 1.1.1 / 1.2.0 | 95.45-95.60% | 95.91% (1,265) | 94.69% (v0.2.4 record) |
| HumanEval 164 | 158 | 160 | 159 | 160 |
| BetterBench combined decode / update p99 | **221.9** / 12.9 ms | 160.6-162.3 / 17.1 ms | 130.7 / 31.8 ms | ~200 (2026-10-02: 197.5 single-stream) |
| concurrency 1 / 2 / 4 / 8 (aggregate tok/s) | **196 / 309 / 402 / 537** | 152-155 / 234-238 / 349-352 / 487-508 | 121 / 219 / 342 / 475 | 174 / 280 / 413 / 519 |
| prefill 2k / 8k / 16k / 32k / 64k (tok/s) | 4,870 / 5,270 / 5,224 / 5,281 / 5,291 | 5,101-5,178 / **7,129-7,182** / **7,127** / **7,245** / (32k max) | 3,025 / 3,332 / 3,317 / 3,256 / 3,055 | **4,190 / 4,191 / 4,072 / 3,837** |
| soak, 16 clients, 25 min | 456 ok, 0 errors, 3,994 prompt tok/s | 628 ok, 0 errors, 6,722 | 262 ok, 0 errors, 2,567 | 314 ok, 0 errors, 3,139 |

Reading it: on four cards radiance's decode step is 12.1 ms against our 16.4 with the same draft acceptance,
so it decodes 37% faster single-stream and leads at every concurrency, while our prefill is 35% faster from 8k
up and radiance runs 64k contexts here. On the 27B it is the other way round everywhere: our decode is ~50%
faster, our prefill 25-40%. Quality is equal within noise on both. The concurrency sanity failures of 1.1.1 are
gone in 1.3.0 as they were in 1.2.0. The "server error lines" counts in the log are the word "default" matching
`fault`; the logs have no errors. Evals `~/.r9keval/radiance130-{fn-tp4,27b-tp2}.json`, BetterBench logs
`~/rad130-*.bb.log` on the test box.
