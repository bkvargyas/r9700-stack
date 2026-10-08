# r9700-stack v0.3.0

**Flash-Next on four cards: 70% more KV cache, 4% more prefill, the same decode, on new measured defaults.**
The code of v0.2.5 plus the decode-step fusions of 2026-10-07 (the P2P all-gather on by default) and the
prefill round of the same day (`notes/prefill-fp8-pipe.md`, `notes/decode-nodes.md`).

## What changed for someone running it

- `serve/flashnext.sh` at TP4: the fp8 hyper-connection up GEMM with the gate mix fused (`R9K_HC_FP8=r9k`),
  memory utilization 0.98, cudagraphs only for prefill chunks up to 256 tokens. KV cache 279,564 -> 442-475k
  tokens (`KVMEM=7.0` pins 475k), 8k prefill 6,769 -> 6,852-7,069 tok/s, decode step 16.4 ms and 199 tok/s
  unchanged, 888-905 tok/s at 16 streams. `UTIL=0.94 CGSIZES=1,...,2048 R9K_HC_FP8=stock` restores v0.2.5.
- The one-shot P2P all-gather for decode-sized gathers at TP > 2 (`R9K_AG`, default on): step 16.70 -> 16.40 ms.
- New kernels in the tree, off or opt-in: fp8 prefill GEMMs on the tiled WMMA kernel (163-182 TFLOPS), an exact
  block-scaled fp8 variant (slower than Triton here), one-launch MoE routing and the fused top-k sum with the
  shared expert, prefill all-reduce pipelining by row parts (correct, a loss on this topology), `R9K_MEMSNAP`.

## Why the KV cache moved

vLLM sizes the KV cache from its profiling run's peak memory, and on this stack that peak was torch.compile and
inductor transients, not activations: 3.83 GiB against a measured 0.6 GiB model peak for a 4096-token chunk
(`compat/memsnap.py`). The graphs for 384-2048-token chunks cost another 1.4 GiB a card and 1.4 GiB of the
estimate for no soak-measured throughput. Utilization 0.98 then leaves about 1 GB at the soaked peak.

## Checked (on this code, 2026-10-07/08, 210 W and -42 mV, prefix caching off)

Flash-Next TP4 on the new defaults, 2026-10-07: GSM8K full set at concurrency 1, paired per question against
the v0.2.6-era record: 95.45% vs 95.60%, 14 vs 16 discordant, McNemar p = 0.86, no detectable difference;
HumanEval 160 / 164 (the record). Mixed-length soak, 16 clients, 25 minutes: 684 requests, 0 errors, 6,666
prompt tok/s, VRAM peak 31,766 of 32,624 MiB a card. Strict sanity: 0 bad of 320 after long prompts, 0 of 1,700
at concurrency 17, 0 of 1,920 at 32. The 2048-chunk, 7.5 GiB and pipelined variants ran the same checks and are
documented as rejected.

The release validation chain on the final tree (`~/validate-030.sh` -> `~/val030.log`): fn4 full BetterBench +
soak + sanity, 27B two cards soak + sanity, Flash-Next two cards soak + sanity, the 4-rank all-reduce /
all-gather / 2-rank race tests, the unit gates. A note on the record's provenance: the chain's first run
(23:24-00:28 UTC) and the 2026-10-07 "v0.2.6" validation launched the serve scripts without `REPO=`, so
`serve/serve.sh` mounted the production copy (`~/r9700-build/repo`, the v0.2.5 plugin) under the new serve
defaults -- the fp8 mix never ran there and that BetterBench (156.8 combined, 2k prefill 5,023) is v0.2.5 code
with the trimmed graphs. The run below mounts the release tree (checked on the container). Record:

| configuration | probe (dec tok/s / c8 / c16 / 8k prefill) | soak, 16 clients, 25 min | strict sanity | VRAM |
|---|---|---|---|---|
| Flash-Next TP4, new defaults (KV 442,575 tokens) | BetterBench: combined 160.6, step p50 16.4 ms, conc 1 / 8 = 154.9 / 486.7 tok/s, prefill 2k / 8k / 32k = 5,178 / 7,182 / 7,246 | 628 ok, 0 errors, 6,722 prompt tok/s | 0 bad of 320 (long prompts), 1,700 (conc 17), 1,440 (24), 1,920 (32) | 31,301 MiB of 32,624, flat |
| 27B two cards (KV 367,494) | 212.8 / 578 / 672 / 3,951 | 314 ok, 0 errors, 3,139 prompt tok/s | 0 bad of 320, 900, 720, 960 | 29.3 GB, flat |
| Flash-Next two cards, experts in host RAM (KV 133,306) | 118.7 / 183 / 150 / 2,608 | 225 ok, 0 errors, 2,339 prompt tok/s | 0 bad of 320, 900, 720, 960 | flat |

Against the previous record (v0.2.5 code, 2026-10-07 morning): the four-card BetterBench 162.8 -> 160.6 is within
the band (160.9 / 162.8 on identical code), 8k / 32k prefill +3.9% / +4.6%, 2k prefill -9% (the graph trim:
`CGSIZES=1,...,2048` restores 5,684 at the cost of ~1.4 GiB of KV cache a card); the 27B is unchanged; the
two-card Flash-Next probe sits in its link-bound band.

Tests on the release tree: the 4-rank all-gather (`test_ag_nrank`, incl. graph replay) PASS, the 2-rank race
(`test_ar_race`, 1,500 replays, exact and compressed) ALL OK, the 4-rank all-reduce (`test_ar_nrank`) PASS after
a fix to the test itself (its burst and graph messages were sized at 80 KB against a 16 KiB one-shot cap and had
been failing with the kernel's -7 since the cap changed; the chain's grep hid it behind the race test's ALL OK --
`b6e1c18`). Single-GPU gates, all ALL OK / PASS: atiled_4bit, fold_mxfp4, prefill_4bit, tuned_cfgs, moe_mxfp4,
nvfp4, cache_moe, gemm_fp8, gdn_merge, moe_route_r9k, moe_sum_r9k, fp8_prefill_r9k. 0 MES timeouts on the boot.

Two BetterBench-only runs for the record (the defaults again, and the defaults with the full cudagraph list) ran
after the chain; their numbers are appended below when in.

## Also in this tag

- `notes/mes-timeouts.md` and the guest kernel parameters that settle the RDNA4 MES timeouts under KVM.
- `notes/radiance-test-run.md`: the radiance engine (1.0.4 / 1.0.8 / 1.1.1 / 1.2.0) through our checklist.
- README cut to highlights and a stats table.
