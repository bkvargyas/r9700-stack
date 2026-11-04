# v0.4.2 -- the DeepSeek-V4.1-Flash step, second pass: eight streams, the expert tier and the KV budget (2026-11-04)

Same pin as v0.4.0 and v0.4.1 (`nightly-rocm100-e4a5b553`, vLLM 0.32 line, torch 2.13, ROCm 10.0), same image. The second pass
of the "2x the Sparks" program (notes/exl3-plan.md "Phase 4"; the two DGX Sparks on TensorFold serve this pack at 80.7 tok/s
prose / 118 code single stream, 154-166 at four streams). Six packages built against the v0.4.1 step ledger (notes/exl3-ledger.md
section "v0.4.1"), each server-validated on a clean tree before it was merged. Measured on 8x R9700 in VM100 (210 W / -42 mV,
eight-card runs at the -500 MHz offset).

## Changed

- **Eight streams are supported** (`NSEQ=8`; the default stays 4). The r9k expert tier now takes steps of up to 48 tokens
  (`kernels/r9k_moe_exl3.hip`, `moe/exl3_decode.py`; package A2): the 48-row grouping of the reverted v0.4.1 kernel plus an
  input-rotation pre-pass above 8 tokens (each local slot's gate / up input rotated once into an f16 scratch instead of in every
  block; `R9K_EXL3_DEC_XR_T`, 0 = off). Card 5, graph replay, us a layer, v0.4.1 -> v0.4.2: 1 row 39.9 -> 37.6, 6 rows 134.8 ->
  128.5, 24 rows 532 -> 391, 48 rows 2,333 (exllamav3's fused tier) -> 700; the DSpark drafter's own geometry 82.4 -> 81.2 at 5 rows,
  1,016 -> 343 at 40. Same ops in the same order (the test file's f32 reference, every width 1..48, the exact drafter geometry,
  graphs). NSEQ 8 on the tagged tree: see Validation.
- **The v0.4.1 "48-token tier breaks the drafter" finding was wrong about the cause** (notes/exl3-drafter-heisenbug.md): a
  graph-pool-layout latent bug makes the DSpark drafter accept nothing (1.0 tokens a step) in particular memory layouts, with the
  target's text still correct. v0.4.1's own tier reproduces it with nothing changed but a bigger scratch reserve; a 64 MiB pad
  hides it; eager mode is clean. Sixteen launches excluded the expert tier, every plugin scratch, the counters, the collectives,
  the weights, the shared-experts stream and the markov / hyper-connection / routing hooks; it is computed at run time inside the
  drafter's first MoE from a bit-identical input, and any change to graph-pool allocations hides it. Not fixed (task 71 in the
  plan). **Guard:** `bench/ds41/drafter_selftest.py` runs after every launch (`SELFTEST=1` in `serve/ds41-exl3.sh`): one greedy
  prose and one code request, acceptance length from `/metrics`, exit 1 below 1.3 prose / 3.0 code with the server left running.
- **The fused MoE combine finally runs** (`moe/exl3_combine.py`; package CB2): under `--enable-expert-parallel` the pinned vLLM's
  MoE parallel config is tp 1 / ep 8 and v0.4.1's `tp_size == 8` guard skipped the hook on all 43 runners. **The drafter's markov
  head serves the adaptive path**: this pack carries `mtp.2.confidence_head`, so with adaptive verification on, stock enables the
  confidence head and v0.4.1's kernel declined; a one-launch confidence tail (`r9k_dsv4_markov_conf`: f64 dot, one f32 rounding,
  f32 sigmoid) replaces stock's f32 projection, 300 -> 7.8 us, probabilities within 1e-6 (`R9K_DSV41_MARKOV_CONF=stock` is
  bit-exact). Ledger step 19.11 -> 18.07 ms at one stream, 32.26 -> 31.25 at four.
- **An expert placement map** (`moe/exl3_placement.py`, `R9K_EXL3_EP_MAP`; package H3): the post-MoE all-reduce waits on the slowest
  rank, and the slow rank is the one whose local experts drew the most picks (93-98 % of calls). A load-time map from recorded
  prose + code routing re-assigns which rank holds which of the 384 experts a layer (same experts, same weights, nothing in the
  forward path): step 19.11 -> 18.52 ms at one stream, 32.43 -> 31.41 at four, the post-MoE all-reduce 86 -> 70 us. On by default
  in the launcher (`EP_MAP=` restores the linear placement); not bit-identical to the linear layout (each rank's bf16 partial sum
  groups the picks differently); the map pays only for traffic like the one it was built from and `bench/ds41/epskew/` rebuilds it.
  Replicas of hot experts and vLLM's EPLB were costed and declined (notes/exl3-ep-skew.md).
- **KV budget: `KVMEM` 2.0 GiB a rank at 32k** (387,069 tokens; v0.4.1 1.7 = 329,008) and **1.0 at 128k** (392,289 tokens, three
  full-length requests; 0.8 = 313,831 before; package E). The load-time dequant path that sized the capture peak (a 12,288-row
  dummy expert at speculator capture) runs in 4,096-row pieces, bit-identical to the one-piece call (hipBLAS's result classes
  checked at 13 row counts). 0.75 GiB free on the worst card through the 32k soak; 0.49-0.52 GiB at the 128k needles and soak.
- **The EXL3 decode GEMV, traced and trimmed** (`kernels/r9k_exl3_gemv.hip`, `quant/exl3_gemv.py`; package F): a per-block trace of
  one launch shows that ~3.5 us of every call is the HIP-graph node floor, and that v0.4.1's per-matrix chunking put the fused
  wqa / wkv launch at 360 blocks of 136 VGPRs, more than are resident at once (some blocks started 14-19 us late). One chunk count
  for all groups of a launch (`R9K_EXL3_GEMV_SHARED_KS`, 280 blocks), the first trellis tile issued before the input rotation
  behind an LDS-only barrier (on gfx12 `__syncthreads()` waits on every outstanding global load), a split epilogue for 1-4 rows and
  scalar load bases (136 -> 85 VGPRs): fused_wqa_wkv 18.9 -> 15.1 us at one row (379 GB/s), wo_a 12.1 -> 10.1, 4-8 rows level
  with v0.4.1, 24 / 48 rows unchanged; about -0.28 ms a single-stream step. The 450 GB/s target does not fit a one-launch,
  last-arriver kernel on these shapes: the node floor plus the store -> counter -> partial-load chain (~1.4 us) leaves less than
  the card's peak for the bytes; fewer launches a layer is the next structure. **The NB2=4 crash on main_proj is fixed**: the WMMA
  kernel's static transpose buffers (6 KiB a 128-column tile) were not counted in the LDS budget (49 + 24 KiB > 64 KiB,
  `HSA_STATUS_ERROR_INVALID_ALLOCATION`); NB2=4 is now the default for k >= 8192 (main_proj 404 -> 299 us at 48 rows, 286 -> 221
  at 24), 2 elsewhere. Bit-exact against exllamav3's reconstruct; graph replay equals eager.
- **Opt-in: a "hub" eight-rank all-reduce for 160 KiB - 1.5 MiB messages** (`R9K_AR8_HUB=1`; package G): the quartet sum on one CPU
  reducer thread a rank over the pinned host pools, one host hop instead of two, bit-identical to the fused split kernel: 240 KiB
  76.7 -> 67.1 us, 480 KiB 122.5 -> 116.3, 1 MiB 230 -> 202. Off by default: the targets (50 / 90 / 150) were not met, the
  server-level gain is within noise at four streams, GLM-5.3 at 16 streams measured 3.5 % lower, and the reducer thread spins.
  The microbenchmarks that bound it (uplink ~14 GB/s a switch each way, 2.7 us a flag hop) are in notes/exl3-plan.md.
- `bench/ds41/spectune.py` handles more streams than prompts (`STREAMS=1,4,8`); `notes/exl3-ledger.md` gains the v0.4.1 ledger
  with per-package ceilings (package Z2); `notes/lsa-study.md` evaluates latent sparse attention for our indexers (parked: it
  pays only above ~512k tokens, changes outputs, and has no DeepSeek-V4 support).

## Speed (DeepSeek-V4.1-Flash, eight cards, launcher defaults; v0.4.1 in parentheses, two DGX Sparks in brackets)

Clean tree = git archive of the tag, VM100, the launcher defaults (MAXLEN 32768, KVMEM 2.0, DSpark k = 5 with adaptive
verification, the placement map on); `~/ds41/exl3/S/validate-42.log` (NSEQ 4, 2026-11-04 17:07-17:35 UTC) and
`validate-42-n8.log` (NSEQ 8, 17:47-18:01); spectune, greedy, best of 2 reps.

| | single stream | per stream x 4 (aggregate) | per stream x 8 (aggregate; NSEQ 8) |
|---|---:|---:|---:|
| prose | **103.5 tok/s** (86.8-96.7) [81] | 56.7-64.5 (**199-221**) (48-57; 179-212) [154] | 45.2 (**317**) |
| code | **214.5** (201.7-202.6) [118] | 113.8 (**422-423**) (99.7-99.9; 368-369) [166] | 75.9 (**555**) |
| GSM8K-style | 153.5 (144.7) | 81.7 (213) (71.7-83.8) | 64.0 (217) |

Acceptance lengths unchanged (prose 1.69-1.77, code 4.24-4.32). Against the two Sparks: single stream 1.3x prose / 1.8x code, four
streams 1.3-1.4x prose / 2.5x code, and at eight streams 2.1x / 3.3x their four-stream aggregates. **Power:** the eight cards draw
1.5-1.6 kW during decode at every stream count (sysfs `power1_average` sampled every 2 s through the NSEQ 8 pass; the cap is 210 W
a card) and ~105 W idle with the model loaded; the host is not metered. Tokens a kWh of card power: ~230k prose / ~480k code at
one stream, ~500k / ~950k at four, ~710k / ~1.25M at eight.

## Validation

**DeepSeek-V4.1-Flash TP8, NSEQ 4, the launcher defaults on the tagged tree** (`validate-42.log`): model load 26.34 GiB a rank in
531 s, RoPE caches trimmed (494 MiB freed a rank), KV cache **387,069 tokens** (11.8 x 32k; v0.4.1 329,008). Launch self-test
prose 1.65 / code 3.85. TTFT 1.28 s on a 1.5k prompt (1,210 tok/s) and 3.42 s on 12.3k (3,599 tok/s); a 4 x 12k prefill burst
leaves 0.52 GiB free on the worst card. spectune: the table above. divergence.py **23 / 24, 24 / 24, 24 / 24** (v0.4.1 22 / 24 / 22).
Needles 3 / 3 at 8k, 16k and 28k. Strict sanity 0 bad of 25 at 5 streams and 0 of 80 at 8, twice. Soak (8 clients, 200-28k-token
prompts, 925 s): 187 requests, 0 errors, median prompt 14.2k tokens, 2,945 prompt tok/s, worst card 0.52 GiB free throughout.
GSM8K 200: **194 / 200 = 97.0 %** (1 truncated; v0.4.1 193). 0 server errors.

**NSEQ 8 on the same tree** (`validate-42-n8.log`, the first eight-stream validation of this configuration): launch self-test
prose 1.66 / code 3.55; spectune 1 / 4 / 8 streams prose 101.7 / 51.8 x 4 (184) / 45.2 x 8 (**317**), code 214.7 / 113.0 x 4
(417) / 75.9 x 8 (**555**), GSM8K-style 158.7 / 83.9 x 4 (204) / 64.0 x 8 (217). Strict sanity 0 bad of 45 at 9 streams and 0 of
80 at 16. Soak (8 clients, 200-28k, 606 s): 125 requests, 0 errors, median prompt 16.0k tokens, 3,179 prompt tok/s. Self-test
after the soak unchanged (1.66 / 3.55). 0 server errors.

**The other three configurations on the same tree and image** (`~/gate-042.sh` -> `~/gate042.log`, 14:47-16:49 UTC; v0.4.1's gate
in parentheses):

- **Qwen3.8-27B-NVFP4 TP2:** probe 220.4 tok/s mean (code 236 / prose 111 / json 281 / math 254), 4.64 tokens a step, 512 / 607
  tok/s at 8 / 16 streams, prefill 3,874 tok/s on 8k. BetterBench combined **195.5** (196.4), update p99 24.7 ms; strict sanity
  0 bad of 2,460; soak 199 requests in 961 s, 0 errors, 3,050 prompt tok/s. 0 server errors.
- **Qwen3.8-Flash-Next TP4:** KV cache 479,076 tokens; probe 222.0 (218 / 160 / 262 / 249), 3.11 tokens a step, 466 / 909 at
  8 / 16 streams, prefill 6,556; BetterBench combined **184.8** (185.3), update p99 15.2 ms; strict sanity 0 bad of 2,460; soak 424
  requests in 906 s, 0 errors, 6,483 prompt tok/s. 0 server errors.
- **GLM-5.3-Flash TP8 (32k production recipe; its small collectives take the v0.4.1 one-kernel paths, the hub path is off):**
  KV cache 203,329 tokens (176,443); decode-only **95.1 tok/s** median (93.2), four streams 176 then **221** tok/s aggregate (172 /
  234), 12.7k needle correct at 118.7 tok/s at depth (123.5) and 83 % acceptance; strict sanity (thinking on) 0 bad of 780; soak
  168 requests in 978 s, 0 errors (170). 0 server errors.

## Known limits

- The drafter Heisenbug above: not fixed, guarded by the self-test. If a launch fails the self-test, change nothing else and
  relaunch with a different `KVMEM` or `NSEQ`; report the launch log.
- 32k and 128k recipes, the capture-time KV budget and GLM above ~75k on this nightly: as in v0.4.0 / v0.4.1. The 128k recipe at
  KVMEM 1.0 sits at the edge of the 0.5 GiB rule (0.49-0.52 GiB free); 0.8 keeps margin.
- The expert tier at 6 rows is 128.5 us a layer (target 120): the per-group trellis stream at 4-5 groups dominates and the launch
  geometry does not move it. The split-path all-reduce at 240-480 KiB stays at 77 / 122 us by default (the hub path above is
  opt-in). What is left in the step at one stream is per-step routing noise across the eight ranks (~1.9 ms), the expert tier's
  bytes, and the collectives' latency floors.
