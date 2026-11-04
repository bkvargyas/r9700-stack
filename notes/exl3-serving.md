# DeepSeek-V4.1-Flash EXL3 2.9 bpw -- serving tuning (package S, Phase 3, 2026-11-02/03)

Eight R9700s on VM100, TP8 + EP, DSpark drafter on, HIP graphs, fp8_ds_mla KV, prefix caching off, NSEQ 4, SCLK -500 MHz.
Tools (bench/ds41/): `memprobe_ext.py` + `memprobe.py` (per-rank allocator state, tensor inventory, raw-hipMalloc headroom;
stock `--worker-extension-cls` + the dev endpoint `/collective_rpc`, launcher `MEMPROBE=1`), `vramwatch.py` (driver
VRAM peak over a workload), `prefillpeak.py` (prefill-only load), `spectune.py` (the draft-length table: prose / code /
GSM8K at 1 and 4 streams, usage-based tok/s, acceptance from the /metrics spec-decode counters).
Served from an isolated copy (VM100 `~/r9700-exl3-S` = committed HEAD + this package's files) because the shared
`~/r9700-exl3` sync carries other packages' uncommitted work (one sync pulled a `r9k_mhc.hip` that did not compile and
the launch failed in the libr9k build).

## 1. Memory audit (rank 0, steady state, k=5 drafter + HIP graphs, KVMEM 1, NBT 2048, MAXLEN 32768, NSEQ 4)

Three views of the same card, and they disagree:
- **sysfs `mem_info_vram_used`** (the driver's VRAM manager) -- the real one. A raw `hipMalloc` loop inside the worker
  (`memprobe.py try`) got **1,568 MiB** on rank 0 while sysfs showed 1,499 MiB free.
- **`hipMemGetInfo` / `torch.cuda.mem_get_info`** reported only **610 MiB** free at that moment (and 54 MiB after a prefill
  burst that left sysfs at 1.0 GiB free): it under-reports by ~0.9-1.3 GiB (KFD-side accounting that is not VRAM; it
  also moves with the number of captured graphs -- see 2a). vLLM's "Graph capturing ... took X GiB" is a
  `mem_get_info` delta, so it inherits the same error. **Budget against sysfs; never against mem_get_info.**
- **torch** (`memory_stats`): allocated / reserved; plus a walk of every device tensor reachable from the model runner,
  the drafter and the plugin / vLLM module globals (`memprobe.py inv`), and a gc scan for tensors the walk missed (none).

| component (rank 0) | MiB | note |
|---|---:|---|
| routed experts (EXL3 trellis + suh/svh + pointer tables, 40 layers x 48 local) | 23,516 | |
| drafter (DSpark mtp.0-2: experts 813, attention 75, shared 25, markov/confidence heads + rest 188, hc 6, router 4) | 1,111 | plan said ~0.3 GiB |
| attention (wq_a/wq_b/wkv/wo_a bf16/wo_b, indexer, compressor) | 977 | |
| **RoPE cos/sin caches** (two: SWA layers and compressed layers, 1,048,576 positions x 64 f32 each) | **512** | see 2c |
| shared experts (bf16, TP-sharded) | 338 | |
| embed / lm_head (bf16, vocab-sharded) | 158 / 158 | |
| router (gate f16 [384, 5120] x 40) | 150 | |
| hyper-connections (hc_* f32) | 75 | |
| other (norms, sinks, Engram device part 29, ...) | 129 | |
| vLLM workspace manager (sparse-MLA / indexer prefill workspace) | 174 | |
| r9k expert-tier scratch (P1's 88f6cb5: 24 MiB) | 24 | was ~1.2 GiB before P1 |
| runner buffers | 24 | |
| **= weights + fixed buffers** | **27,345** | "Model loading took 26.76 GiB" |
| KV cache (KVMEM 1 = 193,534 tokens; one 1,024 MiB pool) | 1,024 | |
| **= inventoried device tensors** | **28,369** | |
| allocated, not reachable from Python (C++-held: BLAS workspaces, compiled-graph statics, ...) | 731 | torch allocated 29,100 |
| torch cache: reserved - allocated (fragmentation; 602 MiB "inactive split") | 840 | reserved 29,940; graph pool 242 of it |
| non-torch VRAM (RCCL + ar8 IPC buffers, HIP graph execs, code objects, runtime scratch) | 1,181 | sysfs - reserved |
| **sysfs used, idle** | **31,121** | 30.4 GiB; worst card 31,213 (card5) |
| sysfs free, idle (32,624 MiB card) | 1,503 | worst card 1,411 |
| prefill transient (4 x 6k-token prompts at NBT 2048) | +400-440 | allocated peak +440, reserved +394 |
| Engram tables (2 x 11.8 GiB fp8 + scales) | (24,170) | pinned HOST memory, mapped; not VRAM |

The reachable-tensor walk accounts for every Python-owned device tensor (gc scan: 0 MiB unreached); the rest of the
footprint is the allocator's cache (0.8 GiB), C++-held allocations (0.7 GiB) and memory outside torch (1.2 GiB).

## 2. Reclaim

| item | how | real VRAM (sysfs, a card) | other effect | status |
|---|---|---:|---|---|
| (a) HIP-graph capture sizes | `CGSIZES` = 1,2,4 + (k+1) x 1..NSEQ (k=5: 1,2,4,6,12,18,24) instead of vLLM's 1,2,4,6,8,12,16,24,32,40,48 | -35 MiB | "Graph capturing took" 1.57 -> 1.17 GiB (mem_get_info), torch graph pool 256 -> 242 MiB, capture 27 -> 19 s; speed unchanged (table 4) | **launcher default** |
| (b) allocator fragmentation | `ALLOC_CONF=expandable_segments:True` (PYTORCH_ALLOC_CONF) | **-470 MiB** (idle 31.12-31.21 -> 30.66-30.73 GiB) | inactive splits 602 -> 0 MiB, "Model loading took" 26.75 -> 26.51 GiB (load-time temporaries returned), the low card (card2, -350 MiB) evens out; speed unchanged; graphs, ar8 IPC (its own hipMalloc) and RCCL fine | **launcher default** |
| (c) drafter length | k=3 vs k=5 | -215 MiB at k=3 | drafter graphs scale with k; k=5 is the speed choice | (k=5 kept) |
| (d) adaptive verification | ADAPTIVE=true | +78 MiB | varlen verify graphs (capture 71 s) | kept (speed, table 4) |
| (e) r9k expert-tier scratch | P1, 88f6cb5 | ~-1.2 GiB vs Phase 2 | 24 MiB a rank now | in HEAD |
| (f) RoPE cos/sin caches | **plugin change, for the owner** (below) | up to -448 MiB at MAXLEN 131072 (-496 at 32768) | | not done |
| (g) MoE tier buffers | -- | ~0 | exllamav3 FusedBuffers (116 MiB) + dq_temp (22.5 MiB) are already shared across the 40 layers and serve the prefill tiering; the mgemv coop buffers are a few MiB | nothing to do |
| (h) wo_a bf16 (320 MiB) | keep (plan) | | | kept |
| (i) prefill transients | NBT / `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | see 3 | | see 3 |

**(f) for the owner -- trim the two RoPE caches to max_model_len (exact).** Both DeepSeek-V4.1 rotary embeddings build
`cos_sin_cache` [1,048,576, 64] f32 = 256 MiB a rank each (vllm/models/deepseek_v41/common/rope.py `build_deepseek_v4_rope`
-> model_executor/layers/rotary_embedding DeepseekScalingRotaryEmbedding / DeepseekV4ScalingRotaryEmbedding
`_compute_cos_sin_cache`: `t = arange(max_position_embeddings * scaling_factor)`). The SWA layers' cache is
max_position_embeddings (1,048,576) x factor 1.0; the compressed layers' is original_max 65,536 x factor 16. The values
at position p do not depend on the cache length, so slicing both to `vllm_config.model_config.max_model_len` rows after
construction (e.g. in `models/deepseek_v41_exl3.py` load_weights / a post-load hook: for every module with a
`rotary_emb` attribute, `rotary_emb.cos_sin_cache = rotary_emb.cos_sin_cache[:max_model_len].contiguous()` once, then drop
the old buffer; the layers share two objects, so two slices) is bit-identical for every position the server can see and
returns 448 MiB at MAXLEN 131072, 496 MiB at 32768. Check that nothing indexes the cache by a position beyond
max_model_len (the drafter's lookahead positions are < max_model_len + k; slice to max_model_len + 64 to be safe). The
launcher route (`--hf-overrides '{"text_config":{"max_position_embeddings":N}}'`) only reaches the SWA cache and fails
anyway: this config class keeps `text_config` as a plain dict at override time, so vLLM replaces it ("text_config ...
does not have num_attention_heads").


## 4. Draft length and adaptive verification (DSpark)

`spectune.py`: greedy, natural stop; prose = ask3's lighthouse essay (thinking off), code = ask3's LRU module (thinking
off), gsm = GSM8K test questions as gsm8k_ds41.py asks them (default template, thinking on; 1 stream: questions 0-3 back
to back, 4 streams: questions 0-15 dealt round-robin). Numbers are **per-stream decode tok/s** (usage-based: completion
tokens over first-token-to-end; the 4-stream aggregate decode rate is ~4x the per-stream figure) and the **mean acceptance
length** (MAL = 1 + accepted / drafts, from the /metrics spec-decode counters around each run). Best of 2 reps (3 where
noted); the other agents' single-card tests made individual reps noisy (a second rep is usually the warm one: the
first request of an AV process pays a ~20 % warm-up, see WARM below).
Each row is one launch (relaunch ~6-8 min); capture list = 1,2,4 + (k+1) x 1..4.
**Rows above the line ran before package H's hyper-connection kernels (5a943a7); rows below, with them
(`mhc:r9k-seam(M<=32)` in the launch log). Timing is not comparable across the line.**

| k | adaptive | prose x1 | prose x4 /stream | code x1 | code x4 /stream | gsm x1 | gsm x4 /stream | MAL prose / code / gsm (x1) | graph capture |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 4 | false | 69.8 | 36.4 | 132.2 | 70.8 | 111.8 | 63.5 | 2.07 / 3.91 / 3.29 | 19 s |
| 5 | false | 68.2 | 36.7 | 158.8 | 80.7 | 112.3 | 67.2 | 1.99 / 4.65 / 3.27 | 27 s (vLLM list) |
| 5 | false, short graph list | 69.6 | 34.8 | 165.2 | 83.1 | 129.9 | 64.3 | 1.93 / 4.62 / 3.59 | 19 s |
| 6 | false | 63.7 | 24.3 | 155.0 | 57.4 | 114.2 | 52.0 | 1.97 / 4.86 / 3.51 | 21 s |
| --- with H --- | | | | | | | | | |
| 3 | false | 63.9 | 44.0 | 115.8 | 75.2 | 101.9 | 65.8 | 1.85 / 3.34 / 2.92 | 13 s, 0.92 GiB |
| 4 | true | 65.7 | 40.3 | 133.5 | 77.1 | 103.3 | 66.6 | 1.88 / 3.97 / 3.04 | 67 s, 1.70 GiB |
| **5** | **false** | **74.6** | 38.3 | 158.1 | 84.3 | **120.9** | 70.7 | 2.06 / 4.39 / 3.33 | 19 s, 1.17 GiB |
| **5** | **true** | 70.1 (3 reps) | **46.8** (3 reps) | **159.3** | 83.6 | 113.7 | 69.6 | 1.78 / 4.50 / 3.14 | 71 s, 1.71 GiB |
| 6 | true | 68.1 | 39.0 | 158.4 | 84.3 | 112.0 | 67.4 | 1.76 / 4.72 / 3.22 | 73 s, 1.70 GiB |

Per-position acceptance (k=5, with H, x1): prose 0.59 / 0.29 / 0.12 / 0.04 / 0.02; code 0.91 / 0.78 / 0.68 / 0.56 / 0.47;
gsm 0.82 / 0.62 / 0.46 / 0.28 / 0.14. The server log's `SpecDecoding metrics` lines agree (e.g. code x4, k=5: mean acceptance
length 4.42-4.47, per-position 0.89 / 0.78 / 0.67-0.69 / 0.58-0.59 / 0.50-0.51).

- **k = 5 wins.** k=4 loses code (-15-20 %), k=3 more; k=6 gains nothing on code (MAL 4.86 vs 4.65, but a 7-token
  verify step a stream costs more) and loses at 4 streams (28-token steps).
- **`enable_adaptive_verification: true` runs on this pin and on RDNA4** (the Phase 1 assumption that AMD must keep it false
  does not hold: the selector check passes for the DeepSeek-V4.1 ROCm backends, and the DSpark cost-curve profiling runs
  at init; capture takes 71 s instead of 19 s and the captured varlen graphs cost +78 MiB of real VRAM). It trims verification
  per request on device: on prose at four streams it is **+22-28 %** (46.3-46.8 vs 36.1-38.3 per stream, three reps each
  side); single-stream prose and GSM8K lose 5-6 % (70.0 vs 74.5; 113.7 vs 120.9); code is unchanged.
  Greedy output is unaffected (verification is lossless; the validation below ran with it on).
- Default: **DSPARK_K=5, ADAPTIVE=true** (four-stream prose is where we trailed the Sparks most; ADAPTIVE=false is the
  knob for a single-user box and buys back ~6 % single-stream prose).

## 3. KV budget (the release launches, image r9700/vllm:0.4.0rc, HEAD with the RoPE trim)

The trim runs ("RoPE cos/sin caches trimmed to 32896 positions (494 MiB freed a rank)") and the idle footprint drops by ~190 MiB
a card (31,380-31,452 MiB at KVMEM 1.7 against 31,574-31,626 on the untrimmed tree), but **the serving peak does not move**:
after a 4-stream 12k prefill burst and after the 15-minute soak the worst card sits at 31,909-31,913 MiB used (0.69-0.70 GiB
free) with or without the trim. The binding constraint is earlier anyway: **the drafter's graph capture**. With the trim in,
`KVMEM=2.2` (425,776 tokens) and `KVMEM=2.0` (387,069 tokens) both die on card 7 while the DSpark speculator captures, in the
expert tier's dequant path for a > 24-token dummy batch (`exl3_moe.py _gemm_dq: torch.empty_like(a)`, 120 MiB, "0 bytes free,
30.37 GiB allocated by PyTorch"); the other ranks then hang in the broadcast. The capture-time peak is above the steady-state
peak by more than the 0.3-0.5 GiB those budgets add, so the trim buys headroom for the capture, not KV.

| MAXLEN | KVMEM | KV tokens | idle used (worst card) | peak used after burst / soak | capture |
|---|---:|---:|---:|---:|---|
| 32768 | 2.2 | 425,776 | -- | -- | OOM card 7 (speculator capture) |
| 32768 | 2.0 | 387,069 | -- | -- | OOM card 7 (speculator capture) |
| **32768** | **1.7** | **329,008** | 31,452 MiB | 31,913 MiB (0.69 GiB free) | 71 s, "took 1.41 GiB" |
| 131072 | 1.5 | 588,434 (vLLM's 128k accounting) | -- | -- | OOM card 7 (speculator capture; the indexer's prefill-logits workspace is ~0.7 GiB larger at 128k) |
| 131072 | 1.0 | 392,289 (2.99 x 131k) | 31,252 MiB | 31,821 at the 64k needle, 31,963 at 100k, 32,407-32,408 MiB (0.21 GiB free, cards 1 and 4) at the 120k needle and the 60-100k soak | 72 s, "took 1.43 GiB" |
| **131072** | **0.8** | **313,831** (2.39 x 131k) | 31,032 MiB | 32,207 at the 120k needle, 32,231 MiB (0.38 GiB free) at the 128.5k needle and the 100-125k soak | 71 s, "took 1.43 GiB" |

Defaults: `KVMEM=1.7` at the launcher's MAXLEN 32768 (10 concurrent 32k requests); `MAXLEN=131072 KVMEM=0.8` for the long-context
launch (313,831 tokens: two full-length requests and change; KVMEM 1.0 retrieves at 120k too but leaves 0.21 GiB). Open item for v0.4.1: capture the speculator with the dequant tier off (or size R9K_EXL3_MOE_R9K_M to the largest
capture width) so the capture peak falls back under the steady-state peak -- that is 0.3-0.5 GiB of KV (60-100k tokens).

## 5. Validation of the defaults (launch A: MAXLEN 32768, KVMEM 1.7, k = 5, adaptive on, graphs, 0.4.0rc image; 2026-11-03 06:51-07:35 UTC)

`validate.sh` -> `S/validate-A.log`; the server up since 06:51 UTC with "r9700_vllm 0.4.0 registered", KV 329,008 tokens.

| | |
|---|---|
| spectune (per-stream tok/s, MAL) | prose x1 **72.5-73.0** (1.86-1.88), x4 **44.3-45.6** (1.72; 166-167 aggregate); code x1 **152.2** (4.31), x4 **81** (4.1; 296-298 aggregate); GSM8K-style x1 **122** (3.4), x4 64-67 (3.04-3.09; 140-159 aggregate) |
| prompt processing | one 12.3k prompt 2,532 tok/s (4.86 s); four 12.3k prompts 3,040 tok/s aggregate; a 1.5k prompt 1.43 s |
| divergence.py (greedy decode vs fresh prefill, 24 tokens) | prose 23/24 (mean dlogprob 0.124), code 24/24 (0.016), math 24/24 (0.045) |
| needle.py 8k / 16k / 28k (3 facts each) | 3/3, 3/3, 3/3 (prefill 2.3 / 4.5 / 7.7 s) |
| strict sanity (NSEQ 4) | 5 streams (5 rounds after 20-24k-token prefills): **0 bad of 25**; 8 streams (10 rounds): **0 bad of 80** |
| soak (8 clients, 200-28k-token prompts, 934 s) | 174 ok, 0 errors, 1 context-length rejection (by design), prompts 409-32,694 tokens (median 14,846), 2,816 prompt tok/s, 47.5 output tok/s, median latency 42.6 s; worst card 0.69 GiB free throughout |
| GSM8K 200 (greedy, thinking on) | **195 / 200 = 97.5 %**, 1 truncated, 3.2 min |
| server log | 0 errors (the transformers "model of type deepseek_v41" notice x 8 is stock's) |
| sysfs after | 31,836-31,916 MiB used a card |

## 5b. The 128k launch (MAXLEN 131072, KVMEM 1.0 = 392,289 KV tokens, otherwise the defaults; `validate-B.sh` -> `S/validate-B.log`, 08:01-08:29 UTC)

| | |
|---|---|
| needle.py 64k / 100k / 120k (3 facts each) | **3/3, 3/3, 3/3**; prefill 63.1k tokens in 17.5-19.0 s (3.4k tok/s), 98.9k in 27.1-27.3 s (3.6k), 118.5k in 32.4-33.0 s (3.6k) |
| soak (2 clients, 60-100k-token prompts, 316 s) | 13 ok, 0 errors, median prompt 78,976 tokens, 3,234 prompt tok/s |
| VRAM | worst card 0.78 GiB free at the 64k needle, 0.65 at 100k, **0.21 GiB at 120k and through the soak** (cards 1 and 4 at 32,391-32,408 MiB): the per-request prefill transient grows with the prompt's context (the indexer's logits over chunk x context) |
| server log | 0 errors |

## 5c. The documented 128k recipe (MAXLEN 131072, KVMEM 0.8 = 313,831 KV tokens; `validate-B3.sh` -> `S/validate-B3.log`, 08:38-08:59 UTC)

| | |
|---|---|
| needle.py 120k / 128.5k (3 facts each) | **3/3, 3/3**; prefill 118.5k tokens in 32.5-33.3 s, 126.7k in 34.7-35.0 s (3.6k tok/s) |
| soak (2 clients, 100-125k-token prompts, 213 s) | 6 ok, 0 errors, median prompt 119,396 tokens, 3,249 prompt tok/s, median latency 67.5 s |
| VRAM | worst card 0.41 GiB free at the 120k needle, **0.38 GiB at 128.5k and through the soak** (cards 1 and 4 run ~0.4 GiB heavier than the rest at every budget) |
| server log | 0 errors |





## 3b. v0.4.2: the capture-time transient removed; KVMEM 2.0 at 32k (package E, 2026-11-04)

The DSpark speculator's graph capture routes a (k + 1) x NBT = 12,288-token dummy batch to one expert, which took the heavy-expert
dequant path `_expert_dq` with per-call `torch.empty` temporaries (`ah` 120 MiB, `g` / `u` 56 MiB each, `xe` 120 MiB, `y` 240 MiB):
that ~0.5 GiB transient, not the serving peak, capped KVMEM at 1.7 (section 3). Now the path runs in pieces of exactly 4,096 rows
(`R9K_EXL3_MOE_DQ_ROWS`), the last piece ending at the expert's last row and overlapping the one before (the atomic path adds the
overlap once), using the fused tier's idle state / intermediate buffers as scratch -- no allocation per call. 4,096 rows and not
1,024 because hipBLAS changes its reduction with M for these f16 shapes below 4,096 (result classes M 1-127 / 128-4095 with
exceptions / >= 4096) while every M >= 4,096 matches M = 4,096 on every row (checked at 13 values to 24,576): the pieces are
bit-identical to v0.4.1's one-piece call (card 7: K = 2 / 3 / 4 at 1 .. 12,289 rows, both paths, graph replay = eager, a full
`apply()` at T = 5,000; `bench/ds41/dqchunk_test.py`, `dqchunk_probe.py`). Timing 12,288 rows 31.8 -> 30.9 ms.

| tree | MAXLEN | KVMEM | KV tokens | capture window, worst card | after the 4 x 12k burst | 15-min 8-client soak |
|---|---:|---:|---:|---|---|---|
| v0.4.1 | 32768 | 1.7 | 329,008 | card 8: 31,978 MiB used, 646 MiB free | | 0.86 GiB free (validate-41) |
| E | 32768 | 1.7 | 329,008 | card 3: 31,787 MiB, 837 MiB free (card 8 -242 MiB) | | |
| **E** | **32768** | **2.0** | **387,069** | card 5: 32,109 MiB, 515 MiB free | 31,854 MiB, **0.75 GiB free** | 184 ok, 0 errors, 2,894 prompt tok/s, 0.75 GiB free; spectune prose 95.7 / code 191 (MAL 1.82 / 4.26) |
| E | 32768 | 2.2 | 425,776 | 32,309 MiB, 315 MiB free | 32,054 MiB, 0.56 GiB free | (soak running at the time of writing) |
| E | 131072 | 1.2 | ~471k | card 7: 31,807 MiB (capture fine) | 120k needle 3/3 at 0.34 GiB free, 128.5k needle 3/3 at **0.32 GiB** | prose 102-103 / code 195-196 single (MAL 1.96-1.99 / 4.43-4.46); below the 0.5 GiB rule -> 1.0 queued |
| **E** | **131072** | **1.0** | **392,289** (2.99 x 131k) | 31,609 MiB at capture end (0.25 GiB free; the 128k launch's capture is the tight point) | 120k needle 3/3 at 0.56 GiB free, 128.5k needle 3/3 at **0.52 GiB** | 5-min 2-client 100-125k soak (`E-128kv10.log`, 07:02-07:10 UTC): 8 ok, 6 over-length rejections, 0 errors, 3,284 prompt tok/s, median latency 78.7 s, **0.49 GiB free** (card 1; the other seven 0.55-1.02); spectune prose 88 / code 198-200 single (MAL 1.65 / 4.51-4.56), prose x4 184-205 / code x4 367-370 aggregate; divergence 23 / 24 / 23 |

Default: **KVMEM 2.0 at MAXLEN 32768** (Brian, 2026-11-04: "commit E at 2.0 when the soak passes"). 2.2 survives capture with only
315 MiB free and its divergence was the weakest run (20 / 24 on prose): not taken. At 128k, 1.2 retrieves at 128.5k but leaves 0.32 GiB
(v0.4.1's wo_a-on-EXL3 freed ~230 MiB a rank since v0.4.0's 1.0 run, which left 0.21). **The 128k recipe is now KVMEM 1.0**
(392,289 tokens, three full-length requests): needles 3/3 at 120k and 128.5k, the 2-client 100-125k soak clean, 0.52 GiB free at the
needles and 0.49 after the soak on the worst card, i.e. at the edge of the 0.5 GiB rule (10 MiB under it in the soak's worst sample);
0.8 (313,831 tokens, v0.4.0's documented recipe) is the margin choice. The 128k capture itself ends at 0.25 GiB free on the worst card
(the indexer's prefill-logits workspace is ~0.7 GiB larger at 128k), which is why 1.5 fails capture and 1.2 is the ceiling that captures.
