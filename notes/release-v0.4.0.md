# v0.4.0 -- DeepSeek-V4.1-Flash on eight R9700s from the published EXL3 pack (2026-11-03)

Same pin as v0.3.7 / v0.3.8 (`nightly-rocm100-e4a5b553`, vLLM 0.32 line, torch 2.13, ROCm 10.0). The image now also builds
exllamav3 v1.6.6's gfx1201 extension (MIT; CREDITS.md). Measured on 8x R9700 in VM100 (384 GiB guest, 210 W / -42 mV; eight-card
runs at the -500 MHz clock offset this box's 120 V feed needs). Brian: "Lets do a 3-bit / 2.9-bit EXL3 version" (2026-11-01),
"begin the payoff process" (11-02), "ship 0.4.0" (11-03). The full record is notes/exl3-plan.md (architecture, bring-up log,
Phase 2 / 3 results), notes/exl3-kernels.md, notes/exl3-layer-map.md and notes/exl3-serving.md.

## Added

- **DeepSeek-V4.1-Flash-1017 at 2.9 bits a weight on eight cards** (`serve/ds41-exl3.sh`). The checkpoint is LibertAIDAI's
  EXL3 pack (196 GiB, 43 shards; 40 MoE layers of 384 routed + 1 shared expert, 6 a token, K = 3 bits with K = 2 on layers
  18-22; MLA with 64 heads x 512, sparse top-k attention with an indexer, hyper-connections (hc_mult 4), Engram n-gram
  layers at 1 and 14, the DSpark three-layer drafter, 1M-position yarn RoPE) plus the base checkpoint's two Engram shards
  and tokenizer, composed into one directory (`bench/ds41/compose-model-dir.sh` on VM100 shows the layout). Serving shape:
  tensor parallel 8 with expert parallel (48 whole experts a rank a layer, no expert sharding), the EXL3 trellis format kept
  as-is for every routed expert and the attention / dense linears (TP slices on the format's 128-wide Hadamard blocks), the
  6-bit lm_head, the shared experts, the output projection `wo_a` (eight 128-wide slices the format cannot split along the
  head dimension), the compressor and the indexer's `wk` dequantized to bf16 at load (0.9 GiB a rank), the Engram tables
  (2 x 11.8 GiB fp8) in pinned host memory through vLLM's offload, fp8 KV (`fp8_ds_mla`), HIP graphs, the DSpark drafter at
  k = 5 with adaptive verification. Memory a card: 27.3 GiB weights and fixed buffers, the rest KV.
- **EXL3 on gfx1201** (`quant/exl3.py`, `quant/exl3_moe.py`, `moe/exl3_decode.py`, `kernels/r9k_moe_exl3.hip`). Dense linears
  through exllamav3's `exl3_gemm` / `exl3_mgemm` (one multi-matrix launch for stacked projections sharing K; 8-row chunks to
  16 rows, dequantize + 1024-row hipBLAS chunks above -- the autotuned trellis GEMM is slower than both on this GPU), static
  per-layer buffers, the input Hadamard in explicit fp32. Routed experts: our batched decode kernel (bit-exact trellis decode
  against `reconstruct`, device-side expert grouping, gate / up / down in two launches a layer, 24 MiB of scratch a rank;
  `R9K_EXL3_MOE_DECODE=mgemv` keeps exllamav3's per-slot tier) for steps of up to 24 tokens, exllamav3's fused MoE tier for
  prefill. `R9K_EXL3_DUMMY=1` synthesizes a pack for launch tests without the weights.
- **Stock's DeepSeek-V4 / V4.1 path on RDNA4** (`models/deepseek_v41_rdna4.py`, `attn/dsv4_decode.py`, `kernels/r9k_mhc.hip`,
  `hc_dsv4.py`; `R9K_DISABLE=dsv41` turns the whole set off): the sparse-attention indexer's architecture gate opened (its op
  is Triton / torch, it runs on gfx12); prefill indexer logits through aiter's Triton `fp8_mqa_logits` instead of the
  [heads, M, N] torch fallback that ran out of memory at 2k tokens; decode attention through stock's split-K Triton kernels
  with RDNA4 tuning (BLOCK_K 16, ~96 workgroups: 180 -> 19 us a layer, flat to 28k of context; `R9K_DSV41_DECODE=ragged|stock`
  for the other paths); the hyper-connection seams in one HIP launch each (post folded into the delayed pre, deterministic
  reduction, bit-exact against the torch reference: 25.7 -> 8.0 us a seam at one token, ~1.4 ms a step; `R9K_DSV41_MHC=stock`).
  **Stock bug found on the way:** on every ROCm GPU outside gfx942 / gfx950 the decode fallback (`rocm_sparse_attn_decode`)
  runs the ragged kernel without the fused inverse RoPE and still reports its rows as rotated, so decode attention output
  reaches `wo_a` rotated -- coherent text that doubles words and loops, GSM8K 0 %. bench/ds41/divergence.py (greedy decode
  against a fresh prefill of every prefix) is the test that found it; the split-K path carries the correct epilogue.
- **Eight-rank all-reduce for 32 KB - 2 MB** (`comm/r9k_ar8.py`, `comm/r9k_ar.py`, `kernels/r9k_ar.hip`): above the one-shot
  host exchange, a pair reduce-scatter, a quartet exchange through the pinned host pools and a pair all-gather in one fused
  kernel. Quiet-box timings (HIP graphs, us): 8 tokens x 5120 bf16 47 (RCCL 99, the old ar8 123), 16 tokens 65 (111), 24 tokens
  83 (125), 1 MB 230 (240); 2 MB and up stay on RCCL. Defaults `R9K_AR8_SPLIT_KB=32`, `R9K_AR8_HOST_MAX_KB=2048`. GLM's
  multi-token steps take the same path (its gate numbers below).
- `bench/ds41/`: divergence.py, needle.py, tprobe.py (usage-based token counting, spec-decode safe), longdecode.py, ask*.py,
  gsm8k_ds41.py, spectune.py (draft-length table), memprobe.py / memprobe_ext.py (per-rank allocator audit through a stock
  worker extension), vramwatch.py, prefillpeak.py.
- `docker/Dockerfile`: exllamav3 v1.6.6 cloned at the tag, its extension built for gfx1201 and import-checked.

## Changed

- The EXL3 loader trims both DeepSeek-V4.1 RoPE cos / sin caches (1,048,576 positions x 64 fp32, two objects) to
  `max_model_len + 128` rows after load: 494 MiB a rank back to the allocator, bit-identical for every reachable position.
  It did not turn into KV cache: the serving peak after a prefill burst is the same 31.16 GiB a card with or without it, and
  the drafter's graph capture is the binding constraint (`KVMEM` 2.0 and 2.2 run card 7 out of memory in the capture's
  dequant tier; notes/exl3-serving.md section 3). The 32k budget stays at 1.7 GiB; recovering the capture-time transient
  is the first v0.4.1 serving item.
- `serve/ds41-exl3.sh` defaults from package S's tuning (notes/exl3-serving.md): HIP-graph capture sizes = 1, 2, 4 and
  (k + 1) x 1..NSEQ (the only widths a DSpark step produces), `PYTORCH_ALLOC_CONF=expandable_segments:True` (470 MiB of
  allocator fragmentation a card), DSPARK_K 5, ADAPTIVE true, KVMEM 1.7 GiB at MAXLEN 32768 (329,008 KV tokens, ten concurrent 32k requests),
  0.8 GiB at MAXLEN 131072 (313,831 tokens, the long-context recipe). Memory is budgeted against the driver's sysfs counter:
  `hipMemGetInfo` under-reports free VRAM by 0.9-1.3 GiB on this pin (and vLLM's "Graph capturing took" inherits the error).
- An oversize message on the flat N-rank collective without its 4-rank object fell over; it goes to RCCL now (`R9kAllReduceN`
  keeps its group).

## Speed (DeepSeek-V4.1-Flash, eight cards, DSpark k = 5, greedy; two DGX Sparks on TensorFold in parentheses)

| | single stream | per stream x 4 (aggregate) |
|---|---:|---:|
| prose (essay, thinking off) | 70-75 tok/s (80.7) | 47 (187) (154 aggregate) |
| code (a module, thinking off) | 158-159 (118) | 83-84 (336) (166 aggregate) |
| GSM8K-style (thinking on) | 111-121 | 67-70 |
| decode after a 2k / 16k / 28k prompt | 82 / 80 / 75 | |
| prompt processing | 3,400 tok/s (2,100) | |

Mean acceptance length 1.8-2.1 prose, 4.3-4.5 code, 3.1-3.3 GSM8K. Adaptive verification is +22-28 % per stream on prose at
four streams and -5-6 % single stream (ADAPTIVE=false for a one-user box). Where the step goes and what is next: the
"Phase 4" section of notes/exl3-plan.md.

## Validation

**DeepSeek-V4.1-Flash TP8, the launcher defaults (MAXLEN 32768, KVMEM 1.7 = 329,008 KV tokens, DSpark k = 5 with adaptive
verification, HIP graphs, NSEQ 4; image r9700/vllm:0.4.0rc, VM100 `~/ds41/exl3/S/validate.sh`, 2026-11-03 06:51-07:21 UTC):**
spectune per-stream decode prose **72.5-73.0** tok/s single (acceptance length 1.86-1.88), **44.3-45.6** per stream at four
(166-167 aggregate); code **152.2** single (4.31), 81 per stream at four (296-298 aggregate); GSM8K-style with thinking **122**
single (3.4), 64-67 at four (140-159 aggregate). Prompt processing 2,532 tok/s on one 12.3k prompt, 3,040 aggregate on four at
once. divergence.py (greedy decode against a fresh prefill of every prefix, 24 tokens): prose 23 / 24 at mean |dlogprob| 0.124,
code 24 / 24 (0.016), math 24 / 24 (0.045). Needles 3 / 3 at 8k, 16k and 28k. Strict sanity after 20-24k-token prefills: 0 bad
of 25 at 5 streams, 0 of 80 at 8 (max_num_seqs 4). Soak (8 clients, 200-28k-token prompts, 934 s): 174 requests, 0 errors, 1
over-length rejection by design, median prompt 14,846 tokens, 2,816 prompt tok/s, worst card 0.69 GiB free throughout. GSM8K
200: **195 / 200 = 97.5 %** (1 truncated) in 3.2 min. 0 server errors; 31.8-31.9 GiB a card used after.

**DeepSeek-V4.1-Flash TP8, the long-context recipe (`MAXLEN=131072 KVMEM=0.8` = 313,831 KV tokens; `validate-B*.sh`, 08:01-08:59
UTC):** needles 3 / 3 at 64k, 100k, 120k and 128.5k (126.7k-token prompts; prefill 3.4-3.6k tok/s, 35 s for the longest);
soaks of 2 clients with 60-100k-token prompts (316 s: 13 requests, 0 errors, 3,234 prompt tok/s) and 100-125k-token prompts
(213 s: 6 requests, 0 errors, 3,249 prompt tok/s); worst card 0.38 GiB free at the 128.5k needle and through the long soak.
`KVMEM=1.0` (392k tokens) also retrieves at 120k but leaves 0.21 GiB; 1.5 fails the drafter's graph capture.

**The other three configurations on the same tree and image (`~/ds41/exl3/gate-040.sh`, VM100):**

Same tree and image as the tag (`r9700/vllm:0.4.0rc` = the tagged `docker/Dockerfile` on `nightly-rocm100-e4a5b553`), overlay on,
nothing else on the box, 2026-11-03 09:00-11:01 UTC. Numbers in parentheses are v0.3.8's gate.

**Qwen3.8-27B-NVFP4 TP2 (cards 0,1, DFlash2 drafter):** probe 218.0 tok/s mean (code 238 / prose 105 / json 274 / math 256), 4.47
tokens a step, 513 / 617 tok/s at 8 / 16 streams, prefill 3,872 tok/s on 8k (217.6 / 4.48 / 516 / 643 / 3,877). BetterBench
combined **195.7** (198.2), update p99 24.5 ms, 1 / 2 / 4 / 8 streams 174.5 / 292.4 / 445.4 / 585.6 (176.9 / 296.4 / 437.2 / 595.5),
prompt processing 3.67-3.91k tok/s at 2k-32k (3.7-3.95k). Strict sanity: 0 bad of 160 after long prompts, 0 of 1,020 at 17
streams, 0 of 1,280 at 32. Soak (16 clients, 200-30k-token prompts, 942 s): 195 requests, 0 errors, median prompt 14.4k tokens,
3,038 prompt tok/s. 0 server errors.

**Qwen3.8-Flash-Next TP4 (cards 0-3, MTP-3):** KV cache 477,832 tokens (477,002). Probe 222.6 tok/s mean (221 / 160 / 261 / 248),
3.11 tokens a step, 469 / 915 tok/s at 8 / 16 streams, prefill 6,762 tok/s on 8k (221.4 / 3.09 / 470 / 924 / 6,743). BetterBench
combined **189.7** (184.6), update p99 15.2 ms, 1 / 2 / 4 / 8 streams 175.4 / 276.6 / 400.0 / 534.9 (174.0 / 276.3 / 402.4 / 535.6),
prompt processing 5.0 / 7.0 / 7.4 / 7.1k tok/s at 2k / 8k / 16k / 32k (4.96 / 6.95 / 7.31 / 7.07k). Strict sanity 0 bad of 160 /
1,020 / 1,280. Soak (16 clients, 907 s): 375 requests, 0 errors (1 over the 32k limit rejected, by design), median prompt 16.5k
tokens, 6,648 prompt tok/s. 0 server errors.

**GLM-5.3-Flash TP8 (32k production recipe: RedHat checkpoint, MTP-3, fp8 KV, NSEQ 4, NBT 8192; the eight-rank collective's
split path now starts at 32 KB):** KV cache 210,051 tokens (203,329 on the v0.3.8 deploy smoke). Decode-only 88.1 tok/s median
over three 512-token runs at 44 % draft acceptance (v0.3.7's same measure: 75.0; the v0.3.8 probe mean is a different tool),
four streams 139 then 177 tok/s aggregate (131 / 157 on v0.3.7's gate, 161 in v0.3.8's BetterBench), 12.7k needle correct
(ttft 5.1 s, prefill 2,495 tok/s, 117 tok/s at depth, 88 % acceptance). Strict sanity (thinking on): 0 bad of 160 after long
prompts, 0 of 300 at 5 streams, 0 of 320 at 8. Soak (16 clients, 1,002 s): 159 requests, 0 errors (1 over-length rejection),
median prompt 14.9k tokens, 2,476 prompt tok/s. 0 server errors.

## Known limits

- DeepSeek-V4.1-Flash: 32k context is the gated recipe; the 131072 launch (`MAXLEN=131072 KVMEM=0.8`) retrieves at 126.7k-token prompts (see above). Four
  streams at 128k do not fit the KV budget on 32 GB cards (the KV cache is replicated on every tensor-parallel rank; context
  parallel over the KV cache is the route to longer contexts and is not in this release). The first prompt of a process
  JIT-compiles four Triton kernels (~7-13 s; `WARM=1` in the launcher sends it). Prefix caching is off (the EXL3 experts'
  prefill tier and the sparse indexer were validated without it).
- Upstream: the RDNA decode-attention bug in stock's DeepSeek-V4 fallback is worked around here, not reported yet
  (notes/exl3-plan.md, bring-up log).
- GLM-5.3-Flash above ~75k prompt tokens on this nightly: unchanged from v0.3.7.
