# Latent Sparse Attention (LSA) -- a study for r9700-stack (2026-11-04)

Brian: "check this repo out: https://github.com/Prefrontal-AI/latent-sparse-attention -- this might be some research that's
worth looking into building into our kernels". Study only: no code, no GPU runs. Sources: the repo at commit 97b82d6
(2026-11-03; README, docs/guide/models.md, docs/reference/notation.md, src/lsa/core, src/lsa/sglang, src/lsa/training/
distillation, configs/evaluation/accuracy/glm53_flash/), the paper arXiv 2610.42147 (HTML, sections 3, 4, App. E, F.1, F.2),
the `d-lsa-weights-v1` release listing, our notes (glm53.md, release-v0.3.7/0.3.8, exl3-plan.md, exl3-ledger.md,
exl3-serving.md) and our plugin (r9700_vllm/models/glm5next.py, deepseek_v41_rdna4.py, attn/indexer_logits.py, attn/sparse_mla.py)
with stock vLLM's indexer (vllm/models/glm5next/amd/sparse_indexer.py, v1/attention/ops/rocm_aiter_mla_sparse.py).
**(est.)** marks my estimates; everything else is quoted or measured.

## 0. Verdict up front

**Not worth building into our kernels now; at most a two-agent-day offline study after v0.4.2.** LSA's gains start where
our cards cannot go: the paper's own end-to-end numbers for GLM-5.3-Flash TP8 are 1.22x TTFT / 0.99x TPOT at 512k and
only become large at 10M (5.8x / 1.9x). At our 32k-262k the indexer is ~8-19 % of GLM prefill and ~4 % of a decode step
(est.), and the 75k-token OOM it is blamed for is fixable without changing what the model attends to. It does not apply
to DeepSeek-V4.1-Flash at all (section 3). Details and the deciding numbers in section 5.

## 1. What LSA computes, and what it replaces

**Data flow (one indexed layer, D-LSA Flat, paper section 3.1-3.3, Fig. 3; src/lsa/sglang/models/glm5next/indexer.py).**
GLM-5.3-Flash defaults from Table 2 and src/lsa/sglang/settings/design.py: d_idx 64, n_pool 4, bucket ~8 keys (cap 16),
cluster block = window 512, sink 4, top-k 4096 / top-p 1024 buckets, and on GLM an early budget of k 2048 / p 512 below
position 1M (lsa_flat_early_budget_tp8.yaml) -- so below 1M a query reads 4 + 2048 + 512 = **2,564 keys** (list width
4,612, padded with -1) against GLM's own 2,048.

Key side (written once per token, clustered once per 512-token block):
1. index key k_idx = normalize(latent @ W_K), latent = the 512-dim MLA key latent the KV pool already holds
   (`indexer.input: key_and_query`, the default; the plugin reads it from the bf16 KV buffer, `key_buffer()`).
2. pool: 4 consecutive index keys summed into one pool key.
3. when a 512-token block closes: Ward agglomerative clustering of its 128 pool keys into 64 buckets (mutual-cheapest
   merges in parallel rounds, cap 16 keys), Algorithm 1; buckets never change afterwards.
4. bucket cache per layer: centroid [m, 64] bf16, members [m, 16] int32 (-1 padded), size [m], paged 8,192 keys a page
   (notation.md "Buckets"; memory.py). ~196 B a bucket = ~24.5 B a token a layer **(est. from the shapes)**; the paper:
   246 MiB a layer at 10M against 331 MiB for GLM's own index (section 4.3) -- same 0.74 ratio.

Query side (every token, prefill and decode):
- A. index query q_idx = normalize(sum over heads of the absorbed queries (q_h @ W_UK_h, 512-dim) @ W_Q). Under
  `query_heads: all` (default) each TP rank sums its 8 heads and **one all-reduce of [tokens, 512]** completes the sum
  (paper F.2, Fig. 18) so every rank selects the same list; `rank` skips the all-reduce and lets ranks differ.
- B. score every closed bucket's centroid: s = c_b . q_idx (64 MACs a bucket = 8 MACs a key), keep the top-p buckets.
  At long context a fused GEMM-with-threshold-epilogue scan never writes the score matrix (Algorithm 3, F.1); **below
  32,768 buckets (~256k tokens) the shipped code writes S with one PyTorch matmul and takes a top-k** (F.1 "Flat prefill").
  Decode: one row per request, a 4-launch sample / threshold / scan / listing pass (Algorithm 4).
- C. expand the picked buckets best-first into key positions until k long-range keys, add sink + window -> J_i, one
  list shared by all heads, handed to the model's own sparse attention as its index list. "The attention kernel is
  unchanged" -- J_i is just a longer `index_topk`.

**What it replaces in GLM-5.3-Flash's DSA indexer** (stock: vllm/models/glm5next/amd/sparse_indexer.py
`sparse_attn_indexer_kpool`; our hooks in glm5next.py and attn/indexer_logits.py):

| GLM's own indexer (goes) | LSA (comes) |
|---|---|
| wq_b-indexer / wk / k_norm / weights_proj (16 heads x 128, our SkinnyWeight mm) and the kpool gate + APE | W_Q, W_K [512, 64] fp32 per indexed layer (2.9 MB for all 11) |
| indexer K cache: fp8 128-dim per pool of 4 tokens + scale (tiled 16x16 pages; our paged_mqa_logits reader) | bucket cache (centroids / members / sizes); plugin sets `index_kpool = 1` and frees the index keys |
| prefill: fp8_mqa_logits over [chunk rows x T/4 pools] (heads padded 16 -> 32 for aiter), torch top-512 pools -> 2,048 tokens | GEMM [rows x 64] x [64 x T/8 buckets] + top-p 512 + expansion; Ward clustering per block |
| decode: paged logits over every pool + top-k (our Triton kernel, 92-100 us a call) | one centroid scan + threshold top-p + expansion |
| nothing on the wire (indexer inputs replicated) | +1 all-reduce of [rows, 512] per indexed layer (`all`) |

What stays: the checkpoint's weights, the MLA latent KV cache and its layout, our r9k_sparse_mla attention kernel (it
reads `indices` + `paged_kv_indptr`, so variable list lengths already work), the MoE / KDA / everything else.
Prefill flops per (query, key) drop from 16 x 128 x 2 / 4 = 1,024 (2,048 with the head padding) to 16 -- **64-128x**.

**DeepSeek-V4.1-Flash** (indexer on 8 layers [2,8,14,20,24,28,32,36], 32 heads x 128, top-512 over *compressed* entries,
selection reused across layers, SWA 128, per-head attention sinks; exl3-plan.md "The checkpoint"): the LSA authors call
DeepSeek-V4 "out of reach" (docs/guide/models.md "What is out of reach": K = V, learnable sinks the read does not model,
interleaved RoPE, compressed entries on the key axis -- "Deciding what a window of 512 means when an entry is eight
tokens is the actual work, and it is not written"). No map, no adapter, no design. Nothing to build on today.

## 2. Where it would pay on our hardware and contexts

**The paper's speedups scaled to our lengths (est.).** One H200 layer, 32,768-query last chunk (section 4.3, Fig. 2):
the indexer is 98 % of sparse prefill and 89 % of decode at 10M ("a free indexer: 58x prefill, 8.9x decode"). With the
indexer linear in T per query and attention fixed by k, share(T) = T / (T + T0) with T0 ~ 204k (prefill) and ~1.26M
(decode). Implied per-layer indexer share and the *ceiling* if LSA cost nothing:

| context | prefill share / ceiling | decode share / ceiling | paper measured |
|---|---|---|---|
| 32k | 14 % / 1.16x | 2.5 % / 1.03x | -- |
| 128k | 39 % / 1.63x | 9 % / 1.10x | -- |
| 256k | 56 % / 2.25x | 17 % / 1.20x | -- |
| 512k | 71 % / 3.5x | 29 % / 1.4x | per layer 2.1x prefill; end-to-end GLM TP8: **TTFT 1.22x, TPOT 0.99x** (Table 3) |
| 1M / 10M | | | TP8 1.51x / 5.83x TTFT, 1.06x / 1.91x TPOT |

A layer is one of 11 indexed layers in 45, so end to end at <= 256k the paper's own trend says ~1.0-1.1x TTFT and no
decode gain **(est.)**; the paper itself: "Below 1M tokens, the gain is smaller ... as the scan of every block is still cheap."

**Our measured shares (GLM-5.3-Flash TP8, v0.3.8 recipe).**
- Prefill: prompt processing 3,361 / 3,289 / 3,140 / 3,102 tok/s at 2k / 8k / 16k / 32k (release-v0.3.8.md), i.e. 297.5 ->
  322.4 us a token; attention reads a fixed 2,048 keys, so the +25 us is the indexer's linear growth: **~8 % of prefill at
  32k (est.)**. The 262k launch on v0.3.5 prefilled 2.7-2.8k tok/s at 67k-216k (glm53.md "Long context"): ~370 us a token,
  **~19 % at 216k, ~12-15 % at 128k (est., interpolated)**. A free indexer would save that much TTFT; LSA's short-context
  path (GEMM + top-k over half the columns, plus Ward) would recover perhaps half to three quarters of it **(est.)**.
- Decode: our tiled paged-logits kernel is 92-100 us a call flat from 760 to 65k pools (3k to 262k tokens; release-v0.3.5.md),
  i.e. launch-bound, not context-bound: 11 calls = ~1.0-1.1 ms of the 28.5 ms step (~4 %), plus the torch decode top-k
  (unmeasured). LSA decode is 4-5 launches + expansion a layer (35 us a layer at 1M on H200 in CuTe; Triton 2.5x slower,
  F.1), so on our launch-bound sizes it saves <= 0.5 ms and adds the 11 all-reduces (~15-20 us each on ar8, ~0.2 ms) and
  +25 % keys read by attention (2,564 vs 2,048; attention is 0.3 ms a step): **net decode change ~0 (est.)**.
- DeepSeek-V4.1-Flash for comparison (exl3-ledger.md): indexer 100-148 us of a 25.4 ms step (0.5 %), decode attention
  0.86-0.95 ms; prefill 3.4k -> 3.6k tok/s from 63k to 118k (exl3-serving.md 5b), i.e. the indexer is invisible in its
  prefill throughput. Its caller-quoted "~1.1 ms at 32k" is attention + indexer measured on short prose under the 32k
  launcher, not at depth.

**Memory.**
- Index storage: GLM's index cache is ~33 B a token a layer (132 B per pool of 4) = 363 B a token over 11 layers; LSA's
  bucket cache ~270 B: **saves ~93 B a token = ~39 MB at the 262k launch's 415k KV tokens (est.)** -- negligible.
- Prefill transient: GLM's logits are [chunk rows x T/4] fp32 = 2 GiB for an 8,192-token step at 262k uncapped; stock
  chunks them to `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (512 MiB default; v1/attention/backends/mla/indexer.py
  `_split_indexer_prefill_chunks`), and our torch top-k (glm5next.py `_topk_rows`: masked_fill copy + two bool
  [rows, n] masks) adds ~1.5-2x on top **(est., from the code)**. LSA's short-context path writes [rows x T/8] fp32 (half);
  only the long-context scan kernel removes it (lists 3p x 8 B + J width x 4 B = ~43 KB a query, ~0.35 GiB for 8,192 rows).
  So LSA would free ~0.5-1.5 GiB at 262k -- the same order as the 1.5 GiB lost to non-PyTorch allocations on this nightly
  (release-v0.3.7.md Known limits) -- but a lower `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` plus an in-place top-k frees the same
  without changing the selection. DeepSeek's 128k recipe: the workspace grows ~0.7 GiB from 32k to 128k (exl3-serving.md
  section 3); same remedy; LSA does not apply there.
- KV layout: unchanged (LSA reads the latent pool). Our pool is fp8 while the released plugin refuses a non-bf16 cache
  (`key_buffer()` NotImplementedError); we would project at KV-write time from the bf16 latent before quantization.
- Collectives: +11 all-reduces a step (`query_heads: all`; decode [<= 16 rows, 512] = 8-16 KB, our one-kernel ar8 path) or
  none (`rank`, untested by the paper). Prefill: [8,192, 512] fp32 = 16 MB a layer through RCCL, < 1 % of a step (est.).

## 3. What it costs

**Released maps** (`d-lsa-weights-v1`): `glm53_flash__long-16k-1M-heavy-d2048-lr3e-4-latent.pt` (2,891,989 B, the paper's
D-LSA) and `...-key-and-query-pca-step0.pt` (2,892,221 B, Training-Free), plus a 23 MB `x`-input research map and two
Qwen3.8-27B maps. Format (src/lsa/sglang/index/weights.py): a flat torch dict `<layer>.W_Q_idx` / `<layer>.W_K_idx`, each
[512, 64] fp32, loaded `weights_only=True`, 11 GLM layers (the layers that run their own top-k). Licence: the repo is
Apache-2.0; the release names no separate licence. Built from `zai-org/GLM-5.3-Flash` revision eb9eb208 (assets/models/
glm53_flash.yaml).
- RedHat NVFP4 GLM: the maps read only the MLA key latent (kv_a + kv_a_layernorm) and the heads' absorbed queries
  (q path + W_UK of kv_b), which the Red Hat checkpoint keeps in bf16 (glm53.md "Where an MTP=3 step goes"); if those
  tensors are the zai fp8 weights dequantized, the latents match to rounding and the map applies. Unverified: needs the
  base revision checked and a latent-agreement test. Neither map uses the checkpoint's own indexer weights.
- EXL3 DeepSeek pack: no map, no design (section 1).

**Training-free PCA map** (paper 3.3; src/lsa/training/distillation/model.py `IndexMap`, trainer.py `build_maps`,
data/native.py `latent_bases`): per indexed layer, 8,192 key-latent rows spread over 8 documents (`Space.INIT_ROWS`,
`Pool.INIT_DOCUMENTS`), carried into the basis L with L L^T = sum_h K_h^T K_h (from the layer's W_UK), row-normalized,
leading 64 directions P; served W_K = L P, W_Q = L^-T P. We could compute it on our box from one prefill of eight long
documents with a latent-dump hook (minutes of GPU, a 8,192 x 512 SVD per layer) -- or just download the released one.

**Accuracy.** RULER 13-task mean for GLM D-LSA (Flat, early budget): 0.973 at 4k, 0.968 at 32k (README; paper 0.973 /
0.968). The native baseline at those lengths appears only in plotted Figures 5 / 11; the text: D-LSA (Flat) "within 3
points of baseline sparse attention up to 1M", trails "by up to 8 points at 2M to 5M"; single-needle retrieval "stays at
95 or above for every method up to 5M"; MultiKey 2/3 and CWE drop for every method, baselines included (App. E.4).
Training-Free "trails D-LSA (Flat) on GLM-5.3-Flash" (4.2; no number in the text), and is within the CIs on knowledge /
reasoning (Table 7: MMLU-Pro +0.4, GPQA +2.0, AIME25 -2.5, MATH500 +1.0, LongBench v2 +1.7 vs native). No sparse
training of the base model was done.

**Our rule** (memory: no vLLM / ROCm / model modifications, unmodified checkpoints). The map file is a load-time artifact,
not a checkpoint edit, and the hooks would be plugin hooks. But it *replaces the model's own trained indexer with a
third-party learned one* and changes outputs -- closer to the MTP graft Brian rejected on 2026-10-13 than to a kernel swap.
Needs Brian's explicit call. If ever built: opt-in only (`R9K_GLM_LSA=off|pca|dlsa`, default off), never in the default
recipe, gated against native on the same tree: divergence.py agreement, needles to 216k, RULER 13 tasks at 4k / 32k /
128k, GSM8K 1,319 paired (McNemar), HumanEval, plus the release-validation stress / soak (feedback_release_validation).

## 4. Build plan (only if the study in 4.3 says go)

**4.1 Hook.** Stock's `sparse_attn_indexer_kpool` (vllm/models/glm5next/amd/sparse_indexer.py) fills
`topk_indices_buffer`, which our `forward_mqa` hook (glm5next.py `_install_sparse_mla`) turns into global slots via
`triton_convert_req_index_to_global_index` and `paged_kv_indptr = cumsum(min(seq_len, topk))`. Replace the indexer op
(the same place `_install_paged_logits` swaps the decode reader) with an LSA op writing the same buffer; set
`index_topk` to the list width at config load (what the sglang plugin does on `hf_text_config`) or keep 2,048 total
(4 + 512 + 1,532) to leave buffers and attention cost untouched (accuracy untested). The KDA layers and the 34 other
layers are untouched; the DeepSeek hooks (`_install_indexer_gate`, `_install_prefill_logits`, split-K decode) are not involved.

**4.2 Kernels (HIP / Triton on gfx1201).**
1. KV-write side: project the bf16 latent to 64 dims, normalize, sum pools of 4 (one small fused kernel next to the KV insert).
2. Ward clustering of a closed 512-token block (128 pools -> 64 buckets, sequential rounds, latency-bound): start from their
   Triton `core/kernels/triton/ward.py`, later one HIP workgroup per block in LDS. Runs every 512 tokens a layer.
3. Prefill selection (<= 256k = <= 32k buckets): hipBLASLt / WMMA GEMM [rows x 64] x [64 x buckets] + a real row top-512
   (not the torch masked_fill topk) + expansion through the members table; row-chunked so the transient stays < 256 MiB.
4. Decode selection: one fused scan of a request's centroids (<= 32k x 128 B = 4 MB a layer at 256k, ~7 us at 640 GB/s),
   in-LDS top-p, expansion -- one or two launches; and the summed-query all-reduce.
5. Memory accounting: bucket cache counted into the KV budget like memory.py does for sglang.

**4.3 HIP graphs.** Our GLM prefill already runs without graphs (FULL_DECODE_ONLY; no torch.compile support for the
model class), so the paper's `--disable-prefill-cuda-graph` costs nothing new, and prefill's host readback of list sizes
can be replaced by fixed-capacity lists (3p) with a device-side fallback flag anyway. Decode: at <= 256k a single graph
sized for max_model_len reads at most 4 MB a layer, so the paper's per-context-range graphs are unnecessary; the
selection runs on device inside the captured step. Clustering a closed block runs before replay in the metadata builder
(host-launched, device-executed, once every 512 tokens a request); under MTP-3 the draft-metadata update hook
(`_install_draft_metadata_update`) has to learn the new list lengths.

**4.4 Size (est.).** E0 + E1 below: 2 agent-days. Then vLLM integration and eager path 4-5, Ward + selection kernels on
gfx1201 3-4, graph-safe decode + MTP metadata 2, gates 2-3: **~13-16 agent-days**, plus Brian's sign-off on the rule.

**4.5 The first experiments that prove or kill it.**
- E0 (0.5 day, one GPU run): rocprof the production GLM at 32k, 128k and 216k prompts on a 262k launch; sum
  indexer kernels (k-gather, fp8_mqa_logits, top-k, paged logits) by phase. Kill if the indexer is < 15 % of TTFT at 128k
  (my estimate says 12-15 %).
- E1 (1.5 days, offline): dump for the 11 indexed layers the key latents, summed absorbed queries, GLM's own top-2,048
  lists and the dense head-averaged attention mass, on ~20 RULER-style needle / multikey prompts at 32k and 128k; run the
  repo's torch reference (`lsa.core.reference.build_buckets` / `read`) with the released PCA and D-LSA maps; compare
  attention-mass recall and needle-key hit rate of J_i against GLM's own lists. Kill if PCA or D-LSA recall < native at
  either length; only then an end-to-end needle / RULER run on an opt-in server.

## 5. Recommendation

**Not worth building now; park it. Optionally run E0 + E1 (2 agent-days) after v0.4.2 if long-context GLM becomes a goal.**
The numbers that decide it:
1. **The indexer's share at our lengths is small:** ~8 % of GLM prefill at 32k, ~19 % at 216k, ~4 % of a decode step, and
   0.5 % of DeepSeek-V4.1's step (est. from release-v0.3.8 / glm53.md / exl3-ledger.md). The paper's own end-to-end GLM TP8
   gain at 512k is 1.22x TTFT and 0.99x TPOT; our best case at <= 256k is ~1.1x TTFT and no decode gain.
2. **Our ceiling is 262k (415k KV tokens over eight 32 GB cards, KV replicated per TP rank)** -- below the 512k-1M where
   LSA starts to pay. The win grows with context we cannot hold.
3. **The memory problem has a cheaper fix:** the 75k-token OOM is ~1.5 GiB of non-PyTorch allocations meeting the indexer's
   prefill transient; lowering `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (512 default) and an in-place `_topk_rows` recover that
   order of memory with bit-identical selection. That is the v0.4.2-class item, not LSA.
Plus two blockers independent of speed: it changes outputs by replacing the model's trained indexer with a third-party
map (Brian's call under the unmodified-checkpoint rule), and it does not exist for DeepSeek-V4 at all.
