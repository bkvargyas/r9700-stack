# DeepSeek-V4.1 decode glue fusion (Phase 4 package B, 2026-11-03)

Brief: notes/exl3-plan.md, Phase 4, package B. Kernels: kernels/r9k_dsv4_glue.hip; Python + hooks: r9700_vllm/dsv4_glue.py;
hook registration: r9700_vllm/models/deepseek_v41_rdna4.py (`_install_route`, `_install_qb_stack`); test:
tests/test_dsv4_glue.py. Logs: VM100 ~/ds41/exl3/B4/.

## Per-layer decode launch sequence (derived from the code; Z's ledger to reconcile)

Pin facts that set the list (server log of 2026-11-03 11:07): compilation mode NONE (the V4.1 model is not
torch.compiled: every torch op is its own launch, captured by the breakable cudagraph), V2 model runner (the narrow eager
region: only `_sparse_indexer_and_attn` runs eagerly at replay), IR op priority `rms_norm = [vllm_c, native]` (RMSNorm is
one `_C` launch), attention aux streams off (`serialize_attention`), shared experts on the MoE aux stream.

One main layer at T tokens (T <= 24), with the Phase 2-4 tiers (H seam, D GEMV, A expert kernel, C collective):

| # | step | launches | owner / note |
|---|---|---|---|
| 1 | attention seam (post of l-1 folded into pre of l) | 1 | H |
| 2 | attn_norm (vllm_c rms_norm) | 1 | stock; foldable into H's seam tail (proposal below) |
| 3 | fused_wqa_wkv (wq_a + wkv) | 1 (D) / 5 (exllamav3: copy, had_in, dot, 1-2 casts) | D |
| 4 | compressor kv_score mm (layers 2,8,14,20) / indexer weights_proj (8 layers) | 1 / 1 | stock hipBLAS |
| 5 | q_norm + kv_norm (fused_q_kv_rmsnorm, Triton) | 1 | stock |
| 6 | wq_b | 1 (D) / 4 | D |
| 7 | fused_qnorm_rope_kv_rope_quant_insert | 1 | stock |
| 8 | compressor + indexer K store / indexer wq_b + q rope/quant (12 layers) | 3-6 | stock; indexer wq_b 1 (D) |
| 9 | sparse indexer + decode attention (eager break; split-K partial + reduce) | 2 + indexer | P3 |
| 10 | wo_a (+ the o slice copy) | 1-2 | D |
| 11 | wo_b + all-reduce | 1 + 1 | D, C |
| 12 | FFN seam | 1 | H |
| 13 | ffn_norm | 1 | **B: folded into 15** |
| 14 | gate GEMM (hipBLASLt bf16 -> f32) | 1 | **B: folded into 15** |
| 15 | topk_softplus_sqrt | 1 | **B: r9k_dsv4_route** |
| 16 | shared expert gate_up (hipBLAS bf16) | 1 | **B: folded into 15** |
| 17 | silu_and_mul_with_clamp | 1 | **B: folded into 15** |
| 18 | shared expert down | 1 | stock |
| 19 | routed experts | 2 | A |
| 20 | combine (shared + routed partials) + all-reduce | 1-3 | C |

## What package B changed (ranked by the ledger's measured time; stop rule: < 20 us a layer left)

The ledger (notes/exl3-ledger.md) moved the premise: the step is GPU-bound (launch + CPU gaps <= 1.9 ms of 25.4), so a
fusion pays in kernel time, not in the node floor. The casts around the EXL3 GEMVs (~19 us a layer, the largest glue
item) are removed by D's GEMV (bf16 in / out, rotation in-kernel); after D and B the remaining per-layer glue is
attn_norm 1.9 + q/kv norm 1.5 + the MoE add 1.4 (C's) us -- below the stop rule.

1. **FFN front, one launch** (`r9k_dsv4_sk`, hook `_RouteHook`; R9K_DSV41_ROUTE=r9k, R9K_DSV41_SHARED_FUSE=1,
   R9K_DSV41_ROUTE_MAX_M=8): ffn_norm + router gate GEMV [5120 -> 384 (drafter 128)] + sqrt-softplus top-k (bias,
   bias_vl sentinel rows, is_padding, renormalize, routed scaling) + the shared expert's gate_up GEMV [5120 -> 2 x 288]
   + silu_and_mul_with_clamp. Ledger rows 22-24, 26, 27 of a plain layer (1.9 + 17.5 + 1.4 + 13.0 + 8.3 = 42 us,
   5 launches) -> 1 launch. Split-K over 512-column slices x 64-row groups (150 blocks), the normalized rows staged in
   LDS, 4 rows x 16-byte interleaved chunks a thread (bf16 dot2), ticketed fixed-order slice sums, the top-k in the last
   router group. Above 8 rows every call takes stock's ops (the kernel loses from 12 rows).
2. **DSpark Markov head** (greedy drafts; `r9k_dsv4_markov_step`, hook on DSparkDeepseekV4ForCausalLM.sample_sequential;
   R9K_DSV41_MARKOV=r9k, R9K_DSV41_MARKOV_FP8=0): one launch a draft step (previous token from the last step's block
   maxima, Markov embedding, bias GEMV over [129280 x 256], + base logits, block argmax) + one final launch, instead of
   embedding + wvSplitK + add + argmax + copy per step. The adaptive-verification estimator gets each step's logits.
   R9K_DSV41_MARKOV_FP8=1 reads an e4m3 copy of the head (per-row scale): faster at 1 row, slower at 4 (decode cost).
3. **Attention input, one launch** on the 8 compressor / indexer layers (`r9k_dsv4_sk`, `_AttnInHook`;
   R9K_DSV41_ATTN_IN=r9k, R9K_DSV41_ATTN_IN_MAX_M=6): attn_norm + compressor kv_score [5120 -> 1024 | 512, f32] +
   indexer weights_proj [5120 -> 32, bf16] (ledger: 23.8 + 20.1 + 1.9 us, 3 launches on layers 2/8/14/20; 2 on 24-36).
4. **wq_b + indexer.wq_b stacked** (R9K_DSV41_QB_STACK=1; needs D's r9k_exl3_gemv): one GEMV launch on indexer layers.
   Not timed here (D's kernel); bit-identical outputs by construction (same kernel, per-part arithmetic).
5. `r9k_dsv4_norm_pre` (RMSNorm + cast + optional exllamav3 rotation; D's `rotated=True` interface) -- not bound into
   the model: with D's GEMV the casts it would fold are gone.

## Results (tests/test_dsv4_glue.py, card 6, clean tree = git archive v041 + B's files; 116 checks, ALL OK)

Graph replay, us a call (the weights repeat across the 20 captured calls, so both sides read them warm from the
64 MB infinity cache; the ledger's cold numbers for stock are ~10 % higher):

| rows | ffn front stock (5 launches) | ffn front r9k (1) | attn input stock (3) | attn input r9k (1) |
|---:|---:|---:|---:|---:|
| 1 | 39.7 | 21.0 | 29.1 | 13.4 |
| 2 | 38.6 | 22.2 | 27.4 | 14.8 |
| 4 | 39.3 | 25.2 | 28.3 | 18.9 |
| 6 | 35.8 | 27.1 | 25.1 | 22.7 |
| 8 | 40.2 | 32.3 | 26.0 | 26.3 (-> stock) |
| 12 | 37.1 | 44.4 (-> stock) | 26.1 | 35.2 (-> stock) |
| 24 | 34.5 | 95.7 (-> stock) | 25.6 | 96.0 (-> stock) |

Markov head, five greedy steps: 1 request 720 -> 245 us (fp8 head 158); 4 requests 747 -> 313 us (fp8 394).

Numerics: normalized rows vs stock's `_C::rms_norm` within 1 bf16 ulp (1 of 163,840 elements at 32 rows; the hooked
module test: 0 differ); router top-k bit-exact given the same logits (ids and weights, incl. bias_vl rows, padding,
int64 ids, the drafter's 128 x top-3); end to end logits within 2e-6 relative of hipBLASLt, identical expert sets on
every tested row; shared-expert activation within 1-2 bf16 ulp of hipBLAS + `_C` (rel. 2.4-4e-3, the GEMM's own order);
kv_score within 2.3e-6 relative, weights_proj bit-exact; Markov drafts token-identical to stock's loop at 1, 2, 4 and
8 requests (step-0 logits: 0 / 0 / 1 / 9 elements a bf16 ulp or a few apart from wvSplitK's sum order); graph replay
== eager bitwise.

Per step (T = 6 verify, single stream, 40 target + 3 drafter layers):
- launches: FFN front -4 a layer (-5 on drafter layers: its router gate is two GEMMs) = -175; attention input -12;
  Markov -19 (25 -> 6); wq_b stack -8 (with D) = **~-214 launches a step** (of 1,754).
- GPU time (warm bench deltas; adaptive verification runs 2-4 rows on 97 % of single-stream prose steps): FFN front
  -14 to -17 us a layer at 1-4 rows, -9 at 6 -> 0.4-0.7 ms; attention input -0.1 ms; Markov head -0.45 to -0.48 ms:
  **~1.0-1.3 ms a step**.
