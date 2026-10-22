# r9700-stack v0.3.4

**GLM-5.3-Flash runs on eight R9700s from the same plugin, on the same stock nightly as v0.3.3: 79.6 tok/s single
stream with three MTP drafts, 158 tok/s aggregate at four streams, GSM8K 97/100, 262k-token launches.** The Qwen
configurations are unchanged: 27B TP2 and Flash-Next TP4 re-validated on this exact tree (`notes/glm53.md`).

## What changed for someone running it

- `serve/serve.sh`: `SERVED_NAME`, `KVDTYPE=fp8`, and `SPEC_EXTRA` (extra speculative-config fields) on the MTP
  branch. GLM-5.3-Flash launch: `MODEL=/models/GLM-5.3-Flash-NVFP4-RedHat TP=8 OFFLOAD_GB=0 MTP=3 KVDTYPE=fp8
  NBT=8192 VLLM_ROCM_USE_AITER=1 R9K_AG_MAX_KB=2048 SPEC_EXTRA='"use_local_argmax_reduction": true'
  R9K_FP8_LINEARS='self_attn\.(in_proj_qkvbfg_a|o_proj|f_b_proj|g_b_proj)$'`; for contexts above 32k give the
  KV cache a fixed budget (`KVMEM`) so the stock indexer's prefill buffer keeps headroom.
- New for this model (`r9700_vllm/models/glm5next.py`, all scoped to the glm5next module path and world size 8):
  the indexer AITER gate, the sparse-MLA kernel over an fp8 or bf16 pool (`kernels/r9k_sparse_mla.hip`), the
  skinny fp32 indexer matmul, post-load per-row fp8 for the KDA projections, the in-graph draft metadata refresh
  that lets the speculator run all draft steps in one graph, and the hierarchical 8-rank all-reduce / all-gather
  (`comm/r9k_ar8.py`: 8 KB cross-stage slices, RCCL above 40 KB).
- NVFP4 MoE experts can be served natively (`R9K_NVFP4_MOE=native`, exact checkpoint bits); the default stays the
  MXFP4 conversion (measured: same speed and accuracy, 1.1 GB a card of KV).
- `bench/sanity_stress.py THINK=1` for reasoning models whose template narrates into the content field.

## Validation

27B TP2: 219.4 tok/s single stream (v0.3.3 ~221), 515 / 585 at 8 / 16 streams, 8k prefill 3,830; soak 286
requests 0 errors; strict sanity 0 bad of 2,900. Flash-Next TP4: 223.1 tok/s (v0.3.3 225.0), BetterBench combined
185.3 (188.0), 610 / 930 at 8 / 16 streams, 8k prefill 6,847; soak 806 requests 0 errors; strict sanity 0 bad of
5,380. GLM-5.3-Flash TP8 at 32k: 16 streams, sanity 96/96 after 24k prefills, 15-minute soak 0 errors, VRAM flat.
Comm tests and every single-GPU kernel gate pass. Image pin unchanged (vLLM nightly fe4df3df, ROCm 10).

## Known

- GLM-5.3-Flash retrieval above roughly 12k-21k tokens is unreliable on this nightly, identically with every
  piece of this stack switched off: the stock GLM long-context path on ROCm, not ours (`notes/glm53.md`).
- Eight cards at the 210 W cap draw ~2,375 W at the wall during prefill; on a 120 V feed with 2+0 supplies that
  is the limit, and the host's protection parks the CPU at 400 MHz until a power cycle. A −500 MHz clock offset
  holds the peak at ~2,175 W; a 208/240 V feed is the real fix.
