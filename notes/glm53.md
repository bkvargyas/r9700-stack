# GLM-5.3-Flash on eight R9700s (glm53 branch, 2026-10-13/14)

Checkpoint: `coolbho3k/GLM-5.3-Flash-NVFP4-Optimized` (compressed-tensors; NVFP4 routed experts, FP8 weight-only
dense / shared-expert / sparse-attention projections, BF16 KDA projections, no MTP layer). Stock vLLM nightly
`fe4df3df`, stock ROCm 10, unmodified weights. TP=8 on four PLX switches, `serve/serve.sh` through VM100's
`~/glm-launch.sh`.

## What the plugin adds for this model

| piece | where | why |
|---|---|---|
| W8A16 fp8 linears -> W8A8 dynamic | `quant/ct.py get_scheme` | stock serves fp8 weight-only through CUDA Marlin only |
| indexer `is_enabled()` gate | `models/glm5next.py` (proxy) | the ROCm indexer refuses to run unless AITER (gfx9) is on; everything it calls on this model is Triton / in-tree |
| 8-rank all-reduce `ar8` | `comm/r9k_ar8.py` | the flat one-shot collapses at eight senders on four switches (126-288 us); pair / evens-cross / pair = 22 us |
| 8-rank all-gather `ag8` | `comm/r9k_ar8.py` | same for the logits gather (1.0-1.3 ms -> ~0.3) |
| indexer head weights | `attn/sparse_mla.py SkinnyWeight` + `r9k_mm_f32_skinny` | hipBLASLt/rocBLAS take ~200 us on the 1 x 4096 x 32 fp32 mm, 11 per step |
| KDA projections -> per-row fp8 | `models/glm5next.py` post-load swap (`R9K_FP8_LINEARS`) | the model builds them with `quant_config=None`; 4.6B params, 3.4 ms/step in bf16 |
| sparse MLA on an fp8 (or bf16) pool | `kernels/r9k_sparse_mla.hip`, `attn/sparse_mla.py`, three backend hooks | stock's fp8 path is CDNA assembly, its Triton path bf16-only, aiter's kernel gfx950-only |

The three sparse-MLA hooks: `forward_mqa` -> our kernel; `supports_quant_query_input = False` (the MLA layer would
otherwise hand over an fp8-pre-quantized query); `_use_rocm_sparse_triton` -> True (the metadata builder would
otherwise call `aiter.get_mla_metadata_v1`, a CDNA kernel, and the pending error surfaces at the next launch).

## Numbers (single stream, 512-token decode, temp 0)

| step | tok/s |
|---|---|
| first light, eager | 6.6 |
| HIP graphs (FULL_DECODE_ONLY; the model class has no torch.compile support) | 24-28 |
| two-shot all-reduce | 40-51 |
| ar8 | 42 (stable probe) |
| + skinny indexer mm + KDA fp8 + ag8 | 52.5 |
| fp8 KV cache (our sparse kernel) | 52.9, KV capacity 291k -> 436k tokens at 8k max-len |

Concurrency (256-token streams): 1 -> 44, 2 -> 76 aggregate, 4 -> 129 aggregate. Reference (djdeniro, forked
vLLM 0.24 + overlay, rfi quant): 34-39 single stream MTP off, 82-88 with MTP-3, 155 at four streams with MTP.

## The fabric cliff (measured, `tests/test_ar_nrank.py`, `tests/test_ar8.py`)

Four ranks on any two switches: one-shot 6.8-10 us. Eight ranks on four: 126-288 us. Four ranks on four switches
with the other four idle: 7.8-10 us. Eight concurrent cross-chain senders at >= 6 KB each: 140-250 us; at 4 KB:
8 us. Any switch-local phase adjacent to a both-quartets cross phase: 65-75 us. Two independent TP=4 groups, or
one TP=4 plus two TP=2 groups, do not interfere at all. So only the single 8-rank group needs the hierarchical
collectives, and only the even ranks ever send cross-chain.

## Knobs

`KVDTYPE=fp8`, `R9K_SPARSE_MLA=auto|r9k|stock`, `R9K_SPARSE_MLA_SPLITS` (8), `R9K_INDEXER_MM=0`, `R9K_AR8=0`,
`R9K_AR8_FP32=1` (exact, 45 us), `R9K_AG8=0`, `R9K_AG_MAX_KB` (2048 for the logits at 4 tokens),
`R9K_FP8_LINEARS=<regex>`, `R9K_DISABLE=glm5next`. Everything keys on world size 8, the glm5next module path or
checkpoint content the Qwen configurations do not have.

## Open

GSM8K / HumanEval gates on this configuration; longer contexts (the fp8 cache is for them); the remaining per-step
cost (profile after fp8 KV); MTP needs a checkpoint that ships the MTP layer (Red Hat's does) and a check of the
stock proposer on this nightly.
