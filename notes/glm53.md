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
| native NVFP4 routed experts | `quant/nvfp4.py prep_native_moe`, `moe/experts.py` (`nv`), `r9k_moe_nvfp4a8` | the load-time NVFP4->MXFP4 conversion is lossy (~1.55x the checkpoint's own weight error); `R9K_NVFP4_MOE=native` keeps the bits for VRAM-resident experts (opt-in: same speed single stream, 127 vs 129 at four streams, GSM8K 96 vs 97 of 100, MTP acceptance 78.3 vs 78.0%, but ~1.1 GB/card of KV: 322k vs 436k tokens at 8k) |
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
| MTP, 1 draft token (Red Hat's unmodified checkpoint, which ships the MTP layer) | 72.3, 75% draft acceptance |
| MTP 1 draft + fp8 KV | 75.2-75.9; four streams 108 aggregate (MTP off: 127-129), so MTP for one or two streams only |
| draft acceptance (1 draft, temp 0) | 72% on a story, 99.5% on a verbatim copy: the proposer path is right, the head is as good as the text is predictable |
| native NVFP4 experts (exact checkpoint bits, opt-in) | 52.8 MTP off, 74.1 MTP 1 draft; GSM8K 96/100 |
| MTP 3 drafts: in-graph draft metadata + 8 KB cross-stage slices + RCCL above 40 KB + local-argmax drafts | **79.6** (78.4 the run before), 2.3 tokens a step on a story; four streams 158 aggregate (second run), 124-129 first; GSM8K 97/100 in 1.6 min; verbatim copy 97.5% accepted |

Concurrency (256-token streams): 1 -> 44, 2 -> 76 aggregate, 4 -> 129 aggregate (native experts: 127).
Kernel level (one MoE layer at the TP8 shapes, both GEMMs + activation): native NVFP4 equals MXFP4 at one row,
is 6-9% slower at 4-256 rows (MT=1) and 37% slower at the MT=2 shape (8,192 routed rows); the exact e4m3 scales
cost ~1.1 GB a card of KV, and draft acceptance with MTP is the same (78.3% vs 78.0%): the default stays MXFP4,
`R9K_NVFP4_MOE=native` is the exact option. Reference (djdeniro, forked
vLLM 0.24 + overlay, rfi quant): 34-39 single stream MTP off, 82-88 with MTP-3, 155 at four streams with MTP.
Ours after the MTP round: 79.6 single stream with 3 drafts, 158 at four streams.

## MTP on this nightly (2026-10-14)

The engine runs the new GPU worker (`vllm/v1/worker/gpu/`) and its `TargetDependentARSpeculator`; the legacy
`EagleProposer` is not on the path. Draft forwards are full-graph captured (decode mode FULL gives the drafter
FULL_DECODE_ONLY). With two or more draft tokens every draft step runs inside ONE graph only when each drafter
attention builder declares `supports_draft_decode_metadata_update`; the ROCm sparse builder does not, so the server
rebuilt metadata eagerly between draft steps. `_install_draft_metadata_update` (glm5next.py) adds the refresh for
the Triton-shaped metadata our kernel serves: the token -> request map and `paged_kv_indptr = cumsum(min(seq_len,
topk))`, both over persistent buffers; `build()` is wrapped to keep the runner's `seq_lens` view. MTP=3: 71.3 -> 73.5
tok/s, acceptance unchanged (1.52 of 3). The drafter is the sparse-attention block only (no KDA), and the indexer
/ kpool builders already supported the update.

Where an MTP=3 step goes (profile, ~34 ms, GPU-bound): the 8-rank all-reduce at four rows is 32 KB a message, above
the 16 KB one-shot limit, and the cross-chain two-shot stage costs 80 us x ~99 = 8 ms; the pair stages 2.3 ms; the
logits all-gather (one per draft step) 2.5 ms; bf16 skinny GEMMs (the Red Hat checkpoint keeps shared experts,
dense MLPs and the MLA projections in bf16) 3.2 ms; MoE 3.5 ms. Draft acceptance itself is fine: 99.5% on a
verbatim copy, 72% on a story at temperature 0 (83% at 0.7 -- stochastic acceptance is looser than argmax equality).

Levers, measured (`tests/test_ar8.py`, us): at 32 KB ar8 96 (RCCL 117), at 64 KB 250 (RCCL 102); a raised one-shot
limit is worse (133 / 311); slicing the cross stage into 8 KB pieces gives 69 / 157 (`R9K_AR8_CROSS_KB`, default 8);
above 40 KB the 8-rank object now declines and RCCL runs the message (`R9K_AR8_MAX_KB`), which alone took the
four-stream MTP=3 aggregate from 87 to 116. `use_local_argmax_reduction` (speculative config; `SPEC_EXTRA` on the
serve.sh MTP branch, the GLM launch helper's default) replaces the per-draft-step logits gather with a per-shard
argmax. Together: 78.4 single stream, 158 at four streams. Not taken: per-row fp8 for the Red Hat checkpoint's
remaining bf16 projections (33 more linears: -6% single stream, lower acceptance, GSM8K 97) and a DMA-engine
all-reduce for 32 KB (the ar4 SDMA mode's flag kernels alone cost 20-28 us each).

## Long context (2026-10-19 to 10-22)

A 262,144-token launch fits (415k KV tokens with a fixed 2.7 GiB KV budget, `KVMEM=2.7`) and serves 236k-token
requests without faults. The fixed budget matters: at 0.94 utilisation a 131k request died in the stock ROCm
indexer's prefill, which materialises a `[step tokens, context]` logits buffer per step and found 0 bytes free.
Prefill: 2,100-2,300 tok/s at 2k-token steps, 3,100 at 8k steps (`NBT=8192`, worth keeping); 32k-token steps run
the cards out of memory in that same buffer. Decode at depth with MTP: 97 tok/s at 73k, 82 at 151k.

Retrieval was unreliable above roughly 12k tokens on this nightly: a unique needle sentence found at 3k and 6k,
missed from 12.7k on, identically with every piece of this stack switched off. The offline reproducer
(`tests/test_indexer_logits_r9k.py`) found the cause on the decode side. The indexer's K cache is written by vLLM's
kpool compress kernel in pages of 64 pooled entries laid out in 16-entry x 16-dim tiles. On gfx942/gfx950 stock reads
it back with aiter's pre-shuffle Gluon kernel; on every other GPU its wrapper falls through to aiter's stage-1
kernel, which reads a linear page. The scores it returns are scrambled: against the torch reference on the same
cache bytes, top-512 overlap 70% at 760 pools, 18% at 3.2k, 8% at 8k, 3% at 32k, and a planted needle is never the
argmax. The prefill side (gather, aiter prefill logits, top-k op) is exact. `attn/indexer_logits.py` is a Triton
kernel that reads the tiled layout and matches the reference to 0.1 on logits of 100-180 at every size up to 65k
pools, for one row and for MTP verify rows, in 92-100 us against the stock kernel's 153-158; `_install_paged_logits`
(glm5next.py) swaps it in where the wrapper would take the fall-through (`R9K_INDEXER_LOGITS=stock` restores).
With it, on a 262k launch, the unique needle is found at 67k, 150k and 216k (prefill 2.7-2.8k tok/s, decode at
depth 61-107 tok/s, acceptance 76-91%); the near-duplicate numbered needle holds to 35k and is hit-or-miss
beyond (the model reasons without concluding), which is the task's difficulty rather than a kernel fault.
GSM8K 95/100, decode probe 74.7 tok/s (the usual 70-80 band).

Stability at 32k: 16 streams, sanity 96/96 after 24k-token prefills (`THINK=1`, the strict check needs the
reasoning in its own field for this model), 15-minute mixed soak 0 errors, VRAM flat.

## Qwen gates on this tree (2026-10-22, `~/validate-glm53-qwen.sh`)

The production configurations must not move; they did not. 27B TP2: single-stream 219.4 tok/s (v0.3.3 ~221),
4.5 tokens a step, 515 / 585 at 8 / 16 streams, 8k prefill 3,830; 26-minute soak 286 requests 0 errors, VRAM flat;
strict sanity 0 bad of 2,900. Flash-Next TP4: 223.1 tok/s single-stream (v0.3.3 225.0), 3.11 tokens a step, 610 /
930 at 8 / 16 streams, 8k prefill 6,847; BetterBench combined 185.3 (v0.3.3 188.0), update p99 15.2 ms, prompt
processing 7.1-7.4k tok/s; 25-minute soak 806 requests 0 errors, VRAM flat; strict sanity 0 bad of 5,380. Comm tests
(ag_nrank, ar_nrank 4 ranks, ar_race) and every single-GPU kernel gate pass. No server errors, no MES timeouts.

## The fabric cliff (measured, `tests/test_ar_nrank.py`, `tests/test_ar8.py`)

Four ranks on any two switches: one-shot 6.8-10 us. Eight ranks on four: 126-288 us. Four ranks on four switches
with the other four idle: 7.8-10 us. Eight concurrent cross-chain senders at >= 6 KB each: 140-250 us; at 4 KB:
8 us. Any switch-local phase adjacent to a both-quartets cross phase: 65-75 us. Two independent TP=4 groups, or
one TP=4 plus two TP=2 groups, do not interfere at all. So only the single 8-rank group needs the hierarchical
collectives, and only the even ranks ever send cross-chain.

## Knobs

`KVDTYPE=fp8`, `R9K_SPARSE_MLA=auto|r9k|stock`, `R9K_SPARSE_MLA_SPLITS` (8), `R9K_INDEXER_MM=0`, `R9K_AR8=0`,
`R9K_AR8_FP32=1` (exact, 45 us), `R9K_AG8=0`, `R9K_AG_MAX_KB` (2048 for the logits at 4 tokens),
`R9K_FP8_LINEARS=<regex>`, `R9K_NVFP4_MOE=mxfp4|native` (default converts; native = exact bits for VRAM-resident experts, offloaded experts
always convert, the expert cache is MXFP4-only), `R9K_DISABLE=glm5next`. Everything keys on world size 8, the glm5next module path or
checkpoint content the Qwen configurations do not have.

## Open

GSM8K (100 questions, four streams, medium reasoning) on the fp8-KV configuration: 97/100, no truncations.
MTP works on this nightly with the stock proposer and the same hooks (no proposer patch, unlike the vLLM 0.24
overlay). Open: more draft tokens, fp8 KV together with MTP, longer contexts, the next profile, HumanEval, and the
QFN / 27B gates on the glm53 tree before any merge.
