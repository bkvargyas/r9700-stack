# Kernel inventory — tcclaviger/vllm:dev (Flash-Next on gfx1201)

Date: 2026-09-18. Image: `tcclaviger/vllm:dev` (Clav 28.06.1, vLLM fork of upstream `55c98e370a`, torch 2.11+rocm10.0, py3.14, LibR4D 0.5.0).
Full local flag/env inventory: `tcclaviger-dev-flags.txt` (`docker run --rm tcclaviger/vllm:dev --flags`).

## How expert offload actually works (from the image's own docs)
One cache space per GPU. ~1/3 holds the hottest experts (ranked by the checkpoint's
`model-expertprofile.safetensors` sidecar) + every shared expert: VRAM-only, never moved.
The rest is a swap region; cold experts live in **pinned host RAM behind UVA**. libr4d's
`moe_lru_*` kernels decide **on the device, every step**, which host experts sit in swap slots
(routing rank x recency; eviction = host read into the victim slot). **Every step runs one GEMM over
the slots and one GEMM directly over the host tensors** — i.e. cold experts are consumed zero-copy
over PCIe. => host->GPU link bandwidth is a first-order decode limiter.
TP only (planner rejects PP != 1: `vllm/model_executor/offloader/expert_plan.py:415`).

Measured link on the .100 box (PLX PEX 8747, Gen3 upstream): 14.0 GB/s per GPU H2D, 27.8 GB/s both
concurrently (each GPU on its own PLX chain). Cards link Gen5 x16 to the switch; switch uplink is 8 GT/s.

## Private compiled extensions and their toggles
| module | role | toggle (default on) | source |
|---|---|---|---|
| clav_attn | RDNA4 attention backend (priority ahead of TRITON_ATTN) | `CLAV_ATTN=0` | private |
| gdn_hip (/app/gdnhip AOT) | GDN conv1d + gated-delta-rule | `NO_AMD_GDN_HIP=1` / `CLAV_GDN=0` | private. NOTE: spec-decode (MTP) batches take the Triton path anyway |
| clav_hc, clav_conv1d, clav_pleconv, clav_silu_quant, clav_memcpy, clav_state_copy, clav_reshape_cache | glue kernels | `CLAV_<NAME>=0` | private |
| clav_ar_ext / clav_ag_ext | P2P all-reduce / all-gather | `CLAV_AR*`, `CLAV_AG=1` (AG default OFF) | private |
| ple_hip / ple_nvme / ple_rocr | PLE row fetch/transport | `VLLM_PLE_*`, `R4D_PLE=0` | private |
| fp8hip (libfp8hip_gemm.so) | FP8 GEMM | `VLLM_DISABLE_FP8HIP=1` | private |
| rdna4 scaled_mm | FP8 w8a8 | `VLLM_DISABLE_RDNA4_FP8_KERNEL=1` | private (python wrapper visible) |
| r4d.so (LibR4D 0.5.0 + private additions) | GEMMs, MoE, expert LRU, attention, GDN, AR | `VLLM_DISABLE_R4D_MXFP4=1`, `R4D_QSA=0`, `R4D_PLE=0`, `R4D_SELECT=0`, `R4D_AR_QUANT=0` | **partly public** (below) |
| _rocm_C | vLLM core ROCm ops | — | upstream vllm `55c98e370a` |
| master switch | | `DISABLE_ALL_CLAV=1` | |

## libr4d: public vs private
Public: https://codeberg.org/StillDeadcode/libr4d (v0.5.0 2026-08-25, no license file; 38 files).
- **Public**: gemm_{bf16_nt_m16,bf16_nt_m64,mxfp4a8_nt_m64,w4a16_nt_m64,w4a8_nt_m64}, quant_act_i8,
  ar_oneshot_2rank_{exact,wht6}, ar_{oneshot,twoshot}_Nrank_{exact,ti8} (N-rank "verified on TP4"),
  attn_{decode,paged,prefill}_h256_gqa6, attn_vit_h72, gdn_{chunk_scan,conv,gated_rmsnorm,kkt_solve,recurrent_update},
  dflash_conv. Build: `./build.sh` inside the image.
- **Private (only in tcclaviger's r4d.so)**: `gemm_moe_mxfp4a8_nt_b16_m64{,_block,_group,_mt1}` (MoE grouped GEMM),
  `moe_lru_{fused,gather,manage}` (device-side expert cache), `moe_route_softmax_bf16`,
  `attn_sparse_h256{,_bf16kv,_fp8kv}`, `attn_sparse_score_h128`, `attn_sparse_topk_expand`, `qsa_index_prep_h128` (QSA),
  `ple_{gather_,}dequant_i6g32_f16`, `gdn_conv_{prep,update}_w4_h128`, `dflash2_prep_*`,
  `{ar,ag}_{oneshot,twoshot}_{4,8}rank_*`.
  => The hard/valuable kernels for Flash-Next (MoE GEMM, expert LRU, sparse attention) are private.
  Public `gemm_mxfp4a8_nt_m64` is the natural base for our own MoE grouped GEMM.

## Related prior art: GGZ14/vllm-mxfp4 ("radiance")
https://github.com/GGZ14/vllm-mxfp4 (checkout on VM100 at ~/vllm-mxfp4, 92eed82 2026-09-15).
Runtime-patch approach (patch_*.py over an image) for Qwen3.8-27B MXFP4 on R9700; builds libr4d from source
pinned at b9e42ab + `r4d_radiance_extras.patch`. Has HIP sources (radiance_mxfp4_fp8.hip, radiance_escha.hip,
radiance_autoround.hip).
**TP=3 implemented** (TP3_PADDING_PLAN.md): zero-padded dummy heads (Megatron-style) so every sharded dim
divides by 3: Q 24->36, KV 4->6, vocab 248320->248448, runtime weight padder (radiance_tp3pad.py,
patch_tp3_pad.py), 3-rank exact AR kernel (libr4d extras rx6 + patch_ar_3rank.py). Validated on GSM8K
(TP2-padded 97.8% = stock). Gotchas they hit: MXFP4 decode GEMM needs per-rank K % 128; GQA ratio is part
of the trained weights (pad keeping GQA); drafter KV bytes/token must equal target's.
Upstream vLLM rejected head padding (vllm-project/vllm#11797).
