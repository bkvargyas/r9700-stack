# r9700-stack — plan

Goal: an optimized **gfx1201 (Radeon AI PRO R9700)** kernel + plugin package that runs on **stock upstream vLLM on
ROCm 10** (no fork), first target Qwen3.8-Flash-Next (MXFP4 experts, expert offload to host RAM), second the dense
Qwen3.8-27B. Built so the hardware layer is ours and upstream bumps are a re-validate, not a rebase.

## Target base
- `vllm/vllm-openai-rocm:nightly-rocm100-dee37d89115db4c94a820a79a78a7828e141c910` (2026-09-18): vLLM main @dee37d89
  (post-0.29.0), torch 2.12.0+rocm10.0.0 (TheRock), triton 3.8.0 rocm10, ROCm SDK 10.0.0 (rocm-systems 6b0e43f —
  same commit our RCCL P2P rebuild uses), py3.12, hipcc/cmake/ninja/pybind11 present, flash_attn 2.8.3 importable,
  amd-aiter 0.1.21. Pin by digest; plugin also tracks v0.29.0 (release) for API reference.
- Upstream moves fast (1506 files differ 0.29.0 -> nightly in 9 days): **pin, and re-validate per bump**.

## What stock vLLM lacks on gfx1201 for Flash-Next (notes/upstream-0.29-rocm10.md)
1. CT MXFP4 MoE -> MarlinExperts (CUDA-only): **fails to load on ROCm.**
2. AMD PLE loader bf16-only (our int6/FP8 PLE fails / mis-scaled; PR #55040), no PLE offload (PR #57497 open).
3. No expert offload / expert cache at all.
4. QSA: BF16 KV only, Triton scalar indexer (slow), hard-wired backend (bypasses selector).
5. MTP k>1 on ROCm: QSA metadata not on spec-decode allowlist (PR #55292).
6. Custom all-reduce only gfx94/95 -> RCCL; GDN on generic FLA Triton.

## Building blocks (licenses!)
| source | what we take | license |
|---|---|---|
| davetha/r9700-lru-expert-cache | device-side LRU expert cache HIP kernels (+tests), cold_gather, fused silu-mul-quant, fused gate-mul, FP8 target lm_head, W4 draft lm_head, profiling tools, hot profile | **Apache-2.0** |
| StillDeadcode/libr4d (codeberg) | gfx1201 GEMMs (mxfp4a8/w4a8/w4a16/bf16), 2/N-rank P2P all-reduce, GDN, attention h256 | **no license — ask author** |
| GGZ14/vllm-mxfp4 | patch ideas (most apply to 0.29), TP=3 padding design, 3-rank AR | **no license — ask author** (escha/ is MIT, DFlash backport Apache) |
| tcclaviger/vllm:dev | reference behaviour + expert-offload algorithm spec (notes/tcclaviger-fork-analysis.md) | private image; re-implement, don't copy |
| upstream open PRs | #57497 AMD PLE UVA, #55040 FP8 PLE scale, #55292 ROCm MTP allowlist, #56005 RDNA4 FP8 FlyDSL, #54972/#57472/#54610 UVA fixes, #46676 RDNA MXFP4 MoE, #55917 RDNA4 AR | Apache-2.0 |

## Package layout (to build)
```
r9700_vllm/                 python package, entry points vllm.general_plugins (+ optional vllm.platform_plugins)
  __init__.py               register(): model override, quant override, monkeypatches (idempotent, version-gated)
  platform.py               R9700Platform(RocmPlatform): P2P communicator, cudagraph/spec defaults
  models/qwen4_exp.py       subclasses of upstream Qwen4Exp{ForConditionalGeneration,ForCausalLM,MTP}: PLE loader
                            (int6/FP8 fused-row), QSA fp8-KV impl, low_latency_gemm hook
  quant/ct_mxfp4_moe.py     CompressedTensors MXFP4 MoE method -> R9700Mxfp4Experts (no Marlin repack)
  moe/experts.py            FusedMoEExperts subclass (grouped MXFP4 GEMM)
  moe/expert_cache.py       offloader + planner + LRU cache (spec: fork analysis §2)
  ple/offload.py            host/UVA PLE + int6 dequant-gather
  ops/                      torch.library("r9700") op registrations + fake impls
kernels/                    HIP sources -> libr9700.so (built in the base image with hipcc for gfx1201, NDEBUG,
                            no device asserts => no hostcall => P2P-safe)
docker/Dockerfile           FROM pinned nightly-rocm100; pip install plugin + kernels; drop in P2P RCCL
bench/ ablation/ profiling/ notes/
```

## Phases
- **P0 (done/running)**: baseline tooling, P2P A/B (P2P ≈ no gain at TP2), kernel ablation of tcclaviger:dev,
  rocprofv3 decode profile, research notes.
- **P1 stack bring-up**: nightly-rocm100 on the R9700 box — smoke test a small model, P2P RCCL drop-in check,
  build libr4d HEAD + davetha LRU kernels inside the image, unit tests on gfx1201.
- **P2 plugin skeleton + load**: entry point, Qwen4Exp override, CT MXFP4 MoE method using a first working experts
  path (candidate: OAI triton_kernels matmul_ogs, which claims gfx1x support, or dequant->Triton), PLE loader for
  the int6 GPTQ checkpoint with `--cpu-offload-params ngram_embedding` UVA. Goal: *correct* output on stock vLLM
  (needs expert offload to fit: P4 may have to come first or use `--cpu-offload-params experts` UVA as the stopgap).
- **P3 MoE GEMM**: grouped MXFP4xFP8 GEMM for gfx1201 from libr4d `gemm_mxfp4a8_nt_m64` (token gather,
  per-expert ptrs, empty-block early exit; K=320 at TP2 needs BK=64).
- **P4 expert cache**: planner + hot/swap/host split + davetha LRU kernels + fused shared expert.
- **P5 perf**: QSA fp8-KV + WMMA indexer, MTP k>1, GDN HIP, P2P AR communicator, fused glue kernels,
  FP8 lm_head, launch-config sweeps. Each gated by rocprof data + A/B bench + GSM8K/needle correctness.
- **P6 packaging**: Dockerfile, wheel, bench suite, docs; TP=3 padding later (3-card Gen4 box).

## Open questions for Brian
- Licenses: ask StillDeadcode (libr4d) and GGZ14 for a license before publishing derived code.
- Which checkpoint is canonical: tcclaviger GPTQ int6-PLE (current), tcclaviger MXFP4-FP8, or davetha heretic2?
