# Upstream vLLM 0.29.0 + ROCm 10 for Qwen3.8-Flash-Next on 2x R9700 (gfx1201): what we can use and what the plugin must supply

Date: 2026-09-18.

**Sources**
- Source tree: `~/src/vllm-0.29.0`. It is a shallow clone at `98dff2a`. Every `path:line` below is relative to `~/src/vllm-0.29.0` unless stated otherwise.
- Web and PR data: AMD wheel index, Docker Hub and the GitHub API, all checked on 2026-09-18.
- Companion note: `kernel-inventory.md` (what the private `tcclaviger/vllm:dev` image does).

**Labels.** "Verified" means read in the source. "Expected" means inferred from the code path but not run on hardware. Nothing in this note was executed on the GPUs.

---

## TL;DR

1. **The `amd/` path is a portable Triton port of the NVIDIA path. It is not tuned for MI355X kernels.**
   - `amd/model.py` is byte-identical to `nvidia/model.py`, and so is `mtp.py`.
   - Only these files differ: QSA ops, the QSA indexer, `ple_layer`, `low_latency_gemm` and comments.
   - gfx950 is named in only two places:
     - an LDS comment at `vllm/models/qwen4_exp/amd/ops/qsa.py:884`;
     - the MXFP4 oracle's "Qwen3.8 TEP8 on gfx950 → emulation" guard at `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:477-510`.
   - The in-tree eval configs are B200/H200 FP8 only (`tests/evals/qwen4_exp/configs/`).
   - Most components will *run* on gfx1201 through Triton. None of them is fast.
2. **Stock 0.29 cannot serve our checkpoint on gfx1201.** There are five blockers:
   - (a) The compressed-tensors MXFP4 MoE path is hard-wired to Marlin (CUDA-only), so it fails at weight processing.
   - (b) The AMD PLE loader accepts only bf16 row shards. Our int6/MXFP4 packed PLE shards fail its shape check, and FP8 PLE loads without its scale.
   - (c) There is no PLE offload and no expert offload on the AMD path. The weights are far larger than 64 GB of VRAM. PR #54371 (UVA PLE offload) is not in the 0.29.0 tree, and it covers NVIDIA only.
   - (d) QSA forces a bf16 KV cache. We run `--kv-cache-dtype fp8` today.
   - (e) MTP with more than one speculative token on ROCm is expected to raise. The QSA `FlashAttentionMetadata` is not in the ROCm spec-decode allowlist.
3. **Almost everything can be supplied without patching vLLM source.** The mechanisms:
   - an OOT `ModelRegistry.register_model` override of `Qwen4ExpForConditionalGeneration`, `Qwen4ExpForCausalLM` and `Qwen4ExpMTP`;
   - `PluggableLayer.register_oot` for `RoutedExperts` (the expert-cache hook) and `QwenGatedDeltaNetAttention`;
   - `register_quantization_config` (custom configs win over built-ins);
   - `register_linear_kernel` for the FP8 and MXFP4 linears;
   - `direct_register_custom_op` plus our own `.so` for kernels.

   The main exceptions are the spec-decode allowlist and the offloader factory, both of which need a monkeypatch.
4. **ROCm 10.** The 0.29 source never mentions ROCm 10.
   - Build pins: torch 2.12.0+rocm7.14.0 (`requirements/build/rock.txt`); CMake expects torch 2.13.0.
   - TheRock stable index has torch 2.11, 2.12 and 2.13 `+rocm10.0.0`, plus `amd-torch-device-gfx1201` wheels.
   - `vllm/vllm-openai-rocm:nightly-rocm100` is a daily ROCm 10 image (torch 2.12.0+rocm10.0.0, gfx1201 in the arch list, no AITER for gfx12).
   - `latest` (v0.29.0) is ROCm 7.2.3.

---

## 1. How Qwen4Exp is built in 0.29

### Dispatch
- `vllm/models/qwen4_exp/__init__.py:21-50` has a lazy `__getattr__`.
  - `current_platform.is_rocm()` selects `.amd.model` and `.amd.mtp`; otherwise `.nvidia.*`. XPU and TPU raise.
- Registry entries point at the package, not a file:
  - `vllm/model_executor/models/registry.py:114` (`Qwen4ExpForCausalLM`)
  - `:580` (`Qwen4ExpForConditionalGeneration`)
  - `:670` (`Qwen4ExpMTP` → `("vllm.models.qwen4_exp", "Qwen4ExpMTP")`)
- `diff amd/ nvidia/` result:
  - Identical: `model.py`, `mtp.py`, `model_state.py`.
  - Comment-only differences: `hyperconnection.py`, `ops/hc.py`.
  - Real differences: `qsa.py`, `ops/qsa.py`, `indexer_qsa.py`, `ple_layer.py`, `low_latency_gemm.py`.
  - `nvidia/ops/qsa_pre_indexer.py` has no AMD equivalent.
- There are **no gfx12, gfx1201 or RDNA checks anywhere in `vllm/models/qwen4_exp/`**. The only ROCm-specific tweaks are:
  - `ops/qsa.py:884-886`: one software-pipelining stage "because gfx942/gfx950 have a 64 KiB LDS limit". gfx1201 also has 64 KiB of LDS per workgroup, so this applies to us too.
  - The top-k fallback at `ops/qsa.py:772-810`.

### Component by component (ROCm path, gfx1201 verdict)

**Decoder layer** (`amd/model.py:174-327`)
- What it uses on ROCm: GDN on `linear_attention` layers; QSA (`Qwen4ExpQSAAttention`) on `full_attention` layers when `indexer_n_heads` is set (`:217-235`); otherwise `Qwen3NextAttention`. PLE on `ple_layer_ids`. Hyper-connection (`GatedResidual`) around both attention and MLP.
- gfx1201: this is structure only.

**MoE experts** (`Qwen4ExpSparseMoeBlock`, `amd/model.py:160-171`)
- What it uses: the `Qwen3NextSparseMoeBlock` block. It contains `FusedMoEFactory(...)` (`vllm/model_executor/models/qwen3_next.py:215-235`), which builds the `RoutedExperts` `PluggableLayer` (`vllm/model_executor/layers/fused_moe/routed_experts.py:44`). The quant method comes from the quantization config.
- gfx1201 by checkpoint:
  - bf16: TritonExperts works.
  - FP8: TritonExperts works. `oracle/fp8.py:408-416` drops AITER on RDNA, and `fused_batched_moe.py:821` lets FP8 run on RDNA4.
  - **MXFP4 (compressed-tensors): fails** (see §2).

**Shared expert** (`Qwen3NextMLP` + `shared_expert_gate`, `qwen3_next.py:186-213`)
- Uses standard linears. BF16 goes through `rocm_unquantized_gemm`: skinny `wvSplitK`/`LLMM1` on gfx1x (`vllm/model_executor/layers/utils.py:319-339`), then torch/hipBLASLt.
- gfx1201: works. FP8 uses the scaled_mm list (§2).

**GDN** (`QwenGatedDeltaNetAttention`, `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:360-405`)
- On ROCm, `_forward_method = forward_hip` (`:401-403`).
- `forward_hip` (`:853-887`) takes the AITER fused path only if `GDN_AITER_TRITON_AVAILABLE` (`:74-85`). That flag requires AITER installed and `VLLM_ROCM_USE_AITER=1`; on RDNA4 the check is `is_rdna_gdn_triton_kernels_available`, `_aiter_ops.py:2067`.
- Even then, the AITER decode fast path requires `gqa_interleaved_layout` (`:1234-1245`). Qwen4Exp passes `gqa_interleaved_layout=False` (`amd/model.py:215`), so it always falls through to generic `_forward_core`: FLA Triton chunked prefill (`vllm/third_party/flash_linear_attention`), the Triton causal_conv1d, and the Triton recurrent decode.
- The prefill backend resolves to "triton" on non-CUDA (`:117-118`).
- The CUDA `fused_gdn_decode_post_conv_mtp` op is built only for CUDA archs (`CMakeLists.txt:1138-1148`). The code checks `hasattr` (`:546`, `:1830`).
- FLA has one RDNA tweak: `chunk_scaled_dot_kkt.py:28` (`_CAST_DOT_TO_K_DTYPE = on_gfx1x()`).
- gfx1201: **works (Triton)**. It is slow at decode. The fork uses gdn_hip here.

**QSA full attention** (`amd/qsa.py`)
- The backend is hard-wired: `self.attn_backend = Qwen4ExpQSAFlashAttentionBackend` (`amd/qsa.py:278`). It **bypasses the platform attention selector**.
- `Qwen4ExpQSAFlashAttentionImpl.__init__` requires `is_flash_attn_varlen_func_available()` (`:107-108`). On ROCm that is just "`import flash_attn` succeeded" (`vllm/v1/attention/backends/fa_utils.py:38-62,365-367`). The attention itself never calls FA; it is Triton `qsa_sparse_paged_attention` (split-K plus merge, `ops/qsa.py:822-960`).
- The KV write uses `reshape_and_cache_flash` from `_C`.
- Constraints (`:186-193, 272-273`): BF16 model dtype, **BF16 KV cache only**, no KV quantization, no context parallelism.
- The fused qk-norm-rope-gate is CUDA-only (`:254-260`).
- gfx1201:
  - **Runs (Triton) if a `flash_attn` package is importable.** The stock images ship flash_attn built with gfx1xxx stripped from GPU_ARCHS (`docker/Dockerfile.rocm_base:323`), but the import still succeeds.
  - **No FP8 KV.**

**QSA indexer** (`amd/indexer_qsa.py`, `amd/ops/qsa.py`)
- Portable Triton kernels:
  - `_qsa_mqa_paged_kernel` scores with a scalar per-head loop, no `tl.dot` (`ops/qsa.py:19-113`). The NVIDIA version uses `tl.dot` plus a multi-stage pipeline.
  - compress-groups and store-rows Triton kernels.
- RMSNorm and RoPE are portable (`indexer_qsa.py:57,66-71`).
- Top-k on ROCm is `ops.top_k_per_row_decode` (`ops/qsa.py:799-808`), which is `_C` `csrc/libtorch_stable/topk.cu` and is built for HIP (`CMakeLists.txt:450`). `cooperative_topk` and `persistent_topk` are CUDA-only.
- gfx1201: **works (Triton plus HIP top-k).** The scoring kernel is a known slow spot.

**QSA metadata** (`common/qsa_cache.py:197-547`)
- A Triton metadata builder. PDL (`gdc_wait`) is gated by `is_arch_support_pdl()`, which is false on ROCm.
- gfx1201: works.

**PLE** (`amd/ple_layer.py`)
- `Qwen4ExpNGramEmbedding` hashes n-grams in torch ops (`:283-356`).
- The table is a **plain bf16 `PLEVocabParallelEmbedding` with no quant method** (`:225-230`), vocab-sharded across TP.
- The lookup is `F.embedding` wrapped in the custom op `qwen4_exp_amd_ple_ngram_embedding` (`:1068-1082`) so Inductor does not copy the weight.
- The short conv is the Mamba-style `PleShortConvAttentionBackend` (`vllm/v1/attention/backends/short_conv_attn.py:94`), torch and Triton.
- Unlike the NVIDIA version, there is no FP8 `Qwen4ExpPLEFp8EmbeddingMethod` (`nvidia/ple_layer.py:83-150`).
- The loader accepts only `ngram_embedding.shard_N.weight` of shape `(rows, head_dim)` (`:384-415`). It then copies with `.to(dtype)`, which **drops FP8 scales silently**; see PR #55040.
- There is no offload.
- gfx1201: compute works. **Loading our packed int6/MXFP4 PLE fails the shape check. FP8 PLE is mis-scaled. The table does not fit in VRAM.**

**Hyper-connection** (`amd/ops/hc.py`)
- Triton: grouped Gemma RMSNorm, hc_silu, gate_mix, combine, combine_norm. The "skinny GEMM padded to 16 rows" is a standard linear.
- gfx1201: works (Triton).

**low_latency_gemm** (`amd/low_latency_gemm.py:1-15`)
- A **no-op on AMD**: "Keep the standard vLLM linear methods". The NVIDIA version is 185 lines.
- gfx1201: nothing to lose. This is a natural hook point for us.

**MTP** (`amd/mtp.py`, identical to NVIDIA)
- Reuses `Qwen4ExpDecoderLayer` (full_attention, so QSA) and `Qwen4ExpSparseMoeBlock`, with PLE forced off (`:5-7, 206-215`).
- Spec decode on ROCm with `num_speculative_tokens > 1` checks drafter metadata against an allowlist (`vllm/v1/spec_decode/llm_base_proposer.py:264-321`, raise at `:655-662`):
  - The allowlist has Triton, ROCM_ATTN, AITER-FA, MLA and Flex metadata.
  - **It does not have `FlashAttentionMetadata`**, which is what `Qwen4ExpQSAMetadataBuilder` produces.
- gfx1201: **k=1 is expected to work. k>1 is expected to raise `ValueError`.** Open PR #55292 addresses this.

**Tensor-parallel all-reduce**
- `RocmPlatform.use_custom_allreduce()` returns True only for gfx94 and gfx95 (`vllm/platforms/rocm.py:982-984`), so gfx1201 uses RCCL.
- gfx1201: works (RCCL only). The fork uses clav_ar.

### ROCm platform facts for gfx1201 (`vllm/platforms/rocm.py`)
- Arch flags (`:217-232`):
  - `_ON_GFX12X` / `on_gfx12x()` (`:322`) and `on_rdna4()` (gfx1200/1201, `:330`).
  - `on_rdna()` and `on_gfx1x()` exclude gfx1250, which is classed as CDNA.
- R9700 device id `0x7551` is mapped (`:81`).
- `supports_fp8()` is `on_cdna() or on_rdna4()`, so True (`:966-967`). `fp8_dtype` is e4m3fn.
- `supports_mx()` is gfx95/gfx1250 only, so **False** (`:962-963`).
- Attention priorities (`:481-495`), used only for non-QSA layers: ROCM_ATTN, then AITER unified on RDNA only when `is_rdna_aiter_enabled`, then TRITON_ATTN, then TURBOQUANT.
  - Do not set `VLLM_ROCM_USE_AITER=1` on gfx12: the AITER unified-attention kernel exceeds LDS (open PR #56040).
- AITER on RDNA4 is Triton-only:
  - `is_aiter_found_and_supported_on_rdna4()` (`vllm/_aiter_ops.py:157-168`)
  - `is_rdna_aiter_enabled()`, `is_rdna_linear_enabled()`, `is_rdna_gdn_triton_kernels_available()` (`:1863-1878, 2067`)
  - `is_triton_gemm_w8a8_tuned` has an rdna4 shape table (`:3039-3072`).
  - The published images build AITER for `gfx942;gfx950` only (`docker/Dockerfile.rock_base:23`), so on gfx12 AITER needs a custom build.
- Other gfx12-aware kernels:
  - `kernels/linear/scaled_mm/rocm.py:82-89` (ROCmFP8ScaledMM allowed on gfx12x)
  - `kernels/linear/scaled_mm/pytorch.py:30` (torch `_scaled_mm` on gfx12x)
  - `kernels/linear/mixed_precision/rdna_hybrid_w4a16.py:41,238` (gfx12 W4A16)
  - `csrc/rocm/skinny_gemms.cu:29-68,1930-1992` (gfx12 wave32 wvSplitK / FP8 dot4)
  - `csrc/rocm/attention.cu:2408-2421,3644` (gfx12 WMMA custom paged attention, head 128 only via `use_rocm_custom_paged_attention` `:405-418`, so not usable for head_dim 256)

---

## 2. Quantization: compressed-tensors MXFP4 MoE plus FP8 on gfx1201

### Compressed-tensors dispatch

MoE:
- `CompressedTensorsMoEMethod.get_moe_method` (`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py:30-67`) routes `_is_mxfp4(weight_quant)` to `CompressedTensorsW4A4Mxfp4MoEMethod`.
  - `_is_mxfp4` means group, float, 4-bit, group 32, symmetric (`compressed_tensors.py:424-436`).
- Backend choice in that class (`compressed_tensors_moe_w4a4_mxfp4.py:46-69`):
  - `moe_backend == "b12x"` → oracle (CUDA).
  - `CutlassExpertsMxfp4._supports_current_device()` → CUTLASS (CUDA SM100).
  - XPU → XPU.
  - **Everything else, including ROCm → `MarlinExperts`.**
- `process_weights_after_loading` calls `prepare_moe_fp4_layer_for_marlin(layer)` (`:200-208`), which uses `gptq_marlin_repack`.
  - Marlin is built only for CUDA (`CMakeLists.txt:609-727`).
  - `MarlinExperts._supports_current_device()` is `is_cuda() and cap >= 7.5` (`vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:605-607`).
- **Result: compressed-tensors MXFP4 MoE fails on any ROCm GPU (verified path; expected missing-op or unsupported error at load).** The CT method never consults the MXFP4 oracle's ROCm list.
- Open PR #46676 states the same thing: "compressed-tensors MXFP4 refuses to load on RDNA".

Dense linears:
- `compressed_tensors.py:745` sends MXFP4 to `CompressedTensorsW4A4Mxfp4`.
- The MXFP4 linear kernel list for ROCm (`vllm/model_executor/kernels/linear/__init__.py:567-570`) is `AiterMxfp4LinearKernel` (gfx950) then `EmulationMxfp4LinearKernel`, so gfx1201 gets **emulation (dequantize, then bf16 GEMM)**.

### MXFP4 MoE oracle (`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`)
Used by Quark, INC, online-mxfp4 and gpt-oss mxfp4. Not used by compressed-tensors except via b12x.

- Backends (`:102-134`): B12X, DEEPGEMM, FLASHINFER_{TRTLLM,CUTLASS}, MARLIN/BATCHED_MARLIN, AITER_MXFP4_{BF16,FP8,MXFP4}, TRITON/TRITON_UNFUSED (OAI triton_kernels), XPU, CPU, EMULATION, HUMMING.
- `backend_to_kernel_cls` (`:154-273`) is a closed if-chain. **There is no registry, so we cannot add a backend without patching or bypassing.**
- ROCm priority (`_get_priority_backends`, `:343-352`) is `[AITER_MXFP4_BF16, EMULATION]`. The gpt-oss list (`:321-340`) adds TRITON; OAI triton_kernels claims `on_gfx9() or on_gfx1x()` support (`experts/gpt_oss_triton_kernels_moe.py:55-56`).
- The Qwen3.8 guard `_requires_qwen38_tep8_emulation` (`:477-510`): E=512, H=2560, N=640, EP8, W4A4 on gfx950 → emulation. This is the only Flash-Next/AMD tuning in the tree, and it is an MI355X correctness workaround.
- EMULATION is `OCP_MXQuantizationEmulationTritonExperts` (`experts/ocp_mx_emulation_moe.py`). It needs the `amd-quark` package and dequantizes weights to bf16 **every forward**, then runs TritonExperts. That works on gfx1201 but is far too slow for 512 experts.

### Other MoE and linear paths on gfx1201
- FP8 MoE: `oracle/fp8.py` goes to TRITON (AITER removed on RDNA, `:408-416`). **Works.**
- WNA16 int4 MoE on ROCm: `rocm_moe_rdna` is gfx1100-only (`compressed_tensors_moe/rocm_moe_rdna.py:21-33`; `_rocm_C` `moe_q_gemm_rdna3.cu` is built only when gfx1100 is in the arch list, `CMakeLists.txt:1514-1521`). Otherwise `CompressedTensorsWNA16MoEMethod` (Triton).
- FP8 dense (ROCm order, `kernels/linear/__init__.py:412-420`):
  1. AITER HipbMM
  2. AITER preshuffled
  3. AITER per-token (these three are unusable without AITER on gfx12)
  4. `ROCmFP8ScaledMMLinearKernel` (skinny wvSplitKQ, gfx12 OK)
  5. PerTensor/RowWise/ChannelWise torch `_scaled_mm` (hipBLASLt FP8 on gfx12)

  Block FP8 order is AITER then Triton. **Works.** FlyDSL RDNA4 FP8 kernels are in open PR #56005.
- **Our checkpoint.** `tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ` = CT `mxfp4-pack-quantized` experts + FP8 dense + int6 PLE.
  - The PLE shards use a custom fused-row format: packed data followed by the scale, int6 with an fp16 scale per 32, or MXFP4 with an E8M0 scale. See the fork's `~/src/tcclaviger-dev/vllm/models/qwen4_exp/common/ple.py:10-140`.
  - That format is neither compressed-tensors nor anything the upstream loader knows.
  - **Upstream 0.29 cannot load it (MoE → Marlin; PLE → shape mismatch).**
  - The official FP8 variant loads its MoE and linears (Triton), but on the AMD path its PLE is mis-scaled (PR #55040), and it needs far more than 64 GB.

### What the MXFP4 MoE must become on gfx1201
We need a new `FusedMoEExperts` subclass: W4A16 (MXFP4 weight, dequantized in-kernel to bf16/fp16 WMMA) or W4A8 (FP8 WMMA). Base it on the public libr4d `gemm_mxfp4a8_nt_m64` and the pattern in #46676 (RDNA3). Wire it in by one of three routes:
- **(a, cleanest)** Register our own quantization config (`register_quantization_config("compressed-tensors")` overrides the built-in, `vllm/model_executor/layers/quantization/__init__.py:58-103,178`), or a new name claimed via `override_quantization_method`; custom methods are probed first (`vllm/config/model.py:1256-1290`). Subclass `CompressedTensorsConfig`, and for MoE layers return our `CompressedTensorsW4A4Mxfp4MoEMethod` subclass with `experts_cls = R9700Mxfp4Experts` and no Marlin repack.
- **(b)** `CustomOp`/`PluggableLayer.register_oot` on `RoutedExperts`, which also carries the expert cache (§3).
- **(c)** Monkeypatch `oracle.mxfp4.backend_to_kernel_cls` and `_get_priority_backends`. Not recommended.

---

## 3. Offloading in 0.29

**Config** (`vllm/config/offload.py`)
- `OffloadConfig.offload_backend` takes `"auto" | "uva" | "prefetch"` (`:12`).
- `UVAOffloadConfig` (`:16-45`): `cpu_offload_gb` plus `cpu_offload_params`, a set of *segment* matches such as `experts.w13_weight` or `ngram_embedding`.
- `PrefetchOffloadConfig` (`:48-80`): `offload_group_size`, `offload_num_in_group`, `offload_prefetch_step`, `offload_params` (async H2D per layer group).

**Factory and singleton**
- `create_offloader` (`vllm/model_executor/offloader/base.py:141-177`) is a closed if-chain, and `set_offloader` is a module global (`:123-139`).
- The worker calls `set_offloader(create_offloader(self.offload_config))` in `GPUModelRunner.__init__` (`vllm/v1/worker/gpu_model_runner.py:994`; model runner v2 at `vllm/v1/worker/gpu/model_runner.py:348`).
- `make_layers` wraps the decoder layers with `get_offloader().wrap_modules(...)` (`vllm/model_executor/models/utils.py:850-858`).

**UVA** (`vllm/model_executor/offloader/uva.py`)
- Per parameter: copy to pinned CPU, then `get_accelerator_view_from_cpu_tensor` → `torch.ops._C.get_cuda_view_from_cpu_tensor` (`vllm/utils/torch_utils.py:915`, cuda_alike so ROCm too). It stops once the byte budget is reached (`:71-150`).
- It knows nothing about experts: whole `w13_weight` tensors per layer, first-come order.
- Open fixes, each trivially monkeypatchable:
  - #54972: `empty_cache` after each module; without it you OOM on load.
  - #57472: respect the per-parameter budget.
  - #54610: prioritise routed experts.

**What does not exist in 0.29.0**
- No MoE-expert-aware cache, no hot/cold expert split, no device-side LRU.
  - The only related code is EPLB/`expert_map_manager.py` and `routed_experts_capturer.py` (routing capture).
  - The design reference is open PR #56177: a shared GPU expert pool with a device-side planner, NVFP4 Marlin, CUDA-only. It hooks `fused_moe/routed_experts.py`, `model_loader/utils.py`, `gpu_worker.py`, `config/offload.py` and `engine/arg_utils.py`.
- **No PLE offload.** There is no `engram` config in the tree, and `VLLM_PLE_CPU_OFFLOAD` does not appear anywhere.
  - PR #54371 (merged 2026-09-09) is **not in the v0.29.0 tree** and only touches the NVIDIA path.
  - Open PR #57497 ports pinned-host plus UVA PLE and next-layer prefetch to the AMD path.
- A workaround available today: `--cpu-offload-params ngram_embedding` (the UVA offloader) moves the PLE table to pinned host memory as a UVA view, because the PLE module lives inside the decoder layers (`layers.N.ple.ple_embedding.ngram_embedding.weight`).
  - `F.embedding` on the UVA view gathers rows over PCIe with zero copy.
  - Expected to work without a patch, once the loader problem in §1 is solved.

**How an expert cache plugs in without patching vLLM**
1. `@RoutedExperts.register_oot` (`PluggableLayer.__new__` swaps the class by name, `vllm/model_executor/custom_op.py:47-100`). The replacement:
   - owns `w13_weight`/`w2_weight` as pinned host tensors (UVA views) plus a VRAM slot bank;
   - wraps `quant_method.apply` to (1) map `topk_ids` to slots with a device-side LRU and (2) run the grouped GEMM over slots plus a zero-copy GEMM over host experts, as the fork does.
   - It is graph-safe provided all bookkeeping stays on the device.
2. Alternatively, do it inside our `FusedMoEExperts` subclass (§2), with weights relocated in `process_weights_after_loading`.
3. A global slot bank across layers (as in #56177) needs one allocation after load. Do it lazily at the first `process_weights_after_loading`, or from the model's `load_weights` in our OOT model class. The KV-cache profiler then sees it, because it runs after load.
4. Custom CLI flags such as `--expert-offload-mem` are **not** possible without patching `arg_utils`. Use env vars or `--additional-config` (a free-form dict on `VllmConfig`) instead.

---

## 4. Plugin extension points in 0.29 (exact)

**Entry points** (`vllm/plugins/__init__.py`)
- `vllm.general_plugins` (`:18`): loaded in every process by `load_general_plugins()` (`:77-89`). It is called from:
  - `engine/arg_utils.py:856,2977`
  - `v1/engine/core.py:119`
  - `v1/worker/worker_base.py:271`
  - `model_executor/models/registry.py:1536`

  Put all registration here. It must be idempotent. It can be filtered with `VLLM_PLUGINS`.
- `vllm.platform_plugins` (`:23`): resolved in `vllm/platforms/__init__.py:233-286`. **One activated OOT platform plugin beats the built-in ROCm plugin** (`:270-272`), and two OOT plugins raise an error.
  - We can return `"r9700_plugin.platform.R9700Platform"`, a **subclass of `RocmPlatform`**. `_enum` stays ROCM, so every `is_rocm()` branch stays live.
  - Useful overrides:
    - `get_attn_backend_cls` (non-QSA layers)
    - `check_and_update_config` (force cudagraph sizes, block size, spec settings)
    - `apply_config_platform_defaults`
    - `get_device_communicator_cls` (custom P2P all-reduce communicator)
    - `use_custom_allreduce`
    - `supports_mx`
    - `import_kernels` (load our `.so`)
    - `verify_quantization`
    - `get_default_ir_op_priority`
    - `is_arch_support_pdl`
    - `opaque_attention_op`
  - Limitation: module-level helpers such as `on_rdna4()`, `on_gfx12x()` and `use_rocm_custom_paged_attention()` in `vllm.platforms.rocm` are plain functions imported by value. **A platform subclass cannot change them**; only monkeypatching can.

**Model registry**
- `ModelRegistry.register_model(arch, "pkg.mod:Cls")` (`vllm/model_executor/models/registry.py:1098-1150`) overwrites an existing arch with a debug log. The lazy string form avoids CUDA init in the parent process.
- Override `Qwen4ExpForConditionalGeneration`, `Qwen4ExpForCausalLM` and `Qwen4ExpMTP` to point at our subclasses. This is **the main lever**, because QSA, PLE, the HC ops and `low_latency_gemm` are all hard-wired inside the model package, not selectable by backend.

**CustomOp / PluggableLayer OOT**
- `CustomOp.register_oot(name=...)` (`custom_op.py:339-360`) and `PluggableLayer.register_oot` (`:84-100`) replace by class name at `__new__`.
- Relevant registered names:
  - `QwenGatedDeltaNetAttention` (PluggableLayer "qwen_gated_delta_net_attention", `qwen_gdn_linear_attn.py:360`)
  - `ChunkGatedDeltaRule` (CustomOp, `:231`)
  - `RoutedExperts` (PluggableLayer, `routed_experts.py:44`)
  - `UnquantizedFusedMoEMethod` / `modular_fused_moe`
  - `GateLinear` (`router/gate_linear.py:17`)
  - `grouped_topk`
  - RMSNorm, SiluAndMul and other CustomOps
- On ROCm, `dispatch_forward` calls `forward_hip` (`custom_op.py:196-197`).

**Attention backends** (`vllm/v1/attention/backends/registry.py`)
- `register_backend(AttentionBackendEnum.X | CUSTOM, class_path, is_mamba=False)` (`:243-292`) overrides via `_ATTN_OVERRIDES`/`_MAMBA_ATTN_OVERRIDES`. The CUSTOM slots are at `:130` and `:193`.
- This helps only for layers that go through the selector: non-QSA attention, `MambaAttentionBackendEnum.GDN_ATTN` (metadata builder) and `SHORT_CONV`.
- **QSA does not go through the selector** (`amd/qsa.py:278`), so it must be replaced through the model override.

**Quantization**
- `register_quantization_config(name)` (`vllm/model_executor/layers/quantization/__init__.py:58-103`). Custom configs override built-ins, including "compressed-tensors" (`:178`), and `override_quantization_method` gives custom configs first claim (`vllm/config/model.py:1285-1290`).

**Linear kernels**
- `register_linear_kernel(cls, PlatformEnum.ROCM, kind)` with kind in {mp, int8, fp8, mxfp8, nvfp4, mxfp4, mxfp6} (`vllm/model_executor/kernels/linear/__init__.py:1115-1160`). It **appends** (lowest priority).
- Getting priority means inserting at index 0 of the module-level `_POSSIBLE_*_KERNELS[PlatformEnum.ROCM]` lists, which is a monkeypatch of private state, or using `--linear-backend`, which filters (`_resolve_backend_kernels`, `:655`).
- There is no registration for `fp8_block` or `wfp8a16`.

**Fused-MoE experts**
- **No registry.** The oracles are closed if-chains (`oracle/mxfp4.py:154-273`, `oracle/fp8.py:189-215`).
- Our options: our own quant method that builds `FusedMoEKernel` with our `experts_cls`, or `RoutedExperts.register_oot`.

**Torch ops**
- `direct_register_custom_op(op_name, op_func, mutates_args, fake_impl, target_lib=..., dispatch_key=...)` (`vllm/utils/torch_utils.py:1041+`). Use our own `torch.library.Library("r9700", "FRAGMENT")` as `target_lib` to avoid clashing with `vllm::`.
- A compiled `.so` can also be loaded with `torch.ops.load_library` inside `import_kernels`.

**Spec decode**
- `method: "custom_class"` plus `speculative_config.model="pkg.Cls"` (`vllm/v1/spec_decode/custom_class_proposer.py:12-60`, `vllm/config/speculative.py:1083,1128`) swaps the whole proposer. It is heavy but patch-free.

**Only possible by patching or monkeypatching**
- Adding QSA `FlashAttentionMetadata` (or our metadata) to the ROCm MTP allowlist (`llm_base_proposer.py:264-321`). Alternatives: monkeypatch `__init__`, make our metadata subclass `TritonAttentionMetadata`, or use custom_class.
- New offloader backends or `--*offload*` CLI flags (`create_offloader` if-chain, `arg_utils`). We can instead monkeypatch `vllm.model_executor.offloader.base.create_offloader` (the gpu_model_runner module imports the name, so patch `vllm.v1.worker.gpu_model_runner.create_offloader`), or avoid it entirely by doing the offload inside `RoutedExperts`/model `load_weights`.
- `on_rdna4()`/`on_gfx12x()` helper behaviour, `use_rocm_custom_paged_attention` and the `_rocm_C` kernel set: a new `csrc` kernel means our own extension, not vLLM's.
- The MXFP4 oracle backend list (only if we insist on the oracle route).
- UVA offloader bug fixes (#54972, #57472): monkeypatch `UVAOffloader._maybe_offload_to_cpu`.
- `QSA` BF16-KV-only asserts and the FA-import gate: bypassed by our QSA subclass through the model override, not by patching.

---

## 5. ROCm 10 and the build

**Source tree**
- **No ROCm 10 reference** anywhere in `CMakeLists.txt`, `setup.py`, `cmake/`, `docker/` or `requirements/`.
- ROCm version handling in `CMakeLists.txt:205-209` only warns if `Torch_VERSION < TORCH_SUPPORTED_VERSION_ROCM` (`2.13.0`, `:70-71`), and only when `ROCM_VERSION_DEV_MAJOR >= 5`. ROCm 10 passes with at most a warning.
- `HIP_SUPPORTED_ARCHS` includes `gfx1200;gfx1201` (`CMakeLists.txt:52`).
- `_rocm_C`:
  - Builds skinny GEMMs for every arch except gfx1250 (`:1484-1511`).
  - Adds RDNA3 `q_gemm`/`moe_q_gemm` only if gfx1100 is present (`:1514-1521`).
  - The custom paged attention has gfx12 WMMA (`csrc/rocm/attention.cu:2408-2421`).
- `_C_stable_libtorch` (topk, cache kernels, ngram, custom_all_reduce and so on) is built for HIP (`:395-457`).
- **gfx1201 compiles.**

**Requirements**
- `requirements/rocm.txt`: common plus `amd-quark==0.12.post1`, `tilelang==0.1.10`, `conch-triton-kernels==1.2.1` and others. No torch pin.
- `requirements/build/rocm.txt` and `rock.txt`: `torch==2.12.0+rocm7.14.0`, `triton==3.7.1+git0263a6a6[.rocm7.14.0]`, `amdsmi==7.0.2`, index `https://repo.amd.com/rocm/whl-multi-arch/`.
- `pyproject.toml:10`: `torch == 2.13.0` (build isolation). Build with `use_existing_torch.py` / `--no-build-isolation`.

**Dockerfiles**
- `docker/Dockerfile.rocm_base`: `rocm/dev-ubuntu-22.04:7.2.3-complete`, ROCm/pytorch@6bbd260 (release/2.12), triton f0b55c0, FA 0e60e394 (gfx1xxx stripped, `:323`), AITER v0.1.19 (gfx942/gfx950).
- `docker/Dockerfile.rock_base` (TheRock pip SDK): torch `2.12.0+rocm7.14.0`, `ROCM_SDK_VERSION=7.14.0`, `PYTORCH_ROCM_ARCH` including gfx1200/gfx1201, `AITER_ROCM_ARCH=gfx942;gfx950`.
- Pointing it at ROCm 10 means changing the index URL and version args.
- `setup.py`:
  - `:97-98`: `torch.version.hip` → `VLLM_TARGET_DEVICE=rocm`.
  - `:316-317`: `-DROCM_PATH`.
  - `:635-653`: precompiled wheel variant naming `rocm<ver>`, e.g. `rocm1000`.

**TheRock stable index** (`https://stable.repo.amd.com/rocm/whl-next/`)
- ROCm **10.0.0 is released** there: `rocm_sdk_core`/`devel`/`libraries` 10.0.0.
- torch generic wheels `2.11.0`, `2.12.0` and `2.13.0 +rocm10.0.0` for cp310–cp314, with arch payload in **`amd-torch-device-gfx1201`** (also `-gfx1200` and the family package `-gfx12-0`). Matching torchvision device wheels exist.
- triton `3.8.0+git4cff872c.rocm10.0.0`.
- Nightlies are at `https://nightly.repo.amd.com/rocm/whl-next/`.

**Docker Hub**
- `vllm/vllm-openai-rocm:latest` is v0.29.0 on **ROCm 7.2.3**.
- **`nightly-rocm100`** (daily; 2026-09-18 build is commit dee37d89115d; also tagged `base-nightly-rocm100`):
  - torch 2.12.0+rocm10.0.0, torchvision 0.27.0, torchaudio 2.11.0, triton 3.8.0+git4cff872c.
  - The SDK comes from pip (`ROCM_PATH=.../dist-packages/_rocm_sdk_devel`).
  - gfx1201 is in the arch list. No AITER for gfx12.
- There is no release-tagged ROCm 10 vLLM image.
- `qwen38-flash-next` tag (2026-08-26): MI-series AITER-QSA build (gfx942/gfx950), useless for us.
- `rocm/vllm`: the only ROCm 10 tag is `rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0` (vLLM 0.27). Newest are `rocm7.14.1_{rdna,cdna}_..._vllm_0.23.0`. These lag upstream.

**Recommended base:** `vllm/vllm-openai-rocm:nightly-rocm100`, pinned to a v0.29.x-compatible date, or build v0.29.0 against TheRock torch 2.12 or 2.13 +rocm10.0.0 with `amd-torch-device-gfx1201`.
- Our plugin wheel must build its HIP extension against the *same* torch and ROCm SDK. Use the pip `_rocm_sdk_devel` hipcc, and build for `--offload-arch=gfx1201` only.
- Keep our patched RCCL (`~/rccl10`) bind-mount approach. The nightly's RCCL comes from the ROCm 10 SDK.

---

## 6. Open upstream PRs worth cherry-picking or vendoring

None of the listed PRs is merged as of 2026-09-18.

**#57497 — [Qwen4Exp][ROCm] PLE n-gram table CPU offload + async prefetch**
- Files: `config/engram.py`, `config/vllm.py`, `qwen4_exp/amd/{model,ple_layer}.py`, `common/ngram_embedding.py`.
- What: AMD-path pinned-host PLE plus UVA view, side-stream prefetch of the next layer's rows, custom ops so compile does not materialize the table.
- Merge risk: medium (new, +751/−519; depends on #54371, which is not in 0.29.0).
- Relevance: **Top.** Vendor its model-side parts into our OOT model.

**#55040 — [Qwen4Exp][ROCm] FP8 PLE n-gram checkpoints on AMD**
- Files: `qwen4_exp/amd/ple_layer.py` plus a test.
- What: dequantizes FP8 PLE shards with `weight_scale` at load. Fixes two loader pitfalls: `weights` is a one-shot generator, and PLE weights arrive across several `load_weights` calls.
- Merge risk: low-medium (needs rebase).
- Relevance: **High**. Required for the FP8 checkpoint, and our int6 loader hits the same pitfalls.

**#56005 — [ROCm][Perf] RDNA4 FP8 block-scale FlyDSL kernels**
- Files: `kernels/linear/scaled_mm/rdna4.py`, `scaled_mm/flydsl_kernels/*`.
- What: FlyDSL FP8 GEMMs for gfx1200/1201 (prefill, small-M, split-K decode, wave32). 1.79x geomean over Triton on R9700.
- Merge risk: medium-high (needs rebase; FlyDSL main@65e5d8e built from source).
- Relevance: **High** for the FP8 linears. It is pure Python plus FlyDSL, so we can vendor it and register it via `register_linear_kernel`.

**#54610 — UVA offload: prioritise sparse MoE experts**
- Files: `offloader/uva.py` (+145/−62).
- What: offloads routed experts first across layers; the shared expert and gates last.
- Merge risk: medium (changes default behaviour).
- Relevance: **High** as a stopgap static offload. Explicit `--cpu-offload-params` gives the same result today.

**#54972 — UVA offloader: release freed accelerator blocks**
- Files: `offloader/uva.py` (+11).
- What: `torch.accelerator.empty_cache()` after each offloaded module; otherwise `expandable_segments` makes loading OOM.
- Merge risk: very low.
- Relevance: **High**. A one-line monkeypatch.

**#56177 — Shared GPU expert pool with a device-side planner (NVFP4 Marlin)**
- Files: new `fused_moe/expert_pool/*`; hooks `routed_experts.py`, `modelopt.py`, `model_loader/utils.py`, `gpu_worker.py`, `config/offload.py`, `arg_utils.py`.
- What: pinned-host experts, one GPU bank shared by all layers, device-side LRU inside CUDA graphs (FULL_DECODE_ONLY).
- Merge risk: high (+5376; CUDA/Marlin only).
- Relevance: **High as an architecture template** for our expert cache. It shows the hook points; the kernels do not transfer.

**#55917 — [ROCm][Perf] FlyDSL RDNA4 all-reduce**
- Files: `device_communicators/rdna4_all_reduce*`, `cuda_communicator.py`.
- What: TP2 one-shot all-reduce, P2P and mapped modes.
- Merge risk: medium-high (+3874).
- Relevance: **High** for TP2 decode. Deliver it through a platform `get_device_communicator_cls` override (a subclassed CudaCommunicator).

**#46676 — Native HIP MXFP4 (Compressed+Quark) dense + MoE for RDNA3**
- Files: `csrc/rocm/{mxfp4_gemm_rdna3,moe_mxfp4_gemm_rdna3}.cu`, `fused_moe/experts/rdna3_mxfp4_moe.py`, `oracle/mxfp4.py`, CT MoE, Quark MoE.
- What: the first ROCm MXFP4 MoE experts plus oracle wiring for compressed-tensors. gfx1100 only.
- Merge risk: high (dirty, +2378).
- Relevance: **High as a template.** The kernels need a gfx12 port (FP8 WMMA). It shows exactly which CT/oracle hooks to take over.

**#55996 / #45916 — RDNA4 FlyDSL SplitKV plus Triton split-KV paged decode for gfx12**
- Files: `rocm_attn.py`, `ops/chunked_prefill_paged_decode.py`, `flydsl_kernels/rdna4_splitkv*`.
- What: split-KV decode. #45916 was written because head_dim 256 fell back to the 2D kernel.
- Merge risk: medium / medium.
- Relevance: medium. QSA bypasses these; they matter only for non-QSA/ViT attention, or as a Triton pattern for our QSA sparse decode.

**#55292 — [ROCm][Spec Decode] metadata builders declare multi-step support**
- Files: 16 files, including `qwen4_exp/common/qsa_cache.py` and `llm_base_proposer.py`.
- What: replaces the hard ROCm allowlist.
- Merge risk: medium.
- Relevance: **High for MTP k>1** on ROCm. Needs a source patch; alternatively monkeypatch the allowlist.

**#57472 — UVA: respect the `--cpu-offload-gb` budget per parameter**
- Files: `offloader/uva.py` (+6/−6).
- Merge risk: very low. Relevance: medium (monkeypatch).

**#54986 — Don't search MFMA-only tuning params on RDNA**
- Files: `benchmarks/kernels/benchmark_moe.py`.
- Merge risk: very low.
- Relevance: medium. Use it to tune `fused_moe` Triton configs for E=512/N=320 on gfx1201 (FP8/bf16 fallback path).

**#53899 — Support PLE-Offload for Qwen3.8-Flash-Next (connector design)**
- Files: `v1/ple_offload/*`, NVIDIA ple_layer.
- What: superseded by #54371. MI355X BF16 validation only.
- Merge risk: dirty.
- Relevance: low (design reference).

**#57286 — gfx1201 M8 BF16 head (interleave M4 groups)**
- Files: `csrc/rocm/skinny_gemms.cu`.
- What: an 8-row wvSplitK on gfx1201. Scoped to [248320,5120]; the Python dispatcher is not widened.
- Merge risk: draft.
- Relevance: low-medium. It is a pattern for our LM head (248320×2560) with M=1+k MTP verify. Needs a C++ rebuild, so reimplement in our extension.

**#34741, #45559, #40827 — RDNA4 custom paged attn FP8 KV; skinny N=5→8 on gfx12 (SWMMAC); LLMM1 RDNA4 correctness**
- Files: `csrc/rocm/*`, `platforms/rocm.py`.
- Merge risk: ready / needs rebase / ready.
- Relevance: medium. #40827 fixes silently wrong LLMM1 results on wave32; check whether our decode hits LLMM1 (`layers/utils.py:336-338`, n==1, k≤8192). Being csrc, these need a vLLM rebuild or a reimplementation.

**#56040 — Honour `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` on RDNA**
- Files: `_aiter_ops.py`, `platforms/rocm.py`.
- Merge risk: low.
- Relevance: low. Workaround: do not set `VLLM_ROCM_USE_AITER=1`.

**#56742 — Fix Qwen4Exp MTP buffer placement, config parsing, Triton warmup**
- Files: `qwen4_exp/*`.
- Merge risk: low-medium.
- Relevance: medium. It is a bugfix to fold into our OOT model subclass.

**Also noted (lower priority)**
- #52869: AITER MoE on gfx12, bf16 only.
- #36659: stale draft.
- QSA fixes #57105, #54912, #56500, #54846 (FP8 KV on the QSA path; worth tracking) and #57431.
- PLE storage variants: #54129 (mmap), #54070 (disk), #56362 (FP8 raw-byte UVA), #56273 (NVFP4 PLE).
- #41197 and #43389 (MoE tuning / int4 repack).

---

## 7. What our plugin must provide, ranked by importance

Delivery shape: the pip package `r9700-vllm`.
- A `vllm.general_plugins` entry point → `r9700.register()`, which does all registrations, idempotently.
- An optional `vllm.platform_plugins` entry point → `R9700Platform(RocmPlatform)`, active only when gfx1201 is detected, and otherwise returning None so it defers to built-in ROCm.
- A HIP extension `.so` built against TheRock ROCm 10 plus torch 2.12/2.13, `--offload-arch=gfx1201`.

1. **An MXFP4 MoE expert kernel for gfx1201 plus a compressed-tensors MoE method that uses it.**
   - This is the blocker for our checkpoint: stock routes to CUDA-only Marlin.
   - A grouped W4A16/W4A8 GEMM (FP8 WMMA) with routing, align and sort. Start from libr4d's public `gemm_mxfp4a8_nt_m64` and the #46676 structure.
   - Wire it through `register_quantization_config` (subclass `CompressedTensorsConfig`, override the MoE method; drop the Marlin repack).
   - Also claim MXFP4 dense linears via `register_linear_kernel(..., "mxfp4")` so they do not use the emulation path.
2. **An expert cache/offload (hot experts in VRAM, cold experts in pinned host).**
   - Replace `RoutedExperts` via `PluggableLayer.register_oot`, or put it inside the experts class.
   - Device-side LRU, graph-safe, sized from an env var or `--additional-config`. Profile-seeded hot set (the `model-expertprofile.safetensors` sidecar). #56177 is the template.
   - The model does not fit in 2×32 GB otherwise.
3. **A PLE loader for our packed int6/MXFP4 (and FP8) shards, plus PLE offload with a fused dequant-gather.**
   - OOT `Qwen4ExpPLELayer` / `Qwen4ExpNGramEmbedding` inside our model subclass.
   - The table lives in pinned host memory as a UVA view (or an NVMe/row cache), read by a fused gather+dequant kernel on a side stream with next-layer prefetch (#57497 design).
   - Must handle the generator and multi-call loader pitfalls (#55040).
4. **An OOT model override** (`ModelRegistry.register_model` for `Qwen4ExpForConditionalGeneration`, `Qwen4ExpForCausalLM`, `Qwen4ExpMTP`). Subclass the upstream `amd.model`/`amd.mtp` classes and swap in our QSA, PLE, HC and low-latency GEMM hooks. This is the only patch-free way to change QSA and PLE.
5. **A QSA implementation for gfx1201 with FP8 KV.**
   - Replaces `Qwen4ExpQSAAttention`, `QSAIndexer` and the `qsa_*` ops.
   - Parts: a WMMA-based indexer scoring kernel (the upstream AMD kernel is a scalar loop), a sparse paged GQA attention kernel for head_dim 256 with fp8 KV, and a fused top-k.
   - Drop the `flash_attn` import gate and the BF16-only asserts (we run `--kv-cache-dtype fp8` today).
6. **MTP k>1 enablement on ROCm.**
   - Monkeypatch the `LLMBaseProposer` ROCm allowlist to add our QSA metadata class. Or make our metadata subclass an allowed type, or backport #55292.
   - Verify the MTP drafter and the HC hidden-state buffer (#56742 fixes).
7. **GDN decode/prefill kernels.**
   - OOT `QwenGatedDeltaNetAttention` (`register_oot`) whose `forward_hip` runs HIP conv1d plus gated-delta-rule for the non-interleaved layout, including the spec-decode (MTP) batch path. The fork's gdn_hip does *not* cover that path.
   - Base it on public libr4d `gdn_*`.
8. **FP8 dense GEMMs for RDNA4.**
   - Vendor #56005 (FlyDSL) and/or libr4d GEMMs, registered first in `_POSSIBLE_FP8_KERNELS[ROCM]` (insert at 0) or selected with `--linear-backend`.
   - Also cover the skinny decode shapes with M=1+k.
9. **TP2 P2P all-reduce** through `R9700Platform.get_device_communicator_cls` (CudaCommunicator subclass). Vendor #55917 or libr4d `ar_oneshot_2rank_*`. RCCL is the fallback; custom all-reduce is disabled on gfx12 upstream.
10. **HC, conv and glue kernels** (grouped RMSNorm plus combine/mix fusions, PLE short-conv, silu+quant) as `direct_register_custom_op` ops in our library, called from our model subclass. Also a fused LM-head/MTP-verify skinny GEMM (#57286 pattern).
11. **Monkeypatch fixes** for the UVA offloader (#54972 `empty_cache`, #57472 budget) and a guard that refuses `VLLM_ROCM_USE_AITER=1` on gfx12 (LDS crash, #56040).
12. **Build and packaging:** base on `vllm/vllm-openai-rocm:nightly-rocm100` or TheRock ROCm 10 torch plus `amd-torch-device-gfx1201`; pin the torch ABI; ship Triton/FlyDSL tuned configs (MoE Triton fallback configs via #54986) and the patched RCCL mount.

**Items that still need vLLM source changes** (keep minimal, or monkeypatch):
- the ROCm spec-decode allowlist (#6);
- CLI flags for offload;
- any change to the MXFP4 oracle or `on_rdna4()` helper semantics;
- csrc-level fixes to `_rocm_C` skinny GEMMs (#40827/#45559). Reimplement these in our own `.so`.
