# Upstream issue text for vllm-project/vllm (final, 2026-10-10) -- posting blocked: the gh token here is a fine-grained PAT without issue rights on other repos

Title: [Bug] DFlash draft model: fp8 `ignored_layers` are not aliased to global layer indices, so excluded bf16 projections are loaded into fp8 parameters (acceptance drops from 4.3 to 1.0)

### Your current environment

vLLM 0.31.1rc1.dev173+g8cbd5d030 (nightly ROCm 10 image `vllm/vllm-openai-rocm:nightly-rocm100-8cbd5d03`), torch 2.12+rocm10.0, 2x Radeon AI PRO R9700 (gfx1201), TP=2. Reproduced with no out-of-tree plugin code active in the model (only a platform plugin that does not touch the draft model); the issue is in the stock draft-model construction path.

### Description

Target: `unsloth/Qwen3.8-27B-NVFP4` (compressed-tensors NVFP4, 64 layers). Drafter: an FP8 block-128 conversion of `z-lab/Qwen3.8-27B-DFlash2` (`DFlash2DraftModel`, 5 layers, `quant_method: fp8`, `weight_block_size: [128, 128]`). The drafter's `quantization_config.modules_to_not_convert` has 45 entries naming the modules its quantization must skip by checkpoint-local layer index, e.g.

```
layers.0.attention_conv.base_kernel
layers.0.attention_conv.kernel_projection
layers.0.mlp_conv.base_kernel
layers.0.mlp_conv.kernel_projection
layers.0.input_layernorm
...
```

vLLM builds the draft layers at global indices after the target's layers (`start_layer_id` = 64 for this target), and `vllm/model_executor/models/qwen3_dflash.py::_add_global_draft_layer_exclusions` (called from the draft model's `__init__` with `start_layer_id`) exists to add the globally-numbered aliases. It only patches `quant_config.exclude_modules`:

```python
exclusions = getattr(quant_config, "exclude_modules", None)
if not isinstance(exclusions, list):
    return
```

`Fp8Config` has no `exclude_modules`; `from_config` reads `modules_to_not_convert` into `ignored_layers`, and `ignored_layers_match_mode` is `"exact"` (`vllm/model_executor/layers/quantization/fp8.py`). So the global name `layers.64.attention_conv.kernel_projection` never matches, the grouped-conv kernel projections are built as fp8 linears, and the loader logs

```
Attempted to load weight layers.0.attention_conv.kernel_projection.weight with dtype torch.bfloat16 into parameter with dtype torch.float8_e4m3fn
```

Outputs stay correct (the target verifies every draft), but the drafter degrades: acceptance falls from 4.3 to 1.03 tokens per step and decode throughput from ~210 to 46 tok/s. On the September nightly (`e97573215`) the same checkpoint was fine, because those projections were constructed with `quant_config=None` there; the regression appeared when the draft quant config started reaching them.

### Fix

Apply the same aliasing to `ignored_layers` (and any other exclusion list a quant config carries), e.g. in `_add_global_draft_layer_exclusions`:

```python
for attr in ("exclude_modules", "ignored_layers"):
    exclusions = getattr(quant_config, attr, None)
    if isinstance(exclusions, list):
        ...  # same offset_local_layer substitution
```

Working around it from outside (adding `layers.{i+start}` and `model.layers.{i+start}` aliases to the draft hf config's `modules_to_not_convert` before construction) restores 4.48 tokens per step on this nightly.

### Reproduction

```
vllm serve unsloth/Qwen3.8-27B-NVFP4 --tensor-parallel-size 2 \
  --speculative-config '{"model": "<fp8 conversion of z-lab/Qwen3.8-27B-DFlash2>", "num_speculative_tokens": 7}'
```

then watch for the dtype-mismatch load message above and `vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_drafts_total` in `/metrics` during a decode (~1.0 per draft instead of ~4.3).
