"""DFlash / DFlash2 drafters with fp8-quantized attention, on stock vLLM (registered over the stock architectures).

Stock `DFlashQwen3Model._build_context_kv_buffers` concatenates the raw `qkv_proj.weight` K/V slices into one fused
bf16 projection and runs it with `F.linear`, which assumes an unquantized drafter. With an fp8 drafter (e.g.
tcclaviger/Qwen3.8-27B-DFlash2-FP8) the fused weight is fp8 (and possibly in a kernel-specific layout), so the
precompute fails ("expected mat1 and mat2 to have the same dtype"). After stock loading, this subclass rebuilds
`_fused_kv_weight` by applying each layer's own qkv_proj quant method to an identity matrix -- exactly the
dequantized weight, independent of the fp8 method's storage layout (identity rows quantize exactly under per-token
fp8). Only the small context-KV projection is dequantized; every other drafter linear stays fp8.
Upstream candidate: dequantize in `_build_context_kv_buffers` when qkv_proj is quantized.
"""
from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM

logger = init_logger("vllm." + __name__)


def _dequant_fused_kv(model) -> None:
    w = getattr(model, "_fused_kv_weight", None)
    if w is None or w.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return
    parts = []
    for layer in model.layers:
        a = layer.self_attn
        qkv = a.qkv_proj
        K = qkv.input_size_per_partition
        eye = torch.eye(K, dtype=torch.bfloat16, device=next(qkv.parameters()).device)
        full = qkv.quant_method.apply(qkv, eye, None)            # [K, out] = dequant(W)^T, bias excluded
        parts.append(full[:, a.q_size:].t().contiguous())
    model._fused_kv_weight = torch.cat(parts, dim=0).to(torch.bfloat16)
    logger.info_once("r9700: DFlash drafter context-KV projection dequantized from %s (%s)", w.dtype,
                     tuple(model._fused_kv_weight.shape))


class R9kDFlashQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    def load_weights(self, weights):
        out = super().load_weights(weights)
        _dequant_fused_kv(self.model)
        return out


class R9kDFlash2Qwen3ForCausalLM(DFlash2Qwen3ForCausalLM):
    def load_weights(self, weights):
        out = super().load_weights(weights)
        _dequant_fused_kv(self.model)
        return out


ARCHS = {
    "DFlashDraftModel": "r9700_vllm.models.dflash:R9kDFlashQwen3ForCausalLM",
    "DFlash2DraftModel": "r9700_vllm.models.dflash:R9kDFlash2Qwen3ForCausalLM",
}
