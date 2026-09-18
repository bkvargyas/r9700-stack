"""Route compressed-tensors MXFP4 MoE layers to R9700Mxfp4Experts on gfx12 (stock vLLM picks CUDA-only Marlin).

Patches CompressedTensorsW4A4Mxfp4MoEMethod in place (idempotent). Other platforms are untouched.

Weight preparation is IN PLACE: the fragment-order permute moves bytes within each expert without changing
the size, so it is done chunk by chunk inside the checkpoint tensors' own storage -- VRAM, or pinned host
memory when the experts are UVA-offloaded (--cpu-offload-params experts). No second host copy of ~40 GB/rank.
The E8M0 scales are repacked to [E, K/32+1, N] on the GPU (~1/16 of the weight bytes).
"""
from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False
_CHUNK_BYTES = 256 << 20


def _permute_in_place(packed: torch.Tensor) -> torch.Tensor:
    """packed [E, N, K/2] uint8 (GPU or UVA) -> same storage viewed as [E, N/16*K/16*32] int32 fragment order."""
    from ..kernels.moe import permute_fragments
    E, N, Kh = packed.shape
    per = N * Kh
    step = max(1, _CHUNK_BYTES // per)
    for e0 in range(0, E, step):
        e1 = min(E, e0 + step)
        tmp = permute_fragments(packed[e0:e1])                    # [c, X] int32, fresh GPU buffer
        packed[e0:e1].view(e1 - e0, -1).copy_(tmp.view(torch.uint8).view(e1 - e0, -1))
        del tmp
    return packed.view(E, -1).view(torch.int32)


def _pack_scales_gpu(scale: torch.Tensor, device) -> torch.Tensor:
    from ..kernels.moe import pack_scales
    E = scale.shape[0]
    per = scale[0].numel()
    step = max(1, _CHUNK_BYTES // per)
    out = torch.empty((E, scale.shape[2] + 1, scale.shape[1]), dtype=torch.uint8, device=device)
    for e0 in range(0, E, step):
        e1 = min(E, e0 + step)
        out[e0:e1].copy_(pack_scales(scale[e0:e1].to(device)))
    return out


def _as_param(t: torch.Tensor, like: torch.nn.Parameter) -> torch.nn.Parameter:
    p = torch.nn.Parameter(t, requires_grad=False)
    if getattr(like, "_vllm_is_uva_offloaded", False):
        p._vllm_is_uva_offloaded = True
    return p


def patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
            compressed_tensors_moe_w4a4_mxfp4 as ctm,
        )
        from vllm.model_executor.layers.fused_moe.config import mxfp4_w4a16_moe_quant_config
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import make_mxfp4_moe_kernel
    except Exception as e:  # API moved: stay stock, say why
        logger.warning("r9700: CT MXFP4 MoE hook not installed (%s)", e)
        return False
    from .experts import R9700Mxfp4Experts, r9k_available

    cls = ctm.CompressedTensorsW4A4Mxfp4MoEMethod
    orig_init = cls.__init__
    orig_pwal = cls.process_weights_after_loading

    def __init__(self, moe):
        orig_init(self, moe)
        self._r9k = (not getattr(self, "use_cutlass_mxfp4", False)) and r9k_available()
        if self._r9k:
            self.experts_cls = R9700Mxfp4Experts
            logger.info_once("r9700: using R9700Mxfp4Experts (libr9k grouped MXFP4xFP8) for CT MXFP4 MoE")

    def process_weights_after_loading(self, layer) -> None:
        if not getattr(self, "_r9k", False):
            return orig_pwal(self, layer)
        dev = torch.device("cuda", torch.cuda.current_device())
        w13p, w2p = layer.w13_weight_packed, layer.w2_weight_packed
        w13 = _permute_in_place(w13p.data)
        w2 = _permute_in_place(w2p.data)
        s13 = _pack_scales_gpu(layer.w13_weight_scale.data, dev)
        s2 = _pack_scales_gpu(layer.w2_weight_scale.data, dev)
        layer.w13_weight = _as_param(w13, w13p)
        layer.w2_weight = _as_param(w2, w2p)
        delattr(layer, "w13_weight_packed")
        delattr(layer, "w2_weight_packed")
        layer.w13_weight_scale = torch.nn.Parameter(s13, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(s2, requires_grad=False)
        self.moe_quant_config = mxfp4_w4a16_moe_quant_config(
            w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale)
        self.moe_kernel = make_mxfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config, moe_config=self.moe, experts_cls=R9700Mxfp4Experts,
            mxfp4_backend=self.mxfp4_backend, routing_tables=layer._expert_routing_tables())
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    cls.__init__ = __init__
    cls.process_weights_after_loading = process_weights_after_loading
    _PATCHED = True
    return True
