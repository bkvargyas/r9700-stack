"""Weight preparation helpers for R9kMxfp4MoEMethod (quant/ct.py).

Weight preparation is IN PLACE: the fragment-order permute moves bytes within each expert without changing
the size, so it is done chunk by chunk inside the checkpoint tensors' own storage -- VRAM, or pinned host
memory when the experts are UVA-offloaded (--cpu-offload-params experts). No second host copy of ~40 GB/rank.
The E8M0 scales are repacked to [E, K/32+1, N] on the GPU (~1/16 of the weight bytes).
"""
from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)

_CHUNK_BYTES = 256 << 20


def permute_in_place(packed: torch.Tensor) -> torch.Tensor:
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


def pack_scales_gpu(scale: torch.Tensor, device) -> torch.Tensor:
    from ..kernels.moe import pack_scales
    E = scale.shape[0]
    per = scale[0].numel()
    step = max(1, _CHUNK_BYTES // per)
    out = torch.empty((E, scale.shape[2] + 1, scale.shape[1]), dtype=torch.uint8, device=device)
    for e0 in range(0, E, step):
        e1 = min(E, e0 + step)
        out[e0:e1].copy_(pack_scales(scale[e0:e1].to(device)))
    return out


def as_param(t: torch.Tensor, like: torch.nn.Parameter) -> torch.nn.Parameter:
    p = torch.nn.Parameter(t, requires_grad=False)
    if getattr(like, "_vllm_is_uva_offloaded", False):
        p._vllm_is_uva_offloaded = True
    return p


def maybe_attach_cache(method, layer) -> None:
    import re
    from . import cache as C
    from ..kernels.moe import GROUP
    s13, s2 = layer.w13_weight_scale, layer.w2_weight_scale
    N1, K1 = s13.shape[2], (s13.shape[1] - 1) * GROUP
    N2, K2 = s2.shape[2], (s2.shape[1] - 1) * GROUP
    per_expert = (layer.w13_weight[0].numel() * 4 + layer.w2_weight[0].numel() * 4 + s13[0].numel() + s2[0].numel())
    slots = C.slots_per_layer(per_expert)
    if slots <= 0:
        return
    if C.host_only() and not (C.is_host(layer.w13_weight) and C.is_host(layer.w2_weight)):
        return          # layer lives in VRAM already: nothing to cache
    name = getattr(layer, "layer_name", "") or ""
    m = re.search(r"layers\.(\d+)\.", name)
    idx = int(m.group(1)) if m else 0
    w13, w2 = layer.w13_weight.data, layer.w2_weight.data
    if not C.is_host(layer.w13_weight):
        w13 = C.to_host(w13)
        layer.w13_weight = as_param(w13, layer.w13_weight)
        layer.w13_weight._vllm_is_uva_offloaded = True
    if not C.is_host(layer.w2_weight):
        w2 = C.to_host(w2)
        layer.w2_weight = as_param(w2, layer.w2_weight)
        layer.w2_weight._vllm_is_uva_offloaded = True
    cache = C.LayerCache(idx, w13, w2, s13.data, s2.data, N1, K1, N2, K2, slots)
    cache.fold = tuple(getattr(layer, "_r9k_fold", (False, False)))
    # the host copies of the scales now back the cold pass; drop the device ones (keep shapes for _dims)
    layer.w13_weight_scale = torch.nn.Parameter(cache.h_s13, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(cache.h_s2, requires_grad=False)
    from vllm.model_executor.layers.fused_moe.config import mxfp4_w4a16_moe_quant_config
    method.moe_quant_config = mxfp4_w4a16_moe_quant_config(w1_scale=layer.w13_weight_scale,
                                                           w2_scale=layer.w2_weight_scale)
    method.moe_kernel.fused_experts.quant_config = method.moe_quant_config
    method.moe_kernel.fused_experts.r9k_cache = cache
    torch.cuda.empty_cache()
    logger.info("r9700: expert cache ON for %s: %d slots (%.2f MiB/expert/rank)", name or idx, cache.S,
                per_expert / 2**20)
