"""Opt-in FP8 serving of selected *unquantized* (bf16) linears on gfx12: R9K_FP8_LINEARS=<regex on layer prefix>.

Flash-Next's hyper-connection linears are excluded from the checkpoint's quantization and cost ~1.33 GB/step of
bf16 weight reads per rank (davetha, docs/HC_QUANT.md); fp8 per-channel/per-tensor costs ~2.5% relative error on
those tensors (int4: 9-16%, not recommended). After loading, matching layers get a per-row-scaled e4m3 copy served
by libr9k's fp8 GEMM (any M, unlike wvSplitKQ which refuses n=5 -- MTP-4 verify batches). The bf16 weight is kept
unless R9K_FP8_LINEARS_FREE=1 (other code may read it; freeing saves ~0.66 GB/rank for the HC set).
Suggested: R9K_FP8_LINEARS='hyper_connection'.
"""
from __future__ import annotations

import os
import re

import torch

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)
_PATCHED = False


def patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    pat = os.environ.get("R9K_FP8_LINEARS", "")
    if not pat:
        return False
    rx = re.compile(pat)
    free = os.environ.get("R9K_FP8_LINEARS_FREE", "0") == "1"
    try:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    except Exception as e:
        logger.warning("r9700: fp8 linear hook not installed (%s)", e)
        return False
    from ..moe.experts import r9k_available

    orig_pwal = UnquantizedLinearMethod.process_weights_after_loading
    orig_apply = UnquantizedLinearMethod.apply
    count = {"n": 0, "bytes": 0}

    def process_weights_after_loading(self, layer):
        orig_pwal(self, layer)
        prefix = getattr(layer, "prefix", "") or ""
        w = getattr(layer, "weight", None)
        if not (prefix and rx.search(prefix) and isinstance(w, torch.Tensor) and w.dim() == 2
                and w.is_cuda and w.shape[0] % 16 == 0 and w.shape[1] % 16 == 0 and r9k_available()):
            return
        from ..kernels import fp8 as F8
        layer._r9k_fp8 = F8.quantize_rows_fp8(w.data)
        count["n"] += 1
        count["bytes"] += w.numel()
        if free:
            layer.weight.data = torch.empty((0, w.shape[1]), dtype=w.dtype, device=w.device)
        logger.info_once("r9700: fp8 serving for unquantized linears matching %r (first: %s %s)", pat, prefix,
                         tuple(w.shape))

    def apply(self, layer, x, bias=None):
        W = getattr(layer, "_r9k_fp8", None)
        if W is None:
            return orig_apply(self, layer, x, bias)
        from ..ops import fp8_linear
        out = fp8_linear(x, W).to(x.dtype)
        return out + bias if bias is not None else out

    UnquantizedLinearMethod.process_weights_after_loading = process_weights_after_loading
    UnquantizedLinearMethod.apply = apply
    _PATCHED = True
    return True
