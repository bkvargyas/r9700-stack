"""Opt-in: serve compressed-tensors W8A8 *block* FP8 linears (128x128 weight blocks, per-token-group-128 activations)
with libr9k's split-K fp8 GEMM on gfx12: R9K_FP8_BLOCK=rowwise.

vLLM's Triton block-FP8 GEMM has no split-K; at decode M it streams a 21 MB weight at ~175 GB/s on the R9700 even
tuned (tuning/tune_fp8_block.py: +7%), while libr9k's fp8 GEMM reaches ~600 GB/s. At load we dequantize with the
block scales and re-quantize per output row (e4m3, one fp32 scale per row), then run activations with a per-token
scale. Precision: per-row instead of per-128-block weight scales and per-token instead of per-128-group activation
scales -- the same trade the fp8 hyper-connection path makes; gate it with bench/quality.py.
The original fp8 weight + block scales are freed (VRAM-neutral).
"""
from __future__ import annotations

import os

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)
_PATCHED = False


def _requant_rowwise(w8: torch.Tensor, bs: torch.Tensor, block=(128, 128)):
    from ..kernels import fp8 as F8
    N, K = w8.shape
    out_rows = []
    step = 4096
    q = torch.empty((N, K), dtype=torch.uint8, device=w8.device)
    s = torch.empty((N,), dtype=torch.float32, device=w8.device)
    for r0 in range(0, N, step):
        r1 = min(N, r0 + step)
        sc = bs.float()[r0 // block[0]:(r1 + block[0] - 1) // block[0]]
        sc = sc.repeat_interleave(block[0], 0)[: r1 - r0].repeat_interleave(block[1], 1)[:, :K]
        x = w8[r0:r1].float() * sc
        amax = x.abs().amax(dim=1).clamp_min(1e-12)
        rs = amax / F8.FP8_MAX
        q[r0:r1] = (x / rs[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
        s[r0:r1] = rs
    del out_rows
    return F8.Fp8Weight(F8.permute_fp8(q), s, N, K)


def patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    mode = os.environ.get("R9K_FP8_BLOCK", "").lower()
    if mode not in ("rowwise", "block"):
        return False
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_w8a8_fp8 as S,
        )
        from compressed_tensors.quantization import QuantizationStrategy
    except Exception as e:
        logger.warning("r9700: fp8 block hook not installed (%s)", e)
        return False
    from ..moe.experts import r9k_available

    cls = S.CompressedTensorsW8A8Fp8
    orig_pwal = cls.process_weights_after_loading
    orig_apply = cls.apply_weights

    def process_weights_after_loading(self, layer):
        if not (self.strategy == QuantizationStrategy.BLOCK and r9k_available()):
            return orig_pwal(self, layer)
        w, bs = layer.weight.data, layer.weight_scale.data
        if w.dim() != 2 or w.shape[0] % 16 or w.shape[1] % 16:
            return orig_pwal(self, layer)
        blk = tuple(getattr(layer, "weight_block_size", None) or (128, 128))
        if mode == "block":
            # exact: same fp8 bytes and block scales, permuted into fragment order (same size)
            if tuple(blk) != (128, 128) or w.shape[1] % 128:
                return orig_pwal(self, layer)
            from ..kernels import fp8 as F8
            layer._r9k_fp8b = (F8.permute_fp8(w.view(torch.uint8)), bs.float().contiguous(), w.shape[0], w.shape[1])
        else:
            layer._r9k_fp8 = _requant_rowwise(w.view(torch.float8_e4m3fn), bs, blk)
        dev = w.device
        layer.weight = Parameter(torch.empty((0,), dtype=w.dtype, device=dev), requires_grad=False)
        layer.weight_scale = Parameter(torch.empty((0,), dtype=torch.float32, device=dev), requires_grad=False)
        layer.input_scale = None
        logger.info_once("r9700: block-fp8 linears -> libr9k split-K fp8 GEMM (R9K_FP8_BLOCK=%s)", mode)

    def apply_weights(self, layer, x, bias=None):
        Wb = getattr(layer, "_r9k_fp8b", None)
        if Wb is not None:
            if not isinstance(x, torch.Tensor):
                raise RuntimeError("r9700 R9K_FP8_BLOCK got a pre-quantized activation; unset R9K_FP8_BLOCK")
            from ..ops import fp8_block_linear
            out = fp8_block_linear(x, *Wb)
            return out + bias if bias is not None else out
        W = getattr(layer, "_r9k_fp8", None)
        if W is None:
            return orig_apply(self, layer, x, bias)
        if not isinstance(x, torch.Tensor):
            raise RuntimeError("r9700 R9K_FP8_BLOCK=rowwise got a pre-quantized activation (a quant-fusion pass is "
                               "on); unset R9K_FP8_BLOCK")
        from ..ops import fp8_linear
        out = fp8_linear(x, W)
        return out + bias if bias is not None else out

    cls.process_weights_after_loading = process_weights_after_loading
    cls.apply_weights = apply_weights
    _PATCHED = True
    return True
