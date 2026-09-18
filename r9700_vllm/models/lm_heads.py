"""Quantized LM heads for Qwen4Exp (Flash-Next) on stock vLLM, served by libr9k (opt-in).

  R9K_TARGET_LMHEAD=fp8          target head: per-row-scaled e4m3 (davetha measured -1.0 ms/step for fp8)
  R9K_DRAFT_LMHEAD=fp8|mxfp4     MTP draft head (runs k times per step with MTP-k)

At TP2 a bf16 head is ~635 MB read per call per rank (vocab 248320 x hidden 2560 / 2). After the weights load,
the model classes in models/qwen4_exp.py quantize a shadow of the head on the GPU and use it for logits; the bf16 original stays (it is also the
embedding for tied models and what the proposer compares when deciding to share heads). fp8 halves the bytes,
mxfp4 quarters them; draft-only quantization cannot change output quality, only MTP acceptance.
"""
from __future__ import annotations

import copy
import torch

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)

_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])   # e2m1 decision thresholds on |x|/scale


def quantize_mxfp4(w: torch.Tensor, rows_per_chunk: int = 4096):
    """bf16/fp16/fp32 [N, K] -> (packed [N, K/2] u8, e8m0 [N, K/32] u8), K % 32 == 0."""
    N, Kd = w.shape
    packed = torch.empty((N, Kd // 2), dtype=torch.uint8, device=w.device)
    scale = torch.empty((N, Kd // 32), dtype=torch.uint8, device=w.device)
    mid = _MID.to(w.device)
    for r0 in range(0, N, rows_per_chunk):
        x = w[r0:r0 + rows_per_chunk].float().reshape(-1, Kd // 32, 32)
        amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(2.0 ** -126)
        e = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127)
        v = x / torch.exp2(e)
        code = torch.bucketize(v.abs(), mid).to(torch.uint8) | ((v < 0).to(torch.uint8) << 3)
        code = code.reshape(x.shape[0], Kd)
        packed[r0:r0 + rows_per_chunk] = code[:, 0::2] | (code[:, 1::2] << 4)
        scale[r0:r0 + rows_per_chunk] = (e.squeeze(-1) + 127).to(torch.uint8)
    return packed, scale


class _HeadMethod:
    """Stand-in quant method: LogitsProcessor only calls .apply(layer, x, bias)."""

    def __init__(self, fmt: str, head: torch.nn.Module):
        from ..kernels import fp8 as F8, moe as K
        w = head.weight.data
        self.fmt, self.N, self.K = fmt, w.shape[0], w.shape[1]
        pad = (-self.N) % 16                               # kernels need N % 16 == 0
        if pad:
            w = torch.cat([w, w.new_zeros((pad, self.K))])
        if fmt == "fp8":
            self.W = F8.quantize_rows_fp8(w)
        elif fmt == "mxfp4":
            packed, scale = quantize_mxfp4(w)
            self.W = K.prepare_mxfp4_weights(packed[None], scale[None])
            from ..linear.mxfp4 import R9700Mxfp4LinearKernel
            self._lin = R9700Mxfp4LinearKernel.__new__(R9700Mxfp4LinearKernel)
            self._lin._ids = {}
        else:
            raise ValueError(fmt)
        self.Np = w.shape[0]

    def apply(self, layer, x, bias=None):
        from ..kernels import fp8 as F8, moe as K
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.K)
        x2 = (x2 if x2.dtype == torch.bfloat16 else x2.to(torch.bfloat16)).contiguous()
        if self.fmt == "fp8":
            q, s = K.quant_rows_fp8(x2)
            out = F8.gemm_fp8(q, s, self.W)
        else:
            fake = type("L", (), {})()
            fake.weight, fake.weight_scale, fake._r9k_nk = self.W.wq, self.W.wsr, (self.Np, self.K)
            out = self._lin.apply_weights(fake, x2)
        if self.Np != self.N:
            out = out[:, : self.N]
        if bias is not None:
            out = out + bias
        return out.reshape(*lead, self.N)


def shadow(head, fmt: str, what: str):
    shadow = copy.copy(head)
    shadow.quant_method = _HeadMethod(fmt, head)
    torch.cuda.empty_cache()
    logger.info_once("r9700: %s LM head -> %s shadow (%d x %d per rank)", what, fmt,
                     head.weight.shape[0], head.weight.shape[1])
    return shadow
