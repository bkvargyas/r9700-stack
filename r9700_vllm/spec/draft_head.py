"""Quantized MTP draft LM head (opt-in: R9K_DRAFT_LMHEAD=mxfp4).

With MTP-k, the drafter runs its LM head k times per step; at TP2 a bf16 head is ~635 MB read per call per rank
(vocab 248320 x 2560 / 2). This builds, after the MTP weights load, a shadow of the drafter's head quantized to
OCP MXFP4 (e2m1 + E8M0 per 32, round-to-nearest) and served by libr9k's MXFP4 x FP8 GEMM: ~4x fewer bytes. Only
the drafter uses it -- the target head is untouched, so output quality is unchanged; the cost is (possibly)
acceptance rate. Same idea as davetha's W4 draft head.
"""
from __future__ import annotations

import copy
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
_PATCHED = False

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


class _Mxfp4HeadMethod:
    def __init__(self, W, N, K):
        from ..linear.mxfp4 import R9700Mxfp4LinearKernel  # reuse its routing tables / launch
        self.W, self.N, self.K = W, N, K
        self._kernel = R9700Mxfp4LinearKernel.__new__(R9700Mxfp4LinearKernel)
        self._kernel._ids = {}

    def apply(self, layer, x, bias=None):
        fake = type("L", (), {})()
        fake.weight, fake.weight_scale, fake._r9k_nk = self.W.wq, self.W.wsr, (self.N, self.K)
        return self._kernel.apply_weights(fake, x, bias)


def _build_shadow(head):
    from ..kernels import moe as K
    w = head.weight.data
    packed, scale = quantize_mxfp4(w)
    W = K.prepare_mxfp4_weights(packed[None], scale[None])
    del packed, scale
    shadow = copy.copy(head)
    shadow.quant_method = _Mxfp4HeadMethod(W, w.shape[0], w.shape[1])
    shadow._r9k_W = W
    torch.cuda.empty_cache()
    logger.info_once("r9700: MTP draft LM head quantized to MXFP4 (%d x %d per rank)", w.shape[0], w.shape[1])
    return shadow


def patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("R9K_DRAFT_LMHEAD", "").lower() != "mxfp4":
        return False
    try:
        from vllm.models.qwen4_exp.amd import mtp as M
    except Exception as e:
        logger.warning("r9700: draft head hook not installed (%s)", e)
        return False
    cls = M.Qwen4ExpMTP
    orig_load = cls.load_weights

    def load_weights(self, weights):
        loaded = orig_load(self, weights)
        head = getattr(self, "lm_head", None)
        if head is not None and isinstance(getattr(head, "weight", None), torch.Tensor):
            self._r9k_head = _build_shadow(head)
        return loaded

    def compute_logits(self, hidden_states):
        head = getattr(self, "_r9k_head", None) or self.lm_head
        return self.logits_processor(head, hidden_states)

    cls.load_weights = load_weights
    cls.compute_logits = compute_logits
    if hasattr(cls, "get_top_tokens"):
        orig_top = cls.get_top_tokens

        def get_top_tokens(self, hidden_states):
            head = getattr(self, "_r9k_head", None)
            if head is None:
                return orig_top(self, hidden_states)
            return self.logits_processor.get_top_tokens(head, hidden_states)

        cls.get_top_tokens = get_top_tokens
    _PATCHED = True
    return True
