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


def requant_rowwise(w8: torch.Tensor, bs: torch.Tensor, block=(128, 128)):
    from ..kernels import fp8 as F8
    N, K = w8.shape
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
    return F8.Fp8Weight(F8.permute_fp8(q), s, N, K)
