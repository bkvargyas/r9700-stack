"""torch.library custom ops wrapping libr9k's ctypes kernels, so they can sit inside vLLM's torch.compile'd model
graph (Dynamo cannot trace ctypes calls) and be captured into CUDA/HIP graphs.

  torch.ops.r9700.fp8_linear(x[M,K] bf16, wq, ws, N, K) -> [M,N] bf16     per-row fp8 weight x per-token fp8 act
  torch.ops.r9700.mxfp4_linear(x[M,K] bf16, wq, wsr, N, K) -> [M,N] bf16  MXFP4 weight x per-token fp8 act
"""
from __future__ import annotations

import torch

from vllm.utils.torch_utils import direct_register_custom_op

_LIB = torch.library.Library("r9700", "FRAGMENT")
_DONE = False


def _fp8_linear(x: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor, N: int, K: int) -> torch.Tensor:
    from .kernels import fp8 as F8, moe as KM
    q, s = KM.quant_rows_fp8(x)
    return F8.gemm_fp8(q, s, F8.Fp8Weight(wq, ws, N, K))


def _fp8_linear_fake(x: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor, N: int, K: int) -> torch.Tensor:
    return x.new_empty((x.shape[0], N))


_MX_TABLES: dict[tuple, tuple] = {}


def _mxfp4_linear(x: torch.Tensor, wq: torch.Tensor, wsr: torch.Tensor, N: int, K: int) -> torch.Tensor:
    from .kernels import moe as KM
    M = x.shape[0]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    if M == 0:
        return out
    q, s = KM.quant_rows_fp8(x)
    MT = 4 if M >= 64 else (2 if M >= 32 else 1)
    blk = KM.MOE_BLOCK * MT
    mpad = (M + blk - 1) // blk * blk
    key = (x.device.index, mpad, blk)
    t = _MX_TABLES.get(key)
    if t is None:
        t = (torch.arange(mpad, dtype=torch.int32, device=x.device),
             torch.zeros(mpad // blk, dtype=torch.int32, device=x.device),
             torch.full((1,), mpad, dtype=torch.int32, device=x.device))
        _MX_TABLES[key] = t
    KM.moe_gemm(q, s, KM.Mxfp4Experts(wq, wsr, N, K), out, *t, M, 1, None, *KM.pick_cfg(N, K), num_experts=1, MT=MT)
    return out


def _mxfp4_linear_fake(x: torch.Tensor, wq: torch.Tensor, wsr: torch.Tensor, N: int, K: int) -> torch.Tensor:
    return x.new_empty((x.shape[0], N))


def register() -> None:
    global _DONE
    if _DONE:
        return
    direct_register_custom_op("fp8_linear", _fp8_linear, mutates_args=[], fake_impl=_fp8_linear_fake, target_lib=_LIB)
    direct_register_custom_op("mxfp4_linear", _mxfp4_linear, mutates_args=[], fake_impl=_mxfp4_linear_fake,
                              target_lib=_LIB)
    _DONE = True


def fp8_linear(x: torch.Tensor, W) -> torch.Tensor:
    register()
    lead = x.shape[:-1]
    x2 = x.reshape(-1, W.K)
    x2 = (x2 if x2.dtype == torch.bfloat16 else x2.to(torch.bfloat16)).contiguous()
    return torch.ops.r9700.fp8_linear(x2, W.wq, W.ws, W.N, W.K).reshape(*lead, W.N)


def mxfp4_linear(x: torch.Tensor, wq: torch.Tensor, wsr: torch.Tensor, N: int, K: int) -> torch.Tensor:
    register()
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K)
    x2 = (x2 if x2.dtype == torch.bfloat16 else x2.to(torch.bfloat16)).contiguous()
    return torch.ops.r9700.mxfp4_linear(x2, wq, wsr, N, K).reshape(*lead, N)
