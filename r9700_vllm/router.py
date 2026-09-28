"""MoE router (gate) GEMM at decode widths on our kernel (kernels/r9k_router.hip).

vLLM's MoE runner computes the router logits with the model's gate module -- for Qwen3-Next / Flash-Next a plain
bf16 ReplicatedLinear (``mlp.gate``), i.e. F.linear through hipBLASLt, which picks a 16x16x32 tile for the
[M <= 32, 2560] x [512, 2560]^T shape: 19.6 us per call for a 2.6 MB weight (~130 GB/s), 51 calls per Flash-Next
decode step (1.0 ms, ~5% of the step). Ours streams the weight at one wave per (expert, K split): fp32
accumulation of exact bf16 products, rounded once to bf16 like stock, so it differs from hipBLASLt only in
summation order.

Registered as torch.ops.r9700.router_gemm (the gate sits inside the compiled model graph); ``install`` binds it
as the forward of every ``mlp.gate`` ReplicatedLinear with bf16 weights and no bias. Rows above R9K_ROUTER_MAX_M
(8: ours re-reads the activations per expert and loses to hipBLASLt from ~16 rows, tests/test_router_r9k.py) or
shapes the kernel does not take use hipBLASLt's bf16 x bf16 -> fp32 out_dtype GEMM rounded to bf16 inside the op
(7.5 us at 32 rows where F.linear's bf16 tile takes 19.6), so the branch is taken at run time, not at trace time.
R9K_ROUTER=stock keeps vLLM's path.
"""
from __future__ import annotations

import ctypes
import os
import types

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger("vllm." + __name__)

_L = None
_DONE = False
MAX_M = int(os.environ.get("R9K_ROUTER_MAX_M", "8"))
SPLIT = int(os.environ.get("R9K_ROUTER_SPLIT", "2"))


def lib():
    global _L
    if _L is None:
        from .kernels import moe as KM
        L = KM.lib()
        L.r9k_router_gemm.restype = ctypes.c_int
        L.r9k_router_gemm.argtypes = [ctypes.c_long] * 6 + [ctypes.c_int] * 6 + [ctypes.c_float, ctypes.c_long]
        L.r9k_router_max_m.restype = ctypes.c_int
        _L = L
    return _L


def available() -> bool:
    try:
        return hasattr(lib(), "r9k_router_gemm")
    except Exception:
        return False


def _fits(x: torch.Tensor, w: torch.Tensor, split: int, max_m: int | None = None) -> bool:
    M, K = x.shape
    return (1 <= M <= min(MAX_M if max_m is None else max_m, lib().r9k_router_max_m()) and x.dtype == torch.bfloat16
            and w.dtype == torch.bfloat16
            and w.dim() == 2 and w.shape[1] == K and w.shape[0] >= 1 and K % (8 * split) == 0
            and x.stride(1) == 1 and w.stride(1) == 1 and x.stride(0) % 8 == 0 and w.stride(0) % 8 == 0)


def router_gemm(x: torch.Tensor, w: torch.Tensor, out_bf16: bool = True, split: int | None = None,
                act_cols: int = 0, hc: float = 1.0, max_m: int | None = None) -> torch.Tensor:
    """[M, E] = x [M, K] bf16 . w [E, K]^T bf16, bf16 (stock's rounding) or fp32 out; hipBLASLt's fp32-out GEMM
    when the kernel does not apply (act_cols > 0 must then be handled by the caller: returns None)."""
    split = SPLIT if split is None else split
    if not _fits(x, w, split, max_m):
        if act_cols > 0:
            return None
        out = torch.mm(x, w.t(), out_dtype=torch.float32)
        return out.to(torch.bfloat16) if out_bf16 else out
    M, K = x.shape
    E = w.shape[0]
    out = torch.empty((M, E), dtype=torch.bfloat16 if out_bf16 else torch.float32, device=x.device)
    rc = lib().r9k_router_gemm(x.data_ptr(), x.stride(0), w.data_ptr(), w.stride(0), out.data_ptr(), out.stride(0),
                               M, E, K, split, 1 if out_bf16 else 0, int(act_cols), float(hc),
                               torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_router_gemm failed ({rc}) M={M} E={E} K={K} split={split}")
    return out


def _router_gemm_op(x: torch.Tensor, w: torch.Tensor, out_bf16: bool) -> torch.Tensor:
    return router_gemm(x, w, out_bf16)


def _router_gemm_fake(x: torch.Tensor, w: torch.Tensor, out_bf16: bool) -> torch.Tensor:
    return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16 if out_bf16 else torch.float32)


def register() -> None:
    global _DONE
    if _DONE:
        return
    from .ops import _LIB
    direct_register_custom_op("router_gemm", _router_gemm_op, mutates_args=[], fake_impl=_router_gemm_fake,
                              target_lib=_LIB)
    _DONE = True


def _forward(self, x: torch.Tensor):
    # ReplicatedLinear.forward's contract with return_bias: (output, output_bias); the gate has no bias
    return torch.ops.r9700.router_gemm(x, self.weight, True), None


def install(model: torch.nn.Module) -> int:
    """Bind our op as the forward of every ``mlp.gate`` under `model` that is an unquantized bf16 linear without
    bias. Returns the count."""
    if os.environ.get("R9K_ROUTER", "r9k") != "r9k" or not available():
        return 0
    register()
    n = 0
    for name, mod in model.named_modules():
        if not name.endswith("mlp.gate") or getattr(mod, "_r9k_router", False):
            continue
        w = getattr(mod, "weight", None)
        qm = type(getattr(mod, "quant_method", None)).__name__
        if type(mod).__name__ not in ("ReplicatedLinear", "GateLinear") or not isinstance(w, torch.Tensor) \
                or w.dtype != torch.bfloat16 or w.dim() != 2 or getattr(mod, "bias", None) is not None \
                or qm != "UnquantizedLinearMethod" or getattr(mod, "out_dtype", None) not in (None, torch.bfloat16) \
                or not getattr(mod, "return_bias", True):
            logger.warning_once("r9700: router GEMM skipped for %s (%s, %s, weight %s)", name, type(mod).__name__,
                                qm, None if w is None else tuple(w.shape))
            continue
        mod.forward = types.MethodType(_forward, mod)
        mod._r9k_router = True
        n += 1
    if n:
        logger.info("r9700: r9k router GEMM installed on %d gates (M <= %d, split %d)", n, MAX_M, SPLIT)
    return n
