"""Hyper-connection glue kernels (kernels/r9k_hc.hip) in place of vLLM's Triton ``hc_gate_mix`` / ``hc_combine_norm``.

In a 4096-token prefill chunk the stock kernels are 88 ms of the 653 ms GPU-busy: one Triton program per row (and
per stream) over 512-wide blocks reaches ~470 GB/s on the 273 MB combine_norm moves (575 us x 97) and ~590 GB/s on
gate_mix (322 us x 100). Ours are one wave per (row, stream) / (row, 256 columns) with 16 B loads and stores.

Registered as torch.ops.r9700.hc_gate_mix / hc_combine_norm (custom ops: the callers sit inside the compiled model
graph) and bound over the names ``hyperconnection.py`` imported (``install``). Same bf16 rounding points as stock
(the combine result is rounded to bf16 before the norm). Geometry outside the kernels' reach (HD % 8, HD <= 4096,
bf16) and small row counts (R9K_HC_MIN_ROWS / _GATE: decode widths, where stock is at the launch floor) fall back
to the stock op per call. R9K_HC=stock keeps vLLM's kernels.
"""
from __future__ import annotations

import ctypes
import os

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger("vllm." + __name__)

_L = None
_DONE = False
# Below these row counts the stock Triton kernels are at the launch floor (~3.4 us in graph replay) and ours are
# 0.5-2 us slower (one workgroup per (row, stream) still pays an LDS reduction); ours win from a few hundred rows
# (4096 rows: combine_norm 607 -> 469 us, gate_mix 318 -> 310). Decode / MTP verify (1-64 rows) stay on stock.
MIN_ROWS_COMBINE = int(os.environ.get("R9K_HC_MIN_ROWS", "256"))
MIN_ROWS_GATE = int(os.environ.get("R9K_HC_MIN_ROWS_GATE", "1024"))


def lib():
    global _L
    if _L is None:
        from .kernels import moe as KM
        L = KM.lib()
        L.r9k_hc_combine_norm.restype = ctypes.c_int
        L.r9k_hc_combine_norm.argtypes = [ctypes.c_long] * 7 + [ctypes.c_int] + [ctypes.c_long] * 4 + [ctypes.c_int] * 3 + \
            [ctypes.c_float, ctypes.c_long]
        L.r9k_hc_gate_mix.restype = ctypes.c_int
        L.r9k_hc_gate_mix.argtypes = [ctypes.c_long] * 6 + [ctypes.c_int] * 3 + [ctypes.c_long]
        _L = L
    return _L


def available() -> bool:
    try:
        return hasattr(lib(), "r9k_hc_combine_norm")
    except Exception:
        return False


def _fits(*ts, hd: int) -> bool:
    return hd % 8 == 0 and hd <= 4096 and all(t.dtype == torch.bfloat16 and t.stride(-1) == 1 for t in ts)


def gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    N, DIM = gate.shape
    HD = DIM // hc_count
    if N < MIN_ROWS_GATE or not _fits(x, gate, hd=HD) or x.shape != gate.shape:
        return torch.ops.vllm.qwen4_exp_hc_gate_mix(x, gate, hc_count)
    y = x.new_empty((N, HD))
    rc = lib().r9k_hc_gate_mix(x.data_ptr(), x.stride(0), gate.data_ptr(), gate.stride(0), y.data_ptr(), y.stride(0),
                               N, hc_count, HD, torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_hc_gate_mix failed ({rc})")
    return y


def combine_norm(residual: torch.Tensor, block_output: torch.Tensor, injection_logits: torch.Tensor,
                 norm_weight: torch.Tensor, eps: float, hc_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    N, DIM = residual.shape
    HD = DIM // hc_count
    if N < MIN_ROWS_COMBINE or not _fits(residual, block_output, injection_logits, norm_weight, hd=HD) \
            or block_output.shape != (N, HD) \
            or injection_logits.shape != (N, hc_count) or norm_weight.numel() not in (HD, DIM) \
            or not norm_weight.is_contiguous():
        return torch.ops.vllm.qwen4_exp_hc_combine_norm(residual, block_output, injection_logits, norm_weight, eps,
                                                        hc_count)
    out = residual.new_empty(residual.shape)
    y = residual.new_empty(residual.shape)
    rc = lib().r9k_hc_combine_norm(residual.data_ptr(), residual.stride(0), block_output.data_ptr(),
                                   block_output.stride(0), injection_logits.data_ptr(), injection_logits.stride(0),
                                   norm_weight.data_ptr(), 1 if norm_weight.numel() == HD else 0,
                                   out.data_ptr(), out.stride(0), y.data_ptr(), y.stride(0), N, hc_count, HD, float(eps),
                                   torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_hc_combine_norm failed ({rc})")
    return out, y


def _gate_mix_fake(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    return x.new_empty((x.shape[0], x.shape[1] // hc_count))


def _combine_norm_fake(residual: torch.Tensor, block_output: torch.Tensor, injection_logits: torch.Tensor,
                       norm_weight: torch.Tensor, eps: float, hc_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    return residual.new_empty(residual.shape), residual.new_empty(residual.shape)


def register() -> None:
    global _DONE
    if _DONE:
        return
    from .ops import _LIB
    direct_register_custom_op("hc_gate_mix", gate_mix, mutates_args=[], fake_impl=_gate_mix_fake, target_lib=_LIB)
    direct_register_custom_op("hc_combine_norm", combine_norm, mutates_args=[], fake_impl=_combine_norm_fake,
                              target_lib=_LIB)
    _DONE = True


def op_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    return torch.ops.r9700.hc_gate_mix(x, gate, hc_count)


def op_combine_norm(residual: torch.Tensor, block_output: torch.Tensor, injection_logits: torch.Tensor,
                    norm_weight: torch.Tensor, eps: float, hc_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.r9700.hc_combine_norm(residual, block_output, injection_logits, norm_weight, eps, hc_count)


def install() -> bool:
    """Point hyperconnection.py's imported names at our ops (before the model is traced). Idempotent."""
    if os.environ.get("R9K_HC", "r9k") != "r9k" or not available():
        return False
    from vllm.models.qwen4_exp.amd import hyperconnection as H
    if getattr(H, "_r9k_hc", False):
        return True
    register()
    H.hc_gate_mix = op_gate_mix
    H.hc_combine_norm = op_combine_norm
    H._r9k_hc = True
    logger.info("r9700: r9k hyper-connection gate_mix / combine_norm installed")
    return True
