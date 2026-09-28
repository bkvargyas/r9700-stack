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
import types

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
# Decode mix (kernels/r9k_router.hip epilogue + r9k_hc_up_mix): the two 6.5 MB skinny GEMMs of every block's input
# mix with silu / sigmoid + gated mean fused in, 2 launches instead of 4 (wvSplitK x2 + hc_silu + hc_gate_mix) for
# rows <= R9K_HC_MIX_MAX_M; above, stock's path inside the same ops (runtime branch, graph-safe).
MIX_MAX_M = int(os.environ.get("R9K_HC_MIX_MAX_M", "8"))
MIX_SPLIT = int(os.environ.get("R9K_HC_MIX_SPLIT", "8"))      # K splits per row of the 336 x 10240 down GEMM
MIX_LPR = int(os.environ.get("R9K_HC_MIX_LPR", "4"))          # lanes per weight row of the up GEMM (2 or 4)


def lib():
    global _L
    if _L is None:
        from .kernels import moe as KM
        L = KM.lib()
        if hasattr(L, "r9k_hc_up_mix"):
            L.r9k_hc_up_mix.restype = ctypes.c_int
            L.r9k_hc_up_mix.argtypes = [ctypes.c_long] * 8 + [ctypes.c_int] * 5 + [ctypes.c_long]
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


def mix_available() -> bool:
    try:
        return hasattr(lib(), "r9k_hc_up_mix") and hasattr(lib(), "r9k_router_gemm")
    except Exception:
        return False


def down_silu(xn: torch.Tensor, w: torch.Tensor, lora_rank: int, hc_count: int) -> torch.Tensor:
    """[M, N] bf16 = xn [M, K] . w [N, K]^T with hc_silu(., hc) applied to the first lora_rank columns (the merged
    down + injection GEMM); stock's F.linear + hc_silu above MIX_MAX_M rows."""
    from . import router as R
    out = R.router_gemm(xn, w, True, MIX_SPLIT, lora_rank, float(hc_count), MIX_MAX_M) if mix_available() else None
    if out is None:
        out = torch.nn.functional.linear(xn, w)
        out[:, :lora_rank] = torch.ops.vllm.qwen4_exp_hc_silu(out[:, :lora_rank], hc_count)
    return out


def up_mix(lora: torch.Tensor, w_up: torch.Tensor, xn: torch.Tensor, hc_count: int, lpr: int | None = None
            ) -> torch.Tensor:
    """[M, HD] bf16 = hc_gate_mix(xn, lora . w_up^T, hc): the up GEMM, sigmoid and gated mean in one launch."""
    lpr = MIX_LPR if lpr is None else lpr
    M, LR = lora.shape
    DIM = xn.shape[1]
    HD = DIM // hc_count
    ok = (mix_available() and 1 <= M <= MIX_MAX_M and hc_count == 4 and LR % (8 * lpr) == 0 and 16 <= LR <= 512
          and HD % 2 == 0 and w_up.shape == (DIM, LR) and lora.dtype == w_up.dtype == xn.dtype == torch.bfloat16
          and lora.stride(1) == 1 and w_up.stride(1) == 1 and xn.stride(1) == 1
          and lora.stride(0) % 8 == 0 and w_up.stride(0) % 8 == 0 and xn.stride(0) % 8 == 0)
    if not ok:
        gate = torch.nn.functional.linear(lora, w_up)
        return torch.ops.vllm.qwen4_exp_hc_gate_mix(xn, gate, hc_count)
    out = xn.new_empty((M, HD))
    rc = lib().r9k_hc_up_mix(lora.data_ptr(), lora.stride(0), w_up.data_ptr(), w_up.stride(0), xn.data_ptr(),
                             xn.stride(0), out.data_ptr(), out.stride(0), M, LR, hc_count, HD, lpr,
                             torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_hc_up_mix failed ({rc}) M={M} LR={LR} HD={HD}")
    return out


def _down_silu_fake(xn: torch.Tensor, w: torch.Tensor, lora_rank: int, hc_count: int) -> torch.Tensor:
    return xn.new_empty((xn.shape[0], w.shape[0]))


def _up_mix_fake(lora: torch.Tensor, w_up: torch.Tensor, xn: torch.Tensor, hc_count: int) -> torch.Tensor:
    return xn.new_empty((xn.shape[0], xn.shape[1] // hc_count))


def register() -> None:
    global _DONE
    if _DONE:
        return
    from .ops import _LIB
    direct_register_custom_op("hc_gate_mix", gate_mix, mutates_args=[], fake_impl=_gate_mix_fake, target_lib=_LIB)
    direct_register_custom_op("hc_combine_norm", combine_norm, mutates_args=[], fake_impl=_combine_norm_fake,
                              target_lib=_LIB)
    direct_register_custom_op("hc_down_silu", down_silu, mutates_args=[], fake_impl=_down_silu_fake, target_lib=_LIB)
    direct_register_custom_op("hc_up_mix", up_mix, mutates_args=[], fake_impl=_up_mix_fake, target_lib=_LIB)
    _DONE = True


def op_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    return torch.ops.r9700.hc_gate_mix(x, gate, hc_count)


def op_combine_norm(residual: torch.Tensor, block_output: torch.Tensor, injection_logits: torch.Tensor,
                    norm_weight: torch.Tensor, eps: float, hc_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.r9700.hc_combine_norm(residual, block_output, injection_logits, norm_weight, eps, hc_count)


def _mix_tail(self, xn: torch.Tensor):
    lr, hc = self.lora_rank, self.hc_count
    if self.use_combine:
        buf = torch.ops.r9700.hc_down_silu(xn, self.input_mix_weight_down_block_inject.weight, lr, hc)
        injection = buf[:, lr:lr + hc]
    else:
        buf = torch.ops.r9700.hc_down_silu(xn, self.input_mix_weight_down.weight, lr, hc)
        injection = None
    block_input = torch.ops.r9700.hc_up_mix(buf[:, :lr], self.input_mix_weight_up.weight, xn, hc)
    return block_input, injection


def _mix(self, hidden_states: torch.Tensor):
    from vllm.models.qwen4_exp.amd import hyperconnection as H
    xn = H.grouped_gemma_rmsnorm(hidden_states, self.hc_norm.weight, self.config.rms_norm_eps, self.hc_count)
    block_input, injection = _mix_tail(self, xn)
    return hidden_states, block_input, injection


def _combine_and_mix(self, hidden_states: torch.Tensor, prev_block_output: torch.Tensor, prev_injection: torch.Tensor):
    from vllm.models.qwen4_exp.amd import hyperconnection as H
    hidden_states, xn = H.hc_combine_norm(hidden_states, prev_block_output, prev_injection, self.hc_norm.weight,
                                          self.config.rms_norm_eps, self.hc_count)
    block_input, injection = _mix_tail(self, xn)
    return hidden_states, block_input, injection


def install_mix(model: torch.nn.Module) -> int:
    """Bind our mix / combine_and_mix (down GEMM + silu, up GEMM + sigmoid + gated mean: 2 launches) on every
    GatedResidual under `model` whose linears are plain bf16 vLLM linears. R9K_HC_MIX=stock keeps vLLM's."""
    if os.environ.get("R9K_HC_MIX", "r9k") != "r9k" or not mix_available():
        return 0
    register()
    n = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ != "GatedResidual" or getattr(mod, "_r9k_mix", False):
            continue
        down = getattr(mod, "input_mix_weight_down_block_inject", None) if getattr(mod, "use_combine", False) \
            else getattr(mod, "input_mix_weight_down", None)
        up = getattr(mod, "input_mix_weight_up", None)
        ok = all(m is not None and isinstance(getattr(m, "weight", None), torch.Tensor)
                 and m.weight.dtype == torch.bfloat16 and getattr(m, "bias", None) is None
                 and type(getattr(m, "quant_method", None)).__name__ == "UnquantizedLinearMethod"
                 for m in (down, up)) and hasattr(mod, "hc_norm") and hasattr(mod, "config") \
            and getattr(mod, "hc_count", 0) == 4 and all(hasattr(mod, a) for a in ("mix", "combine_and_mix"))
        if not ok:
            logger.warning_once("r9700: hc mix skipped for %s", name)
            continue
        mod.mix = types.MethodType(_mix, mod)
        mod.combine_and_mix = types.MethodType(_combine_and_mix, mod)
        mod._r9k_mix = True
        n += 1
    if n:
        logger.info("r9700: r9k hyper-connection mix (down+silu, up+sigmoid+mean) installed on %d modules (M <= %d)",
                    n, MIX_MAX_M)
    return n


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
