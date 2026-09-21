"""Binding for libr9k's paged causal attention (gfx1201): prefill / mixed-batch path of attn/triton3d.py.

Same positional call as the libr4d entries it replaces (see triton3d.py):
  fn(q_ptr, kv_ptr, block_table_ptr, seq_lens_ptr, out_ptr, k_descale_ptr, v_descale_ptr, scratch_ptr,
     nseq, q_len, num_q_heads, num_kv_heads, head_size, block_size, max_blocks, kv_stride0, kv_stride1, scale,
     sliding_window, max_ctx, stream)
"""
from __future__ import annotations

import ctypes
import os

_L = None


def lib() -> ctypes.CDLL:
    global _L
    if _L is None:
        path = os.environ.get("R9K_LIB") or os.path.join(os.path.dirname(__file__), "libr9k.so")
        L = ctypes.CDLL(path)
        argtypes = [ctypes.c_long] * 8 + [ctypes.c_int] * 7 + [ctypes.c_long] * 2 + [ctypes.c_float] + [ctypes.c_int] * 2 + [ctypes.c_long]
        for name in ("r9k_attn_prefill_paged", "r9k_attn_prefill_paged_fp8"):
            if hasattr(L, name):
                fn = getattr(L, name)
                fn.restype = ctypes.c_int
                fn.argtypes = argtypes
        _L = L
    return _L


def _checked(fn):
    def call(*args):
        rc = fn(*args)
        if rc != 0:
            raise RuntimeError(f"{fn.__name__} failed: {rc}")
        return rc
    call.__name__ = fn.__name__
    return call


def prefill_kernels() -> dict:
    """kv dtype name -> prefill callable (bf16 always; fp8_e4m3 when the library has it)."""
    L = lib()
    out = {"bf16": _checked(L.r9k_attn_prefill_paged)}
    if hasattr(L, "r9k_attn_prefill_paged_fp8"):
        out["fp8_e4m3"] = _checked(L.r9k_attn_prefill_paged_fp8)
    return out
