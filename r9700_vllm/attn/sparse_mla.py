"""Sparse rope-free MLA attention on libr9k (kernels/r9k_sparse_mla.hip): bf16 or fp8 e4m3 KV pool, ragged global slot
ids per token. The piece stock vLLM lacks on gfx1201 for an fp8 KV cache on GLM-5.3-Flash's 11 sparse layers."""
from __future__ import annotations

import ctypes
import os

import torch

_L = None


def lib():
    global _L
    if _L is None:
        from ..kernels.moe import lib as _lib      # the shared libr9k handle
        L = _lib()
        L.r9k_sparse_mla.restype = ctypes.c_int
        L.r9k_sparse_mla.argtypes = [ctypes.c_long] * 6 + [ctypes.c_int, ctypes.c_float] + [ctypes.c_int] * 3 + \
            [ctypes.c_long] * 5 + [ctypes.c_float, ctypes.c_int, ctypes.c_long]
        L.r9k_sparse_mla_scratch_bytes.restype = ctypes.c_long
        L.r9k_sparse_mla_scratch_bytes.argtypes = [ctypes.c_int] * 3
        _L = L
    return _L


def available() -> bool:
    try:
        return hasattr(lib(), "r9k_sparse_mla")
    except Exception:
        return False


_SCRATCH: dict = {}


def sparse_mla(q: torch.Tensor, pool: torch.Tensor, indices: torch.Tensor, indptr: torch.Tensor, scale: float,
               descale: float = 1.0, out: torch.Tensor | None = None, nsplit: int | None = None) -> torch.Tensor:
    """q [T, H, 512] bf16; pool [slots, 512] bf16 or float8_e4m3fn/uint8 (scalar descale); indices int32 flat global
    slot ids (-1 masked) ragged by indptr [T + 1] int32; returns [T, H, 512] bf16 (= out if given)."""
    assert q.dtype == torch.bfloat16 and q.dim() == 3 and q.shape[-1] == 512 and q.stride(-1) == 1
    T, H, D = q.shape
    pool2 = pool.view(pool.shape[0], -1) if pool.dim() != 2 else pool
    assert pool2.shape[1] == D and pool2.stride(1) == 1
    fp8 = pool2.dtype != torch.bfloat16
    if out is None:
        out = torch.empty((T, H, D), dtype=torch.bfloat16, device=q.device)
    assert out.stride(-1) == 1 and out.shape == (T, H, D)
    if nsplit is None:
        # decode-sized batches: spread one token's rows over several CUs; prefill: one block per token is plenty
        nsplit = int(os.environ.get("R9K_SPARSE_MLA_SPLITS", "8")) if T <= 16 else 1
    if H > 8:
        nsplit = 1
    scratch = 0
    if nsplit > 1:
        nb = lib().r9k_sparse_mla_scratch_bytes(T, H, nsplit)
        key = (q.device.index, nb)
        buf = _SCRATCH.get(key)
        if buf is None or buf.numel() < nb:
            buf = _SCRATCH[key] = torch.empty(nb, dtype=torch.uint8, device=q.device)
        scratch = buf.data_ptr()
    rc = lib().r9k_sparse_mla(q.data_ptr(), pool2.data_ptr(), indices.data_ptr(), indptr.data_ptr(), out.data_ptr(),
                              scratch, 1 if fp8 else 0, float(descale), T, H, D, q.stride(0), q.stride(1),
                              pool2.stride(0), out.stride(0), out.stride(1), float(scale), nsplit,
                              torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_sparse_mla failed ({rc}) T={T} H={H} fp8={fp8} nsplit={nsplit}")
    return out
