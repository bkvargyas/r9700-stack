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


# ------------------------------------------------------------------------------ skinny fp32 mm (indexer weights)
def _skinny_lib():
    L = lib()
    if not hasattr(L.r9k_mm_f32_skinny, "_r9k_bound"):
        L.r9k_mm_f32_skinny.restype = ctypes.c_int
        L.r9k_mm_f32_skinny.argtypes = [ctypes.c_long] * 3 + [ctypes.c_int] * 3 + [ctypes.c_long] * 4
        L.r9k_mm_f32_skinny._r9k_bound = True
    return L


def mm_f32_skinny(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [T, K] fp32 @ w [K, N] fp32 (N <= 32) -> [T, N] fp32, fp32 accumulate; same math as torch.mm in fp32."""
    assert x.dtype == torch.float32 and w.dtype == torch.float32 and x.dim() == 2 and w.dim() == 2
    assert x.stride(1) == 1 and w.stride(1) == 1 and x.shape[1] == w.shape[0] and w.shape[1] <= 32
    out = torch.empty((x.shape[0], w.shape[1]), dtype=torch.float32, device=x.device)
    rc = _skinny_lib().r9k_mm_f32_skinny(x.data_ptr(), w.data_ptr(), out.data_ptr(), x.shape[0], x.shape[1], w.shape[1],
                                         x.stride(0), w.stride(0), out.stride(0), torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_mm_f32_skinny failed ({rc}) {tuple(x.shape)} x {tuple(w.shape)}")
    return out


class SkinnyWeight(torch.Tensor):
    """An fp32 [K, N] weight whose torch.mm(x, self) runs on r9k_mm_f32_skinny (every other op: plain tensor). Used
    for the indexer's cached head-weight projection, which the model computes with a 1 x 4096 x 32 fp32 torch.mm that
    costs ~200 us per call on hipBLASLt/rocBLAS at decode."""

    @staticmethod
    def __new__(cls, t: torch.Tensor):
        return torch.Tensor._make_subclass(cls, t, require_grad=False)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.mm and len(args) == 2 and isinstance(args[1], SkinnyWeight) and isinstance(args[0], torch.Tensor) \
                and args[0].dtype == torch.float32 and args[0].dim() == 2 and args[0].stride(1) == 1:
            w = torch.Tensor._make_subclass(torch.Tensor, args[1], require_grad=False)
            return mm_f32_skinny(args[0], w)
        plain = tuple(torch.Tensor._make_subclass(torch.Tensor, a, require_grad=False) if isinstance(a, SkinnyWeight) else a
                      for a in args)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*plain, **kwargs)
