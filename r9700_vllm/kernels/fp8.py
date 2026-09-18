"""Binding for libr9k's dense FP8 x FP8 (per-row scales) skinny GEMM, used for LM heads."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import torch

from .moe import lib as _lib, _stream

_BOUND = False
FP8_MAX = 448.0


def _L():
    global _BOUND
    L = _lib()
    if not _BOUND:
        L.r9k_gemm_fp8.restype = ctypes.c_int
        L.r9k_gemm_fp8.argtypes = [ctypes.c_long] * 5 + [ctypes.c_int] * 8 + [ctypes.c_long]
        L.r9k_gemm_fp8_block.restype = ctypes.c_int
        L.r9k_gemm_fp8_block.argtypes = [ctypes.c_long] * 5 + [ctypes.c_int] * 8 + [ctypes.c_long]
        L.r9k_quant_group128_fp8.restype = ctypes.c_int
        L.r9k_quant_group128_fp8.argtypes = [ctypes.c_long] * 3 + [ctypes.c_int] * 3 + [ctypes.c_long]
        _BOUND = True
    return L


@dataclass
class Fp8Weight:
    wq: torch.Tensor   # [N/16 * K/16 * 32 * 2] int32 (fragment order, 8 bytes per lane per k-step)
    ws: torch.Tensor   # [N] fp32 per-row scale
    N: int
    K: int


def permute_fp8(w8: torch.Tensor) -> torch.Tensor:
    """[N, K] uint8/e4m3 row-major -> fragment order (see r9k_gemm_fp8.hip)."""
    N, Kd = w8.shape
    nt, ks = N // 16, Kd // 16
    w = w8.view(torch.uint8).reshape(nt, 16, ks, 2, 8)          # [nt, r, ks, h, 8 bytes]
    return w.permute(0, 2, 3, 1, 4).contiguous().reshape(-1).view(torch.int32)  # [nt, ks, h, r, 8]


def quantize_rows_fp8(w: torch.Tensor, rows_per_chunk: int = 8192) -> Fp8Weight:
    """bf16/fp32 [N, K] -> per-row-scaled e4m3 weight in the kernel layout."""
    N, Kd = w.shape
    assert N % 16 == 0 and Kd % 16 == 0, (N, Kd)
    q = torch.empty((N, Kd), dtype=torch.uint8, device=w.device)
    s = torch.empty((N,), dtype=torch.float32, device=w.device)
    for r0 in range(0, N, rows_per_chunk):
        x = w[r0:r0 + rows_per_chunk].float()
        amax = x.abs().amax(dim=1).clamp_min(1e-12)
        sc = amax / FP8_MAX
        q[r0:r0 + rows_per_chunk] = (x / sc[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
        s[r0:r0 + rows_per_chunk] = sc
    return Fp8Weight(permute_fp8(q), s, N, Kd)


def gemm_fp8(a_q: torch.Tensor, a_s: torch.Tensor, w: Fp8Weight, out: torch.Tensor | None = None,
             WV: int = 4, SK: int = 4, NPW: int = 2):
    M = a_q.shape[0]
    if out is None:
        out = torch.empty((M, w.N), dtype=torch.bfloat16, device=a_q.device)
    MT = 4 if M > 32 else (2 if M > 16 else 1)
    rc = _L().r9k_gemm_fp8(a_q.data_ptr(), a_s.data_ptr(), w.wq.data_ptr(), w.ws.data_ptr(), out.data_ptr(),
                           M, w.K, w.N, out.stride(0), WV, SK, NPW, MT, _stream())
    if rc:
        raise RuntimeError(f"r9k_gemm_fp8 failed ({rc}) M={M} N={w.N} K={w.K}")
    return out


def pick_cfg(kind: str, N: int, K: int, M: int) -> tuple[int, int, int]:
    """kind 'fp8row' | 'fp8block': tuned (tuned.json) or the historical default (4, 4, 2)."""
    from .tuned import lookup
    t = lookup(kind, N, K, M)
    return t if t else (4, 4, 2)


def quant_group128_fp8(x: torch.Tensor):
    """bf16 [M, K] -> (e4m3 [M, K], fp32 [M, K/128]) per-token-group-128 dynamic scales."""
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.stride(1) == 1 and x.shape[1] % 128 == 0
    M, K = x.shape
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((M, K // 128), dtype=torch.float32, device=x.device)
    rc = _L().r9k_quant_group128_fp8(x.data_ptr(), q.data_ptr(), s.data_ptr(), M, K, x.stride(0), _stream())
    if rc:
        raise RuntimeError(f"r9k_quant_group128_fp8 failed ({rc})")
    return q, s


def gemm_fp8_block(a_q: torch.Tensor, a_s: torch.Tensor, wq: torch.Tensor, bs: torch.Tensor, N: int, K: int,
                   out: torch.Tensor | None = None, WV: int = 4, SK: int = 4, NPW: int = 2):
    """Block-scaled fp8: a_s [M, K/128], bs [ceil(N/128), K/128] (weight in permute_fp8 fragment order)."""
    M = a_q.shape[0]
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=a_q.device)
    while SK > 1 and K % (128 * SK):
        SK //= 2
    MT = 4 if M > 32 else (2 if M > 16 else 1)
    rc = _L().r9k_gemm_fp8_block(a_q.data_ptr(), a_s.data_ptr(), wq.data_ptr(), bs.data_ptr(), out.data_ptr(),
                                 M, K, N, out.stride(0), WV, SK, NPW, MT, _stream())
    if rc:
        raise RuntimeError(f"r9k_gemm_fp8_block failed ({rc}) M={M} N={N} K={K}")
    return out
