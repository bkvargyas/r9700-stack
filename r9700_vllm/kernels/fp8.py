"""Binding for libr9k's dense FP8 x FP8 (per-row scales) skinny GEMM, used for LM heads."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import os

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
        if hasattr(L, "r9k_fp8_prefill_mix"):
            L.r9k_fp8_prefill_mix.restype = ctypes.c_int
            L.r9k_fp8_prefill_mix.argtypes = [ctypes.c_long] * 10 + [ctypes.c_int] * 5 + [ctypes.c_long]
        if hasattr(L, "r9k_fp8_prefill_block"):
            L.r9k_fp8_prefill_block.restype = ctypes.c_int
            L.r9k_fp8_prefill_block.argtypes = [ctypes.c_long] * 8 + [ctypes.c_int] * 7 + [ctypes.c_long]
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


# Tile cfgs (kPfCfgs in r9k_moe_4bit_prefill): 9 = 256 x 128 / 8 waves / BK=32 double-buffered is the sweep's best
# on every served shape at 4096 rows (tests/test_fp8_prefill_r9k.py, 2026-10-07: 163-182 TFLOPS); smaller M-tiles
# below 512 rows so a tile is not mostly padding. R9K_FP8_PREFILL_CFGS="N,K=cfg;..." overrides per shape.
PREFILL_CFG = int(os.environ.get("R9K_FP8_PREFILL_CFG", "9"))
PREFILL_MIN_M = int(os.environ.get("R9K_FP8_PREFILL_MINM", "64"))   # below: the split-K decode kernel
MIX_CFG = int(os.environ.get("R9K_FP8_MIX_CFG", "9"))               # TN == 4 cfgs only: 1, 9, 16, 17, 19
BLOCK_CFG = int(os.environ.get("R9K_FP8_BLOCK_CFG", "10"))          # TM*TN <= 8 cfgs only (two accumulator sets)
_TILE: dict[tuple[int, int], int] = {}
for _kv in os.environ.get("R9K_FP8_PREFILL_CFGS", "").split(";"):
    if "=" in _kv:
        _nk, _c = _kv.split("=")
        _TILE[tuple(int(v) for v in _nk.split(","))] = int(_c)


def tile_cfg(N: int, K: int, M: int, mix: bool = False, block: bool = False) -> int:
    c = _TILE.get((N, K))
    if c is not None:
        return c
    if M < 512:
        return 17 if (mix or block or M < 256) else 10              # 64 x 128 (4 waves) / 128 x 128 (8 waves)
    return MIX_CFG if mix else (BLOCK_CFG if block else PREFILL_CFG)


def gemm_fp8_block_tiled(a_q: torch.Tensor, a_s: torch.Tensor, wq: torch.Tensor, bs: torch.Tensor, N: int, K: int,
                         out: torch.Tensor | None = None, cfg: int | None = None, block: int = 128) -> torch.Tensor:
    """Exact block-scaled fp8 GEMM at prefill widths (r9k_fp8_prefill_block): a_q [M, K] e4m3, a_s [M, K/block]
    fp32 (quant_group128_fp8), wq in permute_fp8 order with bs [ceil(N/block), K/block] fp32 -- stock's
    compressed-tensors block-fp8 operands as they are. K % block == 0."""
    from . import moe as KM
    from ..ops import _identity_tables
    M = a_q.shape[0]
    cfg = tile_cfg(N, K, M, block=True) if cfg is None else cfg
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=a_q.device)
    if M == 0:
        return out
    assert a_s.shape == (M, K // block) and a_s.dtype == torch.float32 and a_s.is_contiguous(), a_s.shape
    assert bs.shape[1] == K // block and bs.dtype == torch.float32 and bs.is_contiguous(), bs.shape
    blk = KM.prefill_block(cfg)
    (sid, eid, ntpp), _ = _identity_tables(a_q, M, blk // KM.MOE_BLOCK)
    rc = _L().r9k_fp8_prefill_block(a_q.data_ptr(), a_s.data_ptr(), wq.data_ptr(), bs.data_ptr(), out.data_ptr(),
                                    sid.data_ptr(), eid.data_ptr(), ntpp.data_ptr(), eid.numel(), M, K, N,
                                    block, block, cfg, _stream())
    if rc:
        raise RuntimeError(f"r9k_fp8_prefill_block failed ({rc}) M={M} N={N} K={K} cfg={cfg}")
    return out


def hc4_interleave(w_up: torch.Tensor) -> torch.Tensor:
    """[4*HD, LR] -> rows reordered so each 64-row group holds 16 output columns x the 4 hyper-connection streams
    (r' = g*64 + s*16 + i <- r = s*HD + g*16 + i): the layout gemm_fp8_mix's epilogue expects."""
    DIM, LR = w_up.shape
    assert DIM % 64 == 0, DIM
    return w_up.reshape(4, DIM // 64, 16, LR).permute(1, 0, 2, 3).reshape(DIM, LR).contiguous()


def hc4_perm(DIM: int, device) -> torch.Tensor:
    """perm[r'] = r of hc4_interleave (w_up[perm] == hc4_interleave(w_up))."""
    return torch.arange(DIM, device=device).reshape(4, DIM // 64, 16).permute(1, 0, 2).reshape(-1)


def gemm_fp8_mix(a_q: torch.Tensor, a_s: torch.Tensor, w: Fp8Weight, xn: torch.Tensor,
                 out: torch.Tensor | None = None, cfg: int | None = None) -> torch.Tensor:
    """The hyper-connection up GEMM with hc_gate_mix fused (HC = 4): out[M, N/4] bf16 =
    (1/4) sum_s sigmoid(bf16(fp8 GEMM)[m, s*HD + c]) * xn[m, s*HD + c]; w = quantize_rows_fp8(hc4_interleave(w_up)),
    a_q / a_s the per-row fp8 lora [M, K]. K % 32 == 0, N % 64 == 0."""
    from . import moe as KM
    from ..ops import _identity_tables
    M = a_q.shape[0]
    HD = w.N // 4
    cfg = tile_cfg(w.N, w.K, M, mix=True) if cfg is None else cfg
    if out is None:
        out = torch.empty((M, HD), dtype=torch.bfloat16, device=a_q.device)
    if M == 0:
        return out
    assert xn.shape == (M, w.N) and xn.dtype == torch.bfloat16 and xn.stride(1) == 1 and xn.stride(0) % 8 == 0
    blk = KM.prefill_block(cfg)
    (sid, eid, ntpp), _ = _identity_tables(a_q, M, blk // KM.MOE_BLOCK)
    rc = _L().r9k_fp8_prefill_mix(a_q.data_ptr(), a_s.data_ptr(), w.wq.data_ptr(), w.ws.data_ptr(), out.data_ptr(),
                                  xn.data_ptr(), xn.stride(0), sid.data_ptr(), eid.data_ptr(), ntpp.data_ptr(),
                                  eid.numel(), M, w.K, w.N, cfg, _stream())
    if rc:
        raise RuntimeError(f"r9k_fp8_prefill_mix failed ({rc}) M={M} N={w.N} K={w.K} cfg={cfg}")
    return out


def gemm_fp8_tiled(a_q: torch.Tensor, a_s: torch.Tensor, w: Fp8Weight, out: torch.Tensor | None = None,
                   cfg: int | None = None) -> torch.Tensor:
    """Prefill-width fp8 x fp8 GEMM on the LDS-tiled WMMA kernel (kernels/r9k_moe_mxfp4a8.hip, F8 path):
    out[M, N] bf16 = (a_q[M, K] e4m3 * a_s[M]) . (w e4m3 * w.ws[N])^T. K % 32 == 0, N % 16 == 0."""
    from . import moe as KM
    from ..ops import _identity_tables
    M = a_q.shape[0]
    cfg = tile_cfg(w.N, w.K, M) if cfg is None else cfg
    if out is None:
        out = torch.empty((M, w.N), dtype=torch.bfloat16, device=a_q.device)
    if M == 0:
        return out
    t, _ = _identity_tables(a_q, M, KM.prefill_block(cfg) // KM.MOE_BLOCK)
    KM.moe_gemm(a_q, a_s, w, out, *t, M, 1, None, num_experts=1, prefill=cfg)
    return out
