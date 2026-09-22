"""Python bindings for libr9k's grouped MXFP4 x FP8 MoE GEMM (gfx1201).

Weights come from a compressed-tensors ``mxfp4-pack-quantized`` checkpoint: per expert
``weight_packed [N, K/2] uint8`` (element 2i in the low nibble, 2i+1 in the high nibble, e2m1 codes) and
``weight_scale [N, K/32] uint8`` (E8M0). ``prepare_mxfp4_weights`` converts them once at load to the layout
the kernel reads; see kernels/r9k_moe_mxfp4a8.hip for the format.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import torch

GROUP = 32
MOE_BLOCK = 16

_LIB = None


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        path = os.environ.get("R9K_LIB") or os.path.join(os.path.dirname(__file__), "libr9k.so")
        L = ctypes.CDLL(path)
        L.r9k_moe_mxfp4a8.restype = ctypes.c_int
        L.r9k_moe_mxfp4a8.argtypes = [ctypes.c_long] * 12 + [ctypes.c_int] * 9 + [ctypes.c_long]
        L.r9k_moe_nvfp4a8.restype = ctypes.c_int
        L.r9k_moe_nvfp4a8.argtypes = [ctypes.c_long] * 12 + [ctypes.c_int] * 9 + [ctypes.c_long]
        L.r9k_moe_4bit_prefill.restype = ctypes.c_int
        L.r9k_moe_4bit_prefill.argtypes = [ctypes.c_int] + [ctypes.c_long] * 12 + [ctypes.c_int] * 6 + [ctypes.c_long]
        L.r9k_moe_prefill_bm.restype = ctypes.c_int
        L.r9k_moe_prefill_bm.argtypes = [ctypes.c_int]
        L.r9k_quant_rows_fp8.restype = ctypes.c_int
        L.r9k_quant_rows_fp8.argtypes = [ctypes.c_long] * 3 + [ctypes.c_int] * 3 + [ctypes.c_long]
        L.r9k_silu_mul_quant_fp8.restype = ctypes.c_int
        L.r9k_silu_mul_quant_fp8.argtypes = [ctypes.c_long] * 3 + [ctypes.c_int] * 2 + [ctypes.c_long]
        assert L.r9k_moe_block() == MOE_BLOCK
        if hasattr(L, "r9k_fold_supported"):          # older libr9k.so builds have no folded kernels
            L.r9k_fold_supported.restype = ctypes.c_int
        if hasattr(L, "r9k_moe_4bit_prefill_at"):     # ... or no A-tiled kernels
            L.r9k_moe_4bit_prefill_at.restype = ctypes.c_int
            L.r9k_moe_4bit_prefill_at.argtypes = [ctypes.c_int] + [ctypes.c_long] * 12 + [ctypes.c_int] * 7 + [ctypes.c_long]
            L.r9k_moe_atiled_bm.restype = ctypes.c_int
            L.r9k_moe_atiled_bm.argtypes = [ctypes.c_int]
            L.r9k_quant_rows_fp8_tiled.restype = ctypes.c_int
            L.r9k_quant_rows_fp8_tiled.argtypes = [ctypes.c_long] * 3 + [ctypes.c_int] * 3 + [ctypes.c_long]
        _LIB = L
    return _LIB


def fold_supported() -> bool:
    L = lib()
    return hasattr(L, "r9k_fold_supported") and bool(L.r9k_fold_supported())


def atiled_supported() -> bool:
    return hasattr(lib(), "r9k_moe_4bit_prefill_at")


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


@dataclass
class Mxfp4Experts:
    wq: torch.Tensor    # [E, N/16 * K/16 * 32] int32, fragment order
    wsr: torch.Tensor   # [E, K/32 + 1, N] uint8: E8M0 exponents K-block-major, last row = per-row reference
    N: int
    K: int
    fold: bool = False  # folded-exponent kernels (fold_decide() per tensor at weight prep; layout is unchanged)

    @property
    def estride(self) -> int:
        return (self.K // GROUP + 1) * self.N


@dataclass
class Nvfp4Experts:
    """Native NVFP4 (e2m1 + e4m3 scale per 16 + fp32 global per row), same fragment-order codes as MXFP4."""
    wq: torch.Tensor    # [E, N/16 * K/16 * 32] int32, fragment order
    ws: torch.Tensor    # [E, K/16, N] uint8 (e4m3 bits), K-step-major
    wg: torch.Tensor    # [E, N] fp32 per-row global multiplier (1 / CT's stored divisor)
    N: int
    K: int


def prepare_nvfp4_weights(packed: torch.Tensor, scale16: torch.Tensor, row_mult: torch.Tensor) -> Nvfp4Experts:
    """packed [E, N, K/2] u8, scale16 [E, N, K/16] e4m3, row_mult [E, N] fp32 -> Nvfp4Experts."""
    E, N, Kh = packed.shape
    K = Kh * 2
    assert N % 16 == 0 and K % 16 == 0, (N, K)
    assert scale16.shape == (E, N, K // 16), (scale16.shape, (E, N, K // 16))
    ws = scale16.view(torch.uint8).transpose(1, 2).contiguous()
    return Nvfp4Experts(permute_fragments(packed.view(torch.uint8)), ws,
                        row_mult.float().reshape(E, N).contiguous(), N, K)


def permute_fragments(packed: torch.Tensor) -> torch.Tensor:
    """[E, N, K/2] uint8 checkpoint order -> [E, N/16*K/16*32] int32 fragment order.

    Slot l of tile (nt, ks) holds W[nt*16 + (l & 15)][ks*16 + 8*(l >> 4) .. +8] (4 packed bytes).

    That destination is not a choice: `v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` requires lane l of a wave to hold
    row (l & 15) and k-bytes 8*(l >> 4)..+8 of the B fragment. Permuting the weight into exactly that order on
    the host is the only way a wave can then read its fragment as 32 contiguous dwords, so any correct
    implementation for this builtin produces the same bytes. (An earlier docstring described this as "identical
    to libr4d mxfp4_layout.permute_w" -- true, and true of anyone else's too, because the hardware fixes it.
    See notes/independence.md.)
    """
    E, N, Kh = packed.shape
    K = Kh * 2
    nt, ks = N // 16, K // 16
    w = packed.reshape(E, nt, 16, ks, 2, 4)            # [E, nt, r, ks, h, 4 bytes]
    w = w.permute(0, 1, 3, 4, 2, 5).contiguous()       # [E, nt, ks, h, r, 4] -> lane = h*16 + r
    return w.reshape(E, -1).view(torch.int32)


def prepare_mxfp4_weights(packed: torch.Tensor, scale: torch.Tensor) -> Mxfp4Experts:
    E, N, Kh = packed.shape
    K = Kh * 2
    assert N % 16 == 0 and K % GROUP == 0, (N, K)
    assert scale.shape == (E, N, K // GROUP), (scale.shape, (E, N, K // GROUP))
    return Mxfp4Experts(permute_fragments(packed.view(torch.uint8)), pack_scales(scale), N, K)


def pack_scales(scale: torch.Tensor) -> torch.Tensor:
    """[E, N, K/32] E8M0 -> [E, K/32 + 1, N]: transposed block exponents plus the per-row max as the last row."""
    scale = scale.view(torch.uint8)
    E, N, nb = scale.shape
    out = torch.empty((E, nb + 1, N), dtype=torch.uint8, device=scale.device)
    out[:, :nb].copy_(scale.transpose(1, 2))
    out[:, nb] = scale.max(dim=2).values
    return out


# ------------------------------------------------------------------------------------ folded-exponent MXFP4
# The kernels can fold each block's E8M0 into the e2m1 -> e4m3 unpack (kMag table: value * 2^-d, d = row reference
# exponent - block exponent) and apply only 2^(ref - 127) per row in the epilogue: no per-group fp32 scaling in the
# inner loop. e4m3 reaches 2^-9 below the row's largest block, so the fold is
#   exact     for d <= FOLD_EXACT (8): every e2m1 value (0.5 .. 6) is representable, subnormals included;
#   rounded   for 9 <= d <= FOLD_REACH (12): mantissa bits fall off the bottom of the subnormal range (kMag rounds);
#   flushed   for d >= 13: the block reads as zero (its weights are < 2^-12 of the row's largest block).
# A flushed weight errs by at most 2^-10 * 2^ref, i.e. < 1/6000 of the row's largest weight; the damage is bounded
# per weight but a tensor whose rows are dominated by such blocks would silently lose them, so folding is decided
# per weight tensor from the d distribution (fold_stats) against explicit thresholds (fold_ok), logged at load.
# R9K_FOLD: 0 (default) never fold; 1 fold when the tensor passes the guard; force fold regardless (measurements).
FOLD_EXACT, FOLD_REACH = 8, 12
FOLD_MODE = os.environ.get("R9K_FOLD", "0").lower()
FOLD_MAX_INEXACT = float(os.environ.get("R9K_FOLD_MAX_INEXACT", "0.05"))   # max fraction of blocks with d > 8
FOLD_MAX_FLUSH = float(os.environ.get("R9K_FOLD_MAX_FLUSH", "0.001"))      # max fraction of blocks with d > 12
# Short-K tensors are not FMA-bound (few slabs, weight-stream / launch bound) and the fold's per-slab scale load +
# lane permutes cost more than the FMAs they remove: measured on the Flash-Next routed down GEMM (K=320, cfg 11:
# 1010 -> 1048 us @2048 tokens, 1659 -> 1696 @4096), while every K >= 2560 shape gained 6-16%.
FOLD_MIN_K = int(os.environ.get("R9K_FOLD_MIN_K", "1024"))


@dataclass
class FoldStats:
    blocks: int
    hist: list            # hist[d] = blocks with d = ref - block exponent (d clamped to 0..16; 16 = ">15")
    max_d: int
    K: int = 0            # the GEMM's K (folding is not worth it below FOLD_MIN_K)

    @property
    def p_inexact(self) -> float:
        return sum(self.hist[FOLD_EXACT + 1:]) / max(1, self.blocks)

    @property
    def p_flush(self) -> float:
        return sum(self.hist[FOLD_REACH + 1:]) / max(1, self.blocks)

    @property
    def mean_d(self) -> float:
        return sum(d * n for d, n in enumerate(self.hist)) / max(1, self.blocks)

    def __str__(self):
        return (f"K={self.K} blocks={self.blocks} mean_d={self.mean_d:.2f} max_d={self.max_d} "
                f"inexact(d>{FOLD_EXACT})={100 * self.p_inexact:.3f}% flushed(d>{FOLD_REACH})={100 * self.p_flush:.4f}%")


def fold_stats(wsr: torch.Tensor, chunk: int = 64 << 20) -> FoldStats:
    """Distribution of d = row reference exponent - block exponent over a packed [E, K/32 + 1, N] scale tensor
    (pack_scales layout; the last row is the reference). Chunked over experts, any device."""
    E, nb1, N = wsr.shape
    hist = torch.zeros(17, dtype=torch.int64)
    step = max(1, chunk // (nb1 * N))
    for e0 in range(0, E, step):
        w = wsr[e0:e0 + step].to(torch.int16)
        d = (w[:, -1:, :] - w[:, :-1, :]).clamp_(0, 16)
        hist += torch.bincount(d.reshape(-1), minlength=17).cpu()
    h = hist.tolist()
    return FoldStats(E * (nb1 - 1) * N, h, max(i for i, n in enumerate(h) if n) if any(h) else 0, (nb1 - 1) * GROUP)


def fold_ok(st: FoldStats) -> bool:
    return fold_why(st) == ""


def fold_why(st: FoldStats) -> str:
    """'' when the tensor may fold, else the reason it stays on the exact kernels."""
    if st.p_inexact > FOLD_MAX_INEXACT or st.p_flush > FOLD_MAX_FLUSH:
        return "exceeds R9K_FOLD_MAX_INEXACT/FLUSH"
    if st.K < FOLD_MIN_K:
        return f"K={st.K} < R9K_FOLD_MIN_K={FOLD_MIN_K}: no gain on short K"
    return ""


_FOLD_LOG: list[tuple[str, bool, FoldStats]] = []


def fold_decide(wsr: torch.Tensor, name: str = "", log=None) -> bool:
    """Whether the MXFP4 tensor `wsr` (pack_scales layout) is served with the folded-exponent kernels: R9K_FOLD off
    -> False without looking; on -> fold_ok(fold_stats); force -> True. Every decision is logged (with the stats and
    running totals) so a checkpoint that folds badly is visible at load rather than silently degraded."""
    if FOLD_MODE in ("", "0", "off", "no"):
        return False
    if not fold_supported():
        return False
    st = fold_stats(wsr)
    why = fold_why(st)
    fold = not why or FOLD_MODE == "force"
    _FOLD_LOG.append((name, fold, st))
    n_on = sum(1 for _, f, _ in _FOLD_LOG if f)
    msg = (f"r9700: fold {'ON ' if fold else 'OFF'} {name or 'mxfp4'}: {st}"
           f"{'' if not why else ' (' + why + (', forced)' if fold else ')')}"
           f" [tensors so far: {n_on} folded / {len(_FOLD_LOG) - n_on} exact]")
    (log or _default_log)(msg)
    return fold


def fold_summary() -> str:
    on = [x for x in _FOLD_LOG if x[1]]
    off = [x for x in _FOLD_LOG if not x[1]]
    worst = max(_FOLD_LOG, key=lambda x: x[2].p_flush, default=None)
    return (f"fold: {len(on)} tensors folded, {len(off)} kept exact"
            + (f"; worst flush {100 * worst[2].p_flush:.4f}% ({worst[0]})" if worst else ""))


def _default_log(msg: str) -> None:
    try:
        from vllm.logger import init_logger
        init_logger("vllm." + __name__).info(msg)
    except Exception:
        print(msg)


def quant_rows_fp8(x: torch.Tensor, tiled: bool = False):
    """bf16 [M, K] -> (e4m3fn [M, K], fp32 [M]) with a per-row dynamic scale.

    tiled: the fp8 bytes in the WMMA-fragment-tiled layout the A-tiled prefill kernel reads (16*ceil(M/16) rows;
    fragment (row//16, k//16) = 256 contiguous bytes, see r9k_quant_rows_fp8_tiled); returned as a [16*Mt, K]
    e4m3fn tensor whose bytes are NOT row-major (only moe_gemm(..., a_tiled=True) may consume it). Same values and
    scales as the row-major quant, only the store addresses differ."""
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.stride(1) == 1
    M, K = x.shape
    s = torch.empty((M,), dtype=torch.float32, device=x.device)
    if tiled:
        mt = (M + 15) // 16
        q = torch.empty((mt * 16, K), dtype=torch.float8_e4m3fn, device=x.device)
        rc = lib().r9k_quant_rows_fp8_tiled(x.data_ptr(), q.data_ptr(), s.data_ptr(), M, K, x.stride(0), _stream())
        if rc:
            raise RuntimeError(f"r9k_quant_rows_fp8_tiled failed ({rc}) M={M} K={K}")
        return q, s
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    rc = lib().r9k_quant_rows_fp8(x.data_ptr(), q.data_ptr(), s.data_ptr(), M, K, x.stride(0), _stream())
    if rc:
        raise RuntimeError(f"r9k_quant_rows_fp8 failed ({rc})")
    return q, s


def tile_fp8_ref(q: torch.Tensor) -> torch.Tensor:
    """Pure-torch re-layout of a row-major e4m3 [M, K] into the fragment-tiled [16*ceil(M/16), K] buffer (tests):
    fragment (mt, ks) -> 32 lanes x 8 bytes, lane l = row mt*16 + l%16, bytes ks*16 + 8*(l//16) .. +8."""
    M, K = q.shape
    mt = (M + 15) // 16
    b = torch.zeros((mt * 16, K), dtype=torch.uint8, device=q.device)
    b[:M] = q.view(torch.uint8)
    b = b.reshape(mt, 16, K // 16, 2, 8).permute(0, 2, 3, 1, 4)     # [mt, ks, half, row, 8] -> lane = half*16 + row
    return b.contiguous().reshape(mt * 16, K).view(torch.float8_e4m3fn)


def silu_mul_quant_fp8(gu: torch.Tensor):
    """bf16 [M, 2I] (gate | up) -> (e4m3fn [M, I], fp32 [M]): silu(gate) * up with a per-row dynamic scale."""
    assert gu.dtype == torch.bfloat16 and gu.is_contiguous() and gu.shape[1] % 2 == 0
    M, I2 = gu.shape
    q = torch.empty((M, I2 // 2), dtype=torch.float8_e4m3fn, device=gu.device)
    s = torch.empty((M,), dtype=torch.float32, device=gu.device)
    rc = lib().r9k_silu_mul_quant_fp8(gu.data_ptr(), q.data_ptr(), s.data_ptr(), M, I2 // 2, _stream())
    if rc:
        raise RuntimeError(f"r9k_silu_mul_quant_fp8 failed ({rc})")
    return q, s


# Prefill (LDS-tiled, large-M) kernel tile table: cfg -> (rows per routing block BM, columns BN); mirrors
# kPfCfgs in r9k_moe_mxfp4a8.hip (the kernel is the source of truth: prefill_block() asks it).
PREFILL_TILES = {0: (256, 64), 1: (256, 128), 2: (128, 128), 3: (128, 64), 4: (128, 64), 5: (256, 32),
                 6: (64, 64), 7: (256, 128), 8: (256, 128), 9: (256, 128), 10: (128, 128), 11: (64, 64),
                 12: (256, 128), 13: (128, 128), 14: (64, 64), 15: (64, 128), 16: (256, 128)}
PREFILL_MIN_M = int(os.environ.get("R9K_PREFILL_MIN_M", "128"))   # dense calls at M >= this use the prefill path
# Untuned shapes: 256x128 BK=64 double-buffered (cfg 12) from M >= 512, 128x128 double-buffered (cfg 10) below (grid fill).
PREFILL_DEFAULT = int(os.environ.get("R9K_PREFILL_CFG", "12"))
PREFILL_DEFAULT_SMALL = int(os.environ.get("R9K_PREFILL_CFG_SMALL", "10"))


def prefill_block(cfg: int) -> int:
    """Rows per routing block (moe_align_block_size block) of prefill tile `cfg`."""
    bm = lib().r9k_moe_prefill_bm(cfg)
    if bm <= 0:
        raise ValueError(f"unknown prefill cfg {cfg}")
    return bm


def is_prefill_cfg(cfg) -> bool:
    return bool(cfg) and cfg[0] == "P"


# A-tiled prefill kernel (r9k_moe_4bit_prefill_at: fragment-tiled activations straight from global, weights through
# LDS, folded or with fp32-per-group scaling like the LDS-A kernel). cfg -> (BM, BN, BK); mirrors kAtCfgs in the
# kernel (atiled_block() asks it). Selected by pick_cfg as ("A", cfg) for dense MXFP4 GEMMs at prefill M, folded or
# not, when R9K_ATILED=1 and K % BK == 0; the caller then quantizes with quant_rows_fp8(x, tiled=True) and runs
# moe_gemm(..., a_tiled=True). Everything else keeps the LDS-A tiles.
ATILED_TILES = {0: (256, 64, 64), 1: (256, 128, 64), 2: (256, 64, 128), 3: (256, 128, 64), 4: (128, 128, 64),
                5: (128, 64, 64), 6: (256, 128, 64), 7: (256, 64, 64), 8: (128, 128, 64)}
ATILED = os.environ.get("R9K_ATILED", "1") == "1"
# Untuned shapes, by M band (measured 2026-09-20 on the five served 27B shapes, steady state, vs the folded LDS-A
# tiles: 0.83-0.92x at M >= 2048 on the 256x128 8-wave tile, 0.74-0.83x at 512 on 256x64, 0.74-0.86x at 128-256
# on 128x64; the 256-row tiles lose at M=128 where half the tile is padding).
ATILED_DEFAULT = int(os.environ.get("R9K_ATILED_CFG", "1"))              # M >= 2048: 256 x 128, 8 waves
ATILED_DEFAULT_MID = int(os.environ.get("R9K_ATILED_CFG_MID", "0"))      # 512 <= M < 2048: 256 x 64
ATILED_DEFAULT_SMALL = int(os.environ.get("R9K_ATILED_CFG_SMALL", "5"))  # M < 512: 128 x 64
# The tiled quantizer has no scalar fallback (r9k_quant_rows_fp8_tiled returns -3 past the vectorized kernel's
# reach), so a K it cannot serve must not pick this path at all: RQ_THREADS * RQ_VEC * 8 in the kernel.
ATILED_MAX_K = 20480


def atiled_block(cfg: int) -> int:
    bm = lib().r9k_moe_atiled_bm(cfg)
    if bm <= 0:
        raise ValueError(f"unknown A-tiled cfg {cfg}")
    return bm


def is_atiled_cfg(cfg) -> bool:
    return bool(cfg) and cfg[0] == "A"


def pick_atiled(N: int, K: int, M: int, fold: bool = True) -> tuple[str, int] | None:
    """("A", cfg) for a dense MXFP4 GEMM at M >= PREFILL_MIN_M when the tiled path is on and legal. fold only says
    which kernel variant the caller will run: the tile comes from the tuned "mxfp4_at" rows either way (measured
    2026-09-22: the fold-tuned tile is also the best non-folded tile on every served shape and band, within 3%),
    else the M-band defaults."""
    if not ATILED or M < PREFILL_MIN_M or K % 64 or K > ATILED_MAX_K or not atiled_supported():
        return None
    from .tuned import lookup
    t = lookup("mxfp4_at", N, K, M)
    if t and is_atiled_cfg(t) and K % ATILED_TILES[t[1]][2] == 0:
        return ("A", t[1])
    cfg = ATILED_DEFAULT if M >= 2048 else (ATILED_DEFAULT_MID if M >= 512 else ATILED_DEFAULT_SMALL)
    return ("A", cfg) if K % ATILED_TILES[cfg][2] == 0 else None


def moe_gemm(a_q: torch.Tensor, a_s: torch.Tensor, w: Mxfp4Experts, out: torch.Tensor,
             sorted_ids: torch.Tensor, expert_ids: torch.Tensor, ntpp: torch.Tensor, numel: int,
             a_row_div: int, topk_w: torch.Tensor | None = None, WV: int = 2, SK: int = 4, NPW: int = 2,
             num_experts: int | None = None, MT: int = 1, ldsa: bool = False, prefill: int | None = None,
             a_tiled: bool = False):
    """out[r, :] = dequant(a_q[r // a_row_div]) @ W[expert(r)]^T (* topk_w[r]) for every routed flat row r.

    sorted_ids/expert_ids must come from moe_align_block_size(block_size=16*MT). MT M-tiles per workgroup share
    each weight fragment (use MT>1 when experts see many rows: prefill / wide batches). ldsa: stage the A tile
    through LDS (MT > 1 only; a tuned per-config choice, passed to the kernel as MT | 8).
    prefill: tile cfg of the LDS-tiled large-M kernel (r9k_moe_4bit_prefill) instead; the tables must then come
    from moe_align_block_size(block_size=prefill_block(cfg)) and WV/SK/NPW/MT/ldsa are ignored.
    The grid covers at most ceil(numel/BLK) + min(numel, E) blocks -- the most moe_align_block_size can fill
    with numel routed rows over E experts -- instead of the buffer's worst-case capacity; host-known, so the
    launch stays cudagraph-safe.
    a_tiled: a_q is the fragment-tiled buffer of quant_rows_fp8(tiled=True) in sorted-position order (dense:
    identity tables) and `prefill` is an A-tiled cfg (atiled_block(cfg) rows per block); MXFP4 only, w.fold picks
    the folded or the fp32-per-group variant."""
    assert out.dtype == torch.bfloat16 and out.shape[1] == w.N and a_q.shape[1] == w.K
    E = num_experts if num_experts is not None else w.wq.shape[0]
    if a_tiled:
        assert prefill is not None and isinstance(w, Mxfp4Experts), "a_tiled: MXFP4 prefill only"
        blk = atiled_block(prefill)
        max_blocks = min(expert_ids.numel(), (numel + blk - 1) // blk + min(numel, E))
        rc = lib().r9k_moe_4bit_prefill_at(
            1 if w.fold else 0, a_q.data_ptr(), a_s.data_ptr(), w.wq.data_ptr(), w.wsr.data_ptr(),
            w.wsr.data_ptr() + (w.K // GROUP) * w.N, out.data_ptr(),
            sorted_ids.data_ptr(), expert_ids.data_ptr(), ntpp.data_ptr(),
            topk_w.data_ptr() if topk_w is not None else 0,
            w.estride, w.estride, max_blocks, numel, a_row_div, a_q.shape[0] // 16, w.K, w.N, prefill, _stream())
        if rc:
            raise RuntimeError(f"r9k_moe_4bit_prefill_at failed ({rc}) fold={w.fold} N={w.N} K={w.K} cfg={prefill}")
        return out
    if prefill is not None:
        blk = prefill_block(prefill)
        max_blocks = min(expert_ids.numel(), (numel + blk - 1) // blk + min(numel, E))
        nv = isinstance(w, Nvfp4Experts)
        rc = lib().r9k_moe_4bit_prefill(
            1 if nv else (2 if w.fold else 0), a_q.data_ptr(), a_s.data_ptr(), w.wq.data_ptr(),
            w.ws.data_ptr() if nv else w.wsr.data_ptr(),
            w.wg.data_ptr() if nv else w.wsr.data_ptr() + (w.K // GROUP) * w.N, out.data_ptr(),
            sorted_ids.data_ptr(), expert_ids.data_ptr(), ntpp.data_ptr(),
            topk_w.data_ptr() if topk_w is not None else 0,
            (w.K // 16) * w.N if nv else w.estride, w.N if nv else w.estride,
            max_blocks, numel, a_row_div, w.K, w.N, prefill, _stream())
        if rc:
            raise RuntimeError(f"r9k_moe_4bit_prefill failed ({rc}) nv={nv} N={w.N} K={w.K} cfg={prefill}")
        return out
    blk = MOE_BLOCK * MT
    max_blocks = min(expert_ids.numel(), (numel + blk - 1) // blk + min(numel, E))
    MT = MT | (8 if ldsa and MT > 1 else 0)
    if isinstance(w, Nvfp4Experts):
        rc = lib().r9k_moe_nvfp4a8(
            a_q.data_ptr(), a_s.data_ptr(), w.wq.data_ptr(), w.ws.data_ptr(), w.wg.data_ptr(), out.data_ptr(),
            sorted_ids.data_ptr(), expert_ids.data_ptr(), ntpp.data_ptr(),
            topk_w.data_ptr() if topk_w is not None else 0,
            (w.K // 16) * w.N, w.N, max_blocks, numel, a_row_div, w.K, w.N, WV, SK, NPW, MT, _stream())
        if rc:
            raise RuntimeError(f"r9k_moe_nvfp4a8 failed ({rc}) N={w.N} K={w.K} WV={WV} SK={SK} NPW={NPW} MT={MT}")
        return out
    rc = lib().r9k_moe_mxfp4a8(
        a_q.data_ptr(), a_s.data_ptr(), w.wq.data_ptr(), w.wsr.data_ptr(),
        w.wsr.data_ptr() + (w.K // GROUP) * w.N, out.data_ptr(),
        sorted_ids.data_ptr(), expert_ids.data_ptr(), ntpp.data_ptr(),
        topk_w.data_ptr() if topk_w is not None else 0,
        w.estride, w.estride, max_blocks, numel, a_row_div, w.K, w.N, WV, SK, NPW, MT | (16 if w.fold else 0),
        _stream())
    if rc:
        raise RuntimeError(f"r9k_moe_mxfp4a8 failed ({rc}) N={w.N} K={w.K} WV={WV} SK={SK} NPW={NPW} MT={MT}")
    return out


def pick_cfg(N: int, K: int, group: int = GROUP, M: int | None = None, kind: str | None = None,
             fold: bool = False) -> tuple[int, ...]:
    """A legal (WV, SK, NPW[, MT[, LDSA]]) for any N % 16 == 0, K % group == 0 (32 MXFP4, 16 NVFP4): deep split-K for long K (weight streaming with few
    N tiles per wave), shallow for short K. Mirrors the tuned Flash-Next defaults (2,4,2) @K=2560, (4,2,1) @K=320.
    Large M (dense prefill) returns ("P", cfg): the LDS-tiled prefill kernel (tuned.json rows ["P", cfg], else
    PREFILL_DEFAULT from M >= PREFILL_MIN_M when K % 64 == 0); or ("A", cfg), the A-tiled kernel, for MXFP4
    (folded or not: fold only selects the kernel variant) when R9K_ATILED is on (pick_atiled)."""
    if kind == "mxfp4" and M:
        t = pick_atiled(N, K, M, fold)
        if t:
            return t
    if kind and M:
        from .tuned import lookup
        t = lookup(kind, N, K, M)
        if t:
            return t
        if M >= PREFILL_MIN_M and K % 64 == 0:
            return ("P", PREFILL_DEFAULT if M >= 512 else PREFILL_DEFAULT_SMALL)
    for WV, SK, NPW in ((2, 4, 2), (4, 2, 1), (2, 5, 2), (4, 1, 1)):
        if K % (SK * group) == 0 and (K >= 1024 or SK <= 2):
            return WV, SK, NPW
    return 4, 1, 1


def pick_mt(numel: int, num_experts: int) -> int:
    """M tiles per routing block from the host-known row count: rows per touched expert >= 32 -> 4, >= 16 -> 2."""
    per = numel / max(1, min(num_experts, numel))
    return 4 if per >= 32 else (2 if per >= 16 else 1)


MOE_PREFILL_CFG = int(os.environ.get("R9K_MOE_PREFILL_CFG", "11"))     # -1: routed MoE never uses the prefill tile
MOE_PREFILL_CFG_GATE_UP = int(os.environ.get("R9K_MOE_PREFILL_CFG1", "15"))   # -1: gate_up stays on the MT kernel


def pick_moe_prefill(MT: int, K_down: int, gate_up: bool = False) -> int | None:
    """Prefill tile for the routed-MoE GEMMs of a step whose routing block is 16*MT rows (None: keep the MT kernel).
    Both tiles share block 64 with MT=4, so one moe_align_block_size table serves both GEMMs:
      * down (K=320): the 64x64 double-buffered tile (cfg 11), 1.6-1.7x faster than the split-K decode kernel
        (Flash-Next 2048/4096-token chunks: 1631 -> 1023 / 2802 -> 1669 us steady-state);
      * gate_up (N=640, K=2560): the 64x128 BK=64 double-buffered tile (cfg 15: five column tiles instead of ten,
        half the barriers per K), 1347 -> 1230 / 2282 -> 1907 us at 2048 / 4096 tokens (the 64x64 tiles were a
        wash to -10% against the MT kernel, which is why the first pass left gate_up on it)."""
    cfg = MOE_PREFILL_CFG_GATE_UP if gate_up else MOE_PREFILL_CFG
    if cfg < 0 or K_down % 64 or 16 * MT != prefill_block(cfg):
        return None
    return cfg


def align_block_size_ref(topk_ids: torch.Tensor, num_experts: int, block: int = MOE_BLOCK):
    """Pure-torch equivalent of vLLM's moe_align_block_size (for tests): pads each expert's rows to a
    multiple of `block`, pad value = numel. Returns (sorted_ids, expert_ids, ntpp) sized to capacity."""
    flat = topk_ids.flatten().to(torch.int64)
    numel = flat.numel()
    cap = numel + num_experts * (block - 1)
    cap = (cap + block - 1) // block * block
    sorted_ids = torch.full((cap,), numel, dtype=torch.int32)
    expert_ids = torch.full((cap // block,), -1, dtype=torch.int32)
    pos = 0
    for e in range(num_experts):
        rows = (flat == e).nonzero().flatten()
        if rows.numel() == 0:
            continue
        n = (rows.numel() + block - 1) // block * block
        sorted_ids[pos:pos + rows.numel()] = rows.to(torch.int32)
        expert_ids[pos // block:(pos + n) // block] = e
        pos += n
    return sorted_ids, expert_ids, torch.tensor([pos], dtype=torch.int32)
