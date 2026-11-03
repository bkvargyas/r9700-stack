"""Decode-width EXL3 GEMV: kernels/r9k_exl3_gemv.hip through ctypes (libr9k.so). Package D, notes/exl3-plan.md Phase 4.

``r9k_exl3_gemv(groups, x, out, rotated=False)`` runs y = x @ W for one or more EXL3 matrices of one layer in one launch
per (K, codebook): the input rotation (x * suh, 128-point Hadamard), the trellis decode, the f32 dot, the output rotation
(Hadamard, * svh) and the store in the output dtype all happen in the kernel -- a bf16 [m, k] activation goes in, a bf16
(f16, f32) [m, n] result comes out, nothing around the launch. Rows 1..8 run the fdot row tile (one decode feeds every
row, bandwidth-bound); from R9K_EXL3_GEMV_MMA_FROM rows the dot runs on WMMA, up to 48 rows a group (one decode of a 16x16
tile feeds three 16-row tiles), two 128-column tiles a block sharing the input rotation.

Interface (what Exl3LinearMethod.apply and package B's pre-pass call):
  plan = Exl3GemvPlan(parts, k)          parts: objects with trellis / suh / svh / K / mcg / mul1 / dst / n (quant/exl3.py
                                          Exl3Part): output columns [dst, dst + n) of the layer's output, all reading the
                                          same k inputs. Built once at load (Exl3LinearMethod._warm).
  r9k_exl3_gemv(plan, x, out)            x [m, k] bf16 or f16 (row stride a multiple of 4, unit column stride), out
                                          [m, n_total] bf16 / f16 / f32: part j writes out[:, dst_j:dst_j + n_j].
  r9k_exl3_gemv(plan, xs, outs)          per-part lists: xs[j] [m, k], outs[j] [m, n_j] (separate buffers or views).
  r9k_exl3_gemv(plan, xr, out, rotated=True)
                                          xr (a list, one per part; one tensor for a one-part plan): f16 inputs ALREADY
                                          rotated for that part = had_r_128(x * suh_j) / sqrt(128) per 128-block (what
                                          ``ext.had_r_128(x, xr, suh_j, None, 1.0)`` writes); the kernel only copies them.
                                          The hook for a fused RMSNorm + rotation pre-pass.
  plan.takes(m)                          whether an m-row call should run here (else Exl3LinearMethod's old paths).
No host sync, no per-call allocation (outputs are the caller's), deterministic, HIP-graph safe once the scratch is
reserved (Exl3GemvPlan.reserve / reserve() before capture). Not stream-safe across streams of one device: the f32
partials and arrival counters are one scratch set per device (the EXL3 layers already run on one stream).

Knobs: R9K_EXL3_GEMV_BLOCKS (workgroups a matrix aims for, default 192), R9K_EXL3_GEMV_KS (force the input-chunk count;
diagnostic), R9K_EXL3_GEMV_MMA_FROM (rows from which WMMA runs), R9K_EXL3_GEMV_MMA_ROWS (rows a WMMA group, <= 48),
R9K_EXL3_GEMV_NB2 (128-column tiles a WMMA block, 1 / 2 / 4; default 2), R9K_EXL3_GEMV_ROWS (rows an fdot group, <= 16),
R9K_EXL3_GEMV_ODD_M (row ceiling of plans whose tiles do not pair up). Library: R9K_GEMV_LIB, else R9K_LIB, else
kernels/libr9k.so.
"""
from __future__ import annotations

import ctypes
import os

import torch

_L = None
_scr: dict = {}
_NDESC = 13
BLOCKS = int(os.environ.get("R9K_EXL3_GEMV_BLOCKS", 192))
FORCE_KS = int(os.environ.get("R9K_EXL3_GEMV_KS", 0))
ROWS = int(os.environ.get("R9K_EXL3_GEMV_ROWS", 16))     # max rows a group (row tile; the kernel builds 1..16)
MMA_FROM = int(os.environ.get("R9K_EXL3_GEMV_MMA_FROM", 9))   # rows from which the WMMA form runs (up to 48 a group)
MMA_ROWS = int(os.environ.get("R9K_EXL3_GEMV_MMA_ROWS", 48))   # max rows a WMMA group (16 / 32 / 48)
ODD_M = int(os.environ.get("R9K_EXL3_GEMV_ODD_M", 16))  # max rows for plans with an odd 128-tile count (see max_m)
NB2 = int(os.environ.get("R9K_EXL3_GEMV_NB2", 0))   # WMMA blocks: 128-column tiles sharing one input rotation (0 = auto: 2)
_LDS = 48 * 1024        # rotated-input chunk a block: M rows x kc halves
_LDS_MMA = 50 * 1024    # WMMA form: 16-row tiles x (kc + 8) halves (+ 6 KiB of static transpose buffers)
_XM = {torch.bfloat16: 0, torch.float16: 1}
_YM = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}


def lib() -> ctypes.CDLL:
    global _L
    if _L is None:
        path = (os.environ.get("R9K_GEMV_LIB") or os.environ.get("R9K_LIB")
                or os.path.join(os.path.dirname(__file__), "..", "..", "kernels", "libr9k.so"))
        L = ctypes.CDLL(path)
        if not hasattr(L, "r9k_exl3_gemv"):
            raise RuntimeError(f"{path} has no r9k_exl3_gemv (rebuild kernels/)")
        L.r9k_exl3_gemv.restype = ctypes.c_int
        L.r9k_exl3_gemv.argtypes = [ctypes.c_long] + [ctypes.c_int] * 5 + [ctypes.c_long] * 3 + [ctypes.c_int,
                                                                                                  ctypes.c_long]
        L.r9k_exl3_gemv_dq.restype = ctypes.c_int
        L.r9k_exl3_gemv_dq.argtypes = [ctypes.c_long] * 2 + [ctypes.c_int] * 4 + [ctypes.c_long]
        for f in ("r9k_exl3_gemv_max_groups", "r9k_exl3_gemv_max_rows", "r9k_exl3_gemv_max_rows_mma",
                  "r9k_exl3_gemv_waves"):
            getattr(L, f).restype = ctypes.c_int
        _L = L
    return _L


def available() -> bool:
    try:
        lib()
        return True
    except Exception:
        return False


def ks_for(k: int, n: int, m: int, row_groups: int = 1, mma: bool = False, nb2: int = 1) -> int:
    """Input chunks a matrix is split into: enough workgroups to stream it ((n/128) * ks * row_groups >= BLOCKS), chunk a
    128-multiple whose m x kc f16 rows (WMMA: 16-row tiles, padded rows) fit the LDS budget."""
    segs, tiles = k // 128, n // 128 // nb2 * row_groups

    def lds(kc):
        return (-(-m // 16) * 16 * (kc + 8) * 2 <= _LDS_MMA) if mma else m * kc * 2 <= _LDS
    good = [d for d in range(1, segs + 1) if segs % d == 0 and lds(k // d)]
    if FORCE_KS and FORCE_KS in good:
        return FORCE_KS
    return next((d for d in good if tiles * d >= BLOCKS), good[-1])


def _scratch(dev: torch.device, floats: int, ints: int) -> dict:
    """One f32 partial buffer + one zeroed int32 counter buffer per device, grown only outside graph capture (the
    counters are left zero by every launch)."""
    s = _scr.get(dev.index)
    if s is not None and s["part"].numel() >= floats and s["ctr"].numel() >= ints:
        return s
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("r9k exl3 gemv: scratch must be reserved before graph capture "
                           f"(need {floats} floats / {ints} counters)")
    if s is not None:
        floats, ints = max(floats, s["part"].numel()), max(ints, s["ctr"].numel())
        torch.cuda.current_stream(dev).synchronize()
    s = _scr[dev.index] = {"part": torch.empty(floats, dtype=torch.float32, device=dev),
                           "ctr": torch.zeros(ints, dtype=torch.int32, device=dev)}
    return s


def reserve(dev: torch.device, floats: int, ints: int = 4096) -> None:
    _scratch(dev, floats, ints)


def scratch_bytes(dev: torch.device) -> int:
    s = _scr.get(dev.index)
    return 0 if s is None else s["part"].numel() * 4 + s["ctr"].numel() * 4


class Exl3GemvPlan:
    """The EXL3 matrices of one layer that read the same k inputs (a stacked layer's parts, or one matrix), with the
    launch layout per row count cached: one ctypes descriptor array per (m, launch) whose static fields are filled once."""

    def __init__(self, parts, k: int):
        self.parts = list(parts)
        self.k = k
        self.n_total = max(p.dst + p.n for p in self.parts)
        self.dev = self.parts[0].trellis.device
        for p in self.parts:
            assert p.trellis.shape[0] * 16 == k and p.n % 128 == 0 and k % 128 == 0, (p.trellis.shape, k, p.n)
            assert p.trellis.dtype == torch.int16 and p.trellis.is_contiguous()
            assert p.suh.dtype == torch.float16 and p.svh.dtype == torch.float16
            assert not (p.mcg and p.mul1) and (p.mcg or p.mul1), "exl3 gemv: one codebook marker (mcg or mul1)"
            assert 2 <= p.K <= 6, f"exl3 gemv: K={p.K} not built"
        # widest row count this plan should take: a matrix whose 128-column tiles do not pair up (engram.wkv's 3200 = 25
        # tiles) runs the one-tile WMMA block, which loses to the dequant path above ~16 rows (measured 24 rows: 150 vs
        # 123 us); everything in the attention path pairs up and wins at every width up to 48
        self.max_m = 48 if all(p.n // 128 % 2 == 0 for p in self.parts) else ODD_M
        # rows where exllamav3's multi-row GEMV is faster: very long inputs (the drafter's main_proj, 15360 -> 5120) at
        # 3..8 rows, where our row tile is ALU-bound on the decode (6 rows: 236 us vs 147; 1-2 and 9+ rows: ours wins)
        self.exl3_rows = range(3, 9) if k >= 8192 else range(0)
        self._launches: dict[int, list] = {}

    def takes(self, m: int) -> bool:
        """Whether an m-row call should run here (Exl3LinearMethod.apply asks; otherwise its exllamav3 / dequant path)."""
        return m <= self.max_m and m not in self.exl3_rows

    @staticmethod
    def _cb(p) -> int:
        return 2 if p.mul1 else 1

    def launches(self, m: int) -> list:
        """[(K, cb, M, mma, desc array, [(part index, row0, rows)], floats, ints)] for an m-row call."""
        got = self._launches.get(m)
        if got is not None:
            return got
        L = lib()
        maxg = L.r9k_exl3_gemv_max_groups()
        mma = int(m >= MMA_FROM and m > 1)
        cap = min(MMA_ROWS, L.r9k_exl3_gemv_max_rows_mma()) if mma else max(1, min(ROWS, L.r9k_exl3_gemv_max_rows()))
        ngr = -(-m // cap)                          # row groups, rows spread evenly
        M = -(-m // ngr)
        if mma:
            want = NB2 or 2                         # 2 measured best (4: no better, 1: -25 % at 48 rows)
            mma = next(v for v in (4, 2, 1) if v <= want and all(p.n // 128 % v == 0 for p in self.parts))
        buckets: dict[tuple[int, int], list] = {}
        for j, p in enumerate(self.parts):
            for r0 in range(0, m, M):
                buckets.setdefault((p.K, self._cb(p)), []).append((j, r0, min(M, m - r0)))
        out = []
        for (K, cb), groups in buckets.items():
            for i in range(0, len(groups), maxg):
                gs = groups[i:i + maxg]
                desc = (ctypes.c_long * (_NDESC * len(gs)))()
                floats = ints = 0
                for gi, (j, r0, rows) in enumerate(gs):
                    p = self.parts[j]
                    ks = ks_for(self.k, p.n, M, ngr, bool(mma), max(1, mma))
                    d = gi * _NDESC
                    desc[d + 0], desc[d + 1], desc[d + 2] = p.trellis.data_ptr(), p.suh.data_ptr(), p.svh.data_ptr()
                    desc[d + 5], desc[d + 6], desc[d + 7], desc[d + 10] = self.k, p.n, rows, ks
                    floats += ks * rows * p.n
                    ints += p.n // 128
                out.append((K, cb, M, mma, desc, gs, floats, ints))
        self._launches[m] = out
        return out

    def scratch_need(self, m_max: int) -> tuple[int, int]:
        f = i = 0
        for m in range(1, m_max + 1):
            for *_, floats, ints in self.launches(m):
                f, i = max(f, floats), max(i, ints)
        return f, i

    def reserve(self, m_max: int) -> None:
        f, i = self.scratch_need(m_max)
        reserve(self.dev, f, max(i, 4096))


def _check2(t: torch.Tensor, what: str) -> None:
    assert t.dim() == 2 and t.stride(1) == 1 and t.stride(0) % 4 == 0 and t.data_ptr() % 16 == 0, \
        f"r9k exl3 gemv: {what} {tuple(t.shape)} strides {t.stride()} (need unit columns, row stride % 4, 16-B aligned)"


def r9k_exl3_gemv(groups, x, out, rotated: bool = False):
    """y = x @ W for every part of ``groups`` (an Exl3GemvPlan, or a list of Exl3Part sharing k) -- see the module doc
    for the x / out forms. Returns ``out``."""
    plan = groups if isinstance(groups, Exl3GemvPlan) else Exl3GemvPlan(groups, groups[0].trellis.shape[0] * 16)
    xs = x if isinstance(x, (list, tuple)) else None
    outs = out if isinstance(out, (list, tuple)) else None
    if rotated and xs is None:
        assert len(plan.parts) == 1, "r9k exl3 gemv: rotated inputs are per part (pass a list)"
        xs = [x]
    m = (xs[0] if xs is not None else x).shape[0]
    if m == 0:
        return out
    if xs is not None:
        assert len(xs) == len(plan.parts)
        for t in xs:
            _check2(t, "x")
            assert t.shape == (m, plan.k) and (t.dtype == torch.float16 if rotated else t.dtype in _XM)
    else:
        _check2(x, "x")
        assert x.shape[1] == plan.k and x.dtype in _XM, (x.shape, x.dtype, plan.k)
    if outs is not None:
        assert len(outs) == len(plan.parts)
        for t, p in zip(outs, plan.parts):
            _check2(t, "out")
            assert t.shape == (m, p.n) and t.dtype in _YM
    else:
        _check2(out, "out")
        assert out.shape == (m, plan.n_total) and out.dtype in _YM, (out.shape, out.dtype, plan.n_total)
    L = lib()
    dev = plan.dev
    stream = torch.cuda.current_stream(dev).cuda_stream
    for K, cb, M, mma, desc, gs, floats, ints in plan.launches(m):
        s = _scratch(dev, floats, ints)
        for gi, (j, r0, rows) in enumerate(gs):
            p = plan.parts[j]
            xt = xs[j] if xs is not None else x
            if outs is not None:
                yt, ycol = outs[j], 0
            else:
                yt, ycol = out, p.dst
            d = gi * _NDESC
            desc[d + 3] = xt.data_ptr() + r0 * xt.stride(0) * xt.element_size()
            desc[d + 4] = yt.data_ptr() + (r0 * yt.stride(0) + ycol) * yt.element_size()
            desc[d + 8], desc[d + 9] = xt.stride(0), yt.stride(0)
            desc[d + 11] = 2 if rotated else _XM[xt.dtype]
            desc[d + 12] = _YM[yt.dtype]
        rc = L.r9k_exl3_gemv(ctypes.addressof(desc), len(gs), M, K, cb, mma, s["part"].data_ptr(), s["part"].numel(),
                             s["ctr"].data_ptr(), s["ctr"].numel(), stream)
        if rc != 0:
            raise RuntimeError(f"r9k_exl3_gemv failed ({rc}): m={m} K={K} cb={cb} k={plan.k} "
                               f"n={[plan.parts[j].n for j, _, _ in gs]}")
    return out


def dq(trellis: torch.Tensor, K: int, cb: int = 2) -> torch.Tensor:
    """Validation: the kernel's trellis decode of a whole matrix, f16 [in, out] rotated basis (= ext.reconstruct)."""
    k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
    o = torch.empty((k, n), dtype=torch.float16, device=trellis.device)
    rc = lib().r9k_exl3_gemv_dq(trellis.data_ptr(), o.data_ptr(), k, n, K, cb,
                                torch.cuda.current_stream(trellis.device).cuda_stream)
    assert rc == 0, rc
    return o
