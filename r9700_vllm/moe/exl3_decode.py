"""Decode tier of the EXL3 routed experts: kernels/r9k_moe_exl3.hip through ctypes (libr9k.so).

One MoE layer at decode = two launches (gate|up + activation + down-input rotation; down + output rotation + per-token
weighted sum) for any token count up to MAX_SLOTS / topk, against exllamav3's exl3_mgemm tier's 13 (T = 1) / 26 (T > 1).
Routing goes in as vLLM produced it (global ids int32, weights f32) with the layer's global -> local map; slots not on
this rank are skipped on device, slots sharing an expert are decoded together (up to ROWS a group). No host sync, no
per-call memset: graph-capturable, deterministic. Design and numbers: kernels/r9k_moe_exl3.hip, notes/exl3-plan.md.

Knobs (process-wide): R9K_EXL3_DEC_P (group lanes, grid z; default min(slots, 16)), R9K_EXL3_DEC_KS1 / _KS2 (input
chunks of gate|up and down; default by token count), R9K_EXL3_DEC_ROWS (max slots a group; default by token count:
up to 8 at T <= 8, 2 above; 1 = no sharing).
"""
from __future__ import annotations

import ctypes
import os

import torch

_L = None
_bufs: dict = {}

P_CAP = int(os.environ.get("R9K_EXL3_DEC_P", 16))
KS1 = int(os.environ.get("R9K_EXL3_DEC_KS1", 0))
KS2 = int(os.environ.get("R9K_EXL3_DEC_KS2", 0))
ROWS = int(os.environ.get("R9K_EXL3_DEC_ROWS", 0))      # 0 = by token count (rows_for)
# Above XR_T tokens the gate / up input rotation runs once a local slot (k_rot, a third launch) instead of in every one of
# k_gu's blocks; 0 = never
XR_T = int(os.environ.get("R9K_EXL3_DEC_XR_T", 8))


def lib() -> ctypes.CDLL:
    global _L
    if _L is None:
        path = os.environ.get("R9K_LIB") or os.path.join(os.path.dirname(__file__), "..", "kernels", "libr9k.so")
        L = ctypes.CDLL(path)
        if not hasattr(L, "r9k_moe_exl3_decode"):
            raise RuntimeError(f"{path} has no r9k_moe_exl3_decode (rebuild kernels/)")
        L.r9k_moe_exl3_decode.restype = ctypes.c_int
        L.r9k_moe_exl3_decode.argtypes = ([ctypes.c_long, ctypes.c_int] + [ctypes.c_long] * 3 + [ctypes.c_int] * 6
                                          + [ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_float]
                                          + [ctypes.c_long] * 6 + [ctypes.c_int] * 5 + [ctypes.c_long])
        L.r9k_moe_exl3_ctr_ints.restype = ctypes.c_int
        L.r9k_moe_exl3_ctr_ints.argtypes = [ctypes.c_int] * 4
        L.r9k_moe_exl3_max_slots.restype = ctypes.c_int
        if hasattr(L, "r9k_moe_exl3_decode_xr"):
            L.r9k_moe_exl3_decode_xr.restype = ctypes.c_int
            L.r9k_moe_exl3_decode_xr.argtypes = ([ctypes.c_long, ctypes.c_int] + [ctypes.c_long] * 3 + [ctypes.c_int] * 6
                                                 + [ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_float]
                                                 + [ctypes.c_long] * 6 + [ctypes.c_int] * 5 + [ctypes.c_long] * 2)
        if hasattr(L, "r9k_moe_exl3_groups"):
            L.r9k_moe_exl3_groups.restype = ctypes.c_int
            L.r9k_moe_exl3_groups.argtypes = [ctypes.c_long] * 2 + [ctypes.c_int] * 5 + [ctypes.c_long] * 2
        L.r9k_exl3_dq_tiles.restype = ctypes.c_int
        L.r9k_exl3_dq_tiles.argtypes = [ctypes.c_long] * 2 + [ctypes.c_int] * 4 + [ctypes.c_long]
        _L = L
    return _L


def available() -> bool:
    try:
        lib()
        return True
    except Exception:
        return False


def max_slots() -> int:
    return lib().r9k_moe_exl3_max_slots()


def _fit(segs: int, want: int, ok=lambda k: True) -> int:
    """Smallest chunk count >= want that divides the 128-blocks evenly (and passes ok), else the largest that does."""
    good = [k for k in range(1, segs + 1) if segs % k == 0 and ok(k)]
    return next((k for k in good if k >= want), good[-1])


def rows_for(T: int) -> int:
    """Slots a group (one expert's weights decoded once for up to this many tokens). Above 8 tokens 2: the 2-row
    instance's smaller LDS and register footprint beat sharing a third row (gfx1201, 2026-11-03, T = 48 36 picks over
    25 experts: R 8 790 us, R 4 767, R 2 678; T = 24: 483 / 448 / 430), and 3+ picks of one expert are rare there."""
    if ROWS:
        return max(1, min(ROWS, T, 8))
    return max(1, min(T, 8)) if T <= 8 else 2


def use_xr(T: int) -> bool:
    return 0 < XR_T < T and hasattr(lib(), "r9k_moe_exl3_decode_xr")


def split_for(T: int, H: int, I: int) -> tuple[int, int]:
    """(ks1, ks2): input chunks a block takes for gate|up (of H) and down (of I). Few groups need short chunks to
    put enough blocks in flight (measured on gfx1201, cold weights, 2026-11-02: T = 1 10/6, T = 6 8-10/3, T = 24 8/2;
    2026-11-03 with 2-row groups above 8 tokens: T = 24 / 48 4/2)."""
    mm = rows_for(T)
    # above 8 tokens 4 chunks of gate|up (T = 48: 8 -> 4 with 2 rows, 751 -> 678 us; 2 and 5 lose)
    ks1 = _fit(H // 128, KS1 or (10 if T <= 1 else 8 if T <= 8 else 4), lambda k: 2 * mm * (H // k) * 2 <= 48 * 1024)
    ks2 = _fit(I // 128, KS2 or (6 if T <= 1 else 3 if T <= 8 else 2))
    return ks1, ks2


def _need(T: int, topk: int, H: int, I: int) -> dict:
    """Element counts covering every call of up to T tokens at the splits split_for() picks. One f32 partial buffer
    serves both kernels (k_gu's partials are dead once k_dn starts: same stream, k_dn never reads them)."""
    need = {"part": 0, "ar": 0, "os": 0, "ctr": 0, "xr": 0}
    for t in range(1, T + 1):
        S = t * topk
        ks1, ks2 = split_for(t, H, I)
        need["part"] = max(need["part"], S * ks1 * 2 * I, S * ks2 * H)
        need["ar"] = max(need["ar"], S * I)
        need["os"] = max(need["os"], S * H)
        if use_xr(t):
            need["xr"] = max(need["xr"], S * 2 * H)
        need["ctr"] = max(need["ctr"], lib().r9k_moe_exl3_ctr_ints(t, topk, H, I))
    return need


def _scratch(dev: torch.device, T: int, topk: int, H: int, I: int):
    """Process-wide per device (layers run one after another on one stream; the arrival counters are zero between
    launches). Sized once for the largest call -- reserve() before capture -- never grown step by step: every
    superseded set would stay in the caching allocator's pool (that cost ~1 GiB a rank at 32 rows)."""
    b = _bufs.get(dev.index)
    if b is not None and b["T"] >= T and b["topk"] >= topk and b["H"] == H and b["I"] == I:
        return b
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("exl3 decode: scratch must be sized before graph capture (reserve() the decode sizes)")
    need = _need(T, topk, H, I)
    if b is not None:
        need = {k: max(v, b[k].numel()) for k, v in need.items()}
        T, topk = max(T, b["T"]), max(topk, b["topk"])
        _bufs.pop(dev.index)
        del b
    # one allocation carved into 256-byte aligned views (separate mid-size tensors each round up to an allocator segment)
    layout = (("part", torch.float32), ("os", torch.float32), ("ar", torch.float16), ("xr", torch.float16),
              ("ctr", torch.int32))
    offs, n = {}, 0
    for k, dt in layout:
        offs[k] = n
        n += (need[k] * torch.empty((), dtype=dt).element_size() + 255) // 256 * 256
    raw = torch.zeros(n, dtype=torch.uint8, device=dev)
    b = _bufs[dev.index] = {"T": T, "topk": topk, "H": H, "I": I, "raw": raw}
    for k, dt in layout:
        sz = torch.empty((), dtype=dt).element_size()
        b[k] = raw[offs[k]:offs[k] + need[k] * sz].view(dt)
    return b


def reserve(dev: torch.device, T: int, topk: int, H: int, I: int):
    """Size the scratch for every call of up to T tokens, before capture."""
    _scratch(dev, T, topk, H, I)


def scratch_bytes(dev: torch.device) -> int:
    b = _bufs.get(dev.index)
    return 0 if b is None else b["raw"].numel()


class Tables:
    """The layer's nine int64 pointer tables (gate/up/down x trellis/suh/svh) as the host array the launcher reads."""

    def __init__(self, layer):
        roles = ("gate", "up", "down")
        ts = [getattr(layer, f"exl3_{r}_{f}_ptrs") for r in roles for f in ("trellis", "suh", "svh")]
        self.keep = ts
        self.arr = (ctypes.c_long * 9)(*[t.data_ptr() for t in ts])


def moe_decode(x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, g2l: torch.Tensor, E: int,
               tables: Tables, K: int, cb: int, act_limit: float, I: int, out_f32: bool = False) -> torch.Tensor:
    """x [T, H] bf16/f16, topk_weights [T, topk] f32, topk_ids [T, topk] int32 (global), g2l int64 [G] (>= E or < 0
    = not on this rank) -> [T, H] (x's dtype, or f32): this rank's routed partial sum."""
    T, H = x.shape
    topk = topk_ids.shape[1]
    S = T * topk
    dev = x.device
    if x.dtype not in (torch.bfloat16, torch.float16):
        x = x.to(torch.bfloat16)
    x = x.contiguous()
    ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
    ids = ids.contiguous()
    w = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.float()
    w = w.contiguous()
    ks1, ks2 = split_for(T, H, I)
    b = _scratch(dev, T, topk, H, I)
    y_f32 = out_f32 or x.dtype != torch.bfloat16
    y = torch.empty((T, H), dtype=torch.float32 if y_f32 else torch.bfloat16, device=dev)
    R = rows_for(T)
    P = max(1, min(S, P_CAP))
    rc = lib().r9k_moe_exl3_decode_xr(
        x.data_ptr(), int(x.dtype == torch.float16), ids.data_ptr(), w.data_ptr(), g2l.data_ptr(), g2l.numel(), E,
        T, topk, H, I, ctypes.addressof(tables.arr), K, cb, act_limit,
        b["part"].data_ptr(), b["part"].data_ptr(), b["ar"].data_ptr(), b["os"].data_ptr(), b["ctr"].data_ptr(),
        y.data_ptr(), int(y_f32), R, P, ks1, ks2, b["xr"].data_ptr() if use_xr(T) else 0,
        torch.cuda.current_stream(dev).cuda_stream)
    if rc != 0:
        raise RuntimeError(f"r9k_moe_exl3_decode failed ({rc}): T={T} topk={topk} H={H} I={I} K={K} cb={cb} "
                           f"ks={ks1},{ks2}")
    return y if (out_f32 or y.dtype == x.dtype) else y.to(x.dtype)


def groups(topk_ids: torch.Tensor, g2l: torch.Tensor, E: int, R: int | None = None) -> list:
    """Validation: the slot groups the kernels derive on device for this routing, [(local expert, [slots])] in group
    order. Every group is one pass over its expert's trellis (in each segment / chunk block), so the trellis bytes a
    call reads = len(groups) x an expert's bytes."""
    T, topk = topk_ids.shape
    R = R or rows_for(T)
    MM = 8
    ids = topk_ids.to(torch.int32).contiguous()
    out = torch.full((2 + T * topk * (2 + MM),), -1, dtype=torch.int32, device=ids.device)
    rc = lib().r9k_moe_exl3_groups(ids.data_ptr(), g2l.data_ptr(), g2l.numel(), E, T, topk, R, out.data_ptr(),
                                   torch.cuda.current_stream(ids.device).cuda_stream)
    assert rc == 0, rc
    o = out.cpu().tolist()
    ng = o[1]
    res = []
    for g in range(ng):
        row = o[2 + g * (2 + MM): 2 + (g + 1) * (2 + MM)]
        res.append((row[0], row[2:2 + row[1]]))
    return res


def dq_tiles(trellis: torch.Tensor, cb: int = 2) -> torch.Tensor:
    """Validation: the kernel's trellis decode of a whole matrix, f16 [in, out] rotated basis (= ext.reconstruct)."""
    k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
    out = torch.empty((k, n), dtype=torch.float16, device=trellis.device)
    rc = lib().r9k_exl3_dq_tiles(trellis.data_ptr(), out.data_ptr(), k, n, trellis.shape[-1] // 16, cb,
                                 torch.cuda.current_stream(trellis.device).cuda_stream)
    assert rc == 0, rc
    return out
