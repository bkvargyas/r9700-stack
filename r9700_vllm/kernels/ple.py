"""Binding for libr9k's PLE int6 row gather + dequant."""
from __future__ import annotations

import ctypes

import torch

from .moe import lib as _lib

_BOUND = False


def _L():
    global _BOUND
    L = _lib()
    if not _BOUND:
        L.r9k_ple_gather_int6.restype = ctypes.c_int
        L.r9k_ple_gather_int6.argtypes = [ctypes.c_long] * 4 + [ctypes.c_int] * 2 + [ctypes.c_long] * 2
        _BOUND = True
    return L


def int6_row_bytes(head_dim: int) -> int:
    return head_dim * 6 // 8 + head_dim // 32 * 2


def gather_int6(table: torch.Tensor, ids: torch.Tensor, head_dim: int, out_dtype=torch.bfloat16) -> torch.Tensor:
    """table [rows, row_bytes] uint8 (device or UVA view), ids int64 (any shape) -> [*ids.shape, head_dim] bf16."""
    assert out_dtype == torch.bfloat16
    ids_c = ids.reshape(-1).to(torch.int64).contiguous()
    out = torch.empty((*ids.shape, head_dim), dtype=torch.bfloat16, device=ids.device)
    rc = _L().r9k_ple_gather_int6(table.data_ptr(), ids_c.data_ptr(), out.data_ptr(), ids_c.numel(),
                                  table.shape[1], head_dim, table.shape[0],
                                  torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_ple_gather_int6 failed ({rc})")
    return out
