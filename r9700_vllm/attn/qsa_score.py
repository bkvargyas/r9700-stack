"""Our QSA indexer scoring (kernels/r9k_qsa_score.hip) in place of vLLM's Triton ``_qsa_mqa_paged_kernel``.

The stock scorer launches one program per (query row, 32 columns) over the whole context capacity, re-reading the
compressed keys once per row and writing every column of an fp32 [rows, capacity] logits buffer, although the
top-k that follows (``top_k_per_row_decode``) reads row r only up to its visible count. For a 4096-token prefill
chunk that is ~1.85 ms per layer. Ours stages K once per 16-row tile and stops at the tile's last visible column.

``select_paged_tokens`` mirrors the stock ``qsa_select_paged_tokens`` (same chunking, same top-k and expansion
kernels) with our scorer; ``install_indexers`` binds it as ``_select`` on every ``QSAIndexer`` instance (the
stock ``_select`` imports the op lazily per call, so an instance method is the narrowest hook). R9K_QSA_SCORE=stock
keeps vLLM's kernel.
"""
from __future__ import annotations

import ctypes
import math
import os
import types

import torch

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)

_L = None
_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024       # stock's chunking bound, kept for identical memory behaviour


def lib():
    global _L
    if _L is None:
        from ..kernels import moe as KM
        L = KM.lib()
        L.r9k_qsa_score.restype = ctypes.c_int
        L.r9k_qsa_score.argtypes = [ctypes.c_long] * 3 + [ctypes.c_int] * 2 + [ctypes.c_long] * 3 + [ctypes.c_int] + \
            [ctypes.c_long] * 2 + [ctypes.c_int] + [ctypes.c_long] * 3 + [ctypes.c_int] * 6 + [ctypes.c_float] + \
            [ctypes.c_long] * 4
        _L = L
    return _L


def available() -> bool:
    try:
        return hasattr(lib(), "r9k_qsa_score")
    except Exception:
        return False


def num_splits(rows: int) -> int:
    """Column parts per row tile: enough workgroups to cover the 64 CUs twice when there are few tiles."""
    forced = os.environ.get("R9K_QSA_SCORE_SPLITS")
    if forced:
        return max(1, int(forced))
    tiles = -(-rows // 16)
    return 1 if tiles >= 128 else max(1, min(32, -(-128 // tiles)))


def _i32(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.int32 and t.is_contiguous() else t.to(torch.int32).contiguous()


def score(q, k_cache, page_table, token_to_req, query_positions, sequence_lengths, compress_ratio: int,
          num_columns: int | None = None, score_scale: float | None = None, nsplit: int | None = None):
    """Same contract as the stock ``qsa_mqa_paged``: returns (logits fp32 [rows, columns], visible_blocks i32
    [rows]). Logits are defined for columns below each row's tile-max visible count (-inf between the row's own
    visible count and the tile's); the rest is left uninitialised, as the stock top-k never reads it."""
    rows, heads, hd = q.shape
    if k_cache.ndim != 4 or k_cache.shape[2] != 1 or k_cache.shape[3] != hd:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if q.dtype != torch.bfloat16 or k_cache.dtype != torch.bfloat16:
        raise NotImplementedError("r9k QSA scoring needs bf16 Q and compressed keys")
    assert q.stride(2) == 1 and k_cache.stride(3) == 1 and page_table.stride(1) == 1
    divisor = math.sqrt(hd) if score_scale is None else float(score_scale)
    capacity = page_table.shape[1] * k_cache.shape[1]
    columns = capacity if num_columns is None else int(num_columns)
    logits = torch.empty((rows, columns), dtype=torch.float32, device=q.device)
    visible = torch.empty((rows,), dtype=torch.int32, device=q.device)
    if not rows or not columns:
        return logits, visible
    t2r, pos, sl, bt = _i32(token_to_req), _i32(query_positions), _i32(sequence_lengths), _i32(page_table)
    st = torch.cuda.current_stream().cuda_stream
    rc = lib().r9k_qsa_score(q.data_ptr(), q.stride(0), q.stride(1), heads, hd,
                             k_cache.data_ptr(), k_cache.stride(0), k_cache.stride(1), k_cache.shape[1],
                             bt.data_ptr(), bt.stride(0), bt.shape[1],
                             t2r.data_ptr(), pos.data_ptr(), sl.data_ptr(), bt.shape[0],
                             rows, int(compress_ratio), columns, k_cache.shape[0], nsplit or num_splits(rows), divisor,
                             logits.data_ptr(), logits.stride(0), visible.data_ptr(), st)
    if rc:
        raise RuntimeError(f"r9k_qsa_score failed ({rc})")
    return logits, visible


def select_paged_tokens(q, k_cache, page_table, token_to_req, query_positions, sequence_lengths, token_topk: int,
                        compress_ratio: int, out=None):
    """Stock ``qsa_select_paged_tokens`` with our scorer: score, top-k per row, expand; no host sync."""
    from vllm import _custom_ops as ops
    from vllm.models.qwen4_exp.amd.ops.qsa import expand_qsa_block_indices_cuda
    rows = q.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    if out.shape != (rows, output_width):
        raise ValueError("QSA selection output has an invalid shape")
    if not rows:
        return out
    columns = page_table.shape[1] * k_cache.shape[1]
    block_topk = token_topk // compress_ratio
    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
    blocks_buffer = torch.empty((min(rows, rows_per_chunk), block_topk), dtype=torch.int32, device=q.device)
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        rs = slice(row_start, row_end)
        logits, visible = score(q[rs], k_cache, page_table, token_to_req[rs], query_positions[rs], sequence_lengths,
                                compress_ratio)
        blocks = blocks_buffer[: row_end - row_start]
        ops.top_k_per_row_decode(logits, 1, visible, blocks, blocks.shape[0], logits.stride(0), logits.stride(1),
                                 block_topk)
        expand_qsa_block_indices_cuda(blocks, query_positions[rs], sequence_lengths, token_to_req[rs], compress_ratio,
                                      token_topk, out[rs])
    return out


def _select(self, q, metadata, out):
    return select_paged_tokens(q, self.compressed_key_cache.kv_cache, metadata.block_table, metadata.token_to_req,
                               metadata.logical_positions, metadata.seq_lens, self.token_topk, self.compress_ratio, out)


def install_indexers(model: torch.nn.Module) -> int:
    """Bind our ``_select`` on every QSAIndexer under `model`. Returns how many were switched."""
    if os.environ.get("R9K_QSA_SCORE", "r9k") != "r9k" or not available():
        return 0
    n = 0
    for mod in model.modules():
        if type(mod).__name__ == "QSAIndexer" and hasattr(mod, "_select") and hasattr(mod, "compressed_key_cache"):
            if mod.index_head_dim != 128 or mod.index_n_heads not in (1, 2, 4, 8):
                logger.warning_once("r9700: r9k QSA scoring skipped (heads %d, head_dim %d)", mod.index_n_heads,
                                    mod.index_head_dim)
                return 0
            mod._select = types.MethodType(_select, mod)
            n += 1
    if n:
        logger.info("r9700: r9k QSA indexer scoring installed on %d layers", n)
    return n
