"""Decode-side indexer logits over the tiled (16x16 "pre-shuffled") paged K cache, for GPUs where stock has no reader for it.

GLM-5.3-Flash's indexer K cache is written by vLLM's kpool compress kernel in pages of 64 pooled entries, each page laid
out in 16-entry x 16-dim tiles: byte (t, d) of a page lives at (t//16)*16*D + (d//16)*256 + (t%16)*16 + d%16, followed
by the 64 fp32 scales in entry order. On gfx942/gfx950 vLLM reads it back with aiter's pre-shuffle (Gluon) kernel; on
every other GPU its wrapper falls through to aiter's stage-1 kernel, which reads a linear page -- so the decode logits are
scrambled, the top-k misses the right pools, and retrieval fails once the context is long enough that the needle is no
longer rescued by chance (measured: top-512 overlap with the reference 68% at 760 pools, 18% at 3.2k, 3% at 32k).

paged_mqa_logits() computes logits[row, p] = scale_p * sum_h w[row, h] * relu(q[row, h, :] . k_p) exactly as the torch
reference (bf16 products, fp32 accumulation), reading the tiled layout; -inf beyond each row's context. One program per
(row, page); no host syncs, so it captures into full graphs like the stock kernel it replaces.
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _paged_mqa_logits_tiled_kernel(
    q_ptr, w_ptr, kv_ptr, ks_ptr, bt_ptr, ctx_ptr, out_ptr,
    stride_q_row, stride_q_h, stride_w_row, stride_kv_page, stride_ks_page, stride_bt_b, stride_out_row,
    max_pages, max_model_len,
    NEXT_N: tl.constexpr, CTX_PER_ROW: tl.constexpr, H: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
):
    row = tl.program_id(0)
    pg = tl.program_id(1)
    b = row // NEXT_N
    n = row % NEXT_N
    if CTX_PER_ROW:
        limit = tl.load(ctx_ptr + row)
    else:
        limit = tl.load(ctx_ptr + b) - (NEXT_N - 1 - n)       # row n of a verify batch sees one key fewer per step
    limit = tl.minimum(tl.maximum(limit, 0), max_model_len)
    p = tl.arange(0, PAGE)
    pos = pg * PAGE + p
    valid = pos < limit
    out_row = out_ptr + row * stride_out_row
    if pg * PAGE >= limit:
        tl.store(out_row + pos, tl.full((PAGE,), float("-inf"), tl.float32), mask=pos < max_model_len)
        return
    page = tl.load(bt_ptr + b * stride_bt_b + pg, mask=pg < max_pages, other=0).to(tl.int64)
    h = tl.arange(0, H)
    d = tl.arange(0, D)
    q = tl.load(q_ptr + row * stride_q_row + h[:, None] * stride_q_h + d[None, :]).to(tl.bfloat16)       # [H, D]
    off = (p[:, None] // 16) * (16 * D) + (d[None, :] // 16) * 256 + (p[:, None] % 16) * 16 + (d[None, :] % 16)
    k = tl.load(kv_ptr + page * stride_kv_page + off).to(tl.bfloat16)                                       # [PAGE, D]
    s = tl.dot(q, tl.trans(k)).to(tl.float32)                                                               # [H, PAGE]
    w = tl.load(w_ptr + row * stride_w_row + h).to(tl.float32)
    s = tl.maximum(s, 0.0) * w[:, None]
    logit = tl.sum(s, axis=0)                                                                               # [PAGE]
    sc = tl.load(ks_ptr + page * stride_ks_page + p, mask=valid, other=0.0)
    logit = tl.where(valid, logit * sc, float("-inf"))
    tl.store(out_row + pos, logit, mask=pos < max_model_len)


def paged_mqa_logits(q_fp8: torch.Tensor, kv_cache: torch.Tensor, weights: torch.Tensor, context_lens: torch.Tensor,
                     block_tables: torch.Tensor, max_model_len: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """q_fp8 [B, next_n, H, D] fp8; kv_cache uint8 pages [num_pages, page, (1,) D+4] in the tiled layout; weights
    [B*next_n, H] fp32; context_lens [B] or [B, next_n] int32 (pooled entries); block_tables [B, max_pages] int32.
    Returns logits [B*next_n, max_model_len] fp32, -inf beyond each row's context."""
    B, NN, H, D = q_fp8.shape
    rows = B * NN
    page = kv_cache.shape[1]
    assert page % 16 == 0 and D % 16 == 0 and (H & (H - 1)) == 0, (page, D, H)
    npages = kv_cache.shape[0]
    flat = kv_cache.view(npages, -1)                       # uint8 [npages, page*(D+4)]
    assert flat.shape[1] == page * (D + 4), (flat.shape, page, D)
    kv_vals = flat.view(q_fp8.dtype)                       # fp8 view, same element count
    kv_scales = flat[:, page * D:].view(torch.float32)     # [npages, page] fp32 scales in entry order
    ctx = context_lens
    ctx_per_row = ctx.dim() == 2 and ctx.shape[1] == NN and NN > 1
    if ctx.dim() == 2 and not ctx_per_row:
        ctx = ctx[:, -1].contiguous()
    ctx = ctx.reshape(-1).to(torch.int32)
    if out is None:
        out = torch.empty((rows, max_model_len), dtype=torch.float32, device=q_fp8.device)
    q = q_fp8.reshape(rows, H, D)
    grid = (rows, triton.cdiv(max_model_len, page))
    _paged_mqa_logits_tiled_kernel[grid](
        q, weights, kv_vals, kv_scales, block_tables, ctx, out,
        q.stride(0), q.stride(1), weights.stride(0), kv_vals.stride(0), kv_scales.stride(0), block_tables.stride(0), out.stride(0),
        block_tables.shape[1], max_model_len,
        NEXT_N=NN, CTX_PER_ROW=ctx_per_row, H=H, D=D, PAGE=page, num_warps=4)
    return out
