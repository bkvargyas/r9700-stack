"""DeepSeek-V4/V4.1 sparse-MLA decode on RDNA4 (gfx1201): stock's split-K Triton kernels under our own launcher.

Stock's `_rocm_sparse_attn_decode_ragged_triton` (v1/attention/ops/rocm_aiter_mla_sparse.py) sends every GPU but gfx942 /
gfx950 to its "fallback path for un-tuned architectures": `_sparse_attn_decode_ragged_kernel`, one workgroup per (query,
16-head block) walking the whole SWA window (<= 128 keys) and top-k segment (<= 512) serially -- on the 32 WGPs of an
R9700 one or a few are busy (180 us a layer at SWA 128 + top-k 512, flat up to 24 queries; measured on card 7). That branch also returns before the reduce, so the fused inverse
RoPE it was asked for never happens while `rocm_sparse_attn_decode` reports the rows as rotated (decode rows reach wo_a
still rotated; see models/deepseek_v41_rdna4.py).

Here the same work goes through stock's split-K pair, unmodified: `_sparse_attn_decode_partial_kernel` over (query, split,
head block) writes per-split (m, l, acc) partials, `_sparse_attn_decode_reduce_kernel` merges them, adds the attention
sink and applies the inverse RoPE in its epilogue (FUSE_INV_ROPE) -- so the rows we report as rotated are rotated.
The split count / BLOCK_K / warps are picked for 32 WGPs and wave32 (`_splits`, `_warps`; swept on card 7: ~96
workgroups, BLOCK_K 16, 8 waves while the grid is <= 48 workgroups), not by stock's gfx950-tuned selector.

`rocm_sparse_attn_decode` / `rocm_sparse_attn_decode_bf16` keep stock's signatures and return contracts; the model hook
installs them over the stock names before vLLM imports the DeepSeek-V4.1 ROCm model.
"""
from __future__ import annotations

import math
import os

import torch

from vllm.triton_utils import triton

_BLOCK_H = 16          # tl.dot needs M >= 16 on gfx12 (WMMA 16x16x16); the 8 local heads of TP8 fill half a block
_MAX_SPLITS = 64


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# R9K_DSV4_DECODE_SPLITS / _BLOCK_K / _WARPS / _WG pin the tuning for sweeps; 0 = auto.
_FORCE_SPLITS = _env_int("R9K_DSV4_DECODE_SPLITS", 0)
_BLOCK_K = _env_int("R9K_DSV4_DECODE_BLOCK_K", 16)
_WARPS = _env_int("R9K_DSV4_DECODE_WARPS", 0)
_TARGET_WG = _env_int("R9K_DSV4_DECODE_WG", 96)        # partial workgroups to aim for: three per WGP
_MIN_KEYS = _env_int("R9K_DSV4_DECODE_MIN_KEYS", 16)   # keys a split should own at least (one BLOCK_K tile)


def _warps(workgroups: int) -> int:
    """8 waves a workgroup while the grid is thin (<= 48 workgroups: each one's latency is the step), else 4."""
    return _WARPS if _WARPS > 0 else (8 if workgroups <= 48 else 4)


def _splits(num_queries: int, heads_blocks: int, avg_main: float, avg_extra: float) -> int:
    """Split count from sync-free sizes: enough (query, split, head block) workgroups to cover the WGPs three times, but no
    split shorter than one BLOCK_K tile of the longer segment (more splits than tiles only adds reduce traffic)."""
    if _FORCE_SPLITS > 0:
        return min(_FORCE_SPLITS, _MAX_SPLITS)
    base = max(1, num_queries * heads_blocks)
    want = max(1, math.ceil(_TARGET_WG / base))
    longest = max(avg_main, avg_extra, 1.0)
    cap = max(1, math.ceil(longest / max(_MIN_KEYS, 1)))
    return int(max(1, min(want, cap, _MAX_SPLITS)))


def sparse_decode_fp8(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_indptr: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    inv_rope_positions: torch.Tensor | None = None,
    inv_rope_cos_sin_cache: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> torch.Tensor:
    """Split-K decode over the fp8_ds_mla caches (uint8 [blocks, block, 584]: 448 fp8 NoPE + 64 bf16 RoPE + 8 e8m0
    scale bytes a token) with ragged SWA (main) and top-k (extra) segments; bf16 output [T, H, 512], inverse-RoPE'd when
    `inv_rope_positions` is given. No host syncs: the split count comes from the index tensors' sizes."""
    from vllm.platforms import current_platform
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as O

    assert q.ndim == 3 and main_cache.ndim == 3 and main_cache.dtype == torch.uint8
    main_indices = O._as_int32_contiguous_1d(main_indices)
    main_indptr = O._as_int32_contiguous_1d(main_indptr)
    num_queries, num_heads, head_dim = q.shape
    O._validate_dsv4_sparse_dims(head_dim, nope_head_dim, rope_head_dim, "sparse_decode_fp8")
    has_sink = attn_sink is not None
    attn_sink = attn_sink.contiguous() if has_sink else torch.empty(1, device=q.device, dtype=torch.float32)

    has_extra = extra_cache is not None and extra_indices is not None and extra_indptr is not None
    if has_extra:
        extra_indices = O._as_int32_contiguous_1d(extra_indices)
        extra_indptr = O._as_int32_contiguous_1d(extra_indptr)
    else:
        extra_cache = main_cache
        extra_indices = torch.empty(0, device=q.device, dtype=torch.int32)
        extra_indptr = torch.zeros(num_queries + 1, device=q.device, dtype=torch.int32)

    if out is None:
        out = torch.empty_like(q, dtype=torch.bfloat16)
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    if num_queries == 0:
        return out

    heads_blocks = triton.cdiv(num_heads, _BLOCK_H)
    comb_dim = nope_head_dim + rope_head_dim
    if num_splits is None:
        inv_q = 1.0 / num_queries
        num_splits = _splits(num_queries, heads_blocks, main_indices.numel() * inv_q,
                             extra_indices.numel() * inv_q if has_extra else 0.0)
    is_fnuz = current_platform.is_fp8_fnuz()

    part_m = torch.empty((num_queries, num_splits, num_heads), dtype=torch.float32, device=q.device)
    part_l = torch.empty_like(part_m)
    part_acc = torch.empty((num_queries, num_splits, num_heads, comb_dim), dtype=torch.float32, device=q.device)

    nw = _warps(num_queries * num_splits * heads_blocks)
    O._sparse_attn_decode_partial_kernel[(num_queries, num_splits, heads_blocks)](
        q, main_cache, main_indices, main_indptr, extra_cache, extra_indices, extra_indptr,
        part_m, part_l, part_acc,
        q.stride(0), q.stride(1), main_cache.stride(0), extra_cache.stride(0),
        part_m.stride(0), part_m.stride(1), part_acc.stride(0), part_acc.stride(1), part_acc.stride(2),
        main_cache.shape[0] * main_cache.shape[1], extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1], extra_cache.shape[1],
        scale, num_heads,
        HAS_EXTRA=has_extra, NOPE_DIM=nope_head_dim, NOPE_BLOCK=triton.next_power_of_2(nope_head_dim),
        ROPE_DIM=rope_head_dim,
        IS_FNUZ_MAIN=is_fnuz, IS_FNUZ_EXTRA=False,   # SWA cache: C++ encoder; compressed cache: Triton, OCP everywhere
        BLOCK_H=_BLOCK_H, BLOCK_K=_BLOCK_K, NUM_SPLITS=num_splits, NUM_STAGES=1,
        num_warps=nw,
    )
    _reduce(part_m, part_l, part_acc, attn_sink, has_sink, out, num_heads, num_splits,
            nope_head_dim, rope_head_dim, inv_rope_positions, inv_rope_cos_sin_cache)
    return out


def _reduce(part_m, part_l, part_acc, attn_sink, has_sink, out, num_heads, num_splits, nope, rope,
            inv_rope_positions, inv_rope_cos_sin_cache) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as O

    fuse = inv_rope_positions is not None
    if fuse:
        assert inv_rope_cos_sin_cache is not None and inv_rope_cos_sin_cache.shape[-1] == rope
        assert nope % 2 == 0 and rope % 2 == 0
    num_queries = out.shape[0]
    O._sparse_attn_decode_reduce_kernel[(num_queries, num_heads)](
        part_m, part_l, part_acc, attn_sink, out,
        inv_rope_positions, inv_rope_cos_sin_cache, None,
        out.stride(0), out.stride(1), 0, out.shape[-1] // 32,
        part_m.stride(0), part_m.stride(1), part_acc.stride(0), part_acc.stride(1), part_acc.stride(2),
        inv_rope_cos_sin_cache.stride(0) if fuse else 0,
        num_heads,
        HAS_ATTN_SINK=has_sink, ADAPTIVE_SPLITS=False, COMB_DIM=nope + rope, BLOCK_H=1,
        NUM_SPLITS=num_splits, SPLITS_PAD=triton.next_power_of_2(num_splits),
        FUSE_INV_ROPE=fuse, NOPE=nope, HALF=rope // 2, QUANT_OUT=False, num_warps=4,
    )


def rocm_sparse_attn_decode(
    q, kv_cache, swa_k_cache, swa_only, topk_indices, topk_lens, swa_indices, swa_lens,
    swa_ragged_indices, swa_ragged_indptr, topk_ragged_indices, topk_ragged_indptr,
    attn_sink, scale, head_dim, nope_head_dim, rope_head_dim, output,
    extra_cache_nan_free: bool = False, adaptive_splits: bool = False,
    inv_rope_positions=None, inv_rope_cos_sin_cache=None, output_mxfp8=None,
) -> int:
    """Stock signature and contract (writes `output`, returns how many leading rows came back inverse-RoPE'd), split-K
    on RDNA4. extra_cache_nan_free / adaptive_splits are gfx950 tunings and ignored; MXFP8 output is gfx950-only."""
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as O

    if output_mxfp8 is not None:
        raise RuntimeError("the MXFP8 decode epilogue is gfx950-only")
    assert output is not None
    assert swa_k_cache.dtype == torch.uint8, f"expected a uint8 fp8_ds_mla SWA cache, got {swa_k_cache.dtype}"
    O._validate_dsv4_sparse_dims(head_dim, nope_head_dim, rope_head_dim, "rocm_sparse_attn_decode")

    if swa_ragged_indices is None or swa_ragged_indptr is None:
        main = swa_indices.reshape(swa_indices.shape[0], -1)
        swa_ragged_indices, swa_ragged_indptr = O.build_ragged_indices_from_dense(
            main, swa_lens if swa_lens is not None else (main >= 0).sum(dim=-1, dtype=torch.int32),
            num_rows=swa_k_cache.shape[0] * swa_k_cache.shape[1])
    extra_cache = None
    if not swa_only:
        assert kv_cache is not None and kv_cache.dtype == torch.uint8
        extra_cache = kv_cache
        if topk_ragged_indices is None or topk_ragged_indptr is None:
            assert topk_indices is not None
            ex = topk_indices.reshape(topk_indices.shape[0], -1)
            topk_ragged_indices, topk_ragged_indptr = O.build_ragged_indices_from_dense(
                ex, topk_lens if topk_lens is not None else (ex >= 0).sum(dim=-1, dtype=torch.int32),
                num_rows=kv_cache.shape[0] * kv_cache.shape[1])
    else:
        topk_ragged_indices = topk_ragged_indptr = None

    direct = output.dtype == torch.bfloat16 and output.shape == q.shape
    out = sparse_decode_fp8(
        q, swa_k_cache, swa_ragged_indices, swa_ragged_indptr, scale,
        None if attn_sink is None else attn_sink[: q.shape[1]], nope_head_dim, rope_head_dim,
        extra_cache=extra_cache, extra_indices=topk_ragged_indices, extra_indptr=topk_ragged_indptr,
        out=output if direct else None,
        inv_rope_positions=inv_rope_positions, inv_rope_cos_sin_cache=inv_rope_cos_sin_cache)
    if not direct:
        output.copy_(out.to(output.dtype))
    return output.shape[0] if inv_rope_positions is not None else 0


def sparse_decode_bf16(q, kv, indices, indptr, scale, attn_sink, nope_head_dim, rope_head_dim, out=None,
                       num_splits=None, inv_rope_positions=None, inv_rope_cos_sin_cache=None) -> torch.Tensor:
    """Split-K decode over a bf16 KV cache [skv, d] with one ragged segment, through stock's
    `_sparse_attn_decode_ragged_bf16_partial_kernel` + the shared reduce; splits picked as for fp8."""
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as O

    indices = O._as_int32_contiguous_1d(indices)
    indptr = O._as_int32_contiguous_1d(indptr)
    num_queries, num_heads, head_dim = q.shape
    O._validate_sparse_dims(head_dim, nope_head_dim, rope_head_dim, "sparse_decode_bf16")
    has_sink = attn_sink is not None
    attn_sink = attn_sink.contiguous() if has_sink else torch.empty(1, device=q.device, dtype=torch.float32)
    if out is None:
        out = torch.empty_like(q)
    if num_queries == 0:
        return out
    heads_blocks = triton.cdiv(num_heads, _BLOCK_H)
    if num_splits is None:
        num_splits = _splits(num_queries, heads_blocks, indices.numel() / num_queries, 0.0)
    comb_dim = nope_head_dim + rope_head_dim
    part_m = torch.empty((num_queries, num_splits, num_heads), dtype=torch.float32, device=q.device)
    part_l = torch.empty_like(part_m)
    part_acc = torch.empty((num_queries, num_splits, num_heads, comb_dim), dtype=torch.float32, device=q.device)
    O._sparse_attn_decode_ragged_bf16_partial_kernel[(num_queries, num_splits, heads_blocks)](
        q, kv, indices, indptr, part_m, part_l, part_acc,
        q.stride(0), q.stride(1), q.stride(2), kv.stride(0), kv.stride(1),
        part_m.stride(0), part_m.stride(1), part_acc.stride(0), part_acc.stride(1), part_acc.stride(2),
        num_heads, head_dim, kv.shape[0], float(scale),
        BLOCK_H=_BLOCK_H, BLOCK_D=triton.next_power_of_2(head_dim), BLOCK_K=_BLOCK_K_BF16, NUM_SPLITS=num_splits,
        num_warps=_warps(num_queries * num_splits * heads_blocks),
    )
    _reduce(part_m, part_l, part_acc, attn_sink, has_sink, out, num_heads, num_splits, nope_head_dim, rope_head_dim,
            inv_rope_positions, inv_rope_cos_sin_cache)
    return out


_BLOCK_K_BF16 = _env_int("R9K_DSV4_DECODE_BLOCK_K_BF16", 32)


def rocm_sparse_attn_decode_bf16(q, kv, scale, head_dim, nope_head_dim, rope_head_dim, attn_sink, output,
                                 ragged_indices, ragged_indptr, num_splits) -> None:
    """Stock signature (kv [skv, 1, d], writes `output`); the caller's gfx950-tuned `num_splits` is replaced by ours."""
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as O

    assert kv.ndim == 3 and kv.shape[1] == 1, f"expected kv=[skv,1,d], got {kv.shape}"
    O._validate_sparse_dims(head_dim, nope_head_dim, rope_head_dim, "rocm_sparse_attn_decode_bf16")
    direct = output.shape[-1] == head_dim
    out = output if direct else torch.empty((q.shape[0], q.shape[1], head_dim), dtype=output.dtype, device=output.device)
    sparse_decode_bf16(q, kv.squeeze(1), ragged_indices, ragged_indptr, scale,
                       None if attn_sink is None else attn_sink[: q.shape[1]], nope_head_dim, rope_head_dim, out=out)
    if not direct:
        output.copy_(out[..., : output.shape[-1]])
