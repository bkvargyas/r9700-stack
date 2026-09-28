"""Our QSA sparse attention (kernels/r9k_qsa.hip) for Qwen4Exp full-attention layers, in place of vLLM's Triton
``qsa_sparse_paged_attention``.

Installed per layer by the model subclasses (models/qwen4_exp.py): each ``Qwen4ExpQSAAttention``'s impl gets our
``forward_qsa`` bound on the instance after construction (no class patching). R9K_QSA=stock keeps vLLM's kernel.

Pipeline per call (all on the current stream, no host sync, cudagraph-safe):
  1. r9k_qsa_bitmap: each row's expanded token list -> a bitmap over 4-token groups + the row's position
  2. r9k_qsa_attn:   tiles of 16 rows walk the union of their groups; K/V staged once per tile (see the .hip)
  3. r9k_qsa_merge:  only when the group range is split (decode: few tiles would leave most CUs idle)
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
_SCRATCH: dict = {}
_GS = 4                  # indexer_compress_ratio (the kernel's group size)


def lib():
    global _L
    if _L is None:
        from ..kernels import moe as KM
        L = KM.lib()
        L.r9k_qsa_bitmap.restype = ctypes.c_int
        L.r9k_qsa_bitmap.argtypes = [ctypes.c_long] * 2 + [ctypes.c_int] + [ctypes.c_long] * 3 + \
            [ctypes.c_int] * 2 + [ctypes.c_long] + [ctypes.c_int] + [ctypes.c_long] * 2
        L.r9k_qsa_attn.restype = ctypes.c_int
        L.r9k_qsa_attn.argtypes = [ctypes.c_long] * 8 + [ctypes.c_int] * 2 + [ctypes.c_long] * 5 + \
            [ctypes.c_int] * 4 + [ctypes.c_long] * 5 + [ctypes.c_int] + [ctypes.c_long]
        L.r9k_qsa_merge.restype = ctypes.c_int
        L.r9k_qsa_merge.argtypes = [ctypes.c_long] * 6 + [ctypes.c_int] * 3 + [ctypes.c_long]
        _L = L
    return _L


def available() -> bool:
    try:
        return hasattr(lib(), "r9k_qsa_attn")
    except Exception:
        return False


def _scratch(device, rows: int, bw: int):
    """Bitmap [rows, bw] u32 + positions [rows] i32, grown (never shrunk) per device. The first call is vLLM's
    profile run at max_num_batched_tokens, so capture-time calls reuse it."""
    key = str(device)
    cur = _SCRATCH.get(key)
    if cur is None or cur[0].shape[0] < rows or cur[0].shape[1] < bw:
        r = max(rows, cur[0].shape[0] if cur is not None else 0, int(os.environ.get("R9K_QSA_MIN_ROWS", "4096")))
        w = max(bw, cur[0].shape[1] if cur is not None else 0)
        cur = (torch.empty((r, w), dtype=torch.int32, device=device),
               torch.empty((r,), dtype=torch.int32, device=device))
        _SCRATCH[key] = cur
    return cur


def num_splits(rows: int, hkv: int) -> int:
    """Split the group range when there are too few row tiles to fill the GPU (64 CUs on R9700)."""
    forced = os.environ.get("R9K_QSA_SPLITS")
    if forced:
        return max(1, int(forced))
    tiles = math.ceil(rows / 16) * hkv
    if tiles >= 48:
        return 1
    return max(1, min(16, math.ceil(64 / tiles)))


def sparse_attention(q, key_cache, value_cache, logical_indices, block_table, token_to_req, query_start_loc,
                     seq_lens, out, nsplit: int | None = None):
    """q [rows, hq, 256] bf16; key/value_cache [blocks, page, hkv, 256] bf16 views (any strides, head dim
    contiguous); logical_indices [rows, width] i32 (-1 padded, the stock expanded list); block_table [nreq, W];
    token_to_req [rows]; query_start_loc [nreq+1]; seq_lens [nreq]; out [rows, hq, 256] (pre-zeroed by caller)."""
    rows, hq, hd = q.shape
    if rows == 0:
        return out
    L = lib()
    hkv, page = key_cache.shape[2], key_cache.shape[1]
    assert hd == 256 and q.stride(2) == 1 and key_cache.stride(3) == 1 and value_cache.stride(3) == 1
    assert key_cache.stride() == value_cache.stride() and out.stride(2) == 1
    assert logical_indices.stride(1) == 1 and block_table.stride(1) == 1
    bw = (block_table.shape[1] * page // _GS + 31) // 32
    bm, pos = _scratch(q.device, rows, bw)
    st = torch.cuda.current_stream().cuda_stream
    nreq = seq_lens.shape[0]
    rc = L.r9k_qsa_bitmap(logical_indices.data_ptr(), logical_indices.stride(0), logical_indices.shape[1],
                          token_to_req.data_ptr(), query_start_loc.data_ptr(), seq_lens.data_ptr(), nreq, rows,
                          bm.data_ptr(), bm.shape[1], pos.data_ptr(), st)
    if rc:
        raise RuntimeError(f"r9k_qsa_bitmap failed ({rc})")
    ns = nsplit or num_splits(rows, hkv)
    po = plse = None
    if ns > 1:
        po = torch.empty((ns, rows, hq, hd), dtype=torch.float32, device=q.device)
        plse = torch.empty((ns, rows, hq), dtype=torch.float32, device=q.device)
    rc = L.r9k_qsa_attn(q.data_ptr(), q.stride(0), q.stride(1), key_cache.data_ptr(), value_cache.data_ptr(),
                        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), page, hd,
                        block_table.data_ptr(), block_table.stride(0), token_to_req.data_ptr(), pos.data_ptr(),
                        bm.data_ptr(), bm.shape[1], rows, hq, hkv, out.data_ptr(), out.stride(0), out.stride(1),
                        po.data_ptr() if po is not None else 0, plse.data_ptr() if plse is not None else 0, ns, st)
    if rc:
        raise RuntimeError(f"r9k_qsa_attn failed ({rc})")
    if ns > 1:
        rc = L.r9k_qsa_merge(po.data_ptr(), plse.data_ptr(), out.data_ptr(), out.stride(0), out.stride(1),
                             pos.data_ptr(), rows, hq, ns, st)
        if rc:
            raise RuntimeError(f"r9k_qsa_merge failed ({rc})")
    return out


def _forward_qsa(self, layer, query, key, value, kv_cache, attn_metadata, output, token_to_req,
                 output_scale=None, output_block_scale=None):
    """Drop-in for Qwen4ExpQSAFlashAttentionImpl.forward_qsa: the stock checks and zeroing, our kernels."""
    del key, value
    if output_scale is not None or output_block_scale is not None:
        raise NotImplementedError("QSA does not support fused output quantization")
    num_tokens = attn_metadata.num_actual_tokens
    output.zero_()
    if num_tokens == 0:
        return output
    topk_buffer = getattr(layer, "topk_indices_buffer", None)
    if topk_buffer is None:
        raise RuntimeError("QSA owner did not provide its top-k buffer")
    from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
    key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
    key_cache = canonicalize_singleton_dim_strides(key_cache)
    value_cache = canonicalize_singleton_dim_strides(value_cache)
    if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:
        raise NotImplementedError("Qwen4Exp QSA requires BF16 Q/K/V")
    sparse_attention(query[:num_tokens], key_cache, value_cache, topk_buffer[:num_tokens], attn_metadata.block_table,
                     token_to_req[:num_tokens], attn_metadata.query_start_loc, attn_metadata.seq_lens,
                     output[:num_tokens])
    return output


def install(model: torch.nn.Module) -> int:
    """Bind our forward_qsa on every QSA impl under `model` (and our indexer scoring, qsa_score.py; independent
    knob R9K_QSA_SCORE). Returns how many attention layers were switched."""
    from . import qsa_score
    qsa_score.install_indexers(model)
    if os.environ.get("R9K_QSA", "r9k") != "r9k" or not available():
        return 0
    n = nf = 0
    fuse = os.environ.get("R9K_FUSED_QKROPE", "1") == "1"
    for mod in model.modules():
        impl = getattr(mod, "impl", None)
        if impl is not None and type(impl).__name__ == "Qwen4ExpQSAFlashAttentionImpl":
            impl.forward_qsa = types.MethodType(_forward_qsa, impl)
            n += 1
            # Stock gates its fused Triton QK-RMSNorm + MRoPE + gate kernel on current_platform.is_cuda() and
            # otherwise runs GemmaRMSNorm x2 + the rotary module eagerly (~30 launches per layer at decode). The
            # kernel is plain Triton and handles this model's interleaved MRoPE; tests/test_fused_qk_rope.py checks
            # it against the eager path. Same preconditions as stock's own flag, minus the CUDA check.
            if fuse and getattr(mod, "attn_output_gate", False) and \
                    getattr(getattr(mod, "rotary_emb", None), "is_neox_style", False) and \
                    hasattr(mod, "use_fused_qk_norm_rope_gate"):
                mod.use_fused_qk_norm_rope_gate = True
                nf += 1
    if n:
        logger.info("r9700: r9k QSA sparse attention installed on %d layers (fused qk-norm/rope on %d)", n, nf)
    return n
