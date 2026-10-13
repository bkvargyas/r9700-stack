"""GLM-5.3-Flash (Glm5Next: KDA linear attention + NoPE sparse MLA + 288-expert MoE) on gfx1201.

Stock vLLM's ROCm model copy (vllm.models.glm5next.amd) is written for CDNA + AITER:
  * the sparse-attention indexer raises unless rocm_aiter_ops.is_enabled(), which is gfx9-only by construction
    (@if_aiter_supported) even though everything it then calls on this model is Triton (indexer logits, kpool
    compress, K-cache insert) or in-tree;
  * its top-k calls torch.ops._C.top_k_per_row_{prefill,decode}, CUDA kernels that the ROCm wheel does not build
    (0 symbols in _C / _rocm_C), so even with the gate open the first prefill fails with a missing op.
The sparse MLA attention itself needs nothing: with a bf16 KV cache and qk_rope_head_dim == 0 the backend picks
its rope-free ragged Triton kernels (rocm_sparse_attn_prefill / _decode_bf16), and the logits fall back to
fp8_mqa_logits_torch (or aiter's Triton fp8_mqa_logits with VLLM_ROCM_USE_AITER=1, which upstream already allows
on RDNA4 through is_rdna_aiter_enabled()).

Two runtime patches, both self-disabling and version-gated (compat/gate.py "glm5next_rdna4"):
  1. the glm5next.amd.sparse_indexer module gets a proxy for rocm_aiter_ops whose is_enabled() is True on RDNA4;
     every other predicate (is_indexer_top_k_enabled: gfx950-only, ...) still goes to the real class, so the aiter
     asm top-k is never chosen;
  2. _C::top_k_per_row_prefill / _C::top_k_per_row_decode are defined through a torch.library FRAGMENT of the
     stock _C library (only when the wheel lacks them) with torch implementations that follow the in-tree
     reference: per row, the top-k columns inside [start, end) by value, -1 past the row length; prefill
     indices are request-relative (start subtracted, as the CUDA kernel's consumers expect: vLLM localizes aiter's
     absolute output for the same reason), decode rows start at 0.
R9K_DISABLE=glm5next leaves stock alone.
"""
from __future__ import annotations

import torch
from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)

_PATCHED = False
_LIBS: list = []     # keep torch.library handles alive


# ------------------------------------------------------------------------------------------ top-k (torch)
def _topk_rows(logits: torch.Tensor, row_starts: torch.Tensor, row_ends: torch.Tensor, out: torch.Tensor,
               topk: int, relative: bool) -> None:
    """out[r, :] = top-`topk` columns of logits[r, start_r:end_r] by value (descending), minus start_r when
    `relative`, -1 where the row has fewer than `topk` valid columns."""
    n = logits.shape[1]
    cols = torch.arange(n, device=logits.device, dtype=torch.int32).unsqueeze(0)
    rs = row_starts.to(torch.int32).view(-1, 1)
    re_ = row_ends.to(torch.int32).view(-1, 1)
    masked = logits.float().masked_fill((cols < rs) | (cols >= re_), float("-inf"))
    k = min(topk, n)
    idx = masked.topk(k, dim=-1).indices.to(torch.int32)
    lens = (re_ - rs).clamp_(min=0)
    pos = torch.arange(k, device=logits.device, dtype=torch.int32).unsqueeze(0)
    if relative:
        idx = idx - rs
    idx = torch.where(pos < lens, idx, torch.full_like(idx, -1))
    out[:, :k].copy_(idx)
    if k < topk:
        out[:, k:topk].fill_(-1)


def _prefill_impl(logits, cu_seqlen_ks, cu_seqlen_ke, raw_topk_indices, num_rows: int, stride0: int,
                  stride1: int, topk_tokens: int) -> None:
    _topk_rows(logits[:num_rows], cu_seqlen_ks[:num_rows], cu_seqlen_ke[:num_rows],
               raw_topk_indices[:num_rows], topk_tokens, relative=True)


def _decode_impl(logits, next_n: int, seq_lens, raw_topk_indices, num_rows: int, stride0: int, stride1: int,
                 topk_tokens: int) -> None:
    # same row-length rule as SparseIndexerTopk._row_ends: (B, next_n) effective lens, or (B, 1) + next_n offsets
    if seq_lens.numel() == num_rows:
        ends = seq_lens.reshape(-1)
    else:
        off = torch.arange(next_n, dtype=torch.int32, device=seq_lens.device)
        ends = (seq_lens.reshape(-1, 1).to(torch.int32) - next_n + 1 + off).clamp_(min=0).reshape(-1)
    starts = torch.zeros_like(ends)
    _topk_rows(logits[:num_rows], starts[:num_rows], ends[:num_rows], raw_topk_indices[:num_rows], topk_tokens,
               relative=False)


def _noop(*a, **k) -> None:     # Meta / fake: in-place ops with no outputs
    return None


def _has_op(name: str) -> bool:
    try:
        getattr(torch.ops._C, name)
        return True
    except (AttributeError, RuntimeError):
        return False


def _register_topk_ops() -> list[str]:
    done = []
    lib = torch.library.Library("_C", "FRAGMENT")
    if not _has_op("top_k_per_row_prefill"):
        lib.define("top_k_per_row_prefill(Tensor logits, Tensor cu_seqlen_ks, Tensor cu_seqlen_ke, "
                   "Tensor(a!) raw_topk_indices, int num_rows, int stride0, int stride1, int topk_tokens) -> ()")
        lib.impl("top_k_per_row_prefill", _prefill_impl, "CUDA")
        lib.impl("top_k_per_row_prefill", _noop, "Meta")
        done.append("top_k_per_row_prefill")
    if not _has_op("top_k_per_row_decode"):
        lib.define("top_k_per_row_decode(Tensor logits, int next_n, Tensor seq_lens, Tensor(a!) raw_topk_indices, "
                   "int num_rows, int stride0, int stride1, int topk_tokens) -> ()")
        lib.impl("top_k_per_row_decode", _decode_impl, "CUDA")
        lib.impl("top_k_per_row_decode", _noop, "Meta")
        done.append("top_k_per_row_decode")
    _LIBS.append(lib)
    return done


# ------------------------------------------------------------------------------------- indexer aiter gate
class _AiterOpsProxy:
    """rocm_aiter_ops as seen by glm5next.amd.sparse_indexer: is_enabled() -> True (the module only uses it as
    'may I run the ROCm path'), everything else from the real class."""

    def __init__(self, real):
        self._real = real

    def is_enabled(self) -> bool:
        return True

    def __getattr__(self, name):
        return getattr(self._real, name)


def patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from vllm.platforms.rocm import on_rdna4
        if not on_rdna4():
            return False
    except Exception:
        return False
    from ..compat import gate
    gate.check("glm5next_rdna4")
    done = []
    try:
        ops = _register_topk_ops()
        if ops:
            done.append("ops:" + ",".join(ops))
    except Exception as e:
        logger.warning("r9700: glm5next top-k ops not installed (%s)", e)
    try:
        from vllm.models.glm5next.amd import sparse_indexer as si
        if not isinstance(si.rocm_aiter_ops, _AiterOpsProxy):
            si.rocm_aiter_ops = _AiterOpsProxy(si.rocm_aiter_ops)
            done.append("indexer:aiter-gate")
    except Exception as e:
        logger.warning("r9700: glm5next indexer gate not installed (%s)", e)
    _PATCHED = True
    logger.info("r9700: glm5next on RDNA4: %s", ", ".join(done) or "nothing to do")
    return bool(done)
