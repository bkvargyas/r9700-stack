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

import os

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


# ------------------------------------------------------------------------------- sparse MLA on libr9k
def _install_sparse_mla() -> bool:
    """Route the sparse-MLA backend's attention through kernels/r9k_sparse_mla.hip. Stock's fp8-pool path is CDNA
    assembly and its Triton path is bf16-only, so an fp8 KV cache has no gfx1201 kernel without this. Taken when the
    KV cache is fp8 (R9K_SPARSE_MLA=stock opts out) or when R9K_SPARSE_MLA=r9k asks for it on a bf16 cache too. Only
    the rope-free, kv_lora_rank 512 geometry (GLM-5.3-Flash); anything else runs the stock method."""
    from ..attn import sparse_mla as SM
    if not SM.available():
        return False
    from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as B
    cls = B.ROCMAiterMLASparseImpl
    orig = cls.forward_mqa
    mode = os.environ.get("R9K_SPARSE_MLA", "auto")

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        fp8_attention = self.kv_cache_dtype.startswith("fp8")
        take = mode == "r9k" or (mode == "auto" and fp8_attention)
        if not take or self.kv_lora_rank != 512 or getattr(self, "qk_rope_head_dim", 0) != 0 or self.sinks is not None:
            return orig(self, q, kv_c_and_k_pe_cache, attn_metadata, layer)
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            if q_pe.shape[-1] != 0:
                return orig(self, q, kv_c_and_k_pe_cache, attn_metadata, layer)
            q = ql_nope                                        # rope-free: the latent query is the whole query
        if q.dtype != torch.bfloat16:
            return orig(self, q, kv_c_and_k_pe_cache, attn_metadata, layer)
        n = attn_metadata.num_actual_tokens
        topk_indices = B.fit_kpool_indices_to_aiter(self.topk_indices_buffer[:n], attn_metadata.topk_tokens)
        kv, block_stride_rows = B.flat_kv_row_view(kv_c_and_k_pe_cache, attn_metadata.block_size)
        B.triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token, attn_metadata.block_table, topk_indices, attn_metadata.paged_kv_indptr,
            attn_metadata.paged_kv_indices, BLOCK_SIZE=attn_metadata.block_size, BLOCK_STRIDE_ROWS=block_stride_rows,
            NUM_TOPK_TOKENS=attn_metadata.topk_tokens)
        if fp8_attention:
            kv = kv.view(torch.float8_e4m3fn)
            desc = getattr(layer, "_r9k_kv_descale", None)
            if desc is None:
                desc = layer._r9k_kv_descale = float(layer._k_scale)      # once: fixed scale, no per-call sync
        else:
            desc = 1.0
        out = torch.empty((q.shape[0], self.num_heads, self.kv_lora_rank), dtype=attn_metadata.attn_out_dtype,
                          device=q.device)
        SM.sparse_mla(q[:n].contiguous() if not q[:n].is_contiguous() else q[:n], kv, attn_metadata.paged_kv_indices,
                      attn_metadata.paged_kv_indptr, self.scale, descale=desc, out=out[:n])
        return out, None

    forward_mqa.__r9k_patch__ = "glm5next_rdna4"
    cls.forward_mqa = forward_mqa

    # With an fp8 cache the MLA layer pre-quantizes the query to fp8 (fused concat+quant) when the impl declares
    # supports_quant_query_input, and the stock fp8 path then runs CDNA assembly (a GPU fault on gfx1201). Our
    # kernel wants the bf16 latent query, so the impl declines quantized queries whenever the hook will take over.
    orig_init = cls.__init__

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        fp8 = str(getattr(self, "kv_cache_dtype", "")).startswith("fp8")
        if mode == "r9k" or (mode == "auto" and fp8):
            self.supports_quant_query_input = False

    __init__.__r9k_patch__ = "glm5next_rdna4"
    cls.__init__ = __init__
    return True


# ----------------------------------------------------------------------------- indexer head-weight mm
def _install_indexer_weights() -> bool:
    """The indexer computes its per-head weights as torch.mm(hidden.float(), W_fp32) with W a [4096, 32] exact upcast
    of a bf16 weight: ~200 us per call on hipBLASLt/rocBLAS at decode, 11 calls a step (4.6 ms of a 24 ms step). Hand
    the cached fp32 weight over as a SkinnyWeight so that one torch.mm runs on r9k_mm_f32_skinny (same fp32 math).
    The model fills the cache lazily on its first forward; this wrapper fills it first, the same way."""
    from ..attn.sparse_mla import SkinnyWeight, available
    if not available():
        return False
    from vllm.models.glm5next.common import attention as A
    cls = A.Indexer
    orig = cls.forward

    def forward(self, hidden_states, qr, positions, rotary_emb):
        if getattr(self, "_wp_fp32", None) is None and hasattr(self, "wk_weights_proj") and hasattr(self, "head_dim"):
            w = self.wk_weights_proj.weight.data[self.head_dim:, :].t().contiguous().float()
            self._wp_fp32 = SkinnyWeight(w) if w.shape[1] <= 32 else w
        return orig(self, hidden_states, qr, positions, rotary_emb)

    forward.__r9k_patch__ = "glm5next_rdna4"
    cls.forward = forward
    return True


# --------------------------------------------------------------------------- KDA projections -> fp8 at load
def _install_fp8_linears_postload() -> bool:
    """R9K_FP8_LINEARS=<regex> for GLM: the model builds its KDA attention with vllm_config.quant_config set to None
    (kda.py) and its MLA/indexer projections with quant_config=None, so those linears get UnquantizedLinearMethod
    directly and never pass through the plugin's quant config (where the regex normally applies, quant/ct.py).
    Swap the method on matching modules right after load_weights; vLLM's process_weights_after_loading pass then
    runs the per-row fp8 requant (quant/ct.py R9kFp8UnquantMethod), weights untouched on disk."""
    import re
    pat = os.environ.get("R9K_FP8_LINEARS", "")
    if not pat:
        return False
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from ..quant.ct import R9kFp8UnquantMethod
    from vllm.models import glm5next as G
    rx = re.compile(pat)
    installed = []
    for cname in ("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM"):
        cls = getattr(G, cname, None)
        if cls is None or getattr(cls.load_weights, "__r9k_patch__", None):
            continue
        orig = cls.load_weights

        def load_weights(self, weights, _orig=orig):
            out = _orig(self, weights)
            n = 0
            for name, mod in self.named_modules():
                qm = getattr(mod, "quant_method", None)
                if type(qm) is UnquantizedLinearMethod and rx.search(name) and getattr(mod, "weight", None) is not None \
                        and mod.weight.dtype == torch.bfloat16 and mod.weight.dim() == 2 \
                        and mod.weight.shape[0] % 16 == 0 and mod.weight.shape[1] % 16 == 0:
                    mod.quant_method = R9kFp8UnquantMethod()
                    n += 1
            logger.info("r9700: R9K_FP8_LINEARS post-load: %d bf16 linears -> per-row fp8 on libr9k (pattern %r)", n, pat)
            return out

        load_weights.__r9k_patch__ = "glm5next_rdna4"
        cls.load_weights = load_weights
        installed.append(cname)
    return bool(installed)


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
    try:
        if os.environ.get("R9K_SPARSE_MLA", "auto") != "stock" and _install_sparse_mla():
            done.append("sparse-mla:libr9k(%s)" % os.environ.get("R9K_SPARSE_MLA", "auto"))
    except Exception as e:
        logger.warning("r9700: glm5next sparse MLA hook not installed (%s)", e)
    try:
        if os.environ.get("R9K_INDEXER_MM", "1") == "1" and _install_indexer_weights():
            done.append("indexer-weights:r9k_mm_f32_skinny")
    except Exception as e:
        logger.warning("r9700: glm5next indexer weights hook not installed (%s)", e)
    try:
        if _install_fp8_linears_postload():
            done.append("fp8-linears:post-load")
    except Exception as e:
        logger.warning("r9700: glm5next fp8-linears post-load hook not installed (%s)", e)
    _PATCHED = True
    logger.info("r9700: glm5next on RDNA4: %s", ", ".join(done) or "nothing to do")
    return bool(done)
