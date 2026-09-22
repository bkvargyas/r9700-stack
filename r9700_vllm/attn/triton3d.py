"""TRITON_ATTN with split-KV (3D) attention for small multi-query batches (speculative verify) and our paged
attention (kernels/r9k_attn.hip, or libr4d's) for prefill / mixed batches (head_dim 256, GQA 6, block 16; fp8 or
bf16 KV).

Stock `unified_attention` only takes its split-KV ("3D") path when every sequence contributes ONE query token
(`max_seqlen_q > 1` forces the 2D kernel). Speculative decoding verifies 1+k tokens per sequence, so every verify
step ran the 2D kernel, whose grid is ~(num_seqs, kv_heads): 2 programs per layer for single-stream Qwen3.8-27B at
TP2 -- 93 us per attention call, ~2 ms of a 32 ms step. The 3D kernel itself is generic over query tokens (per-token
causal masks, per-token segment reduce); what ties it to one token per sequence is only (a) that condition and (b)
segment scratch sized [seq_threshold_3D, ...], i.e. one row per sequence.

This backend = stock TritonAttentionBackend + a metadata builder that (a) sizes the segment scratch per TOKEN
(seq_threshold_3D * R9K_ATTN_3D_MAXQ rows) and (b) reports max_query_len = 1 for batches whose sequences all have
<= R9K_ATTN_3D_MAXQ (16) query tokens and fit that scratch -- the only consumer of max_query_len in the stock forward is
unified_attention's 2D/3D switch (its other use, a large-head tuning, is gated to CUDA capability 10.x). Kernels are
stock. Registered as AttentionBackendEnum.CUSTOM through vLLM's register_backend; select with
--attention-backend CUSTOM (serve.sh ATTN=CUSTOM / DRAFT_ATTN=CUSTOM).
Upstream candidate: allow 3D for small max_seqlen_q with token-sized scratch.
"""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadataBuilder,
)

logger = init_logger("vllm." + __name__)

MAXQ = int(os.environ.get("R9K_ATTN_3D_MAXQ", "16"))
# Paged attention (head_dim 256, 6 q heads per kv head, block 16, causal, no window) for the batches the 3D path
# does not take -- prefill chunks and mixed prefill+decode steps; stock unified attention measured 36 ms per
# 4096-token chunk on the 27B. R9K_PAGED_ATTN selects the kernels: "r9k" (default, kernels/r9k_attn.hip: our own,
# bf16 + fp8 KV, no decode-band kernel -- short-q groups of a mixed batch go through the same prefill kernel),
# "r4d" (libr4d's prefill + decode pair, if you have that library -- not used by default and not redistributed
# here) or "0" (stock unified attention). R9K_R4D_ATTN=0 is the legacy switch that disables libr4d.
PAGED_ATTN = os.environ.get("R9K_PAGED_ATTN", "r9k")
R4D_ATTN = os.environ.get("R9K_R4D_ATTN", "1") == "1" and PAGED_ATTN == "r4d"


def _r9k_kernels():
    """(kernels per kv dtype, max decode q_len, scratch fn) for our libr9k paged attention, or None."""
    if PAGED_ATTN != "r9k":
        return None
    try:
        from ..kernels.attn import prefill_kernels
        k = prefill_kernels()
    except Exception as e:           # older libr9k.so without the attention kernels: stock path
        logger.warning_once("r9700: libr9k attention unavailable (%s); prefill stays on unified attention", e)
        return None
    return {kv: (pf, None) for kv, pf in k.items()}, 0, None


def _paged_kernels():
    return _r9k_kernels() if PAGED_ATTN == "r9k" else _r4d_kernels()


def _r4d_kernels():
    """(prefill, decode, max decode q_len, scratch-bytes fn) per kv dtype from our libr4d build, or None."""
    if not R4D_ATTN:
        return None
    try:
        from ..comm.r4d_ar import r4d
        m = r4d()
        geo = dict(head_dim=256, gqa=6, block_size=16, causal=1, q_dtype="bf16")
        out = {}
        for kv in ("bf16", "fp8_e4m3"):
            pf = m.select("attn_prefill_paged", kv_dtype=kv, **geo)
            dc = m.select("attn_decode_paged", kv_dtype=kv, q_len=1, **geo)
            if pf and dc:
                out[kv] = (getattr(m, pf), getattr(m, dc))
        if not out:
            return None
        return out, int(m.ATTN_MAX_DECODE_ROWS) // int(m.ATTN_GQA), m.attn_decode_h256_gqa6_scratch_bytes
    except Exception as e:           # no r4d.so, or a build without the attention registry: stock path
        logger.warning_once("r9700: libr4d attention unavailable (%s); prefill stays on unified attention", e)
        return None


def _plan(qsl_cpu, num_reqs):
    """Maximal runs of consecutive requests with equal query length: (first req, count, q_len, first token)."""
    qs = qsl_cpu.tolist()
    groups, i = [], 0
    while i < num_reqs:
        n = qs[i + 1] - qs[i]
        if n == 0:
            i += 1
            continue
        j = i + 1
        while j < num_reqs and qs[j + 1] - qs[j] == n:
            j += 1
        groups.append((i, j - i, n, qs[i]))
        i = j
    return tuple(groups)


class R9kTriton3DMetadataBuilder(TritonAttentionMetadataBuilder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.softmax_segm_output
        self._cap_tokens = self.seq_threshold_3D * MAXQ
        self.softmax_segm_output = torch.empty((self._cap_tokens, *o.shape[1:]), dtype=o.dtype, device=o.device)
        m = self.softmax_segm_max
        self.softmax_segm_max = torch.empty((self._cap_tokens, *m.shape[1:]), dtype=m.dtype, device=m.device)
        self.softmax_segm_expsum = torch.empty_like(self.softmax_segm_max)
        logger.info_once("r9700: attention 3D split-KV for <= %d query tokens/seq (scratch %d tokens)", MAXQ,
                         self._cap_tokens)
        self._r4d = None
        k = _paged_kernels() if (self.headdim == 256 and self.num_heads_q == 6 * self.num_heads_kv) else None
        if k is not None:
            kernels, self._r4d_maxq, scratch_fn = k
            vc = self.vllm_config
            self._max_ctx = vc.model_config.max_model_len
            self._r4d_scratch = None
            if scratch_fn is not None:
                nbytes = max(scratch_fn(n, self._r4d_maxq, self.num_heads_q, self.num_heads_kv, self.headdim,
                                        self._max_ctx, 0) for n in range(1, vc.scheduler_config.max_num_seqs + 1))
                self._r4d_scratch = torch.empty(nbytes, dtype=torch.uint8, device=self.softmax_segm_output.device)
            self._r4d = kernels
            logger.info_once("r9700: %s paged attention for prefill / mixed batches (%s)",
                             "libr9k" if PAGED_ATTN == "r9k" else "libr4d",
                             ", ".join(f"{kv}: {p.__name__}" for kv, (p, _) in kernels.items()))

    def build(self, common_prefix_len, common_attn_metadata, fast_build: bool = False):
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        num_seqs = md.query_start_loc.shape[0] - 1
        if (1 < md.max_query_len <= MAXQ and md.num_actual_tokens <= self._cap_tokens
                and num_seqs <= self.seq_threshold_3D):
            md.max_query_len = 1          # -> unified_attention's split-KV path (see module doc)
        elif self._r4d is not None and md.max_query_len > MAXQ:
            md.r9k_r4d = (self._r4d, self._r4d_maxq, self._r4d_scratch,
                          _plan(common_attn_metadata.query_start_loc_cpu, common_attn_metadata.num_reqs),
                          common_attn_metadata.max_seq_len)
        return md


class R9kTriton3DImpl(TritonAttentionImpl):
    """Stock Triton attention, except prefill / mixed batches of a matching layer go to the paged kernels."""

    _r4d_ok = None

    def _r4d_layer_ok(self, kv_cache) -> bool:
        if self._r4d_ok is None:
            self._r4d_ok = bool(
                self.head_size == 256 and self.num_heads == 6 * self.num_kv_heads
                and tuple(self.sliding_window) == (-1, -1) and self.alibi_slopes is None
                and not self.logits_soft_cap and getattr(self, "sinks", None) is None
                and kv_cache.dim() == 4 and kv_cache.shape[2] == 16 and kv_cache.shape[3] == 2 * self.head_size
                and kv_cache.stride(3) == 1 and kv_cache.stride(2) == kv_cache.shape[3])
        return self._r4d_ok

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
                output_block_scale=None):
        plan = getattr(attn_metadata, "r9k_r4d", None) if attn_metadata is not None else None
        if plan is not None and os.environ.get("R9K_ATTN_DEBUG") == "1":
            logger.warning_once("r9700 attn debug: causal=%r layer_ok=%r kv=%s strides=%s q_contig=%r o_contig=%r "
                                "scale=%r/%r", attn_metadata.causal, self._r4d_layer_ok(kv_cache), tuple(kv_cache.shape),
                                kv_cache.stride(), query.is_contiguous(), output.is_contiguous(),
                                output_scale is None, output_block_scale is None)
        if (plan is None or output_scale is not None or output_block_scale is not None or kv_cache.numel() == 0
                or attn_metadata.causal is not True or not self._r4d_layer_ok(kv_cache)
                or not query.is_contiguous() or not output.is_contiguous()):
            return super().forward(layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                                   output_block_scale)
        kernels, maxq, scratch, groups, max_ctx = plan
        variant = kernels.get("bf16" if kv_cache.element_size() == 2 else "fp8_e4m3")
        # fp8 KV cache (the 27B's compressed-tensors kv scheme selects it): the kernels read one K and one V descale
        # per (sequence, kv head); vLLM keeps a per-tensor scale, broadcast once into small device buffers here.
        fp8kv = kv_cache.element_size() == 1
        kd = vd = 0
        if variant is None:
            return super().forward(layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                                   output_block_scale)
        if fp8kv:
            ks, vs = float(getattr(layer, "_k_scale_float", 1.0)), float(getattr(layer, "_v_scale_float", 1.0))
            if ks != 1.0 or vs != 1.0:
                bufs = self.__dict__.get("_r9k_descales")
                if bufs is None or bufs[2] != (ks, vs):
                    from vllm.config import get_current_vllm_config
                    try:
                        n = get_current_vllm_config().scheduler_config.max_num_seqs * self.num_kv_heads
                    except Exception:
                        n = 1024 * self.num_kv_heads
                    kb = torch.full((n,), ks, dtype=torch.float32, device=query.device)
                    vb = torch.full((n,), vs, dtype=torch.float32, device=query.device)
                    bufs = (kb, vb, (ks, vs))
                    self.__dict__["_r9k_descales"] = bufs
                kd, vd = bufs[0].data_ptr(), bufs[1].data_ptr()
        prefill, decode = variant
        if os.environ.get("R9K_ATTN_DEBUG") == "1":
            logger.warning_once("r9700 attn debug: r4d launch groups=%s max_ctx=%s", groups[:4], max_ctx)
        bt = attn_metadata.block_table
        maxb = bt.shape[1]
        q_row = self.num_heads * self.head_size * query.element_size()
        o_row = self.num_heads * self.head_size * output.element_size()
        stream = torch.cuda.current_stream().cuda_stream
        scratch_ptr = scratch.data_ptr() if scratch is not None else 0
        for first_req, nseq, q_len, first_tok in groups:
            (decode if (decode is not None and q_len <= maxq) else prefill)(
                query.data_ptr() + first_tok * q_row, kv_cache.data_ptr(), bt.data_ptr() + first_req * maxb * 4,
                attn_metadata.seq_lens.data_ptr() + first_req * 4, output.data_ptr() + first_tok * o_row, kd, vd,
                scratch_ptr, nseq, q_len, self.num_heads, self.num_kv_heads, self.head_size, 16, maxb,
                kv_cache.stride(0), kv_cache.stride(1), self.scale, 0, max_ctx, stream)
        return output


class R9kTriton3DBackend(TritonAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_builder_cls():
        return R9kTriton3DMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return R9kTriton3DImpl

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [16]           # the paged kernels are compiled for block 16; the 3D path is fine with it

    @classmethod
    def supported_kv_cache_layouts(cls):
        """The paged kernels index a key by (block, head, slot) with contiguous K/V-packed slots, i.e. LBHNC
        ([L,B,H,N,C] identity). Without this the engine picked a slot-major layout on the 27B (strides
        (16384, 512, 1024, 1)) and every prefill fell back to unified attention. The 3D / stock kernels take any
        strides."""
        from vllm.v1.kv_cache_layout import KVCacheLayout
        return (KVCacheLayout.LBHNC,)


def register() -> None:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    register_backend(AttentionBackendEnum.CUSTOM, "r9700_vllm.attn.triton3d.R9kTriton3DBackend")
