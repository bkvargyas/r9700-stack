"""TRITON_ATTN with split-KV (3D) attention for small multi-query batches -- speculative verify.

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
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend, TritonAttentionMetadataBuilder

logger = init_logger("vllm." + __name__)

MAXQ = int(os.environ.get("R9K_ATTN_3D_MAXQ", "16"))


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

    def build(self, common_prefix_len, common_attn_metadata, fast_build: bool = False):
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        num_seqs = md.query_start_loc.shape[0] - 1
        if (1 < md.max_query_len <= MAXQ and md.num_actual_tokens <= self._cap_tokens
                and num_seqs <= self.seq_threshold_3D):
            md.max_query_len = 1          # -> unified_attention's split-KV path (see module doc)
        return md


class R9kTriton3DBackend(TritonAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_builder_cls():
        return R9kTriton3DMetadataBuilder


def register() -> None:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    register_backend(AttentionBackendEnum.CUSTOM, "r9700_vllm.attn.triton3d.R9kTriton3DBackend")
