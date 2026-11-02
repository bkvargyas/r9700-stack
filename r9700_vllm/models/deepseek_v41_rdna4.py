"""DeepSeek-V4.1 on RDNA4 (gfx1201): the stock ROCm model copy's architecture gates, opened where the code behind them
is Triton or torch and runs here.

* ``SparseAttnIndexer.forward_hip`` (model_executor/layers/sparse_attn_indexer.py) takes the ROCm indexer op only under
  ``rocm_aiter_ops.is_enabled()`` (gfx9), ``is_rdna_aiter_enabled()`` (RDNA4 with VLLM_ROCM_USE_AITER=1, which would flip
  every other aiter path too), gfx11 or gfx950, and raises otherwise. The op it then calls
  (v1/attention/ops/rocm_aiter_mla_sparse.py rocm_aiter_sparse_attn_indexer) is Triton k-quant/cache-insert, the paged
  logits (our tiled kernel through models/glm5next.py's hook on this GPU), torch fallbacks for the prefill logits and
  the top-k. On RDNA4 we take that op directly.
Installed by r9700_vllm.register() next to the exl3 pieces (R9K_DISABLE=dsv41 turns it off); engages for any
DeepSeek-V4.1 checkpoint, EXL3 or fp8.
"""
import torch
from vllm.logger import init_logger

logger = init_logger("vllm.r9700_vllm.deepseek_v41_rdna4")


def _install_indexer_gate() -> bool:
    from vllm.model_executor.layers import sparse_attn_indexer as S
    cls = S.SparseAttnIndexer
    if getattr(cls.forward_hip, "__r9k_patch__", None):
        return True

    def forward_hip(self, hidden_states, q_quant, k, weights):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
            hidden_states, S._encode_layer_name(self.k_cache.prefix), self.k_cache.kv_cache, q_quant, k, weights,
            self.quant_block_size, self.scale_fmt, self.topk_tokens, self.head_dim, self.max_model_len,
            self.max_total_seq_len, self.topk_indices_buffer, skip_k_cache_insert=self.skip_k_cache_insert,
            compress_ratio=self.compress_ratio, candidate_blocks=self.candidate_blocks,
            candidate_block_size=self.candidate_block_size, candidate_write=self.candidate_write)

    forward_hip.__r9k_patch__ = "dsv41_rdna4"
    cls.forward_hip = forward_hip
    return True


def patch() -> bool:
    try:
        from vllm.platforms.rocm import on_rdna4
        if not on_rdna4():
            return False
    except Exception:
        return False
    done = []
    try:
        if _install_indexer_gate():
            done.append("indexer:rdna4-gate")
    except Exception as e:  # pragma: no cover
        logger.warning("r9700: dsv41 indexer gate not installed (%s)", e)
    if done:
        logger.info("r9700: deepseek_v41 on RDNA4: %s", ", ".join(done))
    return bool(done)
