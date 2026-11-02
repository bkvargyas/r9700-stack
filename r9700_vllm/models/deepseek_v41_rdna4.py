"""DeepSeek-V4.1 on RDNA4 (gfx1201): the stock ROCm model copy's architecture gates, opened where the code behind them
is Triton or torch and runs here.

* ``SparseAttnIndexer.forward_hip`` (model_executor/layers/sparse_attn_indexer.py) takes the ROCm indexer op only under
  ``rocm_aiter_ops.is_enabled()`` (gfx9), ``is_rdna_aiter_enabled()`` (RDNA4 with VLLM_ROCM_USE_AITER=1, which would flip
  every other aiter path too), gfx11 or gfx950, and raises otherwise. The op it then calls
  (v1/attention/ops/rocm_aiter_mla_sparse.py rocm_aiter_sparse_attn_indexer) is Triton k-quant/cache-insert, the paged
  logits (our tiled kernel through models/glm5next.py's hook on this GPU), torch fallbacks for the prefill logits and
  the top-k. On RDNA4 we take that op directly.
* Prefill indexer logits: ``rocm_fp8_mqa_logits`` (same module) takes aiter's Triton ``fp8_mqa_logits`` only under the
  same aiter gates and otherwise runs ``fp8_mqa_logits_torch``, which materializes [heads, M, N] scores in bf16 and f32 --
  32 x 2048 x 2048 x 4 B = 512 MiB for a 2k prompt against 2k of context (OOM at ~31 GiB used), 16 GiB for a 4k chunk
  against 32k of context. The image's aiter ships the Triton kernel with gfx1201 configs; on RDNA4 we call it directly
  (clean_logits=False, as stock does) and keep the torch path only when the import fails.
Installed by r9700_vllm.register() next to the exl3 pieces (R9K_DISABLE=dsv41 turns it off); engages for any
DeepSeek-V4.1 checkpoint, EXL3 or fp8.
"""
import os

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


def _install_prefill_logits() -> bool:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as O
    if getattr(O.rocm_fp8_mqa_logits, "__r9k_patch__", None):
        return True
    try:
        from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits as aiter_fp8_mqa_logits
    except Exception:
        from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits as aiter_fp8_mqa_logits
    orig = O.rocm_fp8_mqa_logits

    def rocm_fp8_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
        k_fp8, scale = kv
        return aiter_fp8_mqa_logits(q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=False)

    rocm_fp8_mqa_logits.__r9k_patch__ = "dsv41_rdna4"
    rocm_fp8_mqa_logits.__r9k_orig__ = orig
    O.rocm_fp8_mqa_logits = rocm_fp8_mqa_logits
    return True


def _install_all_prefill() -> bool:
    """Diagnostic (R9K_DSV41_ALL_PREFILL=1): every request is a prefill -- split_decodes_and_prefills reports zero decodes
    to every metadata builder, so the decode-only kernels (SWA ragged decode, decode indexer logits, decode compressor
    and KV insert, decode Engram hashing) are bypassed and single-token steps run the prefill path. Slow; for bisecting
    decode-vs-prefill divergence (notes/exl3-plan.md bring-up log)."""
    from vllm.v1.attention.backends import utils as U
    if getattr(U.split_decodes_and_prefills, "__r9k_patch__", None):
        return True
    orig = U.split_decodes_and_prefills

    def split_decodes_and_prefills(common_attn_metadata, decode_threshold: int = 1, require_uniform: bool = False):
        return orig(common_attn_metadata, decode_threshold=0, require_uniform=require_uniform)

    split_decodes_and_prefills.__r9k_patch__ = "dsv41_all_prefill"
    U.split_decodes_and_prefills = split_decodes_and_prefills
    return True


def _install_debug_decode() -> bool:
    """Diagnostic (R9K_DSV41_DEBUG_DECODE=1, eager): log the decode attention launches of rank 0 -- per call the
    ragged main (SWA) and extra (top-k compressed) segments: lengths, slot ranges, duplicates, and the full slot lists of
    the first launch of each step -- to check what the decode kernel is told to attend."""
    import inspect
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as O
    state = {"calls": 0}

    def wrap(name):
        orig = getattr(O, name)
        if getattr(orig, "__r9k_patch__", None):
            return
        sig = inspect.signature(orig)

        def fn(*args, **kwargs):
            try:
                from vllm.distributed import get_tensor_model_parallel_rank
                rank0 = get_tensor_model_parallel_rank() == 0
            except Exception:
                rank0 = True
            b = sig.bind(*args, **kwargs).arguments
            pos = b.get("inv_rope_positions")
            real = pos is not None and int(pos.max()) >= 10          # skip the profile / warm-up batches
            if rank0 and real and state["calls"] < 90:
                i = state["calls"]; state["calls"] += 1
                parts = [f"[dsv41-decode #{i} {name}] pos={pos.detach().cpu().tolist()[:4]} swa_only={b.get('swa_only')}"]
                for seg in ("swa", "topk"):
                    ri, rp = b.get(f"{seg}_ragged_indices"), b.get(f"{seg}_ragged_indptr")
                    if ri is None or rp is None:
                        continue
                    rp = rp.detach().cpu(); lens = (rp[1:] - rp[:-1]).tolist()
                    n0 = int(lens[0]) if lens else 0
                    sl = ri.detach().cpu()[:min(n0, 140)].tolist()
                    parts.append(f"{seg}: lens={lens[:4]} slots[:{len(sl)}]={sl}")
                tl_ = b.get("topk_lens"); sl_ = b.get("swa_lens")
                if sl_ is not None: parts.append(f"swa_lens={sl_.detach().cpu().tolist()[:4]}")
                if tl_ is not None: parts.append(f"topk_lens={tl_.detach().cpu().tolist()[:4]}")
                kc = b.get("kv_cache"); sk = b.get("swa_k_cache")
                parts.append(f"swa_cache={tuple(sk.shape) if sk is not None else None} extra_cache={tuple(kc.shape) if kc is not None else None}")
                print(" ".join(parts), flush=True)
            return orig(*args, **kwargs)

        fn.__r9k_patch__ = "dsv41_debug_decode"
        setattr(O, name, fn)

    for name in ("rocm_sparse_attn_decode", "rocm_sparse_attn_decode_bf16"):
        if hasattr(O, name):
            wrap(name)
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
    try:
        if _install_prefill_logits():
            done.append("prefill-logits:aiter-triton")
    except Exception as e:  # pragma: no cover
        logger.warning("r9700: dsv41 prefill logits hook not installed (%s)", e)
    if os.environ.get("R9K_DSV41_DEBUG_DECODE", "0") == "1":
        try:
            if _install_debug_decode():
                done.append("DEBUG-DECODE(diagnostic)")
        except Exception as e:  # pragma: no cover
            logger.warning("r9700: dsv41 debug-decode hook not installed (%s)", e)
    if os.environ.get("R9K_DSV41_ALL_PREFILL", "0") == "1":
        try:
            if _install_all_prefill():
                done.append("ALL-PREFILL(diagnostic)")
        except Exception as e:  # pragma: no cover
            logger.warning("r9700: dsv41 all-prefill hook not installed (%s)", e)
    if done:
        logger.info("r9700: deepseek_v41 on RDNA4: %s", ", ".join(done))
    return bool(done)
