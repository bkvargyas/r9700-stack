"""DeepSeek-V4.1 on RDNA4 (gfx1201): the stock ROCm model copy's architecture gates, opened where the code behind them
is Triton or torch and runs here.

* ``SparseAttnIndexer.forward_hip`` (model_executor/layers/sparse_attn_indexer.py) takes the ROCm indexer op only under
  ``rocm_aiter_ops.is_enabled()`` (gfx9), ``is_rdna_aiter_enabled()`` (RDNA4 with VLLM_ROCM_USE_AITER=1, which would flip
  every other aiter path too), gfx11 or gfx950, and raises otherwise. The op it then calls
  (v1/attention/ops/rocm_aiter_mla_sparse.py rocm_aiter_sparse_attn_indexer) is Triton k-quant/cache-insert, the paged
  logits (our tiled kernel through models/glm5next.py's hook on this GPU), torch fallbacks for the prefill logits and
  the top-k. On RDNA4 we take that op directly.
* Decode attention: stock's launcher sends every GPU but gfx942/gfx950 to a serial one-workgroup-per-query fallback
  (~180 us a layer at SWA 128 + top-k 512 on an R9700) that also never applies the fused inverse RoPE it reports, so
  decode rows reached wo_a still rotated. R9K_DSV41_DECODE selects: ``splitk`` (default) -- attn/dsv4_decode.py runs
  stock's split-K partial + reduce kernels with RDNA4 tuning and the reduce's fused inverse RoPE, reporting rotated rows
  truthfully (_install_splitk_decode); ``ragged`` -- stock's fallback kernel with no inv_rope_positions, so the caller's
  standalone pass un-rotates every decode row (_install_no_fused_invrope, the 2026-11-02 workaround); ``stock`` -- untouched
  (wrong on RDNA4; R9K_DSV41_FUSED_INVROPE=1 is the old spelling).
* Prefill indexer logits: ``rocm_fp8_mqa_logits`` (same module) takes aiter's Triton ``fp8_mqa_logits`` only under the
  same aiter gates and otherwise runs ``fp8_mqa_logits_torch``, which materializes [heads, M, N] scores in bf16 and f32 --
  32 x 2048 x 2048 x 4 B = 512 MiB for a 2k prompt against 2k of context (OOM at ~31 GiB used), 16 GiB for a 4k chunk
  against 32k of context. The image's aiter ships the Triton kernel with gfx1201 configs; on RDNA4 we call it directly
  (clean_logits=False, as stock does) and keep the torch path only when the import fails.
* Hyper-connections: stock's MHCPreDelayedOp / MHCPostOp run TileLang on RDNA4, three launches a seam (27 us at M=1);
  R9K_DSV41_MHC=r9k (default when libr9k has the kernels) binds hc_dsv4.py's one-launch seam for calls of <= 32 rows
  (_install_mhc); ``stock`` keeps TileLang.
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
        if getattr(orig, "__r9k_patch__", None) == "dsv41_debug_decode":
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


def _install_no_fused_invrope() -> bool:
    """The decode-path fix (default on RDNA4; R9K_DSV41_FUSED_INVROPE=1 opts back into stock behaviour).

    Stock's `rocm_sparse_attn_decode` (v1/attention/ops/rocm_aiter_mla_sparse.py) takes `inv_rope_positions` so that the
    split-K reduce kernel can fold the inverse RoPE of the attention output into its epilogue (FUSE_INV_ROPE), and
    reports to the caller how many rows came back rotated; the caller (deepseek_v41/amd/rocm.py forward_mqa) runs the
    standalone inverse-RoPE pass only on the rest. On GPUs that are neither gfx942 nor gfx950 the launcher takes its
    "fallback path for un-tuned architectures" -- the single-pass ragged kernel, which has no such epilogue -- yet the
    wrapper still reports the rows as rotated, so the decode attention output keeps the RoPE rotation on its 64 rope
    dims per head and wo_a sees it. Prefill rows are unaffected. The effect is a per-token error at decode only:
    coherent text that doubles words and loops within a few dozen tokens (found with the EXL3 pack, 2026-11-02;
    divergence.py decode-vs-prefill top-1 agreement 14-20/24 -> 23-24/24 with this fix). Passing no inv_rope_positions
    makes the wrapper report 0 rotated rows and the caller un-rotates every decode row itself."""
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as O
    orig = O.rocm_sparse_attn_decode
    if getattr(orig, "__r9k_patch__", None) == "dsv41_no_fused_invrope":
        return True

    def rocm_sparse_attn_decode(*args, **kwargs):
        kwargs["inv_rope_positions"] = None
        kwargs["inv_rope_cos_sin_cache"] = None
        if kwargs.get("output_mxfp8") is not None:
            raise RuntimeError("R9K_DSV41_NO_FUSED_INVROPE cannot drop the fused inverse RoPE under output_mxfp8")
        return orig(*args, **kwargs)

    rocm_sparse_attn_decode.__r9k_patch__ = "dsv41_no_fused_invrope"
    O.rocm_sparse_attn_decode = rocm_sparse_attn_decode
    return True


def _install_splitk_decode() -> bool:
    """Route `rocm_sparse_attn_decode` (fp8_ds_mla, the DeepSeek-V4.1 path) and `rocm_sparse_attn_decode_bf16` to
    attn/dsv4_decode.py's split-K launcher. Installed over the stock module's names before vLLM imports the model's
    ROCm module; modules that already imported the names by value are re-pointed too."""
    import sys
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as O
    from r9700_vllm.attn import dsv4_decode as D
    for name in ("rocm_sparse_attn_decode", "rocm_sparse_attn_decode_bf16"):
        orig = getattr(O, name)
        if getattr(orig, "__r9k_patch__", None) == "dsv41_splitk_decode":
            continue
        fn = getattr(D, name)

        def wrapped(*args, __fn=fn, **kwargs):
            return __fn(*args, **kwargs)

        wrapped.__r9k_patch__ = "dsv41_splitk_decode"
        wrapped.__r9k_orig__ = orig
        wrapped.__wrapped__ = orig
        setattr(O, name, wrapped)
        for mod in ("vllm.models.deepseek_v41.amd.rocm", "vllm.models.deepseek_v4.amd.rocm",
                    "vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse"):
            m = sys.modules.get(mod)
            if m is not None and getattr(m, name, None) is orig:
                setattr(m, name, wrapped)
    return True


def _install_mhc() -> bool:
    """Hyper-connection seams on our kernels (hc_dsv4.py, kernels/r9k_mhc.hip): MHCPreDelayedOp.forward_hip (the post
    folded into the delayed pre: one launch a seam instead of stock's three TileLang launches) and MHCPostOp.forward_hip,
    for calls of up to 32 rows; wider calls run stock's ops inside the same custom ops. Installed on the classes before
    the model is built (CustomOp binds forward_hip at construction). R9K_DSV41_MHC=stock keeps stock's."""
    from vllm.model_executor.layers import mhc as M
    from r9700_vllm import hc_dsv4 as D
    if getattr(M.MHCPreDelayedOp.forward_hip, "__r9k_patch__", None) == "dsv41_mhc":
        return True
    if not D.available():
        return False
    D.register()
    for cls, fn in ((M.MHCPreDelayedOp, D.pre_delayed_forward_hip), (M.MHCPostOp, D.post_forward_hip)):
        def wrapped(self, *args, __fn=fn, **kwargs):
            return __fn(self, *args, **kwargs)
        wrapped.__r9k_patch__ = "dsv41_mhc"
        wrapped.__r9k_orig__ = cls.forward_hip
        cls.forward_hip = wrapped
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
    mode = os.environ.get("R9K_DSV41_DECODE", "stock" if os.environ.get("R9K_DSV41_FUSED_INVROPE") == "1" else "splitk")
    if mode == "splitk":
        try:
            if _install_splitk_decode():
                done.append("decode:split-k+fused-inverse-rope")
        except Exception as e:  # pragma: no cover
            logger.warning("r9700: dsv41 split-K decode not installed (%s); falling back to ragged", e)
            mode = "ragged"
    if mode == "ragged":
        try:
            if _install_no_fused_invrope():
                done.append("decode:ragged+inverse-rope-unfused")
        except Exception as e:  # pragma: no cover
            logger.warning("r9700: dsv41 no-fused-invrope hook not installed (%s)", e)
    if os.environ.get("R9K_DSV41_MHC", "r9k") == "r9k":
        try:
            if _install_mhc():
                done.append("mhc:r9k-seam(M<=32)")
        except Exception as e:  # pragma: no cover
            logger.warning("r9700: dsv41 r9k mhc kernels not installed (%s); stock TileLang", e)
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
