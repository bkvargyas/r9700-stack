"""The QSA indexer's per-head norm + rope glue on one kernel (kernels/r9k_norm_rope.hip).

Stock ``QSAIndexer.project_qk`` / ``normalize_compressed_keys`` run GemmaRMSNorm through ir.ops.rms_norm's native
path (the fp32 (1 + w) weight rules out the fused kernel: ~10 ATen launches) and, when positions are 1-D, the neox
rotary through ApplyRotaryEmb.forward_static (~9 more; flash_attn's Triton rotary is not in the ROCm image). At
decode these ~30 launches per QSA layer sit in the captured graph as ~1.5 us nodes each (2026-09-28 decode
profile: 15 QSA passes per step). Ours: one launch per (norm [+ rope]) with the same math and rounding points
(tests/test_indexer_glue_r9k.py checks bit-equality against the stock functions). 2-D (MRoPE) positions keep
stock's triton_mrope after our norm. R9K_QSA_GLUE=stock keeps vLLM's path.
"""
from __future__ import annotations

import ctypes
import os
import types

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger("vllm." + __name__)

_L = None
_DONE = False


def lib():
    global _L
    if _L is None:
        from ..kernels import moe as KM
        L = KM.lib()
        L.r9k_gemma_norm_rope.restype = ctypes.c_int
        L.r9k_gemma_norm_rope.argtypes = [ctypes.c_long] * 7 + [ctypes.c_int] * 3 + [ctypes.c_float] + \
            [ctypes.c_long] * 2 + [ctypes.c_int] * 2 + [ctypes.c_long] * 2 + [ctypes.c_int] * 4 + [ctypes.c_long]
        _L = L
    return _L


def available() -> bool:
    try:
        return hasattr(lib(), "r9k_gemma_norm_rope")
    except Exception:
        return False


def _run(x: torch.Tensor, w: torch.Tensor, eps: float, cs: torch.Tensor | None, pos: torch.Tensor | None,
         rotary_dim: int, out: torch.Tensor, section: tuple[int, int, int] = (0, 0, 0), interleaved: bool = False
         ) -> None:
    """x, out: [T, H, D] (x may be a strided view of a wider row; out contiguous). pos: int64 [T] or [3, T], any
    strides (serving passes the indexer a [T] column view of a [T, 3] buffer)."""
    T, H, D = x.shape
    assert x.stride(2) == 1 and out.is_contiguous() and w.is_contiguous()
    if pos is None:
        prow, pstride, ptok = 1, 0, 1
    elif pos.dim() == 1:
        prow, pstride, ptok = 1, 0, pos.stride(0)
    else:
        prow, pstride, ptok = pos.shape[0], pos.stride(0), pos.stride(1)
    rc = lib().r9k_gemma_norm_rope(x.data_ptr(), x.stride(0), x.stride(1), w.data_ptr(), out.data_ptr(), out.stride(0),
                                   out.stride(1), T * H, H, D, float(eps),
                                   cs.data_ptr() if cs is not None else 0, pos.data_ptr() if pos is not None else 0,
                                   rotary_dim, prow, pstride, ptok, int(section[0]), int(section[1]), int(section[2]),
                                   1 if interleaved else 0, torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_gemma_norm_rope failed ({rc}) T={T} H={H} D={D} R2={rotary_dim} pos={tuple(pos.shape) if pos is not None else None}")


def gemma_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """GemmaRMSNorm over the last dim of x [T, H, D] bf16 -> [T, H, D] bf16 (the norm's native numerics)."""
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _run(x, w, eps, None, None, 0, out)
    return out


def gemma_norm_rope(x: torch.Tensor, w: torch.Tensor, eps: float, cos_sin: torch.Tensor, positions: torch.Tensor,
                    rotary_dim: int, sec_t: int, sec_h: int, sec_w: int, interleaved: bool) -> torch.Tensor:
    """gemma_norm then neox rotary on the first `rotary_dim` of every head with cos_sin[positions[t]]; positions
    [T] (plain) or [3, T] (MRoPE with the (sec_t, sec_h, sec_w) section rule, interleaved or concatenated)."""
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _run(x, w, eps, cos_sin, positions, rotary_dim, out, (sec_t, sec_h, sec_w), interleaved)
    return out


def _fits(x: torch.Tensor, w: torch.Tensor) -> bool:
    return (x.dim() == 3 and x.dtype == torch.bfloat16 and x.stride(2) == 1 and x.stride(1) % 8 == 0
            and x.stride(0) % 8 == 0 and 8 <= x.shape[2] <= 256 and x.shape[2] % 8 == 0
            and w.dtype == torch.bfloat16 and w.numel() == x.shape[2] and w.is_contiguous())


def _rope_fits(x: torch.Tensor, cs: torch.Tensor, pos: torch.Tensor, rotary_dim: int) -> bool:
    return (cs.dtype == torch.bfloat16 and cs.dim() == 2 and cs.shape[1] == rotary_dim and cs.is_contiguous()
            and pos.dtype == torch.int64 and pos.shape[-1] == x.shape[0]
            and (pos.dim() == 1 or (pos.dim() == 2 and pos.shape[0] == 3))
            and 16 <= rotary_dim <= x.shape[2] and rotary_dim % 16 == 0)


def _norm_fake(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x.new_empty(x.shape)


def _norm_rope_fake(x: torch.Tensor, w: torch.Tensor, eps: float, cos_sin: torch.Tensor, positions: torch.Tensor,
                    rotary_dim: int, sec_t: int, sec_h: int, sec_w: int, interleaved: bool) -> torch.Tensor:
    return x.new_empty(x.shape)


def register() -> None:
    global _DONE
    if _DONE:
        return
    from ..ops import _LIB
    direct_register_custom_op("qsa_gemma_norm", gemma_norm, mutates_args=[], fake_impl=_norm_fake, target_lib=_LIB)
    direct_register_custom_op("qsa_gemma_norm_rope", gemma_norm_rope, mutates_args=[], fake_impl=_norm_rope_fake,
                              target_lib=_LIB)
    _DONE = True


def _norm_rope_or_stock(self, x: torch.Tensor, norm, positions: torch.Tensor) -> torch.Tensor:
    """x [T, H, D] bf16 -> norm then rope on our kernel (1-D positions, or [3, T] MRoPE with the module's section
    rule); stock's apply_qsa_rope after our norm when the rope does not fit, stock throughout when the norm does not."""
    from vllm.models.qwen4_exp.amd.indexer_qsa import apply_qsa_rope
    re = self.rotary_emb
    w, eps = norm.weight, norm.variance_epsilon
    if not _fits(x, w):
        return apply_qsa_rope(re, positions, norm(x.reshape(-1, x.shape[2])).reshape_as(x))
    if getattr(re, "is_neox_style", False):
        cs = re._match_cos_sin_cache_dtype(x)          # noqa: SLF001 (stock does the same)
        rd = int(re.rotary_dim)
        sec = getattr(re, "mrope_section", None)
        if positions.ndim == 1:
            if _rope_fits(x, cs, positions, rd):
                return torch.ops.r9700.qsa_gemma_norm_rope(x, w, float(eps), cs, positions, rd, 0, 0, 0, False)
        elif sec is not None and len(sec) == 3 and _rope_fits(x, cs, positions, rd):
            return torch.ops.r9700.qsa_gemma_norm_rope(x, w, float(eps), cs, positions, rd, int(sec[0]), int(sec[1]),
                                                       int(sec[2]), bool(getattr(re, "mrope_interleaved", False)))
        logger.warning_once("r9700: indexer rope not fused (stock rope after our norm): positions %s %s stride %s, "
                            "cache %s %s contiguous=%s, rotary_dim %d, x %s, mrope_section %s", tuple(positions.shape),
                            positions.dtype, positions.stride(), tuple(cs.shape), cs.dtype, cs.is_contiguous(), rd,
                            tuple(x.shape), tuple(sec) if sec is not None else None)
    else:
        logger.warning_once("r9700: indexer rope not fused: rotary_emb %s is not neox-style", type(re).__name__)
    return apply_qsa_rope(re, positions, torch.ops.r9700.qsa_gemma_norm(x, w, float(eps)))


def _project_qk(self, hidden_states: torch.Tensor, positions: torch.Tensor):
    qk, _ = self.index_qk_proj(hidden_states)
    T = qk.shape[0]
    nq, nk, D = self.index_n_heads, self.index_kv_heads, self.index_head_dim
    q_raw = qk[:, : nq * D].view(T, nq, D)               # strided view of the projection row: no copy
    token_k = qk[:, nq * D:]
    q = _norm_rope_or_stock(self, q_raw, self.q_layernorm, positions)
    return q, token_k.reshape(-1, 1, D)


def _normalize_compressed_keys(self, compressed_keys: torch.Tensor, first_rope_positions: torch.Tensor):
    D = self.index_head_dim
    keys = compressed_keys.reshape(-1, 1, D)
    if getattr(self.rotary_emb, "mrope_section", None):
        positions = first_rope_positions.transpose(0, 1)
    else:
        positions = first_rope_positions[:, 0]
    return _norm_rope_or_stock(self, keys, self.k_layernorm, positions)


def install(model: torch.nn.Module) -> int:
    """Bind our project_qk / normalize_compressed_keys on every QSAIndexer under `model`. Returns the count."""
    if os.environ.get("R9K_QSA_GLUE", "r9k") != "r9k" or not available():
        return 0
    register()
    n = 0
    for mod in model.modules():
        if type(mod).__name__ != "QSAIndexer" or getattr(mod, "_r9k_glue", False):
            continue
        if not all(hasattr(mod, a) for a in ("index_qk_proj", "q_layernorm", "k_layernorm", "rotary_emb",
                                             "index_n_heads", "index_kv_heads", "index_head_dim")):
            continue
        mod.project_qk = types.MethodType(_project_qk, mod)
        mod.normalize_compressed_keys = types.MethodType(_normalize_compressed_keys, mod)
        mod._r9k_glue = True
        n += 1
    if n:
        logger.info("r9700: r9k QSA indexer norm/rope glue installed on %d indexers", n)
    return n
