"""Gated-DeltaNet input projections as ONE GEMM (both Qwen3.8-27B and Flash-Next use stock QwenGatedDeltaNetAttention).

`in_proj_qkvz` and `in_proj_ba` are two linears over the same hidden state, run back to back in every GDN layer
(48 per forward on the 27B): two activation quantizations + two GEMM launches where one would do. GGZ14 merges
them the same way (radiance_gdnmerge.py). Here: an out-of-tree PluggableLayer subclass (vLLM's documented override
for `qwen_gated_delta_net_attention`) that, once both projections are loaded onto libr9k kernels of the SAME format,
concatenates their weights along N -- MXFP4 (fragment-order tiles + per-row packed scales) or per-row fp8
(fragment-order tiles + row scales), both row-local, so the merged GEMM is bit-identical per row -- and serves:

    in_proj_qkvz(x) -> one GEMM over [qkvz; ba] -> returns the qkvz columns, stashes the ba columns
    in_proj_ba(x)   -> returns the stash      (stock forward calls these two back to back on the same x)

The merge is triggered from in_proj_ba's process_weights_after_loading (stock's loader visits in_proj_qkvz first).
Layers whose formats differ (e.g. Flash-Next's block-fp8 qkvz) stay unmerged. R9K_GDN_MERGE=0 disables.

MTP decode core (kernels/r9k_gdn.hip r9k_gdn_decode_mtp): stock's fused CUDA op for the speculative-decode step
(fused_gdn_decode_post_conv_mtp) is not built on ROCm, so vLLM runs the Triton recurrence inside ~9 glue launches
per layer (b/a contiguous, zeros, q/k/v cat, output copy, gated norm). Ours: the layer's forward becomes
in_proj -> torch.ops.r9700.gdn_core (conv update + one fused gating/recurrence/norm launch when the batch is pure
spec decode, stock's core + norm otherwise) -> out_proj. R9K_GDN_DECODE=stock keeps vLLM's forward.

Short prefills (kernels/r9k_gdn.hip r9k_gdn_seq): a prefill step runs this op eagerly between the pieces of a
piecewise graph, and stock's prefill core is the chunked delta rule -- six Triton kernels plus prep, state
gather/scatter and the norm, ~30 launches and ~1.3 ms of Python per layer whatever the prompt length. On the 27B
(48 GDN layers) that is ~60 of the ~97 ms a 45-token prompt waits for its first token, and the same stall for
every running request when a new one joins the batch. For steps with at most R9K_GDN_PREFILL_MAX prefill tokens
(default 256, serve/27b.sh sets 400; 0 = stock) the core is the token-by-token recurrence in one launch -- the
same recurrence decode uses, so a short prompt goes through the arithmetic its later tokens will -- after the
causal conv in one launch (r9k_gdn_conv, bit-identical to stock's Triton kernel; R9K_GDN_PREFILL_CONV=stock keeps
the Triton one). Longer steps keep the chunked form: token by token costs the GPU ~1.8 us per token and layer,
more than the chunked kernels, so past ~240 tokens (Flash-Next TP4) to ~440 (the 27B) the step is GPU-bound and
the chunked form wins again.
"""
from __future__ import annotations

import ctypes
import os

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.utils.torch_utils import (LayerNameType, _encode_layer_name, _resolve_layer_name,
                                    direct_register_custom_op)

logger = init_logger("vllm." + __name__)

MAX_SPEC_TOKENS = 8            # per sequence (num_spec + 1); stock's MAX_FUSED_GDN_MTP_TOKENS
DEBUG = os.environ.get("R9K_GDN_DEBUG", "0") == "1"   # per-stage syncs + host range checks (slow; diagnosis only)
FUSED = os.environ.get("R9K_GDN_FUSED", "1") == "1"   # 0: always stock's core + norm inside the op (diagnosis)
PREFILL_MAX = int(os.environ.get("R9K_GDN_PREFILL_MAX", "256") or 0)   # prefill tokens per step; 0 = stock's chunked core
CONV = os.environ.get("R9K_GDN_PREFILL_CONV", "r9k") != "stock"         # the conv in front of it: ours, or stock's Triton
ONEPAGE = os.environ.get("R9K_GDN_STATE", "onepage") != "stock"
PAGE_PAD = os.environ.get("R9K_GDN_PAGE_PAD", "0") == "1"
TRACE = os.environ.get("R9K_GDN_TRACE", "0") == "1"   # diagnosis only: per-step checksums of layer 0 (eager steps)   # diagnosis only: size the mamba page as one-page
                                                            # would (bigger attention block) while serving stock pages         # one state page per request under spec decoding
_L = None
_OPS_DONE = False


def lib():
    global _L
    if _L is None:
        from ..kernels import moe as KM
        L = KM.lib()
        L.r9k_gdn_decode_mtp.restype = ctypes.c_int
        L.r9k_gdn_decode_mtp.argtypes = [ctypes.c_long] * 15 + [ctypes.c_int] + [ctypes.c_long] * 2 + [ctypes.c_int] + \
            [ctypes.c_long] * 2 + [ctypes.c_int] * 4 + [ctypes.c_float] * 2 + [ctypes.c_int] + [ctypes.c_long]
        if hasattr(L, "r9k_gdn_seq"):
            L.r9k_gdn_seq.restype = ctypes.c_int
            L.r9k_gdn_seq.argtypes = [ctypes.c_long] * 15 + [ctypes.c_int] + [ctypes.c_long] * 8 + \
                [ctypes.c_int] * 5 + [ctypes.c_float] * 2 + [ctypes.c_int] * 2 + [ctypes.c_long]
        if hasattr(L, "r9k_gdn_conv"):
            L.r9k_gdn_conv.restype = ctypes.c_int
            L.r9k_gdn_conv.argtypes = [ctypes.c_long] * 15 + [ctypes.c_int] * 2 + [ctypes.c_long]
        if hasattr(L, "r9k_gdn_spec_verify"):
            L.r9k_gdn_spec_verify.restype = ctypes.c_int
            L.r9k_gdn_spec_verify.argtypes = [ctypes.c_long] * 15 + [ctypes.c_int] + [ctypes.c_long] * 6 + \
                [ctypes.c_int] + [ctypes.c_long] * 5 + [ctypes.c_int] * 5 + [ctypes.c_float] * 2 + [ctypes.c_int] + \
                [ctypes.c_long]
            L.r9k_gdn_spec_commit.restype = ctypes.c_int
            L.r9k_gdn_spec_commit.argtypes = [ctypes.c_long] * 2 + [ctypes.c_int] + [ctypes.c_long] * 5 + \
                [ctypes.c_int] * 3 + [ctypes.c_long] * 3 + [ctypes.c_int] * 4 + [ctypes.c_long]
        _L = L
    return _L


def available() -> bool:
    try:
        return hasattr(lib(), "r9k_gdn_decode_mtp")
    except Exception:
        return False


def decode_mtp(mixed_qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor,
               w: torch.Tensor, z: torch.Tensor, out: torch.Tensor, state: torch.Tensor, idx: torch.Tensor,
               cu: torch.Tensor, acc: torch.Tensor, Hk: int, Hv: int, scale: float, eps: float,
               gate_sigmoid: bool = False) -> None:
    """Fused gating + delta-rule recurrence + RMSNormGated for N spec-decode sequences.
    mixed_qkv [rows, (2 Hk + Hv) 128] bf16 after the conv (any row stride); b, a [rows, Hv] bf16 views; A_log,
    dt_bias [Hv] fp32; w [128] fp32; z [rows, Hv * 128] bf16 view; out [rows, Hv * 128] bf16 (rows past cu[N] get
    zeros); state [blocks, Hv, 128, 128] fp32 or bf16, updated in place at idx[n, t]; idx [N, S] int32; cu [N + 1]
    int32; acc [N] int32. gate_sigmoid: the norm's gate is sigmoid(z) instead of silu(z)."""
    N = idx.shape[0]
    assert mixed_qkv.dtype == torch.bfloat16 and mixed_qkv.stride(1) == 1 and z.stride(1) == 1 and out.stride(1) == 1
    assert state.dtype in (torch.float32, torch.bfloat16) and state.is_contiguous() or state.stride(-1) == 1
    assert idx.dtype == torch.int32 and cu.dtype == torch.int32 and acc.dtype == torch.int32 and idx.stride(1) == 1
    assert cu.is_contiguous() and acc.is_contiguous() and A_log.dtype == torch.float32 and w.dtype == torch.float32
    rc = lib().r9k_gdn_decode_mtp(mixed_qkv.data_ptr(), mixed_qkv.stride(0), b.data_ptr(), b.stride(0), a.data_ptr(),
                                  a.stride(0), A_log.data_ptr(), dt_bias.data_ptr(), w.data_ptr(), z.data_ptr(),
                                  z.stride(0), out.data_ptr(), out.stride(0), state.data_ptr(), state.stride(0),
                                  1 if state.dtype == torch.float32 else 0, idx.data_ptr(), idx.stride(0),
                                  idx.shape[1], cu.data_ptr(), acc.data_ptr(), N, Hk, Hv, out.shape[0], float(scale),
                                  float(eps), 1 if gate_sigmoid else 0, torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_gdn_decode_mtp failed ({rc}) N={N} S={idx.shape[1]} rows={out.shape[0]} "
                           f"Hk={Hk} Hv={Hv}")


def seq_available() -> bool:
    try:
        return hasattr(lib(), "r9k_gdn_seq")
    except Exception:
        return False


def seq_core(mixed_qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor,
             w: torch.Tensor, z: torch.Tensor, out: torch.Tensor, state: torch.Tensor, idx: torch.Tensor,
             cu: torch.Tensor, Hk: int, Hv: int, scale: float, eps: float, *, acc: torch.Tensor | None = None,
             hasinit: torch.Tensor | None = None, rowmap: torch.Tensor | None = None, zero_from: int | None = None,
             gate_sigmoid: bool = False, flags: torch.Tensor | None = None) -> None:
    """The recurrence + gated norm for N sequences of any length, one launch (kernels/r9k_gdn.hip r9k_gdn_seq).
    acc None -> prefill sequences: idx [N] int32 (any stride) is each sequence's state slot, read when hasinit[n]
    (bool [N]; None = always) and written with the final state. acc [N] -> spec-decode sequences, as decode_mtp
    (idx [N, S]). mixed_qkv rows are the sequences back to back (cu [N + 1] int32); b, a, z, out are indexed by
    rowmap[row] (int64 [cu[N]]) -- the batch's own row order -- or by the same row when rowmap is None. Rows
    zero_from.. of out are zeroed (None: nothing). flags [pages, 2] int32: prefill sequences zero their page's
    one-page record flag."""
    N = cu.shape[0] - 1
    spec = acc is not None
    assert mixed_qkv.dtype == torch.bfloat16 and mixed_qkv.stride(1) == 1 and z.stride(1) == 1 and out.stride(1) == 1
    assert state.dtype in (torch.float32, torch.bfloat16) and state.stride(-1) == 1
    assert idx.dtype == torch.int32 and cu.dtype == torch.int32 and cu.is_contiguous() and idx.shape[0] >= N
    assert A_log.dtype == torch.float32 and w.dtype == torch.float32
    if spec:
        assert acc.dtype == torch.int32 and acc.is_contiguous() and idx.dim() == 2 and idx.stride(1) == 1
    else:
        assert idx.dim() == 1 and (hasinit is None or (hasinit.dtype == torch.bool and hasinit.is_contiguous()
                                                       and hasinit.shape[0] >= N))
    if rowmap is not None:
        rowmap = rowmap.to(torch.int64).contiguous()
    rows = out.shape[0]
    rc = lib().r9k_gdn_seq(mixed_qkv.data_ptr(), mixed_qkv.stride(0), b.data_ptr(), b.stride(0), a.data_ptr(),
                           a.stride(0), A_log.data_ptr(), dt_bias.data_ptr(), w.data_ptr(), z.data_ptr(), z.stride(0),
                           out.data_ptr(), out.stride(0), state.data_ptr(), state.stride(0),
                           1 if state.dtype == torch.float32 else 0, idx.data_ptr(), idx.stride(0), cu.data_ptr(),
                           acc.data_ptr() if spec else 0, hasinit.data_ptr() if hasinit is not None else 0,
                           rowmap.data_ptr() if rowmap is not None else 0,
                           flags.data_ptr() if flags is not None else 0, flags.stride(0) if flags is not None else 0,
                           N, Hk, Hv, rows if zero_from is None else int(zero_from), rows, float(scale), float(eps),
                           1 if gate_sigmoid else 0, 1 if spec else 0, torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_gdn_seq failed ({rc}) N={N} spec={spec} rows={rows} Hk={Hk} Hv={Hv}")


def conv_prefill(x: torch.Tensor, C: int, L: int, weight: torch.Tensor, conv_state: torch.Tensor, idx: torch.Tensor,
                 cu: torch.Tensor, hasinit: torch.Tensor | None = None, rowmap: torch.Tensor | None = None
                 ) -> torch.Tensor:
    """[L, C] bf16 <- silu(causal conv) of the prefill sequences in x[:, :C], conv state updated in place: stock's
    causal_conv1d_fn in one HIP launch (kernels/r9k_gdn.hip r9k_gdn_conv). x rows are the batch's own rows
    (rowmap[i] int64 is the x row of output row i; None = the same row); weight [C, 4] bf16; conv_state
    [lines, C, >= 3] bf16 in either layout (dim-first, or the transposed view of the other) -- the first three
    columns are the prefill history, as in stock; with speculative decoding the state is wider (the conv update's
    rolling window) and the rest is left alone; idx [N] int32 state slots; cu [N + 1] int32; hasinit [N] bool."""
    N = cu.shape[0] - 1
    assert x.dtype == torch.bfloat16 and x.stride(1) == 1 and x.shape[1] >= C
    assert weight.dtype == torch.bfloat16 and weight.shape == (C, 4) and weight.stride(1) == 1
    assert conv_state.dtype == torch.bfloat16 and conv_state.shape[1] == C and conv_state.shape[2] >= 3
    assert idx.dtype == torch.int32 and idx.dim() == 1 and idx.shape[0] >= N and cu.dtype == torch.int32
    assert cu.is_contiguous() and (hasinit is None or (hasinit.dtype == torch.bool and hasinit.is_contiguous()))
    if rowmap is not None:
        rowmap = rowmap.to(torch.int64).contiguous()
    out = torch.empty((L, C), dtype=torch.bfloat16, device=x.device)
    rc = lib().r9k_gdn_conv(x.data_ptr(), x.stride(0), weight.data_ptr(), weight.stride(0), conv_state.data_ptr(),
                            conv_state.stride(0), conv_state.stride(1), conv_state.stride(2), out.data_ptr(), C,
                            idx.data_ptr(), idx.stride(0), hasinit.data_ptr() if hasinit is not None else 0,
                            cu.data_ptr(), rowmap.data_ptr() if rowmap is not None else 0, N, C,
                            torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_gdn_conv failed ({rc}) N={N} C={C} L={L}")
    return out


def _inner_contiguous(t: torch.Tensor) -> bool:
    """[pages, *shape] with each page's elements contiguous (the page stride may be larger: a view of the raw page)."""
    st = 1
    for d in range(t.dim() - 1, 0, -1):
        if t.stride(d) != st:
            return False
        st *= t.shape[d]
    return True


def onepage_available() -> bool:
    try:
        return hasattr(lib(), "r9k_gdn_spec_verify")
    except Exception:
        return False


def spec_verify(mixed_qkv, b, a, A_log, dt_bias, w, z, out, state, rec_qkv, rec_ab, flags, idx, cu, acc, Hk, Hv,
                scale, eps, *, rowmap=None, zero_from=None, gate_sigmoid=False) -> None:
    """One-page verify step (kernels/r9k_gdn.hip r9k_gdn_spec_verify) for N spec-decode sequences: page idx[n]
    (int32 [N] or [N, *], column 0) is first advanced by replaying acc[n] (int32 [N]) tokens of the live record from
    the previous step, then the 1 + spec candidates run from it with outputs through the gated norm, and their
    conv'd rows and b/a become the new record. rec_qkv [pages, 2, S, rowlen], rec_ab [pages, 2, S, 2 Hv] bf16,
    flags [pages, 2] int32. mixed_qkv rows compact (cu [N + 1]); b, a, z, out via rowmap (int64) or the same row.
    Rows of out from zero_from (None: from the batch's last token, rows_total if nothing) are zeroed."""
    N = cu.shape[0] - 1
    S = rec_qkv.shape[2]
    assert mixed_qkv.dtype == torch.bfloat16 and mixed_qkv.stride(1) == 1 and z.stride(1) == 1 and out.stride(1) == 1
    assert state.dtype in (torch.float32, torch.bfloat16) and state.stride(-1) == 1
    assert idx.dtype == torch.int32 and cu.dtype == torch.int32 and cu.is_contiguous() and idx.shape[0] >= N
    assert acc.dtype == torch.int32 and acc.is_contiguous() and acc.shape[0] >= N
    assert rec_qkv.dtype == torch.bfloat16 and _inner_contiguous(rec_qkv) and rec_qkv.shape[1] == 2 \
        and rec_qkv.shape[3] == mixed_qkv.shape[1]
    assert rec_ab.dtype == torch.bfloat16 and _inner_contiguous(rec_ab) and tuple(rec_ab.shape[1:]) == (2, S, 2 * Hv)
    assert flags.dtype == torch.int32 and _inner_contiguous(flags) and flags.shape[1] == 2
    if rowmap is not None:
        rowmap = rowmap.to(torch.int64).contiguous()
    rows = out.shape[0]
    rc = lib().r9k_gdn_spec_verify(mixed_qkv.data_ptr(), mixed_qkv.stride(0), b.data_ptr(), b.stride(0), a.data_ptr(),
                                   a.stride(0), A_log.data_ptr(), dt_bias.data_ptr(), w.data_ptr(), z.data_ptr(),
                                   z.stride(0), out.data_ptr(), out.stride(0), state.data_ptr(), state.stride(0),
                                   1 if state.dtype == torch.float32 else 0, rec_qkv.data_ptr(), rec_qkv.stride(0),
                                   rec_ab.data_ptr(), rec_ab.stride(0), flags.data_ptr(), flags.stride(0), S,
                                   idx.data_ptr(), idx.stride(0), cu.data_ptr(), acc.data_ptr(),
                                   rowmap.data_ptr() if rowmap is not None else 0, N, Hk, Hv,
                                   -1 if zero_from is None else int(zero_from), rows, float(scale), float(eps),
                                   1 if gate_sigmoid else 0, torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_gdn_spec_verify failed ({rc}) N={N} S={S} rows={rows} Hk={Hk} Hv={Hv}")


def spec_commit(layers, idx, ns, Hk, Hv, *, conv_shift=False) -> None:
    """Replay the accepted tokens (ns [N] int32) of every spec sequence on its page idx[n] (int32 [N] or [N, *],
    column 0) for a list of layers = (state, rec_qkv, rec_ab, conv, A_log, dt_bias) tuples of one KV-cache group
    (same page per request), storing the new base state; conv_shift shifts each conv window [pages, C, L] (any of
    the two layouts) left by ns - 1. One launch (kernels/r9k_gdn.hip r9k_gdn_spec_commit)."""
    N = ns.shape[0]
    st0, rq0, rab0, cv0, _, _ = layers[0]
    assert ns.dtype == torch.int32 and ns.is_contiguous() and idx.dtype == torch.int32 and idx.shape[0] >= N
    for st, rq, rab, cv, A_log, dt_bias in layers:
        assert st.dtype == st0.dtype and st.stride(0) == st0.stride(0) and rq.stride(0) == rq0.stride(0)
        assert rab.stride(0) == rab0.stride(0) and cv.stride() == cv0.stride() and A_log.dtype == torch.float32
    tab = torch.tensor([[st.data_ptr(), rq.data_ptr(), rab.data_ptr(), cv.data_ptr(), A_log.data_ptr(),
                         dt_bias.data_ptr()] for st, rq, rab, cv, A_log, dt_bias in layers],
                       dtype=torch.int64, device=idx.device)
    C, L = (cv0.shape[1], cv0.shape[2])
    rc = lib().r9k_gdn_spec_commit(tab.data_ptr(), st0.stride(0), 1 if st0.dtype == torch.float32 else 0,
                                   rq0.stride(0), rab0.stride(0), cv0.stride(0), cv0.stride(1), cv0.stride(2), L, C,
                                   1 if conv_shift else 0, idx.data_ptr(), idx.stride(0), ns.data_ptr(), len(layers),
                                   N, Hk, Hv, torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_gdn_spec_commit failed ({rc}) N={N} layers={len(layers)}")


def _onepage(self) -> bool:
    return bool(getattr(self, "_r9k_onepage", False))


_BACKEND = None


def onepage_from_config(vllm_config) -> bool:
    """Whether the one-page GDN state applies to this configuration (the model classes size the mamba page from
    this before the layers exist; the layers decide the same way, plus their own geometry check)."""
    if not ONEPAGE or not FUSED or not _SPLIT_OK or not onepage_available():
        return False
    spec = int(vllm_config.num_speculative_tokens or 0)
    if spec <= 0 or spec + 1 > MAX_SPEC_TOKENS:
        return False
    return vllm_config.cache_config.mamba_cache_mode == "none"


def onepage_record_shapes(vllm_config):
    """The record's shapes and dtypes per GDN layer page: conv'd q/k/v rows [2, S, rowlen] bf16, b/a rows
    [2, S, 2 Hv] bf16, flags [2] int32 (S = 1 + num_speculative_tokens, per TP rank)."""
    hf = vllm_config.model_config.hf_text_config
    tp = vllm_config.parallel_config.tensor_parallel_size
    S = int(vllm_config.num_speculative_tokens) + 1
    Hk, Hv = hf.linear_num_key_heads // tp, hf.linear_num_value_heads // tp
    rowlen = (2 * Hk * hf.linear_key_head_dim + Hv * hf.linear_value_head_dim)
    return ((2, S, rowlen), (2, S, 2 * Hv), (2,)), (torch.bfloat16, torch.bfloat16, torch.int32)


def onepage_state_shape_from_config(base_cls, vllm_config):
    """For a model class's get_mamba_state_shape_from_config: stock's shapes plus the record."""
    shapes = tuple(base_cls.get_mamba_state_shape_from_config(vllm_config))
    if not (onepage_from_config(vllm_config) or PAGE_PAD):
        logger.info_once("r9700: one-page GDN state not applied at the model level (%s): stock mamba page",
                         base_cls.__name__)
        return shapes
    full = shapes + onepage_record_shapes(vllm_config)[0]
    logger.info_once("r9700: one-page GDN state: mamba page sized from %s for %s", base_cls.__name__, str(full))
    return full


def onepage_state_dtype_from_config(base_cls, vllm_config):
    dtypes = tuple(base_cls.get_mamba_state_dtype_from_config(vllm_config))
    if not (onepage_from_config(vllm_config) or PAGE_PAD):
        return dtypes
    return dtypes + onepage_record_shapes(vllm_config)[1]


def onepage_specs_from_config(base_cls, vllm_config):
    """For a model class's get_mamba_specs_from_config (Flash-Next: GDN spec first, then the PLE conv spec): the GDN
    spec with the record appended."""
    import dataclasses
    specs = tuple(base_cls.get_mamba_specs_from_config(vllm_config))
    if not specs or not (onepage_from_config(vllm_config) or PAGE_PAD):
        logger.info_once("r9700: one-page GDN state not applied at the model level (%s): stock mamba pages",
                         base_cls.__name__)
        return specs
    rs, rd = onepage_record_shapes(vllm_config)
    gdn = dataclasses.replace(specs[0], shapes=tuple(specs[0].shapes) + rs, dtypes=tuple(specs[0].dtypes) + rd)
    logger.info_once("r9700: one-page GDN state: mamba page sized from %s for %s", base_cls.__name__,
                     str([tuple(sh) for sh in gdn.shapes]))
    return (gdn,) + specs[1:]


def onepage_backend():
    """The stock GDN attention backend with a builder that keeps every row's accepted-token count on the metadata
    (stock keeps only the spec rows'); the one-page verify needs it for a batch of draft-less decodes."""
    global _BACKEND
    if _BACKEND is None:
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend, GDNAttentionMetadataBuilder

        class R9kGDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
            def build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens=None,
                      num_decode_draft_tokens_cpu=None, fast_build=False):
                md = super().build(common_prefix_len, common_attn_metadata, num_accepted_tokens,
                                   num_decode_draft_tokens_cpu, fast_build)
                md.r9k_all_num_accepted = num_accepted_tokens
                return md

        class R9kGDNAttentionBackend(GDNAttentionBackend):
            @staticmethod
            def get_builder_cls():
                return R9kGDNAttentionMetadataBuilder

        _BACKEND = R9kGDNAttentionBackend
    return _BACKEND


def _params(self):
    p = self.__dict__.get("_r9k_gdn_params")
    if p is None:
        p = (self.A_log.detach().float().contiguous(), self.dt_bias.detach().float().contiguous(),
             self.norm.weight.detach().float().contiguous())
        self.__dict__["_r9k_gdn_params"] = p
    return p


def _spec_width(self, idx: torch.Tensor) -> int:
    """Candidate columns of the conv window / rows of a spec step: S = 1 + num_speculative_tokens. In stock mode that
    is idx.size(1) (one state slot per candidate); with one page per request idx has a single column, so take S
    from the record (see the note at the fused path)."""
    if _onepage(self) and len(self.kv_cache) > 2:
        return int(self.kv_cache[2].shape[2])
    return int(idx.size(1))


def _prefill_ok(self, md) -> bool:
    return (PREFILL_MAX > 0 and md.num_prefills > 0 and md.num_decodes == 0
            and md.num_prefill_tokens <= PREFILL_MAX and md.prefill_state_indices is not None
            and md.prefill_query_start_loc is not None and md.has_initial_state is not None
            and (md.spec_sequence_masks is None or (md.spec_state_indices_tensor is not None
                                                    and md.num_accepted_tokens is not None
                                                    and md.spec_state_indices_tensor.size(1) <= MAX_SPEC_TOKENS))
            and self.kv_cache[1].dtype in (torch.float32, torch.bfloat16) and seq_available())


def _prefill(self, md, qkvz: torch.Tensor, ba: torch.Tensor, out: torch.Tensor, qkv_size: int, Hv: int) -> None:
    """A step with short prefills: the conv (which also moves the conv state), then one launch for the core + norm
    of the prefill sequences -- and, when running spec-decode sequences share the step, stock's conv update and a
    second launch for theirs. Every launch of ours reads and writes at the batch's own rows, so only the conv
    update's input is gathered."""
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    logger.info_once("r9700: GDN short-prefill core on r9k_gdn_seq (steps with <= %d prefill tokens: conv + one "
                     "recurrence/norm launch per layer)", PREFILL_MAX)
    A_log, dt_bias, w = _params(self)
    n_act = md.num_actual_tokens
    conv_state = self.kv_cache[0] if is_conv_state_dim_first() else self.kv_cache[0].transpose(-1, -2)
    conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
    b, a = self.split_ba(ba)
    z, state = qkvz[:, qkv_size:], self.kv_cache[1]
    Hk, scale, eps = self.num_k_heads // self.tp_size, self.head_k_dim ** -0.5, self.layer_norm_epsilon
    gs = self.norm.activation == "sigmoid"
    x = qkvz[:n_act, :qkv_size]
    spec = md.spec_sequence_masks is not None
    rowmap = md.non_spec_token_indx if spec else None
    if CONV and conv_state.dtype == torch.bfloat16 and conv_weights.dtype == torch.bfloat16 \
            and conv_weights.size(1) == 4 and hasattr(lib(), "r9k_gdn_conv"):
        mixed = conv_prefill(qkvz, qkv_size, md.num_prefill_tokens, conv_weights, conv_state,
                             md.non_spec_state_indices_tensor, md.non_spec_query_start_loc,
                             hasinit=md.has_initial_state, rowmap=rowmap)
    else:
        xn = x.index_select(0, rowmap) if spec else x
        mixed = causal_conv1d_fn(xn.transpose(0, 1), conv_weights, self.conv1d.bias, activation=self.activation,
                                 conv_states=conv_state, has_initial_state=md.has_initial_state,
                                 cache_indices=md.non_spec_state_indices_tensor,
                                 query_start_loc=md.non_spec_query_start_loc, metadata=md).transpose(0, 1)
    onepage = _onepage(self)
    seq_core(mixed, b, a, A_log, dt_bias, w, z, out, state, md.prefill_state_indices, md.prefill_query_start_loc,
             Hk, Hv, scale, eps, hasinit=md.prefill_has_initial_state, rowmap=rowmap,
             zero_from=n_act if spec else md.num_prefill_tokens, gate_sigmoid=gs,
             flags=self.kv_cache[4] if onepage else None)
    if TRACE and (".layers.0." in self.prefix or ".layers.1." in self.prefix):
        torch.cuda.synchronize()
        logger.info("r9700 trace %s prefill N=%d pages=%s cu=%s spec_rows=%s | mixed %.6e out %.6e",
                    self.prefix.split(".layers.")[1].split(".")[0], int(md.num_prefills),
                    md.prefill_state_indices.cpu().tolist(), md.prefill_query_start_loc.cpu().tolist(),
                    int(md.num_spec_decode_tokens) if spec else 0, float(mixed.float().abs().sum()),
                    float(out[:n_act].float().abs().sum()))
    if not spec:
        return
    _spec_rows(self, md, x, b, a, z, out, A_log, dt_bias, w, conv_state, conv_weights, Hk, Hv, scale, eps, gs)


def _spec_rows(self, md, x, b, a, z, out, A_log, dt_bias, w, conv_state, conv_weights, Hk, Hv, scale, eps, gs):
    """The spec-decode rows of a mixed step: stock's conv update on their gathered rows, then one launch that
    reads b/a/z and writes out at the batch's own rows."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    N = md.num_spec_decodes
    idx, cu, acc = md.spec_state_indices_tensor, md.spec_query_start_loc, md.num_accepted_tokens
    ms = causal_conv1d_update(x.index_select(0, md.spec_token_indx), conv_state, conv_weights, self.conv1d.bias,
                              self.activation, conv_state_indices=idx[:, 0][:N], num_accepted_tokens=acc,
                              query_start_loc=cu, max_query_len=_spec_width(self, idx), validate_data=False)
    if _onepage(self):
        spec_verify(ms, b, a, A_log, dt_bias, w, z, out, self.kv_cache[1], self.kv_cache[2], self.kv_cache[3],
                    self.kv_cache[4], idx[:N], cu[: N + 1], acc[:N], Hk, Hv, scale, eps, rowmap=md.spec_token_indx,
                    zero_from=out.shape[0], gate_sigmoid=gs)
    else:
        seq_core(ms, b, a, A_log, dt_bias, w, z, out, self.kv_cache[1], idx[:N], cu[: N + 1], Hk, Hv, scale, eps,
                 acc=acc[:N], rowmap=md.spec_token_indx, gate_sigmoid=gs)
    if TRACE:
        rm = md.spec_token_indx.long()
        _trace(self, "mixed-spec" if _onepage(self) else "mixed-slot", md, ms, b[rm], a[rm], z[rm], out[rm], idx, cu,
               acc, N, self.kv_cache)


def _long_prefill_and_spec(self, md, qkvz, ba, out, qkv_size, Hv):
    """A step that mixes prefills longer than R9K_GDN_PREFILL_MAX with spec-decode rows (one-page state): stock's
    chunked prefill for the prefill rows (gathered), our verify for the spec rows."""
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import fused_post_conv_prep
    A_log, dt_bias, w = _params(self)
    n_act = md.num_actual_tokens
    conv_state = self.kv_cache[0] if is_conv_state_dim_first() else self.kv_cache[0].transpose(-1, -2)
    conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
    b, a = self.split_ba(ba)
    z = qkvz[:, qkv_size:]
    Hk, scale, eps = self.num_k_heads // self.tp_size, self.head_k_dim ** -0.5, self.layer_norm_epsilon
    gs = self.norm.activation == "sigmoid"
    x = qkvz[:n_act, :qkv_size]
    rowmap = md.non_spec_token_indx
    L = rowmap.shape[0]
    D = self.head_v_dim
    xn = x.index_select(0, rowmap)
    mixed = causal_conv1d_fn(xn.transpose(0, 1), conv_weights, self.conv1d.bias, activation=self.activation,
                             conv_states=conv_state, has_initial_state=md.has_initial_state,
                             cache_indices=md.non_spec_state_indices_tensor,
                             query_start_loc=md.non_spec_query_start_loc, metadata=md).transpose(0, 1)
    q, k, v, g_, beta = fused_post_conv_prep(conv_output=mixed, a=a.index_select(0, rowmap).contiguous(),
                                             b=b.index_select(0, rowmap).contiguous(), A_log=self.A_log,
                                             dt_bias=self.dt_bias, num_k_heads=Hk, head_k_dim=self.head_k_dim,
                                             head_v_dim=D, apply_l2norm=True, output_g_exp=False)
    ssm = self.kv_cache[1]
    psi = md.prefill_state_indices.long()
    init = ssm[psi]
    init[~md.prefill_has_initial_state, ...] = 0
    o, final = self.chunk_gated_delta_rule(q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0), g=g_.unsqueeze(0),
                                           beta=beta.unsqueeze(0), initial_state=init, output_final_state=True,
                                           cu_seqlens=md.prefill_query_start_loc, chunk_indices=md.chunk_indices,
                                           chunk_offsets=md.chunk_offsets, use_qk_l2norm_in_kernel=False)
    ssm[psi] = final.to(ssm.dtype)
    self.kv_cache[4][psi] = 0
    y = self.norm(o.squeeze(0)[:L], z.index_select(0, rowmap).reshape(L, Hv, D)).flatten(-2)
    out.index_copy_(0, rowmap, y)
    if n_act < out.shape[0]:
        out[n_act:].zero_()
    _spec_rows(self, md, x, b, a, z, out, A_log, dt_bias, w, conv_state, conv_weights, Hk, Hv, scale, eps, gs)


def _nonspec_decodes(self, md, qkvz, ba, out, qkv_size, Hv):
    """A batch of single-token decodes without draft tokens (one-page state): stock's conv update, then our verify
    with one candidate each -- it replays a live record first, which stock's path would not."""
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    A_log, dt_bias, w = _params(self)
    n_act = md.num_actual_tokens
    N = md.num_decodes
    conv_state = self.kv_cache[0] if is_conv_state_dim_first() else self.kv_cache[0].transpose(-1, -2)
    conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
    b, a = self.split_ba(ba)
    Hk, scale, eps = self.num_k_heads // self.tp_size, self.head_k_dim ** -0.5, self.layer_norm_epsilon
    x = qkvz[:n_act, :qkv_size]
    idx = md.non_spec_state_indices_tensor
    mixed = causal_conv1d_update(x, conv_state, conv_weights, self.conv1d.bias, self.activation,
                                 conv_state_indices=idx[:n_act], validate_data=True, out=torch.empty_like(x))
    acc = md.r9k_all_num_accepted
    spec_verify(mixed, b, a, A_log, dt_bias, w, qkvz[:, qkv_size:], out, self.kv_cache[1], self.kv_cache[2],
                self.kv_cache[3], self.kv_cache[4], idx, md.non_spec_query_start_loc[: N + 1],
                acc[:N].to(torch.int32).contiguous(), Hk, Hv, scale, eps, zero_from=n_act,
                gate_sigmoid=self.norm.activation == "sigmoid")
    if TRACE:
        _trace(self, "nonspec", md, mixed, b, a, qkvz[:, qkv_size:], out, idx.view(-1, 1), md.non_spec_query_start_loc,
               acc[:N].to(torch.int32), N, self.kv_cache)


def _fused_ok(self, md) -> bool:
    idx = md.spec_state_indices_tensor
    return (md.spec_sequence_masks is not None and md.num_prefills == 0 and md.num_decodes == 0
            and md.num_spec_decodes > 0 and idx is not None and idx.size(1) <= MAX_SPEC_TOKENS
            and md.spec_query_start_loc is not None and md.num_accepted_tokens is not None
            and self.kv_cache[1].dtype in (torch.float32, torch.bfloat16))


def gdn_core(qkvz: torch.Tensor, ba: torch.Tensor, out: torch.Tensor, layer_name: LayerNameType) -> None:
    """out [T, Hv * 128] bf16 <- norm(core(conv(qkvz), ba), z): one fused launch after the conv update for a pure
    spec-decode batch; one (two with spec-decode sequences aboard) after stock's conv for a step with short prefills;
    stock's core + gated norm otherwise (long prefills, non-spec decodes, warmup without metadata)."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import GDNAttentionMetadata
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    layer_name = _resolve_layer_name(layer_name)
    fc = get_forward_context()
    self = fc.no_compile_layers[layer_name]
    md = fc.attn_metadata
    md = md.get(self.prefix) if isinstance(md, dict) else None
    T = qkvz.shape[0]
    qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
    Hv, D = self.num_v_heads // self.tp_size, self.head_v_dim
    if FUSED and md is not None and isinstance(md, GDNAttentionMetadata) and _fused_ok(self, md):
        p = _params(self)
        N, n_act = md.num_spec_decodes, md.num_actual_tokens
        idx, cu, acc = md.spec_state_indices_tensor, md.spec_query_start_loc, md.num_accepted_tokens
        conv_state = self.kv_cache[0] if is_conv_state_dim_first() else self.kv_cache[0].transpose(-1, -2)
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        x = qkvz[:n_act, :qkv_size]
        dbg = DEBUG and not torch.cuda.is_current_stream_capturing()
        if dbg:
            _debug_check(self, qkvz, ba, out, x, idx, cu, acc, N, n_act, T)
        # max_query_len sizes the conv window's candidate columns. With one page per request the KV group's block
        # table has ONE column (no speculative blocks), so idx is [N, 1] on every eagerly built step (mixed and
        # non-uniform batches; the cudagraph buffer is [N, S] only because vLLM's copy_ broadcasts the column):
        # idx.size(1) would be 1 and stock's update would roll the window as if one candidate per step. Use S.
        mql = _spec_width(self, idx)
        mixed = causal_conv1d_update(x, conv_state, conv_weights, self.conv1d.bias, self.activation,
                                     conv_state_indices=idx[:N, 0], num_accepted_tokens=acc[:N],
                                     query_start_loc=cu[: N + 1], max_query_len=mql, validate_data=False,
                                     out=torch.empty_like(x))          # qkvz itself is left untouched
        if dbg:
            torch.cuda.synchronize()
            logger.info("r9700 gdn debug %s: conv ok, mixed %s %s", self.prefix, tuple(mixed.shape), mixed.stride())
        b, a = self.split_ba(ba)
        if _onepage(self):
            spec_verify(mixed, b, a, p[0], p[1], p[2], qkvz[:, qkv_size:], out, self.kv_cache[1], self.kv_cache[2],
                        self.kv_cache[3], self.kv_cache[4], idx[:N], cu[: N + 1], acc[:N],
                        self.num_k_heads // self.tp_size, Hv, self.head_k_dim ** -0.5, self.layer_norm_epsilon,
                        gate_sigmoid=self.norm.activation == "sigmoid")
        else:
            decode_mtp(mixed, b, a, p[0], p[1], p[2], qkvz[:, qkv_size:], out, self.kv_cache[1], idx[:N],
                       cu[: N + 1], acc[:N], self.num_k_heads // self.tp_size, Hv, self.head_k_dim ** -0.5,
                       self.layer_norm_epsilon, gate_sigmoid=self.norm.activation == "sigmoid")
        if dbg:
            torch.cuda.synchronize()
            logger.info("r9700 gdn debug %s: kernel ok", self.prefix)
        if TRACE and not torch.cuda.is_current_stream_capturing():
            _trace(self, "spec" if _onepage(self) else "slot", md, mixed, b, a, qkvz[:, qkv_size:], out, idx, cu, acc, N,
                   self.kv_cache)
        return
    if FUSED and md is not None and isinstance(md, GDNAttentionMetadata) and _prefill_ok(self, md):
        _prefill(self, md, qkvz, ba, out, qkv_size, Hv)
        return
    if _onepage(self) and md is not None and isinstance(md, GDNAttentionMetadata):
        # with one state page per request, stock must never see a spec-decode row (its slot indices do not exist)
        if md.spec_sequence_masks is not None and md.num_prefills > 0 and md.num_decodes == 0 \
                and md.spec_state_indices_tensor is not None and md.num_accepted_tokens is not None:
            _long_prefill_and_spec(self, md, qkvz, ba, out, qkv_size, Hv)
            return
        if md.spec_sequence_masks is None and md.num_decodes > 0 and md.num_prefills == 0 \
                and getattr(md, "r9k_all_num_accepted", None) is not None \
                and md.non_spec_state_indices_tensor is not None:
            _nonspec_decodes(self, md, qkvz, ba, out, qkv_size, Hv)
            return
        if md.num_prefills > 0 and md.prefill_state_indices is not None:
            self.kv_cache[4][md.prefill_state_indices.long()] = 0     # stock prefill below: no live record
    dbg = DEBUG and not torch.cuda.is_current_stream_capturing()
    if dbg:
        torch.cuda.synchronize()
        logger.info("r9700 gdn debug %s: fallback T=%d md=%s", self.prefix, T, None if md is None else
                    f"prefills={md.num_prefills} decodes={md.num_decodes} spec={md.num_spec_decodes} act={md.num_actual_tokens}")
    mixed_qkv, z = qkvz.split([qkv_size, self.value_dim // self.tp_size], dim=-1)
    b, a = self.split_ba(ba)
    core = torch.zeros((T, Hv, D), dtype=qkvz.dtype, device=qkvz.device)
    self._forward_core(mixed_qkv=mixed_qkv, b=b.contiguous(), a=a.contiguous(), core_attn_out=core)
    if dbg:
        torch.cuda.synchronize()
        logger.info("r9700 gdn debug %s: fallback core ok", self.prefix)
    out.copy_(self.norm(core, z.reshape(T, Hv, D)).flatten(-2))
    if dbg:
        torch.cuda.synchronize()
        logger.info("r9700 gdn debug %s: fallback norm ok", self.prefix)


def _trace(self, tag, md, mixed, b, a, z, out, idx, cu, acc, N, kv):
    """R9K_GDN_TRACE=1: one log line per eager step for the first GDN layer: inputs, metadata and page checksums."""
    if ".layers.0." not in self.prefix and ".layers.1." not in self.prefix:
        return
    torch.cuda.synchronize()
    i = idx[:N].cpu(); c = cu[: N + 1].cpu(); k = acc[:N].cpu(); L = int(c[-1])
    pg = i[:, 0].long() if i.dim() == 2 else i.long()
    def cs(t): return f"{float(t.float().abs().sum()):.6e}"
    fl = kv[4][pg.to(idx.device)].cpu().tolist() if len(kv) > 4 else None
    st = cs(kv[1][pg.to(idx.device)])
    rec = cs(kv[2][pg.to(idx.device)]) if len(kv) > 2 else "-"
    logger.info("r9700 trace %s %s N=%d pages=%s acc=%s cu=%s | in: mixed %s b %s a %s z %s | out %s | state %s rec %s flags %s",
                self.prefix.split(".layers.")[1].split(".")[0], tag, N, pg.tolist(), k.tolist(), c.tolist(),
                cs(mixed[:L]), cs(b[:L]), cs(a[:L]), cs(z[:L]), cs(out[:L]), st, rec, fl)


def _debug_check(self, qkvz, ba, out, x, idx, cu, acc, N, n_act, T) -> None:
    torch.cuda.synchronize()
    st, cs = self.kv_cache[1], self.kv_cache[0]
    i, c, k = idx[:N].cpu(), cu[: N + 1].cpu(), acc[:N].cpu()
    logger.info("r9700 gdn debug %s: T=%d n_act=%d N=%d qkvz %s %s ba %s %s out %s x %s %s | state %s %s %s conv %s %s"
                " | idx %s %s min %d max %d | cu %s | acc %s | b/a dtypes %s %s",
                self.prefix, T, n_act, N, tuple(qkvz.shape), qkvz.stride(), tuple(ba.shape), ba.stride(),
                tuple(out.shape), tuple(x.shape), x.stride(), tuple(st.shape), st.stride(), st.dtype, tuple(cs.shape),
                cs.dtype, tuple(idx.shape), idx.stride(), int(i.min()), int(i.max()), c.tolist()[:20], k.tolist()[:20],
                idx.dtype, acc.dtype)
    bad = []
    if int(i.max()) >= st.shape[0]:
        bad.append(f"idx max {int(i.max())} >= state blocks {st.shape[0]}")
    if int(c[-1]) > n_act or int(c[-1]) > T:
        bad.append(f"cu[N] {int(c[-1])} > n_act {n_act} / T {T}")
    S = _spec_width(self, idx)
    if (c[1:] - c[:-1]).max() > S:
        bad.append(f"a sequence has more tokens than slots {S}")
    if int(k.min()) < 1 or int(k.max()) > S:
        bad.append(f"acc range {int(k.min())}..{int(k.max())}")
    if bad:
        logger.error("r9700 gdn debug %s: BAD %s", self.prefix, "; ".join(bad))


def _gdn_core_fake(qkvz: torch.Tensor, ba: torch.Tensor, out: torch.Tensor, layer_name: LayerNameType) -> None:
    return None


SPLIT_OP = "r9700::gdn_core"


def register_splitting_op() -> bool:
    """Put our op on vLLM's default piecewise-graph splitting list next to vllm::qwen_gdn_attention_core.

    The GDN core must run eagerly at every step of a piecewise prefill graph (its Triton kernels take that step's
    sequence layout); vLLM lists its own op in CompilationConfig._attention_ops for that. Ours replaces it in the
    traced forward, so without this entry a prefill piece captures our op's kernels for the capture batch and
    replays them for every later one -> GPU memory fault on the first real request (2026-09-29). Runs at plugin
    registration, before any VllmConfig copies the class list. Version-gated on the attribute existing."""
    from vllm.config.compilation import CompilationConfig
    ops = getattr(CompilationConfig, "_attention_ops", None)
    if not isinstance(ops, list):
        logger.warning("r9700: CompilationConfig._attention_ops not a list (%s): GDN core stays on stock's forward",
                       type(ops).__name__)
        return False
    if SPLIT_OP not in ops:
        ops.append(SPLIT_OP)
    return True


_SPLIT_OK = os.environ.get("R9K_GDN_DECODE", "r9k") != "stock" and register_splitting_op()


def register_ops() -> None:
    global _OPS_DONE
    if _OPS_DONE:
        return
    from ..ops import _LIB
    direct_register_custom_op("gdn_core", gdn_core, mutates_args=["out"], fake_impl=_gdn_core_fake,
                              target_lib=_LIB)
    _OPS_DONE = True


def _format(lin):
    """('mxfp4', wq, wsr, N, K) | ('fp8', Fp8Weight) | None for a linear served by a libr9k kernel."""
    mx = getattr(lin, "_r9k_mx", None)
    if mx is not None and getattr(lin, "_r9k_nk", None):
        N, K = lin._r9k_nk
        return ("mxfp4", lin.weight.data, lin.weight_scale.data, N, K, getattr(lin, "_r9k_fold", False))
    W = getattr(lin, "_r9k_fp8", None)
    if W is not None:
        return ("fp8", W)
    return None


class _MergeTrigger(QuantizeMethodBase):
    """Wraps in_proj_ba's quant method: stock behaviour, plus the merge once its own weights are final."""

    def __init__(self, inner, gdn):
        self.inner = inner
        object.__setattr__(self, "_gdn", gdn)

    def create_weights(self, *a, **k):
        return self.inner.create_weights(*a, **k)

    def apply(self, layer, x, bias=None):
        return self.inner.apply(layer, x, bias)

    def process_weights_after_loading(self, layer):
        self.inner.process_weights_after_loading(layer)
        self._gdn._r9k_merge()

    def __getattr__(self, name):          # anything else stock asks the method (e.g. flags) -> inner
        return getattr(self.__dict__["inner"], name)


class _MergedQKVZ(QuantizeMethodBase):
    def __init__(self, gdn, kind, payload, nq, nb):
        object.__setattr__(self, "_gdn", gdn)
        self.kind, self.payload, self.nq, self.nb = kind, payload, nq, nb

    def create_weights(self, *a, **k):
        raise RuntimeError("unused")

    def process_weights_after_loading(self, layer):
        pass

    def apply(self, layer, x, bias=None):
        from .. import ops
        if self.kind == "mxfp4":
            wq, wsr, N, K, fold = self.payload
            out = ops.mxfp4_linear(x, wq, wsr, N, K, fold).to(x.dtype)
        else:
            out = ops.fp8_linear(x, self.payload).to(x.dtype)
        qkvz, ba = out.split([self.nq, self.nb], dim=-1)
        self._gdn._r9k_ba_out = ba
        return qkvz


class _StashedBA(QuantizeMethodBase):
    def __init__(self, gdn):
        object.__setattr__(self, "_gdn", gdn)

    def create_weights(self, *a, **k):
        raise RuntimeError("unused")

    def process_weights_after_loading(self, layer):
        pass

    def apply(self, layer, x, bias=None):
        return self._gdn._r9k_ba_out


# vLLM resolves OOT pluggable layers by the in-tree CLASS name (PluggableLayer.__new__: cls.__name__)
@PluggableLayer.register_oot(name="QwenGatedDeltaNetAttention")
class R9kQwenGatedDeltaNetAttention(QwenGatedDeltaNetAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._r9k_ba_out = None
        ba = getattr(self, "in_proj_ba", None)
        if os.environ.get("R9K_GDN_MERGE", "1") == "1" and ba is not None and getattr(ba, "quant_method", None) \
                is not None and getattr(self, "in_proj_qkvz", None) is not None:
            ba.quant_method = _MergeTrigger(ba.quant_method, self)
        self._r9k_onepage = False
        if _SPLIT_OK and self._r9k_decode_fits() and available():
            register_ops()
            self._forward_method = self._r9k_forward
            self._r9k_onepage = self._r9k_onepage_ok()
            try:
                from vllm.config import get_current_vllm_config
                split = get_current_vllm_config().compilation_config.splitting_ops
                listed = split is not None and SPLIT_OP in split
            except Exception:
                listed = None
            logger.info_once("r9700: GDN MTP decode core on r9k_gdn_decode_mtp (conv update + one fused "
                             "gating/recurrence/norm launch per layer); %s in splitting_ops: %s", SPLIT_OP, listed)

    def _r9k_onepage_ok(self) -> bool:
        """One state page per request under speculative decoding (PROGRESS.md 2026-10-03): needs the kernels, the
        fused paths, a drafter, and vLLM's mamba cache mode "none" (prefix caching off: in "align" mode the runner's
        block-boundary copies pick a spec slot by num_accepted - 1, which no longer exists)."""
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        spec = int(cfg.num_speculative_tokens or 0)
        if not onepage_from_config(cfg):
            if ONEPAGE and FUSED and onepage_available() and 0 < spec < MAX_SPEC_TOKENS \
                    and cfg.cache_config.mamba_cache_mode != "none":
                logger.warning_once("r9700: one state page per request needs mamba cache mode 'none' (prefix "
                                    "caching off, PREFIX_CACHE=0); mode is %r, keeping vLLM's %d state pages per "
                                    "request", cfg.cache_config.mamba_cache_mode, spec + 1)
            return False
        self._r9k_record = onepage_record_shapes(cfg)
        logger.info_once("r9700: GDN state: one page per request under speculative decoding (vLLM: %d); the page "
                         "carries a 2 x %d-row record of the last step's inputs, replayed on acceptance", spec + 1,
                         spec + 1)
        return True

    def _r9k_full_states(self):
        """(shapes, dtypes) of the page: stock's two states plus, with one page per request, the record. Stock
        code unpacks get_state_shape()/get_state_dtype() into exactly two, so those stay stock."""
        shapes, dtypes = tuple(super().get_state_shape()), tuple(super().get_state_dtype())
        if not _onepage(self):
            return shapes, dtypes
        rs, rd = self._r9k_record                     # computed at init, when the config context is set
        return shapes + rs, dtypes + rd

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Stock's page splitting (MambaBase.bind_kv_cache) over the full state list."""
        from math import prod
        from vllm.utils.torch_utils import get_dtype_size
        shapes, dtypes = self._r9k_full_states()
        pages = kv_cache.squeeze(dim=(1, 2))
        states, offset = [], 0
        for shape, dtype in zip(shapes, dtypes):
            nbytes = prod(shape) * get_dtype_size(dtype)
            states.append(pages[:, offset: offset + nbytes].view(dtype).view(-1, *shape))
            offset += nbytes
        self.kv_cache = tuple(states)

    def get_kv_cache_spec(self, vllm_config):
        """Stock's spec with the record appended and no speculative pages. The model class already sized vLLM's
        padded mamba page for the record (get_mamba_state_shape_from_config); if a model class without that
        override is in use, grow the page here as a whole multiple of the attention page (vLLM then scales the
        attention block size by that ratio: unify_kv_cache_spec_page_size)."""
        import dataclasses
        from math import prod
        from vllm.utils.torch_utils import get_dtype_size
        spec = super().get_kv_cache_spec(vllm_config)
        if not _onepage(self) or spec is None:
            return spec
        shapes, dtypes = self._r9k_full_states()
        content = sum(prod(sh) * get_dtype_size(dt) for sh, dt in zip(shapes, dtypes))
        padded = spec.page_size_padded
        if padded is not None and content > padded:
            padded = (content + padded - 1) // padded * padded
            logger.warning_once("r9700: GDN page %d -> %d bytes for the one-page record (the model class did not "
                                "size it; attention block size follows); states %s", spec.page_size_padded, padded,
                                str([tuple(sh) for sh in shapes]))
        return dataclasses.replace(spec, shapes=shapes, dtypes=dtypes, num_speculative_blocks=0,
                                   page_size_padded=padded)

    def get_attn_backend(self):
        if _onepage(self):
            return onepage_backend()
        return super().get_attn_backend()

    def _r9k_decode_fits(self) -> bool:
        conds = {"flat layout": self.qkvz_layout == "flat",
                 "K=V=128": self.head_k_dim == 128 and self.head_v_dim == 128,
                 "Hv % Hk": self.num_v_heads % self.num_k_heads == 0,
                 "conv bias": getattr(self.conv1d, "bias", None) is None,
                 "conv act": self.activation in ("silu", "swish"), "norm group": self.norm.group_size is None,
                 "norm before gate": bool(self.norm.norm_before_gate),
                 "gate act": self.norm.activation in ("silu", "swish", "sigmoid"),
                 "ba tp": not self.disable_tp_for_ba_proj}
        if all(conds.values()):
            return True
        logger.warning_once("r9700: GDN MTP decode core skipped for %s (failed: %s)", self.prefix,
                            ", ".join(k for k, v in conds.items() if not v))
        return False

    def _r9k_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        T = hidden_states.size(0)
        qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        out = torch.empty((T, self.value_dim // self.tp_size), dtype=hidden_states.dtype, device=hidden_states.device)
        torch.ops.r9700.gdn_core(qkvz.view(T, -1), ba.view(T, -1), out, _encode_layer_name(self.prefix))
        output, _ = self.out_proj(out)
        return output

    def _r9k_merge(self) -> None:
        q, b = self.in_proj_qkvz, self.in_proj_ba
        if getattr(q, "bias", None) is not None or getattr(b, "bias", None) is not None:
            return
        fq, fb = _format(q), _format(b)
        if fq is None or fb is None or fq[0] != fb[0]:
            return
        from ..utils import note_shape
        if fq[0] == "mxfp4":
            _, wq1, ws1, n1, k1, f1 = fq
            _, wq2, ws2, n2, k2, f2 = fb
            if k1 != k2:
                return
            wq = torch.cat([wq1.reshape(1, -1), wq2.reshape(1, -1)], dim=1).contiguous()
            wsr = torch.cat([ws1, ws2], dim=2).contiguous()
            q.quant_method = _MergedQKVZ(self, "mxfp4", (wq, wsr, n1 + n2, k1, f1 and f2), n1, n2)
            note_shape("mxfp4", n1 + n2, k1)
        else:
            from ..kernels.fp8 import Fp8Weight
            W1, W2 = fq[1], fb[1]
            if W1.K != W2.K:
                return
            W = Fp8Weight(torch.cat([W1.wq, W2.wq]).contiguous(), torch.cat([W1.ws, W2.ws]).contiguous(),
                          W1.N + W2.N, W1.K)
            q.quant_method = _MergedQKVZ(self, "fp8", W, W1.N, W2.N)
            n1, n2 = W1.N, W2.N
            note_shape("fp8row", W.N, W.K)
        b.quant_method = _StashedBA(self)
        # the merged copy is the only one used from here on: release the separate weights (~1 GB/rank on the 27B)
        for lin in (q, b):
            for name in ("weight", "weight_scale"):
                t = getattr(lin, name, None)
                if isinstance(t, torch.Tensor):
                    setattr(lin, name, torch.nn.Parameter(t.new_empty((0,)), requires_grad=False))
            for name in ("_r9k_fp8", "_r9k_mx"):
                if name in lin.__dict__:
                    lin.__dict__[name] = None
        torch.cuda.empty_cache()     # load-time transients: don't leave them as fragmented reserve before profiling
        logger.info_once("r9700: GDN in_proj_qkvz + in_proj_ba merged into one %s GEMM (N=%d+%d)", fq[0], n1, n2)
