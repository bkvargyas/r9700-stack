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
            L.r9k_gdn_seq.argtypes = [ctypes.c_long] * 15 + [ctypes.c_int] + [ctypes.c_long] * 6 + \
                [ctypes.c_int] * 5 + [ctypes.c_float] * 2 + [ctypes.c_int] * 2 + [ctypes.c_long]
        if hasattr(L, "r9k_gdn_conv"):
            L.r9k_gdn_conv.restype = ctypes.c_int
            L.r9k_gdn_conv.argtypes = [ctypes.c_long] * 15 + [ctypes.c_int] * 2 + [ctypes.c_long]
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
             gate_sigmoid: bool = False) -> None:
    """The recurrence + gated norm for N sequences of any length, one launch (kernels/r9k_gdn.hip r9k_gdn_seq).
    acc None -> prefill sequences: idx [N] int32 (any stride) is each sequence's state slot, read when hasinit[n]
    (bool [N]; None = always) and written with the final state. acc [N] -> spec-decode sequences, as decode_mtp
    (idx [N, S]). mixed_qkv rows are the sequences back to back (cu [N + 1] int32); b, a, z, out are indexed by
    rowmap[row] (int64 [cu[N]]) -- the batch's own row order -- or by the same row when rowmap is None. Rows
    zero_from.. of out are zeroed (None: nothing)."""
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
                           rowmap.data_ptr() if rowmap is not None else 0, N, Hk, Hv,
                           rows if zero_from is None else int(zero_from), rows, float(scale), float(eps),
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


def _params(self):
    p = self.__dict__.get("_r9k_gdn_params")
    if p is None:
        p = (self.A_log.detach().float().contiguous(), self.dt_bias.detach().float().contiguous(),
             self.norm.weight.detach().float().contiguous())
        self.__dict__["_r9k_gdn_params"] = p
    return p


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
    seq_core(mixed, b, a, A_log, dt_bias, w, z, out, state, md.prefill_state_indices, md.prefill_query_start_loc,
             Hk, Hv, scale, eps, hasinit=md.prefill_has_initial_state, rowmap=rowmap,
             zero_from=n_act if spec else md.num_prefill_tokens, gate_sigmoid=gs)
    if not spec:
        return
    N = md.num_spec_decodes
    idx, cu, acc = md.spec_state_indices_tensor, md.spec_query_start_loc, md.num_accepted_tokens
    ms = causal_conv1d_update(x.index_select(0, md.spec_token_indx), conv_state, conv_weights, self.conv1d.bias,
                              self.activation, conv_state_indices=idx[:, 0][:N], num_accepted_tokens=acc,
                              query_start_loc=cu, max_query_len=idx.size(-1), validate_data=False)
    seq_core(ms, b, a, A_log, dt_bias, w, z, out, state, idx[:N], cu[: N + 1], Hk, Hv, scale, eps, acc=acc[:N],
             rowmap=md.spec_token_indx, gate_sigmoid=gs)


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
        mixed = causal_conv1d_update(x, conv_state, conv_weights, self.conv1d.bias, self.activation,
                                     conv_state_indices=idx[:N, 0], num_accepted_tokens=acc[:N],
                                     query_start_loc=cu[: N + 1], max_query_len=idx.size(1), validate_data=False,
                                     out=torch.empty_like(x))          # qkvz itself is left untouched
        if dbg:
            torch.cuda.synchronize()
            logger.info("r9700 gdn debug %s: conv ok, mixed %s %s", self.prefix, tuple(mixed.shape), mixed.stride())
        b, a = self.split_ba(ba)
        decode_mtp(mixed, b, a, p[0], p[1], p[2], qkvz[:, qkv_size:], out, self.kv_cache[1], idx[:N], cu[: N + 1],
                   acc[:N], self.num_k_heads // self.tp_size, Hv, self.head_k_dim ** -0.5, self.layer_norm_epsilon,
                   gate_sigmoid=self.norm.activation == "sigmoid")
        if dbg:
            torch.cuda.synchronize()
            logger.info("r9700 gdn debug %s: kernel ok", self.prefix)
        return
    if FUSED and md is not None and isinstance(md, GDNAttentionMetadata) and _prefill_ok(self, md):
        _prefill(self, md, qkvz, ba, out, qkv_size, Hv)
        return
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
    if (c[1:] - c[:-1]).max() > idx.shape[1]:
        bad.append(f"a sequence has more tokens than slots {idx.shape[1]}")
    if int(k.min()) < 1 or int(k.max()) > idx.shape[1]:
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
        if _SPLIT_OK and self._r9k_decode_fits() and available():
            register_ops()
            self._forward_method = self._r9k_forward
            try:
                from vllm.config import get_current_vllm_config
                split = get_current_vllm_config().compilation_config.splitting_ops
                listed = split is not None and SPLIT_OP in split
            except Exception:
                listed = None
            logger.info_once("r9700: GDN MTP decode core on r9k_gdn_decode_mtp (conv update + one fused "
                             "gating/recurrence/norm launch per layer); %s in splitting_ops: %s", SPLIT_OP, listed)

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
