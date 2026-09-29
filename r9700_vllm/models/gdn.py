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


def _fused_ok(self, md) -> bool:
    idx = md.spec_state_indices_tensor
    return (md.spec_sequence_masks is not None and md.num_prefills == 0 and md.num_decodes == 0
            and md.num_spec_decodes > 0 and idx is not None and idx.size(1) <= MAX_SPEC_TOKENS
            and md.spec_query_start_loc is not None and md.num_accepted_tokens is not None
            and self.kv_cache[1].dtype in (torch.float32, torch.bfloat16))


def gdn_core(qkvz: torch.Tensor, ba: torch.Tensor, out: torch.Tensor, layer_name: LayerNameType) -> None:
    """out [T, Hv * 128] bf16 <- norm(core(conv(qkvz), ba), z): one fused launch after the conv update for a pure
    spec-decode batch; stock's core + gated norm otherwise (prefill, mixed batches, warmup without metadata)."""
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
    if md is not None and isinstance(md, GDNAttentionMetadata) and _fused_ok(self, md):
        p = self.__dict__.get("_r9k_gdn_params")
        if p is None:
            p = (self.A_log.detach().float().contiguous(), self.dt_bias.detach().float().contiguous(),
                 self.norm.weight.detach().float().contiguous())
            self.__dict__["_r9k_gdn_params"] = p
        N, n_act = md.num_spec_decodes, md.num_actual_tokens
        idx, cu, acc = md.spec_state_indices_tensor, md.spec_query_start_loc, md.num_accepted_tokens
        conv_state = self.kv_cache[0] if is_conv_state_dim_first() else self.kv_cache[0].transpose(-1, -2)
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        mixed = causal_conv1d_update(qkvz[:n_act, :qkv_size], conv_state, conv_weights, self.conv1d.bias,
                                     self.activation, conv_state_indices=idx[:N, 0],
                                     num_accepted_tokens=acc[:N], query_start_loc=cu[: N + 1],
                                     max_query_len=idx.size(1), validate_data=False)
        b, a = self.split_ba(ba)
        decode_mtp(mixed, b, a, p[0], p[1], p[2], qkvz[:, qkv_size:], out, self.kv_cache[1], idx[:N], cu[: N + 1],
                   acc[:N], self.num_k_heads // self.tp_size, Hv, self.head_k_dim ** -0.5, self.layer_norm_epsilon,
                   gate_sigmoid=self.norm.activation == "sigmoid")
        return
    mixed_qkv, z = qkvz.split([qkv_size, self.value_dim // self.tp_size], dim=-1)
    b, a = self.split_ba(ba)
    core = torch.zeros((T, Hv, D), dtype=qkvz.dtype, device=qkvz.device)
    self._forward_core(mixed_qkv=mixed_qkv, b=b.contiguous(), a=a.contiguous(), core_attn_out=core)
    out.copy_(self.norm(core, z.reshape(T, Hv, D)).flatten(-2))


def _gdn_core_fake(qkvz: torch.Tensor, ba: torch.Tensor, out: torch.Tensor, layer_name: LayerNameType) -> None:
    return None


def register_ops() -> None:
    global _OPS_DONE
    if _OPS_DONE:
        return
    from ..ops import _LIB
    direct_register_custom_op("gdn_core", gdn_core, mutates_args=["qkvz", "out"], fake_impl=_gdn_core_fake,
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
        if os.environ.get("R9K_GDN_DECODE", "r9k") != "stock" and self._r9k_decode_fits() and available():
            register_ops()
            self._forward_method = self._r9k_forward
            logger.info_once("r9700: GDN MTP decode core on r9k_gdn_decode_mtp (conv update + one fused "
                             "gating/recurrence/norm launch per layer)")

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
