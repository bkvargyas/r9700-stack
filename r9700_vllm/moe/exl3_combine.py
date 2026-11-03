"""The MoE combine of the EXL3 r9k expert tier folded into the eight-rank all-reduce (package C, 2026-11-03).

Stock order for DeepSeek-V4.1's MoE (vLLM MoERunner.forward on this pin; the router scales the pick weights by
routed_scaling_factor, the runner's own scale is 1.0; no residual add here -- the hyper-connection seam mixes the residual
streams AFTER the all-reduce):
    y      = bf16( sum over local picks k, in pick order, of w[t,k] * os[t*topk+k] )   (k_dn's epilogue, fp32)
    result = shared_output + y                                                        (bf16 add: one rounding)
    out    = tensor_model_parallel_all_reduce(result)
Here the last two are one launch: R9kAllReduce8.all_reduce_moe (kernels/r9k_ar.hip r9k_ar8_os_moe) forms `result`
word by word from the tier's per-slot f32 outputs (exl3_decode's scratch `os`), the pick weights and the shared output
inside the collective's first phase, with the same operations in the same order, so the bits are those of the unfused
path (tests/test_ar8.py: T = 1..48, topk 6 and 3). Saved a layer: the bf16 add kernel and its [T, 5120] write + read.
(k_dn still writes y: dropping that write is a flag in kernels/r9k_moe_exl3.hip, package A's file.)

Scope: messages the one-kernel path takes (T x 5120 bf16 <= R9K_AR8_OS_KB 64 KiB, i.e. T <= 6: the single-stream verify step
and the drafter); larger steps and any call this hook cannot prove matches (the tier did not run, other dtype, other
tensor) take the stock add + all-reduce inside the same op. The kernel and the plain one-kernel all-reduce are one
protocol (pools, counters, flags), so a rank that falls back still meets its peers in the same collective.

Mechanics: install(model) rebinds MoERunner.forward of every EXL3 MoE layer (as moe/fold.py does) so that, when the
call has a shared output and no padding, `shared + fused` + the final all-reduce become torch.ops.r9700.exl3_moe_ar
(opaque to torch.compile, captured into the HIP graphs); and wraps the layer's Exl3MoEMethod._apply_r9k to record the
routing it ran with (ids int32, weights f32) next to its output. The op takes the record only if its output IS the
tensor it was handed. R9K_EXL3_MOE_AR=0 (or no one-kernel ar8, or TP != 8) installs nothing.
"""
from __future__ import annotations

import os
import types
from typing import cast

import torch
from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)

ENABLED = os.environ.get("R9K_EXL3_MOE_AR", "1") == "1"

_RUNNERS: list = []
_LAST: dict = {}          # device index -> (y, ids int32 [S], w f32 [S], os f32, layer) of the latest r9k tier call
_STATS = {"fused": 0, "fallback": 0}
_OP_DONE = False


def _ar8():
    """The TP group's one-kernel R9kAllReduce8, or None."""
    try:
        from vllm.distributed import get_tp_group
        ar = getattr(get_tp_group().device_communicator, "_r9k_ar", None)
    except Exception:
        return None
    return ar if type(ar).__name__ == "R9kAllReduce8" and getattr(ar, "os_on", False) else None


def _wrap_method(qm) -> None:
    """Record what the r9k tier ran with (the instance's _apply_r9k; the tier itself is unchanged)."""
    if getattr(qm, "_r9k_moe_ar_wrapped", False):
        return
    orig = qm._apply_r9k

    def _apply_r9k(layer, x, topk_weights, topk_ids):
        ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
        w = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.float()
        ids, w = ids.contiguous(), w.contiguous()            # exactly what moe_decode passes to the kernels
        y = orig(layer, x, w, ids)
        from . import exl3_decode as XD
        b = XD._bufs.get(x.device.index)
        osb = b.get("os") if isinstance(b, dict) else None
        ok = x.dtype == torch.bfloat16 and y.dtype == torch.bfloat16 and osb is not None \
            and osb.numel() >= ids.numel() * x.shape[1]
        _LAST[x.device.index] = (y, ids.view(-1), w.view(-1), osb, layer) if ok else None
        return y

    qm._apply_r9k = _apply_r9k
    qm._r9k_moe_ar_wrapped = True


def _moe_ar(fused: torch.Tensor, shared: torch.Tensor, rid: int) -> torch.Tensor:
    from vllm.distributed import tensor_model_parallel_all_reduce
    rec = _LAST.pop(fused.device.index, None)
    ar = _ar8()
    if rec is not None and ar is not None:
        y, ids, w, osb, layer = rec
        T, H = shared.shape
        if y.data_ptr() == fused.data_ptr() and y.shape == fused.shape and fused.shape == shared.shape \
                and ids.numel() % T == 0 and ar.moe_takes(shared, ids.numel() // T):
            _STATS["fused"] += 1
            if _STATS["fused"] == 1:
                logger.info("r9700: exl3 MoE combine fused into the ar8 one-kernel all-reduce (first call T=%d)", T)
            return ar.all_reduce_moe(osb, w, ids, layer.exl3_g2l, layer._exl3.E, shared.contiguous(), ids.numel() // T)
    _STATS["fallback"] += 1
    return tensor_model_parallel_all_reduce(shared + fused)


def _moe_ar_fake(fused: torch.Tensor, shared: torch.Tensor, rid: int) -> torch.Tensor:
    return torch.empty_like(shared)


def _register() -> None:
    global _OP_DONE
    if _OP_DONE:
        return
    from vllm.utils.torch_utils import direct_register_custom_op
    from .. import ops
    ops.register()
    direct_register_custom_op("exl3_moe_ar", _moe_ar, mutates_args=[], fake_impl=_moe_ar_fake, target_lib=ops._LIB)
    _OP_DONE = True


def _forward(self, hidden_states, router_logits, input_ids=None, shared_experts_input=None):
    """MoERunner.forward (this pin, vllm 0.32.1rc1.dev466 / e4a5b553) with `shared + fused` + the final all-reduce
    as one op. Static decisions only (traced once by torch.compile)."""
    if shared_experts_input is None:
        hidden_states, shared_experts_input = self.apply_routed_input_transform(hidden_states)
    hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = self._maybe_pad_hidden_states(
        shared_experts_input, hidden_states)
    result = self._forward_entry(
        hidden_states, router_logits, shared_experts_input, input_ids, self._encode_layer_name(),
        self.moe_config.hidden_dim_unpadded if self._quant_method.has_unpadded_output else 0)
    shared_output, fused_output = result if isinstance(result, tuple) else (None, result)
    fused_output = cast(torch.Tensor, fused_output)
    if shared_output is not None and og_hidden_dim_pre_xform is None and og_hidden_dim_post_xform is None:
        return torch.ops.r9700.exl3_moe_ar(fused_output, shared_output, self._r9k_moe_ar_id)
    # anything else: the stock tail (install() only takes runners where these steps are the identity but the add)
    if og_hidden_dim_pre_xform is not None:
        fused_output = fused_output[..., :og_hidden_dim_pre_xform]
    fused_output_is_reduced = self._fused_output_is_reduced
    fused_output, fused_output_is_reduced = self._maybe_reduce_routed_output_before_transform(
        fused_output, fused_output_is_reduced)
    shared_output = self._maybe_reduce_shared_expert_output(shared_output, fused_output_is_reduced)
    shared_output, fused_output = self._maybe_apply_routed_scale_to_output(shared_output, fused_output)
    fused_output = self.apply_routed_output_transform(fused_output)
    result = shared_output + fused_output if shared_output is not None else fused_output
    result = self._maybe_reduce_final_output(result, og_hidden_dim_post_xform, fused_output_is_reduced)
    return self._maybe_add_zero_expert_output(result)


def install(model: torch.nn.Module) -> int:
    """Rebind every EXL3 MoE runner under `model` whose tail is exactly `shared + fused` -> all-reduce. Call once after
    the model is built (before torch.compile traces it). Returns the number of runners taken."""
    if not ENABLED or _ar8() is None:
        return 0
    from ..compat import gate
    need = ("apply_routed_input_transform", "_maybe_pad_hidden_states", "_forward_entry", "_encode_layer_name",
            "_fused_output_is_reduced", "_maybe_reduce_routed_output_before_transform",
            "_maybe_reduce_shared_expert_output", "_maybe_apply_routed_scale_to_output",
            "apply_routed_output_transform", "_maybe_reduce_final_output", "_maybe_add_zero_expert_output")
    n = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ != "MoERunner" or getattr(mod, "_r9k_moe_ar_id", None) is not None:
            continue
        qm = getattr(mod, "_quant_method", None)
        cfg = getattr(mod, "moe_config", None)
        ok = (type(qm).__name__ == "Exl3MoEMethod" and hasattr(qm, "_apply_r9k")
              and getattr(mod, "_shared_experts", None) is not None and all(hasattr(mod, a) for a in need)
              and getattr(mod, "routed_output_transform", None) is None
              and getattr(mod, "routed_scaling_factor", 1.0) == 1.0
              and type(getattr(mod, "router", None)).__name__ != "ZeroExpertRouter"
              and cfg is not None and not getattr(cfg, "is_sequence_parallel", True)
              and not getattr(cfg, "skip_final_all_reduce", True) and getattr(cfg, "tp_size", 1) == 8
              and not mod._fused_output_is_reduced and not getattr(mod, "_r9k_fold", False))
        if not ok:
            if type(qm).__name__ == "Exl3MoEMethod":
                logger.warning_once("r9700: exl3 MoE combine fusion skipped for %s (runner shape changed)", name)
            continue
        if n == 0:
            gate.check("moe_shared_fold")                       # the same MoERunner.forward copy as moe/fold.py
            _register()
        _wrap_method(qm)
        mod._r9k_moe_ar_id = len(_RUNNERS)
        _RUNNERS.append(mod)
        mod.forward = types.MethodType(_forward, mod)
        n += 1
    if n:
        logger.info("r9700: exl3 MoE combine -> ar8 one-kernel all-reduce on %d runners (messages <= %d KiB; larger "
                    "steps keep the stock add + all-reduce)", n, _ar8().os_max >> 10)
    return n
