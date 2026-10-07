"""Two launches fewer per MoE layer around the experts' output (kernels/r9k_moe_sum.hip):

1. **Output alias.** vLLM's modular MoE kernel writes the experts' result into a workspace and copies it into the
   layer's output buffer in `finalize` -- on ROCm it only aliases the two when aiter's fused MoE is on
   (`FusedMoEKernelModularImpl._fused_experts`). Our experts write every element of their output, so `alias()`
   rebinds that method on our kernel instance with the alias taken whenever the shapes agree. 48 copies a step.
2. **Shared-expert fold.** The runner adds the shared expert's output to the routed output after the experts
   return (`MoERunner.forward`, `shared_output + fused_output`). The shared expert runs BEFORE routing, so its
   output is ready when our experts reduce over top-k: `r9k_moe_sum` adds it there, in fp32, with one bf16
   rounding instead of stock's two, and the runner's forward (rebound by `install()`) skips the add for calls
   that folded it. 48 adds a step.

Both are monkeypatches of vLLM internals, listed in compat/gate.py; R9K_MOE_FOLD=stock leaves vLLM's path.
"""
from __future__ import annotations

import os
import types
from typing import cast

import torch

from vllm.logger import init_logger

from ..compat import gate

logger = init_logger("vllm." + __name__)

ENABLED = os.environ.get("R9K_MOE_FOLD", "r9k") == "r9k"


# ------------------------------------------------------------------------------------------ 1. output alias
def _fused_experts_alias(self, in_dtype, a1q, a1q_scale, w1, w2, topk_weights, topk_ids, activation,
                         global_num_experts, local_num_experts, expert_map, apply_router_weight_on_input,
                         expert_tokens_meta, output_alias=None):
    """FusedMoEKernelModularImpl._fused_experts (vLLM e97573215) with the output alias taken on ROCm too."""
    _, M_full, N, K, top_k = self.fused_experts.moe_problem_size(a1q, w1, w2, topk_ids)
    if M_full == 0:
        return torch.empty_like(a1q, dtype=in_dtype)
    workspace13, workspace2, fused_out = self._allocate_buffers(
        in_dtype, a1q.device, M_full, M_full, N, K, top_k, global_num_experts, local_num_experts,
        expert_tokens_meta, activation)
    if (output_alias is not None and output_alias.shape == fused_out.shape and output_alias.dtype == fused_out.dtype
            and output_alias.device == fused_out.device and output_alias.is_contiguous()):
        fused_out = output_alias
    self.fused_experts.apply(
        output=fused_out, hidden_states=a1q, w1=w1, w2=w2, topk_weights=topk_weights, topk_ids=topk_ids,
        activation=activation, global_num_experts=global_num_experts, expert_map=expert_map,
        a1q_scale=a1q_scale, a2_scale=self.fused_experts.a2_scale, workspace13=workspace13,
        workspace2=workspace2, expert_tokens_meta=expert_tokens_meta,
        apply_router_weight_on_input=apply_router_weight_on_input)
    return fused_out


def alias(moe_kernel) -> bool:
    """Take the output alias in this kernel's modular impl (our experts fill their output completely)."""
    if not ENABLED:
        return False
    impl = getattr(moe_kernel, "impl", None)
    if impl is None or type(impl).__name__ != "FusedMoEKernelModularImpl" or not hasattr(impl, "_fused_experts") \
            or not hasattr(impl, "_allocate_buffers"):
        return False
    gate.check("moe_output_alias")
    impl._fused_experts = types.MethodType(_fused_experts_alias, impl)
    return True


# ------------------------------------------------------------------------------------- 2. shared-expert fold
def shared_output(experts, M: int, N: int) -> torch.Tensor | None:
    """The runner's shared-expert output for this call, if the fold is installed and the tensor fits."""
    layer = getattr(experts, "r9k_layer", None)
    runner = getattr(layer, "_r9k_runner", None)
    se = getattr(runner, "_shared_experts", None)
    out = getattr(se, "output", None) if se is not None else None
    if (isinstance(out, torch.Tensor) and out.dim() == 2 and out.shape[0] == M and out.shape[1] == N
            and out.dtype == torch.bfloat16 and out.stride(1) == 1 and out.stride(0) % 8 == 0):
        return out
    return None


def mark(experts, folded: bool) -> None:
    layer = getattr(experts, "r9k_layer", None)
    if layer is not None:
        layer._r9k_shared_folded = folded


def _forward(self, hidden_states, router_logits, input_ids=None, shared_experts_input=None):
    """MoERunner.forward (vLLM e97573215) minus the shared add for calls whose experts folded it."""
    if shared_experts_input is None:
        hidden_states, shared_experts_input = self.apply_routed_input_transform(hidden_states)
    hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = self._maybe_pad_hidden_states(
        shared_experts_input, hidden_states)
    result = self._forward_entry(
        hidden_states, router_logits, shared_experts_input, input_ids, self._encode_layer_name(),
        self.moe_config.hidden_dim_unpadded if self._quant_method.has_unpadded_output else 0)
    shared_output, fused_output = result if isinstance(result, tuple) else (None, result)
    fused_output = cast(torch.Tensor, fused_output)
    if shared_output is not None and getattr(self.routed_experts, "_r9k_shared_folded", False):
        shared_output = None                      # already inside fused_output (r9k_moe_sum)
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
    """Rebind the forward of every MoE runner under `model` that has a shared expert, and point its
    RoutedExperts at it so the experts can find the shared output. R9K_MOE_FOLD=stock skips."""
    if not ENABLED:
        return 0
    n = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ != "MoERunner" or getattr(mod, "_shared_experts", None) is None:
            continue
        re_ = getattr(mod, "routed_experts", None)
        need = ("apply_routed_input_transform", "_maybe_pad_hidden_states", "_forward_entry", "_encode_layer_name",
                "_fused_output_is_reduced", "_maybe_reduce_routed_output_before_transform",
                "_maybe_reduce_shared_expert_output", "_maybe_apply_routed_scale_to_output",
                "apply_routed_output_transform", "_maybe_reduce_final_output", "_maybe_add_zero_expert_output")
        if re_ is None or not all(hasattr(mod, a) for a in need):
            logger.warning_once("r9700: shared-expert fold skipped for %s (runner shape changed)", name)
            continue
        gate.check("moe_shared_fold")
        # a plain attribute: nn.Module.__setattr__ would register the runner as a child of its own child and
        # vLLM's tied-weight scan (named_modules) would recurse forever at load
        object.__setattr__(re_, "_r9k_runner", mod)
        re_._r9k_shared_folded = False
        mod.forward = types.MethodType(_forward, mod)
        n += 1
    if n:
        logger.info("r9700: shared-expert fold (r9k_moe_sum adds the shared output; runner add skipped) on %d "
                    "runners", n)
    return n
