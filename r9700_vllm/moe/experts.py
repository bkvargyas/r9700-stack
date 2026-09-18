"""R9700Mxfp4Experts: vLLM modular fused-MoE experts backed by libr9k's grouped MXFP4 x FP8 GEMM (gfx1201).

Layer weights (set by the CT MXFP4 hook's process_weights_after_loading, see ct_mxfp4.py):
  w13_weight        [E, N1/16*K1/16*32] int32   fragment order   (N1 = 2*I, K1 = hidden)
  w2_weight         [E, N2/16*K2/16*32] int32                    (N2 = hidden, K2 = I)
  w13_weight_scale  [E, K1/32 + 1, N1]  uint8   exponents K-major + per-row reference (last row)
  w2_weight_scale   [E, K2/32 + 1, N2]  uint8
Flow per call: moe_align_block_size(16) -> per-token fp8 quant -> grouped gate_up GEMM -> activation ->
per-row fp8 quant -> grouped down GEMM with the router weight folded in -> moe_sum over top-k.
"""
from __future__ import annotations

import os

import torch

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation, apply_moe_activation_supported
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceNoOP
from vllm.platforms import current_platform

from ..kernels import moe as K


def _cfg(env: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    v = os.environ.get(env)
    return tuple(int(x) for x in v.split(",")) if v else default  # type: ignore[return-value]


# (WV, SK, NPW) launch configs; tuned defaults for Flash-Next TP2 shapes, overridable for sweeps.
CFG_GATE_UP = _cfg("R9K_MOE_CFG1", (2, 4, 2))
CFG_DOWN = _cfg("R9K_MOE_CFG2", (4, 2, 1))


def r9k_available() -> bool:
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import on_gfx12x
        if not on_gfx12x():
            return False
        K.lib()
        return True
    except Exception:
        return False


class R9700Mxfp4Experts(mk.FusedMoEExpertsModular):
    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig,
                 max_num_tokens: int | None = None, num_dispatchers: int | None = None):
        super().__init__(moe_config=moe_config, quant_config=quant_config,
                         max_num_tokens=max_num_tokens, num_dispatchers=num_dispatchers)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @staticmethod
    def _supports_current_device() -> bool:
        return r9k_available()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        return activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU and apply_moe_activation_supported(activation)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (moe_parallel_config.use_fi_nvl_two_sided_kernels
                    or moe_parallel_config.use_fi_nvl_one_sided_kernels)

    @staticmethod
    def _dims(ws: torch.Tensor) -> tuple[int, int]:
        """(N, K) of one GEMM from its packed scale tensor [E, K/32 + 1, N]."""
        return ws.shape[2], (ws.shape[1] - 1) * K.GROUP

    def moe_problem_size(self, a1, w1, w2, topk_ids):
        N1, _ = self._dims(self.w1_scale)
        return w1.size(0), a1.size(0), N1 // 2, a1.size(-1), topk_ids.size(1)

    def workspace_shapes(self, M, N, K_, topk, global_num_experts, local_num_experts,
                         expert_tokens_meta, activation):
        # Only the output buffer comes from the MK workspace; intermediates are allocated per call (they are
        # tiny at decode and come from the graph pool under cudagraph capture).
        return ((M, K_), (1,), (M, K_))

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
              expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
              apply_router_weight_on_input):
        assert not apply_router_weight_on_input, "R9700Mxfp4Experts: router weight on input not supported"
        assert hidden_states.dtype == torch.bfloat16 and hidden_states.is_contiguous()
        N1, K1 = self._dims(self.w1_scale)
        N2, K2 = self._dims(self.w2_scale)
        M, topk = hidden_states.shape[0], topk_ids.shape[1]
        E = w1.shape[0]
        if global_num_experts == -1:
            global_num_experts = E
        numel = M * topk
        dev = hidden_states.device

        cache = getattr(self, "r9k_cache", None)
        if cache is None:
            passes = [(K.Mxfp4Experts(w1, self.w1_scale, N1, K1), K.Mxfp4Experts(w2, self.w2_scale, N2, K2),
                       moe_align_block_size(topk_ids, K.MOE_BLOCK, global_num_experts, expert_map))]
        else:
            assert expert_map is None, "r9k expert cache: expert parallelism not supported"
            cache.update(topk_ids)          # LRU manage + gather misses host -> VRAM slots
            (h1, h2), (c1, c2) = cache.hot(), cache.cold()
            passes = [(h1, h2, moe_align_block_size(topk_ids, K.MOE_BLOCK, global_num_experts, cache.table)),
                      (c1, c2, moe_align_block_size(topk_ids, K.MOE_BLOCK, global_num_experts, cache.map_cold))]

        xq, xs = K.quant_rows_fp8(hidden_states)
        gate_up = torch.empty((numel, N1), dtype=torch.bfloat16, device=dev)
        for W1, _, (sid, eid, ntpp) in passes:
            K.moe_gemm(xq, xs, W1, gate_up, sid, eid, ntpp, numel, topk, None, *CFG_GATE_UP,
                       num_experts=global_num_experts)

        act = torch.empty((numel, N1 // 2), dtype=torch.bfloat16, device=dev)
        self.activation(activation, act, gate_up)
        aq, as_ = K.quant_rows_fp8(act)

        down = torch.empty((numel, N2), dtype=torch.bfloat16, device=dev)
        if expert_map is not None:
            down.zero_()   # rows routed to experts on other ranks are never written
        tw = topk_weights.reshape(-1).to(torch.float32)
        for _, W2, (sid, eid, ntpp) in passes:
            K.moe_gemm(aq, as_, W2, down, sid, eid, ntpp, numel, 1, tw, *CFG_DOWN,
                       num_experts=global_num_experts)
        ops.moe_sum(down.view(M, topk, N2), output)
