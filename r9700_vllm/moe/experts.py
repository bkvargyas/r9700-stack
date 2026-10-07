"""R9700Mxfp4Experts: vLLM modular fused-MoE experts backed by libr9k's grouped MXFP4 x FP8 GEMM (gfx1201).

Layer weights (set by R9kMxfp4MoEMethod.process_weights_after_loading, quant/ct.py):
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
from . import fold, route


def _cfg(env: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    v = os.environ.get(env)
    return tuple(int(x) for x in v.split(",")) if v else default  # type: ignore[return-value]


# (WV, SK, NPW) launch configs; tuned defaults for Flash-Next TP2 shapes, overridable for sweeps. Applied through
# K.legal_cfg, which swaps in pick_cfg when another TP size shards K to a shape the tuned SK does not divide.
CFG_GATE_UP = _cfg("R9K_MOE_CFG1", (2, 4, 2))
CFG_DOWN = _cfg("R9K_MOE_CFG2", (4, 2, 1))
# Per-rank shapes at other TP sizes, swept with tuning/decode_moe_sweep.py (2026-09-24, TP=4: w13 320x2560,
# 1-64 tokens): (1, 8, 1) is 8-11% faster than the TP=2 gate_up default; the down GEMM's legal_cfg fallback
# (4, 1, 1) is within 2% of its best, so only gate_up is keyed. An explicit R9K_MOE_CFG1 still wins.
CFG_GATE_UP_BY_N = {320: (1, 8, 1)}


def _legal(cfg: tuple[int, int, int], W) -> tuple[int, int, int]:
    if cfg is CFG_GATE_UP and "R9K_MOE_CFG1" not in os.environ:
        cfg = CFG_GATE_UP_BY_N.get(W.N, cfg)
    return K.legal_cfg(cfg, W.N, W.K, 16 if isinstance(W, K.Nvfp4Experts) else K.GROUP)


FUSED_ACT = os.environ.get("R9K_FUSED_ACT", "1") == "1"
FUSED_SUM = os.environ.get("R9K_FUSED_SUM", "1") == "1"
ZERO_DOWN = os.environ.get("R9K_MOE_ZERO_DOWN", "0") == "1"
# Cold-pass strategy by step width (routed rows = tokens x top-k). Measured 2026-09-18, MTP-3, 270 slots:
#  - few rows (e.g. 4 concurrent = 160 rows): bulk-staging the few cold experts beats latency-bound UVA reads
#    (@4: 182.5 vs 121.0 tok/s);
#  - mid widths (16 concurrent = 640 rows): many cold experts with ~1 row each; UVA reads each once, staging only
#    adds a copy (@16: 195.7 UVA vs 172.5 staged);
#  - prefill chunks: staging avoids re-reading an expert once per 16*MT-row block.
STAGE_MAX_ROWS = int(os.environ.get("R9K_STAGE_MAX_ROWS", "320"))
STAGE_MIN_WIDE = int(os.environ.get("R9K_STAGE_MIN_WIDE", "2048"))


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

    # folded-exponent kernels per GEMM (quant/ct.py decides per tensor at weight prep: layer._r9k_fold)
    fold = (False, False)

    def process_weights_after_loading(self, layer) -> None:
        sup = getattr(super(), "process_weights_after_loading", None)
        if sup is not None:
            sup(layer)
        self.fold = tuple(getattr(layer, "_r9k_fold", (False, False)))

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

        MT = K.pick_mt(numel, global_num_experts)
        blk = K.MOE_BLOCK * MT
        pf_down = K.pick_moe_prefill(MT, K2)      # LDS-tiled kernel for both GEMMs of wide (MT=4) steps
        pf_gate = K.pick_moe_prefill(MT, K1, gate_up=True)
        cache = getattr(self, "r9k_cache", None)
        if cache is None:
            # the fused router (moe/route.py) leaves this call's align tables on the RoutedExperts object
            tabs = route.take(getattr(self, "r9k_layer", None), topk_ids, blk, numel) if expert_map is None else None
            if tabs is None:
                tabs = moe_align_block_size(topk_ids, blk, global_num_experts, expert_map)
            passes = [(K.Mxfp4Experts(w1, self.w1_scale, N1, K1, self.fold[0]),
                       K.Mxfp4Experts(w2, self.w2_scale, N2, K2, self.fold[1]), tabs)]
        else:
            assert expert_map is None, "r9k expert cache: expert parallelism not supported"
            (h1, h2), (c1, c2) = cache.hot(), cache.cold()
            if cache.fused:
                # one launch: LRU manage + align over slots + align over host, then gather misses
                hot_al, cold_al = cache.update_fused(topk_ids, blk)
            else:
                cache.update(topk_ids)          # LRU manage + gather misses host -> VRAM slots
                hot_al = moe_align_block_size(topk_ids, blk, global_num_experts, cache.table)
                cold_al = moe_align_block_size(topk_ids, blk, global_num_experts, cache.map_cold)
            passes = [(h1, h2, hot_al)]
            # If numel <= min(max_distinct, max_inserts), the manager never reads through and never caps
            # inserts, so every routed expert is resident before the GEMMs: the cold pass is provably empty
            # and its two launches can be skipped on the host (no device sync needed; decode is always here).
            if numel > cache.no_cold_limit:
                if cache.stage is not None and (numel <= STAGE_MAX_ROWS or numel >= STAGE_MIN_WIDE):
                    s1, s2, smap = cache.stage_cold(topk_ids)      # bulk-copy cold experts to VRAM once
                    passes.append((s1, s2, moe_align_block_size(topk_ids, blk, global_num_experts, smap)))
                else:
                    passes.append((c1, c2, cold_al))

        xq, xs = K.quant_rows_fp8(hidden_states)
        gate_up = torch.empty((numel, N1), dtype=torch.bfloat16, device=dev)
        for W1, _, (sid, eid, ntpp) in passes:
            K.moe_gemm(xq, xs, W1, gate_up, sid, eid, ntpp, numel, topk, None, *_legal(CFG_GATE_UP, W1),
                       num_experts=global_num_experts, MT=MT, prefill=pf_gate)

        if FUSED_ACT:
            aq, as_ = K.silu_mul_quant_fp8(gate_up)           # one launch: silu*mul + per-row fp8 quant
        else:
            act = torch.empty((numel, N1 // 2), dtype=torch.bfloat16, device=dev)
            self.activation(activation, act, gate_up)
            aq, as_ = K.quant_rows_fp8(act)

        # Every routed row is written by exactly one pass when there is no expert map (the cache's hot and cold
        # tables partition the experts; the plain path covers all of them), so the buffer needs no zeroing:
        # at a 4096-token chunk the fill was 168 MB / 260 us per layer. Rows no pass writes -- other EP ranks --
        # must not feed moe_sum garbage, so expert parallelism (or R9K_MOE_ZERO_DOWN=1) keeps the zeros.
        if expert_map is not None or ZERO_DOWN:
            down = torch.zeros((numel, N2), dtype=torch.bfloat16, device=dev)
        else:
            down = torch.empty((numel, N2), dtype=torch.bfloat16, device=dev)
        tw = topk_weights.reshape(-1).to(torch.float32)
        for _, W2, (sid, eid, ntpp) in passes:
            K.moe_gemm(aq, as_, W2, down, sid, eid, ntpp, numel, 1, tw, *_legal(CFG_DOWN, W2),
                       num_experts=global_num_experts, MT=MT, prefill=pf_down)
        if FUSED_SUM and hasattr(K.lib(), "r9k_moe_sum"):
            shared = fold.shared_output(self, M, N2)      # the runner's shared-expert output, when the fold is on
            K.moe_sum(down.view(M, topk, N2), shared, output)
            fold.mark(self, shared is not None)
        else:
            fold.mark(self, False)
            ops.moe_sum(down.view(M, topk, N2), output)
