"""The MoE block's shared expert in four launches (Flash-Next: `mlp.shared_expert`, a Qwen3NextMLP with an
expert gate, 48 + MTP calls per step).

Stock runs it as gate_up (our dense MXFP4 linear: row quant + GEMM), SiluAndMul, down (quant + GEMM), the
expert gate `F.linear` (bf16 wvSplitK), `sigmoid`, `mul`: eight launches, ~1.7 ms of eager GPU time per step
and as many HIP-graph nodes. Here `torch.ops.r9700.shared_expert` runs
    quant_rows_fp8_gate (row quant + the gate dot + sigmoid)  ->  gate_up GEMM  ->  silu_mul_quant_fp8  ->
    down GEMM with sigmoid(gate) folded into its per-row epilogue (the routed MoE's router-weight fold)
on the same libr9k kernels, at every M (large M takes the same dense prefill / A-tiled configs as the linear).
Numerics: the gate value is stock's (bf16 GEMM output, bf16 sigmoid); the down output is rounded once,
bf16(acc * g), where stock rounds twice, bf16(bf16(acc) * g). R9K_SHARED_EXPERT=stock keeps vLLM's forward.
"""
from __future__ import annotations

import os
import types

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from ..kernels import moe as KM
from ..ops import _LIB, mxfp4_gemm_q

logger = init_logger("vllm." + __name__)
_DONE = False


def available() -> bool:
    try:
        return hasattr(KM.lib(), "r9k_quant_rows_fp8_gate")
    except Exception:
        return False


def shared_expert(x: torch.Tensor, wq1: torch.Tensor, wsr1: torch.Tensor, wq2: torch.Tensor, wsr2: torch.Tensor,
                  wg: torch.Tensor, N1: int, K1: int, N2: int, K2: int, fold1: bool, fold2: bool) -> torch.Tensor:
    """x [M, K1] bf16 -> [M, N2] bf16 = (down(silu(gate) * up) * sigmoid(x . wg)); wq/wsr as the dense MXFP4
    linear stores them (fragment order + packed scales), wg [K1] bf16."""
    M = x.shape[0]
    out = torch.empty((M, N2), dtype=torch.bfloat16, device=x.device)
    if M == 0:
        return out
    cfg1 = KM.pick_cfg(N1, K1, M=M, kind="mxfp4", fold=fold1)
    cfg2 = KM.pick_cfg(N2, K2, M=M, kind="mxfp4", fold=fold2)
    W1 = KM.Mxfp4Experts(wq1, wsr1, N1, K1, fold1)
    W2 = KM.Mxfp4Experts(wq2, wsr2, N2, K2, fold2)
    q, s, g = KM.quant_rows_fp8_gate(x, wg, tiled=KM.is_atiled_cfg(cfg1))
    gate_up = torch.empty((M, N1), dtype=torch.bfloat16, device=x.device)
    mxfp4_gemm_q(q, s, W1, gate_up, M, cfg1)
    if KM.is_atiled_cfg(cfg2):                       # the fused activation writes row-major only
        act = torch.empty((M, N1 // 2), dtype=torch.bfloat16, device=x.device)
        torch.ops._C.silu_and_mul(act, gate_up)
        aq, as_ = KM.quant_rows_fp8(act, tiled=True)
    else:
        aq, as_ = KM.silu_mul_quant_fp8(gate_up)
    mxfp4_gemm_q(aq, as_, W2, out, M, cfg2, topk_w=g)
    return out


def _shared_expert_fake(x: torch.Tensor, wq1: torch.Tensor, wsr1: torch.Tensor, wq2: torch.Tensor,
                        wsr2: torch.Tensor, wg: torch.Tensor, N1: int, K1: int, N2: int, K2: int, fold1: bool,
                        fold2: bool) -> torch.Tensor:
    return x.new_empty((x.shape[0], N2))


def register() -> None:
    global _DONE
    if _DONE:
        return
    direct_register_custom_op("shared_expert", shared_expert, mutates_args=[], fake_impl=_shared_expert_fake,
                              target_lib=_LIB)
    _DONE = True


def _forward(self, x: torch.Tensor) -> torch.Tensor:
    gu, dn = self.gate_up_proj, self.down_proj
    if getattr(gu, "_r9k_nk", None) is None or getattr(dn, "_r9k_nk", None) is None or x.dim() != 2 \
            or x.dtype != torch.bfloat16:
        return self._r9k_stock_forward(x)           # not on our dense kernel (weights not prepared / other format)
    return torch.ops.r9700.shared_expert(x.contiguous(), gu.weight, gu.weight_scale, dn.weight, dn.weight_scale,
                                         self.expert_gate.weight.view(-1), *gu._r9k_nk, *dn._r9k_nk,
                                         bool(getattr(gu, "_r9k_fold", False)), bool(getattr(dn, "_r9k_fold", False)))


def install(model: torch.nn.Module) -> int:
    """Bind the fused forward on every gated shared-expert MLP under `model` whose expert gate is a plain bf16
    linear (the projections are checked at call time, after weight preparation). R9K_SHARED_EXPERT=stock skips."""
    if os.environ.get("R9K_SHARED_EXPERT", "r9k") != "r9k" or not available():
        return 0
    register()
    n = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ not in ("Qwen3NextMLP", "Qwen2MoeMLP") or getattr(mod, "_r9k_shared", False):
            continue
        eg = getattr(mod, "expert_gate", None)
        w = getattr(eg, "weight", None) if eg is not None else None
        if not (isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16 and w.dim() == 2 and w.shape[0] == 1
                and getattr(eg, "bias", None) is None
                and type(getattr(eg, "quant_method", None)).__name__ == "UnquantizedLinearMethod"
                and hasattr(mod, "gate_up_proj") and hasattr(mod, "down_proj")):
            if eg is not None:
                logger.warning_once("r9700: fused shared expert skipped for %s", name)
            continue
        mod._r9k_stock_forward = mod.forward
        mod.forward = types.MethodType(_forward, mod)
        mod._r9k_shared = True
        n += 1
    if n:
        logger.info("r9700: fused shared expert (quant+gate, gate_up, silu_mul_quant, down*gate) installed on %d "
                    "modules", n)
    return n
