"""Dense MXFP4 (weight-only, OCP e2m1 + E8M0/32) linear kernel for gfx1201 on stock vLLM.

Stock vLLM on ROCm gfx12 falls back to EmulationMxfp4LinearKernel: it dequantizes the whole weight to bf16 on
every call (Flash-Next's shared experts hit this 48x per step). This kernel reuses libr9k's grouped
MXFP4 x FP8 GEMM with a single expert: activations are quantized per row to e4m3, the routing tables are the
identity (sorted ids 0..Mpad-1, all blocks -> expert 0), so any M works and the weight stays 4-bit.
Inserted at the front of vLLM's ROCm MXFP4 kernel list; declines anything but weight-only on gfx12.
"""
from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mxfp4.base import MxFp4LinearKernel, MxFp4LinearLayerConfig

from ..kernels import moe as K

logger = init_logger(__name__)


class R9700Mxfp4LinearKernel(MxFp4LinearKernel):
    CFG = (2, 4, 2)

    def __init__(self, config: MxFp4LinearLayerConfig) -> None:
        super().__init__(config)
        self._ids: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        from ..moe.experts import r9k_available
        return (True, None) if r9k_available() else (False, "needs ROCm gfx12 + libr9k")

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        if config.activation_quant_key is not None:
            return False, "weight-only MXFP4 only"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w, s = layer.weight.data, layer.weight_scale.data
        N, Kh = w.shape
        if N % 16 or (2 * Kh) % 32:
            raise ValueError(f"R9700Mxfp4LinearKernel: unsupported shape N={N} K={2 * Kh}")
        layer.weight = Parameter(K.permute_fragments(w.view(torch.uint8)[None]), requires_grad=False)
        layer.weight_scale = Parameter(K.pack_scales(s.view(torch.uint8)[None]), requires_grad=False)
        layer._r9k_nk = (N, 2 * Kh)
        logger.info_once("r9700: dense MXFP4 linears on libr9k (weight-only, fp8 activations)")

    def _tables(self, M: int, dev):
        mpad = (M + K.MOE_BLOCK - 1) // K.MOE_BLOCK * K.MOE_BLOCK
        t = self._ids.get(mpad)
        if t is None:
            t = (torch.arange(mpad, dtype=torch.int32, device=dev),
                 torch.zeros(mpad // K.MOE_BLOCK, dtype=torch.int32, device=dev),
                 torch.full((1,), mpad, dtype=torch.int32, device=dev))
            self._ids[mpad] = t
        return t

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        N, Kd = layer._r9k_nk
        lead = x.shape[:-1]
        x2 = x.reshape(-1, Kd)
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        x2 = x2.contiguous()
        M = x2.shape[0]
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
        if M:
            xq, xs = K.quant_rows_fp8(x2)
            sid, eid, ntpp = self._tables(M, x.device)
            W = K.Mxfp4Experts(layer.weight, layer.weight_scale, N, Kd)
            K.moe_gemm(xq, xs, W, out, sid, eid, ntpp, M, 1, None, *self.CFG, num_experts=1)
        if bias is not None:
            out = out + bias
        return out.reshape(*lead, N).to(x.dtype)


_PATCHED = False


def patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from vllm.model_executor.kernels import linear as L
        from vllm.platforms import PlatformEnum
        lst = L._POSSIBLE_MXFP4_KERNELS.setdefault(PlatformEnum.ROCM, [])
        if R9700Mxfp4LinearKernel not in lst:
            lst.insert(0, R9700Mxfp4LinearKernel)
    except Exception as e:
        logger.warning("r9700: MXFP4 linear kernel not registered (%s)", e)
        return False
    _PATCHED = True
    return True
