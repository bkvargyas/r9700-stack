"""Dense MXFP4 (weight-only, OCP e2m1 + E8M0/32) linear kernel for gfx1201 on stock vLLM.

Stock vLLM on ROCm gfx12 falls back to EmulationMxfp4LinearKernel: it dequantizes the whole weight to bf16 on
every call (Flash-Next's shared experts hit this 48x per step). This kernel reuses libr9k's grouped
MXFP4 x FP8 GEMM with a single expert: activations are quantized per row to e4m3, the routing tables are the
identity (sorted ids 0..Mpad-1, all blocks -> expert 0), so any M works and the weight stays 4-bit.
Handed to stock CT MXFP4 schemes by R9kCompressedTensorsConfig.get_scheme (quant/ct.py); serves stock's W4A4
(kMxfp4Dynamic) requests with fp8 activations (see can_implement).
"""
from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mxfp4.base import MxFp4LinearKernel, MxFp4LinearLayerConfig

from ..kernels import moe as K

logger = init_logger("vllm." + __name__)


class R9700Mxfp4LinearKernel(MxFp4LinearKernel):
    def __init__(self, config: MxFp4LinearLayerConfig) -> None:
        super().__init__(config)
        self._ids: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def input_quant_key(self):
        return None      # quantizes its own activations (fp8 per row); no pre-quantized-input fusion

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        from ..moe.experts import r9k_available
        return (True, None) if r9k_available() else (False, "needs ROCm gfx12 + libr9k")

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        # Stock CompressedTensorsW4A4Mxfp4 always asks for kMxfp4Dynamic activations, even for checkpoints that
        # declare input_activations=null (weight-only, e.g. Flash-Next's shared experts). We run fp8 (e4m3,
        # per-row) activations instead: more precise than MXFP4 activations and what the weights were tuned for.
        from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Dynamic
        if config.activation_quant_key not in (None, kMxfp4Dynamic):
            return False, "weight-only / MXFP4-activation requests only"
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

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        from ..ops import mxfp4_linear
        N, Kd = layer._r9k_nk
        out = mxfp4_linear(x, layer.weight, layer.weight_scale, N, Kd).to(x.dtype)
        return out + bias if bias is not None else out
