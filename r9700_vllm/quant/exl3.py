"""EXL3 (ExLlamaV3 trellis-quantized) weights on stock vLLM: the ``exl3`` quantization config and the linear method.

Serves published EXL3 packs (first target: LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) through exllamav3's compiled
extension (``exllamav3_ext``, the gfx1201 wheel) -- its exl3_gemm / reconstruct kernels, our glue. Design: notes/exl3-plan.md.
Per quantized linear the checkpoint holds ``<name>.trellis`` int16 [in/16, out/16, 16*K], ``.suh`` f16 [in], ``.svh`` f16 [out]
and the codebook marker ``.mul1`` (or ``.mcg``) int32 []. Tiles decode independently; Hadamard blocks are 128 wide on both
sides, so every tensor-parallel split lands on a 128-multiple.

Package L owns this file (see the brief). Public names other packages rely on: Exl3Config, Exl3LinearMethod, ext(), k_of(),
reconstruct_f16(), exl3_gemm_f16().
"""
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase

logger = init_logger("vllm.r9700_vllm.exl3")

_EXT = None


def ext():
    """exllamav3's compiled extension, imported on first use (the serving image carries the gfx1201 wheel)."""
    global _EXT
    if _EXT is None:
        import exllamav3_ext  # noqa: F401
        _EXT = exllamav3_ext
    return _EXT


def k_of(trellis: torch.Tensor) -> int:
    """Bits per weight from the trellis tile width (16*K int16 per 16x16 tile; half-integer K is not used by this pack)."""
    w = trellis.shape[-1]
    assert w % 16 == 0, f"exl3: half-integer bitrate (tile width {w}) not supported"
    return w // 16


def reconstruct_f16(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, K: int, mul1: bool,
                    mcg: bool = False) -> torch.Tensor:
    """Exact dequantization to f16 [in, out] (W^T), the reference every kernel path is checked against. TODO(L)."""
    raise NotImplementedError("package L")


def exl3_gemm_f16(a_f16: torch.Tensor, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, K: int, mul1: bool,
                  mcg: bool = False, out_f32: bool = False) -> torch.Tensor:
    """One trellis GEMM: a_f16 [m, in] contiguous -> [m, out] (f16, or f32 when out_f32). TODO(L)."""
    raise NotImplementedError("package L")


@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):
    """``quantization_config = {"quant_method": "exl3", ...}`` from the pack's config.json."""

    def __init__(self, avg_bits: float, head_bits: int, codebook: str, version: str, out_scales: str = "always",
                 k_map: dict | None = None):
        super().__init__()
        self.avg_bits, self.head_bits, self.codebook, self.version, self.out_scales = avg_bits, head_bits, codebook, version, out_scales
        self.k_map = k_map or {}

    def get_name(self) -> str:  # type: ignore[override]
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        return cls(avg_bits=float(config.get("avg_bits", 0)), head_bits=int(config.get("head_bits", 0)),
                   codebook=str(config.get("codebook", "mul1")), version=str(config.get("version", "")),
                   out_scales=str(config.get("out_scales", "always")))

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        # TODO(L): ParallelLMHead -> dequantize-at-load bf16 head; VocabParallelEmbedding -> UnquantizedEmbeddingMethod;
        # RoutedExperts -> exl3_moe.Exl3MoEMethod; LinearBase -> Exl3LinearMethod. Imports stay lazy (plugin loads everywhere).
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod, VocabParallelEmbedding
        if isinstance(layer, RoutedExperts):
            from .exl3_moe import Exl3MoEMethod
            return Exl3MoEMethod(layer.moe_config, self)
        if isinstance(layer, VocabParallelEmbedding):
            return UnquantizedEmbeddingMethod()
        if isinstance(layer, LinearBase):
            return Exl3LinearMethod(self)
        return None


class Exl3LinearMethod(LinearMethodBase):
    """LinearBase layers with one EXL3 matrix per shard (vLLM's stacked layers fuse matrices whose ``suh`` differ)."""

    def __init__(self, quant_config: Exl3Config):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int],
                       input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs):
        raise NotImplementedError("package L")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError("package L")

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError("package L")
