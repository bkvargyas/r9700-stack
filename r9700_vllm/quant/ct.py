"""compressed-tensors on gfx1201 through vLLM's official quantization-config registry (no class patching).

`R9kCompressedTensorsConfig` subclasses stock `CompressedTensorsConfig` and is registered under the SAME name
("compressed-tensors") with `register_quantization_config`, which vLLM documents as overriding the built-in. It only
changes what it hands out per layer, and only on ROCm gfx12 with libr9k available:

  * config fix-ups (from_config): format-less fp8 groups under a global `mxfp4-pack-quantized` format get
    `float-quantized` (else stock dispatch looks for MXFP4 schemes and fails); optionally drop the MTP-MLP fp8
    group (R9K_MTP_MLP=mxfp4, default) -- its 640-wide experts cannot shard by 128 at TP2, so the model loader
    re-quantizes them to MXFP4 (see models/qwen4_exp.py).
  * MoE layers with MXFP4 experts -> `R9kMxfp4MoEMethod` (libr9k grouped GEMM + expert cache; stock -> CUDA Marlin).
  * dense MXFP4 schemes -> stock scheme object with its kernel set to `R9700Mxfp4LinearKernel` (stock -> per-call
    dequant emulation).
  * NVFP4 dense / MoE -> converted to MXFP4 at load, then the same libr9k paths (quant/nvfp4.py; stock ROCm has
    only NVFP4 emulation for linears and no NVFP4 MoE backend). R9K_DISABLE=nvfp4 opts out.
  * per-channel fp8 + dynamic per-token activations -> `R9kW8A8Fp8` channel mode (exact, libr9k fp8 GEMM;
    stock ROCm -> torch._scaled_mm). R9K_DISABLE=fp8_channel opts out.
  * opt-in R9K_FP8_BLOCK=block|rowwise: block-fp8 schemes -> `R9kW8A8Fp8` (libr9k split-K fp8 GEMM).
  * opt-in R9K_FP8_LINEARS=<regex>: matching unquantized linears -> `R9kFp8UnquantMethod`.
Everything else is exactly the stock object.
"""
from __future__ import annotations

import os
import re

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4 import (  # noqa: E501
    CompressedTensorsW4A4Mxfp4MoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_mxfp4 import (
    CompressedTensorsW4A4Mxfp4,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (
    CompressedTensorsW4A4Fp4,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
    CompressedTensorsW8A8Fp8,
)

from ..moe.experts import R9700Mxfp4Experts, r9k_available
from ..moe import prep

logger = init_logger("vllm." + __name__)


def _mode(env: str) -> str:
    return os.environ.get(env, "").lower()


def _off(name: str) -> bool:
    """R9K_DISABLE=moe,linear leaves that layer family on the stock method."""
    return name in {s.strip() for s in os.environ.get("R9K_DISABLE", "").split(",") if s.strip()}


# ---------------------------------------------------------------------------------------------------------- MoE
class R9kMxfp4MoEMethod(CompressedTensorsW4A4Mxfp4MoEMethod):
    """Stock CT MXFP4 MoE method with the experts class swapped for libr9k and the weight prep done in place
    (fragment-order permute inside the checkpoint tensors' own storage -- VRAM or UVA host), + expert cache."""

    def __init__(self, moe):
        super().__init__(moe)
        self.experts_cls = R9700Mxfp4Experts
        logger.info_once("r9700: CT MXFP4 MoE -> R9700Mxfp4Experts (libr9k grouped MXFP4xFP8)")

    def process_weights_after_loading(self, layer) -> None:
        from vllm.model_executor.layers.fused_moe.config import mxfp4_w4a16_moe_quant_config
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import make_mxfp4_moe_kernel
        dev = torch.device("cuda", torch.cuda.current_device())
        w13p, w2p = layer.w13_weight_packed, layer.w2_weight_packed
        w13 = prep.permute_in_place(w13p.data)
        w2 = prep.permute_in_place(w2p.data)
        s13 = prep.pack_scales_gpu(layer.w13_weight_scale.data, dev)
        s2 = prep.pack_scales_gpu(layer.w2_weight_scale.data, dev)
        layer.w13_weight = prep.as_param(w13, w13p)
        layer.w2_weight = prep.as_param(w2, w2p)
        delattr(layer, "w13_weight_packed")
        delattr(layer, "w2_weight_packed")
        layer.w13_weight_scale = Parameter(s13, requires_grad=False)
        layer.w2_weight_scale = Parameter(s2, requires_grad=False)
        self.moe_quant_config = mxfp4_w4a16_moe_quant_config(w1_scale=layer.w13_weight_scale,
                                                             w2_scale=layer.w2_weight_scale)
        self.moe_kernel = make_mxfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config, moe_config=self.moe, experts_cls=R9700Mxfp4Experts,
            mxfp4_backend=self.mxfp4_backend, routing_tables=layer._expert_routing_tables())
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)
        prep.maybe_attach_cache(self, layer)


# ------------------------------------------------------------------------------------------------ fp8 block
class R9kW8A8Fp8(CompressedTensorsW8A8Fp8):
    """fp8 linears on libr9k's split-K fp8 GEMMs.
    mode 'channel' (default for per-channel weights + dynamic per-token activations): exact -- the checkpoint's
      e4m3 bytes (fragment order) and per-row scales, r9k_gemm_fp8 (stock ROCm picks torch._scaled_mm, slow at
      decode batch sizes);
    mode 'block' (opt-in R9K_FP8_BLOCK=block): exact block math (same bytes + scales, fragment order);
    mode 'rowwise' (opt-in R9K_FP8_BLOCK=rowwise): block weights requantized per output row (lossier)."""

    r9k_mode = "block"

    def _to_mxfp4(self, layer, w, bs) -> bool:
        """R9K_FP8_TO_MXFP4=<regex|1>: requantize this fp8 linear to MXFP4 (half the weight bytes; GGZ14 serves the
        27B this way by default -- same GSM8K). Exact dequant (per-channel or 128x128 block scales), MSE-searched
        E8M0 per 32, then the dense libr9k MXFP4 kernel. The LM head is never converted."""
        pat = os.environ.get("R9K_FP8_TO_MXFP4", "")
        prefix = getattr(layer, "prefix", "") or ""
        if not pat or "lm_head" in prefix or w.dim() != 2 or w.shape[0] % 16 or w.shape[1] % 32:
            return False
        if pat != "1" and not re.search(pat, prefix):
            return False
        from .nvfp4 import quantize_mxfp4_search
        from ..linear.mxfp4 import R9700Mxfp4LinearKernel
        from vllm.model_executor.kernels.linear.mxfp4.base import MxFp4LinearLayerConfig
        N, Kd = w.shape
        packed = torch.empty((N, Kd // 2), dtype=torch.uint8, device=w.device)
        e8 = torch.empty((N, Kd // 32), dtype=torch.uint8, device=w.device)
        bsf = bs.float()
        for r0 in range(0, N, 4096):
            r1 = min(N, r0 + 4096)
            wf = w[r0:r1].float()
            if bsf.numel() == N or bsf.numel() == 1:
                wf = wf * bsf.reshape(-1, 1)[r0:r1] if bsf.numel() == N else wf * bsf.reshape(())
            else:                                            # 128x128 block scales
                sb = bsf.repeat_interleave(128, 0)[r0:r1].repeat_interleave(128, 1)[:, :Kd]
                wf = wf * sb
            p, s = quantize_mxfp4_search(wf)
            packed[r0:r1].copy_(p)
            e8[r0:r1].copy_(s)
        kern = R9700Mxfp4LinearKernel(MxFp4LinearLayerConfig(activation_quant_key=None))
        layer.weight = Parameter(packed, requires_grad=False)
        layer.weight_scale = Parameter(e8, requires_grad=False)
        layer.input_scale = None
        kern.process_weights_after_loading(layer)
        layer._r9k_mx = kern
        logger.info_once("r9700: fp8 linears requantized to MXFP4 (R9K_FP8_TO_MXFP4=%s)", pat)
        return True

    def process_weights_after_loading(self, layer) -> None:
        from ..kernels import fp8 as F8
        from compressed_tensors.quantization import QuantizationStrategy
        w, bs = layer.weight.data, layer.weight_scale.data
        if w.dtype == torch.float8_e4m3fn and self._to_mxfp4(layer, w, bs):
            return
        if self.r9k_mode == "stock":
            return super().process_weights_after_loading(layer)
        if self.r9k_mode == "channel":
            if (self.strategy != QuantizationStrategy.CHANNEL or w.dim() != 2 or w.shape[0] % 16
                    or w.shape[1] % 16 or w.dtype != torch.float8_e4m3fn):
                return super().process_weights_after_loading(layer)
            N, Kd = w.shape
            from ..utils import note_shape
            note_shape("fp8_channel", N, Kd)
            layer._r9k_fp8 = F8.Fp8Weight(F8.permute_fp8(w.view(torch.uint8)), bs.float().reshape(-1).contiguous(),
                                          N, Kd)
            dev = w.device
            layer.weight = Parameter(torch.empty((0,), dtype=w.dtype, device=dev), requires_grad=False)
            layer.weight_scale = Parameter(torch.empty((0,), dtype=torch.float32, device=dev), requires_grad=False)
            layer.input_scale = None
            logger.info_once("r9700: per-channel fp8 linears -> libr9k fp8 GEMM (exact weights, per-token fp8 act)")
            return
        blk = tuple(getattr(layer, "weight_block_size", None) or (128, 128))
        if (self.strategy != QuantizationStrategy.BLOCK or w.dim() != 2 or w.shape[0] % 16 or w.shape[1] % 16
                or (self.r9k_mode == "block" and (blk != (128, 128) or w.shape[1] % 128))):
            return super().process_weights_after_loading(layer)
        from ..utils import note_shape
        note_shape("fp8_" + self.r9k_mode, w.shape[0], w.shape[1])
        if self.r9k_mode == "block":
            layer._r9k_fp8b = (F8.permute_fp8(w.view(torch.uint8)), bs.float().contiguous(), w.shape[0], w.shape[1])
        else:
            from ..linear.fp8_block import requant_rowwise
            layer._r9k_fp8 = requant_rowwise(w.view(torch.float8_e4m3fn), bs, blk)
        dev = w.device
        layer.weight = Parameter(torch.empty((0,), dtype=w.dtype, device=dev), requires_grad=False)
        layer.weight_scale = Parameter(torch.empty((0,), dtype=torch.float32, device=dev), requires_grad=False)
        layer.input_scale = None
        logger.info_once("r9700: block-fp8 linears -> libr9k split-K fp8 GEMM (%s)", self.r9k_mode)

    def apply_weights(self, layer, x, bias=None):
        from .. import ops
        mx = getattr(layer, "_r9k_mx", None)
        if mx is not None:
            return mx.apply_weights(layer, x, bias)
        Wb = getattr(layer, "_r9k_fp8b", None)
        W = getattr(layer, "_r9k_fp8", None)
        if Wb is None and W is None:
            return super().apply_weights(layer, x, bias)
        if not isinstance(x, torch.Tensor):
            raise RuntimeError("r9700 R9K_FP8_BLOCK: got a pre-quantized activation (quant-fusion pass on)")
        out = ops.fp8_block_linear(x, *Wb) if Wb is not None else ops.fp8_linear(x, W)
        return out + bias if bias is not None else out


# ------------------------------------------------------------------------------------------ fp8 unquantized
class R9kFp8UnquantMethod(UnquantizedLinearMethod):
    """bf16 linear served as per-row fp8 on libr9k (opt-in via R9K_FP8_LINEARS regex)."""

    def process_weights_after_loading(self, layer) -> None:
        super().process_weights_after_loading(layer)
        w = getattr(layer, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 2 and w.is_cuda and w.shape[0] % 16 == 0 \
                and w.shape[1] % 16 == 0:
            from ..kernels import fp8 as F8
            layer._r9k_fp8 = F8.quantize_rows_fp8(w.data)
            if os.environ.get("R9K_FP8_LINEARS_FREE", "0") == "1":
                layer.weight.data = torch.empty((0, w.shape[1]), dtype=w.dtype, device=w.device)

    def apply(self, layer, x, bias=None):
        W = getattr(layer, "_r9k_fp8", None)
        if W is None:
            return super().apply(layer, x, bias)
        from .. import ops
        out = ops.fp8_linear(x, W).to(x.dtype)
        return out + bias if bias is not None else out


# ------------------------------------------------------------------------------------------------- config
@register_quantization_config("compressed-tensors")
class R9kCompressedTensorsConfig(CompressedTensorsConfig):

    @classmethod
    def from_config(cls, config):
        if os.environ.get("R9K_MTP_MLP", "mxfp4") == "mxfp4":
            groups = config.get("config_groups") or {}
            for name in list(groups):
                tg = groups[name].get("targets") or []
                if tg and all(str(t).startswith("re:mtp") and "mlp" in str(t) for t in tg):
                    del groups[name]
                    logger.info_once("r9700: CT group %s (MTP MLP, fp8 block-128) dropped; re-quantized to "
                                     "MXFP4 at load", name)
        gfmt = config.get("format")
        if gfmt and gfmt != "float-quantized":
            for name, grp in (config.get("config_groups") or {}).items():
                w = grp.get("weights") or {}
                if grp.get("format") is None and w.get("num_bits") == 8 and w.get("type") == "float":
                    grp["format"] = "float-quantized"
                    logger.info_once("r9700: CT group %s: fp8 weights under global format %r -> float-quantized",
                                     name, gfmt)
        return super().from_config(config)

    def get_scheme(self, layer, layer_name=None):
        scheme = super().get_scheme(layer=layer, layer_name=layer_name)
        if scheme is None or not r9k_available():
            return scheme
        if isinstance(scheme, CompressedTensorsW4A4Mxfp4) and not _off("linear"):
            from ..linear.mxfp4 import R9700Mxfp4LinearKernel
            from vllm.model_executor.kernels.linear.mxfp4.base import MxFp4LinearLayerConfig
            cfg = MxFp4LinearLayerConfig(activation_quant_key=None)
            scheme.kernel = R9700Mxfp4LinearKernel(cfg)       # instance attribute, stock class untouched
            logger.info_once("r9700: dense MXFP4 linears -> libr9k (fp8 activations)")
        elif isinstance(scheme, CompressedTensorsW4A4Fp4) and not _off("nvfp4"):
            from ..linear.mxfp4 import R9700Mxfp4LinearKernel
            from vllm.model_executor.kernels.linear.mxfp4.base import MxFp4LinearLayerConfig
            from .nvfp4 import make_dense_scheme_cls, make_native_dense_scheme_cls
            if (_mode("R9K_NVFP4") or "native") == "native":
                scheme.__class__ = _cached(make_native_dense_scheme_cls, CompressedTensorsW4A4Fp4)
            else:
                scheme.__class__ = _cached(make_dense_scheme_cls, CompressedTensorsW4A4Fp4)   # this instance only
            scheme.kernel = R9700Mxfp4LinearKernel(MxFp4LinearLayerConfig(activation_quant_key=None))
        elif isinstance(scheme, CompressedTensorsW8A8Fp8) and not _off("fp8_channel") \
                and _strategy(scheme) == "channel" and not scheme.is_static_input_scheme:
            scheme.__class__ = R9kW8A8Fp8                        # this instance only
            scheme.r9k_mode = "channel"
        elif isinstance(scheme, CompressedTensorsW8A8Fp8) and _mode("R9K_FP8_BLOCK") in ("block", "rowwise"):
            scheme.__class__ = R9kW8A8Fp8                        # this instance only
            scheme.r9k_mode = _mode("R9K_FP8_BLOCK")
        elif isinstance(scheme, CompressedTensorsW8A8Fp8) and os.environ.get("R9K_FP8_TO_MXFP4"):
            scheme.__class__ = R9kW8A8Fp8                        # requant only; other layers fall back to stock
            scheme.r9k_mode = "stock"
        return scheme

    def get_quant_method(self, layer, prefix):
        if r9k_available() and not _off("nvfp4") and self._is_nvfp4_moe(layer, prefix):
            # before stock dispatch: stock has no NVFP4 MoE backend on ROCm and raises while selecting one
            from .nvfp4 import make_moe_method_cls
            return _cached(make_moe_method_cls, R9kMxfp4MoEMethod)(layer.moe_config)
        m = super().get_quant_method(layer, prefix)
        if not r9k_available():
            return m
        if isinstance(m, CompressedTensorsW4A4Mxfp4MoEMethod) and not isinstance(m, R9kMxfp4MoEMethod) \
                and not getattr(m, "use_cutlass_mxfp4", False) and not _off("moe"):
            return R9kMxfp4MoEMethod(m.moe)
        pat = os.environ.get("R9K_FP8_LINEARS", "")
        if pat and type(m) is UnquantizedLinearMethod and re.search(pat, prefix or ""):
            return R9kFp8UnquantMethod()
        return m

    def _is_nvfp4_moe(self, layer, prefix) -> bool:
        from vllm.model_executor.layers.fused_moe import RoutedExperts
        if not isinstance(layer, RoutedExperts):
            return False
        self._add_fused_moe_to_target_scheme_map()
        sd = self.get_scheme_dict(layer, (prefix or "") + ".0.gate_proj")
        return bool(sd) and self._is_nvfp4_format(sd.get("weights"))


def _strategy(scheme) -> str:
    st = getattr(scheme, "strategy", "")
    return str(getattr(st, "value", st)).lower()


_CLS_CACHE: dict = {}


def _cached(factory, base):
    """One generated subclass per (factory, base), so every layer shares the same class object."""
    key = (factory, base)
    if key not in _CLS_CACHE:
        _CLS_CACHE[key] = factory(base)
    return _CLS_CACHE[key]
