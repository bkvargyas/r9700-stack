"""NVFP4 checkpoints on gfx1201: convert to MXFP4 while loading, then serve with the libr9k MXFP4 kernels.

Stock vLLM on ROCm has only an emulation kernel for NVFP4 linears and no NVFP4 MoE backend. gfx1201 has no FP4
hardware either way; our kernels unpack E2M1 into FP8 WMMA with a power-of-two (E8M0) scale per 32 values folded
into the exponent. NVFP4 = the same E2M1 values with an FP8-E4M3 scale per 16 values and an FP32 global scale
per tensor (per partition for fused qkv / gate_up). Conversion per 32-group:

  exact dequant   w = e2m1 * s16_e4m3 / global      (fp32, global per partition / per expert shard)
  requant         E8M0 exponent from the group max, then round-to-nearest E2M1 -- with a 2-candidate search:
                  e = ceil(log2(amax/6)) (no clipping) vs e-1 (clips the top value(s), finer grid for the rest),
                  keeping whichever has the lower squared error for that group.

This is lossy (two 16-groups share one power-of-two scale, values are re-rounded) -- roughly the error of an
RTN-MXFP4 quantization of the original weights, not better. Gate it with bench/quality.py per checkpoint.
The NVFP4 activation scales (input_global_scale, W4A4 mode) are dropped: activations run in fp8 per row, as for
MXFP4 checkpoints. Disable with R9K_DISABLE=nvfp4 (stock emulation for linears; NVFP4 MoE then fails on ROCm).
"""
from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])      # E2M1 decision thresholds on |x|


def dequant_nvfp4(packed: torch.Tensor, scale16: torch.Tensor, row_div: torch.Tensor) -> torch.Tensor:
    """packed [N, K/2] u8 (low nibble = even element), scale16 [N, K/16] e4m3, row_div [N] fp32 (CT stores the
    global scale as a divisor) -> fp32 [N, K]."""
    N, Kh = packed.shape
    lut = _E2M1.to(packed.device)
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(N, Kh * 2).long()
    v = lut[codes].reshape(N, -1, 16) * scale16.float().unsqueeze(-1)
    return v.reshape(N, Kh * 2) / row_div.float().unsqueeze(-1)


def _encode(v: torch.Tensor, mid: torch.Tensor) -> torch.Tensor:
    """fp32 values already divided by the group scale -> E2M1 codes (u8, sign in bit 3), saturating at 6."""
    return torch.bucketize(v.abs(), mid).to(torch.uint8) | ((v < 0).to(torch.uint8) << 3)


def quantize_mxfp4_search(w: torch.Tensor):
    """fp32 [N, K] (K % 32 == 0) -> (packed [N, K/2] u8, e8m0 [N, K/32] u8), per-group best of {e, e-1}."""
    N, K = w.shape
    mid, lut = _MID.to(w.device), _E2M1.to(w.device)
    x = w.float().reshape(N, K // 32, 32)
    amax = x.abs().amax(-1, keepdim=True).clamp_min(2.0 ** -126)
    e0 = torch.ceil(torch.log2(amax / 6.0)).clamp(-126, 127)
    best_c, best_e, best_err = None, None, None
    for e in (e0, (e0 - 1).clamp(-127, 127)):
        s = torch.exp2(e)
        c = _encode(x / s, mid)
        err = ((lut[c.long()] * s - x) ** 2).sum(-1, keepdim=True)
        if best_c is None:
            best_c, best_e, best_err = c, e, err
        else:
            take = err < best_err
            best_c = torch.where(take, c, best_c)
            best_e = torch.where(take, e, best_e)
            best_err = torch.minimum(err, best_err)
    c = best_c.reshape(N, K)
    return c[:, 0::2] | (c[:, 1::2] << 4), (best_e.squeeze(-1) + 127).to(torch.uint8)


def nvfp4_to_mxfp4(packed, scale16, row_div, rows_per_chunk: int = 2048):
    """Chunked on the GPU. Returns (packed [N, K/2] u8 on packed.device, e8m0 [N, K/32] u8 on packed.device)."""
    dev = torch.device("cuda", torch.cuda.current_device())
    N, Kh = packed.shape
    out_p = torch.empty_like(packed)
    out_s = torch.empty((N, Kh * 2 // 32), dtype=torch.uint8, device=packed.device)
    for r0 in range(0, N, rows_per_chunk):
        r1 = min(N, r0 + rows_per_chunk)
        w = dequant_nvfp4(packed[r0:r1].to(dev), scale16[r0:r1].to(dev), row_div[r0:r1].to(dev))
        p, s = quantize_mxfp4_search(w)
        out_p[r0:r1].copy_(p)
        out_s[r0:r1].copy_(s)
    return out_p, out_s


def _row_div(global_scale: torch.Tensor, widths) -> torch.Tensor:
    g = global_scale.detach().float().reshape(-1)
    if g.numel() == 1:
        return g.expand(sum(widths)).contiguous()
    return torch.repeat_interleave(g, torch.tensor(widths, device=g.device))


# ------------------------------------------------------------------------------------------------ dense
def make_dense_scheme_cls(base):
    """Subclass of stock CompressedTensorsW4A4Fp4: stock create_weights (so the checkpoint loads as-is), MXFP4
    conversion + libr9k kernel after loading."""

    class R9kNvfp4AsMxfp4(base):

        def process_weights_after_loading(self, layer) -> None:
            widths = list(layer.logical_widths)
            div = _row_div(layer.weight_global_scale.data, widths)
            p, s = nvfp4_to_mxfp4(layer.weight_packed.data, layer.weight_scale.data, div)
            del layer.weight_packed
            del layer.weight_global_scale
            if hasattr(layer, "input_global_scale"):
                del layer.input_global_scale
            layer.weight = Parameter(p, requires_grad=False)
            layer.weight_scale = Parameter(s, requires_grad=False)
            logger.info_once("r9700: NVFP4 linears converted to MXFP4 at load -> libr9k (fp8 activations)")
            self.kernel.process_weights_after_loading(layer)

        def apply_weights(self, layer, x, bias=None):
            return self.kernel.apply_weights(layer, x, bias)

    return R9kNvfp4AsMxfp4


# -------------------------------------------------------------------------------------------------- MoE
def make_moe_method_cls(mxfp4_method_cls):
    """NVFP4 checkpoint layout at create time (stock CompressedTensorsW4A4Nvfp4MoEMethod.create_weights), converted
    per expert IN PLACE (MXFP4 codes are the same size) to the MXFP4 layout R9kMxfp4MoEMethod consumes. Works on
    UVA-offloaded experts (the per-expert slice is read to VRAM, converted, written back)."""
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (  # noqa: E501
        CompressedTensorsW4A4Nvfp4MoEMethod as _Nv,
    )

    class R9kNvfp4MoEMethod(mxfp4_method_cls):

        def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype,
                           **extra_weight_attrs):
            saved = self.__dict__.get("group_size", None)
            self.group_size = 16
            try:
                _Nv.create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition,
                                   params_dtype, **extra_weight_attrs)
            finally:
                if saved is None:
                    self.__dict__.pop("group_size", None)
                else:
                    self.group_size = saved

        @staticmethod
        def _convert(packed_p, scale_p, row_div):
            E = packed_p.shape[0]
            s_out = torch.empty((E, packed_p.shape[1], packed_p.shape[2] * 2 // 32), dtype=torch.uint8,
                                device=scale_p.device)
            for e in range(E):
                p, s = nvfp4_to_mxfp4(packed_p.data[e], scale_p.data[e], row_div[e])
                packed_p.data[e].copy_(p)
                s_out[e].copy_(s)
            return s_out

        def process_weights_after_loading(self, layer) -> None:
            I2 = layer.w13_weight_packed.shape[1]
            g13 = layer.w13_weight_global_scale.data.float()                  # [E, shards]
            div13 = g13.repeat_interleave(I2 // g13.shape[1], dim=1)          # [E, 2I]
            g2 = layer.w2_weight_global_scale.data.float()                    # [E]
            div2 = g2[:, None].expand(-1, layer.w2_weight_packed.shape[1])
            s13 = self._convert(layer.w13_weight_packed, layer.w13_weight_scale, div13)
            s2 = self._convert(layer.w2_weight_packed, layer.w2_weight_scale, div2)
            for n in ("w13_weight_global_scale", "w2_weight_global_scale", "w13_input_global_scale",
                      "w2_input_global_scale"):
                if hasattr(layer, n):
                    delattr(layer, n)
            layer.w13_weight_scale = Parameter(s13, requires_grad=False)
            layer.w2_weight_scale = Parameter(s2, requires_grad=False)
            logger.info_once("r9700: NVFP4 MoE experts converted to MXFP4 at load -> libr9k")
            super().process_weights_after_loading(layer)

    return R9kNvfp4MoEMethod
