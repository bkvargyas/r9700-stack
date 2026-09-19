"""DFlash / DFlash2 drafters with fp8-quantized attention, on stock vLLM (registered over the stock architectures).

Stock `DFlashQwen3Model._build_context_kv_buffers` concatenates the raw `qkv_proj.weight` K/V slices into one fused
bf16 projection and runs it with `F.linear`, which assumes an unquantized drafter. With an fp8 drafter (e.g.
tcclaviger/Qwen3.8-27B-DFlash2-FP8) the fused weight is fp8 (and possibly in a kernel-specific layout), so the
precompute fails ("expected mat1 and mat2 to have the same dtype"). After stock loading, this subclass rebuilds
`_fused_kv_weight` by applying each layer's own qkv_proj quant method to an identity matrix -- exactly the
dequantized weight, independent of the fp8 method's storage layout (identity rows quantize exactly under per-token
fp8). Only the small context-KV projection is dequantized; every other drafter linear stays fp8.
Upstream candidate: dequantize in `_build_context_kv_buffers` when qkv_proj is quantized.
"""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM

logger = init_logger("vllm." + __name__)


def _dequant_fused_kv(model) -> None:
    w = getattr(model, "_fused_kv_weight", None)
    if w is None or w.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return
    parts = []
    for layer in model.layers:
        a = layer.self_attn
        qkv = a.qkv_proj
        K = qkv.input_size_per_partition
        eye = torch.eye(K, dtype=torch.bfloat16, device=next(qkv.parameters()).device)
        full = qkv.quant_method.apply(qkv, eye, None)            # [K, out] = dequant(W)^T, bias excluded
        parts.append(full[:, a.q_size:].t().contiguous())
    model._fused_kv_weight = torch.cat(parts, dim=0).to(torch.bfloat16)
    logger.info_once("r9700: DFlash drafter context-KV projection dequantized from %s (%s)", w.dtype,
                     tuple(model._fused_kv_weight.shape))


def _dequant_weight(lin, chunk: int = 2048) -> torch.Tensor:
    """dequant(W) [N, K] bf16 through the layer's own quant method (identity-row inputs)."""
    K = lin.input_size_per_partition
    dev = next(lin.parameters()).device
    cols = []
    for k0 in range(0, K, chunk):
        k1 = min(K, k0 + chunk)
        eye = torch.zeros((k1 - k0, K), dtype=torch.bfloat16, device=dev)
        eye[torch.arange(k1 - k0, device=dev), torch.arange(k0, k1, device=dev)] = 1
        cols.append(lin.quant_method.apply(lin, eye, None))             # [k1-k0, N] = W[:, k0:k1]^T
    return torch.cat(cols, dim=0).t().contiguous()


class _MxMethod:
    """quant_method stand-in for drafter linears requantized to MXFP4 (R9K_DRAFT_W4=1)."""

    def __init__(self, kern):
        self.kern = kern

    def apply(self, layer, x, bias=None):
        return self.kern.apply_weights(layer, x, bias)

    def process_weights_after_loading(self, layer):
        pass


class _BlockMethod:
    """quant_method stand-in for swapped block-fp8 linears (LinearBase.forward only calls .apply)."""

    def apply(self, layer, x, bias=None):
        from .. import ops
        out = ops.fp8_block_linear(x, *layer._r9k_fp8b).to(x.dtype)
        return out + bias if bias is not None else out

    def process_weights_after_loading(self, layer):
        pass


def swap_fp8_linears(root) -> int:
    """Stock Fp8LinearMethod linears (quant_method 'fp8' checkpoints, dynamic activations) -> libr9k per-row fp8
    GEMM with the checkpoint's exact fp8 values: bytes recovered as round_e4m3(dequant(W) / s) with the layer's own
    (per-tensor or per-row) scale -- exact because bf16 dequant error << the e4m3 step. Verified per layer; a layer
    that does not round-trip, has static activation scales or an unsupported shape stays on stock."""
    from vllm.model_executor.layers.linear import LinearBase
    from ..kernels import fp8 as F8
    from ..quant.ct import R9kFp8UnquantMethod
    n = 0
    for name, lin in root.named_modules():
        qm = getattr(lin, "quant_method", None)
        if not isinstance(lin, LinearBase) or type(qm).__name__ != "Fp8LinearMethod":
            continue
        if getattr(lin, "input_scale", None) is not None:
            continue
        W = _dequant_weight(lin)
        N, K = W.shape
        if N % 16 or K % 16:
            continue
        if os.environ.get("R9K_DRAFT_W4", "0") == "1" and K % 32 == 0:
            # drafter-only 4-bit (GGZ14 serves its DFlash2 drafter W4): halves the drafter's weight stream; can
            # only change acceptance, never the target's output
            from ..quant.nvfp4 import mxfp4_linearize
            from ..utils import note_shape
            note_shape("mxfp4", N, K)
            lin.quant_method = _MxMethod(mxfp4_linearize(lin, W))
            n += 1
            continue
        bsc = getattr(lin, "weight_scale_inv", None)
        bsc = bsc if isinstance(bsc, torch.Tensor) else getattr(lin, "weight_scale", None)
        if getattr(qm, "block_quant", False):
            # 128x128 block scales -> exact bytes + r9k_gemm_fp8_block (needs K % 128)
            if bsc is None or bsc.dim() != 2 or K % 128 or bsc.shape != ((N + 127) // 128, K // 128):
                continue
            bs = bsc.detach().float().contiguous()
            full = bs.repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)
            q = (W.float() / full).to(torch.float8_e4m3fn)
            err = ((q.float() * full - W.float()).abs().max() / W.float().abs().max().clamp_min(1e-12)).item()
            if err > 1e-2:
                logger.warning("r9700: %s block-fp8 round-trip error %.3g, left on stock", name, err)
                continue
            lin._r9k_fp8b = (F8.permute_fp8(q.view(torch.uint8)), bs, N, K)
            from ..utils import note_shape
            note_shape("fp8block", N, K)
            lin.weight = torch.nn.Parameter(torch.empty((0,), dtype=torch.float8_e4m3fn, device=W.device),
                                            requires_grad=False)
            lin.quant_method = _BlockMethod()
            n += 1
            continue
        sc = lin.weight_scale.detach().float().reshape(-1)
        if sc.numel() == 1:
            row = sc.expand(N)
        elif sc.numel() == N:
            row = sc
        else:
            continue
        q = (W.float() / row[:, None]).to(torch.float8_e4m3fn)
        err = ((q.float() * row[:, None] - W.float()).abs().max() / W.float().abs().max().clamp_min(1e-12)).item()
        if err > 1e-2:
            logger.warning("r9700: %s fp8 round-trip error %.3g, left on stock", name, err)
            continue
        lin._r9k_fp8 = F8.Fp8Weight(F8.permute_fp8(q.view(torch.uint8)), row.contiguous(), N, K)
        from ..utils import note_shape
        note_shape("fp8row", N, K)
        lin.weight = torch.nn.Parameter(torch.empty((0,), dtype=torch.float8_e4m3fn, device=W.device),
                                        requires_grad=False)
        lin.quant_method = R9kFp8UnquantMethod()
        n += 1
    return n


def _finish(model) -> None:
    _dequant_fused_kv(model.model)          # uses the stock quant methods: before the swap
    from collections import Counter
    from vllm.model_executor.layers.linear import LinearBase
    kinds = Counter(type(getattr(m, "quant_method", None)).__name__ for m in model.modules() if isinstance(m, LinearBase))
    logger.info("r9700: DFlash drafter linear methods: %s", str(dict(kinds)))
    if os.environ.get("R9K_DRAFT_FP8", "1") == "1":
        n = swap_fp8_linears(model)
        if n:
            logger.info_once("r9700: DFlash drafter: %d fp8 linears -> libr9k %s", n,
                             "MXFP4 (R9K_DRAFT_W4=1)" if os.environ.get("R9K_DRAFT_W4", "0") == "1"
                             else "fp8 GEMM (exact bytes)")


class R9kDFlashQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    def load_weights(self, weights):
        out = super().load_weights(weights)
        _finish(self)
        return out


class R9kDFlash2Qwen3ForCausalLM(DFlash2Qwen3ForCausalLM):
    def load_weights(self, weights):
        out = super().load_weights(weights)
        _finish(self)
        return out


ARCHS = {
    "DFlashDraftModel": "r9700_vllm.models.dflash:R9kDFlashQwen3ForCausalLM",
    "DFlash2DraftModel": "r9700_vllm.models.dflash:R9kDFlash2Qwen3ForCausalLM",
}
