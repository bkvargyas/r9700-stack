"""Gated-DeltaNet input projections as ONE GEMM (both Qwen3.8-27B and Flash-Next use stock QwenGatedDeltaNetAttention).

`in_proj_qkvz` and `in_proj_ba` are two linears over the same hidden state, run back to back in every GDN layer
(48 per forward on the 27B): two activation quantizations + two GEMM launches where one would do. GGZ14 merges
them the same way (radiance_gdnmerge.py). Here: an out-of-tree PluggableLayer subclass (vLLM's documented override
for `qwen_gated_delta_net_attention`) that, once both projections are loaded onto libr9k kernels of the SAME format,
concatenates their weights along N -- MXFP4 (fragment-order tiles + per-row packed scales) or per-row fp8
(fragment-order tiles + row scales), both row-local, so the merged GEMM is bit-identical per row -- and serves:

    in_proj_qkvz(x) -> one GEMM over [qkvz; ba] -> returns the qkvz columns, stashes the ba columns
    in_proj_ba(x)   -> returns the stash      (stock forward calls these two back to back on the same x)

The merge is triggered from in_proj_ba's process_weights_after_loading (stock's loader visits in_proj_qkvz first).
Layers whose formats differ (e.g. Flash-Next's block-fp8 qkvz) stay unmerged. R9K_GDN_MERGE=0 disables.
"""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

logger = init_logger("vllm." + __name__)


def _format(lin):
    """('mxfp4', wq, wsr, N, K) | ('fp8', Fp8Weight) | None for a linear served by a libr9k kernel."""
    mx = getattr(lin, "_r9k_mx", None)
    if mx is not None and getattr(lin, "_r9k_nk", None):
        N, K = lin._r9k_nk
        return ("mxfp4", lin.weight.data, lin.weight_scale.data, N, K)
    W = getattr(lin, "_r9k_fp8", None)
    if W is not None:
        return ("fp8", W)
    return None


class _MergeTrigger(QuantizeMethodBase):
    """Wraps in_proj_ba's quant method: stock behaviour, plus the merge once its own weights are final."""

    def __init__(self, inner, gdn):
        self.inner = inner
        object.__setattr__(self, "_gdn", gdn)

    def create_weights(self, *a, **k):
        return self.inner.create_weights(*a, **k)

    def apply(self, layer, x, bias=None):
        return self.inner.apply(layer, x, bias)

    def process_weights_after_loading(self, layer):
        self.inner.process_weights_after_loading(layer)
        self._gdn._r9k_merge()

    def __getattr__(self, name):          # anything else stock asks the method (e.g. flags) -> inner
        return getattr(self.__dict__["inner"], name)


class _MergedQKVZ(QuantizeMethodBase):
    def __init__(self, gdn, kind, payload, nq, nb):
        object.__setattr__(self, "_gdn", gdn)
        self.kind, self.payload, self.nq, self.nb = kind, payload, nq, nb

    def create_weights(self, *a, **k):
        raise RuntimeError("unused")

    def process_weights_after_loading(self, layer):
        pass

    def apply(self, layer, x, bias=None):
        from .. import ops
        if self.kind == "mxfp4":
            wq, wsr, N, K = self.payload
            out = ops.mxfp4_linear(x, wq, wsr, N, K).to(x.dtype)
        else:
            out = ops.fp8_linear(x, self.payload).to(x.dtype)
        qkvz, ba = out.split([self.nq, self.nb], dim=-1)
        self._gdn._r9k_ba_out = ba
        return qkvz


class _StashedBA(QuantizeMethodBase):
    def __init__(self, gdn):
        object.__setattr__(self, "_gdn", gdn)

    def create_weights(self, *a, **k):
        raise RuntimeError("unused")

    def process_weights_after_loading(self, layer):
        pass

    def apply(self, layer, x, bias=None):
        return self._gdn._r9k_ba_out


# vLLM resolves OOT pluggable layers by the in-tree CLASS name (PluggableLayer.__new__: cls.__name__)
@PluggableLayer.register_oot(name="QwenGatedDeltaNetAttention")
class R9kQwenGatedDeltaNetAttention(QwenGatedDeltaNetAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._r9k_ba_out = None
        ba = getattr(self, "in_proj_ba", None)
        if os.environ.get("R9K_GDN_MERGE", "1") == "1" and ba is not None and getattr(ba, "quant_method", None) \
                is not None and getattr(self, "in_proj_qkvz", None) is not None:
            ba.quant_method = _MergeTrigger(ba.quant_method, self)

    def _r9k_merge(self) -> None:
        q, b = self.in_proj_qkvz, self.in_proj_ba
        if getattr(q, "bias", None) is not None or getattr(b, "bias", None) is not None:
            return
        fq, fb = _format(q), _format(b)
        if fq is None or fb is None or fq[0] != fb[0]:
            return
        from ..utils import note_shape
        if fq[0] == "mxfp4":
            _, wq1, ws1, n1, k1 = fq
            _, wq2, ws2, n2, k2 = fb
            if k1 != k2:
                return
            wq = torch.cat([wq1.reshape(1, -1), wq2.reshape(1, -1)], dim=1).contiguous()
            wsr = torch.cat([ws1, ws2], dim=2).contiguous()
            q.quant_method = _MergedQKVZ(self, "mxfp4", (wq, wsr, n1 + n2, k1), n1, n2)
            note_shape("mxfp4", n1 + n2, k1)
        else:
            from ..kernels.fp8 import Fp8Weight
            W1, W2 = fq[1], fb[1]
            if W1.K != W2.K:
                return
            W = Fp8Weight(torch.cat([W1.wq, W2.wq]).contiguous(), torch.cat([W1.ws, W2.ws]).contiguous(),
                          W1.N + W2.N, W1.K)
            q.quant_method = _MergedQKVZ(self, "fp8", W, W1.N, W2.N)
            n1, n2 = W1.N, W2.N
            note_shape("fp8row", W.N, W.K)
        b.quant_method = _StashedBA(self)
        logger.info_once("r9700: GDN in_proj_qkvz + in_proj_ba merged into one %s GEMM (N=%d+%d)", fq[0], n1, n2)
