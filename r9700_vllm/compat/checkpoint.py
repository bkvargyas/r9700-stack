"""Checkpoint-compat hooks for tcclaviger's Qwen3.8-Flash-Next quantizations on stock vLLM.

- Compressed-tensors config groups without their own ``format`` inherit the global one; tcclaviger's
  checkpoints declare ``format: mxfp4-pack-quantized`` globally while their FP8-block groups (attention, GDN
  projections, MTP experts) carry no format, so stock dispatch looks for MXFP4 schemes and fails
  ("No compressed-tensors compatible scheme"). We set ``float-quantized`` on format-less 8-bit float groups.
- Drops ``self_attn.q_scale`` (fork's fp8-attention query scales; no stock slot).
- Drops ``mtp.lm_head.weight_q4 / weight_scale / weight_zero`` (the fork's private 4-bit draft head): stock remaps
  them to ``model.lm_head.*`` which the MTP predictor does not have, failing the load. The MTP head then loads
  from the checkpoint's bf16 ``lm_head.weight`` (the target head) exactly as stock intends; R9K_DRAFT_LMHEAD=mxfp4
  gives a quantized draft head back through our own kernel.
"""
from __future__ import annotations

import re

import torch

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)
_PATCHED = False
_DROP = re.compile(r"(^|\.)mtp\.lm_head\.weight_(q4|scale|zero)$")
# fork-calibrated query scales for its fp8 attention path; stock QSA has no slot for them (k/v scales map fine)
_DROP_MAIN = re.compile(r"\.self_attn\.[qkv]_scale$")   # bf16 KV: all unused (fp8-KV work will need k/v)


def _fix_ct_formats() -> bool:
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )
    except Exception as e:
        logger.warning("r9700: CT format compat not installed (%s)", e)
        return False
    orig = CompressedTensorsConfig.from_config.__func__

    def from_config(cls, config):
        gfmt = config.get("format")
        if gfmt and gfmt != "float-quantized":
            for name, grp in (config.get("config_groups") or {}).items():
                w = grp.get("weights") or {}
                if grp.get("format") is None and w.get("num_bits") == 8 and w.get("type") == "float":
                    grp["format"] = "float-quantized"
                    logger.info_once("r9700: CT group %s: fp8 weights under global format %r -> float-quantized",
                                     name, gfmt)
        return orig(cls, config)

    CompressedTensorsConfig.from_config = classmethod(from_config)
    return True


def _exact_uva_pinning() -> bool:
    """Stock UVAOffloader pins each offloaded param with tensor.pin_memory(), which rounds to a power of two
    (+~22% for Flash-Next's expert tensors). Swap in exact-size hipHostRegister pinning while it offloads."""
    try:
        from vllm.model_executor.offloader import uva
    except Exception:
        return False
    from ..utils.hostmem import pinned_empty
    cls = uva.UVAOffloader
    orig = cls._maybe_offload_to_cpu
    stock_pin = torch.Tensor.pin_memory

    def exact_pin(self, *a, **k):
        if self.device.type != "cpu":
            return stock_pin(self, *a, **k)
        out = pinned_empty(self.shape, self.dtype)
        out.copy_(self)
        return out

    def _maybe_offload_to_cpu(self, module, prefix=""):
        torch.Tensor.pin_memory = exact_pin
        try:
            return orig(self, module, prefix)
        finally:
            torch.Tensor.pin_memory = stock_pin

    cls._maybe_offload_to_cpu = _maybe_offload_to_cpu
    return True


def patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    _fix_ct_formats()
    _exact_uva_pinning()
    try:
        from vllm.models.qwen4_exp.amd import mtp as M
    except Exception as e:
        logger.warning("r9700: checkpoint compat hook not installed (%s)", e)
        return False
    cls = M.Qwen4ExpMTP
    orig = cls.load_weights

    def load_weights(self, weights):
        dropped = []

        def filt():
            for name, w in weights:
                if _DROP.search(name):
                    dropped.append(name)
                    continue
                yield name, w

        out = orig(self, filt())
        if dropped:
            logger.info_once("r9700: ignored fork-private MTP draft head tensors: %s", ", ".join(dropped))
        return out

    cls.load_weights = load_weights

    try:
        from vllm.models.qwen4_exp.amd import model as Mdl
        mcls = Mdl.Qwen4ExpModel
        morig = mcls.load_weights

        def model_load_weights(self, weights):
            return morig(self, ((n, w) for n, w in weights if not _DROP_MAIN.search(n)))

        mcls.load_weights = model_load_weights
    except Exception as e:
        logger.warning("r9700: q_scale compat not installed (%s)", e)
    _PATCHED = True
    return True
