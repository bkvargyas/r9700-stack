"""Checkpoint-compat hooks for tcclaviger's Qwen3.8-Flash-Next quantizations on stock vLLM.

- Compressed-tensors config groups without their own ``format`` inherit the global one; tcclaviger's
  checkpoints declare ``format: mxfp4-pack-quantized`` globally while their FP8-block groups (attention, GDN
  projections, MTP experts) carry no format, so stock dispatch looks for MXFP4 schemes and fails
  ("No compressed-tensors compatible scheme"). We set ``float-quantized`` on format-less 8-bit float groups.
- Drops ``mtp.lm_head.weight_q4 / weight_scale / weight_zero`` (the fork's private 4-bit draft head): stock remaps
  them to ``model.lm_head.*`` which the MTP predictor does not have, failing the load. The MTP head then loads
  from the checkpoint's bf16 ``lm_head.weight`` (the target head) exactly as stock intends; R9K_DRAFT_LMHEAD=mxfp4
  gives a quantized draft head back through our own kernel.
"""
from __future__ import annotations

import re

from vllm.logger import init_logger

logger = init_logger(__name__)
_PATCHED = False
_DROP = re.compile(r"(^|\.)mtp\.lm_head\.weight_(q4|scale|zero)$")


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


def patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    _fix_ct_formats()
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
    _PATCHED = True
    return True
