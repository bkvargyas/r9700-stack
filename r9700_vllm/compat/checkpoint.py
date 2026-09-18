"""Checkpoint-compat hooks for tcclaviger's Qwen3.8-Flash-Next quantizations on stock vLLM.

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


def patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
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
