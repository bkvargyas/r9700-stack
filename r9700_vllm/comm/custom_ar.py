"""Enable vLLM custom (P2P IPC) all-reduce on gfx12 (R9700): R9K_CUSTOM_AR=1. OFF by default -- measured 2026-09-18:
it engages but produces garbage output on 2x R9700 (PCIe P2P, emulated switch); use libr4d AR instead.

Stock RocmPlatform.use_custom_allreduce() is True only for gfx94/gfx95, so TP2 on two R9700s goes through RCCL:
~69 us per decode all-reduce, ~157 of them per MTP-3 step = ~11 ms/step (22% of GPU time). vLLM's CustomAllreduce
already handles 2 PCIe-only GPUs on ROCm (no fully-connected requirement at world size 2); it needs working HIP
IPC (HSA_ENABLE_IPC_MODE_LEGACY=0 on this platform) and P2P between the cards.
"""
from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)
_PATCHED = False


def patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("R9K_CUSTOM_AR", "0") != "1":   # default OFF: vLLM custom AR produces garbage on gfx1201
        return False
    try:
        from vllm.platforms import rocm as R
    except Exception as e:
        logger.warning("r9700: custom AR hook not installed (%s)", e)
        return False
    cls = R.RocmPlatform
    orig = cls.use_custom_allreduce.__func__

    def use_custom_allreduce(c) -> bool:
        try:
            if R.on_gfx12x():
                return True
        except Exception:
            pass
        return orig(c)

    cls.use_custom_allreduce = classmethod(use_custom_allreduce)
    _PATCHED = True
    return True
