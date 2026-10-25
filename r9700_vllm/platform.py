"""R9700Platform: stock RocmPlatform with the plugin's device communicator (libr4d TP=2 all-reduce) and
the gfx1201 answer to the MLA prefill backend question (none).

Registered through the ``vllm.platform_plugins`` entry point; vLLM activates an out-of-tree platform plugin in
preference to the built-in one. Detection defers to vLLM's own ROCm probe, so this is active exactly where stock
would pick RocmPlatform. R9K_PLATFORM=0 opts out (stock RocmPlatform, RCCL all-reduce).
"""
from __future__ import annotations

import os

from vllm.platforms.rocm import RocmPlatform


class R9700Platform(RocmPlatform):

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "r9700_vllm.comm.r4d_ar.R9kCommunicator"

    @classmethod
    def get_mla_prefill_backend_cls(cls, mla_selector_config):
        """No MLA prefill backend runs on gfx1201: AITER FA is gfx942/gfx950, and the CK flash-attn wheel carries
        no gfx1201 code objects -- the 2026-10 nightlies still select it (its validation never looks at the arch)
        and die at the first dense prefill with "device kernel image is invalid". Raising is the API's "none valid"
        answer: a sparse MLA layer (GLM-5.3-Flash) then keeps the top-k MQA path for prefill, exactly as it did
        before upstream added the dense path; a dense MLA model raises at construction, as it would without us.
        R9K_MLA_PREFILL=stock defers to stock's choice."""
        if os.environ.get("R9K_MLA_PREFILL", "none") == "stock":
            return super().get_mla_prefill_backend_cls(mla_selector_config)
        raise ValueError("r9700: no MLA prefill backend runs on gfx1201 (AITER FA is gfx9-only, CK flash-attn ships "
                         "no gfx1201 kernels); sparse MLA layers prefill on the top-k MQA path")


def detect() -> str | None:
    """``vllm.platform_plugins`` entry point: the platform class qualname if this is a ROCm host, else None."""
    if os.environ.get("R9K_PLATFORM", "1") == "0":
        return None
    from vllm.platforms import rocm_platform_plugin
    return "r9700_vllm.platform.R9700Platform" if rocm_platform_plugin() else None
