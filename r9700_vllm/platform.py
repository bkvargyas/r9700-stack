"""R9700Platform: stock RocmPlatform with the plugin's device communicator (libr4d TP=2 all-reduce).

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


def detect() -> str | None:
    """``vllm.platform_plugins`` entry point: the platform class qualname if this is a ROCm host, else None."""
    if os.environ.get("R9K_PLATFORM", "1") == "0":
        return None
    from vllm.platforms import rocm_platform_plugin
    return "r9700_vllm.platform.R9700Platform" if rocm_platform_plugin() else None
