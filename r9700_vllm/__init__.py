"""r9700_vllm: gfx1201 (Radeon AI PRO R9700) kernels and hooks for stock vLLM on ROCm 10.

Loaded by vLLM in every process through the ``vllm.general_plugins`` entry point (see pyproject.toml).
Every hook is idempotent, only engages on ROCm gfx12, and logs (instead of raising) if the vLLM API it
targets has moved, leaving that part of vLLM stock. Disable individual hooks with R9K_DISABLE=moe,ple,...
"""
import os

__version__ = "0.1.0"


def _disabled(name: str) -> bool:
    return name in {s.strip() for s in os.environ.get("R9K_DISABLE", "").split(",") if s.strip()}


def register() -> None:
    from vllm.logger import init_logger
    log = init_logger("r9700_vllm")
    done = []
    try:
        from vllm.platforms import current_platform
        if not current_platform.is_rocm():
            log.info("r9700_vllm: not ROCm, nothing registered")
            return
    except Exception:
        return
    if not _disabled("ple"):
        from .ple import int6
        if int6.patch():
            done.append("ple_int6")
    if not _disabled("moe"):
        from .moe import ct_mxfp4
        if ct_mxfp4.patch():
            done.append("ct_mxfp4_moe")
    log.info("r9700_vllm %s registered: %s", __version__, ", ".join(done) or "nothing")
