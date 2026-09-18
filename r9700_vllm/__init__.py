"""r9700_vllm: gfx1201 (Radeon AI PRO R9700) kernels for stock vLLM on ROCm 10, through vLLM's extension points.

Entry points (pyproject.toml):
  vllm.platform_plugins  r9700 = r9700_vllm.platform:detect   RocmPlatform subclass -> libr4d TP=2 all-reduce
  vllm.general_plugins   r9700 = r9700_vllm:register           everything below, in every vLLM process

register():
  * torch custom ops (r9700::*) for the libr9k kernels                         ops.py
  * quantization config "compressed-tensors" -> R9kCompressedTensorsConfig    quant/ct.py
  * model classes for Qwen4Exp{ForConditionalGeneration,ForCausalLM,MTP}     models/qwen4_exp.py
  * DFlash/DFlash2 drafters with fp8 attention (context-KV dequant)          models/dflash.py
  * one version-gated monkeypatch: MTP k>1 attention-type allowlist          spec/mtp_rocm.py
Only engages on ROCm. R9K_DISABLE=quant,models,mtp turns individual pieces off (R9K_PLATFORM=0 for the platform).
"""
import os

__version__ = "0.2.0"


def _disabled(name: str) -> bool:
    return name in {s.strip() for s in os.environ.get("R9K_DISABLE", "").split(",") if s.strip()}


def register() -> None:
    from vllm.logger import init_logger
    log = init_logger("vllm.r9700_vllm")
    try:
        from vllm.platforms import current_platform
        if not current_platform.is_rocm():
            log.info("r9700_vllm: not ROCm, nothing registered")
            return
    except Exception:
        return
    done = []
    from . import ops
    ops.register()
    if not _disabled("quant"):
        from .quant import ct  # noqa: F401  (registers "compressed-tensors")
        done.append("quant:compressed-tensors")
    if not _disabled("models"):
        from vllm import ModelRegistry
        from .models.qwen4_exp import ARCHS
        from .models.dflash import ARCHS as DF_ARCHS
        for arch, qualname in (ARCHS | DF_ARCHS).items():
            ModelRegistry.register_model(arch, qualname)
        done.append("models:" + ",".join(ARCHS | DF_ARCHS))
    if not _disabled("mtp"):
        from .spec import mtp_rocm
        if mtp_rocm.patch():
            done.append("patch:mtp_allowlist")
    plat = type(current_platform).__name__
    log.info("r9700_vllm %s registered: %s (platform %s)", __version__, ", ".join(done) or "nothing", plat)
