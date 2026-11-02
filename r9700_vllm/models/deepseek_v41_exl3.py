"""DeepSeek-V4.1 from an EXL3 pack: the loader glue over stock vLLM's ROCm DeepseekV41 model classes.

Thin subclasses registered through ``ModelRegistry.register_model`` (the mechanism in models/qwen4_exp.py); they behave exactly
like stock unless the quant config is Exl3Config. Then ``load_weights`` renames the pack's ``attn.wo_a.slice.{g}.*`` tensors to
the ``wo_a`` parameter shards, routes ``<name>.trellis|suh|svh|mul1`` to the EXL3 parameters (quant/exl3.py on linears,
quant/exl3_moe.py on experts), takes only the Engram tables (``layers.{1,14}.engram.embed.*``) from the base checkpoint's
shards, and drops the vision tower when serving text. Design: notes/exl3-plan.md.

Package N owns this file. Public name: ARCHS (architecture -> "module:Class" for __init__.register).
"""
from vllm.logger import init_logger

logger = init_logger("vllm.r9700_vllm.deepseek_v41_exl3")

ARCHS: dict[str, str] = {}  # TODO(N): {"DeepseekV41ForCausalLM": "r9700_vllm.models.deepseek_v41_exl3:DeepseekV41ForCausalLMExl3", ...}
