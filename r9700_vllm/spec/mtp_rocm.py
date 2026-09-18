"""Allow Qwen4Exp (Flash-Next) MTP with num_speculative_tokens > 1 on ROCm.

Stock vLLM keeps a ROCm-only allowlist of attention-metadata types for multi-step drafting
(SpecDecodeBaseProposer.allowed_attn_types). Qwen4Exp's QSA builder emits FlashAttentionMetadata (its builder
subclasses FlashAttentionMetadataBuilder) and QSAForwardMetadata for the indexer; neither is listed, so k>1
raises. Same idea as open upstream PR #55292. Appends both types after the stock __init__ builds the list.
"""
from __future__ import annotations

from vllm.logger import init_logger

logger = init_logger(__name__)
_PATCHED = False


def patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    try:
        from vllm.v1.spec_decode import llm_base_proposer as lbp
    except Exception as e:
        logger.warning("r9700: MTP allowlist hook not installed (%s)", e)
        return False
    cls = lbp.SpecDecodeBaseProposer
    orig = cls.__init__

    def __init__(self, *a, **k):
        orig(self, *a, **k)
        if getattr(self, "allowed_attn_types", None) is None:
            return
        extra = []
        try:
            from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
            extra.append(FlashAttentionMetadata)
        except Exception:
            pass
        try:
            from vllm.models.qwen4_exp.common.qsa_cache import QSAForwardMetadata
            extra.append(QSAForwardMetadata)
        except Exception:
            pass
        self.allowed_attn_types = tuple(self.allowed_attn_types) + tuple(
            t for t in extra if t not in self.allowed_attn_types)

    cls.__init__ = __init__
    _PATCHED = True
    return True
