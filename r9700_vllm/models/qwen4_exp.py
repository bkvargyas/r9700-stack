"""Qwen4Exp (Qwen3.8-Flash-Next) model classes for gfx1201, registered over the stock architectures through
``ModelRegistry.register_model`` (the documented way to override a built-in model). Each is a thin subclass of
the stock ROCm class; forward/compute graphs are stock.

What the subclasses add, all around construction and weight loading:
  * checkpoint compat for tcclaviger's quantizations: ``*.weight_scale_inv`` -> ``*.weight_scale`` (DeepSeek-
    style block-fp8 scale names); drop the fork's fp8-attention q/k/v scales (no stock slot, bf16 KV); MTP: drop
    the fork-private 4-bit draft head (``mtp.lm_head.weight_q4/_scale/_zero``) and, with R9K_MTP_MLP=mxfp4
    (default), re-quantize the MTP experts' fp8 block-128 weights to MXFP4 on load (640-wide experts cannot
    shard by 128 at TP2; quant/ct.py drops that CT group). Draft-only: affects MTP acceptance, never output.
  * construction scope: int6 PLE table (ple/int6.py) when the checkpoint's PLE shards are int6, and exact-size
    pinning for vLLM's UVA expert offload (utils/hostmem.py).
  * optional quantized LM heads, R9K_TARGET_LMHEAD / R9K_DRAFT_LMHEAD = fp8|mxfp4 (models/lm_heads.py).

Unavoidable internals touched (both scoped to ``__init__`` and restored): the ``PLEVocabParallelEmbedding`` name
in ``vllm.models.qwen4_exp.amd.ple_layer`` (stock constructs it without a quant config or class hook) and
``torch.Tensor.pin_memory``. Upstream candidates: an embedding-class hook / quant_config for the PLE table, and
exact-size pinning in the UVA offloader.
"""
from __future__ import annotations

import contextlib
import os
import re

import torch

from vllm.logger import init_logger
from vllm.models.qwen4_exp.amd import ple_layer as _ple_layer
from vllm.models.qwen4_exp.amd.model import Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration
from vllm.models.qwen4_exp.amd.mtp import Qwen4ExpMTP

from . import lm_heads

logger = init_logger("vllm." + __name__)

_DROP_MTP_HEAD = re.compile(r"(^|\.)mtp\.lm_head\.weight_(q4|scale|zero)$")
_DROP_ATTN_SCALE = re.compile(r"\.self_attn\.(attn\.)?[qkv]_scale$")     # bf16 KV: all unused
_MTP_MLP_FP8 = re.compile(r"(^|\.)mtp\.layers\.\d+\.mlp\.(experts\.\d+|shared_expert)\.(gate|up|down)_proj\."
                          r"weight(_scale_inv|_scale)?$")


def rename_scale(name: str) -> str:
    """``weight_scale_inv`` (dequant multiplier, same values/shape as CT's ``weight_scale``) -> CT's name."""
    return name[: -len("_inv")] if name.endswith(".weight_scale_inv") else name


def fp8_block_to_mxfp4(base: str, w: torch.Tensor, scale: torch.Tensor, block: int = 128):
    """fp8 [N, K] + block scales [ceil(N/128), ceil(K/128)] -> CT MXFP4 (``weight_packed`` [N, K/2] u8,
    ``weight_scale`` [N, K/32] E8M0 u8) under ``base``."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wf = w.to(dev).float()
    N, K = wf.shape
    s = scale.to(dev).float().repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    packed, e8m0 = lm_heads.quantize_mxfp4((wf * s).to(torch.bfloat16))
    yield f"{base}.weight_packed", packed.cpu()
    yield f"{base}.weight_scale", e8m0.cpu()


def target_weights(weights):
    """Target-model checkpoint stream: drop unused attention scales, rename block-fp8 scales; log the failing
    tensor if loading raises (stock only says which parameter)."""
    last = [None]

    def gen():
        for n, w in weights:
            if _DROP_ATTN_SCALE.search(n):
                continue
            n = rename_scale(n)
            last[0] = n
            yield n, w
    return gen(), last


def mtp_weights(weights, mtp_mxfp4: bool):
    dropped, pend = [], {}
    for name, w in weights:
        if _DROP_MTP_HEAD.search(name):
            dropped.append(name)
            continue
        if mtp_mxfp4 and _MTP_MLP_FP8.search(name):
            base = name.rsplit(".", 1)[0]
            kind = "scale" if name.endswith(("weight_scale_inv", "weight_scale")) else "w"
            pend.setdefault(base, {})[kind] = w
            if len(pend[base]) == 2:
                d = pend.pop(base)
                yield from fp8_block_to_mxfp4(base, d["w"], d["scale"])
            continue
        yield rename_scale(name), w
    if pend:
        logger.warning("r9700: unpaired MTP fp8 tensors left: %s", list(pend)[:4])
    if dropped:
        logger.info_once("r9700: ignored fork-private MTP draft head tensors: %s", ", ".join(dropped))


@contextlib.contextmanager
def construction_scope():
    """Stock construction with the int6 PLE embedding class (int6 checkpoints only) and exact pinning."""
    from ..ple.int6 import checkpoint_ple_format, make_int6_embedding_cls
    from ..utils.hostmem import exact_pinning
    stock_cls = getattr(_ple_layer, "PLEVocabParallelEmbedding", None)
    swap = stock_cls is not None and checkpoint_ple_format() == "int6"
    if checkpoint_ple_format() == "int6" and stock_cls is None:
        raise RuntimeError("r9700: vLLM's ple_layer no longer exposes PLEVocabParallelEmbedding; int6 PLE needs "
                           "a new construction hook for this vLLM version")
    if swap:
        _ple_layer.PLEVocabParallelEmbedding = make_int6_embedding_cls(stock_cls)
    try:
        with exact_pinning():
            yield
    finally:
        if swap:
            _ple_layer.PLEVocabParallelEmbedding = stock_cls


def _head_fmt(env: str) -> str | None:
    fmt = os.environ.get(env, "").lower()
    return fmt if fmt in ("fp8", "mxfp4") else None


def _make_shadow(owner, head, env: str, what: str) -> None:
    fmt = _head_fmt(env)
    if fmt and isinstance(getattr(head, "weight", None), torch.Tensor) and head.weight.device.type == "cuda":
        # plain attribute (not a registered submodule): the shadow shares lm_head's weight
        object.__setattr__(owner, "_r9k_head", lm_heads.shadow(head, fmt, what))


def _head(owner):
    head = owner.__dict__.get("_r9k_head")
    return owner.lm_head if head is None else head


def _load_target(self, super_load, weights):
    gen, last = target_weights(weights)
    try:
        return super_load(gen)
    except Exception as e:
        logger.error("r9700: weight load failed at checkpoint tensor %s: %s", last[0], e)
        raise


class R9kQwen4ExpForConditionalGeneration(Qwen4ExpForConditionalGeneration):

    def __init__(self, *, vllm_config, prefix: str = "model") -> None:
        with construction_scope():
            super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights):
        loaded = _load_target(self, super().load_weights, weights)
        _make_shadow(self, self.language_model.lm_head, "R9K_TARGET_LMHEAD", "target")
        return loaded

    def compute_logits(self, hidden_states):
        head = self.__dict__.get("_r9k_head")
        if head is None:
            return super().compute_logits(hidden_states)
        return self.language_model.logits_processor(head, hidden_states)


class R9kQwen4ExpForCausalLM(Qwen4ExpForCausalLM):

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        with construction_scope():
            super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights):
        loaded = _load_target(self, super().load_weights, weights)
        _make_shadow(self, self.lm_head, "R9K_TARGET_LMHEAD", "target")
        return loaded

    def compute_logits(self, hidden_states):
        return self.logits_processor(_head(self), hidden_states)


class R9kQwen4ExpMTP(Qwen4ExpMTP):

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        with construction_scope():
            super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights):
        loaded = super().load_weights(mtp_weights(weights, os.environ.get("R9K_MTP_MLP", "mxfp4") == "mxfp4"))
        _make_shadow(self, self.lm_head, "R9K_DRAFT_LMHEAD", "MTP draft")
        return loaded

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        return self.logits_processor(_head(self), hidden_states)


ARCHS = {
    "Qwen4ExpForConditionalGeneration": "r9700_vllm.models.qwen4_exp:R9kQwen4ExpForConditionalGeneration",
    "Qwen4ExpForCausalLM": "r9700_vllm.models.qwen4_exp:R9kQwen4ExpForCausalLM",
    "Qwen4ExpMTP": "r9700_vllm.models.qwen4_exp:R9kQwen4ExpMTP",
}
