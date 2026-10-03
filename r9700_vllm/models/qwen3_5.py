"""Qwen3.5 / Qwen3.8 dense hybrid model classes (the 27B) registered over the stock architectures: thin subclasses
whose only addition is the one-page GDN state's record in the config-level mamba state shape, which vLLM consults
to size the padded mamba page before any layer exists (Platform._align_hybrid_block_size). See models/gdn.py.
"""
from __future__ import annotations

from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

from .gdn import onepage_state_dtype_from_config, onepage_state_shape_from_config


class R9kQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        return onepage_state_shape_from_config(Qwen3_5ForConditionalGeneration, vllm_config)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return onepage_state_dtype_from_config(Qwen3_5ForConditionalGeneration, vllm_config)


class R9kQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        return onepage_state_shape_from_config(Qwen3_5ForCausalLM, vllm_config)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return onepage_state_dtype_from_config(Qwen3_5ForCausalLM, vllm_config)


ARCHS = {
    "Qwen3_5ForConditionalGeneration": "r9700_vllm.models.qwen3_5:R9kQwen3_5ForConditionalGeneration",
    "Qwen3_5ForCausalLM": "r9700_vllm.models.qwen3_5:R9kQwen3_5ForCausalLM",
}
