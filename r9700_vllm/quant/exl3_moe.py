"""EXL3 routed experts on stock vLLM's RoutedExperts layer: exllamav3's exl3_moe kernel behind FusedMoEMethodBase.

Expert parallel: each rank owns whole experts (384 / 8 = 48 a layer for DeepSeek-V4.1-Flash), so no expert matrix is ever
split and the pack's per-expert tensors load as they are. Per local expert: w1 (gate) and w3 (up) [5120 -> 2304], w2 (down)
[2304 -> 5120] as trellis/suh/svh + codebook flag, K uniform within a layer (3, or 2 on layers 18-22). The kernel takes
device-pointer tables per matrix role, exllamav3's sorted routing (expert_count, token_sorted, weight_sorted) and preallocated
temp buffers; above its fused-row limit experts go through the reconstruct-and-GEMM tier. Design: notes/exl3-plan.md.

Package M owns this file. Public name: Exl3MoEMethod(moe_config, quant_config).
"""
import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase

logger = init_logger("vllm.r9700_vllm.exl3_moe")


class Exl3MoEMethod(FusedMoEMethodBase):

    def __init__(self, moe, quant_config):
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(self, layer, num_experts: int, hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        raise NotImplementedError("package M")

    def get_fused_moe_quant_config(self, layer):
        return None

    def process_weights_after_loading(self, layer) -> None:
        raise NotImplementedError("package M")

    def apply(self, layer, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, shared_experts=None,
              shared_experts_input: torch.Tensor | None = None):
        raise NotImplementedError("package M")
