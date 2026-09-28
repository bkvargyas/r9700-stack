"""PLE short conv, prefill path, with the two large transposes on our tiled kernel (kernels/r9k_transpose.hip).

Stock ``Qwen4ExpPLELayer._short_conv_dilated_prefill_batched`` packs the chunk's tokens into [prefills, max_len,
4*hidden], transposes to channels-first for ``F.conv1d`` and transposes the result back. Both transposes go through
ATen's generic permuted copy, which on gfx1201 runs at ~14 GB/s: 5.9 ms each on the 84 MB tensor of a 4096-token
chunk, 23 of the op's 27 ms (the depthwise conv itself is ~1 ms). ``r9k_transpose16`` does the same copy through
64x64 LDS tiles with coalesced 16-byte reads and writes.

Everything else in ``prefill_batched`` mirrors the stock method (same ops, same order, same rounding points), so the
output and the written-back conv state are identical. Bound per PLE layer instance by ``install`` (models/qwen4_exp.py);
R9K_PLE_CONV=stock keeps vLLM's method.
"""
from __future__ import annotations

import ctypes
import os
import types

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)

_L = None


def lib():
    global _L
    if _L is None:
        from ..kernels import moe as KM
        L = KM.lib()
        L.r9k_transpose16.restype = ctypes.c_int
        L.r9k_transpose16.argtypes = [ctypes.c_long] * 4 + [ctypes.c_int] * 3 + [ctypes.c_long]
        _L = L
    return _L


def available() -> bool:
    try:
        return hasattr(lib(), "r9k_transpose16")
    except Exception:
        return False


def transpose12(x: torch.Tensor) -> torch.Tensor:
    """x [B, R, C] (16-bit dtype, C contiguous, any batch/row strides) -> contiguous [B, C, R]; == x.transpose(1, 2).contiguous()."""
    assert x.dim() == 3 and x.element_size() == 2 and x.stride(2) == 1
    B, R, C = x.shape
    out = torch.empty((B, C, R), dtype=x.dtype, device=x.device)
    if x.numel() == 0:
        return out
    rc = lib().r9k_transpose16(x.data_ptr(), x.stride(0), x.stride(1), out.data_ptr(), B, R, C,
                               torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_transpose16 failed ({rc})")
    return out


def prefill_batched(self, x_p, metadata, conv_state, conv_weights, state_indices_tensor_p, num_prefills,
                    num_decode_tokens, num_prefill_tokens):
    """Drop-in for Qwen4ExpPLELayer._short_conv_dilated_prefill_batched (vLLM, Apache-2.0): the stock algorithm with
    ``transpose(1, 2).contiguous()`` on our kernel."""
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
    non_spec_query_start_loc = metadata.non_spec_query_start_loc
    if non_spec_query_start_loc is None:
        raise ValueError("query_start_loc is required for prefill short-conv")
    query_start_loc_p = non_spec_query_start_loc[-num_prefills - 1:] - num_decode_tokens
    has_initial_states_p = metadata.has_initial_states_p
    if has_initial_states_p is None:
        raise ValueError("has_initial_states_p is required for prefill short-conv")

    output = torch.empty_like(x_p)
    q_starts = query_start_loc_p.to(torch.int64)
    if state_indices_tensor_p.numel() < num_prefills:
        raise ValueError(f"state_indices_tensor_p size mismatch: got {state_indices_tensor_p.numel()}, "
                         f"need >= {num_prefills}.")
    if has_initial_states_p.numel() < num_prefills:
        raise ValueError(f"has_initial_states_p size mismatch: got {has_initial_states_p.numel()}, "
                         f"need >= {num_prefills}.")
    if num_prefills == 0 or x_p.numel() == 0:
        return output
    lengths = q_starts[1:] - q_starts[:-1]
    max_len = metadata.max_prefill_query_len
    if max_len <= 0:
        return output

    hidden_size = x_p.shape[1]
    positions = torch.arange(num_prefill_tokens, device=x_p.device, dtype=torch.int64)
    req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
    col_indices = positions - q_starts[req_indices]

    packed_tokens = x_p.new_zeros((num_prefills, max_len, hidden_size))
    packed_tokens[req_indices, col_indices] = x_p
    packed_tokens = transpose12(packed_tokens)                     # stock: .transpose(1, 2).contiguous()

    state_indices = state_indices_tensor_p[:num_prefills].to(device=conv_state.device, dtype=torch.int64)
    valid_state = state_indices != NULL_BLOCK_ID
    state_indices = torch.where(valid_state, state_indices, torch.zeros_like(state_indices))
    has_initial = has_initial_states_p[:num_prefills].to(device=conv_state.device, dtype=torch.bool)
    if self.conv_state_len > 0:
        if conv_state.shape[0] == 0:
            state = conv_state.new_zeros((num_prefills, hidden_size, self.conv_state_len), dtype=x_p.dtype)
        else:
            state = conv_state.index_select(0, state_indices)[..., : self.conv_state_len].to(x_p.dtype)
        use_initial_mask = (valid_state & has_initial).view(num_prefills, 1, 1)
        initial_state = torch.where(use_initial_mask, state, torch.zeros_like(state))
        history = torch.cat((initial_state, packed_tokens), dim=-1)
    else:
        history = packed_tokens

    conv_output = F.conv1d(history, conv_weights.unsqueeze(1).contiguous(), groups=history.size(1),
                           dilation=self.short_conv_dilation)
    conv_output = transpose12(F.silu(conv_output))                 # stock: .transpose(1, 2).contiguous()

    token_positions = torch.arange(max_len, device=x_p.device, dtype=torch.int64)
    valid_tokens = token_positions.view(1, max_len) < lengths.view(num_prefills, 1)
    valid_output_mask = valid_tokens & valid_state.to(device=x_p.device).view(num_prefills, 1)
    conv_output.masked_fill_(~valid_output_mask.unsqueeze(-1), 0)
    output.copy_(conv_output[req_indices, col_indices])

    if self.conv_state_len > 0 and conv_state.shape[0] > 0:
        state_starts = lengths.to(device=history.device, dtype=torch.int64).view(num_prefills, 1, 1)
        state_offsets = torch.arange(self.conv_state_len, device=history.device, dtype=torch.int64).view(
            1, 1, self.conv_state_len)
        next_state = history.gather(dim=2, index=(state_starts + state_offsets).expand(-1, history.size(1), -1))
        existing_state = conv_state.index_select(0, state_indices)
        existing_base_state = existing_state[..., : self.conv_state_len]
        update_mask = valid_state & (lengths.to(device=conv_state.device) > 0)
        safe_next_state = torch.where(update_mask.view(num_prefills, 1, 1), next_state.to(conv_state.dtype),
                                      existing_base_state)
        existing_state[..., : self.conv_state_len] = safe_next_state
        conv_state.index_copy_(0, state_indices, existing_state)
    return output


def install(model: torch.nn.Module) -> int:
    """Bind our prefill short conv on every Qwen4ExpPLELayer under `model`. Returns how many were switched."""
    if os.environ.get("R9K_PLE_CONV", "r9k") != "r9k" or not available():
        return 0
    n = 0
    for mod in model.modules():
        if type(mod).__name__ == "Qwen4ExpPLELayer" and hasattr(mod, "_short_conv_dilated_prefill_batched") \
                and hasattr(mod, "conv_state_len") and hasattr(mod, "short_conv_dilation"):
            mod._short_conv_dilated_prefill_batched = types.MethodType(prefill_batched, mod)
            n += 1
    if n:
        logger.info("r9700: r9k PLE short-conv prefill (tiled transposes) installed on %d layers", n)
    return n
