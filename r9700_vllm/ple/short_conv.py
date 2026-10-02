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


# A step's prefills are packed into [prefills, longest, 4*hidden] and the function holds about six tensors of that
# size at once. Stock packs ALL of them together, so one 3,800-token chunk next to seven 40-token prompts is eight
# rows of 3,800: 600 MiB per tensor for 4,080 tokens of work, where vLLM's memory profile (one sequence) saw 84 MiB.
# On four cards a soak of mixed-length prompts at 16 running sequences took the cards from 29.4 to 32.2 GiB in 40
# seconds and died in exactly this function (2026-10-02). We pack by length instead: groups, longest first, each
# at most SLACK times the step's own token count (a group of one is always allowed). Each sequence's convolution
# reads only its own tokens and state, so the result is the same bits; one prefill, or prefills of similar length,
# still run as a single group, exactly as stock.
SLACK = float(os.environ.get("R9K_PLE_CONV_SLACK", "1.25"))     # 0: one group always (stock's packing)


def length_groups(lengths: list[int], cap: int) -> list[list[int]]:
    """Indices grouped longest first so that len(group) * longest <= cap; a single sequence is always a group."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    groups: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        if cur and (len(cur) + 1) * lengths[cur[0]] > cap:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    return groups


def _conv_group(self, x_tok, rows, cols, n, max_len, lengths, state_indices, valid_state, has_initial, conv_state,
                conv_weights):
    """The stock algorithm for n prefills whose tokens are x_tok [tokens, 4*hidden] at (rows, cols) of the packed
    batch: returns the conv output for those tokens, in the same order, and writes the group's conv state back."""
    hidden_size = x_tok.shape[1]
    packed_tokens = x_tok.new_zeros((n, max_len, hidden_size))
    packed_tokens[rows, cols] = x_tok
    packed_tokens = transpose12(packed_tokens)                     # stock: .transpose(1, 2).contiguous()

    if self.conv_state_len > 0:
        if conv_state.shape[0] == 0:
            state = conv_state.new_zeros((n, hidden_size, self.conv_state_len), dtype=x_tok.dtype)
        else:
            state = conv_state.index_select(0, state_indices)[..., : self.conv_state_len].to(x_tok.dtype)
        use_initial_mask = (valid_state & has_initial).view(n, 1, 1)
        initial_state = torch.where(use_initial_mask, state, torch.zeros_like(state))
        history = torch.cat((initial_state, packed_tokens), dim=-1)
    else:
        history = packed_tokens

    conv_output = F.conv1d(history, conv_weights.unsqueeze(1).contiguous(), groups=history.size(1),
                           dilation=self.short_conv_dilation)
    conv_output = transpose12(F.silu(conv_output))                 # stock: .transpose(1, 2).contiguous()

    token_positions = torch.arange(max_len, device=x_tok.device, dtype=torch.int64)
    valid_tokens = token_positions.view(1, max_len) < lengths.view(n, 1)
    valid_output_mask = valid_tokens & valid_state.to(device=x_tok.device).view(n, 1)
    conv_output.masked_fill_(~valid_output_mask.unsqueeze(-1), 0)
    out = conv_output[rows, cols]

    if self.conv_state_len > 0 and conv_state.shape[0] > 0:
        state_starts = lengths.to(device=history.device, dtype=torch.int64).view(n, 1, 1)
        state_offsets = torch.arange(self.conv_state_len, device=history.device, dtype=torch.int64).view(
            1, 1, self.conv_state_len)
        next_state = history.gather(dim=2, index=(state_starts + state_offsets).expand(-1, history.size(1), -1))
        existing_state = conv_state.index_select(0, state_indices)
        existing_base_state = existing_state[..., : self.conv_state_len]
        update_mask = valid_state & (lengths.to(device=conv_state.device) > 0)
        safe_next_state = torch.where(update_mask.view(n, 1, 1), next_state.to(conv_state.dtype),
                                      existing_base_state)
        existing_state[..., : self.conv_state_len] = safe_next_state
        conv_state.index_copy_(0, state_indices, existing_state)
    return out


def prefill_batched(self, x_p, metadata, conv_state, conv_weights, state_indices_tensor_p, num_prefills,
                    num_decode_tokens, num_prefill_tokens):
    """Drop-in for Qwen4ExpPLELayer._short_conv_dilated_prefill_batched (vLLM, Apache-2.0): the stock algorithm with
    ``transpose(1, 2).contiguous()`` on our kernel, packed by length when the prefills of a step differ (see SLACK)."""
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

    positions = torch.arange(num_prefill_tokens, device=x_p.device, dtype=torch.int64)
    req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
    col_indices = positions - q_starts[req_indices]

    state_indices = state_indices_tensor_p[:num_prefills].to(device=conv_state.device, dtype=torch.int64)
    valid_state = state_indices != NULL_BLOCK_ID
    state_indices = torch.where(valid_state, state_indices, torch.zeros_like(state_indices))
    has_initial = has_initial_states_p[:num_prefills].to(device=conv_state.device, dtype=torch.bool)

    cap = max(int(max_len), int(SLACK * num_prefill_tokens))
    if SLACK <= 0 or num_prefills == 1 or num_prefills * int(max_len) <= cap:
        output.copy_(_conv_group(self, x_p, req_indices, col_indices, num_prefills, max_len, lengths, state_indices,
                                 valid_state, has_initial, conv_state, conv_weights))
        return output

    lens = lengths.tolist()                       # host sync, only on a step whose prefills differ in length
    for grp in length_groups(lens, cap):
        gt = torch.tensor(grp, device=x_p.device, dtype=torch.int64)
        local = torch.full((num_prefills,), -1, device=x_p.device, dtype=torch.int64)
        local[gt] = torch.arange(len(grp), device=x_p.device, dtype=torch.int64)
        rows = local[req_indices]
        sel = rows >= 0
        gs = gt.to(conv_state.device)
        output[sel] = _conv_group(self, x_p[sel], rows[sel], col_indices[sel], len(grp), max(lens[i] for i in grp),
                                  lengths[gt], state_indices[gs], valid_state[gs], has_initial[gs], conv_state,
                                  conv_weights)
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
