"""Exact-size pinned host memory + UVA views.

torch's pinned allocator (tensor.pin_memory(), torch.empty(pin_memory=True)) rounds every block up to a power of
two: a 19.4 GiB PLE table costs 32 GiB of locked RAM (the first stock bring-up was OOM-killed at 49.9 GB shmem per
worker). Here we allocate ordinary host memory and lock it with hipHostRegister(Portable|Mapped) at its exact size,
then take vLLM's UVA device view of it -- the same approach tcclaviger's expert offloader uses.
"""
from __future__ import annotations

import torch

_KEEP: list[torch.Tensor] = []
_REGISTER_PORTABLE_MAPPED = 3


def pinned_empty(shape, dtype) -> torch.Tensor:
    t = torch.empty(shape, dtype=dtype, device="cpu")
    n = t.numel() * t.element_size()
    if n:
        rc = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), n, _REGISTER_PORTABLE_MAPPED)
        if int(rc) != 0:  # fall back to torch's (rounding) allocator rather than fail
            t = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    _KEEP.append(t)       # registered memory must outlive every view; these are process-lifetime buffers
    return t


def uva_empty(shape, dtype) -> torch.Tensor:
    """Device-typed (UVA) view of a fresh exact-size pinned host buffer."""
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    host = pinned_empty(shape, dtype)
    v = get_accelerator_view_from_cpu_tensor(host)
    v._r9k_host = host
    return v
