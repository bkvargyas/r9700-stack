"""Exact-size pinned host memory + UVA views.

torch's pinned allocator (tensor.pin_memory(), torch.empty(pin_memory=True)) rounds every block up to a power of
two: a 19.4 GiB PLE table costs 32 GiB of locked RAM (the first stock bring-up was OOM-killed at 49.9 GB shmem per
worker). Here we allocate ordinary host memory and lock it with hipHostRegister(Portable|Mapped) at its exact size,
then take vLLM's UVA device view of it -- the same approach tcclaviger's expert offloader uses.
"""
from __future__ import annotations

import contextlib

import torch

_KEEP: list[torch.Tensor] = []
_REGISTER_PORTABLE_MAPPED = 3


def pinned_empty(shape, dtype) -> torch.Tensor:
    t = torch.empty(shape, dtype=dtype, device="cpu")
    n = t.numel() * t.element_size()
    if n:
        rc = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), n, _REGISTER_PORTABLE_MAPPED)
        if int(rc) != 0:  # fall back to torch's (rounding) allocator rather than fail
            import warnings
            warnings.warn(f"r9700: hipHostRegister failed ({int(rc)}); using torch pinned memory (rounds to 2^k)")
            t = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    if n and not t.is_pinned():
        raise RuntimeError("r9700: host buffer is not recognised as pinned; a UVA view would silently be a copy")
    _KEEP.append(t)       # registered memory must outlive every view; these are process-lifetime buffers
    return t


def uva_empty(shape, dtype) -> torch.Tensor:
    """Device-typed (UVA) view of a fresh exact-size pinned host buffer."""
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    host = pinned_empty(shape, dtype)
    v = get_accelerator_view_from_cpu_tensor(host)
    v._r9k_host = host
    return v


@contextlib.contextmanager
def exact_pinning():
    """Within the block, CPU ``Tensor.pin_memory()`` pins at exact size (hipHostRegister) instead of torch's 2^k
    rounding. Used around model construction, where vLLM's UVA offloader pins every offloaded parameter
    (+~22% host RAM for Flash-Next's experts otherwise). Restored on exit; not thread-scoped (construction is
    single-threaded per worker)."""
    stock = torch.Tensor.pin_memory

    def exact(self, *a, **k):
        if self.device.type != "cpu":
            return stock(self, *a, **k)
        out = pinned_empty(self.shape, self.dtype)
        out.copy_(self)
        return out

    torch.Tensor.pin_memory = exact
    try:
        yield
    finally:
        torch.Tensor.pin_memory = stock
