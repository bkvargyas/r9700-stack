"""Hierarchical exact all-reduce for eight R9700s on four PCIe switches (decode sizes).

Measured 2026-10-13/14 (tests/test_ar_nrank.py, tests/test_ar8.py, HIP graphs, bf16 8 KB): the flat one-shot N-rank kernel
costs 6.8-10 us at four ranks on any two switches but 126-288 us at eight ranks on four, and the flat two-shot 45-55 us.
The fabric, not the kernel: once roughly 100-150 KB of GPU-issued stores are in flight across the four root complexes
at once (8 ranks x 3 cross-chain peers x 6 KB already), the stores stall for ~150-250 us; and a switch-local phase
adjacent to a cross-chain phase in which both quartets send costs 65-75 us however it is ordered. Four senders are
fine: 4 ranks on 4 switches push 8 KB each in 10 us. A decode step of GLM-5.3-Flash makes 90 all-reduces.

So only the EVEN ranks ever send cross-chain:
  1. PAIR   one-shot all-reduce inside each switch-local pair (TP ranks 2i, 2i+1): both hold the pair sum   ~6 us
  2. CROSS  one-shot all-reduce of the whole message among the four even ranks (one per switch); odds idle  ~10 us
  3. PAIR   one-shot all-reduce again: the even contributes the total, the odd a zero tensor -> both hold it  ~6 us
Measured 22 us for 8 KB bf16 (H2 in tests/test_ar8.py). Every sub-collective sums in fp32 in rank order and
rounds once, so all eight outputs are bit-identical. The bf16 message is rounded after step 1 and after step 2 (two
roundings; the flat kernels round once, RCCL's ring seven times). R9K_AR8_FP32=1 runs steps 1-2 on an fp32 copy
and casts once before step 3: bit-identical to the flat kernel's rank-ordered sum, but 45 us instead of 22 (the
fp32 cross step doubles the bytes in flight) -- opt-in; the quality gates (GSM8K / sanity) judge the bf16 default. Messages above the sub-group caps fall back to the flat instance (two-shot), as do the
all-gathers. R9K_AR8=0 keeps the flat kernel.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from .r9k_ar import R9kAllReduceN, _DTYPE

logger = init_logger("vllm." + __name__)


class R9kAllReduce8:
    def __init__(self, group, device, flat: R9kAllReduceN | None = None):
        self.disabled = True
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if self.world_size != 8:
            return
        self.flat = flat if flat is not None else R9kAllReduceN(group, device)
        if self.flat.disabled:
            return
        self.fp32 = os.environ.get("R9K_AR8_FP32", "0") == "1"      # bf16: 22 us; fp32 (exact): 45 us (tests/test_ar8.py)
        g = dist.get_process_group_ranks(group)           # global ranks of the TP group, in TP-rank order
        # every rank must create every sub-group (new_group is collective over the world); keep ours
        pairs = [dist.new_group(ranks=[g[2 * i], g[2 * i + 1]], backend="gloo") for i in range(4)]
        crosses = [dist.new_group(ranks=[g[0], g[2], g[4], g[6]], backend="gloo"),
                   dist.new_group(ranks=[g[1], g[3], g[5], g[7]], backend="gloo")]
        self.pair_pos = self.rank % 2                      # 0 = even: the switch's cross-chain sender; 1 = odd
        self.pair = R9kAllReduceN(pairs[self.rank // 2], device)
        # the odd quartet's group exists (new_group is collective) but never carries traffic
        self.cross = R9kAllReduceN(crosses[0], device) if self.pair_pos == 0 else None
        ok = not self.pair.disabled and (self.cross is None or not self.cross.disabled)
        oks = [None] * self.world_size
        dist.all_gather_object(oks, ok, group=group)
        if not all(oks):
            logger.warning("r9700: ar8 sub-group setup failed on some rank; flat kernel kept")
            return
        self._zeros: dict = {}
        self.disabled = False
        logger.info("r9700: ar8 hierarchical all-reduce installed (rank %d: pair %d, %s, %s intermediate)", self.rank,
                    self.rank // 2, "cross-chain sender" if self.pair_pos == 0 else "idle in the cross step",
                    "fp32" if self.fp32 else "bf16")

    # the flat instance decides for anything we do not take (and for every all-gather)
    def should_ag(self, x, dim):
        return self.flat.should_ag(x, dim)

    def all_gather(self, x, dim=-1):
        return self.flat.all_gather(x, dim)

    def _takes(self, x: torch.Tensor) -> bool:
        if self.disabled or x.dtype not in _DTYPE or not x.is_contiguous():
            return False
        n = x.numel() * x.element_size()
        if n == 0 or n % 16:
            return False
        nw = x.numel() * (4 if self.fp32 else x.element_size())      # bytes of the working copy
        cap = self.pair.max_bytes if self.cross is None else min(self.pair.max_bytes, self.cross.max_bytes)
        return nw <= cap and n <= self.pair.max_bytes

    def should(self, x: torch.Tensor) -> bool:
        return self._takes(x) or self.flat.should(x)

    def _zero(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.numel(), x.dtype)
        z = self._zeros.get(key)
        if z is None:
            z = self._zeros[key] = torch.zeros(x.numel(), dtype=x.dtype, device=x.device)
        return z

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if not self._takes(x):
            return self.flat.all_reduce(x)
        flat = x.view(-1)
        work = flat.float() if self.fp32 and x.dtype != torch.float32 else flat
        y = self.pair.all_reduce(work)                       # both pair members: the pair sum
        if self.pair_pos == 0:
            z = self.cross.all_reduce(y)                     # evens: the total, four senders on four switches
            if z.dtype != x.dtype:
                z = z.to(x.dtype)
            out = self.pair.all_reduce(z)                    # odd adds zeros -> both hold the total, unrounded
        else:
            out = self.pair.all_reduce(self._zero(flat))
        return out.view_as(x)
