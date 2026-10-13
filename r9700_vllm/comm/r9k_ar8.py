"""Hierarchical exact all-reduce for eight R9700s on four PCIe switches (decode sizes).

Measured 2026-10-13 (tests/test_ar_nrank.py, HIP graphs): the flat one-shot N-rank kernel costs 6.8-10 us at four ranks on
any two switches, but 126 us at 5 KB and 288 us at 10 KB at eight ranks -- the 7-way fan-out of shader stores through
four root complexes collapses (fences make no difference; two-shot is 55-82 us, RCCL 82 us). A decode step of
GLM-5.3-Flash makes 90 of them, 22.9 of its 40 ms.

So the eight-rank reduce is three small collectives on sub-groups, each of which stays in its fast regime:
  1. PAIR   one-shot all-reduce inside each switch-local pair (ranks 2i, 2i+1 of the TP group): ~5 us
  2. CROSS  one-shot all-reduce of HALF the message among the four even ranks (first half) and, concurrently,
            among the four odd ranks (second half) -- four ranks on four switches: ~8-10 us
  3. PAIR   all-gather of the two halves inside the pair: ~5 us
Every rank ends with [evens' half, odds' half], bit-identical across ranks. By default the pair and cross steps
run on an fp32 copy and the result is rounded once at the end: tests/test_ar8.py shows it bit-identical to the
flat kernel's rank-ordered fp32 sum (2x the bytes in the first two steps plus two casts). R9K_AR8_FP32=0 keeps
the message in bf16 (the pair sum is rounded before the cross step: still identical across ranks, but up to a
few ulps from the once-rounded result where pair sums cancel; RCCL's ring rounds seven times).

Messages that cannot be split into 16-byte-aligned halves, or that exceed the sub-group caps, fall back to the flat
N-rank instance (two-shot) and the logits all-gather stays on it too. R9K_AR8=0 keeps the flat kernel.
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
        self.fp32 = os.environ.get("R9K_AR8_FP32", "1") == "1"      # exact by default (tests/test_ar8.py: max diff 0)
        g = dist.get_process_group_ranks(group)           # global ranks of the TP group, in TP-rank order
        # every rank must create every sub-group (new_group is collective over the world); keep ours
        pairs = [dist.new_group(ranks=[g[2 * i], g[2 * i + 1]], backend="gloo") for i in range(4)]
        crosses = [dist.new_group(ranks=[g[0], g[2], g[4], g[6]], backend="gloo"),
                   dist.new_group(ranks=[g[1], g[3], g[5], g[7]], backend="gloo")]
        self.pair_pos = self.rank % 2                      # 0 = even (first half), 1 = odd (second half)
        self.pair = R9kAllReduceN(pairs[self.rank // 2], device)
        self.cross = R9kAllReduceN(crosses[self.pair_pos], device)
        if self.pair.disabled or self.cross.disabled or self.pair._spg is None:
            logger.warning("r9700: ar8 sub-group setup failed (pair %s, cross %s, pair all-gather %s); flat kernel kept",
                           not self.pair.disabled, not self.cross.disabled, self.pair._spg is not None)
            return
        self.disabled = False
        logger.info("r9700: ar8 hierarchical all-reduce installed (rank %d: pair %d/%s, cross %s, %s)", self.rank,
                    self.rank // 2, "even" if self.pair_pos == 0 else "odd", "evens" if self.pair_pos == 0 else "odds",
                    "fp32 intermediate" if self.fp32 else "bf16 intermediate")

    # the flat instance decides for anything we do not take (and for every all-gather)
    def should_ag(self, x, dim):
        return self.flat.should_ag(x, dim)

    def all_gather(self, x, dim=-1):
        return self.flat.all_gather(x, dim)

    def _splits(self, x: torch.Tensor) -> bool:
        if self.disabled or x.dtype not in _DTYPE or not x.is_contiguous():
            return False
        n = x.numel()
        esz = 4 if self.fp32 else x.element_size()
        half = (n // 2) * esz
        if n % 2 or half % 16 or half == 0:
            return False
        # pair step carries the whole message, cross step a half, gather a half (ag cap in bytes of x's dtype)
        return n * esz <= self.pair.max_bytes and half <= self.cross.max_bytes and \
            (n // 2) * x.element_size() <= self.pair.ag_max

    def should(self, x: torch.Tensor) -> bool:
        return self._splits(x) or self.flat.should(x)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if not self._splits(x):
            return self.flat.all_reduce(x)
        flat = x.view(-1)
        work = flat.float() if self.fp32 and x.dtype != torch.float32 else flat
        y = self.pair.all_reduce(work)                       # both pair members hold the pair sum
        n = y.numel()
        h = y[: n // 2] if self.pair_pos == 0 else y[n // 2:]
        z = self.cross.all_reduce(h)                         # evens: first-half total, odds: second-half total
        if self.fp32 and x.dtype != torch.float32:
            z = z.to(x.dtype)
        out = self.pair.all_gather(z, dim=0)                 # [evens' half, odds' half] on both pair members
        return out.view_as(x)
