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
        # cross-stage slice (tests/test_ar8.py: 32 KB messages 96 -> 69 us, 64 KB 250 -> 157 with 8 KB slices; 0 = whole)
        self.cross_kb = int(float(os.environ.get("R9K_AR8_CROSS_KB", "8")) * 1024)
        # above this the hierarchy loses to RCCL (64 KB: 157 us sliced vs RCCL 104) and so does the flat two-shot (449):
        # should() declines and vLLM's communicator runs the message on RCCL. Only this 8-rank object looks at it.
        self.max_kb = int(float(os.environ.get("R9K_AR8_MAX_KB", "40")) * 1024)
        g = dist.get_process_group_ranks(group)           # global ranks of the TP group, in TP-rank order
        # every rank must create every sub-group (new_group is collective over the world); keep ours
        pairs = [dist.new_group(ranks=[g[2 * i], g[2 * i + 1]], backend="gloo") for i in range(4)]
        crosses = [dist.new_group(ranks=[g[0], g[2], g[4], g[6]], backend="gloo"),
                   dist.new_group(ranks=[g[1], g[3], g[5], g[7]], backend="gloo")]
        self.pair_pos = self.rank % 2                      # 0 = even: the switch's cross-chain sender; 1 = odd
        # R9K_AR8_PAIR_1S_KB (64): the pair stage is switch-local, where a 32-64 KB one-shot is fine; without it the
        # 4-row verify step's 32 KB ran the pair stage as a 14 us two-shot twice an all-reduce (profile 2026-10-30).
        pair_1s = float(os.environ.get("R9K_AR8_PAIR_1S_KB", "64"))
        self.pair = R9kAllReduceN(pairs[self.rank // 2], device, one_shot_kb=pair_1s)
        # the odd quartet's group exists (new_group is collective) but never carries traffic
        # R9K_AR8_CROSS_HOST (default 1; 0 keeps peer writes): a second cross object whose pools sit in pinned host memory, so the evens
        # push over one link to the host each instead of across four root complexes, where the peer stores stall
        # (the cliff above). Measured 2026-10-30 (tests/test_ar8.py, us, whole message): 8 KB 31.6 (P2P 21.6),
        # 16 KB 35.9 (41.6), 32 KB 47.4 (71.7), 64 KB 60.3 (RCCL 107). So messages from R9K_AR8_HOST_MIN_KB (12) up
        # take the host stage whole (one-shot to R9K_AR8_HOST_1S_KB, 128) up to R9K_AR8_HOST_MAX_KB (96): at 128 KB the
        # host path (103-148 us) no longer beats RCCL (109) and four-stream steps (16 rows) lost 5-10% with it, so 128 KB
        # and up stay on RCCL; 1-row messages keep P2P. GLM TP8 2026-10-30: 86.5 -> 100-102 tok/s single stream.
        self.cross = R9kAllReduceN(crosses[0], device) if self.pair_pos == 0 else None
        # Every rank must route a message the same way (an even in the ar8 kernels while its odd runs RCCL is a
        # deadlock -- 2026-10-30), so the host stage's existence and caps are agreed over the whole group and the
        # size rules below read only group-wide state (host_on, host_min, host_max), never the evens' objects.
        self.cross_host = None
        self.host_min = int(float(os.environ.get("R9K_AR8_HOST_MIN_KB", "12")) * 1024)
        self.host_max = int(float(os.environ.get("R9K_AR8_HOST_MAX_KB", "96")) * 1024)
        self.host_on = False
        if os.environ.get("R9K_AR8_CROSS_HOST", "1") == "1":
            h1 = float(os.environ.get("R9K_AR8_HOST_1S_KB", "128"))
            ch = R9kAllReduceN(crosses[0], device, host=True, one_shot_kb=h1, max_kb=self.host_max / 1024) \
                if self.pair_pos == 0 else None
            ok_h = ch is None or not ch.disabled
            oks_h = [None] * self.world_size
            dist.all_gather_object(oks_h, ok_h, group=group)
            if all(oks_h):
                self.cross_host = ch
                self.host_on = True
                self.max_kb = max(self.max_kb, self.host_max)
            else:
                logger.warning("r9700: ar8 host-memory cross stage unavailable on some rank; peer-write cross stage kept")
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

    # ------------------------------------------------------------------------------------ all-gather
    # The flat 8-rank one-shot all-gather (the logits every step, ~38 KB per rank) hits the same fabric cliff:
    # 1.34 ms per call in the 2026-10-14 profile. Same shape as the reduce: pair all-gather (2 pieces), cross
    # all-gather among the evens (8 pieces, four senders), pair all-gather again with the odd contributing zeros
    # so the even's full result lands as the first half on both. Concatenation order = TP rank order, as vLLM's.
    def should_ag(self, x, dim):
        if self.disabled or os.environ.get("R9K_AG8", "1") != "1":
            return self.flat.should_ag(x, dim)
        return self._ag_takes(x, dim) or self.flat.should_ag(x, dim)

    def _ag_takes(self, x, dim) -> bool:
        if not self.pair.should_ag(x, dim):
            return False
        d = dim + x.dim() if dim < 0 else dim
        shape2 = list(x.shape); shape2[d] *= 2
        shape8 = list(x.shape); shape8[d] *= 8
        n2 = x.numel() * 2 * x.element_size(); n8 = x.numel() * 8 * x.element_size()
        return (self.cross is None or n2 <= self.cross.ag_max) and n8 <= self.pair.ag_max

    def all_gather(self, x, dim=-1):
        if not (os.environ.get("R9K_AG8", "1") == "1" and self._ag_takes(x, dim)):
            return self.flat.all_gather(x, dim)
        d = dim + x.dim() if dim < 0 else dim
        y = self.pair.all_gather(x, d)                      # [.., 2 v, ..] = [even, odd] of this pair
        if self.pair_pos == 0:
            # R9K_AR8_HOST_AG=0 keeps the logits all-gather on the peer path (A/B knob; the eight-stream probe moved
            # 150 -> 129-140 with the host stage on, and the all-gather is the one message that grows with the batch)
            ch = self.cross_host if (self.host_on and os.environ.get("R9K_AR8_HOST_AG", "1") == "1"
                                     and y.numel() * y.element_size() >= self.host_min
                                     and self.cross_host.should_ag(y, d)) else self.cross
            z = ch.all_gather(y, d)                          # [.., 8 v, ..] in TP rank order (pairs in order)
            full = self.pair.all_gather(z, d)                # [.., 16 v, ..]: [evens' z | odd's zeros]
        else:
            full = self.pair.all_gather(self._zero_like(y, d, 4), d)
        return full.narrow(d, 0, x.shape[d] * 8).contiguous()

    def _zero_like(self, y, d, mult):
        shape = list(y.shape); shape[d] *= mult
        key = ("ag", tuple(shape), y.dtype)
        z = self._zeros.get(key)
        if z is None:
            z = self._zeros[key] = torch.zeros(shape, dtype=y.dtype, device=y.device)
        return z

    def _takes(self, x: torch.Tensor) -> bool:
        if self.disabled or x.dtype not in _DTYPE or not x.is_contiguous():
            return False
        n = x.numel() * x.element_size()
        if n == 0 or n % 16:
            return False
        nw = x.numel() * (4 if self.fp32 else x.element_size())      # bytes of the working copy
        if self.host_on and nw >= self.host_min:
            cap = min(self.pair.max_bytes, self.host_max)           # the same rule on evens and odds
        else:
            cap = self.pair.max_bytes if self.cross is None else min(self.pair.max_bytes, self.cross.max_bytes)
        return nw <= cap and n <= self.pair.max_bytes

    def should(self, x: torch.Tensor) -> bool:
        if x.numel() * x.element_size() > self.max_kb:
            return False                                     # RCCL: 104 us at 64 KB against 157 here, 449 flat
        return self._takes(x) or self.flat.should(x)

    def _zero(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.numel(), x.dtype)
        z = self._zeros.get(key)
        if z is None:
            z = self._zeros[key] = torch.zeros(x.numel(), dtype=x.dtype, device=x.device)
        return z

    def _cross(self, y: torch.Tensor) -> torch.Tensor:
        """The evens' 4-chain stage. Four senders x the message is what the fabric sees in flight; at 32 KB a message
        that is 128 KB, inside the measured cliff (tests/test_ar8.py: 95 us at 4 rows against 41 at 2). R9K_AR8_CROSS_KB
        (0 = off) caps the bytes a sender has in flight by running the stage over consecutive slices."""
        n = y.numel() * y.element_size()
        if self.host_on and n >= self.host_min and n <= self.host_max:
            return self.cross_host.all_reduce(y)              # whole: no cliff on the host path
        if not self.cross_kb or n <= self.cross_kb:
            return self.cross.all_reduce(y)
        per = max(16, (self.cross_kb // y.element_size() // 8) * 8)    # elements a slice (16-byte multiples)
        parts = [self.cross.all_reduce(y[i:i + per]) for i in range(0, y.numel(), per)]
        return torch.cat(parts)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if not self._takes(x):
            return self.flat.all_reduce(x)                   # callers check should() first: never above max_kb
        flat = x.view(-1)
        work = flat.float() if self.fp32 and x.dtype != torch.float32 else flat
        y = self.pair.all_reduce(work)                       # both pair members: the pair sum
        if self.pair_pos == 0:
            z = self._cross(y)                               # evens: the total, four senders on four switches
            if z.dtype != x.dtype:
                z = z.to(x.dtype)
            out = self.pair.all_reduce(z)                    # odd adds zeros -> both hold the total, unrounded
        else:
            out = self.pair.all_reduce(self._zero(flat))
        return out.view_as(x)
