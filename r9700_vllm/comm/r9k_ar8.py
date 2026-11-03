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
fp32 cross step doubles the bytes in flight) -- opt-in; the quality gates (GSM8K / sanity) judge the bf16 default.
Since 2026-10-30 the cross step goes through pinned host memory (R9K_AR8_CROSS_HOST, below) from 12 KB.

Large messages (2026-11-02, DeepSeek-V4.1's [tokens, 5120] bf16 at 10-200 tokens a step), above R9K_AR8_SPLIT_KB (32)
and up to R9K_AR8_HOST_MAX_KB (2048) -- the SPLIT path, one fused kernel (kernels/r9k_ar.hip r9k_ar8_fused):
  1. pair reduce-scatter: each member of a pair ends with the pair sum of its half (even: first, odd: second)
  2. each quartet (evens / odds) all-reduces its half through its host pools: write, reduce our quarter, pull
  3. pair all-gather of the two halves
The pair traffic (GPU <-> switch links) overlaps the cross traffic (the switch's uplink) sub-step by sub-step. Same two
roundings in the same order as the small path, so the bits are those of the three-stage composition. Per uplink the
cross step moves 1 S up / 1.5 S down, the all-reduce floor for four switches with GPU-side sums. Measured (HIP graphs,
us, tests/test_ar8.py): 240 KB 83 (RCCL 124), 640 KB 157 (182), 1 MB 230 (240), 2 MB 418 (402). Larger messages
(prefill) and anything this object declines run on RCCL (should() is False); R9K_AR8=0 keeps the flat kernel.
"""
from __future__ import annotations

import ctypes
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
        self.max_kb = self.max_kb_env = int(float(os.environ.get("R9K_AR8_MAX_KB", "40")) * 1024)
        g = dist.get_process_group_ranks(group)           # global ranks of the TP group, in TP-rank order
        # every rank must create every sub-group (new_group is collective over the world); keep ours
        pairs = [dist.new_group(ranks=[g[2 * i], g[2 * i + 1]], backend="gloo") for i in range(4)]
        crosses = [dist.new_group(ranks=[g[0], g[2], g[4], g[6]], backend="gloo"),
                   dist.new_group(ranks=[g[1], g[3], g[5], g[7]], backend="gloo")]
        self.pair_pos = self.rank % 2                      # 0 = even: the switch's cross-chain sender; 1 = odd
        # R9K_AR8_PAIR_1S_KB (64): the pair stage is switch-local, where a 32-64 KB one-shot is fine; without it the
        # 4-row verify step's 32 KB ran the pair stage as a 14 us two-shot twice an all-reduce (profile 2026-10-30).
        pair_1s = float(os.environ.get("R9K_AR8_PAIR_1S_KB", "64"))
        self.host_max = int(float(os.environ.get("R9K_AR8_HOST_MAX_KB", "2048")) * 1024)
        # R9K_AR8_PAIR_MAX_KB: the pair stage's two-shot cap; by default it covers every message the host stage takes
        # (x2 for the fp32 working copy), else those sizes would fall to the flat 8-rank kernel.
        pair_max = float(os.environ.get("R9K_AR8_PAIR_MAX_KB", "0")) or \
            max(float(os.environ.get("R9K_ARN_MAX_KB", "512")), self.host_max / 1024 * (2 if self.fp32 else 1))
        self.pair = R9kAllReduceN(pairs[self.rank // 2], device, one_shot_kb=pair_1s, max_kb=pair_max, sub=True,
                                  extra2=4)          # [6 s, 10 s): r9k_ar8_fused's RS / AG areas
        # R9K_AR8_PAIR_WPB: the pair two-shot's words a block per chunk. The N-rank default (1024) gives an 80 KB pair
        # message 2 blocks and 38 us; 8+ blocks take 14 us (tests/test_ar8.py PARTS=4, 2026-11-02).
        self.pair.words_per_block2 = int(os.environ.get("R9K_AR8_PAIR_WPB", "256"))
        # the odd quartet carries traffic only in the split path (its host pools; never peer writes)
        # R9K_AR8_CROSS_HOST (default 1; 0 keeps peer writes): a second cross object whose pools sit in pinned host memory, so the evens
        # push over one link to the host each instead of across four root complexes, where the peer stores stall
        # (the cliff above). Measured 2026-10-30 (tests/test_ar8.py, us, whole message): 8 KB 31.6 (P2P 21.6),
        # 16 KB 35.9 (41.6), 32 KB 47.4 (71.7), 64 KB 60.3 (RCCL 107). So messages from R9K_AR8_HOST_MIN_KB (12) up
        # take the host stage whole; 1-row messages keep P2P. GLM TP8 2026-10-30: 86.5 -> 100-102 tok/s single stream.
        # To 2026-11-01 the stage stopped at 96 KB (the one-shot moves 3 S a rank and lost to RCCL at 128 KB), which put
        # every DeepSeek-V4.1 step above ~9 tokens on RCCL at 1-2 ms an all-reduce. Since 2026-11-02: one-shot to
        # R9K_AR8_HOST_1S_KB (96), above it the host-memory two-shot (each even writes its own pool, reduces its quarter
        # from the peers' pools, publishes it, pulls the other three: 2.5 S of PCIe a rank) up to R9K_AR8_HOST_MAX_KB
        # (2048; at 1.28 MB the split path ties RCCL and at 2 MB loses 4 %, not worth a second rule); larger messages (prefill) stay on RCCL.
        # SPLIT (R9K_AR8_SPLIT_KB, 32: the quiet-box sweep of 2026-11-02 -- 8 tokens 78 -> 47 us, 6 tokens 37 us; 64 and 96 lose on 60-90 KB): above it the two pair stages are a pair REDUCE-SCATTER and a pair ALL-GATHER
        # (each moves half the message once instead of the whole message's all-reduce), and the cross stage runs on
        # the halves in BOTH quartets at once -- evens the first half, odds the second, each through its quartet's
        # host pools. The host path has no cross-root-complex stores, so the odds may send too; the two GPUs of a
        # switch share its uplink, which still carries 1 S up / 1.5 S down, but their two links issue it. Same
        # roundings as the whole-message path (pair sum, then the four pair sums in rank order), same bits.
        self.cross = R9kAllReduceN(crosses[0], device, sub=True) if self.pair_pos == 0 else None
        # Every rank must route a message the same way (an even in the ar8 kernels while its odd runs RCCL is a
        # deadlock -- 2026-10-30), so the host stage's existence and caps are agreed over the whole group and the
        # size rules below read only group-wide state (host_on, host_min, host_max), never the evens' objects.
        self.cross_host = None
        self.host_min = int(float(os.environ.get("R9K_AR8_HOST_MIN_KB", "12")) * 1024)
        self.host_on = False
        self.split_min = int(float(os.environ.get("R9K_AR8_SPLIT_KB", "32")) * 1024)
        self.split = os.environ.get("R9K_AR8_SPLIT", "1") == "1" and not self.fp32 and self.split_min < self.host_max \
            and self.pair.can_split()
        if os.environ.get("R9K_AR8_CROSS_HOST", "1") == "1":
            h1 = float(os.environ.get("R9K_AR8_HOST_1S_KB", "96"))
            # pool cap: whole messages to split_min, halves above it (+16 bytes: an odd number of words)
            hcap = max(self.split_min, (self.host_max + 1) // 2 + 16) if self.split else self.host_max
            # every rank builds its quartet's host stage: the odds carry the second halves in the split path
            ch = R9kAllReduceN(crosses[self.pair_pos], device, host=True, one_shot_kb=h1, max_kb=hcap / 1024, sub=True)
            ok_h = not ch.disabled
            oks_h = [None] * self.world_size
            # the evens must also agree on the one-shot / two-shot boundary (each picks it locally per message)
            dist.all_gather_object(oks_h, (ok_h, ch.max1 if ok_h else None, self.split), group=group)
            if all(o[0] for o in oks_h) and len({o[1] for o in oks_h}) == 1:
                self.split = all(o[2] for o in oks_h)
                self.cross_host = ch
                self.host_on = True
            else:
                logger.warning("r9700: ar8 host-memory cross stage unavailable on some rank; peer-write cross stage kept")
        if not self.host_on:
            self.split = False
        # R9K_AR8_FUSED (1): the split path as one kernel (kernels/r9k_ar.hip r9k_ar8_fused), whose role blocks
        # overlap the pair traffic with the cross traffic; 0 = three kernels. R9K_AR8_FUSED_K / _W: sub-steps a lane
        # (8) and words a sub-step per part (512), which set the lanes (2 MB: 458 us at 8 x 256, 418 at 8 x 512, 530
        # at 2 x 512). The kernel addresses both pools by fixed offsets, so the pool sizes must match across the
        # pair and the quartet (agreed below).
        L = self.pair.L
        fz = self.split and os.environ.get("R9K_AR8_FUSED", "1") == "1" and hasattr(L, "r9k_ar8_fused") \
            and self.pair.can_split() and self.cross_host is not None and self.cross_host._two_host
        if fz:
            L.r9k_ar8_fused.restype = ctypes.c_int
            L.r9k_ar8_fused.argtypes = [ctypes.POINTER(ctypes.c_long)] * 4 + [ctypes.c_int] + [ctypes.c_long] * 6 + \
                [ctypes.c_int, ctypes.c_long] + [ctypes.c_int] * 3
            L.r9k_ar8_fused_lanes.restype = ctypes.c_int
            if hasattr(L, "r9k_ar8_fused_t"):         # the same kernel with a stage trace (tests/test_ar8.py PARTS=5)
                L.r9k_ar8_fused_t.restype = ctypes.c_int
                L.r9k_ar8_fused_t.argtypes = L.r9k_ar8_fused.argtypes + [ctypes.c_long]
                L.r9k_ar8_fused_trace_words.restype = ctypes.c_int
                L.r9k_wallclock_khz.restype = ctypes.c_int
        fzs = [None] * self.world_size
        dist.all_gather_object(fzs, (fz, self.pair.slot2 if fz else 0, self.cross_host.slot2 if fz else 0), group=group)
        self.fused = all(f[0] for f in fzs) and len({f[1:] for f in fzs}) == 1
        if self.fused:
            self._fseq = torch.zeros(5 * L.r9k_ar8_fused_lanes(), dtype=torch.int32, device=self.pair.device)
            self._flanes = L.r9k_ar8_fused_lanes()
            self._fk = int(os.environ.get("R9K_AR8_FUSED_K", "8"))
            self._fw = int(os.environ.get("R9K_AR8_FUSED_W", "512"))
            self._fw1 = int(os.environ.get("R9K_AR8_FUSED_W1", "1024"))
        ok = not self.pair.disabled and (self.cross is None or not self.cross.disabled)
        oks = [None] * self.world_size
        dist.all_gather_object(oks, (ok, self.pair.max_bytes, self.host_max, self.split_min, self.host_min), group=group)
        if not all(o[0] for o in oks):
            logger.warning("r9700: ar8 sub-group setup failed on some rank; flat kernel kept")
            return
        # group-wide caps for the size rules (a rank with a different environment must still route like the others)
        self.pair_cap = min(o[1] for o in oks)
        self.host_max = min(o[2] for o in oks)
        self.split_min = min(o[3] for o in oks)
        self.host_min = min(o[4] for o in oks)
        if self.host_on:
            self.max_kb = max(self.max_kb_env, self.host_max)
        self._zeros: dict = {}
        # the odd's zero contribution to the last pair stage: one buffer of the largest message, allocated before any
        # graph capture (a buffer grown later would leave captured graphs pointing at freed memory)
        self._zbuf = torch.zeros(self.pair_cap, dtype=torch.uint8, device=self.pair.device) if self.pair_pos else None
        self._setup_oneshot(group, pairs, crosses, device)     # package C: the one-kernel small path (collective)
        self.disabled = False
        logger.info("r9700: ar8 hierarchical all-reduce installed (rank %d: pair %d, %s, %s intermediate; %s)",
                    self.rank, self.rank // 2, "cross-chain sender" if self.pair_pos == 0 else "idle in the small path",
                    "fp32" if self.fp32 else "bf16",
                    f"split {'fused' if self.fused else '3 kernels'} {self.split_min >> 10}-{self.host_max >> 10} KiB"
                    if self.split else "no split path")

    # ------------------------------------------------------------------------------------ all-gather
    # The flat 8-rank one-shot all-gather (the logits every step, ~38 KB per rank) hits the same fabric cliff:
    # 1.34 ms per call in the 2026-10-14 profile. Same shape as the reduce: pair all-gather (2 pieces), cross
    # all-gather among the evens (8 pieces, four senders), pair all-gather again with the odd contributing zeros
    # so the even's full result lands as the first half on both. Concatenation order = TP rank order, as vLLM's.
    def should_ag(self, x, dim):
        if self.disabled or os.environ.get("R9K_AG8", "1") != "1":
            return self.flat.should_ag(x, dim)
        if self._ag_os_takes(x, dim):
            return True
        # the flat 8-rank kernel only below R9K_AG8_FLAT_KB gathered (default never: the fabric cliff, see _setup_ag_os)
        return self._ag_takes(x, dim) or (self.flat.should_ag(x, dim)
                                          and x.numel() * x.element_size() * 8 <= self.ag_flat_max)

    def _ag_takes(self, x, dim) -> bool:
        if not self.pair.should_ag(x, dim):
            return False
        d = dim + x.dim() if dim < 0 else dim
        shape2 = list(x.shape); shape2[d] *= 2
        shape8 = list(x.shape); shape8[d] *= 8
        n2 = x.numel() * 2 * x.element_size(); n8 = x.numel() * 8 * x.element_size()
        return (self.cross is None or n2 <= self.cross.ag_max) and n8 <= self.pair.ag_max

    def all_gather(self, x, dim=-1):
        if os.environ.get("R9K_AG8", "1") == "1" and self._ag_os_takes(x, dim):
            return self._ag_oneshot(x, dim)
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
            cap = min(self.pair_cap, self.host_max)                 # the same rule on evens and odds
        else:
            cap = self.pair_cap if self.cross is None else min(self.pair_cap, self.cross.max_bytes)
        return nw <= cap and n <= self.pair_cap

    def should(self, x: torch.Tensor) -> bool:
        if x.numel() * x.element_size() > self.max_kb:
            return False                                     # RCCL (prefill sizes)
        return self._takes(x) or self.flat.should(x)

    def _zero(self, x: torch.Tensor) -> torch.Tensor:
        return self._zbuf[: x.numel() * x.element_size()].view(x.dtype)

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

    def fused_lanes(self, nbytes: int) -> tuple[int, int]:
        """(lanes, sub-steps) of the fused split kernel for a message: each lane is an independent pipeline of
        five role blocks over its slice of every part, cut into sub-steps of ~_fw words."""
        cq = -(-(-(-(nbytes // 16) // 2)) // 4)               # words of a part: ceil(ceil(n16 / 2) / 4)
        nl = max(1, min(self._flanes, -(-cq // (self._fk * self._fw))))
        k = max(1, min(64, -(-cq // (nl * self._fw))))
        if nl == 1:
            # one lane (parts <= _fk x _fw words, messages <= ~512 KiB): sub-steps of R9K_AR8_FUSED_W1 (1024) words.
            # The stage trace (tests/test_ar8.py PARTS=5, 2026-11-03) shows each role's host-memory reads as the cost
            # and every extra sub-step or lane as one more round of host flag polls: 240 KiB 81.7 us at 4 x 480 words
            # -> 77.1 at 2 x 960; 480 KiB 140.0 at 8 x 480 -> 122.5 at 4 x 960; more lanes lose at both sizes.
            k = max(1, min(64, -(-cq // self._fw1)))
        return nl, k

    def _split(self, flat: torch.Tensor, trace: torch.Tensor | None = None) -> torch.Tensor:
        """Pair reduce-scatter -> host cross stage on our half (our quartet) -> pair all-gather; one fused kernel
        (self.fused) or three. trace (tests): uint64 [r9k_ar8_fused_trace_words()] for the stage timestamps."""
        n, es = flat.numel(), flat.element_size()
        if self.fused:
            out = torch.empty_like(flat)
            nl, k = self.fused_lanes(n * es)
            ch = self.cross_host
            fn = self.pair.L.r9k_ar8_fused if trace is None else self.pair.L.r9k_ar8_fused_t
            extra = () if trace is None else (trace.data_ptr(),)
            rc = fn(
                self.pair._sp2, self.pair._fp2, ch._sp2, ch._fp2, self.rank, self._fseq.data_ptr(), self.pair.slot2,
                ch.slot2, flat.data_ptr(), out.data_ptr(), n, _DTYPE[flat.dtype], torch.cuda.current_stream().cuda_stream,
                nl, k, 0, *extra)
            if rc:
                raise RuntimeError(f"r9k_ar8_fused failed ({rc}) n={n}")
            return out
        buf = torch.empty_like(flat)
        self.pair.reduce_scatter(flat, buf)                  # our half of the pair sum
        off, ln = self.pair.chunk(n, es)
        part = buf[off:off + ln]
        self.cross_host.all_reduce(part, out=part)           # our quartet's total of that half, in place
        return self.pair.all_gather_chunks(buf)

    # ------------------------------------------------------------------ one-kernel small path (package C, 2026-11-03)
    # R9K_AR8_ONESHOT (1; 0 = the stage kernels as before): every message up to R9K_AR8_OS_KB (64) is ONE kernel
    # (kernels/r9k_ar.hip r9k_ar8_os) whose three phases are the split path's stages -- pair reduce-scatter (P2P),
    # both quartets' exchange of the halves (pinned host pools), pair all-gather (P2P) -- joined by release / acquire
    # flags instead of kernel boundaries. Same two roundings in the same order as the stage kernels, so the same bits.
    # Quiet box, HIP graphs, wpb 256 (notes/exl3-p4-C.md): 8 / 16 / 32 / 40 / 60 / 64 KiB 14.6 / 16.5 / 22.1 / 25.2 /
    # 34.2 / 36.7 us against the stage / split kernels' 21 / 34.5 / 46 / 34.7 / 37.6 / 38.3 (at wpb 128 the 60-64 KiB
    # sizes lost, 39.8 vs 37.6: wpb 256 is what makes the 64 KiB cap pay). R9K_AR8_OS_P2P_KB (0): peer writes for phase B up
    # to that size -- measured 83-153 us at every size (eight cross-chain senders: the fabric cliff); keep 0.
    # R9K_AR8_OS_WPB (256): words of a half a block (256 beat 128: 32 KiB 22.3 vs 23.6, 60 KiB 34.5 vs 39.2),
    # R9K_AR8_OS_NB (16) the block cap, R9K_AR8_OS_NT (0 = by size) threads a block.
    # R9K_AR_SPIN_TIMEOUT_MS (0 = off): a flag wait longer than this traps, after printing who waited for what.
    # all_reduce_moe: the same kernel, its message formed from the r9k expert tier's per-slot f32 outputs + the shared
    # expert's output (r9700_vllm/moe/exl3_combine.py), bit-identical to the unfused add + all-reduce.
    os_on = False
    os_max = 0

    def _setup_oneshot(self, group, pairs, crosses, device) -> None:
        self._os_tmo = int(os.environ.get("R9K_AR_SPIN_TIMEOUT_MS", "0"))
        self._os_diag = 0
        if self._os_tmo > 0:
            from .r9k_ar import host_mapped_words
            hm = host_mapped_words(16)
            if hm is not None:
                self._os_diag_keep = hm
                self._os_diag = hm[2]
                self._os_watchdog(hm[1])
        self._setup_ar_os(group, pairs, crosses, device)
        self._setup_ag_os(group, pairs, crosses, device)

    def _setup_ar_os(self, group, pairs, crosses, device) -> None:
        from .r9k_ar import R9kPool
        L = self.pair.L
        want = os.environ.get("R9K_AR8_ONESHOT", "1") == "1" and not self.fp32 and hasattr(L, "r9k_ar8_os")
        cfg = (want, float(os.environ.get("R9K_AR8_OS_KB", "64")), float(os.environ.get("R9K_AR8_OS_P2P_KB", "0")),
               int(os.environ.get("R9K_AR8_OS_WPB", "256")), int(os.environ.get("R9K_AR8_OS_NB", "16")),
               int(os.environ.get("R9K_AR8_OS_NT", "0")))
        cfgs = [None] * self.world_size
        dist.all_gather_object(cfgs, cfg, group=group)       # every rank must take the same branch (pools are collective)
        if not all(c[0] for c in cfgs) or len(set(cfgs)) != 1:
            if any(c[0] for c in cfgs):
                logger.warning("r9700: ar8 one-kernel path off: ranks disagree on R9K_AR8_ONESHOT / _OS_* (%s)", cfgs)
            return
        _, os_kb, xp_kb, self._os_wpb, nbcap, self._os_nt = cfg
        P = ctypes.POINTER(ctypes.c_long)
        L.r9k_ar8_os.restype = ctypes.c_int
        L.r9k_ar8_os.argtypes = [P] * 6 + [ctypes.c_int] + [ctypes.c_long] * 5 + [ctypes.c_int, ctypes.c_long] + \
            [ctypes.c_int] * 4 + [ctypes.c_long]
        L.r9k_ar8_os_moe.restype = ctypes.c_int
        L.r9k_ar8_os_moe.argtypes = [P] * 6 + [ctypes.c_int] + [ctypes.c_long] * 6 + [ctypes.c_int] * 5 + \
            [ctypes.c_long] * 3 + [ctypes.c_int] * 4 + [ctypes.c_long]
        for f in ("r9k_ar8_os_max_blocks", "r9k_ar8_os_flag_row", "r9k_ar8_os_max_slots"):
            getattr(L, f).restype = ctypes.c_int
        MB, FR = L.r9k_ar8_os_max_blocks(), L.r9k_ar8_os_flag_row()
        self._os_nb = max(1, min(nbcap, MB))
        self._os_slots = L.r9k_ar8_os_max_slots()
        slot = max(1, int(os_kb * 1024) // 16)               # words: the largest message, whole
        self._os_slot = slot
        pp = R9kPool(pairs[self.rank // 2], device, 4 * slot * 16, 2 * FR * 4)
        hp = R9kPool(crosses[self.pair_pos], device, 2 * slot * 16, 4 * FR * 4, host=True)
        xp = R9kPool(crosses[self.pair_pos], device, 8 * slot * 16, 4 * FR * 4) if xp_kb > 0 else None
        ok = pp.ok and hp.ok and (xp is None or xp.ok)
        oks = [None] * self.world_size
        dist.all_gather_object(oks, ok, group=group)
        if not all(oks):
            logger.warning("r9700: ar8 one-kernel path: pool setup failed on some rank; stage kernels kept")
            return
        self._os_pools = (pp, hp, xp)
        self._os_args = (pp.sp, pp.fp, hp.sp, hp.fp, xp.sp if xp else None, xp.fp if xp else None)
        self._os_seq = torch.zeros(MB, dtype=torch.int32, device=self.pair.device)
        self.os_max = slot * 16
        self.os_xp = int(xp_kb * 1024) if xp is not None else 0
        self.os_on = True
        logger.info("r9700: ar8 one-kernel all-reduce for messages <= %d KiB (rank %d; phase-B transport: host pools%s; "
                    "%d words a block, <= %d blocks; spin timeout %s)", self.os_max >> 10, self.rank,
                    f", peer writes <= {self.os_xp >> 10} KiB" if self.os_xp else "", self._os_wpb, self._os_nb,
                    f"{self._os_tmo} ms" if self._os_tmo > 0 else "off")

    def _os_watchdog(self, arr) -> None:
        """R9K_AR_SPIN_TIMEOUT_MS: print the kernel's timeout record before its trap takes the process down."""
        import threading
        import time as _t

        def run():
            while True:
                if arr[0] == 0x52394B54:
                    phase = {1: "A (partner's reduce-scatter push)", 2: "B (quartet peers' pair sums)",
                             3: "C (partner's all-gather push)", 4: "all-gather 1 (quartet peers' pieces)",
                             5: "all-gather 2 (partner's pieces)"}.get(arr[3], str(arr[3]))
                    msg = (f"r9700: ar8 one-kernel all-reduce SPIN TIMEOUT ({self._os_tmo} ms): TP rank {arr[1]} "
                           f"block {arr[2]} phase {phase} waited for flag {arr[4]} (call {arr[6]}), saw {arr[5]} -- a "
                           f"peer is not running this collective (routing disagreement or a hung rank); trapping")
                    logger.error(msg)
                    print(msg, flush=True)
                    return
                _t.sleep(0.02)
        threading.Thread(target=run, name="r9k-ar8-watchdog", daemon=True).start()

    # ------------------------------------------------------------------ one-kernel all-gather (package C, 2026-11-03)
    # R9K_AG8_ONESHOT (1): gathers of up to R9K_AG8_OS_KB (192: 6 rows of the logits; at 8 rows it ties RCCL, 234 vs 229
    # us) a rank as ONE kernel (kernels/r9k_ar.hip r9k_ag8_os):
    # own piece -> own host pool + the partner (P2P); the three quartet peers' pieces from host memory -> out + the
    # partner; the partner quartet's four from our pair pool. Nothing crosses a root complex. Before it, gathers whose
    # 8x size exceeded the pair path's R9K_AG_MAX_KB fell to the FLAT eight-rank one-shot (every rank pushing into all
    # seven peers: the fabric cliff), 0.65-1.07 ms for Engram's [T, 3200] and 3.2-3.6 ms for the logits [<= 4, 16160]
    # against RCCL's 0.2 ms (notes/exl3-ledger.md 1b). R9K_AG8_FLAT_KB (0): the largest gathered size (8x the piece)
    # still allowed on that flat kernel; above it the call goes to RCCL. R9K_AG8_OS_WPB (256) words a block.
    ag_os_on = False
    ag_os_max = 0
    ag_flat_max = 0
    _os_tmo = 0
    _os_diag = 0

    def _setup_ag_os(self, group, pairs, crosses, device) -> None:
        from .r9k_ar import R9kPool
        L = self.pair.L
        self.ag_flat_max = int(float(os.environ.get("R9K_AG8_FLAT_KB", "0")) * 1024)
        want = os.environ.get("R9K_AG8_ONESHOT", "1") == "1" and hasattr(L, "r9k_ag8_os") and hasattr(L, "r9k_ar8_os")
        cfg = (want, float(os.environ.get("R9K_AG8_OS_KB", "192")), int(os.environ.get("R9K_AG8_OS_WPB", "256")),
               int(os.environ.get("R9K_AG8_OS_NB", "16")), int(os.environ.get("R9K_AG8_OS_NT", "0")),
               self.ag_flat_max)
        cfgs = [None] * self.world_size
        dist.all_gather_object(cfgs, cfg, group=group)
        if not all(c[0] for c in cfgs) or len(set(cfgs)) != 1:
            if any(c[0] for c in cfgs):
                logger.warning("r9700: ag8 one-kernel path off: ranks disagree on R9K_AG8_* (%s)", cfgs)
            self.ag_flat_max = min(c[5] for c in cfgs)
            return
        _, kb, self._ag_wpb, nbcap, self._ag_nt, _ = cfg
        P = ctypes.POINTER(ctypes.c_long)
        L.r9k_ag8_os.restype = ctypes.c_int
        L.r9k_ag8_os.argtypes = [P] * 4 + [ctypes.c_int] + [ctypes.c_long] * 6 + [ctypes.c_int, ctypes.c_long] + \
            [ctypes.c_int] * 3 + [ctypes.c_long]
        L.r9k_ar8_os_max_blocks.restype = ctypes.c_int
        L.r9k_ar8_os_flag_row.restype = ctypes.c_int
        MB, FR = L.r9k_ar8_os_max_blocks(), L.r9k_ar8_os_flag_row()
        self._ag_nb = max(1, min(nbcap, MB))
        slot = max(1, int(kb * 1024) // 16)
        pp = R9kPool(pairs[self.rank // 2], device, 8 * slot * 16, 2 * FR * 4)
        hp = R9kPool(crosses[self.pair_pos], device, 2 * slot * 16, 4 * FR * 4, host=True)
        oks = [None] * self.world_size
        dist.all_gather_object(oks, pp.ok and hp.ok, group=group)
        if not all(oks):
            logger.warning("r9700: ag8 one-kernel path: pool setup failed on some rank; previous all-gather paths kept")
            return
        self._ag_pools = (pp, hp)
        self._ag_args = (pp.sp, pp.fp, hp.sp, hp.fp)
        self._ag_slot = slot
        self._ag_seq = torch.zeros(MB, dtype=torch.int32, device=self.pair.device)
        self.ag_os_max = slot * 16
        self.ag_os_on = True
        logger.info("r9700: ag8 one-kernel all-gather for pieces <= %d KiB a rank (rank %d; flat 8-rank kernel %s)",
                    self.ag_os_max >> 10, self.rank,
                    f"<= {self.ag_flat_max >> 10} KiB gathered" if self.ag_flat_max else "never (RCCL instead)")

    def _ag_os_takes(self, x, dim) -> bool:
        if not self.ag_os_on or not x.is_contiguous() or x.element_size() not in (2, 4) or x.numel() == 0 \
                or not (-x.dim() <= dim < x.dim()):
            return False
        rows, cols = R9kAllReduceN._ag_shape(x, dim)
        return (cols * x.element_size()) % 16 == 0 and x.numel() * x.element_size() <= self.ag_os_max

    def _ag_oneshot(self, x, dim):
        d = dim + x.dim() if dim < 0 else dim
        rows, cols = R9kAllReduceN._ag_shape(x, d)
        shape = list(x.shape)
        shape[d] *= 8
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        n16 = x.numel() * x.element_size() // 16
        nd = max(1, min(self._ag_nb, -(-n16 // self._ag_wpb)))
        rc = self.pair.L.r9k_ag8_os(*self._ag_args, self.rank, self._ag_seq.data_ptr(), self._ag_slot, x.data_ptr(),
                                    out.data_ptr(), rows, cols, x.element_size(), torch.cuda.current_stream().cuda_stream,
                                    nd, self._ag_nt, self._os_tmo, self._os_diag)
        if rc:
            raise RuntimeError(f"r9k_ag8_os failed ({rc}) shape={tuple(x.shape)} dim={dim}")
        return out

    def _os_nd(self, nbytes: int) -> int:
        c2 = -(-(nbytes // 16) // 2)
        return max(1, min(self._os_nb, -(-c2 // self._os_wpb)))

    def _oneshot(self, flat: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(flat)
        n = flat.numel() * flat.element_size()
        rc = self.pair.L.r9k_ar8_os(
            *self._os_args, self.rank, self._os_seq.data_ptr(), self._os_slot, flat.data_ptr(), out.data_ptr(),
            flat.numel(), _DTYPE[flat.dtype], torch.cuda.current_stream().cuda_stream, self._os_nd(n), self._os_nt,
            int(n <= self.os_xp), self._os_tmo, self._os_diag)
        if rc:
            raise RuntimeError(f"r9k_ar8_os failed ({rc}) n={n}")
        return out

    def moe_takes(self, shared: torch.Tensor, topk: int) -> bool:
        """all_reduce_moe can take this call: the message (shared's shape, bf16) is one the plain call would send
        through the one-kernel path too (so a rank that falls back still runs the same protocol)."""
        if not self.os_on or shared.dtype != torch.bfloat16 or shared.dim() != 2 or not shared.is_contiguous():
            return False
        T, H = shared.shape
        return T * topk <= self._os_slots and H % 8 == 0 and T * H * 2 <= self.os_max and self.should(shared) \
            and self._takes(shared)

    def all_reduce_moe(self, os_buf: torch.Tensor, w: torch.Tensor, ids: torch.Tensor, g2l: torch.Tensor, E: int,
                       shared: torch.Tensor, topk: int) -> torch.Tensor:
        """out [T, H] bf16 = all-reduce of bf16(bf16(sum_k w * os over the local picks, pick order) + shared): the
        r9k expert tier's per-token sum + the runner's shared add, formed inside the collective's first phase.
        os_buf f32 [>= T*topk, H] (the tier's per-slot outputs), w f32 [T*topk], ids int32 [T*topk] (global),
        g2l int64 [G] (the layer's global -> local map; outside [0, E) = not here), shared bf16 [T, H]."""
        T, H = shared.shape
        out = torch.empty_like(shared)
        rc = self.pair.L.r9k_ar8_os_moe(
            *self._os_args, self.rank, self._os_seq.data_ptr(), self._os_slot, os_buf.data_ptr(), w.data_ptr(),
            ids.data_ptr(), g2l.data_ptr(), g2l.numel(), E, T, topk, H, shared.data_ptr(), out.data_ptr(),
            torch.cuda.current_stream().cuda_stream, self._os_nd(T * H * 2), self._os_nt, int(T * H * 2 <= self.os_xp),
            self._os_tmo, self._os_diag)
        if rc:
            raise RuntimeError(f"r9k_ar8_os_moe failed ({rc}) T={T} H={H} topk={topk}")
        return out

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if not self._takes(x):
            return self.flat.all_reduce(x)                   # callers check should() first: never above max_kb
        flat = x.view(-1)
        if self.os_on and x.numel() * x.element_size() <= self.os_max:      # group-wide state only (_setup_oneshot)
            return self._oneshot(flat).view_as(x)
        if self.split and x.numel() * x.element_size() > self.split_min:     # group-wide state only (see above)
            return self._split(flat).view_as(x)
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
