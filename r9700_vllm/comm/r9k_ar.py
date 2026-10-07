"""Our own TP=2 one-shot P2P all-reduce (kernels/r9k_ar.hip), interface-compatible with R4dAllReduce.

Independent of libr4d: the kernel and these IPC helpers are ours (see kernels/r9k_ar.hip and notes/independence.md).
Selected with R9K_AR_IMPL=r9k; R9K_AR_IMPL=r4d (default for now) keeps the libr4d backend.

Two paths, as libr4d's backend has: small messages go exact (bit-identical to RCCL); with R9K_AR_QUANT=1,
messages at or above R9K_AR_QUANT_MIN_KB are rotated and shipped at 4.25 bits/element (kernels/r9k_ar_wht.hip;
R9K_AR_QUANT_BITS=6 for the more conservative width),
which is what makes a prefill-sized all-reduce fit the PCIe 3 link. The compressed path is lossy, so both ranks
reduce the same PACKED pair and stay bit-identical to each other, and decode-size messages stay exact.
"""
from __future__ import annotations

import ctypes
import os

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)
_DTYPE = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}


def _lib():
    from ..kernels import moe as KM
    L = KM.lib()
    if not hasattr(L, "r9k_ar_oneshot_2rank"):
        raise RuntimeError("libr9k.so has no r9k_ar_oneshot_2rank (rebuild kernels/)")
    L.r9k_ar_max_blocks.restype = ctypes.c_int
    L.r9k_ar_ipc_handle_size.restype = ctypes.c_int
    L.r9k_ar_ipc_alloc.restype = ctypes.c_int
    L.r9k_ar_ipc_alloc.argtypes = [ctypes.c_long, ctypes.c_int, ctypes.POINTER(ctypes.c_long), ctypes.c_void_p]
    L.r9k_ar_ipc_open.restype = ctypes.c_int
    L.r9k_ar_ipc_open.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long)]
    L.r9k_ar_oneshot_2rank.restype = ctypes.c_int
    L.r9k_ar_oneshot_2rank.argtypes = [ctypes.c_long] * 9 + [ctypes.c_int, ctypes.c_long] + [ctypes.c_int] * 4
    if not hasattr(L, "r9k_ar_oneshot_2rank_g"):
        raise RuntimeError("libr9k.so predates the fixed-grid all-reduce (r9k_ar_oneshot_2rank_g): rebuild kernels/")
    L.r9k_ar_oneshot_2rank_g.restype = ctypes.c_int
    L.r9k_ar_oneshot_2rank_g.argtypes = [ctypes.c_long] * 9 + [ctypes.c_int, ctypes.c_long] + [ctypes.c_int] * 5
    if hasattr(L, "r9k_wht_pack"):
        for n in ("r9k_wht_group", "r9k_wht_group_bytes"):
            getattr(L, n).restype = ctypes.c_int
        L.r9k_wht_pack.restype = ctypes.c_int
        L.r9k_wht_pack.argtypes = [ctypes.c_long] * 3 + [ctypes.c_int] * 2 + [ctypes.c_long]
        L.r9k_wht_bytes_for.restype = ctypes.c_int
        L.r9k_wht_bytes_for.argtypes = [ctypes.c_int]
        L.r9k_wht_reduce_at.restype = ctypes.c_int
        L.r9k_wht_reduce_at.argtypes = [ctypes.c_long] * 6 + [ctypes.c_int] * 2 + [ctypes.c_long]
        L.r9k_ar_push_2rank.restype = ctypes.c_int
        # (peer_scratch, peer_flags, flags, seq, slot16, src, nbytes, stream) + (nb, nt, drain, acq) + slot_out
        L.r9k_ar_push_2rank.argtypes = [ctypes.c_long] * 8 + [ctypes.c_int] * 4 + [ctypes.c_long]
        L.r9k_ar_push_2rank_g.restype = ctypes.c_int
        L.r9k_ar_push_2rank_g.argtypes = [ctypes.c_long] * 8 + [ctypes.c_int] * 4 + [ctypes.c_long, ctypes.c_int]
    return L


class R9kAllReduce:
    """Same contract as R4dAllReduce: .disabled, .should(x), .all_reduce(x)."""

    def __init__(self, group, device):
        self.disabled = True
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if self.world_size != 2:
            return
        self.L = _lib()
        self.device = torch.device(f"cuda:{device}") if isinstance(device, int) else device
        torch.cuda.set_device(self.device)
        self.max_bytes = (int(float(os.environ.get("R9K_R4D_AR_MAX_MB", "48")) * 2**20) // 16) * 16
        self.slot16 = self.max_bytes // 16
        self.max_nb = min(24, self.L.r9k_ar_max_blocks())
        self.words_per_block, self.min_nb = 1400, 4
        hsz = self.L.r9k_ar_ipc_handle_size()

        def alloc(nbytes, fine):
            ptr, h = ctypes.c_long(0), (ctypes.c_char * hsz)()
            rc = self.L.r9k_ar_ipc_alloc(nbytes, 1 if fine else 0, ctypes.byref(ptr), ctypes.byref(h))
            return (ptr.value, bytes(h)) if rc == 0 else (None, None)

        self._scratch, sh = alloc(2 * self.max_bytes, True)
        fine = self._scratch is not None
        if not fine:
            self._scratch, sh = alloc(2 * self.max_bytes, False)
        if self._scratch is None:
            logger.warning("r9700: r9k all-reduce scratch alloc failed; staying on RCCL")
            return
        self._flags, fh = alloc(self.max_nb * 4, fine)
        if self._flags is None:
            logger.warning("r9700: r9k all-reduce flag alloc failed; staying on RCCL")
            return

        shs, fhs = [None, None], [None, None]
        dist.all_gather_object(shs, sh, group=group)
        dist.all_gather_object(fhs, fh, group=group)
        peer = 1 - self.rank
        ps, pf = ctypes.c_long(0), ctypes.c_long(0)
        if self.L.r9k_ar_ipc_open(shs[peer], ctypes.byref(ps)) or \
           self.L.r9k_ar_ipc_open(fhs[peer], ctypes.byref(pf)):
            logger.warning("r9700: r9k all-reduce IPC open failed; staying on RCCL")
            return
        self._peer_scratch, self._peer_flags = ps.value, pf.value
        self._histon = os.environ.get("R9K_AR_HIST") == "1"
        self._seq = torch.zeros(self.max_nb, dtype=torch.int32, device=self.device)
        # Every call launches max_nb blocks and advances all max_nb counters, whatever its size (blocks beyond the
        # data blocks do nothing else): the double buffering is only valid if all blocks agree on the buffer half.
        # With the grid following the message size, a prompt joining a batch of running decodes garbled the last
        # sequence of the batch (kernels/r9k_ar.hip, protocol note). R9K_AR_FIXED_GRID=0 restores that for bisecting.
        self._grid = self.max_nb if os.environ.get("R9K_AR_FIXED_GRID", "1") == "1" else 0
        # 4/2 = release store on the flag, acquire load on the poll: expresses exactly the ordering the
        # handshake needs at system scope, instead of draining everything with __threadfence_system(). Measured
        # ~54 us/call cheaper at 2 MB. R9K_AR_FENCE=drain,acq overrides for experiments (see tuning/ar_profile.py;
        # drain=1/2 are agent-scope and are NOT valid for a peer device even though they measure correct).
        self.drain, self.acq = 4, 2
        if os.environ.get("R9K_AR_FENCE"):
            self.drain, self.acq = (int(v) for v in os.environ["R9K_AR_FENCE"].split(",")[:2])

        # Compressed path for large (prefill) messages: rotate + 6 bits, so the wire carries 2.56x fewer bytes.
        self._wht = None
        if os.environ.get("R9K_AR_QUANT", "0") == "1" and hasattr(self.L, "r9k_wht_pack"):
            self._qgroup = self.L.r9k_wht_group()
            # 4.25 bits/elem (34 bytes per 64-element group) instead of 6.25: 1.47x fewer bytes and +5.3%
            # prefill, the only lever once the link is saturated. Default since 2026-09-22 on two independent
            # paired evals at conc=1, both null and with point estimates in OPPOSITE directions:
            #   short answers, GSM8K 1319:      94.77% -> 94.39%, 25/20 discordant, McNemar p=0.551
            #   full chain-of-thought, 800:     96.88% -> 97.00%,  4/5  discordant, McNemar p=1.000
            # The long-chain result is the load-bearing one. The worry was that a ~4x larger per-call
            # perturbation (rel 0.108 vs 0.024) would COMPOUND over a long reasoning chain; it does the
            # opposite -- only 1.1% of answers changed outcome under thinking against 3.4% on short answers,
            # because subsequent reasoning catches and corrects a perturbed intermediate step.
            # Bound honestly: these resolve ~1%, so "smaller than we can measure", not "exactly zero".
            # R9K_AR_QUANT_BITS=6 restores the conservative width. Re-evaluate ONLY at conc=1 -- at conc=8 this
            # stack is ~32% self-consistent with itself, so a conc-8 comparison measures nothing.
            self._qbits = int(os.environ.get("R9K_AR_QUANT_BITS", "4"))
            self._qbytes = self.L.r9k_wht_bytes_for(self._qbits)
            if self._qbytes <= 0:
                raise RuntimeError(f"R9K_AR_QUANT_BITS={self._qbits} unsupported (6 or 4)")
            self._qmin = int(float(os.environ.get("R9K_AR_QUANT_MIN_KB", "128")) * 1024)
            # packed bytes for the largest message we accept, rounded to the 16-byte exchange granularity
            self._qcap = ((self.max_bytes // 2 // self._qgroup) * self._qbytes + 15) // 16 * 16
            self._locpk = torch.empty(self._qcap, dtype=torch.uint8, device=self.device)
            self._slot = torch.zeros(1, dtype=torch.int32, device=self.device)
            # The compressed path needs EVERY block at the same double-buffer parity, because one slot index
            # (recorded by block 0) addresses the whole payload for the separate reduce kernel. It shares the exact
            # path's scratch, so it must share its counters too: with separate counters the two paths could pick the
            # same half for consecutive calls. Both paths advance all max_nb counters on every call (self._grid).
            # (R9K_AR_FIXED_GRID=0: the old separate counter, as before 2026-10-02.)
            self._qseq = self._seq if self._grid else torch.zeros(self.max_nb, dtype=torch.int32, device=self.device)
            self._wht = True
            logger.info("r9700: r9k compressed all-reduce for messages >= %d KB (%d-bit codes, %.2f bits/elem)",
                        self._qmin >> 10, self._qbits, self._qbytes * 8.0 / self._qgroup)
        self.disabled = False
        logger.info("r9700: r9k 2-rank P2P all-reduce installed (rank %d, max %d MiB, fine-grained=%s)",
                    self.rank, self.max_bytes >> 20, fine)

    def should(self, x: torch.Tensor) -> bool:
        if self.disabled or x.dtype not in _DTYPE or not x.is_contiguous():
            return False
        n = x.numel() * x.element_size()
        return 0 < n <= self.max_bytes and n % 16 == 0

    def _hist(self, nbytes: int, compressed: bool) -> None:
        """R9K_AR_HIST=1: histogram the message sizes this workload actually uses, so tuning targets the
        sizes that dominate rather than the size that was convenient to benchmark."""
        h = self.__dict__.setdefault("_hist_d", {})
        bucket = 1 << (nbytes.bit_length() - 1)          # power-of-two bucket
        k = (bucket, compressed)
        h[k] = h.get(k, 0) + 1
        n = self.__dict__.get("_hist_n", 0) + 1
        self.__dict__["_hist_n"] = n
        if n % int(os.environ.get("R9K_AR_HIST_EVERY", "2000")) == 0:
            rows = sorted(h.items(), key=lambda kv: -kv[1])
            tot = sum(h.values())
            logger.info("r9700 AR sizes after %d calls: %s", tot, "  ".join(
                f"{(b >> 20) or b >> 10}{'MiB' if b >> 20 else 'KiB'}{'/c' if c else '/x'}:{v}"
                f"({100.0 * v / tot:.0f}%)" for (b, c), v in rows[:8]))

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        nbytes = x.numel() * x.element_size()
        use_wht = (self._wht and x.dtype in (torch.bfloat16, torch.float16) and nbytes >= self._qmin
                   and x.numel() % self._qgroup == 0)
        if self._histon:                     # resolved once in __init__: this runs ~157x per decode step
            self._hist(nbytes, use_wht)
        if use_wht:
            return self._all_reduce_wht(x, out)
        n16 = x.numel() * x.element_size() // 16
        nb = max(self.min_nb, min(self.max_nb, n16 // self.words_per_block))
        nb = max(1, min(nb, n16))
        rc = self.L.r9k_ar_oneshot_2rank_g(
            self._peer_scratch, self._scratch, self._peer_flags, self._flags, self._seq.data_ptr(),
            self.slot16, x.data_ptr(), out.data_ptr(), x.numel(), _DTYPE[x.dtype],
            torch.cuda.current_stream().cuda_stream, nb, 0, self.drain, self.acq, self._grid)
        if rc:
            raise RuntimeError(f"r9k_ar_oneshot_2rank failed ({rc})")
        return out

    def _all_reduce_wht(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """pack locally -> exchange the packed bytes -> reduce the two packed payloads.

        Three launches instead of one; at prefill sizes that is ~6 us against the ~640 us of link time the
        smaller payload saves. Both ranks reduce the same packed pair, so their outputs agree bit for bit."""
        st = torch.cuda.current_stream().cuda_stream
        ng = x.numel() // self._qgroup
        pk_bytes = (ng * self._qbytes + 15) // 16 * 16
        rc = self.L.r9k_wht_pack(x.data_ptr(), self._locpk.data_ptr(), x.numel(), _DTYPE[x.dtype],
                                 self._qbits, st)
        if rc:
            raise RuntimeError(f"r9k_wht_pack failed ({rc})")
        # fixed block count (see _qseq above); the smallest compressed payload is >= 50 KB = 3200 16-byte words,
        # so max_nb blocks are always all non-empty and the kernel never clamps the count underneath us
        rc = self.L.r9k_ar_push_2rank_g(
            self._peer_scratch, self._peer_flags, self._flags, self._qseq.data_ptr(), self.slot16,
            self._locpk.data_ptr(), pk_bytes, st, self.max_nb, 0, self.drain, self.acq, self._slot.data_ptr(),
            self._grid)
        if rc:
            raise RuntimeError(f"r9k_ar_push_2rank failed ({rc})")
        rc = self.L.r9k_wht_reduce_at(
            self._locpk.data_ptr(), self._scratch, self._slot.data_ptr(), self.max_bytes,
            out.data_ptr(), x.numel(), _DTYPE[x.dtype], self._qbits, st)
        if rc:
            raise RuntimeError(f"r9k_wht_reduce_at failed ({rc})")
        return out


class R9kAllReduceN:
    """P2P all-reduce for TP > 2 (kernels/r9k_ar.hip), same contract as R9kAllReduce. Exact only.

    Three regimes by message size, measured 2026-09-24 on 4x R9700 (two PEX 8747 switches), bf16, HIP graphs
    (tests/test_ar_nrank.py):
      <= R9K_ARN_1S_KB  one-shot: every rank pushes its whole input to every peer, one handshake. Latency-optimal
                        (20 KB: 13.4 us vs RCCL 72.9) but moves (N-1)x the bytes, so it loses above ~190 KB.
      <= R9K_ARN_MAX_KB two-shot: reduce-scatter + all-gather in one kernel, 2(N-1)/N the bytes, two handshakes.
      larger            RCCL.
    Graph-timed, us (tokens x 2560 bf16):  rccl / one-shot / two-shot
       1 tok    5 KB    59.2 /   6.8 /   9.7
       4 tok   20 KB    72.6 /  13.4 /  11.3
      16 tok   80 KB    71.2 /  37.2 /  24.2
      64 tok  320 KB   102.0 / 130.4 /  73.2
     128 tok  640 KB   142.1 / 252.0 / 139.9
    1024 tok    5 MB   770.1 /1965.3 /1140.7   (the switch uplinks are the limit; RCCL's ring spreads it better)"""

    def __init__(self, group, device):
        self.disabled = True
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        L = _lib()
        if not hasattr(L, "r9k_ar_oneshot_nrank"):
            logger.warning("r9700: libr9k.so has no r9k_ar_oneshot_nrank; TP=%d stays on RCCL", self.world_size)
            return
        L.r9k_ar_max_ranks.restype = ctypes.c_int
        if self.world_size not in (2, 4, 8) or self.world_size > L.r9k_ar_max_ranks():
            return
        at = [ctypes.POINTER(ctypes.c_long)] * 2 + [ctypes.c_int, ctypes.c_int] + \
            [ctypes.c_long] * 5 + [ctypes.c_int, ctypes.c_long] + [ctypes.c_int] * 4
        for fn in ("r9k_ar_oneshot_nrank", "r9k_ar_twoshot_nrank"):
            if hasattr(L, fn):
                getattr(L, fn).restype = ctypes.c_int
                getattr(L, fn).argtypes = at
        for fn in ("r9k_ar_oneshot_nrank_g", "r9k_ar_twoshot_nrank_g"):
            if not hasattr(L, fn):
                logger.warning("r9700: libr9k.so predates the fixed-grid all-reduce (%s); TP=%d stays on RCCL", fn,
                               self.world_size)
                return
            getattr(L, fn).restype = ctypes.c_int
            getattr(L, fn).argtypes = at + [ctypes.c_int]
        self.L = L
        self.device = torch.device(f"cuda:{device}") if isinstance(device, int) else device
        torch.cuda.set_device(self.device)
        kb = lambda k, d: (int(float(os.environ.get(k, d)) * 1024) // 16) * 16
        self.max1 = kb("R9K_ARN_1S_KB", "16")
        self.max_bytes = max(self.max1, kb("R9K_ARN_MAX_KB", "512")) if hasattr(L, "r9k_ar_twoshot_nrank") \
            else self.max1
        self.max_nb = min(int(os.environ.get("R9K_ARN_MAX_BLOCKS", "16")), L.r9k_ar_max_blocks())
        self.words_per_block = int(os.environ.get("R9K_ARN_WPB", "512"))
        self.min_nb = int(os.environ.get("R9K_ARN_MIN_BLOCKS", "2"))
        self.max_nb2 = min(int(os.environ.get("R9K_ARN2_MAX_BLOCKS", "64")), L.r9k_ar_max_blocks())
        self.words_per_block2 = int(os.environ.get("R9K_ARN2_WPB", "1024"))
        N = self.world_size
        self.slot1 = self.max1 // 16
        self.slot2 = self.max_bytes // 16
        one = self._share(group, 2 * N * self.max1, N * L.r9k_ar_max_blocks() * 4)
        two = self._share(group, 2 * (N + 1) * self.max_bytes, N * L.r9k_ar_max_blocks() * 4) \
            if self.max_bytes > self.max1 else None
        if one is None or (self.max_bytes > self.max1 and two is None):
            logger.warning("r9700: r9k N-rank all-reduce IPC setup failed; staying on RCCL")
            return
        self._sp, self._fp = one
        self._sp2, self._fp2 = two or (None, None)
        self._seq = torch.zeros(L.r9k_ar_max_blocks(), dtype=torch.int32, device=self.device)
        self._seq2 = torch.zeros(L.r9k_ar_max_blocks(), dtype=torch.int32, device=self.device)
        # all-gather pool (kernels/r9k_ar.hip r9k_ag_oneshot_nrank): its own scratch / flags / sequence counters,
        # so the all-reduce protocol is untouched. R9K_AG_MAX_KB (128) bounds the per-rank message; larger
        # gathers (the logits at wide steps) stay on RCCL. R9K_AG=0 keeps RCCL for every all-gather.
        self._spg = None
        self.ag_max = kb("R9K_AG_MAX_KB", "128")
        if os.environ.get("R9K_AG", "1") == "1" and hasattr(L, "r9k_ag_oneshot_nrank_g"):
            L.r9k_ag_oneshot_nrank_g.restype = ctypes.c_int
            L.r9k_ag_oneshot_nrank_g.argtypes = [ctypes.POINTER(ctypes.c_long)] * 2 + [ctypes.c_int] * 2 + \
                [ctypes.c_long] * 6 + [ctypes.c_int, ctypes.c_long] + [ctypes.c_int] * 5
            ag = self._share(group, 2 * N * self.ag_max, N * L.r9k_ar_max_blocks() * 4)
            if ag is not None:
                self._spg, self._fpg = ag
                self._seqg = torch.zeros(L.r9k_ar_max_blocks(), dtype=torch.int32, device=self.device)
                self.slotg = self.ag_max // 16
            else:
                logger.warning("r9700: r9k all-gather IPC setup failed; all-gathers stay on RCCL")
        # fixed launch grids, one per (scratch, seq) pair: see R9kAllReduce._grid and kernels/r9k_ar.hip
        self._fixed = os.environ.get("R9K_AR_FIXED_GRID", "1") == "1"
        self.drain, self.acq = 4, 2
        if os.environ.get("R9K_AR_FENCE"):
            self.drain, self.acq = (int(v) for v in os.environ["R9K_AR_FENCE"].split(",")[:2])
        # Prefill-sized messages: the compressed hierarchical path (comm/r9k_ar4.py), which beats RCCL's ring 2.2x
        # at 4 bits / 1.6x at 6 bits (tests/test_ar4.py); the exact kernels cannot on this topology. Default since
        # 2026-09-25 on two null paired evals at conc=1 (300 short: 96.67 -> 96.00%, p=0.69; 800 chain-of-thought:
        # 97.62 -> 97.62%, 1/1 discordant, p=1.0, 75% of outputs token-identical). R9K_AR4=0 keeps RCCL.
        # Its constructor is collective (all ranks must take the same branch): R9K_AR4 is read on every rank.
        self.ar4 = None
        if N == 4 and os.environ.get("R9K_AR4", "1") == "1":
            from .r9k_ar4 import R9kAllReduce4
            a4 = R9kAllReduce4(group, device)
            if not a4.disabled:
                self.ar4 = a4
        self.disabled = False
        logger.info("r9700: r9k %d-rank P2P all-reduce installed (rank %d, one-shot <= %d KiB, two-shot <= %d KiB%s)",
                    N, self.rank, self.max1 >> 10, self.max_bytes >> 10,
                    f", compressed {self.ar4.bits}-bit above" if self.ar4 is not None else "")

    def _share(self, group, scratch_bytes, flag_bytes):
        """Allocate fine-grained scratch + flags, exchange IPC handles, open every peer's. Collective."""
        L, N = self.L, self.world_size
        hsz = L.r9k_ar_ipc_handle_size()

        def alloc(nbytes):
            ptr, h = ctypes.c_long(0), (ctypes.c_char * hsz)()
            rc = L.r9k_ar_ipc_alloc(nbytes, 1, ctypes.byref(ptr), ctypes.byref(h))
            return (ptr.value, bytes(h)) if rc == 0 else (None, None)

        scratch, sh = alloc(scratch_bytes)
        flags, fh = alloc(flag_bytes)
        oks = [None] * N
        dist.all_gather_object(oks, scratch is not None and flags is not None, group=group)
        if not all(oks):
            return None
        shs, fhs = [None] * N, [None] * N
        dist.all_gather_object(shs, sh, group=group)
        dist.all_gather_object(fhs, fh, group=group)
        sp, fp, good = [0] * N, [0] * N, True
        for q in range(N):
            if q == self.rank:
                sp[q], fp[q] = scratch, flags
                continue
            ps, pf = ctypes.c_long(0), ctypes.c_long(0)
            if L.r9k_ar_ipc_open(shs[q], ctypes.byref(ps)) or L.r9k_ar_ipc_open(fhs[q], ctypes.byref(pf)):
                good = False
                break
            sp[q], fp[q] = ps.value, pf.value
        goods = [None] * N
        dist.all_gather_object(goods, good, group=group)
        if not all(goods):
            return None
        return (ctypes.c_long * N)(*sp), (ctypes.c_long * N)(*fp)

    def should(self, x: torch.Tensor) -> bool:
        if self.disabled or x.dtype not in _DTYPE or not x.is_contiguous():
            return False
        n = x.numel() * x.element_size()
        if n > self.max_bytes:
            return self.ar4 is not None and self.ar4.should(x)
        return 0 < n <= self.max_bytes and n % 16 == 0

    def nblocks(self, nbytes: int, two: bool = False) -> int:
        if two:
            c16 = nbytes // 16 // self.world_size
            return max(1, min(max(self.min_nb, min(self.max_nb2, c16 // self.words_per_block2)), max(c16, 1)))
        n16 = nbytes // 16
        return max(1, min(max(self.min_nb, min(self.max_nb, n16 // self.words_per_block)), n16))

    @staticmethod
    def _ag_shape(x: torch.Tensor, dim: int) -> tuple[int, int]:
        """(rows, cols) of x seen as [prod(shape[:dim]), prod(shape[dim:])], the gather concatenating cols."""
        d = dim + x.dim() if dim < 0 else dim
        rows = 1
        for n in x.shape[:d]:
            rows *= n
        return rows, x.numel() // max(rows, 1)

    def should_ag(self, x: torch.Tensor, dim: int) -> bool:
        if self.disabled or self._spg is None or not x.is_contiguous() or x.element_size() not in (2, 4) \
                or x.numel() == 0 or not (-x.dim() <= dim < x.dim()):
            return False
        rows, cols = self._ag_shape(x, dim)
        n = x.numel() * x.element_size()
        return (cols * x.element_size()) % 16 == 0 and n <= self.ag_max

    def all_gather(self, x: torch.Tensor, dim: int = -1, nb: int | None = None, nt: int = 0) -> torch.Tensor:
        """vLLM's concat all-gather along `dim` (out.shape[dim] = world * x.shape[dim]) in one launch."""
        d = dim + x.dim() if dim < 0 else dim
        rows, cols = self._ag_shape(x, d)
        shape = list(x.shape)
        shape[d] *= self.world_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        nbytes = x.numel() * x.element_size()
        rc = self.L.r9k_ag_oneshot_nrank_g(self._spg, self._fpg, self.world_size, self.rank, self._seqg.data_ptr(),
                                           self.slotg, x.data_ptr(), out.data_ptr(), rows, cols, x.element_size(),
                                           torch.cuda.current_stream().cuda_stream, nb or self.nblocks(nbytes), nt,
                                           self.drain, self.acq, self.max_nb if self._fixed else 0)
        if rc:
            raise RuntimeError(f"r9k_ag_oneshot_nrank failed ({rc}) shape={tuple(x.shape)} dim={dim}")
        return out

    def all_reduce(self, x: torch.Tensor, nb: int | None = None, nt: int = 0, mode: int | None = None) -> torch.Tensor:
        """mode: None = by size, 1 = force one-shot, 2 = force two-shot (tests/benchmarks)."""
        nbytes = x.numel() * x.element_size()
        if mode is None and nbytes > self.max_bytes:
            return self.ar4.all_reduce(x)
        out = torch.empty_like(x)
        two = (nbytes > self.max1) if mode is None else mode == 2
        if two:
            fn, sp, fp, seq, slot = self.L.r9k_ar_twoshot_nrank_g, self._sp2, self._fp2, self._seq2, self.slot2
            grid = self.max_nb2
        else:
            fn, sp, fp, seq, slot = self.L.r9k_ar_oneshot_nrank_g, self._sp, self._fp, self._seq, self.slot1
            grid = self.max_nb
        rc = fn(sp, fp, self.world_size, self.rank, seq.data_ptr(), slot, x.data_ptr(), out.data_ptr(),
                x.numel(), _DTYPE[x.dtype], torch.cuda.current_stream().cuda_stream,
                nb or self.nblocks(nbytes, two), nt, self.drain, self.acq, grid if self._fixed else 0)
        if rc:
            raise RuntimeError(f"r9k_ar_{'two' if two else 'one'}shot_nrank failed ({rc})")
        return out
