"""Our own TP=2 one-shot P2P all-reduce (kernels/r9k_ar.hip), interface-compatible with R4dAllReduce.

Independent of libr4d: the kernel and these IPC helpers are ours (see kernels/r9k_ar.hip and notes/independence.md).
Selected with R9K_AR_IMPL=r9k; R9K_AR_IMPL=r4d (default for now) keeps the libr4d backend.

Two paths, as libr4d's backend has: small messages go exact (bit-identical to RCCL); with R9K_AR_QUANT=1,
messages at or above R9K_AR_QUANT_MIN_KB are rotated and shipped at 6.25 bits/element (kernels/r9k_ar_wht.hip),
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
    if hasattr(L, "r9k_wht_pack"):
        for n in ("r9k_wht_group", "r9k_wht_group_bytes"):
            getattr(L, n).restype = ctypes.c_int
        L.r9k_wht_pack.restype = ctypes.c_int
        L.r9k_wht_pack.argtypes = [ctypes.c_long] * 3 + [ctypes.c_int, ctypes.c_long]
        L.r9k_wht_reduce_at.restype = ctypes.c_int
        L.r9k_wht_reduce_at.argtypes = [ctypes.c_long] * 6 + [ctypes.c_int, ctypes.c_long]
        L.r9k_ar_push_2rank.restype = ctypes.c_int
        # (peer_scratch, peer_flags, flags, seq, slot16, src, nbytes, stream) + (nb, nt, drain, acq) + slot_out
        L.r9k_ar_push_2rank.argtypes = [ctypes.c_long] * 8 + [ctypes.c_int] * 4 + [ctypes.c_long]
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
            self._qbytes = self.L.r9k_wht_group_bytes()
            self._qmin = int(float(os.environ.get("R9K_AR_QUANT_MIN_KB", "128")) * 1024)
            # packed bytes for the largest message we accept, rounded to the 16-byte exchange granularity
            self._qcap = ((self.max_bytes // 2 // self._qgroup) * self._qbytes + 15) // 16 * 16
            self._locpk = torch.empty(self._qcap, dtype=torch.uint8, device=self.device)
            self._slot = torch.zeros(1, dtype=torch.int32, device=self.device)
            # The compressed path needs EVERY block at the same double-buffer parity, because one slot index
            # (recorded by block 0) addresses the whole payload for the separate reduce kernel. seq is per block,
            # so a varying block count would desynchronise parities -- blocks that sat out a call lag behind.
            # Hence: its own counter, never shared with the exact path, and a fixed block count on every call.
            self._qseq = torch.zeros(self.max_nb, dtype=torch.int32, device=self.device)
            self._wht = True
            logger.info("r9700: r9k compressed all-reduce for messages >= %d KB (%d bits/elem)",
                        self._qmin >> 10, self._qbytes * 8 // self._qgroup)
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
        rc = self.L.r9k_ar_oneshot_2rank(
            self._peer_scratch, self._scratch, self._peer_flags, self._flags, self._seq.data_ptr(),
            self.slot16, x.data_ptr(), out.data_ptr(), x.numel(), _DTYPE[x.dtype],
            torch.cuda.current_stream().cuda_stream, nb, 0, self.drain, self.acq)
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
        rc = self.L.r9k_wht_pack(x.data_ptr(), self._locpk.data_ptr(), x.numel(), _DTYPE[x.dtype], st)
        if rc:
            raise RuntimeError(f"r9k_wht_pack failed ({rc})")
        # fixed block count (see _qseq above); the smallest compressed payload is >= 50 KB = 3200 16-byte words,
        # so max_nb blocks are always all non-empty and the kernel never clamps the count underneath us
        rc = self.L.r9k_ar_push_2rank(
            self._peer_scratch, self._peer_flags, self._flags, self._qseq.data_ptr(), self.slot16,
            self._locpk.data_ptr(), pk_bytes, st, self.max_nb, 0, self.drain, self.acq, self._slot.data_ptr())
        if rc:
            raise RuntimeError(f"r9k_ar_push_2rank failed ({rc})")
        rc = self.L.r9k_wht_reduce_at(
            self._locpk.data_ptr(), self._scratch, self._slot.data_ptr(), self.max_bytes,
            out.data_ptr(), x.numel(), _DTYPE[x.dtype], st)
        if rc:
            raise RuntimeError(f"r9k_wht_reduce_at failed ({rc})")
        return out
