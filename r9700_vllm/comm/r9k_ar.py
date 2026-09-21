"""Our own TP=2 one-shot P2P all-reduce (kernels/r9k_ar.hip), interface-compatible with R4dAllReduce.

Independent of libr4d: the kernel and these IPC helpers are ours (see kernels/r9k_ar.hip and notes/independence.md).
Selected with R9K_AR_IMPL=r9k; R9K_AR_IMPL=r4d (default for now) keeps the libr4d backend.

Exact path only -- there is no compressed (wht6) variant here yet, so R9K_AR_QUANT has no effect on this backend
and large prefill messages ship uncompressed.
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
        self._seq = torch.zeros(self.max_nb, dtype=torch.int32, device=self.device)
        self.drain, self.acq = (3, 0) if fine else (3, 1)
        self.disabled = False
        logger.info("r9700: r9k 2-rank P2P all-reduce installed (rank %d, max %d MiB, fine-grained=%s)",
                    self.rank, self.max_bytes >> 20, fine)

    def should(self, x: torch.Tensor) -> bool:
        if self.disabled or x.dtype not in _DTYPE or not x.is_contiguous():
            return False
        n = x.numel() * x.element_size()
        return 0 < n <= self.max_bytes and n % 16 == 0

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
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
