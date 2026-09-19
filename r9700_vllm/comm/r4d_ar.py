"""TP=2 all-reduce on libr4d's one-shot P2P kernel (gfx1201), as a CudaCommunicator subclass.

libr4d (codeberg.org/StillDeadcode/libr4d, license pending) ships r4d_ar_oneshot_2rank_exact: each rank pushes its
input into the peer's fine-grained IPC scratch and reduces locally (fp32 accumulate, bit-identical to RCCL's sum).
Scratch is double-buffered by a device-resident per-block sequence counter the kernel increments, so it is
cudagraph-safe with no buffer registration (only scratch/flags are IPC-shared). davetha measured ~3.2 us p50 per
call vs RCCL's ~69 us here (157 calls per MTP-3 step). Integration pattern follows GGZ14/vllm-mxfp4's
radiance_allreduce.py (exact path only here).

Env: R9K_R4D_AR (1 = on), R9K_R4D_AR_MAX_MB (48). Needs HSA_ENABLE_IPC_MODE_LEGACY=0 on this platform.
"""
from __future__ import annotations

import importlib.util
import os

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)
_DTYPE = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}
_R4D = None


def r4d():
    """libr4d's pybind module (r4d.so next to libr9k.so, or R9K_R4D_SO)."""
    global _R4D
    if _R4D is None:
        path = os.environ.get("R9K_R4D_SO") or os.path.join(os.path.dirname(__file__), "..", "kernels", "r4d.so")
        spec = importlib.util.spec_from_file_location("r4d", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _R4D = mod
    return _R4D


class R4dAllReduce:
    def __init__(self, group, device):
        self.disabled = True
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if self.world_size != 2:
            return
        ext = r4d()
        name = ext.select("allreduce", world_size=2, exact=1, dtype="bf16")
        if name is None:
            logger.warning("r9700: libr4d has no exact 2-rank all-reduce in this build")
            return
        self._ar = getattr(ext, name)
        self.device = torch.device(f"cuda:{device}") if isinstance(device, int) else device
        self.max_bytes = (int(float(os.environ.get("R9K_R4D_AR_MAX_MB", "48")) * 2**20) // 16) * 16
        self.slot16 = self.max_bytes // 16
        self.nt, self.min_nb, self.words_per_block = 1024, 4, 1400
        maxb = int(ext.AR_MAX_BLOCKS)
        self.max_nb = min(24, maxb)
        torch.cuda.set_device(self.device)

        def alloc(nbytes):
            try:
                ptr, h = ext.ar_ipc_alloc(nbytes, True)          # fine-grained: posts straight to the fabric
                return ptr, h, True
            except Exception:
                ptr, h = ext.ar_ipc_alloc(nbytes, False)
                return ptr, h, False

        self._scratch, sh, fine = alloc(2 * self.max_bytes)
        self._flags, fh, _ = alloc(maxb * 4)
        shs, fhs = [None, None], [None, None]
        dist.all_gather_object(shs, sh, group=group)
        dist.all_gather_object(fhs, fh, group=group)
        peer = 1 - self.rank
        self._peer_scratch = ext.ar_ipc_open(shs[peer])
        self._peer_flags = ext.ar_ipc_open(fhs[peer])
        self._seq = torch.zeros(maxb, dtype=torch.int32, device=self.device)
        self.drain, self.acq = (3, 0) if fine else (3, 1)
        # Optional compressed payload for large messages (R9K_AR_QUANT=1): libr4d's ar_oneshot_2rank_wht6 rotates
        # each 64-element group by a Walsh-Hadamard transform and ships 6 bits + a bf16 scale (6.25 bits/elem),
        # ~2.6x fewer bytes over PCIe; both ranks reduce the identical packed pair, so they stay bit-identical to
        # each other (not to the exact sum). Same threshold as GGZ14's radiance default (128 KB): decode-size
        # messages stay exact. Lossy -> opt-in, gated on GSM8K.
        self._wht6 = None
        if os.environ.get("R9K_AR_QUANT", "0") == "1":
            try:
                self._qgroup = int(ext.AR_WHT6_GROUP)
                qname = ext.select("allreduce", world_size=2, exact=0, dtype="bf16", numel=self._qgroup)
                if qname is None:
                    raise RuntimeError("no lossy all-reduce kernel in this libr4d build")
                self._wht6 = getattr(ext, qname)
                self._qmin = int(float(os.environ.get("R9K_AR_QUANT_MIN_KB", "128")) * 1024)
                self._locpk = torch.empty(self.max_bytes // 2 + self.max_bytes // 32 + 4096, dtype=torch.uint8,
                                          device=self.device)
                logger.info("r9700: libr4d wht6 compressed all-reduce for messages >= %d KB (%d-bit)",
                            self._qmin >> 10, int(ext.AR_WHT6_BITS))
            except Exception as e:
                logger.warning("r9700: R9K_AR_QUANT requested but unavailable (%s); exact all-reduce only", e)
                self._wht6 = None
        self.disabled = False
        logger.info("r9700: libr4d 2-rank P2P all-reduce installed (rank %d, max %d MiB, fine-grained=%s)",
                    self.rank, self.max_bytes >> 20, fine)

    def should(self, x: torch.Tensor) -> bool:
        if self.disabled or x.dtype not in _DTYPE or not x.is_contiguous():
            return False
        n = x.numel() * x.element_size()
        return 0 < n <= self.max_bytes and n % 16 == 0

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        nbytes = x.numel() * x.element_size()
        if (self._wht6 is not None and x.dtype in (torch.bfloat16, torch.float16) and nbytes >= self._qmin
                and x.numel() % self._qgroup == 0):
            nb = max(self.min_nb, min(48, nbytes // (self.words_per_block * 16)))
            self._wht6(self._peer_scratch, self._scratch, self._peer_flags, self._flags, self._seq.data_ptr(),
                       self._locpk.data_ptr(), self.max_bytes, self.max_bytes // 2, x.data_ptr(), out.data_ptr(),
                       x.numel(), _DTYPE[x.dtype], torch.cuda.current_stream().cuda_stream, nb, 1024, self.drain,
                       self.acq)
            return out
        n16 = x.numel() * x.element_size() // 16
        nb = max(self.min_nb, min(self.max_nb, n16 // self.words_per_block))
        nb = max(1, min(nb, n16))
        self._ar(self._peer_scratch, self._scratch, self._peer_flags, self._flags, self._seq.data_ptr(),
                 self.slot16, x.data_ptr(), out.data_ptr(), x.numel(), _DTYPE[x.dtype],
                 torch.cuda.current_stream().cuda_stream, nb, self.nt, self.drain, self.acq)
        return out


class R9kCommunicator(CudaCommunicator):
    """Stock CudaCommunicator (RCCL + vLLM's paths) with the TP=2 all-reduce routed to libr4d when it fits.
    Installed by R9700Platform.get_device_communicator_cls (platform.py); no class patching."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._r9k_ar = None
        if os.environ.get("R9K_R4D_AR", "1") != "1":
            return
        try:
            from ..moe.experts import r9k_available
            if "tp" in (getattr(self, "unique_name", "") or "") and getattr(self, "world_size", 1) == 2 \
                    and r9k_available():
                ar = R4dAllReduce(self.cpu_group, self.device)
                if not ar.disabled:
                    self._r9k_ar = ar
        except Exception as e:
            logger.warning("r9700: r4d AR setup failed, staying on RCCL (%s)", e)

    def all_reduce(self, input_):
        ar = self._r9k_ar
        if ar is not None and ar.should(input_):
            return ar.all_reduce(input_)
        return super().all_reduce(input_)
