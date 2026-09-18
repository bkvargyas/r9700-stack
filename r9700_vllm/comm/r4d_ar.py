"""TP=2 all-reduce on libr4d's one-shot P2P kernel (gfx1201), wrapped around vLLM's CudaCommunicator.

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

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)
_PATCHED = False
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
        n16 = x.numel() * x.element_size() // 16
        nb = max(self.min_nb, min(self.max_nb, n16 // self.words_per_block))
        nb = max(1, min(nb, n16))
        self._ar(self._peer_scratch, self._scratch, self._peer_flags, self._flags, self._seq.data_ptr(),
                 self.slot16, x.data_ptr(), out.data_ptr(), x.numel(), _DTYPE[x.dtype],
                 torch.cuda.current_stream().cuda_stream, nb, self.nt, self.drain, self.acq)
        return out


def patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("R9K_R4D_AR", "1") != "1":
        return False
    try:
        from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
        from ..moe.experts import r9k_available
    except Exception as e:
        logger.warning("r9700: r4d AR hook not installed (%s)", e)
        return False
    orig_init = CudaCommunicator.__init__
    orig_ar = CudaCommunicator.all_reduce

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        self._r9k_ar = None
        try:
            if "tp" in (getattr(self, "unique_name", "") or "") and getattr(self, "world_size", 1) == 2 \
                    and r9k_available():
                ar = R4dAllReduce(self.cpu_group, self.device)
                if not ar.disabled:
                    self._r9k_ar = ar
        except Exception as e:
            logger.warning("r9700: r4d AR setup failed, staying on RCCL (%s)", e)

    def all_reduce(self, input_):
        ar = getattr(self, "_r9k_ar", None)
        if ar is not None and ar.should(input_):
            return ar.all_reduce(input_)
        return orig_ar(self, input_)

    CudaCommunicator.__init__ = __init__
    CudaCommunicator.all_reduce = all_reduce
    _PATCHED = True
    return True
