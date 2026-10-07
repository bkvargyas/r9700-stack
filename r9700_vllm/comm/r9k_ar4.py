"""Compressed, topology-aware 4-rank all-reduce for prefill-sized messages (kernels/r9k_ar4.hip).

Hierarchical 2x2 (see the kernel header and notes/ar4-plan.md): pairs share a PCIe switch, cross partners sit on
the other switch. Seven launches, four handshakes, 4- or 6-bit Walsh-Hadamard wire, outputs bit-identical on all
ranks. Selected by R9kAllReduceN for messages above its exact range (default; R9K_AR4=0 reverts), for
bf16/fp16 messages whose element count is a multiple of 4 * 64 and at most R9K_AR4_MAX_MB.

Env:
  R9K_AR4 (1)            0 = keep RCCL for large messages (read by R9kAllReduceN)
  R9K_AR4_BITS (4)       4 or 6 bits per element on the wire
  R9K_AR4_MAX_MB (64)    largest message (bf16 bytes) this path accepts; scratch is sized from it
  R9K_AR4_PAIRS          "0,1;2,3": the two same-switch pairs by TP rank (default: consecutive pairs)
  R9K_AR4_BLOCKS (128)   blocks per phase kernel (64: pushes unchanged, decode/reduce 2x slower; 256: regresses)
  R9K_AR4_SDMA (0)       1 = pack locally and move the bytes with the DMA engine (hipMemcpyAsync, 11.6 GB/s) with a
                         one-block flag handshake. Measured slower end to end (11 MB: 757 vs 691 us): pack and copy
                         no longer overlap and each flag kernel costs 20-28 us. Kept for experiments.
"""
from __future__ import annotations

import ctypes
import os

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)
_DTYPE = {torch.bfloat16: 0, torch.float16: 1}
GROUP = 64


def _lib():
    from ..kernels import moe as KM
    L = KM.lib()
    if not hasattr(L, "r9k_ar4_pack_push"):
        raise RuntimeError("libr9k.so has no r9k_ar4_* (rebuild kernels/)")
    c_long, c_int = ctypes.c_long, ctypes.c_int
    L.r9k_ar4_packed_bytes.restype = c_long
    L.r9k_ar4_packed_bytes.argtypes = [c_long, c_int]
    L.r9k_ar4_max_blocks.restype = c_int
    L.r9k_ar4_pack_push.restype = c_int
    L.r9k_ar4_pack_push.argtypes = [c_long] * 7 + [c_int, c_int, c_long]
    L.r9k_ar4_reduce_bf16.restype = c_int
    L.r9k_ar4_reduce_bf16.argtypes = [c_long] * 4 + [c_int, c_long, c_long, c_int, c_int, c_long]
    L.r9k_ar4_reduce_pack_push.restype = c_int
    L.r9k_ar4_reduce_pack_push.argtypes = [c_long] * 4 + [c_int, c_long] + [c_long] * 6 + [c_int, c_int, c_long]
    L.r9k_ar4_decode.restype = c_int
    L.r9k_ar4_decode.argtypes = [c_long] * 3 + [c_int, c_int, c_long, c_long, c_int, c_int, c_long]
    L.r9k_ar4_push2.restype = c_int
    L.r9k_ar4_push2.argtypes = [c_long] * 5 + [c_int] + [c_long] * 5 + [c_int, c_long]
    L.r9k_ar4_pack.restype = c_int
    L.r9k_ar4_pack.argtypes = [c_long] * 3 + [c_int, c_int, c_long]
    L.r9k_ar4_reduce_pack.restype = c_int
    L.r9k_ar4_reduce_pack.argtypes = [c_long] * 4 + [c_int, c_int, c_long]
    L.r9k_ar4_flag.restype = c_int
    L.r9k_ar4_flag.argtypes = [c_long] * 5
    L.r9k_ar4_dma.restype = c_int
    L.r9k_ar4_dma.argtypes = [c_long] * 4
    # IPC helpers shared with r9k_ar.hip
    L.r9k_ar_ipc_handle_size.restype = c_int
    L.r9k_ar_ipc_alloc.restype = c_int
    L.r9k_ar_ipc_alloc.argtypes = [c_long, c_int, ctypes.POINTER(c_long), ctypes.c_void_p]
    L.r9k_ar_ipc_open.restype = c_int
    L.r9k_ar_ipc_open.argtypes = [ctypes.c_void_p, ctypes.POINTER(c_long)]
    return L


def _pairs(world: int) -> list[list[int]]:
    spec = os.environ.get("R9K_AR4_PAIRS")
    if spec:
        return [[int(v) for v in p.split(",")] for p in spec.split(";")]
    return [[2 * i, 2 * i + 1] for i in range(world // 2)]


class R9kAllReduce4:
    """.disabled / .should(x) / .all_reduce(x) like the other backends. world_size must be 4."""

    # phase indices into the flag / seq arrays
    P1, P2, P3, P4 = 0, 1, 2, 3

    def __init__(self, group, device, max_mb: float | None = None):
        """max_mb: the largest message (bf16 bytes) this instance accepts (default R9K_AR4_MAX_MB); a second,
        smaller instance serves the pipelined prefill parts on their own stream (comm/pipe.py)."""
        self.disabled = True
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if self.world != 4:
            return
        L = _lib()
        self.L = L
        self.device = torch.device(f"cuda:{device}") if isinstance(device, int) else device
        torch.cuda.set_device(self.device)
        self.bits = int(os.environ.get("R9K_AR4_BITS", "4"))
        if self.bits not in (4, 6):
            raise RuntimeError(f"R9K_AR4_BITS={self.bits} unsupported (4 or 6)")
        self.max_bytes = int(float(os.environ.get("R9K_AR4_MAX_MB", "64") if max_mb is None else max_mb) * 2**20)
        self.nblocks = min(int(os.environ.get("R9K_AR4_BLOCKS", "128")), L.r9k_ar4_max_blocks())
        self.sdma = os.environ.get("R9K_AR4_SDMA", "0") == "1"
        pairs = _pairs(self.world)
        pos = {r: (pi, h) for pi, p in enumerate(pairs) for h, r in enumerate(p)}
        if len(pos) != 4:
            raise RuntimeError(f"R9K_AR4_PAIRS must cover 4 ranks: {pairs}")
        self.pair_idx, self.half = pos[self.rank]
        self.pair_peer = pairs[self.pair_idx][1 - self.half]
        self.cross_peer = pairs[1 - self.pair_idx][self.half]
        self.quarter = 2 * self.half + self.pair_idx          # final quarter this rank owns

        # scratch: receive areas for P1 (packed half), P2 (packed quarter), P3 (packed quarter), P4 (packed half),
        # each double-buffered; plus local: bf16 pair partial (half), packed own quarter, packed own half (for P4
        # we forward own packed quarter + received quarter, laid out contiguously as the pair partner's half)
        max_groups = self.max_bytes // 2 // GROUP
        gh, gq = max_groups // 2, max_groups // 4
        pb = lambda g: int(L.r9k_ar4_packed_bytes(g, self.bits))
        self.stride = {self.P1: pb(gh), self.P2: pb(gq), self.P3: pb(gq), self.P4: 2 * pb(gq)}   # P4 = two quarters
        self.off = {}
        total = 0
        for ph in (self.P1, self.P2, self.P3, self.P4):
            self.off[ph] = total
            total += 2 * self.stride[ph]
        hsz = L.r9k_ar_ipc_handle_size()

        def alloc(nbytes):
            ptr, h = ctypes.c_long(0), (ctypes.c_char * hsz)()
            rc = L.r9k_ar_ipc_alloc(nbytes, 1, ctypes.byref(ptr), ctypes.byref(h))
            return (ptr.value, bytes(h)) if rc == 0 else (None, None)

        scratch, sh = alloc(total)
        flags, fh = alloc(4 * L.r9k_ar4_max_blocks() * 4)
        oks = [None] * 4
        dist.all_gather_object(oks, scratch is not None and flags is not None, group=group)
        if not all(oks):
            logger.warning("r9700: ar4 scratch alloc failed on some rank; large messages stay on RCCL")
            return
        shs, fhs = [None] * 4, [None] * 4
        dist.all_gather_object(shs, sh, group=group)
        dist.all_gather_object(fhs, fh, group=group)
        peers = {}
        good = True
        for q in (self.pair_peer, self.cross_peer):
            ps, pf = ctypes.c_long(0), ctypes.c_long(0)
            if L.r9k_ar_ipc_open(shs[q], ctypes.byref(ps)) or L.r9k_ar_ipc_open(fhs[q], ctypes.byref(pf)):
                good = False
                break
            peers[q] = (ps.value, pf.value)
        goods = [None] * 4
        dist.all_gather_object(goods, good, group=group)
        if not all(goods):
            logger.warning("r9700: ar4 IPC open failed; large messages stay on RCCL")
            return
        self.scratch, self.flags = scratch, flags
        self.peers = peers
        self.max_blocks = L.r9k_ar4_max_blocks()
        # one sequence counter array per phase (device-resident, graph-safe)
        self.seq = torch.zeros((4, self.max_blocks), dtype=torch.int32, device=self.device)
        self.partial = torch.empty(max_groups // 2 * GROUP, dtype=torch.bfloat16, device=self.device)
        self.own_packed = torch.empty(self.stride[self.P3], dtype=torch.uint8, device=self.device)
        # DMA mode sources: packed half (P1) and packed quarter (P2)
        self.pk_half = torch.empty(self.stride[self.P1], dtype=torch.uint8, device=self.device)
        self.pk_quarter = torch.empty(self.stride[self.P2], dtype=torch.uint8, device=self.device)
        self.disabled = False
        logger.info("r9700: ar4 compressed all-reduce installed (rank %d, %d-bit, pair %d cross %d, <= %d MiB, %s)",
                    self.rank, self.bits, self.pair_peer, self.cross_peer, self.max_bytes >> 20,
                    "dma pushes" if self.sdma else "fused pushes")

    def should(self, x: torch.Tensor) -> bool:
        if self.disabled or x.dtype not in _DTYPE or not x.is_contiguous():
            return False
        n = x.numel()
        return n % (4 * GROUP) == 0 and 0 < n * 2 <= self.max_bytes

    def _recv(self, phase: int, who: int) -> int:
        """address of `who`'s receive area for `phase` (parity handled in the kernels via the stride)"""
        base = self.scratch if who == self.rank else self.peers[who][0]
        return base + self.off[phase]

    def _flags(self, phase: int, who: int) -> int:
        base = self.flags if who == self.rank else self.peers[who][1]
        return base + phase * self.max_blocks * 4

    def all_reduce(self, x: torch.Tensor, timing: list | None = None, out: torch.Tensor | None = None
                   ) -> torch.Tensor:
        """timing: if a list is given, a CUDA event is recorded after each launch and appended (eager use only;
        tests/test_ar4.py PHASES=1 prints the per-phase breakdown). out: a contiguous tensor of x's shape and dtype
        to write the result into (a row slice of a larger buffer); runs on the current stream."""
        L, st = self.L, torch.cuda.current_stream().cuda_stream
        n = x.numel()
        G = n // GROUP                 # groups; multiple of 4
        gh, gq = G // 2, G // 4
        h, q = self.half, self.quarter
        if out is None:
            out = torch.empty_like(x)
        else:
            assert out.shape == x.shape and out.dtype == x.dtype and out.is_contiguous(), (out.shape, x.shape)
        seq = self.seq
        nb = self.nblocks
        bits = self.bits
        xp = x.data_ptr()

        def chk(rc, what):
            if rc:
                raise RuntimeError(f"r9k_ar4 {what} failed ({rc})")
            if timing is not None:
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                timing.append((what, e))

        if timing is not None:
            e0 = torch.cuda.Event(enable_timing=True)
            e0.record()
            timing.append(("start", e0))
        if self.sdma:
            return self._all_reduce_dma(x, out, G, chk)

        # P1: push the partner's half (the one it owns) to the pair partner; receive ours
        other_half = 1 - h
        chk(L.r9k_ar4_pack_push(xp + other_half * gh * GROUP * 2, gh, self._recv(self.P1, self.pair_peer),
                                self.stride[self.P1], self._flags(self.P1, self.pair_peer),
                                self._flags(self.P1, self.rank), seq[self.P1].data_ptr(), bits, nb, st), "P1")
        # R1: pair partial of our half = own half + received (bf16)
        chk(L.r9k_ar4_reduce_bf16(xp + h * gh * GROUP * 2, self._recv(self.P1, self.rank), self.stride[self.P1],
                                  seq[self.P1].data_ptr(), 0, gh, self.partial.data_ptr(), bits, nb, st), "R1")
        # P2: push the cross partner's quarter of our partial (quarter index within the half = its pair index)
        other_q = 1 - self.pair_idx
        chk(L.r9k_ar4_pack_push(self.partial.data_ptr() + other_q * gq * GROUP * 2, gq,
                                self._recv(self.P2, self.cross_peer), self.stride[self.P2],
                                self._flags(self.P2, self.cross_peer), self._flags(self.P2, self.rank),
                                seq[self.P2].data_ptr(), bits, nb, st), "P2")
        # R2 + P3: final quarter (own partial quarter + received) -> packed locally and pushed to the cross partner
        chk(L.r9k_ar4_reduce_pack_push(self.partial.data_ptr() + self.pair_idx * gq * GROUP * 2,
                                       self._recv(self.P2, self.rank), self.stride[self.P2], seq[self.P2].data_ptr(), 0,
                                       gq, self.own_packed.data_ptr(), self._recv(self.P3, self.cross_peer),
                                       self.stride[self.P3], self._flags(self.P3, self.cross_peer),
                                       self._flags(self.P3, self.rank), seq[self.P3].data_ptr(), bits, nb, st), "R2P3")
        # D3: decode own quarter (from our own packed bytes) and the cross partner's quarter into out
        chk(L.r9k_ar4_decode(self.own_packed.data_ptr(), 0, seq[self.P3].data_ptr(), 0, 0, gq,
                             out.data_ptr() + q * gq * GROUP * 2, bits, nb, st), "D3 own")
        cross_q = 2 * h + (1 - self.pair_idx)
        chk(L.r9k_ar4_decode(self._recv(self.P3, self.rank), self.stride[self.P3], seq[self.P3].data_ptr(), 0, 1, gq,
                             out.data_ptr() + cross_q * gq * GROUP * 2, bits, nb, st), "D3 cross")
        # P4: forward our half to the pair partner as two quarter messages [q_lo][q_hi]: our own packed quarter and
        # the cross partner's (in our P3 receive area; parity resolved on device). One kernel, one handshake.
        pb_q = int(L.r9k_ar4_packed_bytes(gq, bits))
        chk(L.r9k_ar4_push2(self.own_packed.data_ptr(), self._recv(self.P3, self.rank), self.stride[self.P3],
                            seq[self.P3].data_ptr(), pb_q, 1 if self.pair_idx == 0 else 0,
                            self._recv(self.P4, self.pair_peer), self.stride[self.P4],
                            self._flags(self.P4, self.pair_peer), self._flags(self.P4, self.rank),
                            seq[self.P4].data_ptr(), nb, st), "P4")
        # D4: decode the partner's two quarters
        other_h = 1 - h
        for k in range(2):
            qk = 2 * other_h + k
            chk(L.r9k_ar4_decode(self._recv(self.P4, self.rank) + k * pb_q, self.stride[self.P4], seq[self.P4].data_ptr(),
                                 0, 1, gq, out.data_ptr() + qk * gq * GROUP * 2, bits, nb, st), f"D4 q{qk}")
        return out

    def _all_reduce_dma(self, x: torch.Tensor, out: torch.Tensor, G: int, chk) -> torch.Tensor:
        """Same four phases, bytes moved by the DMA engine; receive areas single-buffered (see the kernel header)."""
        L, st = self.L, torch.cuda.current_stream().cuda_stream
        gh, gq = G // 2, G // 4
        h, q, nb, bits, seq = self.half, self.quarter, self.nblocks, self.bits, self.seq
        xp, pp = x.data_ptr(), self.partial.data_ptr()
        pb_h, pb_q = int(L.r9k_ar4_packed_bytes(gh, bits)), int(L.r9k_ar4_packed_bytes(gq, bits))
        r1_peer, r1_me = self._recv(self.P1, self.pair_peer), self._recv(self.P1, self.rank)
        r2_peer, r2_me = self._recv(self.P2, self.cross_peer), self._recv(self.P2, self.rank)
        r3_peer, r3_me = self._recv(self.P3, self.cross_peer), self._recv(self.P3, self.rank)
        r4_peer, r4_me = self._recv(self.P4, self.pair_peer), self._recv(self.P4, self.rank)
        # P1: pack the partner's half locally, DMA it over, handshake
        chk(L.r9k_ar4_pack(xp + (1 - h) * gh * GROUP * 2, gh, self.pk_half.data_ptr(), bits, nb, st), "P1 pack")
        chk(L.r9k_ar4_dma(r1_peer, self.pk_half.data_ptr(), pb_h, st), "P1 dma")
        chk(L.r9k_ar4_flag(self._flags(self.P1, self.pair_peer), self._flags(self.P1, self.rank),
                           seq[self.P1].data_ptr(), r1_peer + pb_h - 16, st), "P1 flag")
        # R1: pair partial of our half (single-buffered area: stride 0 makes the parity offset vanish)
        chk(L.r9k_ar4_reduce_bf16(xp + h * gh * GROUP * 2, r1_me, 0, seq[self.P1].data_ptr(), 0, gh, pp, bits, nb, st),
            "R1")
        # P2: the cross partner's quarter of our partial
        other_q = 1 - self.pair_idx
        chk(L.r9k_ar4_pack(pp + other_q * gq * GROUP * 2, gq, self.pk_quarter.data_ptr(), bits, nb, st), "P2 pack")
        chk(L.r9k_ar4_dma(r2_peer, self.pk_quarter.data_ptr(), pb_q, st), "P2 dma")
        chk(L.r9k_ar4_flag(self._flags(self.P2, self.cross_peer), self._flags(self.P2, self.rank),
                           seq[self.P2].data_ptr(), r2_peer + pb_q - 16, st), "P2 flag")
        # R2 + P3: final quarter packed locally, DMA to the cross partner, handshake
        chk(L.r9k_ar4_reduce_pack(pp + self.pair_idx * gq * GROUP * 2, r2_me, gq, self.own_packed.data_ptr(), bits, nb,
                                  st), "R2")
        chk(L.r9k_ar4_dma(r3_peer, self.own_packed.data_ptr(), pb_q, st), "P3 dma")
        chk(L.r9k_ar4_flag(self._flags(self.P3, self.cross_peer), self._flags(self.P3, self.rank),
                           seq[self.P3].data_ptr(), r3_peer + pb_q - 16, st), "P3 flag")
        # D3: our quarter (own packed bytes) and the cross partner's
        chk(L.r9k_ar4_decode(self.own_packed.data_ptr(), 0, seq[self.P3].data_ptr(), 0, 0, gq,
                             out.data_ptr() + q * gq * GROUP * 2, bits, nb, st), "D3 own")
        cross_q = 2 * h + (1 - self.pair_idx)
        chk(L.r9k_ar4_decode(r3_me, 0, seq[self.P3].data_ptr(), 0, 0, gq, out.data_ptr() + cross_q * gq * GROUP * 2,
                             bits, nb, st), "D3 cross")
        # P4: forward our half as [q_lo][q_hi] to the pair partner: own packed quarter + the cross partner's
        mine_off, other_off = (0, pb_q) if self.pair_idx == 0 else (pb_q, 0)
        chk(L.r9k_ar4_dma(r4_peer + mine_off, self.own_packed.data_ptr(), pb_q, st), "P4 dma own")
        chk(L.r9k_ar4_dma(r4_peer + other_off, r3_me, pb_q, st), "P4 dma fwd")
        chk(L.r9k_ar4_flag(self._flags(self.P4, self.pair_peer), self._flags(self.P4, self.rank),
                           seq[self.P4].data_ptr(), r4_peer + 2 * pb_q - 16, st), "P4 flag")
        # D4: the partner's two quarters
        other_h = 1 - h
        for k in range(2):
            qk = 2 * other_h + k
            chk(L.r9k_ar4_decode(r4_me + k * pb_q, 0, seq[self.P4].data_ptr(), 0, 0, gq,
                                 out.data_ptr() + qk * gq * GROUP * 2, bits, nb, st), f"D4 q{qk}")
        return out
