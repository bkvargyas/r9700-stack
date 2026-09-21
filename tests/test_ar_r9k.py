"""Our own 2-rank P2P all-reduce (kernels/r9k_ar.hip) vs RCCL: exactness, rank agreement, and latency.

Run with two ranks on one node:
    torchrun --nproc-per-node=2 tests/test_ar_r9k.py

Checks:
  * bit-exactness vs torch.distributed's RCCL all_reduce for fp32, and vs an fp32-accumulated reference for
    bf16/fp16 (a 2-rank sum rounded once, which is what RCCL does too)
  * both ranks produce bit-identical outputs
  * repeated calls stay correct (exercises the device-resident sequence counter and the double buffering)
  * a varying block count between calls, which is what the serving path does
"""
import ctypes
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from r9700_vllm.kernels import moe as KM  # noqa: E402

DTYPES = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}
MAX_MB = 48


def lib():
    L = KM.lib()
    L.r9k_ar_max_blocks.restype = ctypes.c_int
    L.r9k_ar_ipc_handle_size.restype = ctypes.c_int
    L.r9k_ar_ipc_alloc.restype = ctypes.c_int
    L.r9k_ar_ipc_alloc.argtypes = [ctypes.c_long, ctypes.c_int, ctypes.POINTER(ctypes.c_long), ctypes.c_void_p]
    L.r9k_ar_ipc_open.restype = ctypes.c_int
    L.r9k_ar_ipc_open.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long)]
    L.r9k_ar_oneshot_2rank.restype = ctypes.c_int
    L.r9k_ar_oneshot_2rank.argtypes = [ctypes.c_long] * 9 + [ctypes.c_int, ctypes.c_long] + [ctypes.c_int] * 4
    return L


class R9kAR:
    def __init__(self, group, device, max_bytes=MAX_MB * 2**20):
        self.L = lib()
        self.rank, self.device = dist.get_rank(group), device
        self.max_bytes = (max_bytes // 16) * 16
        self.slot16 = self.max_bytes // 16
        self.max_nb = min(24, self.L.r9k_ar_max_blocks())
        hsz = self.L.r9k_ar_ipc_handle_size()

        def alloc(nbytes, fine):
            ptr, h = ctypes.c_long(0), (ctypes.c_char * hsz)()
            rc = self.L.r9k_ar_ipc_alloc(nbytes, 1 if fine else 0, ctypes.byref(ptr), ctypes.byref(h))
            if rc:
                print(f"  r9k_ar_ipc_alloc({nbytes}, fine={fine}) -> rc {rc}")
            return (ptr.value, bytes(h)) if rc == 0 else (None, None)

        self.scratch, sh = alloc(2 * self.max_bytes, True)
        self.fine = self.scratch is not None
        if not self.fine:                                     # no fine-grained memory: plain device alloc + acquire
            self.scratch, sh = alloc(2 * self.max_bytes, False)
        assert self.scratch, "r9k_ar_ipc_alloc failed"
        self.flags, fh = alloc(self.max_nb * 4, self.fine)
        assert self.flags, "flag alloc failed"

        shs, fhs = [None, None], [None, None]
        dist.all_gather_object(shs, sh, group=group)
        dist.all_gather_object(fhs, fh, group=group)
        peer = 1 - self.rank
        self.peer_scratch, self.peer_flags = ctypes.c_long(0), ctypes.c_long(0)
        assert self.L.r9k_ar_ipc_open(shs[peer], ctypes.byref(self.peer_scratch)) == 0, "ipc_open scratch"
        assert self.L.r9k_ar_ipc_open(fhs[peer], ctypes.byref(self.peer_flags)) == 0, "ipc_open flags"
        self.seq = torch.zeros(self.max_nb, dtype=torch.int32, device=device)
        self.drain, self.acq = (3, 0) if self.fine else (3, 1)

    def all_reduce(self, x, nb=None, nt=0):
        out = torch.empty_like(x)
        n16 = x.numel() * x.element_size() // 16
        if nb is None:
            nb = max(4, min(self.max_nb, n16 // 1400))
        nb = max(1, min(nb, n16, self.max_nb))
        rc = self.L.r9k_ar_oneshot_2rank(
            self.peer_scratch.value, self.scratch, self.peer_flags.value, self.flags, self.seq.data_ptr(),
            self.slot16, x.data_ptr(), out.data_ptr(), x.numel(), DTYPES[x.dtype],
            torch.cuda.current_stream().cuda_stream, nb, nt, self.drain, self.acq)
        assert rc == 0, f"r9k_ar_oneshot_2rank failed ({rc})"
        return out


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2, "this test is the 2-rank kernel"
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    ar = R9kAR(dist.group.WORLD, dev)
    if rank == 0:
        print(f"fine-grained scratch: {ar.fine}  max_blocks {ar.max_nb}")
    bad = 0

    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for numel in (8, 1024, 4096, 40960, 1 << 20, 5_242_880):
            if numel * torch.finfo(dtype).bits // 8 % 16:
                continue
            g = torch.Generator(device=dev).manual_seed(1234 + numel)
            x = (torch.rand(numel, generator=g, device=dev, dtype=torch.float32) - 0.5).to(dtype)
            mine = ar.all_reduce(x)

            ref = x.float().clone()                            # fp32-accumulated 2-rank sum, rounded once
            dist.all_reduce(ref)
            ref = ref.to(dtype)
            rccl = x.clone()
            dist.all_reduce(rccl)

            ok_ref = torch.equal(mine, ref)
            ok_rccl = torch.equal(mine, rccl)
            # both ranks must agree bit-for-bit
            other = mine.clone()
            dist.broadcast(other, src=0)
            ok_same = torch.equal(mine, other)
            if not (ok_ref and ok_same):
                bad += 1
            if rank == 0:
                print(f"  {str(dtype):>16} numel {numel:>9}: ==fp32ref {ok_ref}  ==rccl {ok_rccl}  "
                      f"ranks agree {ok_same}" + ("" if ok_ref and ok_same else "   <-- FAIL"))

    # repeated calls with a varying block count: exercises seq[] and the double buffer
    g = torch.Generator(device=dev).manual_seed(7)
    x = (torch.rand(1 << 20, generator=g, device=dev, dtype=torch.float32) - 0.5).to(torch.bfloat16)
    ref = x.float().clone()
    dist.all_reduce(ref)
    ref = ref.to(torch.bfloat16)
    for i in range(50):
        if not torch.equal(ar.all_reduce(x, nb=4 + (i % 5) * 5), ref):
            bad += 1
            if rank == 0:
                print(f"  repeat {i}: MISMATCH   <-- FAIL")
            break
    if rank == 0 and bad == 0:
        print("  50 repeated calls with varying block counts: ok")

    # latency vs RCCL
    def timeit(fn, n=200):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(n):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1e3 / n

    def graph_time(fn, n=200):
        """Time inside a captured graph: how the serving path actually replays these calls."""
        s_ = torch.cuda.Stream()
        s_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_):
            for _ in range(5):
                fn()
        torch.cuda.current_stream().wait_stream(s_)
        torch.cuda.synchronize()
        dist.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        dist.barrier()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(n):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) * 1e3 / n

    if rank == 0:
        print("  latency (us/call):")
    for numel in (5120, 40960, 1 << 20):
        x = torch.randn(numel, device=dev, dtype=torch.bfloat16)
        t_ours = timeit(lambda: ar.all_reduce(x))
        y = torch.randn(numel, device=dev, dtype=torch.bfloat16)
        t_rccl = timeit(lambda: dist.all_reduce(y))
        try:
            gz = torch.randn(numel, device=dev, dtype=torch.bfloat16)
            t_graph = f"{graph_time(lambda: ar.all_reduce(gz)):7.1f}"
        except Exception as e:
            t_graph = f"n/a ({type(e).__name__})"
        if rank == 0:
            print(f"    {numel*2:>9} B: ours eager {t_ours:7.1f}  in-graph {t_graph}  "
                  f"rccl eager {t_rccl:7.1f}")

    dist.barrier()
    if rank == 0:
        print("ALL OK" if bad == 0 else f"FAILURES: {bad}")
    dist.destroy_process_group()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
