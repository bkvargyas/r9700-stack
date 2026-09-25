#!/usr/bin/env python3
"""Raw P2P push bandwidth between the four R9700s, per (source, destination) and per concurrency pattern.

    torchrun --nproc-per-node=4 tuning/ar_push_bench.py [MB=8]

The exact two-shot all-reduce moves 3x the bytes RCCL's ring does and every rank pushes to both cross-switch peers
at once, so at 20 MB it is slower than RCCL (4.05 vs 2.97 ms) even though a single push runs at link rate. Before
designing the compressed / topology-aware version, this measures what the fabric actually gives:
  * one rank pushing to one peer, every (src, dst) pair: same-switch (0<->1, 2<->3) vs cross-switch
  * every rank pushing to ONE peer at once, in a pairing (a permutation), e.g. the ring 0->1->3->2->0
  * every rank pushing to all three peers at once (what the two-shot does)
  * two ranks pushing across the switch in the same direction (uplink sharing)
Uses R9kAllReduceN's IPC scratch (peers' fine-grained buffers) and the two-shot push kernel's pattern through a
tiny per-lane copy launched here; times are wall-clock over graph replays, reported as GB/s per pushing rank.
"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 4
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    mb = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    os.environ["R9K_ARN_MAX_KB"] = str(mb * 1024 * 2 + 64)      # scratch big enough for the payload
    from r9700_vllm.comm.r9k_ar import R9kAllReduceN
    ar = R9kAllReduceN(dist.group.WORLD, dev)
    assert not ar.disabled
    nbytes = mb << 20
    src = torch.randn(nbytes // 2, device=dev).to(torch.bfloat16)
    # peer scratch pointers (two-shot area is the larger one): write into slot `rank` of peer q
    ptrs = [int(ar._sp2[q]) for q in range(world)]
    slot_bytes = ar.slot2 * 16

    # Destinations: slot `rank` of each peer's IPC-mapped scratch; pushed with hipMemcpyAsync D2D (peer mapping).
    import ctypes
    dsts = {q: ptrs[q] + rank * slot_bytes for q in range(world) if q != rank}
    hip = None
    for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"):
        try:
            hip = ctypes.CDLL(name)
            break
        except OSError:
            continue
    assert hip is not None, "libamdhip64 not found"
    hip.hipMemcpyAsync.restype = ctypes.c_int
    hip.hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    D2D = 3

    def copy_to(q):
        st = torch.cuda.current_stream().cuda_stream
        rc = hip.hipMemcpyAsync(dsts[q], src.data_ptr(), nbytes, D2D, st)
        assert rc == 0, rc

    def timed(fn, iters=10):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        ms = a.elapsed_time(b) / iters
        dist.barrier()
        return ms

    def report(label, ms, n_dst):
        # gather all ranks' times so rank 0 prints one line
        t = [None] * world
        dist.all_gather_object(t, ms)
        if rank == 0:
            gbs = [nbytes * n_dst / (x * 1e-3) / 1e9 if x else 0.0 for x in t]
            print(f"  {label:<44} " + "  ".join(f"r{i} {t[i]:6.2f} ms {gbs[i]:5.1f} GB/s" for i in range(world)))

    if rank == 0:
        print(f"push bandwidth, {mb} MB per destination (GPU 0,1 = switch A; 2,3 = switch B)")
    # 1. single pusher, single destination: rank s pushes to q, others idle
    for s in range(world):
        for q in range(world):
            if q == s:
                continue
            ms = timed(lambda: copy_to(q)) if rank == s else timed(lambda: None)
            report(f"only r{s} -> r{q}", ms if rank == s else 0.0, 1)
    # 2. permutations: every rank pushes to exactly one peer at once
    for name, perm in (("ring 0>1>3>2>0", [1, 3, 0, 2]), ("ring 0>2>3>1>0", [2, 0, 3, 1]),
                       ("pairs same-switch (0<>1, 2<>3)", [1, 0, 3, 2]), ("pairs cross (0<>2, 1<>3)", [2, 3, 0, 1]),
                       ("pairs cross (0<>3, 1<>2)", [3, 2, 1, 0])):
        ms = timed(lambda: copy_to(perm[rank]))
        report(name, ms, 1)
    # 3. all-to-all: every rank pushes to all three peers back to back
    def all3():
        for q in range(world):
            if q != rank:
                copy_to(q)
    report("all-to-all (3 dsts each)", timed(all3), 3)
    # 4. staged all-to-all: three rounds, each a permutation (same-switch, cross a, cross b)
    def staged():
        for perm in ([1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]):
            copy_to(perm[rank])
    report("staged 3 rounds (1 dst per round)", timed(staged), 3)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
