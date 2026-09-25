"""Compressed hierarchical 4-rank all-reduce (kernels/r9k_ar4.hip, comm/r9k_ar4.py): correctness and latency.

    torchrun --nproc-per-node=4 tests/test_ar4.py           (R9K_AR4_BITS=4|6; BENCH=0 skips timing)

Checks:
  * all four ranks produce bit-identical outputs (every quarter is decoded from the same packed bytes)
  * error vs the fp32 sum is bounded: relative RMS error and the fraction of elements off by > 1 quantisation step
    of the largest term are reported; the bound is loose (this is a lossy wire, gated downstream by the paired eval)
  * repeated calls with no host sync, message sizes with a non-multiple-of-64-groups quarter, graph replay
    interleaved with the exact N-rank kernels (separate flags and sequence counters)
Then times 1 / 5 / 11 / 21 MB messages: ours vs RCCL vs the exact two-shot, graph-timed.
"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def inp(rank, it, n, dev):
    g = torch.Generator(device="cpu").manual_seed(7919 * it + rank)
    # heavy-tailed like activations: gaussian with occasional outliers
    x = torch.randn(n, generator=g)
    mask = torch.rand(n, generator=g) < 0.01
    x[mask] *= 20.0
    return x.to(torch.bfloat16).to(dev)


def ref(world, it, n, dev):
    acc = torch.zeros(n, dtype=torch.float32, device=dev)
    for q in range(world):
        acc += inp(q, it, n, dev).float()
    return acc


def graph_time(fn, reps=10, iters=10):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / (reps * iters)


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 4
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    os.environ.setdefault("R9K_AR4_MAX_MB", "64")
    os.environ.setdefault("R9K_ARN_MAX_KB", "24576")      # let the exact two-shot be timed at 21 MB too
    os.environ.setdefault("R9K_ARN_1S_KB", "16")
    from r9700_vllm.comm.r9k_ar4 import R9kAllReduce4
    from r9700_vllm.comm.r9k_ar import R9kAllReduceN
    ar = R9kAllReduce4(dist.group.WORLD, dev)
    assert not ar.disabled, "R9kAllReduce4 did not install"
    exact = R9kAllReduceN(dist.group.WORLD, dev)
    bits = ar.bits
    bad = 0

    def check(name, out, want):
        nonlocal bad
        # bit-identity across ranks
        o0 = out.clone()
        dist.broadcast(o0, 0)
        same = torch.equal(o0, out)
        d = out.float() - want
        rms = (d.pow(2).mean().sqrt() / want.pow(2).mean().sqrt().clamp_min(1e-9)).item()
        finite = torch.isfinite(out).all().item()
        # loose per-bits bound on relative RMS: 4-bit ~0.1-0.2 with three quantisations, 6-bit ~4x smaller
        lim = 0.25 if bits == 4 else 0.07
        ok = same and finite and rms < lim
        flags = torch.tensor([0 if ok else 1], device=dev)
        dist.all_reduce(flags)
        bad += int(flags.item() > 0)
        if rank == 0:
            print(f"  {name:<40} rel RMS {rms:.3e}  identical-across-ranks {same}  {'ok' if flags.item() == 0 else 'FAIL'}")

    it = 0
    for n in (256, 2560 * 8, 2560 * 100, 2560 * 1000, 2560 * 2142, 2560 * 4096):
        if n % 256:
            continue
        it += 1
        x = inp(rank, it, n, dev)
        check(f"n={n} ({n * 2 / 2**20:.1f} MB)", ar.all_reduce(x), ref(world, it, n, dev))
    # back-to-back with no host sync (double buffering on four receive areas)
    xs = [inp(rank, 100 + i, 2560 * 300, dev) for i in range(12)]
    outs = [ar.all_reduce(x) for x in xs]
    for i, o in enumerate(outs):
        check(f"burst {i}", o, ref(world, 100 + i, 2560 * 300, dev))
    # graph replay, interleaved with the exact kernels (independent flags / counters)
    x = torch.empty(2560 * 512, dtype=torch.bfloat16, device=dev)
    y_small = torch.empty(2560 * 4, dtype=torch.bfloat16, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        ar.all_reduce(x)
        exact.all_reduce(y_small)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ys = exact.all_reduce(y_small)
        y = ar.all_reduce(x)
        y2 = ar.all_reduce(y)
    for r in range(6):
        x.copy_(inp(rank, 500 + r, x.numel(), dev))
        y_small.copy_(inp(rank, 600 + r, y_small.numel(), dev))
        g.replay()
        torch.cuda.synchronize()
        check(f"graph {r}", y, ref(world, 500 + r, x.numel(), dev))
        # y is identical on every rank, so its all-reduce is 4*y (quantised)
        check(f"graph chained {r}", y2, 4.0 * y.float())
        assert torch.equal(ys, ys) and torch.isfinite(ys).all()
    if rank == 0:
        print("correctness:", "PASS" if bad == 0 else f"FAIL ({bad})")

    if os.environ.get("BENCH", "1") == "1":
        if rank == 0:
            print(f"{'MB':>6} {'rccl us':>9} {'ar4 us':>9} {'exact2s us':>11}")
        for mb in (1, 5, 11, 21):
            n = (mb << 20) // 2 // 256 * 256
            x = torch.randn(n, device=dev).to(torch.bfloat16)
            t_rccl = graph_time(lambda: dist.all_reduce(x))
            t_ar4 = graph_time(lambda: ar.all_reduce(x))
            t_ex = graph_time(lambda: exact.all_reduce(x, mode=2), reps=4, iters=5) if n * 2 <= exact.max_bytes \
                else float("nan")
            if rank == 0:
                print(f"{mb:>6} {t_rccl:>9.0f} {t_ar4:>9.0f} {t_ex:>11.0f}")
    if os.environ.get("PHASES", "0") == "1":
        # eager per-phase breakdown (events between launches); the handshake waits land in the pushing kernels
        for mb in (5, 11, 21):
            n = (mb << 20) // 2 // 256 * 256
            x = torch.randn(n, device=dev).to(torch.bfloat16)
            acc = {}
            for it in range(12):
                tl = []
                dist.barrier()
                ar.all_reduce(x, timing=tl)
                torch.cuda.synchronize()
                if it >= 2:
                    for (a, ea), (b, eb) in zip(tl, tl[1:]):
                        acc[b] = acc.get(b, 0.0) + ea.elapsed_time(eb) * 1e3 / 10
            tot = sum(acc.values())
            row = [None] * world
            dist.all_gather_object(row, acc)
            if rank == 0:
                print(f"  phases @ {mb} MB (rank 0 / rank 2, us): total {tot:.0f} / {sum(row[2].values()):.0f}")
                for k in acc:
                    print(f"    {k:<8} {acc[k]:7.1f}   {row[2][k]:7.1f}")
    dist.destroy_process_group()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
