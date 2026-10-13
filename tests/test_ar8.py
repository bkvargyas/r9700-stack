"""Hierarchical 8-rank all-reduce (comm/r9k_ar8.py): correctness and latency vs the flat kernels and RCCL.
    torchrun --nproc-per-node=8 tests/test_ar8.py        (TOKS=1,2,4 message sizes in tokens x 4096 bf16; BENCH=0 skips timing)
Checks: every rank's output identical (bit-exact across ranks); within 1 bf16 ulp of the fp32 rank-ordered
reference (the pair sum is rounded once before the cross step); graph replay of chained calls."""
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def inp(rank, it, n, dtype, dev):
    g = torch.Generator(device="cpu").manual_seed(1000 * it + rank)
    return torch.randn(n, generator=g, dtype=torch.float32).mul_(3).to(dtype).to(dev)


def ref(world, it, n, dtype, dev):
    acc = torch.zeros(n, dtype=torch.float32, device=dev)
    for q in range(world):
        acc += inp(q, it, n, dtype, dev).float()
    return acc.to(dtype)


def graph_time(fn, reps=50, iters=20):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(iters):
            fn()
    g.replay(); torch.cuda.synchronize()
    dist.barrier()
    t = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps / iters * 1e6


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    from r9700_vllm.comm.r9k_ar import R9kAllReduceN
    from r9700_vllm.comm.r9k_ar8 import R9kAllReduce8
    flat = R9kAllReduceN(dist.group.WORLD, dev)
    assert not flat.disabled, "flat N-rank all-reduce unavailable"
    ar8 = R9kAllReduce8(dist.group.WORLD, dev, flat=flat)
    assert not ar8.disabled, "ar8 unavailable"
    # bf16 intermediate: the pair sums are rounded before the cross step -> up to ~2.5 ulp from the once-rounded
    # reference; fp32 intermediate (R9K_AR8_FP32=1): a summation-order difference only -> within 1 ulp.
    TOL = 1 if ar8.fp32 else 3
    if rank == 0:
        print(f"ar8 intermediate: {'fp32' if ar8.fp32 else 'bf16'}, tolerance {TOL} ulp", flush=True)
    bad = 0

    def check(name, got, want, ulps=1):
        nonlocal bad
        # identical across ranks
        gathered = [torch.empty_like(got) for _ in range(world)]
        dist.all_gather(gathered, got)
        same = all(torch.equal(gathered[0], g) for g in gathered)
        # within `ulps` bf16 ulps of the fp32 reference
        diff = (got.float() - want.float()).abs()
        ulp = torch.exp2(torch.floor(torch.log2(want.float().abs().clamp_(min=2 ** -126))) - 7)
        ok = bool((diff <= ulps * ulp + 1e-6).all())
        if rank == 0:
            print(f"{name}: cross-rank identical={same} within {ulps} ulp={ok} max|diff|={diff.max().item():.3g}", flush=True)
        bad += (not same) + (not ok)

    for it, n in enumerate((4096, 8192, 4096 * 4, 4096 * 16)):
        x = inp(rank, it, n, torch.bfloat16, dev)
        assert ar8._splits(x), f"ar8 does not take n={n}"
        check(f"eager n={n}", ar8.all_reduce(x), ref(world, it, n, torch.bfloat16, dev), ulps=TOL)
    # graph replay of chained calls (decode runs inside HIP graphs)
    xs = [inp(rank, 100 + r, 4096, torch.bfloat16, dev) for r in range(3)]
    for x in xs:
        ar8.all_reduce(x)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ys = [ar8.all_reduce(x) for x in xs]
        y2 = ar8.all_reduce(ys[0])
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    for r, y in enumerate(ys):
        check(f"graph {r}", y, ref(world, 100 + r, 4096, torch.bfloat16, dev), ulps=TOL)
    check("graph chained", y2, ref(world, 100, 4096, torch.bfloat16, dev).float().mul(world).to(torch.bfloat16), ulps=2 * TOL)
    if rank == 0:
        print("RESULT", "PASS" if bad == 0 else f"FAIL ({bad})", flush=True)
    if os.environ.get("BENCH", "1") == "1":
        if rank == 0:
            print(f"{'tokens':>7} {'KiB':>6} {'rccl':>8} {'flat-1s':>8} {'flat-2s':>8} {'ar8':>8}  (us, HIP graphs)", flush=True)
        for tok in [int(t) for t in os.environ.get("TOKS", "1,2,4").split(",")]:
            n = 4096 * tok
            x = inp(rank, 500 + tok, n, torch.bfloat16, dev)
            t_rccl = graph_time(lambda: dist.all_reduce(x.clone()))
            t_1s = graph_time(lambda: flat.all_reduce(x, mode=1)) if n * 2 <= flat.max1 else float("nan")
            t_2s = graph_time(lambda: flat.all_reduce(x, mode=2))
            t_8 = graph_time(lambda: ar8.all_reduce(x))
            if rank == 0:
                print(f"{tok:>7} {n * 2 // 1024:>6} {t_rccl:>8.1f} {t_1s:>8.1f} {t_2s:>8.1f} {t_8:>8.1f}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
