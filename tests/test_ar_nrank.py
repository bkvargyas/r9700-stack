"""Our N-rank P2P all-reduce (one-shot + two-shot, R9kAllReduceN) vs RCCL: exactness and latency.

Needs every GPU of the TP group (NOT part of the single-GPU gate suite):
    torchrun --nproc-per-node=4 tests/test_ar_nrank.py            # BENCH=0 to skip the latency table

Checks:
  * bit-exact vs an fp32 sum in rank order, rounded once (every rank regenerates every rank's input from seeds)
  * repeated calls and a varying block count stay correct (device-resident sequence counters, double buffering)
  * the same, replayed from a captured HIP graph (how vLLM runs decode), with one-shot and two-shot calls
    interleaved (separate sequence counters and flag arrays)
Then times ours vs RCCL, both captured in HIP graphs (decode runs inside graphs, so eager timings would mislead),
at the message sizes Flash-Next produces: hidden 2560 x {4, 16, 64, 256, 1024, 4096} tokens.
"""
import os
import sys

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
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    for _ in range(3):
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
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    from r9700_vllm.comm.r9k_ar import R9kAllReduceN
    ar = R9kAllReduceN(dist.group.WORLD, dev)
    assert not ar.disabled, "R9kAllReduceN did not install"
    bad = 0

    def check(name, got, want):
        nonlocal bad
        ok = torch.equal(got, want)
        if not ok:
            bad += 1
            d = (got.float() - want.float()).abs()
            print(f"  rank {rank} FAIL {name}: {int((d > 0).sum())} differ, max {float(d.max()):.3g}")
        return ok

    it = 0
    for mode in (1, 2):
        cap = (ar.max1 if mode == 1 else ar.max_bytes)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            esz = torch.tensor([], dtype=dtype).element_size()
            for n in (8, 24, 2560 * 4 - 8, 2560 * 16, 2560 * 64, 2560 * 200 + 8, cap // esz):
                if n * esz % 16 or n * esz > cap:
                    continue
                for nb in (None, 1, 3, 16):
                    it += 1
                    check(f"mode{mode} {dtype} n={n} nb={nb}",
                          ar.all_reduce(inp(rank, it, n, dtype, dev), nb=nb, mode=mode),
                          ref(world, it, n, dtype, dev))
    # back-to-back calls with no host sync in between (the double buffer is what keeps them apart). Sizes from
    # the caps: the one-shot cap is 16 KiB by default (R9K_ARN_1S_KB) -- a fixed 2560 x 16 bf16 message was -7.
    n1 = min(2560 * 16, ar.max1 // 2 // 2 // 8 * 8)          # bf16 elements, under the one-shot slot, x8 aligned
    n2 = min(2560 * 64, ar.max_bytes // 2 // 2 // 8 * 8)
    for mode, n in ((1, n1), (2, n2)):
        xs = [inp(rank, 5000 + i, n, torch.bfloat16, dev) for i in range(40)]
        outs = [ar.all_reduce(x, mode=mode) for x in xs]
        for i, o in enumerate(outs):
            check(f"burst mode{mode} {i}", o, ref(world, 5000 + i, n, torch.bfloat16, dev))
    # graph replay: one-shot -> two-shot -> one-shot chained (a one-shot-sized message)
    x = torch.empty(n1, dtype=torch.bfloat16, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        ar.all_reduce(x, mode=1)
        ar.all_reduce(x, mode=2)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = ar.all_reduce(x, mode=1)
        y2 = ar.all_reduce(y, mode=2)
        y3 = ar.all_reduce(y2, mode=1)
    for r in range(10):
        x.copy_(inp(rank, 7000 + r, x.numel(), x.dtype, dev))
        g.replay()
        check(f"graph {r}", y, ref(world, 7000 + r, x.numel(), x.dtype, dev))
        torch.cuda.synchronize()
        # later calls: every rank holds the same input, so the sum is world * input, computed the same way
        for a, bb, nm in ((y, y2, "chained2"), (y2, y3, "chained3")):
            w = torch.zeros_like(a, dtype=torch.float32)
            for _ in range(world):
                w += a.float()
            check(f"graph {nm} {r}", bb, w.to(a.dtype))
    torch.cuda.synchronize()
    tb = torch.tensor([bad], device=dev)
    dist.all_reduce(tb)
    if rank == 0:
        print(f"correctness: {'PASS' if tb.item() == 0 else f'FAIL ({tb.item()} checks)'} ({world} ranks)")

    if os.environ.get("BENCH", "1") == "1":
        if rank == 0:
            print(f"{'tokens':>7} {'KiB':>7} {'rccl':>8} {'1-shot':>8} {'2-shot':>8}  (us, best block count; per-nb)")
        for tok in [int(t) for t in os.environ.get("TOKS", "1,4,16,32,64,128,256,1024,4096").split(",")]:
            n = 2560 * tok
            nbytes = n * 2
            x = torch.randn(n, device=dev).to(torch.bfloat16)
            t_rccl = graph_time(lambda: dist.all_reduce(x))
            res = {}
            for mode, cap in ((1, ar.max1), (2, ar.max_bytes)):
                per = {}
                if nbytes <= cap:
                    for nb in ((1, 2, 4, 8, 16) if mode == 1 else (4, 8, 16, 32, 64, 128, 256)):
                        if nb <= ar.L.r9k_ar_max_blocks():
                            per[nb] = graph_time(lambda: ar.all_reduce(x, nb=nb, mode=mode), reps=20 if tok >= 1024 else 50)
                res[mode] = per
            f = lambda p: f"{min(p.values()):7.1f}/{min(p, key=p.get):<2}" if p else f"{'-':>10}"
            if rank == 0:
                print(f"{tok:>7} {nbytes / 1024:>7.0f} {t_rccl:>8.1f} {f(res[1])} {f(res[2])}  "
                      + " | ".join(" ".join(f"{k}:{v:.0f}" for k, v in res[m].items()) for m in (1, 2)))
    dist.destroy_process_group()
    sys.exit(1 if tb.item() else 0)


if __name__ == "__main__":
    main()
