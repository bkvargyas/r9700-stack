"""Our N-rank one-shot P2P all-gather (R9kAllReduceN.all_gather) vs RCCL: exactness and latency.

Needs every GPU of the group (NOT part of the single-GPU gate suite):
    torchrun --nproc-per-node=4 tests/test_ag_nrank.py            # BENCH=0 to skip the latency table

Checks, against vLLM's concat layout (all_gather_into_tensor + movedim + reshape), bit for bit:
  * every shape Flash-Next gathers at decode: the MTP head's gather_output ([M, 640] bf16 at dim -1) and the
    logits ([M, vocab/4] bf16 at dim -1), plus dim 0 and a 3-D dim 1 case
  * repeated calls with varying block counts (device-resident sequence counters, double buffering)
  * the same replayed from a captured HIP graph, interleaved with our all-reduce on its own pool
Then times ours vs RCCL inside HIP graphs.
"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def inp(rank, it, shape, dtype, dev):
    g = torch.Generator(device="cpu").manual_seed(7000 * it + rank)
    return torch.randn(shape, generator=g, dtype=torch.float32).to(dtype).to(dev)


def ref_gather(x, dim, world, group):
    """vLLM's GroupCoordinator.all_gather, verbatim semantics."""
    d = dim + x.dim() if dim < 0 else dim
    size = x.size()
    out = torch.empty((size[0] * world,) + size[1:], dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out, x, group=group)
    out = out.reshape((world,) + size).movedim(0, d)
    return out.reshape(size[:d] + (world * size[d],) + size[d + 1:]).contiguous()


def graph_time(fn, reps=50, iters=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000 / (reps * iters)


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    from r9700_vllm.comm.r9k_ar import R9kAllReduceN
    ar = R9kAllReduceN(dist.group.WORLD, dev)
    assert not ar.disabled and ar._spg is not None, "r9k all-gather pool not installed"
    fails = 0

    def check(name, got, want):
        nonlocal fails
        ok = got.shape == want.shape and torch.equal(got, want)
        flag = torch.tensor([0 if ok else 1], device=dev)
        dist.all_reduce(flag)
        if rank == 0:
            print(f"{'OK  ' if flag.item() == 0 else 'FAIL'} {name}")
        fails += int(flag.item() != 0)

    cases = [((4, 640), -1), ((16, 640), -1), ((1, 640), -1), ((4, 62016), -1), ((64, 640), -1),
             ((4, 640), 0), ((2, 3, 320), 1), ((4, 2560), -1)]
    it = 0
    for shape, dim in cases:
        for dtype in (torch.bfloat16, torch.float32):
            x = inp(rank, it, shape, dtype, dev)
            if not ar.should_ag(x, dim):
                if rank == 0:
                    print(f"skip {shape} dim {dim} {dtype}: outside the pool (ag_max {ar.ag_max >> 10} KiB)")
                continue
            got = ar.all_gather(x, dim)
            want = ref_gather(x, dim, world, dist.group.WORLD)
            check(f"{shape} dim {dim} {dtype}", got, want)
            it += 1
    # repeated calls, varying block counts, interleaved with the all-reduce on its own pool
    for it2 in range(12):
        x = inp(rank, 100 + it2, (4 + it2, 640), torch.bfloat16, dev)
        got = ar.all_gather(x, -1, nb=1 + it2 % 4)
        y = inp(rank, 200 + it2, (4 * 2560,), torch.bfloat16, dev)
        _ = ar.all_reduce(y)
        want = ref_gather(x, -1, world, dist.group.WORLD)
        check(f"repeat {it2} nb={1 + it2 % 4}", got, want)
    # graph replay: capture an all-gather + all-reduce pair, replay several times, compare each replay
    xs = [inp(rank, 300 + k, (4, 640), torch.bfloat16, dev) for k in range(3)]
    xin = torch.empty_like(xs[0])
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        xin.copy_(xs[0])
        _ = ar.all_gather(xin, -1)
        _ = ar.all_reduce(xin.reshape(-1))
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gout = ar.all_gather(xin, -1)
        _ = ar.all_reduce(xin.reshape(-1))
    for k in range(3):
        xin.copy_(xs[k])
        g.replay()
        torch.cuda.synchronize()
        want = ref_gather(xs[k], -1, world, dist.group.WORLD)
        check(f"graph replay {k}", gout.clone(), want)

    if os.environ.get("BENCH", "1") == "1":
        if rank == 0:
            print(f"{'shape':>14} {'bytes/rank':>10} {'r9k ag us':>10} {'rccl us':>9}")
        for shape in ((4, 640), (16, 640), (64, 640), (4, 62016)):
            x = inp(rank, 500, shape, torch.bfloat16, dev)
            if not ar.should_ag(x, -1):
                continue
            ours = graph_time(lambda: ar.all_gather(x, -1))
            rccl = graph_time(lambda: ref_gather(x, -1, world, dist.group.WORLD))
            if rank == 0:
                print(f"{str(shape):>14} {x.numel() * 2:>10} {ours:>10.1f} {rccl:>9.1f}")
    dist.barrier()
    if rank == 0:
        print("test_ag_nrank:", "FAIL" if fails else "PASS")
    dist.destroy_process_group()
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
