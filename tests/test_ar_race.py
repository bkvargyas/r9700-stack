"""Back-to-back all-reduces of DIFFERENT sizes must stay exact when the ranks are out of step.

The bug this guards (found 2026-10-02): the kernels chose the half of the double buffer from a per-block counter,
and launched only as many blocks as the message had data for. A block that sat out a small message kept its old
parity; on the next, larger message it wrote into the same half as the message before, over that message's tail,
and a rank that was one call ahead overwrote rows its peer had not summed yet. Serving: a prompt joining a batch
of running decodes garbled the last sequence of the batch (5-8% of answers once requests exceeded max_num_seqs).

Each rank captures ONE graph holding pairs of all-reduces (a small message, then a larger one whose block slices
are shorter) and replays it many times; rank 1 burns GPU time before every replay so rank 0 runs ahead. Every
output of every replay must equal the fp32-accumulated sum of the two ranks' inputs, which each rank can compute
alone (inputs are seeded by rank).

    torchrun --nproc-per-node=2 tests/test_ar_race.py            # R9K_AR_FIXED_GRID=0 shows the old behaviour
"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from r9700_vllm.comm.r9k_ar import R9kAllReduce  # noqa: E402

HIDDEN = 2560
# (rows of the first message, rows of the second): the second has more blocks and shorter slices than the first
PAIRS = [(28, 58), (28, 64), (32, 58), (8, 28), (24, 100), (4, 32)]
REPLAYS = int(os.environ.get("REPLAYS", "1500"))


def make(rows, salt, rank):
    g = torch.Generator().manual_seed(100003 * salt + 7 * rows + rank)
    return (torch.rand(rows * HIDDEN, generator=g) - 0.5).to(torch.bfloat16)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    warm = torch.ones(16, device=dev)
    dist.all_reduce(warm)
    ar = R9kAllReduce(dist.group.WORLD, dev)
    assert not ar.disabled, "r9k all-reduce did not come up"
    fixed = bool(ar._grid)
    if rank == 0:
        print(f"2-rank all-reduce, fixed grid: {fixed} (grid {ar._grid}), compressed path: {bool(ar._wht)}")

    xs, refs = [], []
    for k, (a, b) in enumerate(PAIRS):
        for rows in (a, b):
            mine = make(rows, k, rank).to(dev)
            other = make(rows, k, 1 - rank).to(dev)
            xs.append(mine)
            refs.append((mine.float() + other.float()).to(torch.bfloat16))

    # eager first (also the warm-up a capture needs)
    bad_eager = 0
    for _ in range(20):
        outs = [ar.all_reduce(x) for x in xs]
        bad_eager += sum(int(not torch.equal(o, r)) for o, r in zip(outs, refs))
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [ar.all_reduce(x) for x in xs]
    burn = torch.randn(1536, 1536, device=dev)
    bad, first = 0, None
    per_call = [0] * len(xs)
    for it in range(REPLAYS):
        if rank == 1 and it % 3:                 # the lagging rank: rank 0 reaches the next all-reduce first
            burn @ burn
        graph.replay()
        torch.cuda.synchronize()
        for i, (o, r) in enumerate(zip(outs, refs)):
            if not torch.equal(o, r):
                bad += 1
                per_call[i] += 1
                if first is None:
                    d = (o != r).nonzero().flatten()
                    first = (it, i, int(d[0]) // HIDDEN, int(d[-1]) // HIDDEN, o.numel() // HIDDEN)
    tot = torch.tensor([bad, bad_eager], dtype=torch.int64, device=dev)
    dist.all_reduce(tot)
    dist.barrier()
    print(f"  rank {rank}: {bad} wrong outputs of {REPLAYS * len(xs)} in graph replay, {bad_eager} eager; per call "
          f"{per_call}" + ("" if first is None else f"; first: replay {first[0]}, call {first[1]}, rows "
                           f"{first[2]}..{first[3]} of {first[4]}"), flush=True)
    ok = int(tot[0]) == 0 and int(tot[1]) == 0
    if rank == 0:
        print("ALL OK" if ok else f"FAILURES: {int(tot[0])} graph, {int(tot[1])} eager (both ranks)")
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
