"""Back-to-back all-reduces of DIFFERENT sizes must stay correct when the ranks are out of step.

The bug this guards (found 2026-10-02): the kernels chose the half of the double buffer from a per-block counter,
and launched only as many blocks as the message had data for. A block that sat out a small message kept its old
parity; on the next, larger message it wrote into the same half as the message before, over that message's tail,
and a rank that was one call ahead overwrote rows its peer had not summed yet. The exact and the compressed path
also shared one scratch with separate counters. Serving: a prompt joining a batch of running decodes garbled the
last sequence of the batch (5-8% of answers once requests exceeded max_num_seqs).

Each rank captures ONE graph holding pairs of all-reduces (a small message, then a larger one whose block slices
are shorter) and replays it many times; rank 1 burns GPU time before most replays so rank 0 runs ahead. Twice:
all messages exact (R9K_AR_QUANT=0), then with the compressed path on for messages >= 128 KB.
  * an exact message must equal the fp32-accumulated sum of the two inputs (each rank computes it alone: the
    inputs are seeded by rank);
  * a compressed message must be the same bits on both ranks and within the codec's error of that sum.

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
PAIRS = [(28, 58), (28, 64), (32, 58), (8, 28), (24, 100), (4, 32), (8, 32)]
REPLAYS = int(os.environ.get("REPLAYS", "1500"))
REL_MAX = 0.2                     # 4-bit codec: ~0.11 rel RMS on this input


def make(rows, salt, rank):
    g = torch.Generator().manual_seed(100003 * salt + 7 * rows + rank)
    return (torch.rand(rows * HIDDEN, generator=g) - 0.5).to(torch.bfloat16)


def run(label, quant, rank, dev):
    os.environ["R9K_AR_QUANT"] = quant
    ar = R9kAllReduce(dist.group.WORLD, dev)
    assert not ar.disabled, "r9k all-reduce did not come up"
    xs, refs, comp = [], [], []
    for k, (a, b) in enumerate(PAIRS):
        for rows in (a, b):
            mine, other = make(rows, k, rank).to(dev), make(rows, k, 1 - rank).to(dev)
            xs.append(mine)
            refs.append((mine.float() + other.float()).to(torch.bfloat16))
            nbytes = mine.numel() * 2
            comp.append(bool(ar._wht) and nbytes >= ar._qmin and mine.numel() % ar._qgroup == 0)
    for _ in range(10):                                   # eager warm-up, as before any capture
        [ar.all_reduce(x) for x in xs]
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [ar.all_reduce(x) for x in xs]
    burn = torch.randn(1536, 1536, device=dev)
    sums = torch.zeros((REPLAYS, len(xs)), dtype=torch.int64, device=dev)
    wrong = [0] * len(xs)
    far = [0] * len(xs)
    for it in range(REPLAYS):
        if rank == 1 and it % 3:                          # the lagging rank: rank 0 reaches the next call first
            burn @ burn
        graph.replay()
        torch.cuda.synchronize()
        for i, (o, r) in enumerate(zip(outs, refs)):
            sums[it, i] = o.view(torch.int16).to(torch.int64).sum()
            if comp[i]:
                rel = float((o.float() - r.float()).norm() / r.float().norm())
                far[i] += int(not rel < REL_MAX)
            else:
                wrong[i] += int(not torch.equal(o, r))
    d = sums if rank == 0 else -sums                      # zero after the reduce iff both ranks hold the same bits
    dist.all_reduce(d)
    differ = (d != 0).sum(0).tolist()
    n_wrong, n_far, n_diff = sum(wrong), sum(far), sum(differ)
    tot = torch.tensor([n_wrong, n_far], dtype=torch.int64, device=dev)
    dist.all_reduce(tot)
    dist.barrier()
    if rank == 0:
        kinds = "".join("c" if c else "x" for c in comp)
        print(f"  {label}: grid {ar._grid or 'legacy'}; calls {kinds} (x exact, c compressed); {REPLAYS} replays: "
              f"exact outputs wrong {int(tot[0])}, compressed outputs off {int(tot[1])}, outputs differing between "
              f"the ranks {n_diff}" + ("" if not (int(tot[0]) or int(tot[1]) or n_diff) else
                                      f"   <-- FAIL  per call: wrong {wrong} differ {differ}"), flush=True)
    return int(tot[0]) + int(tot[1]) + n_diff


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    warm = torch.ones(16, device=dev)
    dist.all_reduce(warm)
    bad = run("all exact      ", "0", rank, dev)
    bad += run("exact + 4-bit  ", "1", rank, dev)
    if rank == 0:
        print("ALL OK" if bad == 0 else f"FAILURES: {bad}")
    dist.destroy_process_group()
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
