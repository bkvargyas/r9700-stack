"""The expert cache must not keep memory per batch shape (v0.2.1 leak: one set of align buffers per (rows, block)
per layer, for good -- 28 MiB per rank at a 4,096-token chunk, OOM after minutes of mixed-length prompts).

1. Eager steps of 900 different sizes, smallest first (the worst order): a handful of buffer sets, under twice the
   largest, and none per shape. Largest first (serving: the profile run): no growth at all, one address throughout.
2. A captured HIP graph replays new routings equal to an eager twin, also with eager steps of another size between
   replays (they share the buffers).

Run in the ROCm 10 image:
    PYTHONPATH=/repo:/repo/tests R9K_LIB=/repo/r9700_vllm/kernels/libr9k.so python3 /repo/tests/test_cache_shapes.py
"""
import os
import sys

os.environ["R9K_LRU_THRESH"] = "0.5"

import torch

from r9700_vllm.kernels import moe as K
from r9700_vllm.moe import cache as C
from test_moe_mxfp4 import make_experts

E, S, topk = 64, 48, 6
N1, K1, N2, K2 = 640, 2560, 2560, 320


def make_cache(g):
    p1, s1, _ = make_experts(E, N1, K1, g)
    p2, s2, _ = make_experts(E, N2, K2, g)
    w1 = K.prepare_mxfp4_weights(p1.cuda(), s1.cuda())
    w2 = K.prepare_mxfp4_weights(p2.cuda(), s2.cuda())
    return C.LayerCache(0, C.to_host(w1.wq), C.to_host(w2.wq), w1.wsr, w2.wsr, N1, K1, N2, K2, S)


def route(M, g):
    return torch.stack([torch.randperm(E, generator=g)[:topk] for _ in range(M)]).to(torch.int32).cuda()


def main():
    g = torch.Generator().manual_seed(7)
    ok = True

    # 1. eager, 300 different step sizes, three block sizes
    cache = make_cache(g)
    cache.update_fused(route(4, g), K.MOE_BLOCK)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    for M in range(1, 301):
        for mt in (1, 2, 4):
            hot, cold = cache.update_fused(route(M, g), K.MOE_BLOCK * mt)
            del hot, cold
    torch.cuda.synchronize()
    grown = torch.cuda.memory_allocated() - base
    held = sum(sum(x.numel() * 4 for x in o[2:]) for o in [cache._one] + cache._one_old if o is not None)
    eager_ok = grown <= 2 * held and held < (1 << 20) and len(cache._one_old) <= 12
    ok &= eager_ok
    print(f"  eager, 900 shapes: allocated memory grew {grown / 2**10:.0f} KiB, align buffers held "
          f"{held / 2**10:.0f} KiB in {1 + len(cache._one_old)} set(s)  ok={eager_ok}")
    # the same shapes again, largest first: nothing may grow at all once the largest step has been seen
    c2 = make_cache(g)
    c2.update_fused(route(300, g), K.MOE_BLOCK * 4)
    torch.cuda.synchronize()
    base2, ptr = torch.cuda.memory_allocated(), c2._one[2].data_ptr()
    for M in range(1, 301, 7):
        for mt in (1, 2, 4):
            hot, cold = c2.update_fused(route(M, g), K.MOE_BLOCK * mt)
            stable = hot[0].data_ptr() == ptr
            ok &= stable
            del hot, cold
    torch.cuda.synchronize()
    flat = torch.cuda.memory_allocated() == base2 and not c2._one_old
    ok &= flat
    print(f"  largest step first: no growth afterwards and one address throughout: {flat}")

    # 2. captured graph: buffers kept, replays follow new routing exactly as an eager twin does
    ga, gb = torch.Generator().manual_seed(11), torch.Generator().manual_seed(11)
    a, b = make_cache(ga), make_cache(gb)
    M = 8
    ids = route(M, g)
    a.update_fused(ids, K.MOE_BLOCK)                 # eager warm-up, as vLLM does before a capture
    b.update_fused(ids, K.MOE_BLOCK)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        (sh, eh, nh), (sc, ec, nc) = a.update_fused(ids, K.MOE_BLOCK)
    b.update_fused(ids, K.MOE_BLOCK)                 # the capture pass ran the step once: keep the twin in step
    same = True
    for trial in range(12):
        ids.copy_(route(M, g))
        graph.replay()
        (rh, reh, rnh), (rc, rec, rnc) = b.update_fused(ids, K.MOE_BLOCK)
        torch.cuda.synchronize()
        same &= all(torch.equal(x, y) for x, y in ((sh, rh), (eh, reh), (nh, rnh), (sc, rc), (ec, rec), (nc, rnc)))
        same &= torch.equal(a.table, b.table) and torch.equal(a.slot_expert, b.slot_expert)
    ok &= same
    print(f"  captured graph: 12 replays equal to the eager twin: {same}")
    # an eager step of another size between replays must not disturb the graph (they share the buffers)
    mixed = True
    for trial in range(6):
        big = route(40, g)
        a.update_fused(big, K.MOE_BLOCK * 2); b.update_fused(big, K.MOE_BLOCK * 2)
        ids.copy_(route(M, g))
        graph.replay()
        (rh, reh, rnh), (rc, rec, rnc) = b.update_fused(ids, K.MOE_BLOCK)
        torch.cuda.synchronize()
        mixed &= all(torch.equal(x, y) for x, y in ((sh, rh), (eh, reh), (nh, rnh), (sc, rc), (ec, rec), (nc, rnc)))
        mixed &= torch.equal(a.table, b.table)
    ok &= mixed
    print(f"  eager steps of another size between replays: replays still equal: {mixed}")

    print("ALL OK" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
