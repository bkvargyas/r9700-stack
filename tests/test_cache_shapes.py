"""The expert cache must not keep memory per batch shape (v0.2.1 leak: one set of align buffers per (rows, block)
per layer, for good -- 28 MiB per rank at a 4,096-token chunk, OOM after minutes of mixed-length prompts).

1. Eager steps of many different sizes leave nothing behind (no persistent buffers, no growth in allocated memory).
2. A captured HIP graph still works: its buffers are kept, and replays with new routing match an eager twin.

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
    peak_keep = 0
    for M in range(1, 301):
        for mt in (1, 2, 4):
            hot, cold = cache.update_fused(route(M, g), K.MOE_BLOCK * mt)
            del hot, cold
        peak_keep = max(peak_keep, len(cache._align))
    torch.cuda.synchronize()
    grown = torch.cuda.memory_allocated() - base
    eager_ok = peak_keep == 0 and grown < (1 << 20)
    ok &= eager_ok
    print(f"  eager, 900 shapes: persistent buffer sets {peak_keep}, allocated memory grew {grown / 2**10:.0f} KiB  ok={eager_ok}")

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
    kept = len(a._align)
    b.update_fused(ids, K.MOE_BLOCK)                 # the capture pass ran the step once: keep the twin in step
    same = True
    for trial in range(12):
        ids.copy_(route(M, g))
        graph.replay()
        (rh, reh, rnh), (rc, rec, rnc) = b.update_fused(ids, K.MOE_BLOCK)
        torch.cuda.synchronize()
        same &= all(torch.equal(x, y) for x, y in ((sh, rh), (eh, reh), (nh, rnh), (sc, rc), (ec, rec), (nc, rnc)))
        same &= torch.equal(a.table, b.table) and torch.equal(a.slot_expert, b.slot_expert)
    graph_ok = kept == 1 and len(a._align) == 1 and same
    ok &= graph_ok
    print(f"  captured graph: buffer sets kept {kept}, 12 replays equal to the eager twin: {same}  ok={graph_ok}")

    print("ALL OK" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
