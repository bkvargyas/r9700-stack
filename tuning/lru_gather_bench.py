"""Host -> VRAM throughput of the expert-cache insert copy (r4d_lru_gather) against the number of inserts and the
launch grid (R9K_LRU_GATHER="chunks,lanes"), at Flash-Next's expert size (1.245 MiB per expert per rank).

    PYTHONPATH=/repo R9K_LIB=/repo/r9700_vllm/kernels/libr9k.so python3 /repo/tuning/lru_gather_bench.py
"""
import os
import sys
import time

os.environ.setdefault("R9K_EXPERT_CACHE_STATS", "0")

import torch

from r9700_vllm.moe import cache as C

E, S = 512, 270
N1, K1, N2, K2 = 640, 2560, 2560, 320
GRIDS = [(8, 16), (16, 16), (32, 16), (64, 16), (8, 32), (16, 32), (32, 32), (8, 64), (16, 64), (32, 64)]
NS = [1, 2, 4, 8, 13, 16, 32, 64]
REPS = int(os.environ.get("REPS", "30"))


def host(shape):
    t = C._uva_empty(shape, torch.uint8)
    t.random_(0, 256)
    return t


def main():
    g = torch.Generator().manual_seed(0)
    w13, w2 = host((E, N1 * K1 // 2)), host((E, N2 * K2 // 2))
    s13, s2 = torch.randint(0, 256, (E, N1, K1 // 32), dtype=torch.uint8), torch.randint(0, 256, (E, N2, K2 // 32), dtype=torch.uint8)
    cache = C.LayerCache(0, w13, w2, s13, s2, N1, K1, N2, K2, S)
    per = sum(cache.bytes)
    print(f"expert {per / 2**20:.3f} MiB, slots {cache.S}, insert cap {cache.max_inserts}")
    ok = True
    print(f"{'inserts':>8s} " + " ".join(f"{f'{c}x{l}':>12s}" for c, l in GRIDS) + "    (us per call / GB/s)")
    for n in NS:
        row = []
        for c, l in GRIDS:
            cache.chunks, cache.lanes = c, l
            ex = [torch.randperm(E, generator=g)[:n].to(torch.int32) for _ in range(REPS + 3)]
            sl = torch.arange(n, dtype=torch.int32)
            pairs = [torch.stack([e, sl], 1).cuda() for e in ex]
            cache.n_miss.fill_(n)
            for p in pairs[:3]:                                  # warm
                cache.miss[:n].copy_(p); cache._gather()
            torch.cuda.synchronize()
            t = 0.0
            for p in pairs[3:]:
                cache.miss[:n].copy_(p)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                cache._gather()
                torch.cuda.synchronize()
                t += time.perf_counter() - t0
            us = t / REPS * 1e6
            row.append(f"{us:6.0f}/{n * per / (us * 1e-6) / 1e9:5.2f}")
            e_last = ex[-1].long()
            ok &= torch.equal(cache.a_w13[:n].cpu(), w13[e_last.to(w13.device)].cpu())
            ok &= torch.equal(cache.a_w2[:n].cpu(), w2[e_last.to(w2.device)].cpu())
        print(f"{n:8d} " + " ".join(f"{r:>12s}" for r in row), flush=True)
    print("DATA OK" if ok else "DATA MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
