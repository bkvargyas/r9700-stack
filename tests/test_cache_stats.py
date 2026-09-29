"""R9K_EXPERT_CACHE_STATS counters (moe/cache.py) against a host-side recount of the same steps.

Independent identity per step: the distinct routed experts are exactly those resident BEFORE the step, plus the
inserted ones, plus the ones read through from host; a step over the insert threshold inserts nothing.

Run in the ROCm 10 image:
    PYTHONPATH=/repo R9K_LIB=/repo/r9700_vllm/kernels/libr9k.so python3 /repo/tests/test_cache_stats.py
"""
import os
import sys

os.environ["R9K_EXPERT_CACHE_STATS"] = "1"
os.environ["R9K_EXPERT_CACHE_STATS_SEC"] = "0"      # no reader thread in the test
os.environ["R9K_LRU_THRESH"] = "0.5"                # max_distinct = 24: the steps over all E must read through

import torch

from r9700_vllm.kernels import moe as K
from r9700_vllm.moe import cache as C
from test_moe_mxfp4 import make_experts


def main():
    g = torch.Generator().manual_seed(3)
    E, S, topk = 64, 48, 6                      # max_distinct = 24
    N1, K1, N2, K2 = 640, 2560, 2560, 320
    p1, s1, _ = make_experts(E, N1, K1, g)
    p2, s2, _ = make_experts(E, N2, K2, g)
    w1 = K.prepare_mxfp4_weights(p1.cuda(), s1.cuda())
    w2 = K.prepare_mxfp4_weights(p2.cuda(), s2.cuda())
    cache = C.LayerCache(0, C.to_host(w1.wq), C.to_host(w2.wq), w1.wsr, w2.wsr, N1, K1, N2, K2, S)
    assert cache.stats is not None and C.stats_snapshot().shape == (1, len(C.STAT_FIELDS))
    want = [0, 0, 0, 0, 0]
    ok = True
    work = torch.randperm(E, generator=g)[:20]
    for step in range(60):
        if step == 40:
            work = torch.randperm(E, generator=g)[:20]                       # working-set shift
        M = (1, 4, 8)[step % 3]
        pool = torch.arange(E) if step % 5 == 0 else work                     # every 5th step routes over all E
        ids = torch.stack([pool[torch.randperm(pool.numel(), generator=g)[:topk]] for _ in range(M)])
        ids = ids.to(torch.int32).cuda()
        uniq = torch.unique(ids.flatten().long())
        before = int((cache.table[uniq] >= 0).sum())
        if step % 2:
            cache.update_fused(ids, K.MOE_BLOCK)
        else:
            cache.update(ids)
        torch.cuda.synchronize()
        d = uniq.numel()
        ins = int(cache.n_miss.item())
        cold = int((cache.table[uniq] < 0).sum())
        wide = int(d > cache.max_distinct)
        ident = (d == before + ins + cold) and (ins == 0 if wide else ins <= cache.max_inserts)
        for i, v in enumerate((1, d, ins, cold, wide)):
            want[i] += v
        got = cache.stats.tolist()
        same = got == want
        ok &= ident and same
        if not (ident and same) or step % 10 == 0:
            print(f"  step {step:2d}: M={M} distinct {d:2d} = resident {before:2d} + inserted {ins:2d} + read-through "
                  f"{cold:2d}  wide={wide}  identity={ident}  counters={same}")
    snap = C.stats_snapshot()
    print("  totals:", dict(zip(C.STAT_FIELDS, snap[0].tolist())))
    print("  " + C.stats_line(snap, sum(cache.bytes) / 2**20))
    ok &= want[4] > 0 and want[2] > 0 and want[3] > 0          # the test exercised wide steps, inserts and read-through
    print("ALL OK" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
