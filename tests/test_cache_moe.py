"""Expert cache (LRU + two-pass hot/cold grouped GEMM) must equal a single pass over the full host store, bit for bit.

Run in the ROCm 10 image (needs vllm for moe_align_block_size):
    PYTHONPATH=/repo R9K_LIB=/repo/r9700_vllm/kernels/libr9k.so python3 /repo/tests/test_cache_moe.py
"""
import sys

import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

from r9700_vllm.kernels import moe as K
from r9700_vllm.moe import cache as C
from test_moe_mxfp4 import make_experts


CFG_UP, CFG_DOWN = (2, 4, 2), (4, 2, 1)     # same as production (K=320 needs SK in {1,2,5,10})


def gemm_pass(xq, xs, W, out, topk_ids, E, mapping, numel, a_row_div, tw=None):
    sid, eid, ntpp = moe_align_block_size(topk_ids, K.MOE_BLOCK, E, mapping)
    cfg = CFG_UP if W.K != 320 else CFG_DOWN
    K.moe_gemm(xq, xs, W, out, sid, eid, ntpp, numel, a_row_div, tw, *cfg, num_experts=E)


def main():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    E, S, M, topk = 64, 16, 4, 6
    N1, K1, N2, K2 = 640, 2560, 2560, 320
    p1, s1, _ = make_experts(E, N1, K1, g)
    p2, s2, _ = make_experts(E, N2, K2, g)
    w1 = K.prepare_mxfp4_weights(p1.cuda(), s1.cuda())
    w2 = K.prepare_mxfp4_weights(p2.cuda(), s2.cuda())
    hw13, hw2 = C.to_host(w1.wq), C.to_host(w2.wq)
    cache = C.LayerCache(0, hw13, hw2, w1.wsr, w2.wsr, N1, K1, N2, K2, S)
    ref1 = K.Mxfp4Experts(hw13, cache.h_s13, N1, K1)
    ref2 = K.Mxfp4Experts(hw2, cache.h_s2, N2, K2)
    ok = True
    hot_pref = torch.randperm(E, generator=g)[:24]            # routing locality: mostly a 24-expert working set
    for step in range(40):
        if step == 30:
            hot_pref = torch.randperm(E, generator=g)[:24]    # working-set shift -> burst of misses
        pool = hot_pref if step % 7 else torch.arange(E)      # every 7th step: uniform (read-through-ish)
        topk_ids = torch.stack([pool[torch.randperm(pool.numel(), generator=g)[:topk]] for _ in range(M)])
        topk_ids = topk_ids.to(torch.int32).cuda()
        tw = torch.rand(M * topk, generator=g).cuda()
        x = (torch.randn(M, K1, generator=g) * 0.5).to(torch.bfloat16).cuda()
        h = (torch.randn(M * topk, K2, generator=g) * 0.5).to(torch.bfloat16).cuda()
        xq, xs = K.quant_rows_fp8(x)
        hq, hs = K.quant_rows_fp8(h)
        numel = M * topk
        # reference: one pass over the full host store
        r1 = torch.zeros(numel, N1, dtype=torch.bfloat16, device="cuda")
        r2 = torch.zeros(numel, N2, dtype=torch.bfloat16, device="cuda")
        gemm_pass(xq, xs, ref1, r1, topk_ids, E, None, numel, topk)
        gemm_pass(hq, hs, ref2, r2, topk_ids, E, None, numel, 1, tw)
        # cached: LRU update, then hot pass over slots + cold pass over host (alternate split / fused paths)
        (h1, h2), (c1, c2) = cache.hot(), cache.cold()
        o1 = torch.zeros(numel, N1, dtype=torch.bfloat16, device="cuda")
        o2 = torch.zeros(numel, N2, dtype=torch.bfloat16, device="cuda")
        if step % 2:
            (sh, eh, nh), (sc, ec, nc) = cache.update_fused(topk_ids, K.MOE_BLOCK)
            for W, o, a, d, t, cfg in ((h1, o1, (xq, xs), topk, None, CFG_UP), (h2, o2, (hq, hs), 1, tw, CFG_DOWN)):
                K.moe_gemm(a[0], a[1], W, o, sh, eh, nh, numel, d, t, *cfg, num_experts=E)
            for W, o, a, d, t, cfg in ((c1, o1, (xq, xs), topk, None, CFG_UP), (c2, o2, (hq, hs), 1, tw, CFG_DOWN)):
                K.moe_gemm(a[0], a[1], W, o, sc, ec, nc, numel, d, t, *cfg, num_experts=E)
        else:
            cache.update(topk_ids)
            gemm_pass(xq, xs, h1, o1, topk_ids, E, cache.table, numel, topk)
            gemm_pass(xq, xs, c1, o1, topk_ids, E, cache.map_cold, numel, topk)
            gemm_pass(hq, hs, h2, o2, topk_ids, E, cache.table, numel, 1, tw)
            gemm_pass(hq, hs, c2, o2, topk_ids, E, cache.map_cold, numel, 1, tw)
        torch.cuda.synchronize()
        same = torch.equal(o1, r1) and torch.equal(o2, r2)
        # invariants: table/map_cold complementary; slot_expert bijective with table
        t, mc, se = cache.table.cpu(), cache.map_cold.cpu(), cache.slot_expert.cpu()
        inv = bool(((t >= 0) ^ (mc >= 0)).all())
        for s_, e_ in enumerate(se.tolist()):
            if e_ >= 0:
                inv &= int(t[e_]) == s_
        resident_routed = int((t[topk_ids.cpu().long().flatten()] >= 0).sum())
        ok &= same and inv
        print(f"  step {step:2d}: nmiss={int(cache.n_miss.item()):2d} resident-routed {resident_routed:2d}/{numel} "
              f"bit-identical={same} invariants={inv}")
    print("ALL OK" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
