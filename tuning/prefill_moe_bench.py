#!/usr/bin/env python3
"""Routed-MoE prefill timing (Flash-Next per-rank shapes: E=512, top-10, gate_up 640x2560, down 2560x320):
old kernel (MT=4, CFG_GATE_UP / CFG_DOWN) vs the prefill tile cfgs, real moe_align_block_size tables, graph-timed.
Weights are E experts (420 MB per GEMM at E=512), so every call streams from DRAM without rotation.

usage: prefill_moe_bench.py [--tokens 512,1024,2048,4096] [--E 512] [--topk 10] [--cfgs old,P11,P14,P15]
                            [--warm 10 --rounds 3]   (steady-state clocks, interleaved rounds)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import tune_dense as T
from r9700_vllm.kernels import moe as K

try:
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
except ImportError:
    moe_align_block_size = None


def align(topk_ids, blk, E):
    if moe_align_block_size is not None:
        return moe_align_block_size(topk_ids, blk, E, None)
    return tuple(v.cuda() for v in K.align_block_size_ref(topk_ids.cpu(), E, blk))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="512,1024,2048,4096")
    ap.add_argument("--E", type=int, default=512)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--cfgs", default="old,P0,P2,P3,P4,P6")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warm", type=float, default=0.0, help="seconds of load before timing (steady-state clocks)")
    ap.add_argument("--rounds", type=int, default=1, help="interleaved timing rounds per cfg (min reported)")
    a = ap.parse_args()
    E, topk = a.E, a.topk
    g = torch.Generator(device="cuda").manual_seed(0)
    N1, K1, N2, K2 = 640, 2560, 2560, 320
    p1 = torch.randint(0, 256, (E, N1, K1 // 2), dtype=torch.uint8, device="cuda", generator=g)
    s1 = torch.randint(118, 128, (E, N1, K1 // 32), dtype=torch.uint8, device="cuda", generator=g)
    p2 = torch.randint(0, 256, (E, N2, K2 // 2), dtype=torch.uint8, device="cuda", generator=g)
    s2 = torch.randint(118, 128, (E, N2, K2 // 32), dtype=torch.uint8, device="cuda", generator=g)
    W1, W2 = K.prepare_mxfp4_weights(p1, s1), K.prepare_mxfp4_weights(p2, s2)
    b1 = E * (N1 * K1 // 2 + N1 * K1 // 32)
    b2 = E * (N2 * K2 // 2 + N2 * K2 // 32)
    for M in (int(v) for v in a.tokens.split(",")):
        numel = M * topk
        # a mildly skewed router (real routing is not uniform): softmax of gaussian logits with a per-expert bias
        bias = torch.randn(E, device="cuda", generator=g) * 0.7
        scores = torch.randn(M, E, device="cuda", generator=g) + bias
        topk_w, topk_ids = torch.softmax(scores, -1).topk(topk, dim=-1)
        x = torch.randn(M, K1, device="cuda", generator=g).to(torch.bfloat16)
        xq, xs = K.quant_rows_fp8(x)
        h = torch.randn(numel, K2, device="cuda", generator=g).to(torch.bfloat16)
        hq, hs = K.quant_rows_fp8(h)
        tw = topk_w.reshape(-1).float().contiguous()
        o1 = torch.empty(numel, N1, dtype=torch.bfloat16, device="cuda")
        o2 = torch.empty(numel, N2, dtype=torch.bfloat16, device="cuda")
        flop1, flop2 = 2.0 * numel * N1 * K1, 2.0 * numel * N2 * K2
        fns = []
        for c in a.cfgs.split(","):
            if c == "old":
                MT = K.pick_mt(numel, E)
                blk = 16 * MT
                sid, eid, ntpp = align(topk_ids, blk, E)
                f1 = lambda sid=sid, eid=eid, ntpp=ntpp, MT=MT: K.moe_gemm(xq, xs, W1, o1, sid, eid, ntpp, numel, topk, None, 2, 4, 2, num_experts=E, MT=MT)
                f2 = lambda sid=sid, eid=eid, ntpp=ntpp, MT=MT: K.moe_gemm(hq, hs, W2, o2, sid, eid, ntpp, numel, 1, tw, 4, 2, 1, num_experts=E, MT=MT)
            else:
                cfg = int(c[1:])
                blk = K.prefill_block(cfg)
                sid, eid, ntpp = align(topk_ids, blk, E)
                f1 = lambda sid=sid, eid=eid, ntpp=ntpp, cfg=cfg: K.moe_gemm(xq, xs, W1, o1, sid, eid, ntpp, numel, topk, None, num_experts=E, prefill=cfg)
                f2 = lambda sid=sid, eid=eid, ntpp=ntpp, cfg=cfg: K.moe_gemm(hq, hs, W2, o2, sid, eid, ntpp, numel, 1, tw, num_experts=E, prefill=cfg)
            fns.append((c, blk, int(ntpp.item()) // blk, f1, f2))
        if a.warm > 0:
            import time
            t0 = time.time()
            while time.time() - t0 < a.warm:
                for _ in range(10):
                    fns[0][3]()
                torch.cuda.synchronize()
        best = {}
        for r in range(a.rounds):
            for c, blk, nblk, f1, f2 in fns:
                u1 = T.graph_time(f1, reps=10, iters=a.iters)
                u2 = T.graph_time(f2, reps=10, iters=a.iters)
                b = best.setdefault(c, [1e9, 1e9])
                b[0], b[1] = min(b[0], u1), min(b[1], u2)
        for c, blk, nblk, f1, f2 in fns:
            u1, u2 = best[c]
            print(f"tokens={M:5d} rows={numel:6d} {c:4s} blk={blk:3d} blocks={nblk:5d} pad={nblk * blk / numel:.2f}x | "
                  f"gate_up {u1:7.1f} us {flop1 / u1 / 1e6:5.1f} TF {b1 / u1 / 1e3:4.0f} GB/s | "
                  f"down {u2:7.1f} us {flop2 / u2 / 1e6:5.1f} TF {b2 / u2 / 1e3:4.0f} GB/s", flush=True)

if __name__ == "__main__":
    main()
