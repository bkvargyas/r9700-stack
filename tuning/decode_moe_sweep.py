#!/usr/bin/env python3
"""Decode-shape sweep of the routed MXFP4 MoE kernel's (WV, SK, NPW) launch configs, graph-timed with real
moe_align_block_size tables (the same path experts.py takes at decode: MT kernel, CFG_GATE_UP / CFG_DOWN).

The fixed decode configs in moe/experts.py were tuned for Flash-Next at TP=2 (w13 640x2560, w2 2560x320). At TP=4 the
per-rank shapes are w13 320x2560 and w2 2560x160, where the tuned down config (SK=2) is illegal and legal_cfg falls
back to pick_cfg. This sweeps every legal config at decode row counts and prints the best per GEMM per step width,
next to what the served path currently uses, so the winners can go into R9K_MOE_CFG1 / R9K_MOE_CFG2 or the defaults.

usage: decode_moe_sweep.py [--tp 4] [--tokens 1,4,16,64] [--E 512] [--topk 10] [--iters 5] [--rounds 2]
       (inside the ROCm image, one GPU; ~3 min)
"""
import argparse
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune_dense as T  # noqa: E402
from r9700_vllm.kernels import moe as K  # noqa: E402
from r9700_vllm.moe.experts import CFG_DOWN, CFG_GATE_UP  # noqa: E402

try:
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
except ImportError:
    moe_align_block_size = None


def align(topk_ids, blk, E):
    if moe_align_block_size is not None:
        return moe_align_block_size(topk_ids, blk, E, None)
    return tuple(v.cuda() for v in K.align_block_size_ref(topk_ids.cpu(), E, blk))


def legal(N, Kd):
    for WV, SK, NPW in itertools.product((1, 2, 4, 8, 16), (1, 2, 4, 5, 8, 10, 16, 20), (1, 2, 4, 8)):
        if Kd % (SK * K.GROUP) == 0 and WV * SK * 32 <= 1024:
            yield WV, SK, NPW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--tokens", default="1,4,16,64")
    ap.add_argument("--E", type=int, default=512)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    E, topk = a.E, a.topk
    inter = 640 // a.tp                                  # moe_intermediate_size per rank
    N1, K1, N2, K2 = 2 * inter, 2560, 2560, inter
    g = torch.Generator(device="cuda").manual_seed(0)
    p1 = torch.randint(0, 256, (E, N1, K1 // 2), dtype=torch.uint8, device="cuda", generator=g)
    s1 = torch.randint(118, 128, (E, N1, K1 // 32), dtype=torch.uint8, device="cuda", generator=g)
    p2 = torch.randint(0, 256, (E, N2, K2 // 2), dtype=torch.uint8, device="cuda", generator=g)
    s2 = torch.randint(118, 128, (E, N2, K2 // 32), dtype=torch.uint8, device="cuda", generator=g)
    W1, W2 = K.prepare_mxfp4_weights(p1, s1), K.prepare_mxfp4_weights(p2, s2)
    cur1, cur2 = K.legal_cfg(CFG_GATE_UP, N1, K1), K.legal_cfg(CFG_DOWN, N2, K2)
    print(f"TP={a.tp}: gate_up {N1}x{K1}, down {N2}x{K2}; served configs now: gate_up {cur1} down {cur2}")
    for M in (int(v) for v in a.tokens.split(",")):
        numel = M * topk
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
        MT = K.pick_mt(numel, E)
        sid, eid, ntpp = align(topk_ids, 16 * MT, E)
        res = {1: {}, 2: {}}
        for r in range(a.rounds):
            for which, (N, Kd, fn) in {
                1: (N1, K1, lambda c: K.moe_gemm(xq, xs, W1, o1, sid, eid, ntpp, numel, topk, None, *c, num_experts=E, MT=MT)),
                2: (N2, K2, lambda c: K.moe_gemm(hq, hs, W2, o2, sid, eid, ntpp, numel, 1, tw, *c, num_experts=E, MT=MT)),
            }.items():
                for c in legal(N, Kd):
                    try:
                        us = T.graph_time(lambda c=c: fn(c), reps=10, iters=a.iters)
                    except Exception:
                        continue
                    res[which][c] = min(res[which].get(c, 1e9), us)
        for which, name, cur in ((1, "gate_up", cur1), (2, "down", cur2)):
            top = sorted(res[which].items(), key=lambda kv: kv[1])[:5]
            now = res[which].get(cur, float("nan"))
            print(f"tokens={M:3d} rows={numel:4d} MT={MT} {name:8s} served {cur} {now:7.1f} us | best "
                  + "  ".join(f"{c} {u:6.1f}" for c, u in top), flush=True)


if __name__ == "__main__":
    main()
