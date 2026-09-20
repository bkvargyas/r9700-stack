#!/usr/bin/env python3
"""DRAM-fed timing of the 4-bit GEMMs at prefill M: old (decode-shaped) kernel default vs the prefill tile cfgs.
Same make / runner / graph_time as tune_dense.py (weight copies rotated between calls).

usage: prefill_bench.py [--kind mxfp4,nvfp4] [--shapes 17408,5120;5120,8704] [--M 512,1024,2048,4096]
                        [--cfgs old,P0,P1,...] [--iters 5] [--json out]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import tune_dense as T
from r9700_vllm.kernels import moe as K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="nvfp4,mxfp4")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--M", default="512,1024,2048,4096")
    ap.add_argument("--cfgs", default="old," + ",".join(f"P{c}" for c in sorted(K.PREFILL_TILES)))
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--ncopies", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = {}
    for kind in a.kind.split(","):
        shapes = [tuple(int(v) for v in s.split(",")) for s in a.shapes.split(";") if s] or \
            {"nvfp4": [(17408, 5120), (5120, 8704)],
             "mxfp4": [(8240, 5120), (7168, 5120), (5120, 3072), (17408, 5120), (5120, 8704)]}[kind]
        for (N, Kd) in shapes:
            W, group, nbytes = T.make(kind, N, Kd)
            nc = a.ncopies or max(1, min(15, (256 << 20) // nbytes))
            Ws = [W] + [T.make(kind, N, Kd)[0] for _ in range(nc)]
            for M in (int(v) for v in a.M.split(",")):
                flop = 2.0 * M * N * Kd
                for c in a.cfgs.split(","):
                    if c == "old":
                        os.environ["R9K_TUNED"] = "/nonexistent"
                        cfg = (2, 4, 2, 4, 1) if T.ldsa_ok((2, 4, 2), 4, Kd) else (2, 4, 2, 4, 0)
                        if c == "old" and M <= 64:
                            cfg = tuple(T.default_cfg(kind, N, Kd, M))
                    elif c.startswith("P"):
                        cfg = ("P", int(c[1:]))
                    else:
                        cfg = tuple(int(v) for v in c.split("/"))
                    try:
                        us = T.graph_time(T.runner(kind, Ws, N, Kd, M, cfg), reps=2 * len(Ws), iters=a.iters)
                    except RuntimeError as ex:
                        print(f"{kind} N={N} K={Kd} M={M} {c}: FAILED {ex}")
                        continue
                    res.setdefault(kind, {}).setdefault(f"{N},{Kd}", {}).setdefault(str(M), {})[c] = us
                    print(f"{kind:6s} N={N:6d} K={Kd:5d} M={M:5d} {c:6s} {str(cfg):16s} {us:8.1f} us "
                          f"{flop / us / 1e6:6.1f} TFLOPS {nbytes / us / 1e3:5.0f} GB/s(w)", flush=True)
            del W, Ws
            torch.cuda.empty_cache()
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
