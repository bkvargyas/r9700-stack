#!/usr/bin/env python3
"""Interleaved A/B timing of the 4-bit prefill GEMM between two libr9k.so builds and several tile cfgs.

The card's clocks drift with load and temperature (the first measurement of a fresh process on a cold GPU ran
2300 us where the steady state is 2640), so single-shot numbers from separate runs cannot be compared. This
harness loads the libraries side by side (ctypes, same ABI), warms the GPU to a steady state, then times every
(lib, cfg) case round-robin for several rounds and reports the per-case min and median over the rounds.

usage: prefill_ab.py --libs head=/opt/r9700/kernels_head/libr9k.so,new=/opt/r9700/kernels/libr9k.so
                     --cases "nvfp4:17408,5120:2048:head/P8,new/P8,new/P12" [--rounds 6] [--warm 10]
cfg names: P<n> prefill tile n, F<n> the same tile with folded exponents (MXFP4), A<n> the A-tiled kernel (folded,
fragment-tiled activation), old the MT kernel, foldold folded MT. kind "quant" times the activation quantizer alone
(cfgs row / tiled; the N of the shape is ignored, K is the row length).
"""
import argparse
import ctypes
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import tune_dense as T
from r9700_vllm.kernels import moe as K


def load_lib(path):
    os.environ["R9K_LIB"] = path
    K._LIB = None
    return K.lib()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libs", required=True)
    ap.add_argument("--cases", required=True, help="; separated: kind:N,K:M:lib/cfg,lib/cfg,...")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--warm", type=float, default=10.0)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--json", default="", help="write {kind: {\"N,K\": {M: {cfg: min_us}}}}")
    a = ap.parse_args()
    libs = {}
    for spec in a.libs.split(","):
        name, path = spec.split("=")
        libs[name] = load_lib(path)
    out = {}
    warmed = False
    for spec in a.cases.split(";"):                    # one shape at a time (weights + outputs are freed between)
        spec = spec.strip()
        if not spec:
            continue
        kind, shape, M, cfgs = spec.split(":")
        N, Kd = (int(v) for v in shape.split(","))
        M = int(M)
        W, group, nbytes = T.make(kind, N, Kd)
        nc = max(1, min(15, (256 << 20) // nbytes))
        Ws = [W] + [T.make(kind, N, Kd)[0] for _ in range(nc)]
        runners = []
        for c in cfgs.split(","):
            lname, cfg = c.split("/")
            fold = cfg.startswith("F") or cfg.startswith("A") or cfg == "foldold"
            if kind == "quant":
                cc = cfg
            elif cfg in ("old", "foldold"):
                mt = T.mt_for(M)
                cc = (2, 4, 2, mt, 1) if mt > 1 and T.ldsa_ok((2, 4, 2), mt, Kd) else (2, 4, 2, mt, 0)
            elif cfg.startswith("A"):
                cc = ("A", int(cfg[1:]))
            else:
                cc = ("P", int(cfg[1:]))
            runners.append((f"{kind} {N}x{Kd} M={M} {lname}/{cfg}", lname, T.runner(kind, Ws, N, Kd, M, cc, fold),
                            2.0 * M * N * Kd if kind != "quant" else 2.0 * M * Kd, len(Ws),
                            (kind, f"{N},{Kd}", str(M), cfg)))
        # steady state: hammer the GPU for --warm seconds before the first shape, a few seconds before the others
        K._LIB = libs[runners[0][1]]
        t0 = time.time()
        while time.time() - t0 < (a.warm if not warmed else min(a.warm, 3.0)):
            for _ in range(20):
                runners[0][2]()
            torch.cuda.synchronize()
        warmed = True
        res = {}
        for r in range(a.rounds):
            for name, lname, fn, flop, nw, _ in runners:
                K._LIB = libs[lname]
                us = T.graph_time(fn, reps=2 * nw, iters=a.iters)
                res.setdefault(name, []).append(us)
                print(f"  r{r} {name:48s} {us:8.1f} us {flop / us / 1e6:6.1f} TF", flush=True)
        for name, lname, fn, flop, nw, key in runners:
            v = res[name]
            print(f"{name:48s} min {min(v):8.1f} us {flop / min(v) / 1e6:6.1f} TF | "
                  f"med {statistics.median(v):8.1f} us {flop / statistics.median(v) / 1e6:6.1f} TF", flush=True)
            out.setdefault(key[0], {}).setdefault(key[1], {}).setdefault(key[2], {})[key[3]] = min(v)
        del W, Ws, runners
        torch.cuda.empty_cache()
    if a.json:
        import json
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
