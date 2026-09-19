#!/usr/bin/env python3
"""DRAM-fed timing of libr9k GEMMs on chosen shapes/configs (same make/runner/graph_time as tune_dense.py).

usage: fable_bench.py --kind mxfp4 --shape 17408,5120 --M 8,64 [--cfg tuned | WV,SK,NPW[,MT] | sweep]
                      [--tuned PATH] [--json OUT]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import tune_dense as T                       # sets R9K_TUNED=/nonexistent before importing the bindings

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TUNED = os.path.join(os.path.dirname(HERE), "r9700_vllm", "kernels", "tuned.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="mxfp4")
    ap.add_argument("--shape", default="17408,5120")
    ap.add_argument("--M", default="8,64")
    ap.add_argument("--cfg", default="tuned")
    ap.add_argument("--tuned", default=os.environ.get("FABLE_TUNED", DEFAULT_TUNED))
    ap.add_argument("--reps", type=int, default=0)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--ncopies", type=int, default=0, help="weight copies rotated; 0 = tune_dense rule (>=256 MB, defeats the MALL)")
    ap.add_argument("--mts", default="1,2,4")
    ap.add_argument("--json", default="")
    ap.add_argument("--arow0", action="store_true", help="route every row to A row 0 (A-footprint diagnostic)")
    a = ap.parse_args()
    os.environ["R9K_TUNED"] = a.tuned          # tuned.py reads it lazily at first lookup
    from r9700_vllm.kernels import moe as K
    if a.arow0:
        _orig = T._runner

        def _r(kind, W, N, Kd, M, cfg):
            fn = _orig(kind, W, N, Kd, M, cfg)
            if kind in ("mxfp4", "nvfp4"):
                fn.__closure__[list(fn.__code__.co_freevars).index("t")].cell_contents[0].zero_()
            return fn
        T._runner = _r
    results = {}
    for kind in a.kind.split(","):
        for shp in a.shape.replace("+", ";").split(";"):
            N, Kd = (int(v) for v in shp.split(","))
            W, group, nbytes = T.make(kind, N, Kd)
            nc = a.ncopies or min(15, (256 << 20) // nbytes)
            Ws = [W] + [T.make(kind, N, Kd)[0] for _ in range(nc)]
            reps = a.reps or 2 * len(Ws)
            for M in (int(v) for v in a.M.split(",")):
                if a.cfg == "tuned":
                    cands = [tuple(T.default_cfg(kind, N, Kd, M))]
                elif a.cfg == "sweep":
                    base = T.configs(Kd, group)
                    mts = [int(v) for v in a.mts.split(",")]
                    cands = [c + (mt, ld) for c in base for mt in mts if 16 * mt <= max(16, 2 * M)
                             for ld in ((0, 1) if mt > 1 and T.ldsa_ok(c, mt, Kd) else (0,))] \
                        if kind in ("mxfp4", "nvfp4") else base
                else:
                    cands = [tuple(int(v) for v in c.split(",")) for c in a.cfg.replace("/", ";").split(";")]
                rows = []
                for cfg in cands:
                    try:
                        us = T.graph_time(T.runner(kind, Ws, N, Kd, M, cfg), reps=reps, iters=a.iters)
                    except RuntimeError as e:
                        if a.cfg != "sweep":
                            print(f"{kind} N={N} K={Kd} M={M} cfg={cfg}: FAILED {e}")
                        continue
                    rows.append((us, cfg))
                rows.sort()
                for us, cfg in rows[:a.top]:
                    print(f"{kind:6s} N={N:6d} K={Kd:5d} M={M:3d} cfg={str(cfg):18s} {us:7.1f} us {nbytes / us / 1e3:6.0f} GB/s",
                          flush=True)
                if rows:
                    results[f"{kind}:{N},{Kd}:{M}"] = [list(rows[0][1]), rows[0][0]]
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
