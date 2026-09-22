#!/usr/bin/env python3
"""Tile-config sweep for libr9k's GEMMs on the per-rank shapes of BOTH served models, timed kernel-only under HIP
graph replay (launch overhead excluded, as in the served decode graphs). Writes r9700_vllm/kernels/tuned.json, which
kernels/moe.py:pick_cfg and kernels/fp8.py read at runtime ({kind: {"N,K": {"M": [WV, SK, NPW]}}}; M is the next
bucket >= the call's M).

kinds: mxfp4 / nvfp4 (grouped kernel, single expert = dense), fp8row (r9k_gemm_fp8), fp8block (r9k_gemm_fp8_block).
usage: tune_dense.py [--quick] [--out PATH]   (inside the ROCm image, one GPU)
"""
import argparse
import json
import os
import sys
import time

import torch

os.environ["R9K_TUNED"] = "/nonexistent"    # "default" column = untuned behaviour

from r9700_vllm.kernels import fp8 as F8, moe as K

MS = [1, 4, 8, 16, 32, 64]
# per-rank shapes at TP2: (N, K)
SHAPES = {
    # Qwen3.8-27B-NVFP4: NVFP4 MLPs (fp8 in the last 8 layers), fp8 attention / GDN / LM head
    "nvfp4": [(17408, 5120), (5120, 8704)],
    "fp8row": [(17408, 5120), (5120, 8704), (7168, 5120), (5120, 3072), (8192, 5120), (124160, 5120),
               # Flash-Next: fp8 LM head shadows (target + MTP draft)
               (124160, 2560),
               # GDN in_proj_ba (27B: 2*48 heads / TP2) when served fp8 (R9K_FP8_LINEARS=in_proj_ba)
               (48, 5120)],
    # Flash-Next shared expert (dense MXFP4) + 27B converted path (R9K_NVFP4=mxfp4)
    "mxfp4": [(640, 2560), (2560, 320), (17408, 5120), (5120, 8704),
              # R9K_FP8_TO_MXFP4: 27B attention / GDN, Flash-Next attention / GDN (block fp8 requantized)
              (7168, 5120), (8192, 5120), (5120, 3072), (8192, 2560), (6656, 2560), (2560, 3072),
              # GDN in_proj_qkvz + in_proj_ba merged (models/gdn.py), 27B
              (8240, 5120),
              # R9K_DRAFT_W4: DFlash2 drafter qkv / o_proj
              (3072, 5120), (5120, 2048)],
    # Flash-Next attention / GDN block-fp8 projections
    "fp8block": [(8192, 2560), (6656, 2560), (2560, 3072),
                 # DFlash2 drafter for the 27B (block fp8): qkv, o, gate_up, down per rank
                 (3072, 5120), (5120, 2048), (17408, 5120), (5120, 8704)],
}


def configs(K_, group):
    out = []
    for WV in (1, 2, 4, 8):
        for SK in (1, 2, 4, 5, 8, 10, 16):
            if WV * SK * 32 > 1024 or K_ % (SK * group):
                continue
            for NPW in (1, 2, 4):
                if WV * NPW * SK * 256 * 4 > 64 * 1024:
                    continue
                out.append((WV, SK, NPW))
    return out


def graph_time(fn, reps=20, iters=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(iters):
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / reps)
    return best * 1e6


def make(kind, N, Kd):
    g = torch.Generator(device="cuda").manual_seed(0)
    if kind == "quant":                    # no weight: the "shape" is (rows M, K) of the activation, see _runner
        return None, 16, 1 << 20
    if kind in ("mxfp4", "nvfp4"):
        packed = torch.randint(0, 256, (1, N, Kd // 2), dtype=torch.uint8, device="cuda", generator=g)
        if kind == "mxfp4":
            sc = torch.randint(118, 128, (1, N, Kd // 32), dtype=torch.uint8, device="cuda", generator=g)
            return K.prepare_mxfp4_weights(packed, sc), 32, N * Kd // 2 + N * Kd // 32
        s16 = (torch.rand(1, N, Kd // 16, device="cuda", generator=g) + 0.5).to(torch.float8_e4m3fn)
        return K.prepare_nvfp4_weights(packed, s16, torch.ones(1, N, device="cuda")), 16, N * Kd // 2 + N * Kd // 16
    w8 = (torch.randn(N, Kd, device="cuda", generator=g)).to(torch.float8_e4m3fn)
    wq = F8.permute_fp8(w8.view(torch.uint8))
    if kind == "fp8row":
        return F8.Fp8Weight(wq, torch.ones(N, device="cuda"), N, Kd), 16, N * Kd
    return (wq, torch.rand((N + 127) // 128, Kd // 128, device="cuda") + 0.5), 128, N * Kd


def ldsa_ok(cfg, mt, Kd, dbk=64):
    """Mirror of the kernel's LDS-A eligibility rule (moe_launch)."""
    WV, SK = cfg[0], cfg[1]
    return Kd % (SK * dbk) == 0 and WV * SK <= 8 and (2 * mt) % WV == 0 and (2 * mt) // WV <= 4


def mt_for(M):
    return 4 if M >= 64 else (2 if M >= 32 else 1)


def runner(kind, Ws, N, Kd, M, cfg, fold=False):
    """Cycles through the weight copies so every call streams from DRAM (R9700: ~64 MB of on-die cache would
    otherwise serve repeated calls on one small weight and inflate GB/s). fold: folded-exponent MXFP4 kernels."""
    if fold:
        import dataclasses
        Ws = [dataclasses.replace(W, fold=True) for W in Ws]
    fns = [_runner(kind, W, N, Kd, M, cfg) for W in Ws]
    it = [0]

    def f():
        fns[it[0] % len(fns)]()
        it[0] += 1
    return f


def _runner(kind, W, N, Kd, M, cfg):
    x = torch.randn(M, Kd, device="cuda").to(torch.bfloat16)
    if kind == "quant":                                          # activation quantizer alone: cfg "row" / "tiled"
        tiled = cfg == "tiled"
        return lambda: K.quant_rows_fp8(x, tiled=tiled)
    if kind == "mxfp4" and K.is_atiled_cfg(cfg):                 # ("A", cfg): A-tiled prefill kernel (W.fold: variant)
        q, s = K.quant_rows_fp8(x, tiled=True)
        blk = K.atiled_block(cfg[1])
        mpad = (M + blk - 1) // blk * blk
        t = (torch.arange(mpad, dtype=torch.int32, device="cuda"), torch.zeros(mpad // blk, dtype=torch.int32,
             device="cuda"), torch.full((1,), mpad, dtype=torch.int32, device="cuda"))
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        return lambda: K.moe_gemm(q, s, W, out, *t, M, 1, None, num_experts=1, prefill=cfg[1], a_tiled=True)
    if kind in ("mxfp4", "nvfp4") and K.is_prefill_cfg(cfg):   # ("P", tile cfg): LDS-tiled prefill kernel
        q, s = K.quant_rows_fp8(x)
        blk = K.prefill_block(cfg[1])
        mpad = (M + blk - 1) // blk * blk
        t = (torch.arange(mpad, dtype=torch.int32, device="cuda"), torch.zeros(mpad // blk, dtype=torch.int32,
             device="cuda"), torch.full((1,), mpad, dtype=torch.int32, device="cuda"))
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        return lambda: K.moe_gemm(q, s, W, out, *t, M, 1, None, num_experts=1, prefill=cfg[1])
    WV, SK, NPW = cfg[:3]
    if kind in ("mxfp4", "nvfp4"):
        q, s = K.quant_rows_fp8(x)
        MT = cfg[3] if len(cfg) > 3 else mt_for(M)
        ldsa = bool(cfg[4]) if len(cfg) > 4 else False
        blk = 16 * MT
        mpad = (M + blk - 1) // blk * blk
        t = (torch.arange(mpad, dtype=torch.int32, device="cuda"), torch.zeros(mpad // blk, dtype=torch.int32,
             device="cuda"), torch.full((1,), mpad, dtype=torch.int32, device="cuda"))
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        return lambda: K.moe_gemm(q, s, W, out, *t, M, 1, None, WV, SK, NPW, num_experts=1, MT=MT, ldsa=ldsa)
    if kind == "fp8row":
        q, s = K.quant_rows_fp8(x)
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        return lambda: F8.gemm_fp8(q, s, W, out, WV, SK, NPW)
    q, s = F8.quant_group128_fp8(x)
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    return lambda: F8.gemm_fp8_block(q, s, W[0], W[1], N, Kd, out, WV, SK, NPW)


def default_cfg(kind, N, Kd, M):
    if kind in ("mxfp4", "nvfp4"):
        return K.pick_cfg(N, Kd, 16 if kind == "nvfp4" else 32, M=M, kind=kind)
    return F8.pick_cfg(kind, N, Kd, M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(K.__file__), "tuned.json"))
    ap.add_argument("--kinds", default=",".join(SHAPES))
    ap.add_argument("--shapes", default="", help="only these N,K pairs, e.g. 7168,5120;8192,5120")
    ap.add_argument("--ms", default="", help="only these M values, e.g. 32,64")
    a = ap.parse_args()
    ms = [int(v) for v in a.ms.split(",")] if a.ms else ([1, 8, 64] if a.quick else MS)
    table = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for kind in a.kinds.split(","):
        only = {tuple(int(v) for v in p.split(",")) for p in a.shapes.split(";") if p}
        for (N, Kd) in SHAPES[kind]:
            if only and (N, Kd) not in only:
                continue
            W, group, nbytes = make(kind, N, Kd)
            Ws = [W] + [make(kind, N, Kd)[0] for _ in range(min(15, (256 << 20) // nbytes))]
            base = configs(Kd, group)
            for M in ms:
                # 4-bit kernels: the M-tile count per routing block is tuned too (MT tiles share each weight
                # fragment; fewer, wider blocks trade parallelism for weight re-reads), and for MT > 1 whether
                # the A tile is staged through LDS (kernel-legal configs only)
                cand = [c + (mt, ld) for c in base for mt in (1, 2, 4) if 16 * mt <= max(16, 2 * M)
                        for ld in ((0, 1) if mt > 1 and ldsa_ok(c, mt, Kd) else (0,))] \
                    if kind in ("mxfp4", "nvfp4") else base
                d = default_cfg(kind, N, Kd, M)
                best, bcfg = 1e9, None
                for cfg in cand:
                    try:
                        us = graph_time(runner(kind, Ws, N, Kd, M, cfg), reps=2 * len(Ws))
                    except RuntimeError:
                        continue
                    if us < best:
                        best, bcfg = us, cfg
                try:
                    dus = graph_time(runner(kind, Ws, N, Kd, M, d), reps=2 * len(Ws))
                except RuntimeError:
                    dus = float("nan")
                if dus == dus and dus <= best:          # the default wins (or ties): keep it
                    best, bcfg = dus, tuple(d)
                table.setdefault(kind, {}).setdefault(f"{N},{Kd}", {})[str(M)] = list(bcfg)
                print(f"{kind:8s} N={N:6d} K={Kd:5d} M={M:3d}: best {bcfg} {best:7.1f} us {nbytes / best / 1e3:6.0f} GB/s"
                      f" | default {tuple(d)} {dus:7.1f} us ({dus / best:.2f}x)", flush=True)
            json.dump(table, open(a.out, "w"), indent=1, sort_keys=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
