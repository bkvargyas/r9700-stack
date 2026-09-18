#!/usr/bin/env python3
"""Tune vLLM's Triton W8A8 block-FP8 GEMM (_w8a8_triton_block_scaled_mm) for gfx1201 and write vLLM-format configs.

vLLM ships no configs for AMD_Radeon_R9700 ("Using default W8A8 Block FP8 kernel config"); Flash-Next's attention
and GDN projections (block 128x128) run ~130 of these per decode step. Output files go in
vllm/model_executor/layers/quantization/utils/configs/ (the launcher bind-mounts tuning/configs/ there).

usage: tune_fp8_block.py [OUTDIR]   (run inside the ROCm 10 image, one GPU)
"""
import itertools
import json
import os
import sys
import time

import torch
import triton

from vllm.model_executor.layers.quantization.utils import fp8_utils as F

OUT = sys.argv[1] if len(sys.argv) > 1 else "configs"
os.makedirs(OUT, exist_ok=True)
SHAPES = [(8192, 2560), (6656, 2560), (2560, 3072)]          # Flash-Next per-rank at TP2
MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
BLK = [128, 128]
dev = "cuda"
fp8 = torch.float8_e4m3fn


def space(M):
    bms = [16] if M <= 16 else ([16, 32] if M <= 32 else ([32, 64] if M <= 128 else [64, 128]))
    for bm, bn, bk, g, w, s in itertools.product(bms, [32, 64, 128], [64, 128], [1, 8], [2, 4, 8], [1, 2]):
        if bm * bn >= 128 * 128 and w == 2:
            continue
        yield {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": g,
               "num_warps": w, "num_stages": s}


def bench(A, B, As, Bs, cfg):
    orig = F.get_w8a8_block_fp8_configs
    F.get_w8a8_block_fp8_configs = lambda *a, **k: {A.shape[0]: cfg}
    try:
        fn = lambda: F.w8a8_triton_block_scaled_mm(A, B, As, Bs, BLK, torch.bfloat16)  # noqa: E731
        fn()
        return triton.testing.do_bench(fn, warmup=5, rep=20)
    except Exception:
        return float("inf")
    finally:
        F.get_w8a8_block_fp8_configs = orig


dn = F.get_device_name_as_file_name()
for N, K in SHAPES:
    B = torch.randn(N, K, device=dev).to(fp8)
    Bs = torch.rand(triton.cdiv(N, 128), triton.cdiv(K, 128), device=dev) * 0.01
    best = {}
    t0 = time.time()
    for M in MS:
        A = torch.randn(M, K, device=dev).to(fp8)
        As = torch.rand(M, triton.cdiv(K, 128), device=dev) * 0.01
        default = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32,
                   "num_warps": 4, "num_stages": 2}
        tdef = bench(A, B, As, Bs, default)
        res = sorted(((bench(A, B, As, Bs, c), c) for c in space(M)), key=lambda x: x[0])
        t, c = res[0]
        best[str(M)] = c
        print(f"N={N} K={K} M={M:5d}: best {t*1e3:8.1f} us (default {tdef*1e3:8.1f} us, {tdef/t:4.2f}x) {c}",
              flush=True)
    fn = f"N={N},K={K},device_name={dn},dtype=fp8_w8a8,block_shape=[128,128].json"
    json.dump(best, open(os.path.join(OUT, fn), "w"), indent=2)
    print(f"wrote {fn} ({time.time()-t0:.0f}s)", flush=True)
