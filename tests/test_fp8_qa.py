"""The fused-quant block-fp8 GEMM (r9k_gemm_fp8_block_qa) against quant_group128_fp8 + r9k_gemm_fp8_block on the
Flash-Next dense shapes at decode widths: the operands are the same fp8 bytes, so the outputs must agree to the
summation order; and the one-launch vs two-launch time at M = 8."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from r9700_vllm.kernels import fp8 as F8                                # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda")
fails = 0
SHAPES = [(2560, 1536), (2560, 3072), (3072, 5120), (3584, 2560), (4096, 2560), (5120, 2048), (5120, 8704),
          (6656, 2560), (8192, 2560), (17408, 5120)]


def block_weight(N, K):
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    W = F8.quantize_rows_fp8(w)                                  # fragment order; per-row scales are not used
    bs = (torch.rand((N + 127) // 128, K // 128, device=dev) * 0.5 + 0.5) * 0.01
    return W.wq, bs


def us(fn, reps=100):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e6


for N, K in SHAPES:
    wq, bs = block_weight(N, K)
    for M in (1, 3, 8, 16):
        x = (torch.randn(M, K, device=dev) * (1 + 3 * torch.rand(M, 1, device=dev))).to(torch.bfloat16)
        cfg = F8.pick_cfg("fp8block", N, K, M)
        q, s = F8.quant_group128_fp8(x)
        ref = F8.gemm_fp8_block(q, s, wq, bs, N, K, None, *cfg)
        out = F8.gemm_fp8_block_qa(x, wq, bs, N, K, None, *cfg)
        torch.cuda.synchronize()
        rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
        ok = rel < 1e-5
        fails += 0 if ok else 1
        if M == 8 or not ok:
            print(f"  N={N:6d} K={K:5d} M={M:2d} cfg={cfg}: rel {rel:.2e} {'ok' if ok else '<-- FAIL'}")
    M = 8
    x = torch.randn(M, K, device=dev).to(torch.bfloat16)
    cfg = F8.pick_cfg("fp8block", N, K, M)

    def two():
        q, s = F8.quant_group128_fp8(x)
        return F8.gemm_fp8_block(q, s, wq, bs, N, K, None, *cfg)
    t2 = us(two)
    t1 = us(lambda: F8.gemm_fp8_block_qa(x, wq, bs, N, K, None, *cfg))
    print(f"      M=8: quant+gemm {t2:6.1f} us   fused {t1:6.1f} us")
print("test_fp8_qa:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
