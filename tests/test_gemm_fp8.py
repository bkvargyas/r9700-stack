"""r9k dense fp8 x fp8 GEMM (per-row scales) + LM-head quantizers vs torch (gfx1201)."""
import sys, time
import torch
from r9700_vllm.kernels import moe as K, fp8 as F8
from r9700_vllm.spec.draft_head import quantize_mxfp4

def rel(a, b): return ((a.float() - b.float()).norm() / b.float().norm()).item()
ok = True
g = torch.Generator().manual_seed(0)
for (M, N, Kd) in [(1, 4096, 2560), (4, 124160, 2560), (37, 1024, 2560), (100, 512, 1024)]:
    w = (torch.randn(N, Kd, generator=g) * 0.02).to(torch.bfloat16).cuda()
    x = torch.randn(M, Kd, generator=g).to(torch.bfloat16).cuda()
    ref = x.float() @ w.float().T
    W = F8.quantize_rows_fp8(w)
    q, s = K.quant_rows_fp8(x)
    out = F8.gemm_fp8(q, s, W)
    r = rel(out, ref); good = r < 3e-2; ok &= good

    if N % 16 == 0:
        pk, sc = quantize_mxfp4(w)
        Wm = K.prepare_mxfp4_weights(pk[None], sc[None])
        mpad = (M + 15) // 16 * 16
        o2 = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        K.moe_gemm(q, s, Wm, o2, torch.arange(mpad, dtype=torch.int32, device="cuda"),
                   torch.zeros(mpad // 16, dtype=torch.int32, device="cuda"),
                   torch.tensor([mpad], dtype=torch.int32, device="cuda"), M, 1, None, num_experts=1)
        r2 = rel(o2, ref); good = r2 < 0.2; ok &= good
        print(f"  mxfp4 (quantized head) M={M:3d} N={N:6d}: rel {r2:.2e} {'ok' if good else '<-- FAIL'}")
# bench LM-head shape per rank at TP2
w = (torch.randn(124160, 2560) * 0.02).to(torch.bfloat16).cuda()
W = F8.quantize_rows_fp8(w)
for M in (1, 4):
    x = torch.randn(M, 2560).to(torch.bfloat16).cuda(); q, s = K.quant_rows_fp8(x)
    for _ in range(5): F8.gemm_fp8(q, s, W)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(50): F8.gemm_fp8(q, s, W)
    torch.cuda.synchronize(); us = (time.perf_counter() - t) / 50 * 1e6
    for _ in range(5): x @ w.T
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(50): x @ w.T
    torch.cuda.synchronize(); ub = (time.perf_counter() - t) / 50 * 1e6
    print(f"  lm_head 124160x2560 M={M}: fp8 {us:.0f} us ({124160*2560/us/1e3:.0f} GB/s) vs bf16 torch {ub:.0f} us")
print("ALL OK" if ok else "FAILURES"); sys.exit(0 if ok else 1)
