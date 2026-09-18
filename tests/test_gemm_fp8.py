"""r9k dense fp8 x fp8 GEMM (per-row scales) + LM-head quantizers vs torch (gfx1201)."""
import sys, time
import torch
from r9700_vllm.kernels import moe as K, fp8 as F8
from r9700_vllm.models.lm_heads import quantize_mxfp4

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
    # kernel exactness: against the dequantized fp8 operands (only accumulation-order differences remain)
    wq = W.wq.view(torch.uint8).reshape(N // 16, Kd // 16, 2, 16, 8).permute(0, 3, 1, 2, 4).reshape(N, Kd)
    wdq = wq.view(torch.float8_e4m3fn).float() * W.ws[:, None]
    refq = (q.float() * s[:, None]) @ wdq.T
    rk = rel(out, refq); good = rk < 5e-3; ok &= good
    r = rel(out, ref)
    print(f"  fp8  M={M:3d} N={N:6d} K={Kd}: kernel rel {rk:.2e} {'ok' if good else '<-- FAIL'} "
          f"(vs bf16 {r:.2e} = fp8 W+A quantization)")

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
# block-scaled (128x128 weight blocks, per-token-group-128 activations): exactness vs dequantized operands
for (M, N, Kd) in [(1, 8192, 2560), (4, 6656, 2560), (7, 2560, 3072), (100, 2560, 3072)]:
    w = (torch.randn(N, Kd, generator=g) * 0.05).cuda()
    bs = (torch.rand((N + 127) // 128, Kd // 128, generator=g) * 0.02 + 0.001).cuda()
    wq8 = w.to(torch.float8_e4m3fn)
    x = torch.randn(M, Kd, generator=g).to(torch.bfloat16).cuda()
    q, s = F8.quant_group128_fp8(x)
    out = F8.gemm_fp8_block(q, s, F8.permute_fp8(wq8.view(torch.uint8)), bs, N, Kd)
    wd = wq8.float() * bs.repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)
    xd = q.float() * s.repeat_interleave(128, 1)
    refq = xd @ wd.T
    rk = rel(out, refq); good = rk < 5e-3; ok &= good
    print(f"  fp8-block M={M:3d} N={N:5d} K={Kd}: kernel rel {rk:.2e} {'ok' if good else '<-- FAIL'}")
    for _ in range(3): F8.gemm_fp8_block(q, s, F8.permute_fp8(wq8.view(torch.uint8)), bs, N, Kd)
wp = F8.permute_fp8(wq8.view(torch.uint8))
for M in (1, 4):
    x = torch.randn(M, 2560).to(torch.bfloat16).cuda(); w8 = torch.randn(8192, 2560).cuda().to(torch.float8_e4m3fn)
    bsb = torch.rand(64, 20).cuda() * 0.01 + 0.001; wpb = F8.permute_fp8(w8.view(torch.uint8))
    q, s = F8.quant_group128_fp8(x)
    for _ in range(5): F8.gemm_fp8_block(q, s, wpb, bsb, 8192, 2560)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(100): F8.gemm_fp8_block(q, s, wpb, bsb, 8192, 2560)
    torch.cuda.synchronize(); us = (time.perf_counter() - t) / 100 * 1e6
    print(f"  fp8-block 8192x2560 M={M}: {us:.0f} us ({8192*2560/us/1e3:.0f} GB/s)  [vLLM Triton tuned: ~120 us]")
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
