"""The LDS-tiled fp8 x fp8 prefill GEMM (r9k_moe_4bit_prefill, F8 path) against an exact reference, and its speed
against hipBLASLt bf16 (what the hyper-connection GEMMs run at prefill today) and the split-K decode kernel.

Shapes are the Flash-Next prefill ones: the hyper-connection down (10240 -> 336) and up (320 -> 10240), the
QSA / dense projections (2560 x 2560, 2560 -> 1536, 4096 -> 2560), at 4096 and 333 rows, and 64 rows (the
hand-over from the decode kernel). Reference: dequantised operands in fp32. Tolerance as test_moe_mxfp4."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from r9700_vllm.kernels import fp8 as F8                                # noqa: E402
from r9700_vllm.kernels import moe as KM                                # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda")
fails = 0


def check(name, got, exp, tol=2e-2):
    rel = ((got.float() - exp).norm() / exp.norm().clamp_min(1e-9)).item()
    ok = rel < tol
    print(f"  {name:52s} rel {rel:.3e} {'ok' if ok else '<-- FAIL'}")
    return ok


def run(M, N, K, cfg=None):
    global fails
    x = (torch.randn(M, K, device=dev) * 0.7).to(torch.bfloat16)
    w = (torch.randn(N, K, device=dev) * 0.05).to(torch.bfloat16)
    W = F8.quantize_rows_fp8(w)
    q, s = KM.quant_rows_fp8(x)
    out = F8.gemm_fp8_tiled(q, s, W, cfg=cfg)
    torch.cuda.synchronize()
    # exact reference from the same quantised operands
    xd = q.view(torch.float8_e4m3fn).float() * s[:, None].float()
    # un-permute: permute_fp8 is [nt, ks, h, r, 8] from [nt, r, ks, h, 8]
    wf = W.wq.view(torch.uint8).reshape(N // 16, K // 16, 2, 16, 8).permute(0, 3, 1, 2, 4).reshape(N, K)
    wd = wf.view(torch.float8_e4m3fn).float() * W.ws[:, None]
    ref = xd @ wd.T
    fails += 0 if check(f"fp8 tiled M={M:4d} N={N:5d} K={K:5d} cfg={F8.PREFILL_CFG if cfg is None else cfg}", out, ref) else 1
    if M >= 64:
        # the decode kernel on the same operands: same math, different accumulation order
        d = F8.gemm_fp8(q, s, W, None, *F8.pick_cfg("fp8row", N, K, M))
        rel = ((out.float() - d.float()).norm() / d.float().norm()).item()
        print(f"  {'   vs decode kernel':52s} rel {rel:.3e} {'ok' if rel < 5e-3 else '<-- FAIL'}")
        fails += 0 if rel < 5e-3 else 1
    return x, w, W, q, s


for M, N, K in ((4096, 336, 10240), (4096, 10240, 320), (4096, 2560, 2560), (4096, 1536, 2560), (4096, 2560, 4096),
                (333, 2560, 2560), (64, 2560, 2560), (2048, 336, 10240)):
    run(M, N, K)


# the fused up GEMM + gate mix against (a) the stock formula on the same fp8 gate and (b) the unfused fp8 path
def gate_mix_ref(xn, gate, hc=4):
    M, DIM = xn.shape
    HD = DIM // hc
    acc = torch.zeros((M, HD), device=xn.device)
    for s_ in range(hc):
        acc += torch.sigmoid(gate[:, s_ * HD:(s_ + 1) * HD].float()) * xn[:, s_ * HD:(s_ + 1) * HD].float()
    return acc / hc


for M in (4096, 333, 256):
    LR, DIM = 320, 10240
    lora = (torch.randn(M, LR, device=dev) * 0.5).to(torch.bfloat16)
    w_up = (torch.randn(DIM, LR, device=dev) * 0.05).to(torch.bfloat16)
    xn = torch.randn(M, DIM, device=dev).to(torch.bfloat16)
    Wm = F8.quantize_rows_fp8(F8.hc4_interleave(w_up))
    q, s = KM.quant_rows_fp8(lora)
    for cfg in ((9, 17, 1, 16, 19) if M == 4096 else (17, 9)):
        out = F8.gemm_fp8_mix(q, s, Wm, xn, cfg=cfg)
        torch.cuda.synchronize()
        # the same fp8 gate, un-interleaved, through the stock formula
        gperm = F8.gemm_fp8_tiled(q, s, Wm)                       # [M, DIM] bf16 in interleaved column order
        gate = torch.empty_like(gperm)
        gate[:, F8.hc4_perm(DIM, dev)] = gperm
        ref = gate_mix_ref(xn, gate)
        fails += 0 if check(f"fp8 up GEMM + gate mix M={M:4d} cfg={cfg:2d} vs stock formula", out, ref, 1e-2) else 1
    # against bf16 hipBLASLt + the formula (the path it replaces): fp8 operands differ, so only loosely
    ref_bf16 = gate_mix_ref(xn, torch.nn.functional.linear(lora, w_up))
    check(f"   (fp8 mix vs the bf16 path, informational)", out, ref_bf16, 1.0)


def us(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e6


print("\n  bench (M=4096): tiled fp8 (incl. nothing else) vs hipBLASLt bf16 F.linear")
for N, K in ((336, 10240), (10240, 320), (2560, 2560), (4096, 2560), (1536, 2560)):
    x = (torch.randn(4096, K, device=dev) * 0.7).to(torch.bfloat16)
    w = (torch.randn(N, K, device=dev) * 0.05).to(torch.bfloat16)
    W = F8.quantize_rows_fp8(w)
    q, s = KM.quant_rows_fp8(x)
    out = torch.empty((4096, N), dtype=torch.bfloat16, device=dev)
    best = None
    for cfg in (17, 19, 11, 18, 10, 13, 15, 16, 8, 9, 12, 0, 1, 2, 5, 6):
        try:
            t = us(lambda: F8.gemm_fp8_tiled(q, s, W, out, cfg=cfg))
        except RuntimeError:
            continue
        if best is None or t < best[0]:
            best = (t, cfg)
    t_ours, cfg = best
    t_bf16 = us(lambda: torch.nn.functional.linear(x, w))
    t_quant = us(lambda: KM.quant_rows_fp8(x))
    tflops = 2 * 4096 * N * K / t_ours / 1e6
    print(f"  N={N:5d} K={K:5d}: tiled fp8 {t_ours:7.0f} us cfg {cfg:2d} ({tflops:5.1f} TFLOPS)   bf16 hipBLASLt "
          f"{t_bf16:7.0f} us   row quant of x {t_quant:5.0f} us")
print("\n  bench (M=4096): fused fp8 up GEMM + gate mix vs bf16 hipBLASLt F.linear + r9k_hc_gate_mix")
from r9700_vllm import hc as HC                                          # noqa: E402
M, LR, DIM = 4096, 320, 10240
lora = (torch.randn(M, LR, device=dev) * 0.5).to(torch.bfloat16)
w_up = (torch.randn(DIM, LR, device=dev) * 0.05).to(torch.bfloat16)
xn = torch.randn(M, DIM, device=dev).to(torch.bfloat16)
Wm = F8.quantize_rows_fp8(F8.hc4_interleave(w_up))
q, s = KM.quant_rows_fp8(lora)
outm = torch.empty((M, DIM // 4), dtype=torch.bfloat16, device=dev)
for cfg in (9, 17, 1, 16, 19):
    try:
        t = us(lambda: F8.gemm_fp8_mix(q, s, Wm, xn, outm, cfg=cfg))
    except RuntimeError as e:
        print(f"  cfg {cfg}: {e}"); continue
    print(f"  fused mix cfg {cfg:2d}: {t:7.0f} us")
t_lin = us(lambda: torch.nn.functional.linear(lora, w_up))
gate = torch.nn.functional.linear(lora, w_up)
t_mix = us(lambda: HC.gate_mix(xn, gate, 4))
t_q = us(lambda: KM.quant_rows_fp8(lora))
print(f"  bf16 path: F.linear {t_lin:7.0f} us + r9k_hc_gate_mix {t_mix:7.0f} us = {t_lin + t_mix:7.0f} us"
      f"   (fp8 path adds the lora row quant: {t_q:4.0f} us)")
print("test_fp8_prefill_r9k:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
