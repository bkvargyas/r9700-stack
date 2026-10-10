"""The hyper-connection decode GEMMs on fp8 fragment-order weights (kernels/r9k_hc_f8.hip; bf16 WMMA, W8A16) against
the exact dequantised reference, and their speed against the bf16 decode kernels (r9k_router_gemm, r9k_hc_up_mix).
Flash-Next shapes: down 336 x 10240 (lora 320 + 16 injection logits, hc_silu on the first 320), up 10240 x 320."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from r9700_vllm.kernels import fp8 as F8                                # noqa: E402
from r9700_vllm import hc as HC                                         # noqa: E402
from r9700_vllm import router as R                                      # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda")
fails = 0
E, K, LR, HD, HC_ = 336, 10240, 320, 2560, 4


def check(name, got, exp, tol=2e-2):
    rel = ((got.float() - exp.float()).norm() / exp.float().norm().clamp_min(1e-9)).item()
    ok = rel < tol
    print(f"  {name:52s} rel {rel:.3e} {'ok' if ok else '<-- FAIL'}")
    return ok


def dequant(W):   # Fp8Weight (fragment order) -> [N, K] fp32
    N, Kd = W.N, W.K
    wf = W.wq.view(torch.uint8).reshape(N // 16, Kd // 16, 2, 16, 8).permute(0, 3, 1, 2, 4).reshape(N, Kd)
    return wf.view(torch.float8_e4m3fn).float() * W.ws[:, None]


def hc_silu_ref(y, lr, hc):   # y bf16 [M, E]: hc_silu on the first lr columns (s = y/hc; s*sigmoid(s)), bf16 out
    y = y.float().clone()
    a = y[:, :lr] / hc
    y[:, :lr] = a * torch.sigmoid(a)
    return y.to(torch.bfloat16)


def gate_mix_ref(xn, gate, hc=4):
    M, DIM = xn.shape
    hd = DIM // hc
    acc = torch.zeros((M, hd), device=xn.device)
    for s in range(hc):
        acc += torch.sigmoid(gate[:, s * hd:(s + 1) * hd].float()) * xn[:, s * hd:(s + 1) * hd].float()
    return (acc / hc).to(torch.bfloat16)


w_down = (torch.randn(E, K, device=dev) * 0.02).to(torch.bfloat16)
w_up = (torch.randn(HD * HC_, LR, device=dev) * 0.05).to(torch.bfloat16)
Wd = F8.quantize_rows_fp8(w_down)
Wu = F8.quantize_rows_fp8(F8.hc4_interleave(w_up))
wd_f = dequant(Wd)
wu_f = torch.empty((HD * HC_, LR), device=dev)
wu_f[F8.hc4_perm(HD * HC_, dev)] = dequant(Wu)          # back to the natural row order
for M in (1, 2, 4, 8, 13, 16):
    xn = torch.randn(M, K, device=dev).to(torch.bfloat16)
    for split, kb in ((16, 1), (8, 1), (4, 1), (32, 1), (16, 2), (16, 4), (8, 8)):
        HC.FP8_DECODE_SPLIT, HC.FP8_DECODE_KB = split, kb
        out = HC.down_f8(xn, Wd.wq, Wd.ws, E, LR, float(HC_))
        out2 = HC.down_f8(xn, Wd.wq, Wd.ws, E, LR, float(HC_))      # the counters must come back reset
        torch.cuda.synchronize()
        ref = hc_silu_ref((xn.float() @ wd_f.T).to(torch.bfloat16), LR, float(HC_))
        fails += 0 if check(f"down+silu M={M} split={split:2d} kb={kb} vs exact fp8 ref", out, ref) else 1
        fails += 0 if torch.equal(out, out2) else (print("  <-- FAIL: second launch differs (counters)"), 1)[1]
    HC.FP8_DECODE_SPLIT, HC.FP8_DECODE_KB = 16, 1
    lora = (torch.randn(M, LR, device=dev) * 0.5).to(torch.bfloat16)
    out = HC.up_mix_f8(lora, Wu.wq, Wu.ws, xn, HD)
    torch.cuda.synchronize()
    ref = gate_mix_ref(xn, (lora.float() @ wu_f.T).to(torch.bfloat16))
    fails += 0 if check(f"up+mix M={M} vs exact fp8 ref", out, ref) else 1
    if M <= 8:   # against the bf16 kernels on the bf16 weights (fp8 quantisation error only, informational)
        ref_bf = HC.up_mix(lora, w_up, xn, HC_)
        check(f"   (up+mix M={M} vs the bf16 kernel, informational)", out, ref_bf, 1.0)
    HC.FP8_DECODE_SPLIT = 16


def us(fn, n=50, reps=20):
    """GPU time per call inside a replayed graph (n launches captured; Python launch cost excluded)."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(n):
            fn()
    torch.cuda.synchronize()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps / n * 1e6


for M in (4, 8):
    print(f"\n  graph-timed bench at M={M} (one hc module; a layer has two, a step 97):")
    xn = torch.randn(M, K, device=dev).to(torch.bfloat16)
    lora = (torch.randn(M, LR, device=dev) * 0.5).to(torch.bfloat16)
    for split, kb in ((8, 1), (16, 1), (32, 1), (16, 2), (32, 2), (16, 4)):
        HC.FP8_DECODE_SPLIT, HC.FP8_DECODE_KB = split, kb
        print(f"  down fp8 split {split:2d} kb {kb}: {us(lambda: HC.down_f8(xn, Wd.wq, Wd.ws, E, LR, 4.0)):6.2f} us")
    HC.FP8_DECODE_SPLIT, HC.FP8_DECODE_KB = 16, 1
    print(f"  down bf16 (r9k_router_gemm, split {HC.MIX_SPLIT}): "
          f"{us(lambda: R.router_gemm(xn, w_down, True, HC.MIX_SPLIT, LR, 4.0, 8)):6.2f} us")
    print(f"  up+mix fp8: {us(lambda: HC.up_mix_f8(lora, Wu.wq, Wu.ws, xn, HD)):6.2f} us")
    print(f"  up+mix bf16 (r9k_hc_up_mix): {us(lambda: HC.up_mix(lora, w_up, xn, HC_)):6.2f} us")
print("test_hc_f8:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
