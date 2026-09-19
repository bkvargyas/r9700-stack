"""models/gdn.py merge rule: concatenating two libr9k weights along N (MXFP4 fragments + packed scales, or fp8 fragments
+ row scales) gives the two separate GEMMs' outputs side by side (up to split-K summation order)."""
import sys
import torch
from r9700_vllm import ops
from r9700_vllm.kernels import moe as K, fp8 as F8
from r9700_vllm.quant.nvfp4 import quantize_mxfp4_search

ok = True
g = torch.Generator(device="cuda").manual_seed(0)
Kd = 5120
for M in (1, 8, 64):
    x = torch.randn(M, Kd, device="cuda", generator=g).to(torch.bfloat16)
    ws = [torch.randn(n, Kd, device="cuda", generator=g) * 0.02 for n in (8192, 48)]
    # MXFP4
    Ws = [K.prepare_mxfp4_weights(*(t[None] for t in quantize_mxfp4_search(w))) for w in ws]
    sep = [ops.mxfp4_linear(x, W.wq, W.wsr, W.N, Kd) for W in Ws]
    wq = torch.cat([W.wq.reshape(1, -1) for W in Ws], dim=1)
    wsr = torch.cat([W.wsr for W in Ws], dim=2)
    mer = ops.mxfp4_linear(x, wq, wsr, sum(W.N for W in Ws), Kd)
    ref = torch.cat(sep, dim=1).float()
    same = ((mer.float() - ref).abs().max() / ref.abs().max()).item() < 5e-3   # split-K order + bf16 output rounding
    ok &= same
    # fp8 row
    Fs = [F8.quantize_rows_fp8(w) for w in ws]
    sep8 = [ops.fp8_linear(x, W) for W in Fs]
    Wm = F8.Fp8Weight(torch.cat([W.wq for W in Fs]), torch.cat([W.ws for W in Fs]), sum(W.N for W in Fs), Kd)
    ref8 = torch.cat(sep8, dim=1).float()
    same8 = ((ops.fp8_linear(x, Wm).float() - ref8).abs().max() / ref8.abs().max()).item() < 5e-3
    ok &= same8
    print(f"  M={M:2d}: mxfp4 merged == separate {same}   fp8 merged == separate {same8}")
print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
