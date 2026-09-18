"""NVFP4 -> MXFP4 load-time conversion (quant/nvfp4.py): exact NVFP4 dequant vs stock's reference, conversion error
vs a direct RTN-MXFP4 quantization of the same weights, and a GEMM through the libr9k MXFP4 kernel."""
import sys
import torch
from r9700_vllm.quant import nvfp4 as NV
from r9700_vllm.models.lm_heads import quantize_mxfp4
from r9700_vllm.kernels import moe as K

ok = True
g = torch.Generator().manual_seed(0)
LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def nvfp4_quant(w):
    """CT-style NVFP4: global divisor gd = 448*6/amax, per-16 e4m3 scale, RTN E2M1. -> packed, scale16, gd"""
    N, Kd = w.shape
    gd = (448.0 * 6.0 / w.abs().max()).float()
    x = w.float().reshape(N, Kd // 16, 16)
    s16 = (x.abs().amax(-1) / 6.0 * gd).clamp(min=2 ** -9).to(torch.float8_e4m3fn)
    v = x * gd / s16.float().unsqueeze(-1)
    code = torch.bucketize(v.abs(), NV._MID.to(w.device)).to(torch.uint8) | ((v < 0).to(torch.uint8) << 3)
    code = code.reshape(N, Kd)
    return code[:, 0::2] | (code[:, 1::2] << 4), s16, gd


def mx_dequant(p, e8):
    N, Kh = p.shape
    lut = NV._E2M1.to(p.device)
    c = torch.stack((p & 15, p >> 4), -1).reshape(N, Kh * 2).long()
    return (lut[c].reshape(N, -1, 32) * torch.exp2(e8.float() - 127).unsqueeze(-1)).reshape(N, Kh * 2)


from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import break_fp4_bytes
for (N, Kd) in [(256, 512), (4096, 2560), (1536, 8704)]:
    # heavy-ish tails like real weights
    w = (torch.randn(N, Kd, generator=g) * torch.exp(torch.randn(N, 1, generator=g) * 0.3) * 0.02).cuda()
    p16, s16, gd = nvfp4_quant(w)
    div = gd.expand(N).contiguous()
    deq = NV.dequant_nvfp4(p16, s16, div)
    ref = (break_fp4_bytes(p16, torch.float32).reshape(N, -1, 16) * s16.float().unsqueeze(-1)).reshape(N, Kd) / gd
    exact = torch.equal(deq, ref)
    ok &= exact
    pm, em = NV.nvfp4_to_mxfp4(p16, s16, div)
    conv = mx_dequant(pm, em)
    pr, er = quantize_mxfp4(w.to(torch.bfloat16))
    rtn = mx_dequant(pr, er)
    e_nv, e_conv, e_rtn, e_c2n = rel(deq, w), rel(conv, w), rel(rtn, w), rel(conv, deq)
    good = e_conv < 1.15 * e_rtn and e_c2n < 0.2
    ok &= good
    print(f"  {N}x{Kd}: dequant exact={exact}  rel err vs bf16: nvfp4 {e_nv:.4f}  nvfp4->mxfp4 {e_conv:.4f}  "
          f"rtn-mxfp4 {e_rtn:.4f}  (conv vs nvfp4 {e_c2n:.4f}) {'ok' if good else '<-- FAIL'}")

    # GEMM through the libr9k MXFP4 kernel on the converted weights
    M = 4
    x = torch.randn(M, Kd, generator=g).to(torch.bfloat16).cuda()
    W = K.prepare_mxfp4_weights(pm[None], em[None])
    q, s = K.quant_rows_fp8(x)
    mpad = 16
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    K.moe_gemm(q, s, W, out, torch.arange(mpad, dtype=torch.int32, device="cuda"),
               torch.zeros(1, dtype=torch.int32, device="cuda"), torch.tensor([mpad], dtype=torch.int32, device="cuda"),
               M, 1, None, num_experts=1)
    r = rel(out, x.float() @ conv.T)
    good = r < 0.05
    ok &= good
    print(f"      kernel on converted weights vs fp32 matmul: rel {r:.4f} {'ok' if good else '<-- FAIL'}")

print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
