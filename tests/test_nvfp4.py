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
    # two independent roundings: the floor is ~sqrt(e_nv^2 + e_rtn^2); fail only if clearly above it
    floor = (e_nv ** 2 + e_rtn ** 2) ** 0.5
    alt = {c: rel(mx_dequant(*NV.quantize_mxfp4_search(deq, c)), w) for c in ((0,), (0, -1), (0, -1, -2), (1, 0, -1))}
    print("      scale candidates:", "  ".join(f"{c}: {v:.4f}" for c, v in alt.items()))
    good = e_conv < 1.05 * floor
    ok &= good
    print(f"  {N}x{Kd}: dequant exact={exact}  rel err vs bf16: nvfp4 {e_nv:.4f}  nvfp4->mxfp4 {e_conv:.4f}  "
          f"rtn-mxfp4 {e_rtn:.4f}  independent-floor {floor:.4f} {'ok' if good else '<-- FAIL'}")

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

# ---- native NVFP4 kernel: exact weight math -> only fp32-accumulation-order differences vs dequantized operands
import time
print("native NVFP4 kernel")
for (N, Kd, M) in [(256, 512, 3), (4096, 2560, 4), (1536, 8704, 37), (8704, 5120, 100)]:
    w = (torch.randn(N, Kd, generator=g) * 0.02).cuda()
    p16, s16, gd = nvfp4_quant(w)
    deq = NV.dequant_nvfp4(p16, s16, gd.expand(N).contiguous())
    Wn = K.prepare_nvfp4_weights(p16[None], s16[None], (1.0 / gd).expand(N)[None])
    x = torch.randn(M, Kd, generator=g).to(torch.bfloat16).cuda()
    from r9700_vllm import ops
    out = ops.nvfp4_linear(x, Wn.wq, Wn.ws, Wn.wg, N, Kd)
    q, s = K.quant_rows_fp8(x)
    refq = (q.float() * s[:, None]) @ deq.T
    rk = rel(out, refq)
    good = rk < 5e-3
    ok &= good
    print(f"  dense M={M:3d} N={N:5d} K={Kd}: rel vs dequantized operands {rk:.2e} {'ok' if good else '<-- FAIL'}")

# grouped: 8 experts, top-2 routing over 40 tokens, gate_up-style a_row_div=topk, every MT
E, N, Kd, T, top = 8, 512, 1024, 40, 2
ws_ = [(torch.randn(N, Kd, generator=g) * 0.02).cuda() for _ in range(E)]
qs = [nvfp4_quant(w) for w in ws_]
P = torch.stack([q[0] for q in qs]); S = torch.stack([q[1] for q in qs])
G = torch.stack([(1.0 / q[2]).expand(N) for q in qs])
Wn = K.prepare_nvfp4_weights(P, S, G)
deqs = [NV.dequant_nvfp4(q[0], q[1], q[2].expand(N).contiguous()) for q in qs]
x = torch.randn(T, Kd, generator=g).to(torch.bfloat16).cuda()
q, s = K.quant_rows_fp8(x)
xd = q.float() * s[:, None]
topk = torch.stack([torch.randperm(E, generator=g)[:top] for _ in range(T)]).cuda()
ref = torch.stack([torch.stack([xd[t] @ deqs[int(topk[t, j])].T for j in range(top)]) for t in range(T)]).reshape(T * top, N)
for MT in (1, 2, 4):
    sid, eid, ntpp = K.align_block_size_ref(topk.cpu(), E, 16 * MT)
    out = torch.zeros(T * top, N, dtype=torch.bfloat16, device="cuda")
    K.moe_gemm(q, s, Wn, out, sid.cuda(), eid.cuda(), ntpp.cuda(), T * top, top, None, *K.pick_cfg(N, Kd, 16),
               num_experts=E, MT=MT)
    rk = rel(out, ref)
    good = rk < 5e-3
    ok &= good
    print(f"  grouped E={E} top{top} T={T} MT={MT}: rel {rk:.2e} {'ok' if good else '<-- FAIL'}")

# speed: native NVFP4 vs MXFP4 (converted) on the same weights, dense decode shapes
from r9700_vllm import ops


def bench(fn, n=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


for (N, Kd) in [(4096, 2560), (8704, 5120), (5120, 8704)]:
    w = (torch.randn(N, Kd, generator=g) * 0.02).cuda()
    p16, s16, gd = nvfp4_quant(w)
    Wn = K.prepare_nvfp4_weights(p16[None], s16[None], (1.0 / gd).expand(N)[None])
    pm, em = NV.nvfp4_to_mxfp4(p16, s16, gd.expand(N).contiguous())
    Wm = K.prepare_mxfp4_weights(pm[None], em[None])
    for M in (1, 4, 64):
        x = torch.randn(M, Kd, generator=g).to(torch.bfloat16).cuda()
        tn = bench(lambda: ops.nvfp4_linear(x, Wn.wq, Wn.ws, Wn.wg, N, Kd))
        tm = bench(lambda: ops.mxfp4_linear(x, Wm.wq, Wm.wsr, N, Kd))
        print(f"  speed {N}x{Kd} M={M:2d}: native nvfp4 {tn:6.1f} us  mxfp4 {tm:6.1f} us  ({tn / tm:.2f}x)")

print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
