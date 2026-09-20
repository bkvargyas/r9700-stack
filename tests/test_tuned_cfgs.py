"""Every tuned.json config of the dense 4-bit kernels (incl. MT and LDS-A) against a dequantized reference: configs are
only reachable through the table at serving batch sizes, so they need their own correctness gate."""
import json, os, sys
import torch
from r9700_vllm.kernels import moe as K
from r9700_vllm.quant.nvfp4 import quantize_mxfp4_search, dequant_nvfp4
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tuning"))

ok = True
g = torch.Generator(device="cuda").manual_seed(0)
T = json.load(open(os.path.join(os.path.dirname(K.__file__), "tuned.json")))
LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")


def mx_deq(p, e):
    N, Kh = p.shape
    c = torch.stack((p & 15, p >> 4), -1).reshape(N, Kh * 2).long()
    return (LUT[c].reshape(N, -1, 32) * torch.exp2(e.float() - 127).unsqueeze(-1)).reshape(N, Kh * 2)


for kind in ("mxfp4", "nvfp4"):
    for shape, per_m in T.get(kind, {}).items():
        N, Kd = map(int, shape.split(","))
        w = torch.randn(N, Kd, device="cuda", generator=g) * 0.02
        if kind == "mxfp4":
            p, e = quantize_mxfp4_search(w)
            W, wd = K.prepare_mxfp4_weights(p[None], e[None]), mx_deq(p, e)
        else:
            gd = 448 * 6 / w.abs().max()
            x16 = w.reshape(N, -1, 16)
            s16 = (x16.abs().amax(-1) / 6 * gd).clamp(min=2 ** -9).to(torch.float8_e4m3fn)
            p, _ = quantize_mxfp4_search(w)                    # any valid e2m1 codes; reference uses the same bytes
            W = K.prepare_nvfp4_weights(p[None], s16[None], (1 / gd).expand(N)[None])
            wd = dequant_nvfp4(p, s16, gd.expand(N).contiguous())
        for Ms, cfg in per_m.items():
            M = int(Ms)
            x = torch.randn(M, Kd, device="cuda", generator=g).to(torch.bfloat16)
            q, s = K.quant_rows_fp8(x)
            if K.is_prefill_cfg(cfg):                       # ["P", tile]: LDS-tiled prefill kernel
                blk, MT, ld = K.prefill_block(cfg[1]), None, False
            else:
                MT = cfg[3] if len(cfg) > 3 else (4 if M >= 64 else 2 if M >= 32 else 1)
                ld = bool(cfg[4]) if len(cfg) > 4 else False
                blk = 16 * MT
            mpad = (M + blk - 1) // blk * blk
            t = (torch.arange(mpad, dtype=torch.int32, device="cuda"), torch.zeros(mpad // blk, dtype=torch.int32,
                 device="cuda"), torch.full((1,), mpad, dtype=torch.int32, device="cuda"))
            out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
            if MT is None:
                K.moe_gemm(q, s, W, out, *t, M, 1, None, num_experts=1, prefill=cfg[1])
            else:
                K.moe_gemm(q, s, W, out, *t, M, 1, None, *cfg[:3], num_experts=1, MT=MT, ldsa=ld)
            ref = (q.float() * s[:, None]) @ wd.T
            r = ((out.float() - ref).norm() / ref.norm()).item()
            good = r < 5e-3
            ok &= good
            if not good or M >= 32:
                print(f"  {kind} {N}x{Kd} M={M:2d} cfg={cfg}: rel {r:.2e} {'ok' if good else '<-- FAIL'}")
print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
