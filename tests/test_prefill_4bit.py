"""Large-M (prefill) path of the 4-bit kernels: r9k_moe_4bit_prefill on every tile cfg, dense MXFP4 and NVFP4 at
M >= 256 (incl. ragged M and N tails) against a dequantized reference, a routed MoE case with many rows per expert
(gate_up with a_row_div = topk, down with the router weight folded in), and the ops-level dispatch at prefill M.

Run inside the ROCm image: PYTHONPATH=/repo R9K_LIB=/repo/r9700_vllm/kernels/libr9k.so python3 tests/test_prefill_4bit.py
"""
import sys

import torch

from r9700_vllm.kernels import moe as K
from r9700_vllm.quant.nvfp4 import quantize_mxfp4_search, dequant_nvfp4

ok = True
g = torch.Generator(device="cuda").manual_seed(0)
LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")


def mx_deq(p, e):
    N, Kh = p.shape
    c = torch.stack((p & 15, p >> 4), -1).reshape(N, Kh * 2).long()
    return (LUT[c].reshape(N, -1, 32) * torch.exp2(e.float() - 127).unsqueeze(-1)).reshape(N, Kh * 2)


def make_w(kind, N, Kd, E=1):
    """E experts of an (N, K) weight -> (Experts, dequantized [E, N, K] fp32)."""
    Ws, wd = [], []
    for _ in range(E):
        w = torch.randn(N, Kd, device="cuda", generator=g) * 0.02
        if kind == "mxfp4":
            p, e = quantize_mxfp4_search(w)
            Ws.append((p, e))
            wd.append(mx_deq(p, e))
        else:
            gd = 448 * 6 / w.abs().max()
            s16 = (w.reshape(N, -1, 16).abs().amax(-1) / 6 * gd).clamp(min=2 ** -9).to(torch.float8_e4m3fn)
            p, _ = quantize_mxfp4_search(w)
            Ws.append((p, s16, (1 / gd).expand(N).contiguous()))
            wd.append(dequant_nvfp4(p, s16, gd.expand(N).contiguous()))
    if kind == "mxfp4":
        W = K.prepare_mxfp4_weights(torch.stack([a for a, _ in Ws]), torch.stack([b for _, b in Ws]))
    else:
        W = K.prepare_nvfp4_weights(torch.stack([a for a, _, _ in Ws]), torch.stack([b for _, b, _ in Ws]),
                                    torch.stack([c for _, _, c in Ws]))
    return W, torch.stack(wd)


def check(name, got, ref, tol=5e-3):
    global ok
    r = ((got.float() - ref).norm() / ref.norm()).item()
    good = r < tol and torch.isfinite(got.float()).all().item()
    ok &= good
    print(f"  {name:70s} rel {r:.2e} {'ok' if good else '<-- FAIL'}", flush=True)
    return good


# ---- dense: every cfg on a mid shape, then the served shapes on the default cfgs
for kind in ("mxfp4", "nvfp4"):
    W, wd = make_w(kind, 1040, 1024)                     # N=1040: 16-column tail for every BN
    for cfg in sorted(K.PREFILL_TILES):
        for M in (256, 300, 1024):
            blk = K.prefill_block(cfg)
            x = torch.randn(M, 1024, device="cuda", generator=g).to(torch.bfloat16)
            q, s = K.quant_rows_fp8(x)
            mpad = (M + blk - 1) // blk * blk
            t = (torch.arange(mpad, dtype=torch.int32, device="cuda"),
                 torch.zeros(mpad // blk, dtype=torch.int32, device="cuda"),
                 torch.full((1,), mpad, dtype=torch.int32, device="cuda"))
            out = torch.full((M, 1040), float("nan"), dtype=torch.bfloat16, device="cuda")
            K.moe_gemm(q, s, W, out, *t, M, 1, None, num_experts=1, prefill=cfg)
            check(f"{kind} dense 1040x1024 M={M} cfg={cfg} {K.PREFILL_TILES[cfg]}", out, (q.float() * s[:, None]) @ wd[0].T)
    for (N, Kd) in ((17408, 5120), (5120, 8704), (8240, 5120)) if kind == "mxfp4" else ((17408, 5120), (5120, 8704)):
        W, wd = make_w(kind, N, Kd)
        for M in (512, 2048):
            # this section exercises the tuned LDS-A tile ("P" rows); with R9K_ATILED on, MXFP4 dispatches to the
            # A-tiled kernel at these M whether folded or not (tests/test_atiled_4bit.py gates that path), so the
            # tiled pick is switched off for the lookup
            atiled, K.ATILED = K.ATILED, False
            try:
                cfg = K.pick_cfg(N, Kd, 32 if kind == "mxfp4" else 16, M=M, kind=kind)
            finally:
                K.ATILED = atiled
            assert K.is_prefill_cfg(cfg), (kind, N, Kd, M, cfg)
            blk = K.prefill_block(cfg[1])
            x = torch.randn(M, Kd, device="cuda", generator=g).to(torch.bfloat16)
            q, s = K.quant_rows_fp8(x)
            mpad = (M + blk - 1) // blk * blk
            t = (torch.arange(mpad, dtype=torch.int32, device="cuda"),
                 torch.zeros(mpad // blk, dtype=torch.int32, device="cuda"),
                 torch.full((1,), mpad, dtype=torch.int32, device="cuda"))
            out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
            K.moe_gemm(q, s, W, out, *t, M, 1, None, num_experts=1, prefill=cfg[1])
            check(f"{kind} dense {N}x{Kd} M={M} tuned cfg={cfg}", out, (q.float() * s[:, None]) @ wd[0].T)
        del W, wd

# ---- routed MoE (Flash-Next shapes): E experts, many rows per expert, padding blocks, a_row_div = topk, topk_w
E, M, topk = 16, 1200, 4
for cfg in (0, 2, 3, 6, 11, 15):      # 11 / 15 = the tiles experts.py uses for the routed down / gate_up GEMMs (block 64)
    W1, wd1 = make_w("mxfp4", 640, 2560, E)
    W2, wd2 = make_w("mxfp4", 2560, 320, E)
    blk = K.prefill_block(cfg)
    scores = torch.randn(M, E, device="cuda", generator=g)
    topk_w, topk_ids = torch.softmax(scores, -1).topk(topk, dim=-1)
    sid, eid, ntpp = (v.cuda() for v in K.align_block_size_ref(topk_ids.cpu(), E, blk))
    numel = M * topk
    x = torch.randn(M, 2560, device="cuda", generator=g).to(torch.bfloat16)
    xq, xs = K.quant_rows_fp8(x)
    out1 = torch.full((numel, 640), float("nan"), dtype=torch.bfloat16, device="cuda")
    K.moe_gemm(xq, xs, W1, out1, sid, eid, ntpp, numel, topk, None, num_experts=E, prefill=cfg)
    xd = xq.float() * xs[:, None]
    fl = topk_ids.flatten()

    def routed_ref(ad, wd, a_row_div):        # per-expert matmuls (a gathered [numel, N, K] would be 29 GB)
        ref = torch.empty(numel, wd.shape[1], device="cuda")
        for ex in range(E):
            rows = (fl == ex).nonzero().flatten()
            ref[rows] = ad[rows // a_row_div] @ wd[ex].T
        return ref
    ref1 = routed_ref(xd, wd1, topk)
    check(f"mxfp4 routed gate_up E={E} M={M} top{topk} cfg={cfg} blk={blk}", out1, ref1)
    h = torch.randn(numel, 320, device="cuda", generator=g).to(torch.bfloat16)
    hq, hs = K.quant_rows_fp8(h)
    tw = topk_w.flatten().float().contiguous()
    out2 = torch.full((numel, 2560), float("nan"), dtype=torch.bfloat16, device="cuda")
    K.moe_gemm(hq, hs, W2, out2, sid, eid, ntpp, numel, 1, tw, num_experts=E, prefill=cfg)
    ref2 = routed_ref(hq.float() * hs[:, None], wd2, 1) * tw[:, None]
    check(f"mxfp4 routed down    E={E} M={M} top{topk} cfg={cfg} blk={blk}", out2, ref2)

# ---- ops-level dispatch (torch.ops.r9700.*_linear) picks the prefill path at large M and the old one at decode M
try:
    from r9700_vllm import ops
    for kind in ("mxfp4", "nvfp4"):
        W, wd = make_w(kind, 1024, 2048)
        for M in (8, 64, 256, 1000):
            x = torch.randn(M, 2048, device="cuda", generator=g).to(torch.bfloat16)
            if kind == "mxfp4":
                out = ops.mxfp4_linear(x, W.wq[0], W.wsr[0], 1024, 2048)
            else:
                out = ops.nvfp4_linear(x, W.wq[0], W.ws[0], W.wg[0], 1024, 2048)
            q, s = K.quant_rows_fp8(x)
            check(f"ops.{kind}_linear 1024x2048 M={M}", out, (q.float() * s[:, None]) @ wd[0].T)
except ImportError as ex:          # no vLLM in this environment
    print("  (ops-level dispatch skipped:", ex, ")")

print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
