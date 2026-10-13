"""The fused decode MoE path (kernels/r9k_moe_mxfp4a8.hip FUSE 1 / 2) against the three-launch path: the gate_up GEMM
with silu(g)*u in its epilogue (fp32 act) and the down GEMM quantising its A rows on load must give the SAME bytes as
gate_up -> r9k_silu_mul_quant_fp8 -> down (bit-identical outputs), plain and folded MXFP4, at decode widths."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from r9700_vllm.kernels import moe as K                                 # noqa: E402
from test_moe_mxfp4 import make_experts                                 # noqa: E402

dev = "cuda"
fails = 0


def case(E, M, topk, I, Kd, seed, fold, cfg1=(1, 8, 1), cfg2=(4, 2, 1)):
    global fails
    g = torch.Generator().manual_seed(seed)
    N1, N2, K1, K2 = 2 * I, Kd, Kd, I
    cfg1, cfg2 = K.legal_cfg(cfg1, N1, K1, K.GROUP), K.legal_cfg(cfg2, N2, K2, K.GROUP)   # as experts.py picks them
    p1, s1, _ = make_experts(E, N1, K1, g)
    p2, s2, _ = make_experts(E, N2, K2, g)
    w1 = K.prepare_mxfp4_weights(p1.to(dev), s1.to(dev)); w1.fold = fold
    w2 = K.prepare_mxfp4_weights(p2.to(dev), s2.to(dev)); w2.fold = fold
    x = (torch.randn(M, K1, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    scores = torch.randn(M, E, generator=g)
    topk_w, topk_ids = torch.softmax(scores, -1).topk(topk, dim=-1)
    sid, eid, ntpp = (t.to(dev) for t in K.align_block_size_ref(topk_ids, E, K.MOE_BLOCK))
    numel = M * topk
    tw = topk_w.reshape(-1).float().to(dev)
    xq, xs = K.quant_rows_fp8(x)
    # three launches
    gu = torch.zeros(numel, N1, dtype=torch.bfloat16, device=dev)
    K.moe_gemm(xq, xs, w1, gu, sid, eid, ntpp, numel, topk, None, *cfg1, MT=1)
    aq, as_ = K.silu_mul_quant_fp8(gu)
    down_ref = torch.zeros(numel, N2, dtype=torch.bfloat16, device=dev)
    K.moe_gemm(aq, as_, w2, down_ref, sid, eid, ntpp, numel, 1, tw, *cfg2, MT=1)
    # two launches
    act = torch.zeros(numel, I, dtype=torch.float32, device=dev)
    K.moe_gemm(xq, xs, w1, act, sid, eid, ntpp, numel, topk, None, *cfg1, MT=1, fuse=1)
    down = torch.zeros(numel, N2, dtype=torch.bfloat16, device=dev)
    K.moe_gemm(act, None, w2, down, sid, eid, ntpp, numel, 1, tw, *cfg2, MT=1, fuse=2)
    torch.cuda.synchronize()
    # the act the separate kernel saw: silu(bf16 g) * bf16 u in fp32
    gf, uf = gu[:, :I].float(), gu[:, I:].float()
    act_ref = gf / (1 + torch.exp(-gf)) * uf
    # the kernels compute silu with __expf, torch with its own exp: fp32-ulp differences in the act are expected; the
    # down output (through the same fp8 quant as before) is what has to be bit-equal
    a_rel = ((act - act_ref).norm() / act_ref.norm().clamp_min(1e-9)).item()
    a_ok = a_rel < 1e-6
    d_ok = torch.equal(down, down_ref)
    d_rel = ((down.float() - down_ref.float()).norm() / down_ref.float().norm().clamp_min(1e-9)).item()
    print(f"  E={E} M={M} topk={topk} I={I} K={Kd} fold={int(fold)}: act rel {a_rel:.1e} ({'ok' if a_ok else 'FAIL'}), "
          f"down {'bit-equal' if d_ok else f'rel {d_rel:.2e}'} {'ok' if a_ok and d_ok else '<-- FAIL'}")
    fails += 0 if (a_ok and d_ok) else 1


for fold in (False, True):
    case(8, 4, 4, 160, 2560, 1, fold)          # Flash-Next TP4 shapes at MTP-3 decode
    case(16, 8, 10, 160, 2560, 2, fold)
    case(8, 3, 4, 320, 2560, 3, fold)          # TP2 intermediate
    case(32, 16, 10, 160, 2560, 4, fold)
print("test_moe_fused:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
