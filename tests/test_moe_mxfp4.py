"""Correctness of r9k grouped MXFP4 x FP8 MoE GEMM + per-row fp8 quant vs exact dequantisation (gfx1201).

Run inside the ROCm 10 image with R9K_LIB pointing at libr9k.so:
    PYTHONPATH=/repo python3 /repo/tests/test_moe_mxfp4.py
"""
import sys
import time

import torch

from r9700_vllm.kernels import moe as K

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def make_experts(E, N, Kd, g, max_d=8):
    nb = Kd // K.GROUP
    ref = torch.randint(120, 136, (E, N, 1), generator=g, dtype=torch.int32)
    drop = torch.randint(0, max_d + 1, (E, N, nb), generator=g, dtype=torch.int32)
    e8m0 = (ref - drop).clamp(0, 254)                                  # [E, N, K/32]
    codes = torch.randint(0, 16, (E, N, Kd), generator=g, dtype=torch.uint8)
    mag = E2M1[(codes & 0x7).long()]
    sign = torch.where((codes & 0x8) > 0, -1.0, 1.0)
    Wf = (mag * sign).reshape(E, N, nb, K.GROUP) * torch.exp2(e8m0.float() - 127.0).unsqueeze(-1)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()  # [E, N, K/2]
    return packed, e8m0.to(torch.uint8), Wf.reshape(E, N, Kd)


def check(name, got, exp, tol=2e-2):
    rel = ((got.float() - exp).norm() / exp.norm().clamp_min(1e-9)).item()
    ok = rel < tol
    print(f"  {name:48s} rel {rel:.3e} {'ok' if ok else '<-- FAIL'}")
    return ok


def run_case(E, M, topk, N1, K1, N2, K2, seed, cfg1=(2, 4, 2), cfg2=(4, 2, 1)):
    g = torch.Generator().manual_seed(seed)
    dev = "cuda"
    p1, s1, W1 = make_experts(E, N1, K1, g)
    p2, s2, W2 = make_experts(E, N2, K2, g)
    w1 = K.prepare_mxfp4_weights(p1.to(dev), s1.to(dev))
    w2 = K.prepare_mxfp4_weights(p2.to(dev), s2.to(dev))
    x = (torch.randn(M, K1, generator=g) * 0.5).to(torch.bfloat16)
    scores = torch.randn(M, E, generator=g)
    topk_w, topk_ids = torch.softmax(scores, -1).topk(topk, dim=-1)
    sorted_ids, expert_ids, ntpp = K.align_block_size_ref(topk_ids, E)
    numel = M * topk
    ok = True

    # quant
    xq, xs = K.quant_rows_fp8(x.to(dev))
    xd = xq.float() * xs.unsqueeze(1)
    ok &= check(f"quant_rows_fp8 M={M} K={K1}", xd.cpu(), x.float(), tol=6e-2)

    # gate_up: rows r -> token r // topk
    out1 = torch.zeros(numel, N1, dtype=torch.bfloat16, device=dev)
    K.moe_gemm(xq, xs, w1, out1, sorted_ids.to(dev), expert_ids.to(dev), ntpp.to(dev), numel, topk,
               None, *cfg1)
    Aq = xd.cpu()
    ref1 = torch.stack([Aq[r // topk] @ W1[topk_ids.flatten()[r]].T for r in range(numel)])
    ok &= check(f"gate_up E={E} M={M} top{topk} N={N1} K={K1} cfg{cfg1}", out1.cpu(), ref1)

    # down: per (token, j) input, router weight folded into the epilogue
    h = (torch.randn(numel, K2, generator=g) * 0.5).to(torch.bfloat16)
    hq, hs = K.quant_rows_fp8(h.to(dev))
    hd = (hq.float() * hs.unsqueeze(1)).cpu()
    tw = topk_w.flatten().float()
    out2 = torch.zeros(numel, N2, dtype=torch.bfloat16, device=dev)
    K.moe_gemm(hq, hs, w2, out2, sorted_ids.to(dev), expert_ids.to(dev), ntpp.to(dev), numel, 1,
               tw.to(dev), *cfg2)
    ref2 = torch.stack([(hd[r] @ W2[topk_ids.flatten()[r]].T) * tw[r] for r in range(numel)])
    ok &= check(f"down    E={E} M={M} top{topk} N={N2} K={K2} cfg{cfg2}", out2.cpu(), ref2)
    return ok, (w1, w2, xq, xs, hq, hs, sorted_ids, expert_ids, ntpp, numel, tw)


def bench(E, M, topk, N1, K1, N2, K2, cfg1, cfg2, iters=200):
    ok, (w1, w2, xq, xs, hq, hs, sid, eid, ntpp, numel, tw) = run_case(E, M, topk, N1, K1, N2, K2, 7, cfg1, cfg2)
    dev = "cuda"
    sid, eid, ntpp, tw = sid.to(dev), eid.to(dev), ntpp.to(dev), tw.to(dev)
    o1 = torch.empty(numel, N1, dtype=torch.bfloat16, device=dev)
    o2 = torch.empty(numel, N2, dtype=torch.bfloat16, device=dev)
    for _ in range(10):
        K.moe_gemm(xq, xs, w1, o1, sid, eid, ntpp, numel, topk, None, *cfg1)
        K.moe_gemm(hq, hs, w2, o2, sid, eid, ntpp, numel, 1, tw, *cfg2)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        K.moe_gemm(xq, xs, w1, o1, sid, eid, ntpp, numel, topk, None, *cfg1)
    torch.cuda.synchronize()
    t1 = (time.perf_counter() - t) / iters * 1e6
    t = time.perf_counter()
    for _ in range(iters):
        K.moe_gemm(hq, hs, w2, o2, sid, eid, ntpp, numel, 1, tw, *cfg2)
    torch.cuda.synchronize()
    t2 = (time.perf_counter() - t) / iters * 1e6
    nexp = int((eid.cpu() >= 0).sum()) and len(set(eid.cpu().tolist()) - {-1})
    b1 = nexp * N1 * K1 / 2 + nexp * N1 * K1 / 32
    b2 = nexp * N2 * K2 / 2 + nexp * N2 * K2 / 32
    print(f"  bench M={M} experts touched={nexp}: gate_up {t1:.1f} us ({b1/t1/1e3:.0f} GB/s), "
          f"down {t2:.1f} us ({b2/t2/1e3:.0f} GB/s)")
    return ok


if __name__ == "__main__":
    torch.manual_seed(0)
    allok = True
    # Flash-Next per-rank at TP2: w13 N=2*320=640 K=2560, w2 N=2560 K=320; 512 experts top-10.
    for (E, M, topk) in [(8, 1, 2), (8, 5, 3), (32, 4, 10), (64, 16, 10)]:
        allok &= run_case(E, M, topk, 640, 2560, 2560, 320, seed=E + M)[0]
    # odd shapes: N=48 tail, K=320 with SK=5/10
    allok &= run_case(8, 3, 2, 48, 1024, 2560, 320, 3, cfg1=(4, 4, 1), cfg2=(2, 5, 2))[0]
    allok &= run_case(8, 3, 2, 640, 2560, 2560, 320, 4, cfg1=(4, 2, 4), cfg2=(1, 10, 4))[0]
    print("-- perf (Flash-Next TP2 shapes, E=512)")
    for M in (1, 4, 16):
        allok &= bench(512, M, 10, 640, 2560, 2560, 320, (2, 4, 2), (4, 2, 1))
    print("ALL OK" if allok else "FAILURES")
    sys.exit(0 if allok else 1)
