"""Our fused shared expert (r9700_vllm/moe/shared.py: quant+gate, gate_up GEMM, silu_mul_quant, down GEMM with the
sigmoid gate folded) vs the stock composition on the same dense MXFP4 kernels: mxfp4_linear -> silu_and_mul ->
mxfp4_linear -> sigmoid(F.linear(x, wg)) * out.

Single GPU:  python3 tests/test_shared_expert_r9k.py            (BENCH=0 skips the timing table)

Flash-Next shared expert at TP=4 (gate_up 320x2560, down 2560x160) and TP=2 (640x2560, 2560x320), M = 1 / 4 / 8 /
16 / 64 / 200 / 1024 (decode, MT and prefill / A-tiled configs). The gate values must equal stock's bit for bit
(same rounding points; only the dot's fp32 order differs -> flips counted). The output differs from stock by its
single rounding, bf16(acc * g) vs bf16(bf16(acc) * g), and by the activation's fp8 codes (silu_mul_quant_fp8 vs
silu_and_mul + quant_rows_fp8: a code apart on some elements, as the routed MoE already runs): ours must be at
least as close (within 5%) to the fp64 reference -- stock's semantics from the same fp8 input quant and the
UNquantized activation -- as stock is, and within 3% of stock's output in relative norm.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from r9700_vllm.kernels import moe as K  # noqa: E402
from r9700_vllm.moe import shared as S  # noqa: E402
from r9700_vllm.ops import _mxfp4_linear  # noqa: E402

dev = torch.device("cuda")
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
g = torch.Generator(device="cpu").manual_seed(0)
bad = 0


def make_weight(N, Kd, max_d=8):
    """random MXFP4 weight -> (wq, wsr as the dense linear stores them, dequantized fp32 [N, K])"""
    nb = Kd // K.GROUP
    ref = torch.randint(120, 128, (1, N, 1), generator=g, dtype=torch.int32)
    drop = torch.randint(0, max_d + 1, (1, N, nb), generator=g, dtype=torch.int32)
    e8m0 = (ref - drop).clamp(0, 254)
    codes = torch.randint(0, 16, (1, N, Kd), generator=g, dtype=torch.uint8)
    mag = E2M1[(codes & 0x7).long()]
    sign = torch.where((codes & 0x8) > 0, -1.0, 1.0)
    Wf = (mag * sign).reshape(1, N, nb, K.GROUP) * torch.exp2(e8m0.float() - 127.0).unsqueeze(-1)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()
    wq = K.permute_fragments(packed.to(dev))
    wsr = K.pack_scales(e8m0.to(torch.uint8).to(dev))
    return wq, wsr, Wf.reshape(N, Kd).to(dev)


def stock(x, w1, w2, wg, N1, K1, N2, K2):
    gu = _mxfp4_linear(x, w1[0], w1[1], N1, K1, False)
    act = torch.empty((x.shape[0], N1 // 2), dtype=torch.bfloat16, device=dev)
    torch.ops._C.silu_and_mul(act, gu)
    out = _mxfp4_linear(act, w2[0], w2[1], N2, K2, False)
    return torch.sigmoid(F.linear(x, wg.view(1, -1))) * out


def ours(x, w1, w2, wg, N1, K1, N2, K2):
    return torch.ops.r9700.shared_expert(x, w1[0], w1[1], w2[0], w2[1], wg, N1, K1, N2, K2, False, False)


def ref64(x, w1d, w2d, wg):
    """fp64 of stock's semantics from the same fp8-quantized activations (per-row scale, e4m3)."""
    def quant(t):
        q, s = K.quant_rows_fp8(t)
        return q.double() * s.double()[:, None]
    gu = (quant(x) @ w1d.double().T).to(torch.bfloat16).double()
    gate, up = gu.chunk(2, dim=-1)
    act = F.silu(gate) * up                    # unquantized: both paths' fp8 activation codes count as error
    dn = act @ w2d.double().T
    gate_v = torch.sigmoid((x.double() @ wg.double()).to(torch.bfloat16).double()).to(torch.bfloat16).double()
    return dn * gate_v[:, None]


def check(M, N1, K1, N2, K2):
    global bad
    w1 = make_weight(N1, K1)
    w2 = make_weight(N2, K2)
    wg = (torch.randn(K1, generator=g) * 0.02).to(torch.bfloat16).to(dev)
    x = torch.randn((M, K1), generator=g).to(torch.bfloat16).to(dev)
    with torch.no_grad():
        s = stock(x, w1[:2], w2[:2], wg, N1, K1, N2, K2)
        o = ours(x, w1[:2], w2[:2], wg, N1, K1, N2, K2)
        # the gate alone: stock's value vs the one our quant kernel produced
        gs = torch.sigmoid(F.linear(x, wg.view(1, -1))).view(-1)
        _, _, gk = K.quant_rows_fp8_gate(x, wg)
        r = ref64(x, w1[2], w2[2], wg)
    torch.cuda.synchronize()
    gflips = (gk != gs.float()).sum().item()
    rel = ((o.float() - s.float()).norm() / s.float().norm().clamp_min(1e-9)).item()
    within = rel <= 3e-2
    eo = (o.double() - r).abs().mean().item()
    es = (s.double() - r).abs().mean().item()
    nf = (o != s).sum().item()
    ok = within and bool(torch.isfinite(o).all()) and eo <= es * 1.05 + 1e-9 and gflips <= max(1, M // 64)
    bad += not ok
    print(f"  M {M:<5} {N1}x{K1} / {N2}x{K2}   vs stock: {nf}/{o.numel()} differ, rel {rel:.1e}  "
          f"err ours {eo:.3e} stock {es:.3e}  gate flips {gflips}/{M}  {'ok' if ok else 'FAIL'}")


def graph_us(fn, reps=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, stream=st):
            for _ in range(reps):
                fn()
    torch.cuda.synchronize()
    gr.replay()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(5):
        gr.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / 5 / reps * 1e6


def bench(M, N1, K1, N2, K2):
    w1 = make_weight(N1, K1)
    w2 = make_weight(N2, K2)
    wg = (torch.randn(K1, generator=g) * 0.02).to(torch.bfloat16).to(dev)
    x = torch.randn((M, K1), generator=g).to(torch.bfloat16).to(dev)
    with torch.no_grad():
        for name, f in (("stock (8 launches)", lambda: stock(x, w1[:2], w2[:2], wg, N1, K1, N2, K2)),
                        ("ours (4 launches)", lambda: ours(x, w1[:2], w2[:2], wg, N1, K1, N2, K2))):
            print(f"  M {M:<5} {N1}x{K1}  {name:<20} {graph_us(f):7.1f} us (graph replay)")


def main():
    if not S.available():
        print("libr9k.so has no r9k_quant_rows_fp8_gate: rebuild kernels/")
        sys.exit(1)
    S.register()
    for N1, K1, N2, K2 in ((320, 2560, 2560, 160), (640, 2560, 2560, 320)):
        for M in (1, 4, 8, 16, 64, 200, 1024):
            check(M, N1, K1, N2, K2)
    print(f"correctness: {'PASS' if bad == 0 else 'FAIL'}")
    if os.environ.get("BENCH", "1") != "0":
        for M in (1, 4, 16):
            bench(M, 320, 2560, 2560, 160)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
