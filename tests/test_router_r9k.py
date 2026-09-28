"""Our MoE router GEMM (kernels/r9k_router.hip, router.py) vs stock's bf16 F.linear (hipBLASLt on ROCm; what
vLLM's MoE runner runs for Qwen3-Next's ReplicatedLinear gate) and torch.mm's bf16 -> fp32 path.

Single GPU:  python3 tests/test_router_r9k.py            (BENCH=0 skips the timing table)

Against an fp64 reference: ours (bf16 out) must be within 1.5x of stock's error and differ from stock only where
summation order flips a bf16 rounding (counted, rare); the fp32 variant within 2x of torch.mm's. Top-10 expert sets
must agree with the reference except on near-ties. Shapes: Flash-Next K=2560, E=512 at rows 1 / 2 / 3 / 4 / 8 / 13
/ 16 / 32, E=500 (partial block), strided x, each split; then timing per split.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from r9700_vllm import router as R  # noqa: E402

dev = torch.device("cuda")
g = torch.Generator(device="cpu").manual_seed(0)
bad = 0


def rnd(*shape, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(torch.bfloat16).to(dev)


def topk_ok(ref, o, k=10):
    """Every expert ours picks but the fp64 reference does not must be a near-tie with the reference's k-th."""
    tr = ref.topk(k, dim=-1).indices
    to = o.float().topk(k, dim=-1).indices
    agree = 0
    for m in range(ref.shape[0]):
        a, b = set(tr[m].tolist()), set(to[m].tolist())
        agree += len(a & b)
        kth = ref[m].topk(k).values[-1].item()
        for j in a ^ b:
            if abs(ref[m, j].item() - kth) > 4e-3 * max(1.0, abs(kth)):     # bf16 logits: one ulp at |x|~1
                return False, agree / (ref.shape[0] * k)
    return True, agree / (ref.shape[0] * k)


def check(name, M, E=512, K=2560, strided=False, split=2):
    global bad
    x = rnd(M, K)
    if strided:
        x = rnd(M, K + 64)[:, :K]
    w = rnd(E, K, scale=0.05)
    ref = x.double() @ w.double().t()
    s_bf = F.linear(x, w)
    t_32 = torch.mm(x, w.t(), out_dtype=torch.float32)
    o_bf = R.router_gemm(x, w, True, split=split)
    o_32 = R.router_gemm(x, w, False, split=split)
    torch.cuda.synchronize()
    e = dict(s=(s_bf.double() - ref).abs().max().item(), o=(o_bf.double() - ref).abs().max().item(),
             t=(t_32.double() - ref).abs().max().item(), o32=(o_32.double() - ref).abs().max().item())
    flips = (o_bf != s_bf).sum().item()
    tk, agree = topk_ok(ref, o_bf)
    ok = (e["o"] <= 1.5 * e["s"] + 1e-6 and e["o32"] <= 2.0 * e["t"] + 1e-7 and o_bf.dtype == torch.bfloat16
          and o_32.dtype == torch.float32 and o_bf.shape == (M, E) and torch.isfinite(o_bf).all()
          and flips <= max(2, o_bf.numel() * 2e-3) and tk)
    bad += not ok
    print(f"  {name:<28} max|err| bf16 ours {e['o']:.2e} stock {e['s']:.2e} (flips {flips}) | fp32 ours {e['o32']:.2e}"
          f" torch {e['t']:.2e} | top-10 agreement {agree:.4f}  {'ok' if ok else 'FAIL'}")


def graph_us(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, stream=s):
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


def bench(M, E=512, K=2560):
    ws = [rnd(E, K) for _ in range(16)]          # rotate weights so the 2.6 MB gate is not served from L2
    x = rnd(M, K)
    i = [0]

    def nxt():
        i[0] = (i[0] + 1) % len(ws)
        return ws[i[0]]
    rows = [("stock F.linear bf16", lambda: F.linear(x, nxt())),
            ("torch.mm fp32-out", lambda: torch.mm(x, nxt().t(), out_dtype=torch.float32))]
    for sp in (1, 2, 4, 8):
        rows.append((f"ours bf16 split {sp}", lambda sp=sp: R.router_gemm(x, nxt(), True, split=sp)))
    for name, f in rows:
        us = graph_us(f)
        print(f"  M {M:<3} {name:<22} {us:7.1f} us  {E * K * 2 / us / 1e3:6.0f} GB/s")


def main():
    if not R.available():
        print("libr9k.so has no r9k_router_gemm: rebuild kernels/")
        sys.exit(1)
    R.register()
    x, w = rnd(4, 2560), rnd(512, 2560)
    global bad
    bad += not torch.equal(torch.ops.r9700.router_gemm(x, w, True), R.router_gemm(x, w, True))
    big = rnd(64, 2560)                          # above MAX_M: hipBLASLt's fp32-out GEMM rounded to bf16
    fb = torch.ops.r9700.router_gemm(big, w, True)
    bad += not (torch.equal(fb, torch.mm(big, w.t(), out_dtype=torch.float32).to(torch.bfloat16))
                and (fb != F.linear(big, w)).float().mean().item() < 2e-3)
    print(f"  {'torch.ops.r9700.router_gemm registered, equal to the direct call, M > MAX_M = fp32 GEMM':<28} "
          f"{'ok' if bad == 0 else 'FAIL'}")
    for M in (1, 2, 3, 4, 8, 13, 16, 32):
        check(f"rows {M}", M)
    check("rows 4 E=500 (partial block)", 4, E=500)
    check("rows 5 strided x", 5, strided=True)
    for sp in (1, 4, 8):
        check(f"rows 4 split {sp}", 4, split=sp)
    print("correctness:", "PASS" if bad == 0 else f"FAIL ({bad})")
    if os.environ.get("BENCH", "1") == "1":
        for M in (1, 4, 16, 32):
            bench(M)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
