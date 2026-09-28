"""Our fused hyper-connection input mix (router-GEMM silu epilogue + kernels/r9k_hc.hip r9k_hc_up_mix, hc.py) vs the
stock composition: F.linear (merged down + injection) -> hc_silu -> F.linear (up) -> hc_gate_mix.

Single GPU:  python3 tests/test_hc_mix_r9k.py            (BENCH=0 skips the timing table)

Flash-Next shapes: HC=4 x HD=2560, lora rank 320, merged down rows 320 + 4 + 12 pad. The down+silu output must match
stock except where the GEMM's summation order flips a bf16 rounding (counted, rare); the mixed block input is checked
against an fp64 reference of the stock semantics from the SAME silu output (ours within 1.5x of stock's error). Rows
1 / 3 / 4 / 8, a strided xn; rows 16 must take stock's path inside the ops.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vllm.models.qwen4_exp.amd.ops.hc import hc_gate_mix, hc_silu  # noqa: E402

from r9700_vllm import hc as R  # noqa: E402

dev = torch.device("cuda")
HC, HD, LR, PAD = 4, 2560, 320, 12
DIM = HC * HD
NDOWN = LR + HC + PAD
g = torch.Generator(device="cpu").manual_seed(0)
bad = 0


def rnd(*shape, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(torch.bfloat16).to(dev)


def ref_mix(xn, lora, w_up):
    """fp64 stock semantics of up GEMM (bf16-rounded) -> sigmoid -> gated mean."""
    gate = (lora.double() @ w_up.double().t()).to(torch.bfloat16).double().view(-1, HC, HD)
    return (torch.sigmoid(gate) * xn.double().view(-1, HC, HD)).sum(1) / HC


def err(a, ref):
    return ((a.double() - ref).abs() / ref.abs().clamp_min(1e-2)).max().item()


def close_to_stock(o, s, ref):
    """ours vs stock against the fp64 reference: mean relative error within 1.2x of stock's, and elements beyond
    2x stock's max error (a bf16 ulp flipped by the up GEMM's summation order before the sigmoid) rarer than 1e-3."""
    rel_o = (o.double() - ref).abs() / ref.abs().clamp_min(1e-2)
    rel_s = (s.double() - ref).abs() / ref.abs().clamp_min(1e-2)
    return rel_o.mean().item() <= 1.2 * rel_s.mean().item() + 1e-6 and \
        (rel_o > 2 * rel_s.max()).double().mean().item() <= 1e-3


def check(name, M, strided=False):
    global bad
    xn = rnd(M, DIM + 64)[:, :DIM] if strided else rnd(M, DIM)
    w_down = rnd(NDOWN, DIM, scale=0.02)
    w_up = rnd(DIM, LR, scale=0.06)
    with torch.no_grad():
        s_buf = F.linear(xn, w_down)
        s_lora = hc_silu(s_buf[:, :LR], HC)
        s_gate = F.linear(s_lora, w_up)
        s_mix = hc_gate_mix(xn, s_gate, HC)
        o_buf = torch.ops.r9700.hc_down_silu(xn, w_down, LR, HC)
        o_lora = o_buf[:, :LR]
        o_mix = torch.ops.r9700.hc_up_mix(o_lora, w_up, xn, HC)
        # the mix kernel on stock's silu output isolates the up GEMM + mix from the down GEMM's flips
        o_mix_s = torch.ops.r9700.hc_up_mix(s_lora, w_up, xn, HC)
    torch.cuda.synchronize()
    lora_flips = (o_lora != s_lora).sum().item()
    inj_flips = (o_buf[:, LR:LR + HC] != s_buf[:, LR:LR + HC]).sum().item()
    ref = ref_mix(xn, s_lora, w_up)
    e_s, e_o = err(s_mix, ref), err(o_mix_s, ref)
    ref_full = ref_mix(xn, o_lora, w_up)
    e_full = err(o_mix, ref_full)
    ok = (o_buf.shape == s_buf.shape and o_buf.dtype == torch.bfloat16 and o_mix.shape == (M, HD)
          and torch.isfinite(o_mix).all() and lora_flips <= max(4, M * LR * 5e-3) and inj_flips <= 1
          and close_to_stock(o_mix_s, s_mix, ref) and close_to_stock(o_mix, s_mix, ref_full))
    bad += not ok
    print(f"  {name:<26} lora flips {lora_flips}/{M * LR}  inj flips {inj_flips}  mix err ours {e_o:.2e} (full path "
          f"{e_full:.2e}) stock {e_s:.2e}  {'ok' if ok else 'FAIL'}")


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


def bench(M):
    xn = rnd(M, DIM)
    wd = [rnd(NDOWN, DIM, scale=0.02) for _ in range(6)]       # rotate the 6.5 MB weights out of the caches
    wu = [rnd(DIM, LR, scale=0.06) for _ in range(6)]
    i = [0]

    def nxt():
        i[0] = (i[0] + 1) % 6
        return wd[i[0]], wu[i[0]]

    def stock():
        a, b = nxt()
        buf = F.linear(xn, a)
        lora = hc_silu(buf[:, :LR], HC)
        return hc_gate_mix(xn, F.linear(lora, b), HC)

    def ours():
        a, b = nxt()
        buf = torch.ops.r9700.hc_down_silu(xn, a, LR, HC)
        return torch.ops.r9700.hc_up_mix(buf[:, :LR], b, xn, HC)

    def down_s():
        return F.linear(xn, nxt()[0])

    def down_o():
        return torch.ops.r9700.hc_down_silu(xn, nxt()[0], LR, HC)
    lora = rnd(M, LR)

    def up_s():
        return F.linear(lora, nxt()[1])

    def up_o():
        return torch.ops.r9700.hc_up_mix(lora, nxt()[1], xn, HC)

    def up_o2():
        return R.up_mix(lora, nxt()[1], xn, HC, lpr=2)

    def up_o4():
        return R.up_mix(lora, nxt()[1], xn, HC, lpr=4)
    from r9700_vllm import router as RT
    with torch.no_grad():
        rows = [("stock mix (4 launches)", stock), ("ours mix (2 launches)", ours), ("stock down GEMM", down_s),
                ("ours down+silu", down_o), ("stock up GEMM", up_s), ("ours up+mix", up_o),
                ("ours up+mix 2 lanes/row", up_o2), ("ours up+mix 4 lanes/row", up_o4)]
        for sp in (1, 2, 4, 8):
            rows.append((f"ours down+silu split {sp}", lambda sp=sp: RT.router_gemm(xn, nxt()[0], True, sp, LR, 4.0, 8)))
        for name, f in rows:
            print(f"  rows {M:<3} {name:<24} {graph_us(f):7.1f} us (graph replay)")


def main():
    if not R.mix_available():
        print("libr9k.so has no r9k_hc_up_mix / r9k_router_gemm: rebuild kernels/")
        sys.exit(1)
    R.register()
    global bad
    for M in (1, 3, 4, 8):
        check(f"rows {M}", M)
    check("rows 5 strided xn", 5, strided=True)
    xn, w_down, w_up = rnd(16, DIM), rnd(NDOWN, DIM, scale=0.02), rnd(DIM, LR, scale=0.06)
    with torch.no_grad():
        buf = torch.ops.r9700.hc_down_silu(xn, w_down, LR, HC)
        s_buf = F.linear(xn, w_down)
        ok = torch.equal(buf[:, LR:], s_buf[:, LR:]) and torch.equal(buf[:, :LR], hc_silu(s_buf[:, :LR], HC)) \
            and torch.equal(torch.ops.r9700.hc_up_mix(buf[:, :LR], w_up, xn, HC),
                            hc_gate_mix(xn, F.linear(buf[:, :LR], w_up), HC))
    bad += not ok
    print(f"  {'rows 16: stock path inside the ops':<26} {'ok' if ok else 'FAIL'}")
    print("correctness:", "PASS" if bad == 0 else f"FAIL ({bad})")
    if os.environ.get("BENCH", "1") == "1":
        bench(4)
        bench(1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
