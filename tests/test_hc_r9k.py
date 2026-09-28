"""Our hyper-connection kernels (kernels/r9k_hc.hip, hc.py) vs vLLM's Triton hc_gate_mix / hc_combine_norm.

Single GPU:  python3 tests/test_hc_r9k.py            (BENCH=0 skips the timing table)

Both against an fp64 reference of the stock semantics; ours must be within 1.5x of stock's error (bf16 outputs, so
both are dominated by the final rounding) and the combine output `out` must match stock bit-for-bit except where an
fp32 fma-vs-mul-add difference flips a bf16 rounding (counted, must be rare). Shapes: Flash-Next HC=4 x 2560, rows
1 / 5 / 333 / 4096, shared and per-stream norm weights, a strided residual; then timing at 4096 rows.
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vllm.models.qwen4_exp.amd.ops.hc import hc_combine_norm, hc_gate_mix  # noqa: E402

from r9700_vllm import hc as R  # noqa: E402

dev = torch.device("cuda")
HC, HD = 4, 2560
EPS = 1e-6
g = torch.Generator(device="cpu").manual_seed(0)
bad = 0


def rnd(*shape, scale=1.0):
    return (torch.randn(shape, generator=g) * scale).to(torch.bfloat16).to(dev)


def ref_gate_mix(x, gate):
    x, gt = x.double(), gate.double()
    N = x.shape[0]
    return (torch.sigmoid(gt.view(N, HC, HD)) * x.view(N, HC, HD)).sum(1) / HC


def ref_combine_norm(res, blk, inj, w):
    N = res.shape[0]
    injf = 2 * torch.sigmoid(inj.double() / HC)                                  # [N, HC]
    out = (res.double().view(N, HC, HD) + blk.double().view(N, 1, HD) * injf.view(N, HC, 1)).to(torch.bfloat16).double()
    rrms = torch.rsqrt((out * out).mean(-1, keepdim=True) + EPS)
    wv = w.double().view(1, 1, HD) if w.numel() == HD else w.double().view(1, HC, HD)
    y = out * rrms
    y = y + y * wv
    return out.view(N, HC * HD), y.view(N, HC * HD)


def err(a, ref):
    return ((a.double() - ref).abs() / ref.abs().clamp_min(1e-2)).max().item()


def check(name, N, shared=True, strided=False):
    global bad
    x = rnd(N, HC * HD)
    if strided:
        big = rnd(N, HC * HD + 64)
        x = big[:, :HC * HD]
    gate = rnd(N, HC * HD, scale=2.0)
    blk = rnd(N, HD)
    inj = rnd(N, HC, scale=3.0)
    w = rnd(HD if shared else HC * HD, scale=0.5).contiguous()
    r_gm = ref_gate_mix(x, gate)
    s_gm = hc_gate_mix(x, gate, HC)
    o_gm = R.gate_mix(x, gate, HC)
    r_out, r_y = ref_combine_norm(x, blk, inj, w)
    s_out, s_y = hc_combine_norm(x, blk, inj, w, EPS, HC)
    o_out, o_y = R.combine_norm(x, blk, inj, w, EPS, HC)
    torch.cuda.synchronize()
    e = dict(gm_s=err(s_gm, r_gm), gm_o=err(o_gm, r_gm), out_s=err(s_out, r_out), out_o=err(o_out, r_out),
             y_s=err(s_y, r_y), y_o=err(o_y, r_y))
    flips = (o_out != s_out).sum().item()
    ok = e["gm_o"] <= 1.5 * e["gm_s"] + 1e-6 and e["out_o"] <= 1.5 * e["out_s"] + 1e-6 and e["y_o"] <= 1.5 * e["y_s"] + 1e-6 \
        and flips <= o_out.numel() * 1e-4 and torch.isfinite(o_y).all() and o_gm.shape == s_gm.shape
    bad += not ok
    print(f"  {name:<34} gate_mix err ours {e['gm_o']:.1e} stock {e['gm_s']:.1e} | out {e['out_o']:.1e} / {e['out_s']:.1e} "
          f"(bf16 flips {flips}) | y {e['y_o']:.1e} / {e['y_s']:.1e}  {'ok' if ok else 'FAIL'}")


def bench(N):
    x, gate, blk, inj, w = rnd(N, HC * HD), rnd(N, HC * HD), rnd(N, HD), rnd(N, HC), rnd(HD).contiguous()
    for name, f in (("gate_mix stock", lambda: hc_gate_mix(x, gate, HC)), ("gate_mix ours", lambda: R.gate_mix(x, gate, HC)),
                    ("combine_norm stock", lambda: hc_combine_norm(x, blk, inj, w, EPS, HC)),
                    ("combine_norm ours", lambda: R.combine_norm(x, blk, inj, w, EPS, HC))):
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(20):
            f()
        torch.cuda.synchronize()
        us = (time.perf_counter() - t) / 20 * 1e6
        mb = (N * HC * HD * 2 * 2 + N * HD * 2) / 1e6 if "gate" in name else (N * HC * HD * 2 * 3 + N * HD * 2) / 1e6
        print(f"  rows {N:<5} {name:<20} {us:8.0f} us  {mb / us * 1e3:5.0f} GB/s")


def main():
    if not R.available():
        print("libr9k.so has no r9k_hc_combine_norm: rebuild kernels/")
        sys.exit(1)
    # the registered custom ops (what the compiled model graph calls) must build and agree with the direct calls
    R.register()
    x, gate, blk, inj, w = rnd(64, HC * HD), rnd(64, HC * HD), rnd(64, HD), rnd(64, HC), rnd(HD).contiguous()
    global bad
    bad += not torch.equal(torch.ops.r9700.hc_gate_mix(x, gate, HC), R.gate_mix(x, gate, HC))
    o1, y1 = torch.ops.r9700.hc_combine_norm(x, blk, inj, w, EPS, HC)
    o2, y2 = R.combine_norm(x, blk, inj, w, EPS, HC)
    bad += not (torch.equal(o1, o2) and torch.equal(y1, y2))
    print(f"  {'torch.ops.r9700.hc_* registered and equal to direct calls':<34} {'ok' if bad == 0 else 'FAIL'}")
    check("rows 1 shared w", 1)
    check("rows 5 per-stream w", 5, shared=False)
    check("rows 333 shared w", 333)
    check("rows 4096 shared w", 4096)
    check("rows 4096 per-stream w", 4096, shared=False)
    check("rows 1024 strided residual", 1024, strided=True)
    print("correctness:", "PASS" if bad == 0 else f"FAIL ({bad})")
    if os.environ.get("BENCH", "1") == "1":
        bench(4096)
        bench(64)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
