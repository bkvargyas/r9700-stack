"""A-tiled prefill path (fragment-tiled activations, r9k_quant_rows_fp8_tiled + r9k_moe_4bit_prefill_at):
  (a) the tiled quantizer's bytes and scales equal the row-major quantizer's, re-laid out (bit-exact);
  (b) every A-tiled cfg on a mid shape (ragged M and N tails), folded AND fp32-per-group (fold=False), against the
      dequantized reference AND bit-identical to the LDS-A prefill kernel of the same variant (same WMMA accumulation
      order, so the outputs must match exactly);
  (c) the served 27B shapes on the cfg pick_cfg selects for BOTH fold settings (pick_cfg(..., fold=False) must return
      an ("A", cfg) too -- a knob that looks set while the LDS-A kernel runs is the failure this guards), plus every
      tuned.json "mxfp4_at" row;
  (d) the public wrappers the model calls: ops.mxfp4_linear (torch.ops.r9700.mxfp4_linear) at decode and prefill M with
      fold on/off (A-tiled at prefill M either way), and R9700Mxfp4LinearKernel.apply_weights on a fake layer with
      fold on/off (the served path, needs vLLM).

Run inside the ROCm image: PYTHONPATH=/repo R9K_LIB=/repo/r9700_vllm/kernels/libr9k.so python3 tests/test_atiled_4bit.py
"""
import dataclasses
import json
import os
import sys

os.environ.setdefault("R9K_ATILED", "1")
os.environ.setdefault("R9K_FOLD", "1")

import torch

from r9700_vllm.kernels import moe as K
from r9700_vllm.quant.nvfp4 import quantize_mxfp4_search

ok = True
g = torch.Generator(device="cuda").manual_seed(0)
LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")


def mx_deq(p, e):
    N, Kh = p.shape
    c = torch.stack((p & 15, p >> 4), -1).reshape(N, Kh * 2).long()
    return (LUT[c].reshape(N, -1, 32) * torch.exp2(e.float() - 127).unsqueeze(-1)).reshape(N, Kh * 2)


def check(name, got, ref, tol=5e-3):
    global ok
    r = ((got.float() - ref).norm() / ref.norm()).item()
    good = r < tol and torch.isfinite(got.float()).all().item()
    ok &= good
    print(f"  {name:80s} rel {r:.2e} {'ok' if good else '<-- FAIL'}", flush=True)
    return good


def check_eq(name, a, b):
    global ok
    good = a.shape == b.shape and torch.equal(a.view(torch.uint8) if a.dtype != torch.float32 else a,
                                              b.view(torch.uint8) if b.dtype != torch.float32 else b)
    ok &= good
    print(f"  {name:80s} {'bit-identical' if good else '<-- FAIL (differs)'}", flush=True)
    return good


def identity(M, blk):
    mpad = (M + blk - 1) // blk * blk
    return (torch.arange(mpad, dtype=torch.int32, device="cuda"), torch.zeros(mpad // blk, dtype=torch.int32, device="cuda"),
            torch.full((1,), mpad, dtype=torch.int32, device="cuda"))


def make_w(N, Kd, fold=True):
    w = torch.randn(N, Kd, device="cuda", generator=g) * 0.02
    p, e = quantize_mxfp4_search(w)
    W = dataclasses.replace(K.prepare_mxfp4_weights(p[None], e[None]), fold=fold)
    return W, mx_deq(p, e)


def both(W):
    """(fold, W) for the folded and the fp32-per-group variant of the same weights."""
    return ((True, dataclasses.replace(W, fold=True)), (False, dataclasses.replace(W, fold=False)))


def run_at(W, qt, s, M, cfg):
    out = torch.full((M, W.N), float("nan"), dtype=torch.bfloat16, device="cuda")
    K.moe_gemm(qt, s, W, out, *identity(M, K.atiled_block(cfg)), M, 1, None, num_experts=1, prefill=cfg, a_tiled=True)
    return out


def run_lds(W, q, s, M, cfg):          # the LDS-A prefill kernel, folded or fp32-per-group per W.fold
    out = torch.full((M, W.N), float("nan"), dtype=torch.bfloat16, device="cuda")
    K.moe_gemm(q, s, W, out, *identity(M, K.prefill_block(cfg)), M, 1, None, num_experts=1, prefill=cfg)
    return out


assert K.atiled_supported(), "libr9k without r9k_moe_4bit_prefill_at"
assert K.ATILED, "R9K_ATILED must be on for this gate"

# ---- (a) tiled quantizer == row-major quantizer re-laid out
for M, Kd in ((256, 1024), (300, 1024), (1000, 5120), (4096, 8704), (17, 3072)):
    x = torch.randn(M, Kd, device="cuda", generator=g).to(torch.bfloat16) * 3
    q, s = K.quant_rows_fp8(x)
    qt, st = K.quant_rows_fp8(x, tiled=True)
    ref = K.tile_fp8_ref(q)
    # rows M..16*ceil(M/16) of the tiled buffer are unwritten: compare only the fragments' written lanes
    mt = (M + 15) // 16                          # tiled byte order is [mt][ks][half][row16][8]: lane row = mt*16 + row16
    mask = (torch.arange(mt * 16, device="cuda").reshape(mt, 1, 1, 16, 1) < M).expand(mt, Kd // 16, 2, 16, 8) \
        .reshape(mt * 16, Kd)
    good = torch.equal(qt.view(torch.uint8)[mask], ref.view(torch.uint8)[mask]) and torch.equal(s, st)
    ok &= good
    print(f"  {'tiled quant M=%d K=%d bytes + scales' % (M, Kd):80s} {'bit-identical' if good else '<-- FAIL'}", flush=True)

# ---- (b) every A-tiled cfg, both variants: vs reference and bit-identical to the LDS-A kernel of the same variant
# (cfg 12 / 16 = same tiles)
N, Kd = 1040, 1024                                  # 16-column tail for every BN
W0, wd = make_w(N, Kd)
for M in (256, 300, 1024):
    x = torch.randn(M, Kd, device="cuda", generator=g).to(torch.bfloat16)
    q, s = K.quant_rows_fp8(x)
    qt, _ = K.quant_rows_fp8(x, tiled=True)
    ref = (q.float() * s[:, None]) @ wd.T
    for fold, W in both(W0):
        base = run_lds(W, q, s, M, 12)
        fs = "fold" if fold else "nofold"
        for cfg in sorted(K.ATILED_TILES):
            out = run_at(W, qt, s, M, cfg)
            check(f"atiled 1040x1024 M={M} {fs} cfg={cfg} {K.ATILED_TILES[cfg]}", out, ref)
            check_eq(f"atiled 1040x1024 M={M} {fs} cfg={cfg} == {fs} LDS-A cfg 12", out, base)
    # the two variants differ only in where the block exponent is applied: on ordinary weights (every block within
    # e4m3's reach) the folded product is exact, so both must agree to fp8-WMMA accumulation tolerance
    check(f"atiled 1040x1024 M={M} fold vs nofold (cfg 1)", run_at(W0, qt, s, M, 1),
          run_at(dataclasses.replace(W0, fold=False), qt, s, M, 1).float(), tol=1e-3)
del W0, wd

# ---- (c) served shapes on the picked cfg for both fold settings, and every tuned "mxfp4_at" row
T = json.load(open(os.path.join(os.path.dirname(K.__file__), "tuned.json")))
for (N, Kd) in ((17408, 5120), (5120, 8704), (8240, 5120), (7168, 5120), (5120, 3072)):
    W0, wd = make_w(N, Kd)
    cases = {(M, fold): K.pick_cfg(N, Kd, 32, M=M, kind="mxfp4", fold=fold) for M in (512, 2048) for fold in (True, False)}
    for Ms, cfg in T.get("mxfp4_at", {}).get(f"{N},{Kd}", {}).items():
        for fold in (True, False):
            cases[(int(Ms), fold)] = tuple(cfg)
    for (M, fold), cfg in sorted(cases.items(), key=lambda kv: (kv[0][0], not kv[0][1])):
        fs = "fold" if fold else "nofold"
        good = K.is_atiled_cfg(cfg)
        ok &= good
        print(f"  {'pick_cfg(%d,%d,M=%d,%s) -> %s' % (N, Kd, M, fs, cfg):80s} {'ok' if good else '<-- FAIL (not A-tiled)'}")
        if not good:
            continue
        W = dataclasses.replace(W0, fold=fold)
        x = torch.randn(M, Kd, device="cuda", generator=g).to(torch.bfloat16)
        q, s = K.quant_rows_fp8(x)
        qt, _ = K.quant_rows_fp8(x, tiled=True)
        out = run_at(W, qt, s, M, cfg[1])
        check(f"atiled dense {N}x{Kd} M={M} {fs} cfg={cfg}", out, (q.float() * s[:, None]) @ wd.T)
        check_eq(f"atiled dense {N}x{Kd} M={M} {fs} cfg={cfg} == {fs} LDS-A cfg 12", out, run_lds(W, q, s, M, 12))
    del W0, wd

# ---- (d) the public wrappers, the way the model calls them
try:
    from r9700_vllm import ops
    N, Kd = 1024, 2048
    W, wd = make_w(N, Kd)
    for M in (8, 64, 256, 1000, 2048):
        x = torch.randn(M, Kd, device="cuda", generator=g).to(torch.bfloat16)
        q, s = K.quant_rows_fp8(x)
        ref = (q.float() * s[:, None]) @ wd.T
        for fold in (False, True):
            out = ops.mxfp4_linear(x, W.wq[0], W.wsr[0], N, Kd, fold)
            cfg = K.pick_cfg(N, Kd, M=M, kind="mxfp4", fold=fold)
            check(f"ops.mxfp4_linear {N}x{Kd} M={M} fold={fold} -> {cfg}", out, ref)
        for fold in (True, False):
            good = K.is_atiled_cfg(K.pick_cfg(N, Kd, M=M, kind="mxfp4", fold=fold)) == (M >= K.PREFILL_MIN_M)
            ok &= good
            print(f"  {'dispatch M=%d fold=%s: A-tiled iff M >= PREFILL_MIN_M' % (M, fold):80s} {'ok' if good else '<-- FAIL'}")
    # 3-D input through the wrapper (the model's [B, T, K] hidden states)
    x3 = torch.randn(2, 600, Kd, device="cuda", generator=g).to(torch.bfloat16)
    out3 = ops.mxfp4_linear(x3, W.wq[0], W.wsr[0], N, Kd, True)
    q, s = K.quant_rows_fp8(x3.reshape(-1, Kd))
    check("ops.mxfp4_linear 3-D [2, 600, K] fold", out3.reshape(-1, N), (q.float() * s[:, None]) @ wd.T)
    try:
        from r9700_vllm.linear.mxfp4 import R9700Mxfp4LinearKernel
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(W.wq, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(W.wsr, requires_grad=False)
        layer._r9k_nk = (N, Kd)
        kern = R9700Mxfp4LinearKernel.__new__(R9700Mxfp4LinearKernel)
        x = torch.randn(1500, Kd, device="cuda", generator=g).to(torch.bfloat16)
        q, s = K.quant_rows_fp8(x)
        for fold in (True, False):
            layer._r9k_fold = fold
            out = kern.apply_weights(layer, x, None)
            check(f"R9700Mxfp4LinearKernel.apply_weights M=1500 fold={fold} (served path)", out,
                  (q.float() * s[:, None]) @ wd.T)
    except ImportError as ex:
        print("  (R9700Mxfp4LinearKernel skipped:", ex, ")")
except ImportError as ex:          # no vLLM in this environment
    print("  (ops-level dispatch skipped:", ex, ")")

print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
