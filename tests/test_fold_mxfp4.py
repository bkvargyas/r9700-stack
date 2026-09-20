"""Folded-exponent MXFP4 (R9K_FOLD): the prefill tiles and the decode MT kernel with the block exponent folded into the
e4m3 weights against (a) the exact dequantized reference on ordinary weights (every block within e4m3's reach: the
fold is exact) and (b) a table-emulating reference on weights crafted to have rounded (9 <= d <= 12) and flushed
(d >= 13) blocks, where the kernel must reproduce the kMag semantics; plus the guard's accounting (fold_stats /
fold_ok) on the crafted tensor, and the R9K_FOLD=0 default (never fold).

Run inside the ROCm image: PYTHONPATH=/repo R9K_LIB=/repo/r9700_vllm/kernels/libr9k.so python3 tests/test_fold_mxfp4.py
"""
import dataclasses
import sys

import torch

from r9700_vllm.kernels import moe as K
from r9700_vllm.quant.nvfp4 import quantize_mxfp4_search

ok = True
g = torch.Generator(device="cuda").manual_seed(0)
LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
# kernels/r9k_moe_mxfp4a8.hip:kMag -- e4m3 bytes of e2m1 magnitude c (0..7) scaled by 2^-d, row d
KMAG = [(0x3c383000, 0x4c484440), (0x34302800, 0x44403c38), (0x2c282000, 0x3c383430), (0x24201800, 0x34302c28),
        (0x1c181000, 0x2c282420), (0x14100800, 0x24201c18), (0x0c080400, 0x1c181410), (0x06040200, 0x14100c08),
        (0x03020100, 0x0c080604), (0x01010000, 0x06040302), (0x01000000, 0x03020101), (0x00000000, 0x01010100),
        (0x00000000, 0x01000000), (0, 0), (0, 0), (0, 0)]


def e4m3(b):
    em = b & 0x7F
    return (2.0 ** ((em >> 3) - 7) * (1 + (em & 7) / 8)) if em >= 8 else em * 2.0 ** -9


FOLD_LUT = torch.tensor([[e4m3((t0 >> (8 * c)) & 255) for c in range(4)] + [e4m3((t1 >> (8 * c)) & 255) for c in range(4)]
                         for t0, t1 in KMAG], device="cuda")            # [16 d][8 c]


def mx_deq(p, e):
    N, Kh = p.shape
    c = torch.stack((p & 15, p >> 4), -1).reshape(N, Kh * 2).long()
    return (LUT[c].reshape(N, -1, 32) * torch.exp2(e.float() - 127).unsqueeze(-1)).reshape(N, Kh * 2)


def mx_deq_folded(p, e):
    """What the folded kernels compute: kMag(c, d) * 2^(ref - 127), d = ref - e clamped to 0..15."""
    N, Kh = p.shape
    c = torch.stack((p & 15, p >> 4), -1).reshape(N, Kh * 2).long()
    ref = e.max(dim=1, keepdim=True).values
    d = (ref.int() - e.int()).clamp(0, 15).long()                            # [N, K/32]
    mag = FOLD_LUT[d.repeat_interleave(32, 1), c & 7]
    sign = torch.where(c >= 8, -1.0, 1.0)
    return sign * mag * torch.exp2(ref.float() - 127)


def check(name, got, ref, tol=5e-3):
    global ok
    r = ((got.float() - ref).norm() / ref.norm()).item()
    good = r < tol and torch.isfinite(got.float()).all().item()
    ok &= good
    print(f"  {name:78s} rel {r:.2e} {'ok' if good else '<-- FAIL'}", flush=True)
    return r


def identity(M, blk):
    mpad = (M + blk - 1) // blk * blk
    return (torch.arange(mpad, dtype=torch.int32, device="cuda"), torch.zeros(mpad // blk, dtype=torch.int32, device="cuda"),
            torch.full((1,), mpad, dtype=torch.int32, device="cuda"))


def run_prefill(W, q, s, M, cfg):
    out = torch.full((M, W.N), float("nan"), dtype=torch.bfloat16, device="cuda")
    K.moe_gemm(q, s, W, out, *identity(M, K.prefill_block(cfg)), M, 1, None, num_experts=1, prefill=cfg)
    return out


def run_mt(W, q, s, M, cfg, MT, ldsa):
    out = torch.full((M, W.N), float("nan"), dtype=torch.bfloat16, device="cuda")
    K.moe_gemm(q, s, W, out, *identity(M, 16 * MT), M, 1, None, *cfg, num_experts=1, MT=MT, ldsa=ldsa)
    return out


assert K.fold_supported(), "libr9k without r9k_fold_supported"

# ---- (a) ordinary weights: fold exact -> same 5e-3 gate as the exact kernels, every prefill tile and MT config
N, Kd = 1040, 1024
w = torch.randn(N, Kd, device="cuda", generator=g) * 0.02
p, e = quantize_mxfp4_search(w)
st = K.fold_stats(K.pack_scales(e[None]))
print(f"  gaussian 1040x1024: {st}")
Wx = K.prepare_mxfp4_weights(p[None], e[None])
Wf = dataclasses.replace(Wx, fold=True)
wd = mx_deq(p, e)
assert (mx_deq_folded(p, e) - wd).abs().max().item() == 0 or st.p_inexact > 0, "table emulation disagrees on exact blocks"
for M in (256, 300, 1024):
    x = torch.randn(M, Kd, device="cuda", generator=g).to(torch.bfloat16)
    q, s = K.quant_rows_fp8(x)
    ref = (q.float() * s[:, None]) @ wd.T
    for cfg in sorted(K.PREFILL_TILES):
        check(f"fold prefill 1040x1024 M={M} cfg={cfg} {K.PREFILL_TILES[cfg]}", run_prefill(Wf, q, s, M, cfg), ref)
for M in (1, 8, 64):
    x = torch.randn(M, Kd, device="cuda", generator=g).to(torch.bfloat16)
    q, s = K.quant_rows_fp8(x)
    ref = (q.float() * s[:, None]) @ wd.T
    for MT in (1, 2, 4):
        for cfg in ((2, 4, 2), (4, 2, 1), (4, 1, 1)):
            for ldsa in ((False, True) if MT > 1 else (False,)):
                check(f"fold MT kernel 1040x1024 M={M} MT={MT} cfg={cfg} ldsa={ldsa}", run_mt(Wf, q, s, M, cfg, MT, ldsa), ref)

# ---- served dense shapes on the default prefill cfgs, folded vs exact kernel vs reference
for (N, Kd) in ((17408, 5120), (5120, 8704), (8240, 5120)):
    w = torch.randn(N, Kd, device="cuda", generator=g) * 0.02
    p, e = quantize_mxfp4_search(w)
    Wx = K.prepare_mxfp4_weights(p[None], e[None])
    Wf = dataclasses.replace(Wx, fold=True)
    wd = mx_deq(p, e)
    for M in (512, 2048):
        cfg = K.pick_cfg(N, Kd, 32, M=M, kind="mxfp4")
        x = torch.randn(M, Kd, device="cuda", generator=g).to(torch.bfloat16)
        q, s = K.quant_rows_fp8(x)
        ref = (q.float() * s[:, None]) @ wd.T
        check(f"fold dense {N}x{Kd} M={M} cfg={cfg}", run_prefill(Wf, q, s, M, cfg[1]), ref)
    del w, p, e, Wx, Wf, wd

# ---- (b) crafted exponents: rows with blocks at every d 0..16 -> the guard counts them and the kernel matches kMag
N, Kd = 512, 2048
nb = Kd // 32
p = torch.randint(0, 256, (N, Kd // 2), dtype=torch.uint8, device="cuda", generator=g)
d_tab = torch.randint(0, 17, (N, nb), device="cuda", generator=g)       # d = 16 -> clamped to 15 (flushed)
d_tab[:, 0] = 0                                                         # the reference block
e = (120 - d_tab).clamp(1, 254).to(torch.uint8)
wsr = K.pack_scales(e[None])
st = K.fold_stats(wsr)
exp_hist = torch.bincount(d_tab.reshape(-1).clamp(0, 16), minlength=17).tolist()
good = st.hist == exp_hist and st.blocks == N * nb and st.max_d == 16
ok &= good
print(f"  guard accounting: {st} {'ok' if good else '<-- FAIL'}")
p_in, p_fl = sum(exp_hist[9:]) / (N * nb), sum(exp_hist[13:]) / (N * nb)
good = abs(st.p_inexact - p_in) < 1e-12 and abs(st.p_flush - p_fl) < 1e-12 and not K.fold_ok(st)
ok &= good
print(f"  guard refuses: inexact {st.p_inexact:.3f} > {K.FOLD_MAX_INEXACT} / flush {st.p_flush:.4f} > {K.FOLD_MAX_FLUSH}"
      f" {'ok' if good else '<-- FAIL'}")
# fold_decide honours R9K_FOLD: default off never folds (and never computes stats), 'force' folds a refused tensor
saved = K.FOLD_MODE
for mode, want in (("0", False), ("1", False), ("force", True)):
    K.FOLD_MODE = mode
    got = K.fold_decide(wsr, f"crafted mode={mode}", log=lambda m: print("   ", m))
    ok &= got == want
    print(f"  fold_decide R9K_FOLD={mode}: {got} {'ok' if got == want else '<-- FAIL'}")
K.FOLD_MODE = saved
Wx = K.prepare_mxfp4_weights(p[None], e[None])
Wf = dataclasses.replace(Wx, fold=True)
wd, wf = mx_deq(p, e), mx_deq_folded(p, e)
print(f"  crafted weights: folded-vs-exact weight rel err {((wf - wd).norm() / wd.norm()).item():.2e}")
for M in (64, 512):
    x = torch.randn(M, Kd, device="cuda", generator=g).to(torch.bfloat16)
    q, s = K.quant_rows_fp8(x)
    ref_exact, ref_fold = (q.float() * s[:, None]) @ wd.T, (q.float() * s[:, None]) @ wf.T
    for cfg in (11, 12, 15):
        out = run_prefill(Wf, q, s, M, cfg)
        check(f"crafted prefill cfg={cfg} M={M} vs kMag emulation", out, ref_fold)
        r = ((out.float() - ref_exact).norm() / ref_exact.norm()).item()
        print(f"  {'':4s}(same vs exact dequant: rel {r:.2e} -- the flush/rounding error, reported not gated)")
        check(f"exact  prefill cfg={cfg} M={M} (fold=False) vs exact dequant", run_prefill(Wx, q, s, M, cfg), ref_exact)
    for MT in (1, 4):
        check(f"crafted MT={MT} M={M} vs kMag emulation", run_mt(Wf, q, s, M, (2, 4, 2), MT, MT > 1), ref_fold)

# ---- routed MoE (Flash-Next shapes) folded: gate_up (a_row_div = topk) and down (router weight) on the served tiles
E, M, topk = 16, 1200, 4
Ws1, Ws2, wd1, wd2 = [], [], [], []
for _ in range(E):
    for (Nn, Kk, Ws, wdl) in ((640, 2560, Ws1, wd1), (2560, 320, Ws2, wd2)):
        w = torch.randn(Nn, Kk, device="cuda", generator=g) * 0.02
        pp, ee = quantize_mxfp4_search(w)
        Ws.append((pp, ee))
        wdl.append(mx_deq(pp, ee))
W1 = dataclasses.replace(K.prepare_mxfp4_weights(torch.stack([a for a, _ in Ws1]), torch.stack([b for _, b in Ws1])), fold=True)
W2 = dataclasses.replace(K.prepare_mxfp4_weights(torch.stack([a for a, _ in Ws2]), torch.stack([b for _, b in Ws2])), fold=True)
wd1, wd2 = torch.stack(wd1), torch.stack(wd2)
scores = torch.randn(M, E, device="cuda", generator=g)
topk_w, topk_ids = torch.softmax(scores, -1).topk(topk, dim=-1)
numel = M * topk
fl = topk_ids.flatten()


def routed_ref(ad, wd, a_row_div):
    ref = torch.empty(numel, wd.shape[1], device="cuda")
    for ex in range(E):
        rows = (fl == ex).nonzero().flatten()
        ref[rows] = ad[rows // a_row_div] @ wd[ex].T
    return ref


x = torch.randn(M, 2560, device="cuda", generator=g).to(torch.bfloat16)
xq, xs = K.quant_rows_fp8(x)
h = torch.randn(numel, 320, device="cuda", generator=g).to(torch.bfloat16)
hq, hs = K.quant_rows_fp8(h)
tw = topk_w.flatten().float().contiguous()
ref1, ref2 = routed_ref(xq.float() * xs[:, None], wd1, topk), routed_ref(hq.float() * hs[:, None], wd2, 1) * tw[:, None]
for cfg in (11, 15):
    blk = K.prefill_block(cfg)
    sid, eid, ntpp = (v.cuda() for v in K.align_block_size_ref(topk_ids.cpu(), E, blk))
    out1 = torch.full((numel, 640), float("nan"), dtype=torch.bfloat16, device="cuda")
    K.moe_gemm(xq, xs, W1, out1, sid, eid, ntpp, numel, topk, None, num_experts=E, prefill=cfg)
    check(f"fold routed gate_up E={E} M={M} top{topk} prefill cfg={cfg}", out1, ref1)
    out2 = torch.full((numel, 2560), float("nan"), dtype=torch.bfloat16, device="cuda")
    K.moe_gemm(hq, hs, W2, out2, sid, eid, ntpp, numel, 1, tw, num_experts=E, prefill=cfg)
    check(f"fold routed down    E={E} M={M} top{topk} prefill cfg={cfg}", out2, ref2)
for MT in (1, 4):
    sid, eid, ntpp = (v.cuda() for v in K.align_block_size_ref(topk_ids.cpu(), E, 16 * MT))
    out1 = torch.full((numel, 640), float("nan"), dtype=torch.bfloat16, device="cuda")
    K.moe_gemm(xq, xs, W1, out1, sid, eid, ntpp, numel, topk, None, 2, 4, 2, num_experts=E, MT=MT, ldsa=MT > 1)
    check(f"fold routed gate_up E={E} M={M} top{topk} MT={MT}", out1, ref1)
    out2 = torch.full((numel, 2560), float("nan"), dtype=torch.bfloat16, device="cuda")
    K.moe_gemm(hq, hs, W2, out2, sid, eid, ntpp, numel, 1, tw, 4, 2, 1, num_experts=E, MT=MT, ldsa=MT > 1)
    check(f"fold routed down    E={E} M={M} top{topk} MT={MT}", out2, ref2)

print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
