"""r9k decode GEMV for EXL3 matrices (kernels/r9k_exl3_gemv.hip, r9700_vllm/quant/exl3_gemv.py) on one gfx1201 card.

1. Trellis decode: the kernel's copy of the decode (r9k_exl3_gemv_dq) == ext.reconstruct bit for bit, K 1-8, mul1 / mcg.
2. Numerics per shape (synthetic K = 4 / 5 / 6 at the TP8 per-rank attention shapes, plus real pack tensors), M = 1..8,
   12, 16:
   (a) same rotated input: x_rot = f16(had(x * suh) / sqrt 128) computed in torch and fed with rotated=True ->
       vs f32 torch: x_rot @ reconstruct(W) (f32), output Hadamard, * svh. Only the f32 summation order differs
       (gate: rms rel <= 4e-6);
   (b) the in-kernel rotation from bf16 x vs the full f32 reference x @ W_f32 (gate rms rel 2e-3, P1's discipline) and
       exllamav3's exl3_gemm (f32 C) on the same x for comparison;
   (c) bf16 / f16 outputs == the f32 output rounded (bitwise);
   (d) stacked parts (fused_wqa_wkv: [1280, 512] with their own suh) into one [m, 1792] output, per-part lists.
3. Exl3LinearMethod.apply with R9K_EXL3_GEMV=r9k vs =exl3 (exllamav3) on the same layer: both vs the f32 reference.
4. HIP graph: several plans captured (M = 1, 6, 8, 16), replay with new inputs == eager bitwise, chained.
5. Timing (graph replay over a ring of distinct matrices > the 64 MB Infinity Cache, so weights come from DRAM):
   us a call and GB/s of trellis per shape, exllamav3's apply path vs ours; then a synthetic attention layer
   (fused_wqa_wkv + wq_b + wo_a + wo_b; wo_a bf16 einsum before / EXL3 after) at M = 1 and 6: GPU us a layer and
   launches a layer (torch profiler, eager), and the step projection.

Run (VM100, card 7): ~/ds41/exl3/gpu-run.sh 7 python3 tests/test_exl3_gemv.py   (R9K_GEMV_LIB / R9K_LIB = a libr9k.so with
r9k_exl3_gemv; ONLY=dq,num,real,apply,graph,bench selects parts; EXL3_REAL=/ds41/exl3/D4/real-attn.safetensors)
"""
import os
import sys
import time
from types import SimpleNamespace

import torch

import exllamav3_ext as ext  # noqa: E402
from r9700_vllm.quant import exl3 as E  # noqa: E402
from r9700_vllm.quant import exl3_gemv as G  # noqa: E402

DEV = torch.device("cuda", 0)
BF, F16, F32 = torch.bfloat16, torch.float16, torch.float32
ONLY = os.environ.get("ONLY", "")
REAL = os.environ.get("EXL3_REAL", "/ds41/exl3/D4/real-attn.safetensors")
FAIL = []


def want(tag):
    return not ONLY or tag in ONLY.split(",")


def check(name, ok, msg=""):
    print(("  ok   " if ok else "  FAIL ") + name + (("  " + msg) if msg else ""), flush=True)
    if not ok:
        FAIL.append(name)


def rr(y, ref):
    y, ref = y.float(), ref.float()
    return ((y - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def mr(y, ref):
    y, ref = y.float(), ref.float()
    return ((y - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()


def synth(k, n, K, seed, dst=0):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    tr = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * K), dtype=torch.int32, device=DEV, generator=g).to(torch.int16)
    suh = ((torch.randint(0, 2, (k,), device=DEV, generator=g) * 2 - 1).float()
           * (0.02 + 0.03 * torch.rand(k, device=DEV, generator=g))).half()          # the pack's suh carries a scale
    svh = ((torch.rand(n, device=DEV, generator=g) - 0.5) * 2 * (0.6 + 1.2 * torch.rand(n, device=DEV, generator=g))
           * 30 / k ** 0.5).half()
    return E.Exl3Part(tr, suh, svh, K, False, True, dst, n)


def real_parts():
    if not os.path.exists(REAL):
        print(f"  (real tensors {REAL} missing: skipped)")
        return {}
    from safetensors.torch import load_file
    t = load_file(REAL, device="cuda:0")
    out = {}
    for name in sorted({k.rsplit(".", 1)[0] for k in t}):
        tr = t[name + ".trellis"].contiguous()
        out[name] = E.Exl3Part(tr, t[name + ".suh"].contiguous(), t[name + ".svh"].contiguous(), E.k_of(tr), False,
                               name + ".mul1" in t, 0, tr.shape[1] * 16)
    return out


def rot_in(x, suh):
    H = E._had128(DEV)
    m, k = x.shape
    return ((x.float() * suh.float()).view(m, k // 128, 128) @ H).view(m, k).half()


def ref_same_rot(xr16, p):
    """f32: the kernel's own math on a given f16 rotated input (exact decode, f32 accumulation, output rotation)."""
    H = E._had128(DEV)
    w = torch.empty((p.trellis.shape[0] * 16, p.n), dtype=F16, device=DEV)
    ext.reconstruct(w, p.trellis, p.K, p.mcg, p.mul1)
    acc = xr16.float() @ w.float()
    m = acc.shape[0]
    return (acc.view(m, p.n // 128, 128) @ H).view(m, p.n) * p.svh.float()


def wf32(p):
    return E.dequant_f32(p.trellis, p.suh, p.svh, p.K, p.mul1, p.mcg)


def exl3_gemm_ref(x, p):
    return E.exl3_gemm_f16(x.half().contiguous(), p.trellis, p.suh, p.svh, p.K, p.mul1, p.mcg, out_f32=True)


# ----------------------------------------------------------------------------------------------------------- 1. dq
def check_dq():
    g = torch.Generator(device=DEV)
    g.manual_seed(7)
    for K in range(1, 9):
        for mcg, mul1 in ((False, True), (True, False)):
            for k, n in ((1280, 4096), (4096, 1024), (256, 384)):
                t = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * K), dtype=torch.int32, device=DEV,
                                  generator=g).to(torch.int16)
                ref = torch.empty((k, n), dtype=F16, device=DEV)
                ext.reconstruct(ref, t, K, mcg, mul1)
                got = G.dq(t, K, 2 if mul1 else 1)
                same = torch.equal(ref.view(torch.int16), got.view(torch.int16))
                if not same or (k, n) == (1280, 4096):
                    check(f"decode K={K} {'mul1' if mul1 else 'mcg '} {k}x{n} == ext.reconstruct", same)


# ------------------------------------------------------------------------------------------------------ 2. numerics
MS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 33, 48)


def check_matrix(name, p, Ms=MS, seed=0):
    k = p.trellis.shape[0] * 16
    p = E.Exl3Part(p.trellis, p.suh, p.svh, p.K, p.mcg, p.mul1, 0, p.n)
    plan = G.Exl3GemvPlan([p], k)
    plan.reserve(48)
    W = wf32(p)
    g = torch.Generator(device=DEV)
    g.manual_seed(1000 + seed)
    worst_a = worst_b = worst_x = 0.0
    ok = True
    for m in Ms:
        x = (torch.randn((m, k), device=DEV, generator=g)).to(BF)
        xr = rot_in(x, p.suh)
        y_a = torch.empty((m, p.n), dtype=F32, device=DEV)
        G.r9k_exl3_gemv(plan, [xr], [y_a], rotated=True)
        ra = rr(y_a, ref_same_rot(xr, p))
        y32 = torch.empty((m, p.n), dtype=F32, device=DEV)
        G.r9k_exl3_gemv(plan, x, y32)
        full = x.float() @ W
        rb = rr(y32, full)
        rx = rr(exl3_gemm_ref(x, p), full)
        ybf = torch.empty((m, p.n), dtype=BF, device=DEV)
        y16 = torch.empty((m, p.n), dtype=F16, device=DEV)
        G.r9k_exl3_gemv(plan, x, ybf)
        G.r9k_exl3_gemv(plan, x, y16)
        same = torch.equal(ybf, y32.to(BF)) and torch.equal(y16, y32.half())
        good = ra <= 4e-6 and rb <= 2e-3 and same and bool(torch.isfinite(y32).all())
        ok &= good
        worst_a, worst_b, worst_x = max(worst_a, ra), max(worst_b, rb), max(worst_x, rx)
        if not good:
            print(f"    {name} m={m}: same-rot {ra:.2e} full {rb:.2e} (exllamav3 {rx:.2e}) bf16/f16==round {same}")
    check(f"{name} [{k}->{p.n} K{p.K}] M={Ms[0]}..{Ms[-1]}", ok,
          f"same rotated input rms rel <= {worst_a:.1e}; vs f32 x@W {worst_b:.2e} (exllamav3 exl3_gemm {worst_x:.2e}); "
          f"bf16/f16 out == f32 rounded")
    return ok


def check_stacked(name, p0, p1, Ms=(1, 3, 6, 8, 16)):
    k = p0.trellis.shape[0] * 16
    p1 = E.Exl3Part(p1.trellis, p1.suh, p1.svh, p1.K, p1.mcg, p1.mul1, p0.n, p1.n)
    plan = G.Exl3GemvPlan([p0, p1], k)
    plan.reserve(48)
    W = torch.cat([wf32(p0), wf32(p1)], 1)
    g = torch.Generator(device=DEV)
    g.manual_seed(77)
    ok, worst = True, 0.0
    for m in Ms:
        x = torch.randn((m, k), device=DEV, generator=g).to(BF)
        y = torch.empty((m, p0.n + p1.n), dtype=BF, device=DEV)
        G.r9k_exl3_gemv(plan, x, y)
        y0, y1 = torch.empty((m, p0.n), dtype=F32, device=DEV), torch.empty((m, p1.n), dtype=F32, device=DEV)
        G.r9k_exl3_gemv(plan, [x, x], [y0, y1])
        same = torch.equal(y, torch.cat([y0, y1], 1).to(BF))
        r = rr(torch.cat([y0, y1], 1), x.float() @ W)
        worst = max(worst, r)
        ok &= same and r <= 2e-3
    check(f"{name} stacked [{k}->{p0.n}+{p1.n}] one launch, M={Ms}", ok,
          f"vs f32 {worst:.2e}; [m, n_total] out == per-part lists")


# ------------------------------------------------------------------------------------------- 3. Exl3LinearMethod
def fake_layer(parts, name):
    k = parts[0].trellis.shape[0] * 16
    layer = SimpleNamespace(exl3_dense=False, exl3_parts=parts, exl3_k=k, exl3_n=sum(p.n for p in parts), prefix=name)
    E.Exl3LinearMethod._warm(layer)
    return layer


def check_apply(name, parts, Ms=(1, 2, 6, 8, 9, 16, 24, 48)):
    layer = fake_layer(parts, name)
    meth = E.Exl3LinearMethod(None)
    W = torch.cat([wf32(p) for p in parts], 1)
    g = torch.Generator(device=DEV)
    g.manual_seed(5)
    ok = layer.exl3_gemv is not None
    msg = []
    for m in Ms:
        x = torch.randn((m, layer.exl3_k), device=DEV, generator=g).to(BF)
        ref = x.float() @ W
        E.GEMV = "r9k"
        y = meth.apply(layer, x)
        E.GEMV = "exl3"
        y0 = meth.apply(layer, x)
        E.GEMV = "r9k"
        a, b = rr(y, ref), rr(y0, ref)
        ok &= y.dtype == BF and a <= 4e-3
        msg.append(f"M{m} {a:.1e}/{b:.1e}")
    check(f"apply {name} r9k vs exllamav3 (bf16 out, rms rel vs f32)", ok, " ".join(msg))
    return layer


def check_wo_a_hook(p0, p1):
    """models/deepseek_v41_rdna4.wo_a_exl3: TP8 (one o-group: the layer's apply) and two local o-groups (TP4: per-group
    inputs into one launch; reconstruct per part above GEMV_M) vs the per-group f32 reference."""
    from r9700_vllm.models import deepseek_v41_rdna4 as R
    meth = E.Exl3LinearMethod(None)
    one = fake_layer([p0], "wo_a")
    one.quant_method = meth
    q1 = E.Exl3Part(p1.trellis, p1.suh, p1.svh, p1.K, p1.mcg, p1.mul1, p0.n, p1.n)
    two = fake_layer([p0, q1], "wo_a2")
    two.quant_method = meth
    W0, W1 = wf32(p0), wf32(p1)
    ok, msg = True, []
    for T in (1, 6, 16, 40, 64):
        o = torch.randn((T, 8192), device=DEV).to(BF)
        z1 = R.wo_a_exl3(one, o[:, :4096].contiguous())
        z2 = R.wo_a_exl3(two, o)
        r1 = rr(z1, o[:, :4096].float() @ W0)
        r2 = rr(z2, torch.cat([o[:, :4096].float() @ W0, o[:, 4096:].float() @ W1], 1))
        ok &= r1 <= 4e-3 and r2 <= 4e-3 and z1.dtype == BF and z2.shape == (T, 2048)
        msg.append(f"T{T} {r1:.1e}/{r2:.1e}")
    check("wo_a_exl3 hook, 1 and 2 local o-groups (bf16 out vs f32)", ok, " ".join(msg))


# --------------------------------------------------------------------------------------------------------- 4. graph
def check_graph(parts_list):
    plans = [G.Exl3GemvPlan(ps, ps[0].trellis.shape[0] * 16) for ps in parts_list]
    for pl in plans:
        pl.reserve(16)
    ok = True
    for m in (1, 6, 8, 16, 24, 48):
        xs = [torch.randn((m, pl.k), device=DEV).to(BF) for pl in plans]
        ys = [torch.empty((m, pl.n_total), dtype=BF, device=DEV) for pl in plans]
        # chain: plan 0's output feeds plan 1 when the widths line up (wq_a 1280 -> wq_b 1280)
        def run():
            for i, pl in enumerate(plans):
                src = xs[i]
                if i > 0 and plans[i - 1].n_total >= pl.k and i == 1:
                    src = ys[0][:, :pl.k]
                G.r9k_exl3_gemv(pl, src, ys[i])
        run()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            run()
        diffs = []
        for t in range(3):
            for x in xs:
                x.copy_(torch.randn(x.shape, device=DEV).to(BF))
            gr.replay()
            got = [y.clone() for y in ys]
            gr.replay()
            again = [y.clone() for y in ys]
            run()
            torch.cuda.synchronize()
            diffs += [(a.float() - y.float()).abs().max().item() for a, y in zip(got, ys)]
            diffs += [(a.float() - b.float()).abs().max().item() for a, b in zip(got, again)]
        good = max(diffs) == 0.0
        ok &= good
    check(f"HIP graph: {len(plans)} plans (one chained) at M = 1, 6, 8, 16: replay x2 == eager bitwise", ok)


# --------------------------------------------------------------------------------------------------------- 5. bench
def graph_us(fns, iters=30):
    """fns: list of callables (one ring step each) captured in one graph; us per callable."""
    for f in fns:
        f()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for f in fns:
            f()
    for _ in range(3):
        gr.replay()
    torch.cuda.synchronize()
    best = 1e30
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(iters):
            gr.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters / len(fns) * 1e6)
    return best


def count_kernels(fn):
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
    return len(names), names


RING = int(os.environ.get("R9K_TEST_RING", 12))
RING_MB = float(os.environ.get("R9K_TEST_RING_MB", 192))     # > the 64 MB Infinity Cache: weights stream from DRAM


def bench_shapes():
    """us a call (GB/s of trellis) per shape over a ring of RING distinct matrices (cold weights)."""
    shapes = [("fused_wqa_wkv", 5120, (1280, 512), 5), ("fused_wqa_wkv L0/drafter", 5120, (1280, 512), 6),
              ("wq_b / indexer wq_b", 1280, (4096,), 5), ("wq_b drafter", 1280, (4096,), 6),
              ("wo_a slice", 4096, (1024,), 5), ("wo_b rows", 1024, (5120,), 5), ("wo_b drafter", 1024, (5120,), 6),
              ("engram wkv cols", 6144, (3200,), 4), ("mtp main_proj", 15360, (5120,), 6)]
    only = os.environ.get("R9K_TEST_SHAPES")
    if only:
        shapes = [s for s in shapes if s[0].replace(" ", "_") in only.split(",")]
    Ms = tuple(int(v) for v in os.environ.get("R9K_TEST_MS", "1,2,4,6,8,16,24,48").split(","))   # > 16: exllamav3's side
                                          # is the reconstruct path (RECON_M 16); ours is 8-row groups
    print(f"  {'shape':28s} {'MB':>5s} | " + " | ".join(f"M={m:<2d} exl3 / r9k us (GB/s)" for m in Ms))
    rows = []
    for name, k, ns, K in shapes:
        layers = []
        mb = sum(k * n * K / 8 for n in ns) / 1e6
        for i in range(max(RING if mb < 32 else 3, int(RING_MB / mb + 0.999))):   # < 1 GiB of VRAM
            parts, dst = [], 0
            for j, n in enumerate(ns):
                parts.append(synth(k, n, K, 100 * i + j, dst))
                dst += n
            layers.append(fake_layer(parts, name))
        meth = E.Exl3LinearMethod(None)
        cells = []
        for m in Ms:
            x = torch.randn((m, k), device=DEV).to(BF)
            E.GEMV = "exl3"
            t_x = float("nan") if os.environ.get("R9K_TEST_NOEXL3") else \
                graph_us([lambda l=l: meth.apply(l, x) for l in layers])
            E.GEMV, E.GEMV_M = "r9k", max(E.GEMV_M, m)
            t_r = graph_us([lambda l=l: meth.apply(l, x) for l in layers])
            E.GEMV_M = 48
            tag = "" if layers[0].exl3_gemv.takes(m) else "*"
            cells.append(f"{t_x:6.1f} / {t_r:5.1f}{tag:1s}({mb / t_r * 1e3:4.0f})")
            rows.append((name, k, ns, K, m, t_x, t_r, mb))
        print(f"  {name:28s} {mb:5.2f} | " + " | ".join(cells), flush=True)   # * = plan.takes(m) is False: apply's old path
        del layers
        torch.cuda.empty_cache()
    return rows


def bench_layer():
    """A synthetic TP8 attention layer: fused_wqa_wkv (5120 -> 1280 + 512), wq_b (1280 -> 4096), wo_a (4096 -> 1024),
    wo_b (1024 -> 5120), K = 5. Before: exllamav3's apply for the three EXL3 linears + stock's bf16 einsum for wo_a;
    after: r9k for all four. GPU us a layer over a ring of layers, launches a layer (eager, profiler)."""
    L = []
    for i in range(RING):
        qa, kv = synth(5120, 1280, 5, 11 * i), synth(5120, 512, 5, 11 * i + 1, 1280)
        wq_b, wo_a, wo_b = synth(1280, 4096, 5, 11 * i + 2), synth(4096, 1024, 5, 11 * i + 3), synth(1024, 5120, 5, 11 * i + 4)
        L.append({"qkv": fake_layer([qa, kv], "fused_wqa_wkv"), "wq_b": fake_layer([wq_b], "wq_b"),
                  "wo_a": fake_layer([wo_a], "wo_a"), "wo_b": fake_layer([wo_b], "wo_b"),
                  "wo_a_bf16": wf32(wo_a).t().contiguous().to(BF).view(1, 1024, 4096)})
    meth = E.Exl3LinearMethod(None)
    res = {}
    for m in (1, 6, 24, 48):
        x = torch.randn((m, 5120), device=DEV).to(BF)
        qr = torch.randn((m, 1280), device=DEV).to(BF)
        o = torch.randn((m, 4096), device=DEV).to(BF)

        def before(l):
            meth.apply(l["qkv"], x)
            meth.apply(l["wq_b"], qr)
            z = torch.einsum("tgd,grd->tgr", o.view(m, 1, 4096), l["wo_a_bf16"])
            meth.apply(l["wo_b"], z.flatten(1))

        def after(l):
            meth.apply(l["qkv"], x)
            meth.apply(l["wq_b"], qr)
            z = meth.apply(l["wo_a"], o)
            meth.apply(l["wo_b"], z)

        E.GEMV = "r9k"
        t_ein = graph_us([lambda l=l: torch.einsum("tgd,grd->tgr", o.view(m, 1, 4096), l["wo_a_bf16"]) for l in L])
        t_woa = graph_us([lambda l=l: meth.apply(l["wo_a"], o) for l in L])
        print(f"  wo_a M={m}: stock bf16 einsum {t_ein:6.1f} us (8.39 MB) | EXL3 r9k {t_woa:6.1f} us (2.62 MB)", flush=True)
        E.GEMV = "exl3"
        tb = graph_us([lambda l=l: before(l) for l in L])
        nb, names_b = count_kernels(lambda: before(L[0]))
        E.GEMV = "r9k"
        ta = graph_us([lambda l=l: after(l) for l in L])
        na, names_a = count_kernels(lambda: after(L[0]))
        res[m] = (tb, nb, ta, na)
        print(f"  layer M={m}: before {tb:6.1f} us, {nb} launches | after {ta:6.1f} us, {na} launches", flush=True)
        if m == 1:
            print("    before kernels:", ", ".join(n[:40] for n in names_b))
            print("    after kernels: ", ", ".join(n[:40] for n in names_a))
    return res


if __name__ == "__main__":
    print("device", torch.cuda.get_device_name(), "| lib", os.environ.get("R9K_GEMV_LIB") or os.environ.get("R9K_LIB"),
          "| waves a block", G.lib().r9k_exl3_gemv_waves(), "| blocks target", G.BLOCKS, flush=True)
    if want("dq"):
        print("-- trellis decode vs ext.reconstruct")
        check_dq()
    syn = {"wq_a K5": synth(5120, 1280, 5, 1), "wkv K5": synth(5120, 512, 5, 2, 1280), "wq_b K5": synth(1280, 4096, 5, 3),
           "wo_a K5": synth(4096, 1024, 5, 4), "wo_b K5": synth(1024, 5120, 5, 5), "wq_a K6": synth(5120, 1280, 6, 6),
           "wkv K6": synth(5120, 512, 6, 7, 1280), "wq_b K6": synth(1280, 4096, 6, 8), "wo_b K6": synth(1024, 5120, 6, 9),
           "engram K4": synth(6144, 3200, 4, 10), "K3 5120->2304": synth(5120, 2304, 3, 11),
           "K2 2304->5120": synth(2304, 5120, 2, 12)}
    if want("num"):
        print("-- numerics, synthetic (TP8 per-rank shapes)")
        for i, (nm, p) in enumerate(syn.items()):
            check_matrix(nm, p, seed=i)
        check_stacked("fused_wqa_wkv K5", syn["wq_a K5"], syn["wkv K5"])
        check_stacked("fused_wqa_wkv K6", syn["wq_a K6"], syn["wkv K6"])
    real = real_parts() if (want("real") or want("apply")) else {}
    if want("real") and real:
        print("-- numerics, real pack tensors (TP8 rank 0 slices)")
        for i, (nm, p) in enumerate(real.items()):
            check_matrix(nm, p, seed=50 + i)
        for lay in ("layers.0", "layers.39", "mtp.0"):
            if f"{lay}.attn.wq_a" in real:
                check_stacked(f"{lay} fused_wqa_wkv", real[f"{lay}.attn.wq_a"], real[f"{lay}.attn.wkv"])
    if want("apply"):
        print("-- Exl3LinearMethod.apply: R9K_EXL3_GEMV=r9k vs exl3")
        check_apply("fused_wqa_wkv K5", [syn["wq_a K5"], syn["wkv K5"]])
        check_apply("wo_a K5", [syn["wo_a K5"]])
        check_wo_a_hook(syn["wo_a K5"], synth(4096, 1024, 5, 40))
        if real:
            check_apply("real layers.0.attn.wo_a.slice.0", [real["layers.0.attn.wo_a.slice.0"]])
    if want("graph"):
        print("-- HIP graph")
        check_graph([[syn["wq_a K5"], syn["wkv K5"]], [syn["wq_b K5"]], [syn["wo_a K5"]], [syn["wo_b K6"]]])
    del syn
    torch.cuda.empty_cache()
    if want("bench"):
        print(f"-- timing: graph replay over a ring of {RING} matrices a shape, us a call (GB/s of trellis, ours)")
        bench_shapes()
        if not os.environ.get("R9K_TEST_NOLAYER"):
            print("-- a synthetic attention layer (K = 5, TP8 rank shapes)")
            bench_layer()
    print("peak VRAM %.0f MiB" % (torch.cuda.max_memory_allocated() / 2**20))
    print("ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)
