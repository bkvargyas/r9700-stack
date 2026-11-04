"""r9k EXL3 decode tier (kernels/r9k_moe_exl3.hip, r9700_vllm/moe/exl3_decode.py) on one gfx1201 card.

1. Trellis decode: the kernel's decode of whole matrices (r9k_exl3_dq_tiles) == ext.reconstruct bit for bit, K 1-8,
   mul1 and mcg codebooks, random trellis (every bit stream is a valid code).
2. MoE numerics: Exl3MoEMethod layer as rank 0 of EP 8 (tests/test_exl3_moe.make_layer: 48 local of 384 experts,
   synthetic K = 3 / 2 / 4, real pack experts K = 3 (layer 0) / K = 2 (layer 18)) vs the per-expert f32 reference;
   kernel output in f32 (rms rel gate 2e-3) and through Exl3MoEMethod.apply in bf16 (gate 1e-2 like test_exl3_moe).
   Routings: random 6-of-384, all-local, none-local (exact zeros), shared experts across tokens (group rows), x*6
   (act_limit clipping).
3. HIP graph: capture at T, replay with new inputs / routing == eager bitwise, chained layers, T = 1 .. 48.
4. Grouping: the device-side group table (r9k_moe_exl3_groups, the setup() both kernels run) == the reference
   grouping (each expert's local slots in slot order, runs of at most ROWS): one trellis pass a group, so the trellis
   bytes a call reads = groups x an expert's bytes; T = 1 .. 48, repeats, all-local, none.
5. The drafter's geometry (128 experts, 3 a token, K = 4: 16 local at EP 8), numerics + graphs, T = 1 .. 48.
6. Timing (graph replay, us a layer) r9k vs mgemv at T = 1 .. 48, EP routing with a fixed number of local picks over
   a fixed number of distinct experts (the expected counts for uniform routing at EP 8: T = 48 -> 36 picks over ~25),
   K = 3 and K = 2; at T = 48 also 36 picks over 12 / 36 experts (time follows the distinct experts, not the picks).

Run (VM100, card 5): ~/ds41/exl3/gpu-run.sh 5 python3 tests/test_exl3_moe_r9k.py   (R9K_LIB=libr9k.so with the kernel)
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_exl3_moe as TM  # noqa: E402

import exllamav3_ext as ext  # noqa: E402
from r9700_vllm.moe import exl3_decode as XD  # noqa: E402
from r9700_vllm.quant import exl3_moe as M  # noqa: E402
from r9700_vllm.quant.exl3 import dequant_f32, k_of  # noqa: E402

# R9K_TEST_EP: expert-parallel width of the simulated rank (8 = 48 local experts; 96 = 4, for a < 1 GiB run beside a
# server: the f32 reference holds 141 MB an expert). R9K_TEST_BENCH_EP: the same for the timing layer, which keeps no
# reference (24 = 16 experts, 212 MB at K = 3).
EP = int(os.environ.get("R9K_TEST_EP", 8))
BENCH_EP = int(os.environ.get("R9K_TEST_BENCH_EP", EP))


def set_ep(ep):
    global E
    TM.EP, TM.E = ep, TM.G // ep
    E = TM.E


def set_geom(g, topk, ep):
    """Main layers (384 experts, 6 a token) or the drafter's (128, 3)."""
    global G, TOPK
    TM.G, TM.TOPK = G, TOPK = g, topk
    set_ep(ep)


DEV = TM.DEV
G, TOPK, H, I = TM.G, TM.TOPK, TM.H, TM.I
set_ep(EP)
TS_NUM = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 36, 48)
TS_GRAPH = (1, 2, 4, 6, 8, 16, 24, 36, 48)


def check_decode():
    ok = True
    g = torch.Generator(device=DEV)
    g.manual_seed(7)
    for K in range(1, 9):
        for mcg, mul1 in ((False, True), (True, False)):
            for k, n in ((5120, 2304), (2304, 5120), (256, 384)):
                t = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * K), dtype=torch.int32, device=DEV,
                                  generator=g).to(torch.int16)
                ref = torch.empty((k, n), dtype=torch.float16, device=DEV)
                ext.reconstruct(ref, t, K, mcg, mul1)
                got = XD.dq_tiles(t, 2 if mul1 else 1)
                same = torch.equal(ref.view(torch.int16), got.view(torch.int16))
                ok &= same
                if not same or (k, n) == (5120, 2304):
                    print(f"  decode K={K} {'mul1' if mul1 else 'mcg '} {k}x{n}: "
                          f"{'bit-identical' if same else 'DIFFERS <-- FAIL'}")
    return ok


# Lazy f32 reference: make_layer keeps each local expert's (trellis, suh, svh) on the host and the reference dequantizes one
# expert at a time (141 MB transient) -- so the numerics run at EP 8 with every local expert (48 main / 16 drafter) under
# 1 GiB. (The eager reference held 141 MB an expert, which pushed earlier runs to R9K_TEST_EP 96 / R9K_TEST_DRAFT_EP 32:
# 4 local experts, a few local picks a call -- the many-group paths of T > 8 never ran against a reference.)
TM.weight_f32 = lambda t, su, sv: (t.cpu(), su.cpu(), sv.cpu())


def lazy_reference(x, w, ids, ref):
    xf = x.float()
    out = torch.zeros((x.shape[0], H), dtype=torch.float32, device=DEV)
    for e in range(E):
        rows, slot = (ids == e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        Wt = {}
        for ck in ("w1", "w3", "w2"):
            t, su, sv = (a.to(DEV) for a in ref[e, ck])
            Wt[ck] = dequant_f32(t, su, sv, k_of(t), True)
        xe = xf[rows]
        gt = torch.nn.functional.silu(xe @ Wt["w1"]).clamp(max=TM.LIMIT)
        u = (xe @ Wt["w3"]).clamp(-TM.LIMIT, TM.LIMIT)
        out.index_add_(0, rows, ((gt * u) @ Wt["w2"]) * w[rows, slot].unsqueeze(1))
        del Wt
    return out


TM.reference = lazy_reference


def run_r9k(layer, x, w, ids, out_f32=True):
    meth = layer.quant_method
    tab = XD.Tables(layer)
    return XD.moe_decode(x, w, ids, layer.exl3_g2l, E, tab, meth.K[0], 2 if meth.mul1 else 1, meth.act_limit, I,
                         out_f32=out_f32)


def routing(T, kind, g):
    if kind == "warm":      # exl3_moe._warm's routing: every token the first TOPK local experts (runs of T rows)
        ids = torch.arange(TOPK, device=DEV, dtype=torch.int32).remainder(E).expand(T, -1).contiguous()
        return torch.full((T, TOPK), 1.0 / TOPK, device=DEV), ids
    if kind == "shared":    # every token picks local experts 0..5 in a rotated order (groups of up to T rows)
        base = torch.arange(TOPK, device=DEV, dtype=torch.int32)
        ids = torch.stack([(base + t) % TOPK for t in range(T)]).contiguous()
        ids[:, -1] += 100   # one non-local pick each
        w = torch.rand((T, TOPK), device=DEV, generator=g) + 0.05
        return (w / w.sum(1, keepdim=True) * 1.5).float(), ids
    if kind == "half":      # half the slots local with repeats, half elsewhere
        ids = torch.stack([torch.randperm(G, generator=torch.Generator().manual_seed(int(t)))[:TOPK]
                           for t in range(T)]).to(DEV, torch.int32)
        ids[:, :3] = torch.stack([torch.randperm(12, generator=torch.Generator().manual_seed(int(t) + 99))[:3]
                                  for t in range(T)]).to(DEV, torch.int32)
        w = torch.rand((T, TOPK), device=DEV, generator=g) + 0.05
        return (w / w.sum(1, keepdim=True) * 1.5).float(), ids
    if kind == "all-local" and E < TOPK:    # fewer local experts than picks: every one of them, the rest elsewhere
        w, ids = TM.routing(T, False, g)
        ids[:, :E] = torch.stack([torch.randperm(E, generator=torch.Generator().manual_seed(int(t) + 7))
                                  for t in range(T)]).to(DEV, torch.int32)
        ids[:, E:] = torch.where(ids[:, E:] < E, ids[:, E:] + E, ids[:, E:])
        return w, ids
    return TM.routing(T, {"random": False, "all-local": True, "none": "none"}[kind], g)


def ref_groups(ids, R):
    """Each local expert's slots in slot order, cut into runs of R; groups ordered by their first slot."""
    flat = ids.reshape(-1).tolist()
    seen, out = {}, []
    for i, gid in enumerate(flat):
        if 0 <= gid < E:
            n = seen.get(gid, 0)
            seen[gid] = n + 1
            if n % R == 0:
                out.append((gid, [i]))
            else:
                k = max(j for j, (e, _) in enumerate(out) if e == gid)
                out[k][1].append(i)
    return out


def check_groups(layer):
    """The kernels' device-side grouping: one trellis pass a group, never one a pick."""
    ok = True
    g = torch.Generator(device=DEV)
    g.manual_seed(42)
    g2l = layer.exl3_g2l
    for T in range(1, 49):                  # every width the tier takes
        for kind in ("random", "half", "shared", "all-local", "none"):
            w, ids = routing(T, kind, g)
            R = XD.rows_for(T)
            got = XD.groups(ids, g2l, E, R)
            want = ref_groups(ids, R)
            nloc = int(((ids >= 0) & (ids < E)).sum())
            npick = sum(len(r) for _, r in got)
            nexp = len({e for e, _ in want})
            good = got == want and npick == nloc
            ok &= good
            if not good or kind in ("random", "half") and T in (6, 24, 48):
                print(f"  groups T={T:2d} {kind:9s}: {nloc:3d} local picks over {nexp:2d} experts -> {len(got):2d} groups "
                      f"(trellis passes; {len(got) - nexp} beyond one an expert, R = {R}) "
                      f"{'== reference' if good else '<-- FAIL ' + str(got[:4]) + ' vs ' + str(want[:4])}")
    return ok


def check_scratch():
    """reserve() footprint for the default decode rows (48) and 24, 32: sized once, no allocator growth."""
    ok = True
    dev = DEV
    for rows in (24, 32, 48):
        XD._bufs.clear()
        torch.cuda.empty_cache()
        r0 = torch.cuda.memory_reserved()
        XD.reserve(dev, rows, TOPK, H, I)
        for t in range(1, rows + 1):        # every call size reuses it
            XD._scratch(dev, t, TOPK, H, I)
        mib = XD.scratch_bytes(dev) / 2**20
        pool = (torch.cuda.memory_reserved() - r0) / 2**20
        good = mib <= 256 and pool <= mib + 4
        ok &= good
        print(f"  reserve({rows} tokens x {TOPK}): {mib:.1f} MiB of scratch, {pool:.1f} MiB taken from the allocator "
              f"{'ok' if good else '<-- FAIL'}")
    XD._bufs.clear()
    torch.cuda.empty_cache()
    return ok


KINDS = ("random", "all-local", "none", "shared", "half", "warm")


def check_numerics(layer, ref, K, Ts=TS_NUM, tol=2e-3, kinds=KINDS, method=True):
    ok = True
    g = torch.Generator(device=DEV)
    g.manual_seed(500 + K)
    worst = 0.0
    for T in Ts:
        for kind in kinds:
            for scale in ((1.0, 6.0) if T in (1, 8) else (1.0,)):
                w, ids = routing(T, kind, g)
                x = (torch.randn((T, H), device=DEV, generator=g) * scale).to(torch.bfloat16)
                r = TM.reference(x, w, ids, ref)
                y = run_r9k(layer, x, w, ids, out_f32=True)
                ma, mr, rr = TM.errs(y, r)
                local = int((ids < E).sum())
                if local == 0:
                    good = bool((y == 0).all())
                    rr = 0.0
                else:
                    good = rr < tol and bool(torch.isfinite(y).all())
                    worst = max(worst, rr)
                ok &= good
                if not good or T in (1, 6, 24, 48) or kind == "shared":
                    print(f"  K={K} T={T:3d} {kind:9s} x*{scale:<3g} local {local:3d} | f32 out: max/|ref| {mr:.3e} "
                          f"rms rel {rr:.3e} {'ok' if good else '<-- FAIL'}")
    # through the method tier (bf16 out), default knobs
    M.DECODE = "r9k"
    for T in ((1, 4, 8, 16, 36, 48) if method else ()):
        for kind in ("random", "all-local"):
            w, ids = routing(T, kind, g)
            x = torch.randn((T, H), device=DEV, generator=g).to(torch.bfloat16)
            y = layer.quant_method.apply(layer, x, w, ids, None, None)
            r = TM.reference(x, w, ids, ref)
            ma, mr, rr = TM.errs(y, r)
            good = rr < 1e-2 and y.dtype == torch.bfloat16
            ok &= good
            print(f"  K={K} T={T:3d} {kind:9s} apply(R9K_EXL3_MOE_DECODE=r9k) bf16: rms rel {rr:.3e} "
                  f"{'ok' if good else '<-- FAIL'}")
    print(f"  K={K}: worst f32 rms rel {worst:.3e} (gate {tol})")
    return ok


def check_graphs(layer, K, Ts=TS_GRAPH):
    ok = True
    g = torch.Generator(device=DEV)
    g.manual_seed(600 + K)
    for T in Ts:
        w, ids = routing(T, "random", g)
        ids[0, 0] = 3
        x = torch.randn((T, H), device=DEV, generator=g).to(torch.bfloat16)
        sx, sw, si = x.clone(), w.clone(), ids.clone()
        for _ in range(2):
            run_r9k(layer, sx, sw, si, out_f32=False)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y1 = run_r9k(layer, sx, sw, si, out_f32=False)
            sy = run_r9k(layer, y1, sw, si, out_f32=False)     # chained: the second layer reads the first's output
        diffs = []
        for trial in range(4):
            w2, ids2 = routing(T, ("random", "all-local", "shared", "half")[trial], g)
            x2 = torch.randn((T, H), device=DEV, generator=g).to(torch.bfloat16)
            sx.copy_(x2)
            sw.copy_(w2)
            si.copy_(ids2)
            graph.replay()
            r1 = sy.clone()
            graph.replay()
            e1 = run_r9k(layer, x2, w2, ids2, out_f32=False)
            eager = run_r9k(layer, e1, w2, ids2, out_f32=False)
            torch.cuda.synchronize()
            diffs.append(max((sy.float() - eager.float()).abs().max().item(), (r1.float() - sy.float()).abs().max().item()))
        good = max(diffs) == 0.0
        ok &= good
        print(f"  K={K} graph T={T}: replay (x2) vs eager, two chained layers, 4 routings: max diff {max(diffs):.3e} "
              f"{'ok' if good else '<-- FAIL'}")
    return ok


def ep_routing(T, nloc, seed, shift=0, nexp=None):
    """T tokens, exactly nloc local picks, the rest elsewhere. nexp None: experts spread over the E local ones
    (rotated by shift); else exactly nexp distinct local experts (each picked at least once, never twice by a token)."""
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)
    ids = (torch.rand((T, G - E), device=DEV, generator=gen).topk(TOPK, dim=1).indices + E).to(torch.int32)
    flat = ids.view(-1)
    pos = torch.randperm(T * TOPK, generator=torch.Generator().manual_seed(seed))[:nloc].tolist()
    if nexp is None:
        for i, p in enumerate(pos):
            flat[p] = (7 * i + 3 + p // TOPK + shift) % E
    else:
        pool = [(shift + 7 * j) % E for j in range(E)]
        pool = list(dict.fromkeys(pool))[:nexp]
        rng = torch.Generator().manual_seed(seed + 1)
        used = {}
        for i, p in enumerate(sorted(pos, key=lambda q: torch.rand((), generator=rng).item())):
            t = p // TOPK
            have = used.setdefault(t, set())
            order = [pool[(i + k) % nexp] for k in range(nexp)]
            e = next((x for x in order if x not in have), order[0])
            have.add(e)
            flat[p] = e
    w = torch.full((T, TOPK), 0.25, device=DEV)
    x = torch.randn((T, H), device=DEV, generator=gen).to(torch.bfloat16)
    return x, w, ids


def graph_us(f, iters=200, reps=10):
    """f(i) for i in 0..reps-1 captured in one graph; us a call."""
    for i in range(reps):
        f(i)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for i in range(reps):
            f(i)
    for _ in range(5):
        gr.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters // reps):
        gr.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (iters // reps) / reps * 1e6


BENCH_CASES = [(1, 0, None), (1, 1, None), (1, 2, None), (2, 1, None), (2, 2, None), (4, 3, None), (4, 3, 3), (6, 4, 4),
               (6, 5, 5),
               (8, 6, None), (16, 12, None), (24, 18, 15), (24, 18, None), (36, 27, 21), (48, 36, 25), (48, 36, 12),
               (48, 36, 36), (1, 6, None), (6, 36, None), (24, 144, None)]
# the drafter (128 experts, top-3, 16 local at EP 8): T = 5 a stream (block 5), ~3T/8 local picks
DRAFT_CASES = [(1, 1, None), (2, 1, None), (4, 2, None), (5, 2, None), (6, 2, None), (10, 4, None), (20, 8, 7),
               (40, 15, 10)]


def bench(layer, K, cases=None):
    """Graph replay of 10 layer calls, each with its own routing over different experts (consecutive calls rotate
    the experts by 5, so a replay touches all of them: weights come from DRAM as in a real step, not the 64 MB
    Infinity Cache). Local picks at EP 8: T = 1 -> 1, T = 6 -> ~4.5 (here 5), T = 24 -> ~18 over ~15 experts,
    T = 36 -> 27 over ~21, T = 48 -> 36 over ~25; plus all-local rows. GB/s = distinct experts x an expert's trellis
    bytes / time (the grouping reads each expert once: check_groups)."""
    meth = layer.quant_method
    mb_e = 3 * H * I * K / 8 / 1e6
    print(f"  K={K}: {mb_e:.2f} MB of trellis an expert")
    print(f"  {'T':>3} {'local':>5} {'experts':>7} | {'r9k us':>7} {'GB/s':>6} | {'mgemv us':>8}")
    rows = []
    sel = os.environ.get("BENCH_T")
    for T, nloc, nx in (cases or BENCH_CASES):
        if sel and str(T) not in sel.split(","):
            continue
        if nx is not None and nx > E or nloc > T * TOPK:
            continue
        calls = [ep_routing(T, nloc, 1000 + T * 7 + nloc + i, shift=5 * i, nexp=nx) for i in range(10)]
        nexp = sum(int(torch.unique(c[2][c[2] < E]).numel()) for c in calls) / len(calls)
        M.DECODE = "r9k"
        us = graph_us(lambda i: meth.apply(layer, *calls[i], None, None))
        if T * TOPK <= M.MGEMV_SLOTS and os.environ.get("BENCH_MGEMV", "1") == "1":
            M.DECODE = "mgemv"
            us_m = graph_us(lambda i: meth.apply(layer, *calls[i], None, None))
        else:
            us_m = float("nan")
        gbs = nexp * mb_e / us * 1e3 if nexp else 0.0
        rows.append((T, nloc, nexp, us, gbs, us_m))
        print(f"  {T:3d} {nloc:5d} {nexp:7.1f} | {us:7.1f} {gbs:6.0f} | {us_m:8.1f}", flush=True)
    M.DECODE = "r9k"
    return rows


def check_server_sequence():
    """The DSpark drafter's experts the way the server builds them (package A2, 2026-11-04): a target layer first (its
    warm-up reserves the process-wide scratch at topk 6), then the drafter's K = 4 layer with the drafter geometry
    exactly -- 128 experts, top-3, 16 local at EP 8 -- and its own warm-up (exl3_moe._warm: every T of the tier, incl.
    36 / 48, with every token on the first 3 local experts), then every width the tier takes (T = 1..48) against the
    lazy f32 reference, the group table at every T, both input-rotation paths (in-block LDS and the k_rot pre-pass),
    and graph replay at the drafter's captured widths (5 a stream: 5 / 10 / 15 / 20 at NSEQ 4, to 40 at NSEQ 8).
    Earlier drafter tests ran at 4 local experts (R9K_TEST_DRAFT_EP 32) with an eager f32 reference and never saw
    16-expert groups against a reference."""
    ok = True
    XD._bufs.clear()
    M._warmed.clear()
    torch.cuda.empty_cache()
    set_geom(384, 6, 96)
    main, _ = TM.make_layer(3, 1)
    set_geom(128, 3, 8)
    layer, ref = TM.make_layer(4, 77)
    print(f"-- drafter geometry, server order: {G} experts, top-{TOPK}, K=4, {E} local; R9K_ROWS {M.R9K_ROWS}, "
          f"scratch {XD.scratch_bytes(DEV) / 2**20:.1f} MiB")
    ok &= check_groups(layer)
    for xr in (XD.XR_T, 0):
        x0, XD.XR_T = XD.XR_T, xr
        try:
            print(f"   input rotation: {'k_rot pre-pass above ' + str(xr) + ' tokens' if xr else 'in every k_gu block'}")
            ok &= check_numerics(layer, ref, 4, Ts=tuple(range(1, 49)), method=(xr == x0))
        finally:
            XD.XR_T = x0
    ok &= check_graphs(layer, 4, Ts=(5, 10, 15, 20, 25, 30, 40, 48))
    del layer, ref, main
    torch.cuda.empty_cache()
    set_geom(384, 6, EP)
    return ok


if __name__ == "__main__":
    print("device", torch.cuda.get_device_name(), "| lib", os.environ.get("R9K_LIB"), "| max slots", XD.max_slots())
    allok = True
    only = os.environ.get("ONLY", "")
    if not only or "dq" in only:
        print("-- trellis decode vs ext.reconstruct")
        allok &= check_decode()
    if not only or "scratch" in only:
        print("-- decode scratch")
        allok &= check_scratch()
    M.DECODE = "r9k"
    synth = [] if only == "real" else [(3, 1), (2, 18), (4, 40)]
    for K, seed in synth:
        if not only or "num" in only or "graph" in only:
            set_ep(EP if K != 4 else max(EP, 16))      # K = 4 at 48 local is 850 MB of trellis: 24 keeps the run < 1 GiB
            layer, ref = TM.make_layer(K, seed)
            print(f"   ({E} local experts, EP {TM.EP})")
            if K == 3 and (not only or "num" in only or "group" in only):
                print("-- device-side grouping (one trellis pass a group)")
                allok &= check_groups(layer)
            if not only or "num" in only:
                print(f"-- numerics K={K} synthetic")
                allok &= check_numerics(layer, ref, K)
            if not only or "graph" in only:
                print(f"-- HIP graph K={K}")
                allok &= check_graphs(layer, K)
            del layer, ref
            torch.cuda.empty_cache()
            set_ep(EP)
        if (not only or "bench" in only) and K in (3, 2):
            set_ep(BENCH_EP)
            wf = TM.weight_f32
            TM.weight_f32 = lambda t, su, sv: torch.zeros(1, device=t.device)    # timing only: no reference
            layer, _ = TM.make_layer(K, seed)
            TM.weight_f32 = wf
            print(f"-- timing K={K}, graph replay, us a layer ({E} local experts)")
            bench(layer, K)
            del layer
            torch.cuda.empty_cache()
            set_ep(EP)
    if not only or "draft" in only:
        allok &= check_server_sequence()
    if not only or "bench" in only:
        set_geom(128, 3, 8)
        wf = TM.weight_f32
        TM.weight_f32 = lambda t, su, sv: torch.zeros(1, device=t.device)
        layer, _ = TM.make_layer(4, 77)
        TM.weight_f32 = wf
        print(f"-- timing, the drafter's experts (K=4, {E} local of {G}, top-{TOPK}), graph replay, us a layer")
        bench(layer, 4, DRAFT_CASES)
        del layer
        torch.cuda.empty_cache()
        set_geom(384, 6, EP)
    real_dir = os.environ.get("EXL3_REAL_DIR", "/ds41/exl3/kerneltest/real")
    if not only or only == "real" or "num" in only:
        for K, f in ((3, "l0-sample.safetensors"), (2, "l18-sample.safetensors")):
            path = os.path.join(real_dir, f)
            if not os.path.exists(path):
                print(f"-- real experts: {path} missing, skipped")
                continue
            layer, ref = TM.make_layer(K, 0 if K == 3 else 18, real=path)
            print(f"-- numerics on real pack experts ({f}, K={K})")
            allok &= check_numerics(layer, ref, K, Ts=(1, 2, 6, 8, 24, 36, 48))
            allok &= check_graphs(layer, K, Ts=(1, 6, 48))
            del layer, ref
            torch.cuda.empty_cache()
    print("ALL OK" if allok else "FAILURES")
    sys.exit(0 if allok else 1)

# Results, VM100 card 5 (R9700 gfx1201, SCLK offset -500 MHz as set on the box), r9700/vllm:exl3, 2026-11-02, ALL OK:
#   decode == ext.reconstruct bit for bit, K 1-8, mul1 and mcg. f32 output vs the f32 reference: rms rel 3.2e-4 - 4.0e-4
#   (K = 2 / 3 / 4 synthetic, real layer-0 K=3 and layer-18 K=2 experts; random, all-local, none-local = exact zeros,
#   shared experts, x*6 clipping; T = 1 .. 32); through apply() in bf16 1.7e-3 (bf16 output rounding). Graph replay ==
#   eager bitwise, two chained layers, T = 1 .. 24. Graph replay, us a layer, cold weights (10 calls over different
#   experts), K=3 / K=2, r9k vs mgemv:
#     T/local picks   1/0        1/1          2/2           4/3           6/4           6/5           8/6
#     r9k K=3         9.3   43.0 (308 GB/s)  67.5 (393)   98.8 (403)   122.8 (432)   149.5 (444)   189.8 (420)
#     mgemv K=3      45.7   93.2            143.9        182.6        265.9         289.9         411.5
#     T/local         16/12          24/18          1/6           6/36           24/144
#     r9k K=3       364.8 (407)    538.3 (372)    172.6 (461)    798.1 (424)    2218.9 (277)
#     mgemv K=3    1000.2           -             201.4          989.1             -
#     r9k K=2: 1/1 39.8, 6/4 110.9, 6/5 141.1, 16/12 350.3, 24/18 520.1 (ALU-bound: K=2 is only ~5% faster than K=3)
#
# Phase 4 package A (2026-11-03, VM100 card 5, r9700/vllm:dev, SCLK offset 0 at the time; ALL OK): decode == reconstruct
# bit for bit K 1-8; device grouping == the reference at T = 1-48 (one trellis pass a group); f32 rms rel <= 4.2e-4 at
# T = 1-48 incl. the drafter geometry (128 experts, top-3, K = 4) and real layer-0 / layer-18 experts; graph replay ==
# eager bitwise to T = 48; scratch at 48 rows 47.5 MiB. Timing, graph replay, us a layer, K = 3 (A4sweep.py: T / local
# picks / distinct experts; before = the v0.4.0 kernel and tier, which sends T = 48 to exllamav3's fused tier):
#     T/picks/experts   6/4/4    6/5/5    24/18/15   36/27/21   48/36/25          48/36/12   48/36/36
#     v0.4.0           123.0    145.6     518.4      795.3     2324.0 (fused)        -          -
#     package A        118-124  141-152   414-435    616-620    674-741 (449 GB/s)   676       990
#   K = 2: 24/18/15 548 -> 369, 36/27/21 871 -> 597, 48/36/25 2309 -> 717; drafter (K = 4, 16 local, top-3): T = 20
#   223 -> 213, T = 40 407 -> 342. Above 8 tokens 2-row groups and 4 gate|up chunks (exl3_decode.rows_for / split_for).
