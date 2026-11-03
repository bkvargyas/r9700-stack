"""Hierarchical 8-rank all-reduce (comm/r9k_ar8.py): correctness and latency vs the flat kernels and RCCL.
    torchrun --nproc-per-node=8 tests/test_ar8.py        (BENCH=0 skips timing; TOKS=1,24x5120,... sets the table's
                                                           sizes, tokens x WIDTH (4096) bf16)
Checks: every rank's output identical (bit-exact across ranks); within 1 bf16 ulp of the fp32 rank-ordered
reference (the pair sum is rounded once before the cross step); graph replay of chained calls, including mixed
sizes that switch between the host stage's one-shot and two-shot; a burst of back-to-back calls of varying size
(ranks drift out of step); every message size routes the same way on every rank (should()/_takes agree);
DeepSeek-V4.1 sizes [tokens, 5120] up to 400 tokens (4 MB: above R9K_AR8_HOST_MAX_KB, so RCCL)."""
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def inp(rank, it, n, dtype, dev):
    g = torch.Generator(device="cpu").manual_seed(1000 * it + rank)
    return torch.randn(n, generator=g, dtype=torch.float32).mul_(3).to(dtype).to(dev)


def ref(world, it, n, dtype, dev):
    acc = torch.zeros(n, dtype=torch.float32, device=dev)
    for q in range(world):
        acc += inp(q, it, n, dtype, dev).float()
    return acc.to(dtype)


def graph_time(fn, reps=50, iters=20):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(iters):
            fn()
    g.replay(); torch.cuda.synchronize()
    best = float("inf")
    for _ in range(int(os.environ.get("TREPS", "3"))):      # best of 3: other jobs on the cards only add time
        dist.barrier()
        t = time.perf_counter()
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / reps / iters * 1e6)
    return best


def _rccl_gather(x, dim, world):
    """vLLM's GroupCoordinator.all_gather on RCCL: all_gather_into_tensor + movedim + reshape."""
    d = dim + x.dim() if dim < 0 else dim
    out = torch.empty((world,) + tuple(x.shape), dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out, x)
    out = out.movedim(0, d)
    shape = list(x.shape)
    shape[d] *= world
    return out.reshape(shape)


def _ag_before(ar8, x, dim):
    """The all-gather as v0.4.0 routed it: pair/cross/pair if _ag_takes, else the flat 8-rank kernel if it fits, else RCCL."""
    if ar8._ag_takes(x, dim):
        on, ar8.ag_os_on = ar8.ag_os_on, False
        try:
            return ar8.all_gather(x, dim)
        finally:
            ar8.ag_os_on = on
    if ar8.flat.should_ag(x, dim):
        return ar8.flat.all_gather(x, dim)
    return None


AG_SHAPES = [((t, 3200), -1) for t in range(1, 7)] + [((t, 16160), -1) for t in (1, 2, 4, 6, 8, 24)] + \
    [((1, 19360), -1), ((4, 19360), -1), ((4, 4096), 0), ((2, 3, 64), 1), ((3, 8), -1)]


def ag_cases(ar8, rank, world, dev):
    bad = 0
    dec = []
    for i, (shp, dim) in enumerate(AG_SHAPES):
        n = 1
        for v in shp:
            n *= v
        x = inp(rank, 11000 + i, n, torch.bfloat16, dev).view(shp)
        want = torch.cat([inp(q, 11000 + i, n, torch.bfloat16, dev).view(shp) for q in range(world)], dim=dim)
        takes = ar8.should_ag(x, dim)
        dec.append((shp, takes, ar8._ag_os_takes(x, dim)))
        got = ar8.all_gather(x, dim) if takes else _rccl_gather(x, dim, world)
        ok = torch.equal(got, want)
        oks = [None] * world
        dist.all_gather_object(oks, ok)
        if rank == 0:
            path = "one-kernel" if ar8._ag_os_takes(x, dim) else ("ar8 pair/cross" if takes else "RCCL")
            print(f"ag8 {tuple(shp)} dim {dim} ({n * 2 >> 10} KiB a rank) via {path}: {'OK' if all(oks) else 'MISMATCH'}",
                  flush=True)
        bad += not all(oks)
    alld = [None] * world
    dist.all_gather_object(alld, dec)
    if any(d != alld[0] for d in alld):
        bad += 1
        if rank == 0:
            print("ag8: ROUTING DISAGREES across ranks", flush=True)
    # graph replay: gathers chained with all-reduces (the decode step's mix), mixed sizes, back to back
    xs = [inp(rank, 11500 + i, t * c, torch.bfloat16, dev).view(t, c) for i, (t, c) in
          enumerate(((4, 16160), (6, 3200), (2, 16160), (1, 3200)))]
    wants = [torch.cat([inp(q, 11500 + i, x.numel(), torch.bfloat16, dev).view(x.shape) for q in range(world)], -1)
             for i, x in enumerate(xs)]
    a = inp(rank, 11600, 6 * 5120, torch.bfloat16, dev)
    for x in xs:
        ar8.all_gather(x, -1)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ys = []
        for x in xs:
            ys.append(ar8.all_gather(x, -1))
            ar8.all_reduce(a)
        y2 = ar8.all_gather(ys[1][:, :3200].contiguous(), -1)
    for _ in range(7):
        g.replay()
    torch.cuda.synchronize()
    ok = all(torch.equal(y, w) for y, w in zip(ys, wants)) and torch.equal(
        y2, torch.cat([wants[1][:, :3200]] * world, -1))
    oks = [None] * world
    dist.all_gather_object(oks, ok)
    bad += not all(oks)
    if rank == 0:
        print(f"ag8 one-kernel graph replay (7 x 9 calls, with all-reduces): {'OK' if all(oks) else 'FAIL'}", flush=True)
    return bad


def ag_bench(ar8, rank, world, dev):
    if rank == 0:
        print(f"{'shape':>14} {'KiB/rank':>8} {'rccl':>8} {'v0.4.0':>8} {'one-kernel':>10}   (us, HIP graphs, best of 3; "
              f"v0.4.0 = pair/cross path or the flat 8-rank kernel; - = RCCL)", flush=True)
    for i, (shp, dim) in enumerate(AG_SHAPES[:12]):
        n = 1
        for v in shp:
            n *= v
        x = inp(rank, 12000 + i, n, torch.bfloat16, dev).view(shp)
        t_r = graph_time(lambda: _rccl_gather(x, dim, world))
        t_b = graph_time(lambda: _ag_before(ar8, x, dim)) if (ar8._ag_takes(x, dim) or ar8.flat.should_ag(x, dim)) \
            else float("nan")
        t_o = graph_time(lambda: ar8.all_gather(x, dim)) if ar8._ag_os_takes(x, dim) else float("nan")
        ts = [None] * world
        dist.all_gather_object(ts, (t_r, t_b, t_o))
        if rank == 0:
            m = [max(t[k] for t in ts) for k in range(3)]
            f = lambda v: f"{v:>8.1f}" if v == v else f"{'-':>8}"
            print(f"{str(tuple(shp)):>14} {n * 2 / 1024:>8.1f} {m[0]:>8.1f} {f(m[1])} {f(m[2]):>10}", flush=True)


def oneshot_cases(ar8, rank, world, dev, check):
    """Package C (2026-11-03): the one-kernel small path (R9K_AR8_ONESHOT) and the fused MoE reduction. Returns the
    number of failures. Bits must equal the stage kernels' (the same message with os_on off), across ranks too."""
    bad = 0

    def stage(x):                     # the same message through the stage kernels (every rank flips together)
        ar8.os_on = False
        try:
            return ar8.all_reduce(x)
        finally:
            ar8.os_on = True

    def same_all(t):
        g = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(g, t)
        return all(torch.equal(g[0], v) for v in g)

    def eq(name, got, want):
        nonlocal bad
        ok = torch.equal(got, want)
        allok = [None] * world
        dist.all_gather_object(allok, ok)
        if rank == 0 and not all(allok):
            print(f"{name}: NOT bit-identical on ranks {[i for i, v in enumerate(allok) if not v]}", flush=True)
        bad += not all(allok)
        return all(allok)

    # routing: every size routes the same way on every rank
    sizes = [8, 16, 24, 56, 4096, 4104, 8192, 12288, 16384, 24576, 32768] + [5120 * t for t in range(1, 7)] + [32760, 32768, 32776]
    dec = [(n, ar8.should(torch.empty(n, dtype=torch.bfloat16, device=dev)) and ar8._takes(torch.empty(n, dtype=torch.bfloat16, device=dev)),
            n * 2 <= ar8.os_max) for n in sizes]
    alld = [None] * world
    dist.all_gather_object(alld, dec)
    if any(d != alld[0] for d in alld):
        bad += 1
        if rank == 0:
            print("oneshot: ROUTING DISAGREES across ranks", flush=True)
    # eager, against the stage kernels (bits) and the fp32 reference (1-2 ulp)
    nos = 0
    for it, n in enumerate(sizes):
        for dt in ((torch.bfloat16, torch.float16, torch.float32) if n in (4096, 30720) else (torch.bfloat16,)):
            if n * torch.empty((), dtype=dt).element_size() % 16:
                continue
            x = inp(rank, 5000 + it, n, dt, dev)
            if not (ar8.should(x) and ar8._takes(x)):
                continue
            nos += x.numel() * x.element_size() <= ar8.os_max
            y = ar8.all_reduce(x)
            ok = eq(f"oneshot n={n} {dt}", y, stage(x))
            if dt == torch.bfloat16 and (n in (4096, 8192, 16384, 30720, 32768)):
                check(f"oneshot eager n={n} ({n * 2 >> 10} KiB) bits=stage:{ok}", y, ref(world, 5000 + it, n, dt, dev), ulps=2)
    if rank == 0:
        print(f"oneshot: {nos} eager messages through the one-kernel path, bit-identical to the stage kernels: "
              f"{'yes' if bad == 0 else 'NO'}", flush=True)
    # burst: no host sync, mixed one-kernel / split / stage sizes (double buffering, parity, shared counters)
    bs = [5120 * t for t in (1, 6, 24, 2, 4, 10, 1, 3, 64, 5, 6, 1, 2)] + [4096, 8, 32768, 16]
    xs = [inp(rank, 6000 + i, n, torch.bfloat16, dev) for i, n in enumerate(bs)]
    torch.cuda.synchronize(); dist.barrier()
    ys = [ar8.all_reduce(x) for x in xs]
    torch.cuda.synchronize()
    for i, (x, y) in enumerate(zip(xs, ys)):
        eq(f"oneshot burst {i} n={bs[i]}", y, stage(x))
    # graph replay of chained one-kernel calls with mixed sizes (and a split call between)
    gx = [inp(rank, 6100 + i, n, torch.bfloat16, dev) for i, n in enumerate((4096, 5120 * 6, 16, 5120 * 2, 5120 * 24))]
    for x in gx:
        ar8.all_reduce(x)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gy = [ar8.all_reduce(x) for x in gx]
        gz = ar8.all_reduce(gy[1].mul(0.125))
        gw = ar8.all_reduce(gz)
    for _ in range(7):
        g.replay()
    torch.cuda.synchronize()
    for i, (x, y) in enumerate(zip(gx, gy)):
        eq(f"oneshot graph {i}", y, stage(x))
    eq("oneshot graph chained", gz, stage(gy[1].mul(0.125)))
    eq("oneshot graph chained x2", gw, stage(gz))
    if rank == 0:
        print(f"oneshot graph replay (7 x 8 calls): {'OK' if bad == 0 else 'FAIL'}", flush=True)

    # ---- the fused MoE reduction against the unfused path (k_dn's per-token sum -> bf16, + shared (bf16 add), AR)
    H, G = 5120, 384
    E = G // world
    nb0 = bad
    for topk, Ge in ((6, 384), (3, 128)):              # main layers: 384 experts, 6 a token; the drafter: 128, 3
        El = Ge // world
        g2l = torch.full((Ge,), -1, dtype=torch.int64, device=dev)
        g2l[rank * El:(rank + 1) * El] = torch.arange(El, dtype=torch.int64, device=dev)
        g2l[(rank * El + El // 2) % Ge] = El          # ">= E" also means not here
        for T in range(1, 49):
            S = T * topk
            gen = torch.Generator(device="cpu").manual_seed(7000 + 100 * topk + T)
            ids = torch.randint(0, Ge, (T, topk), generator=gen, dtype=torch.int32)
            # every rank must agree on the routing (it does in the model): draw it from a seed shared by all ranks;
            # force some local picks on this rank so the sums are not all empty
            if T % 3 == 0:
                ids[0, 0] = rank * El + 1
            ids = ids.to(dev)
            w = (torch.rand((T, topk), generator=torch.Generator(device="cpu").manual_seed(8000 + T)) * 1.5).to(dev)
            osb = inp(rank, 9000 + T, S * H, torch.float32, dev).view(S, H)
            loc = (g2l[ids.long()] >= 0) & (g2l[ids.long()] < El)
            osb[~loc.view(-1)] = float("nan")          # non-local rows are never read
            sh = inp(rank, 9100 + T, T * H, torch.bfloat16, dev).view(T, H)
            if not ar8.moe_takes(sh, topk):
                assert T * H * 2 > ar8.os_max, (T, topk)
                continue
            # unfused: the tier's sum in pick order (multiply, then add: two torch kernels, no contraction)
            acc = torch.zeros((T, H), dtype=torch.float32, device=dev)
            o3 = osb.view(T, topk, H)
            for k in range(topk):
                acc = torch.where(loc[:, k:k + 1], acc + w[:, k:k + 1] * o3[:, k], acc)
            X = sh + acc.to(torch.bfloat16)
            want = ar8.all_reduce(X)
            got = ar8.all_reduce_moe(osb, w.reshape(-1).contiguous(), ids.reshape(-1).contiguous(), g2l, El, sh, topk)
            ok = eq(f"moe fused T={T} topk={topk}", got, want)
            if not ok and rank == 0:
                print(f"  moe T={T} topk={topk}: max|d|={(got.float() - want.float()).abs().max().item():.3g}", flush=True)
    # fused MoE inside a graph, chained with plain calls (the largest T the one-kernel cap takes)
    T, topk = max(1, min(6, ar8.os_max // (H * 2))), 6
    ids = torch.randint(0, G, (T, topk), generator=torch.Generator(device="cpu").manual_seed(4242), dtype=torch.int32).to(dev)
    g2l = torch.full((G,), -1, dtype=torch.int64, device=dev)
    g2l[rank * E:(rank + 1) * E] = torch.arange(E, dtype=torch.int64, device=dev)
    w = torch.rand((T, topk), generator=torch.Generator(device="cpu").manual_seed(4243)).to(dev)
    osb = inp(rank, 4244, T * topk * H, torch.float32, dev).view(T * topk, H)
    sh = inp(rank, 4245, T * H, torch.bfloat16, dev).view(T, H)
    wf, idf = w.reshape(-1).contiguous(), ids.reshape(-1).contiguous()
    e1 = ar8.all_reduce_moe(osb, wf, idf, g2l, E, sh, topk)
    e2 = ar8.all_reduce(e1.mul(0.125))
    torch.cuda.synchronize()
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        r1 = ar8.all_reduce_moe(osb, wf, idf, g2l, E, sh, topk)
        r2 = ar8.all_reduce(r1.mul(0.125))
        r3 = ar8.all_reduce_moe(osb, wf, idf, g2l, E, r2, topk)
    for _ in range(5):
        g2.replay()
    torch.cuda.synchronize()
    eq("moe graph", r1, e1)
    eq("moe graph chained", r2, e2)
    eq("moe graph chained 2", r3, ar8.all_reduce_moe(osb, wf, idf, g2l, E, e2, topk))
    if rank == 0:
        print(f"moe fused reduction (T 1..48, topk 6 / 3, graph replay): bit-identical to the unfused path: "
              f"{'yes' if bad == nb0 else 'NO'} (T <= {ar8.os_max // (H * 2)} fused; above: declined -> the stock add + "
              f"all-reduce)", flush=True)
    return bad


def oneshot_bench(ar8, flat, rank, world, dev):
    """OSBENCH=1: us an all-reduce, HIP graphs, best of 3: RCCL / stage kernels (R9K_AR8_ONESHOT=0) / one kernel;
    and the MoE combine: unfused (shared + y; all-reduce) vs fused."""
    if rank == 0:
        print(f"{'KiB':>6} {'rccl':>8} {'stages':>8} {'oneshot':>8}   (us, HIP graphs, best of 3)", flush=True)
    for spec in os.environ.get("OSTOKS", "1x4096,2x4096,4x4096,6x5120,8x4096,1x5120,2x5120,4x5120").split(","):
        tok, _, wd = spec.partition("x")
        n = int(tok) * int(wd or 4096)
        x = inp(rank, 7700 + n, n, torch.bfloat16, dev)
        t_r = graph_time(lambda: dist.all_reduce(x.clone()))
        ar8.os_on = False
        t_s = graph_time(lambda: ar8.all_reduce(x))
        ar8.os_on = True
        t_o = graph_time(lambda: ar8.all_reduce(x)) if n * 2 <= ar8.os_max else float("nan")
        ts = [None] * world
        dist.all_gather_object(ts, (t_r, t_s, t_o))
        if rank == 0:
            m = [max(t[i] for t in ts) for i in range(3)]
            print(f"{n * 2 / 1024:>6.1f} {m[0]:>8.1f} {m[1]:>8.1f} {m[2]:>8.1f}   ({spec})", flush=True)
    H, G, topk = 5120, 384, 6
    E = G // world
    g2l = torch.full((G,), -1, dtype=torch.int64, device=dev)
    g2l[rank * E:(rank + 1) * E] = torch.arange(E, dtype=torch.int64, device=dev)
    for T in sorted({1, max(1, min(6, ar8.os_max // (H * 2)))}):
        ids = torch.randint(0, G, (T * topk,), generator=torch.Generator(device="cpu").manual_seed(31 + T),
                            dtype=torch.int32).to(dev)
        w = torch.rand((T * topk,), generator=torch.Generator(device="cpu").manual_seed(37 + T)).to(dev)
        osb = inp(rank, 41 + T, T * topk * H, torch.float32, dev).view(T * topk, H)
        sh = inp(rank, 43 + T, T * H, torch.bfloat16, dev).view(T, H)
        y = inp(rank, 47 + T, T * H, torch.bfloat16, dev).view(T, H)
        t_u = graph_time(lambda: ar8.all_reduce(sh + y))
        ar8.os_on = False
        t_u0 = graph_time(lambda: ar8.all_reduce(sh + y))
        ar8.os_on = True
        t_f = graph_time(lambda: ar8.all_reduce_moe(osb, w, ids, g2l, E, sh, topk))
        ts = [None] * world
        dist.all_gather_object(ts, (t_u0, t_u, t_f))
        if rank == 0:
            m = [max(t[i] for t in ts) for i in range(3)]
            print(f"moe combine T={T} ({T * H * 2 >> 10} KiB): add + stage kernels {m[0]:.1f}, add + one kernel {m[1]:.1f}, "
                  f"fused {m[2]:.1f} us", flush=True)
    if os.environ.get("OSSWEEP"):
        # R9K_AR8_OS_WPB / NB / NT and the phase-B transport at the decode sizes
        wpb0, nb0, nt0, xp0 = ar8._os_wpb, ar8._os_nb, ar8._os_nt, ar8.os_xp
        for kb in (8, 16, 32, 60):
            n = kb * 512
            x = inp(rank, 7900 + kb, n, torch.bfloat16, dev)
            for v in os.environ["OSSWEEP"].split(","):
                wpb, nb, nt, xpk = (int(t) for t in v.split("/"))
                ar8._os_wpb, ar8._os_nb, ar8._os_nt = wpb, nb, nt
                ar8.os_xp = xpk * 1024 if ar8._os_args[4] is not None else 0
                y = ar8.all_reduce(x)
                same = [None] * world
                ar8.os_on = False
                yy = ar8.all_reduce(x)
                ar8.os_on = True
                dist.all_gather_object(same, torch.equal(y, yy))
                t = graph_time(lambda: ar8.all_reduce(x))
                ts = [None] * world
                dist.all_gather_object(ts, t)
                if rank == 0:
                    print(f"  sweep {kb:3d} KiB wpb {wpb:4d} nb {nb:2d} nt {nt:4d} xp<= {xpk:3d} KiB: {max(ts):7.1f} us"
                          f"  bits=stage {all(same)}", flush=True)
        ar8._os_wpb, ar8._os_nb, ar8._os_nt, ar8.os_xp = wpb0, nb0, nt0, xp0


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    from r9700_vllm.comm.r9k_ar import R9kAllReduceN
    from r9700_vllm.comm.r9k_ar8 import R9kAllReduce8
    flat = R9kAllReduceN(dist.group.WORLD, dev)
    assert not flat.disabled, "flat N-rank all-reduce unavailable"
    ar8 = R9kAllReduce8(dist.group.WORLD, dev, flat=flat)
    assert not ar8.disabled, "ar8 unavailable"
    # bf16 intermediate: the pair sums are rounded before the cross step -> up to ~2.5 ulp from the once-rounded
    # reference; fp32 intermediate (R9K_AR8_FP32=1): a summation-order difference only -> within 1 ulp.
    TOL = 1 if ar8.fp32 else 2
    if rank == 0:
        print(f"ar8 intermediate: {'fp32' if ar8.fp32 else 'bf16'}, tolerance {TOL} ulp", flush=True)
    bad = 0

    def check(name, got, want, ulps=1):
        nonlocal bad
        # identical across ranks
        gathered = [torch.empty_like(got) for _ in range(world)]
        dist.all_gather(gathered, got)
        same = all(torch.equal(gathered[0], g) for g in gathered)
        # within `ulps` bf16 ulps of the fp32 reference. For the bf16 intermediate the intermediate roundings
        # are at the scale of the PAIR SUMS, so where the total cancels the error can exceed the result's own
        # ulp: measure it against the ulp of the largest magnitude that occurs (8 x max|input|).
        diff = (got.float() - want.float()).abs()
        scale = want.float().abs() if ar8.fp32 else torch.full_like(want.float(), float(world * 12.0))
        ulp = torch.exp2(torch.floor(torch.log2(scale.clamp_(min=2 ** -126))) - 7)
        ok = bool((diff <= ulps * ulp + 1e-6).all())
        if rank == 0:
            print(f"{name}: cross-rank identical={same} within {ulps} ulp={ok} max|diff|={diff.max().item():.3g}", flush=True)
        bad += (not same) + (not ok)

    if os.environ.get("CHECK", "1") == "1":       # CHECK=0: timing only (e.g. against an older ar8)
        for it, n in enumerate((4096, 8192, 4096 * 4, 4096 * 16)):
            x = inp(rank, it, n, torch.bfloat16, dev)
            assert ar8._takes(x), f"ar8 does not take n={n}"
            check(f"eager n={n}", ar8.all_reduce(x), ref(world, it, n, torch.bfloat16, dev), ulps=TOL)
        # DeepSeek-V4.1 decode / verify sizes: [tokens, 5120] bf16, through the host stage's one-shot (<= 96 KB) and
        # two-shot (above) up to R9K_AR8_HOST_MAX_KB; 400 tokens (4 MB) must be declined (RCCL)
        for tok in (1, 9, 10, 24, 64, 100, 200, 400):
            n = tok * 5120
            x = inp(rank, 200 + tok, n, torch.bfloat16, dev)
            takes = ar8.should(x) and ar8._takes(x)
            allt = [None] * world
            dist.all_gather_object(allt, takes)
            if len(set(allt)) != 1:
                bad += 1
                if rank == 0:
                    print(f"ds41 {tok} tok: ROUTING DISAGREES across ranks {allt}", flush=True)
                continue
            if not takes:
                if rank == 0:
                    print(f"ds41 {tok} x 5120 ({n * 2 >> 10} KiB): declined (RCCL)", flush=True)
                continue
            check(f"ds41 eager {tok} x 5120 ({n * 2 >> 10} KiB)", ar8.all_reduce(x), ref(world, 200 + tok, n, torch.bfloat16, dev),
                  ulps=TOL)
        # burst: back-to-back calls of varying size with no host sync in between (double buffering, one-/two-shot pools)
        sizes = [5120 * t for t in (24, 1, 200, 10, 64, 2, 100, 24, 200, 4, 64, 9)]
        outs = [ar8.all_reduce(inp(rank, 600 + i, n, torch.bfloat16, dev)) for i, n in enumerate(sizes)]
        for i, (n, y) in enumerate(zip(sizes, outs)):
            check(f"burst {i} n={n}", y, ref(world, 600 + i, n, torch.bfloat16, dev), ulps=TOL)
        # graph replay of chained calls (decode runs inside HIP graphs)
        xs = [inp(rank, 100 + r, 4096, torch.bfloat16, dev) for r in range(3)]
        for x in xs:
            ar8.all_reduce(x)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            ys = [ar8.all_reduce(x) for x in xs]
            y2 = ar8.all_reduce(ys[0])
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        for r, y in enumerate(ys):
            check(f"graph {r}", y, ref(world, 100 + r, 4096, torch.bfloat16, dev), ulps=TOL)
        check("graph chained", y2, ref(world, 100, 4096, torch.bfloat16, dev).float().mul(world).to(torch.bfloat16), ulps=2 * TOL)
        # the same at DeepSeek-V4.1 sizes, mixed: two-shot / one-shot / two-shot of a different size / chained two-shot
        gsz = [5120 * t for t in (64, 4, 200, 24)]
        xs = [inp(rank, 700 + r, n, torch.bfloat16, dev) for r, n in enumerate(gsz)]
        for x in xs:
            ar8.all_reduce(x)
        torch.cuda.synchronize()
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            ys = [ar8.all_reduce(x) for x in xs]
            y2 = ar8.all_reduce(ys[0].mul(0.125))
        for _ in range(5):
            g2.replay()
        torch.cuda.synchronize()
        for r, (n, y) in enumerate(zip(gsz, ys)):
            check(f"graph ds41 {n // 5120} tok", y, ref(world, 700 + r, n, torch.bfloat16, dev), ulps=TOL)
        # chained: the input is the (identical) all-reduced result / 8 on every rank -> want 8 x that, i.e. the result back
        check("graph ds41 chained", y2, ys[0].float().mul(0.125).to(torch.bfloat16).float().mul(world).to(torch.bfloat16),
              ulps=TOL)
        # all-gather: vLLM concat semantics along dim=-1, TP rank order (logits: [tokens, vocab/8] bf16 per rank)
        os.environ.setdefault("R9K_AG_MAX_KB", "1024")
        for rows, cols in ((1, 19360), (4, 19360), (2, 4096)):
            x = inp(rank, 3000 + rows, rows * cols, torch.bfloat16, dev).view(rows, cols)
            want = torch.cat([inp(q, 3000 + rows, rows * cols, torch.bfloat16, dev).view(rows, cols) for q in range(world)], dim=-1)
            if ar8._ag_takes(x, -1):
                got = ar8.all_gather(x, -1)
                ok = torch.equal(got, want)
                if rank == 0:
                    print(f"ag8 [{rows}, {cols}] -> {tuple(got.shape)}: {'OK' if ok else 'MISMATCH'}", flush=True)
                bad += (not ok)
            elif rank == 0:
                print(f"ag8 [{rows}, {cols}]: not taken (caps: pair ag_max {ar8.pair.ag_max >> 10} KiB)", flush=True)
        if os.environ.get("OS", "1") == "1" and getattr(ar8, "os_on", False):
            bad += oneshot_cases(ar8, rank, world, dev, check)
        if os.environ.get("AG", "1") == "1" and getattr(ar8, "ag_os_on", False):
            bad += ag_cases(ar8, rank, world, dev)
    if rank == 0:
        print("RESULT", "PASS" if bad == 0 else f"FAIL ({bad})", flush=True)
    if os.environ.get("OSBENCH", "0") == "1" and getattr(ar8, "os_on", False):
        oneshot_bench(ar8, flat, rank, world, dev)
    if os.environ.get("AGBENCH", "0") == "1" and getattr(ar8, "ag_os_on", False):
        ag_bench(ar8, rank, world, dev)
    if os.environ.get("BENCH", "1") == "1":
        if rank == 0:
            print(f"{'tokens':>7} {'width':>6} {'KiB':>6} {'rccl':>8} {'flat-1s':>8} {'flat-2s':>8} {'ar8':>8} {'cross':>8}"
                  "  (us, HIP graphs, best of 3; ar8 - = declined -> RCCL; cross = the host stage alone: evens on the whole message, or both quartets on the halves in the split path)", flush=True)
        dflt = "1,2,4,8x5120,9x5120,10x5120,16x5120,24x5120,32x5120,64x5120,100x5120,128x5120,200x5120,400x5120"
        for spec in os.environ.get("TOKS", dflt).split(","):
            tok, _, w = spec.partition("x")
            tok, w = int(tok), int(w or 4096)
            n = w * tok
            x = inp(rank, 500 + tok, n, torch.bfloat16, dev)
            nb = n * 2
            t_rccl = graph_time(lambda: dist.all_reduce(x.clone()))
            t_1s = graph_time(lambda: flat.all_reduce(x, mode=1)) if nb <= flat.max1 else float("nan")
            t_2s = graph_time(lambda: flat.all_reduce(x, mode=2)) if nb <= min(flat.max_bytes, 512 << 10) else float("nan")
            t_8 = graph_time(lambda: ar8.all_reduce(x)) if ar8.should(x) and ar8._takes(x) else float("nan")
            t_c = float("nan")
            if ar8.host_on and getattr(ar8, "split", False) and ar8.split_min < nb <= ar8.host_max:
                off, ln = ar8.pair.chunk(n, 2)
                xh = x[off:off + ln]
                tc = graph_time(lambda: ar8.cross_host.all_reduce(xh))      # both quartets, on the halves
                ts = [None] * world
                dist.all_gather_object(ts, tc)
                t_c = max(ts)
            elif ar8.host_on and hasattr(ar8, "split") and ar8.host_min <= nb <= min(ar8.host_max, ar8.cross_host.max_bytes):
                tc = graph_time((lambda: ar8.cross_host.all_reduce(x)) if ar8.pair_pos == 0 else (lambda: None))
                ts = [None] * world
                dist.all_gather_object(ts, tc)
                t_c = max(t for t in ts if t is not None)
            if rank == 0:
                f = lambda v: f"{v:>8.1f}" if v == v else f"{'-':>8}"
                print(f"{tok:>7} {w:>6} {nb // 1024:>6} {t_rccl:>8.1f} {f(t_1s)} {f(t_2s)} {f(t_8)} {f(t_c)}", flush=True)
        xg = inp(rank, 4000, 19360, torch.bfloat16, dev).view(1, 19360)
        if ar8._ag_takes(xg, -1):
            t_ag_flat = graph_time(lambda: flat.all_gather(xg, -1)) if flat.should_ag(xg, -1) else float("nan")
            t_ag8 = graph_time(lambda: ar8.all_gather(xg, -1))
            out_r = [torch.empty_like(xg) for _ in range(world)]
            t_ag_rccl = graph_time(lambda: dist.all_gather(out_r, xg))
            if rank == 0:
                print(f"all-gather [1, 19360] bf16 (38 KB/rank): rccl {t_ag_rccl:.1f}  flat-ag {t_ag_flat:.1f}  ag8 {t_ag8:.1f} us", flush=True)
    if os.environ.get("PARTS", "0") == "3":
        # stage costs at DeepSeek-V4.1 sizes: pair all-reduce, the evens' host cross stage, both quartets' host cross
        # stages on half the message at once (one per quartet), the pair all-gather of the halves
        for tok in [int(t) for t in os.environ.get("PTOKS", "8,16,24,64,100,200").split(",")]:
            n = tok * 5120
            x = inp(rank, 950 + tok, n, torch.bfloat16, dev)
            h = x[: n // 2].contiguous()
            buf = torch.empty_like(x)
            cap = ar8.cross_host.max_bytes
            rows = [("pair AR", lambda: ar8.pair.all_reduce(x)),
                    ("pair RS", lambda: ar8.pair.reduce_scatter(x, buf)),
                    ("pair AG chunks", lambda: ar8.pair.all_gather_chunks(x, buf)),
                    ("pair RS + AG", lambda: (ar8.pair.reduce_scatter(x, buf), ar8.pair.all_gather_chunks(buf))),
                    ("cross evens S", (lambda: ar8.cross_host.all_reduce(x)) if ar8.pair_pos == 0 and n * 2 <= cap
                     else (lambda: None)),
                    ("cross evens S/2", (lambda: ar8.cross_host.all_reduce(h)) if ar8.pair_pos == 0 else (lambda: None)),
                    ("cross both quartets S/2", lambda: ar8.cross_host.all_reduce(h)),
                    ("ar8", lambda: ar8.all_reduce(x))]
            for name, fn in rows:
                t = graph_time(fn)
                ts = [None] * world
                dist.all_gather_object(ts, t)
                if rank == 0:
                    print(f"  {tok:4d} tok {n * 2 >> 10:5d} KiB  {name:28s} {max(ts):8.1f} us", flush=True)
    if os.environ.get("PARTS", "0") == "5" and ar8.fused:
        # the fused split kernel at DeepSeek-V4.1 verify sizes: (1) the lane / sub-step sweep (K sub-steps of W words
        # a lane per part -> lanes), (2) the stage trace: per role (R0 push to the partner, R1 pair sum -> host,
        # R2 quartet reduce of our part, R3 pull the peers' parts, R4 receive the partner's half) the time its
        # waits end and its work ends, us from the kernel's first block, median of 20 calls, ranks 0 (even) and 1
        L = ar8.pair.L
        khz = L.r9k_wallclock_khz() or 100000
        tw = L.r9k_ar8_fused_trace_words()
        LM = ar8._flanes
        names = ("R0 push", "R1 pair-sum", "R2 reduce", "R3 gather", "R4 receive")
        fk0, fw0 = ar8._fk, ar8._fw
        for tok in [int(t) for t in os.environ.get("PTOKS", "24,48").split(",")]:
            n = tok * 5120
            x = inp(rank, 990 + tok, n, torch.bfloat16, dev)
            want = ref(world, 990 + tok, n, torch.bfloat16, dev)
            for kw in os.environ.get("SWEEP", "8x512,4x256,2x256,1x256,4x128,2x128,1x128,2x64,1x64").split(","):
                k_, w_ = (int(v) for v in kw.split("x"))
                ar8._fk, ar8._fw = k_, w_
                nl, k = ar8.fused_lanes(n * 2)
                y = ar8.all_reduce(x)
                gathered = [torch.empty_like(y) for _ in range(world)]
                dist.all_gather(gathered, y)
                same = all(torch.equal(gathered[0], g_) for g_ in gathered)
                t = graph_time(lambda: ar8.all_reduce(x))
                ts = [None] * world
                dist.all_gather_object(ts, t)
                if rank == 0:
                    print(f"  sweep {tok:3d} tok {n * 2 >> 10:4d} KiB  K={k_:2d} W={w_:4d} -> lanes {nl:2d} sub-steps {k:2d}: "
                          f"{max(ts):7.1f} us  identical={same} max|d|={(y.float() - want.float()).abs().max().item():.3g}",
                          flush=True)
            ar8._fk, ar8._fw = fk0, fw0
            if os.environ.get("TRACE_KW"):
                ar8._fk, ar8._fw = (int(v) for v in os.environ["TRACE_KW"].split("x"))
            nl, k = ar8.fused_lanes(n * 2)
            trs = [torch.zeros(tw, dtype=torch.int64, device=dev) for _ in range(20)]
            for _ in range(3):
                ar8.all_reduce(x)
            torch.cuda.synchronize()
            dist.barrier()
            for tr in trs:
                ar8._split(x.view(-1), trace=tr)
            torch.cuda.synchronize()
            rows = []
            for tr in trs:
                tt = tr.view(5, LM, -1)[:, :nl, : 1 + 2 * k].cpu().double()
                t0 = tt[:, :, 0].min()
                tt = (tt - t0) / (khz / 1000.0)
                rows.append(torch.stack([tt[:, :, 0].min(1).values, tt[:, :, 1].max(1).values,
                                         tt[:, :, 2 * k].max(1).values,
                                         (tt[:, :, 1::2] - torch.cat([tt[:, :, :1], tt[:, :, 2:2 * k:2]], 2)).sum(2).max(1).values],
                                        1))
            med = torch.stack(rows).median(0).values          # [5 roles, (start, first wait end, end, waits)]
            allm = [None] * world
            dist.all_gather_object(allm, med.tolist())
            if rank == 0:
                ar8._fk, ar8._fw = fk0, fw0
                print(f"  trace {tok:3d} tok {n * 2 >> 10:4d} KiB, lanes {nl} x {k} sub-steps (us from the first block; "
                      f"start / first wait done / done / time spent waiting)", flush=True)
                for r_ in (0, 1):
                    for i, nm in enumerate(names):
                        a_ = allm[r_][i]
                        print(f"    rank {r_} {nm:12s} {a_[0]:7.1f} {a_[1]:7.1f} {a_[2]:7.1f}   wait {a_[3]:7.1f}", flush=True)
                print(f"    kernel span (max over ranks of R4 done): {max(m_[4][2] for m_ in allm):7.1f} us", flush=True)
            ar8._fk, ar8._fw = fk0, fw0
    if os.environ.get("PARTS", "0") == "4":
        # the pair stage alone: one-shot / two-shot by size and block count
        for kb in (8, 16, 32, 64, 80, 128, 256, 512):
            n = kb * 512
            x = inp(rank, 970 + kb, n, torch.bfloat16, dev)
            row = []
            for mode, nb in ((1, None), (2, None), (2, 4), (2, 8), (2, 16)):
                if mode == 1 and n * 2 > ar8.pair.max1:
                    row.append(float("nan")); continue
                t = graph_time(lambda: ar8.pair.all_reduce(x, mode=mode, nb=nb))
                ts = [None] * world
                dist.all_gather_object(ts, t)
                row.append(max(ts))
            if rank == 0:
                print(f"  pair {kb:4d} KiB  1s {row[0]:7.1f}  2s {row[1]:7.1f}  2s/nb4 {row[2]:7.1f}  2s/nb8 {row[3]:7.1f}"
                      f"  2s/nb16 {row[4]:7.1f} us", flush=True)
    if os.environ.get("PARTS", "0") in ("1", "2"):
        # component timing: where does the composed cost come from?
        n = 4096
        xb = inp(rank, 900, n, torch.bfloat16, dev)
        xf = xb.float()
        hb, hf = xb[: n // 2].contiguous(), xf[: n // 2].contiguous()
        xb2 = inp(rank, 901, 2 * n, torch.bfloat16, dev)
        zb = torch.zeros_like(xb)
        xb6 = inp(rank, 903, 3 * n // 2, torch.bfloat16, dev)
        rows = [
            ("pair AR bf16 8KB", lambda: ar8.pair.all_reduce(xb)),
            ("pair AR fp32 16KB", lambda: ar8.pair.all_reduce(xf)),
            ("cross AR both quartets bf16 4KB", lambda: ar8.cross.all_reduce(hb)),
            ("cross AR both quartets fp32 8KB", lambda: ar8.cross.all_reduce(hf)),
            ("cross AR evens only bf16 4KB", (lambda: ar8.cross.all_reduce(hb)) if ar8.pair_pos == 0 else (lambda: None)),
            ("cross AR evens only fp32 8KB", (lambda: ar8.cross.all_reduce(hf)) if ar8.pair_pos == 0 else (lambda: None)),
            ("pair AG bf16 4KB", lambda: ar8.pair.all_gather(hb, dim=0)),
            ("casts bf16->fp32->bf16", lambda: xb.float().to(torch.bfloat16)),
            ("ar8 composed", lambda: ar8.all_reduce(xb)),
            ("seq: pair AR -> cross AR (bf16)", lambda: ar8.cross.all_reduce(ar8.pair.all_reduce(xb)[: n // 2] if ar8.pair_pos == 0 else ar8.pair.all_reduce(xb)[n // 2:])),
            ("seq: pair AR -> cross AR, UNRELATED input", lambda: (ar8.pair.all_reduce(xb), ar8.cross.all_reduce(hb))),
            ("seq: pair AR -> add kernel -> cross AR", lambda: ar8.cross.all_reduce((ar8.pair.all_reduce(xb)[: n // 2] + 0) if ar8.pair_pos == 0 else (ar8.pair.all_reduce(xb)[n // 2:] + 0))),
            ("seq: cross AR -> pair AR (bf16)", lambda: (ar8.cross.all_reduce(hb), ar8.pair.all_reduce(xb))),
            ("seq: pair AG -> cross AR", lambda: (ar8.pair.all_gather(hb, dim=0), ar8.cross.all_reduce(hb))),
            ("seq: pair AR -> pair AG", lambda: (ar8.pair.all_reduce(xb), ar8.pair.all_gather(hb, dim=0))),
            ("seq: pair AR x2 -> cross AR", lambda: (ar8.pair.all_reduce(xb), ar8.pair.all_reduce(xb), ar8.cross.all_reduce(hb))),
            ("seq: pair AR -> cross AR -> cross AR", lambda: (ar8.pair.all_reduce(xb), ar8.cross.all_reduce(hb), ar8.cross.all_reduce(hb))),
            ("H2: pair AR -> evens-only cross AR 8KB -> pair AR(odd zeros)", (lambda: ar8.pair.all_reduce(ar8.cross.all_reduce(ar8.pair.all_reduce(xb)))) if ar8.pair_pos == 0 else (lambda: (ar8.pair.all_reduce(xb), ar8.pair.all_reduce(zb)))),
            ("H2a: pair AR -> evens-only cross AR 8KB", lambda: (ar8.cross.all_reduce(ar8.pair.all_reduce(xb)) if ar8.pair_pos == 0 else ar8.pair.all_reduce(xb))),
            ("H2b: evens-only cross AR 8KB -> pair AR", lambda: (ar8.pair.all_reduce(ar8.cross.all_reduce(xb)) if ar8.pair_pos == 0 else ar8.pair.all_reduce(xb))),
            ("H2c: pair AR -> evens-only cross AR 4KB", lambda: (ar8.cross.all_reduce(ar8.pair.all_reduce(xb)[: n // 2]) if ar8.pair_pos == 0 else ar8.pair.all_reduce(xb))),
            ("pair AR 16KB bf16 (2 tok)", lambda: ar8.pair.all_reduce(xb2)),
            ("cross both quartets bf16 8KB (2 tok)", lambda: ar8.cross.all_reduce(xb)),
            ("cross both quartets bf16 6KB", lambda: ar8.cross.all_reduce(xb6)),
        ]
        if os.environ.get("PARTS") == "2":
            rows = [r for r in rows if r[0].startswith("H2") or r[0].startswith("pair AR bf16") or r[0].startswith("ar8 composed")]
        for name, fn in rows:
            t = graph_time(fn)
            ts = [None] * world
            dist.all_gather_object(ts, t)
            if rank == 0:
                print(f"  {name:36s} {max(ts):8.1f} us (max over ranks; rank0 {t:.1f})", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
