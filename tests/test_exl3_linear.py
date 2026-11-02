"""EXL3 linear method (r9700_vllm/quant/exl3.py) on one card: synthetic EXL3 matrices through vLLM's own linear layers.

Plain script (not pytest): ``python3 tests/test_exl3_linear.py [--quick]``, inside the serving image with the exllamav3
wheel installed, e.g. on VM100: ``~/ds41/exl3/gpu-run.sh 6 bash -c 'pip install -q --no-deps /ds41/exl3/dist/*.whl &&
python3 tests/test_exl3_linear.py'``. Exits non-zero on a failed check; the numbers recorded at the bottom.

Synthetic matrices: any int16 trellis is a valid EXL3 encoding (every bit pattern decodes through the codebook), so the
test draws random trellis words, +-1 ``suh`` and signed-scale ``svh`` -- no quantizer needed. Reference: the rotated
decode (ext.reconstruct, plain trellis decode) with both 128-point Hadamards and the sign vectors applied here in f32
(Sylvester H, checked against ext.had_r_128), i.e. LinearEXL3.get_weight_tensor recomputed; y_ref = x @ W in f32.

Checks: (1) reconstruct_f16 == reference W; (2) Replicated / Column / Row layers at M in {1, 8, 64, 4096} for
5120->1280 K5, 1280->32768 K5, 8192->5120 K5, 5120->2304 K3; (3) TP8 simulated (layers built as rank r of 8, the same
full tensors through param.weight_loader): concatenated column outputs and summed row partials == the whole matrix
(wq_b 1280->32768, wo_b 8192->5120, engram.wkv 6144->25600 K4); (4) stacked: fused_wqa_wkv (5120 -> [1280 K5, 512 K6],
different suh), wo_a (eight 4096->1024 slices by index, TP1 and TP8), shared expert gate_up / down at TP8
(dequantized at load); (5) lm_head 5120->129280 K6 at TP8 ranks 0, 3, 7; (6) HIP graph capture of apply straight after
load, M = 1..8, 16, 33, 64, 144 (GEMV, tuned GEMM, reconstruct path), plus a cold-capture negative control;
(7) timings (gemm f32 C vs f16 C, gemm vs reconstruct crossover); (8) two-stream concurrency (informational).
"""
import math
import os
import sys
import time
from unittest import mock

import torch

QUICK = "--quick" in sys.argv
FAIL = []
BF = torch.bfloat16
TOL = 8e-3          # relative Frobenius error vs the f32 reference (f16 activations + f16 rotated input in the kernel)


def check(name, ok, msg=""):
    print(("  ok   " if ok else "  FAIL ") + name + (("  " + msg) if msg else ""), flush=True)
    if not ok:
        FAIL.append(name)


def rel(y, ref):
    y, ref = y.float(), ref.float()
    return ((y - ref).norm() / ref.norm().clamp_min(1e-30)).item()


# ------------------------------------------------------------------------------------------------------ setup
def init_vllm():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    init_distributed_environment(world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:29571", local_rank=0,
                                 backend="nccl")
    ensure_model_parallel_initialized(1, 1)
    import r9700_vllm.quant.exl3 as E
    return E


def make_exl3(k, n, K, seed, dev="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    tr = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * K), generator=g, dtype=torch.int32).to(torch.int16)
    suh = (torch.randint(0, 2, (k,), generator=g) * 2 - 1).to(torch.float16)
    svh = ((torch.randint(0, 2, (n,), generator=g) * 2 - 1) * (0.5 + torch.rand(n, generator=g)) / math.sqrt(k))
    mul1 = torch.tensor(-2082680531, dtype=torch.int32)
    return {"trellis": tr, "suh": suh, "svh": svh.to(torch.float16), "mul1": mul1}


_H = None


def sylvester128(dev):
    global _H
    if _H is None:
        h = torch.ones(1, 1)
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _H = (h / math.sqrt(128)).to(dev)
    return _H


def ref_weight(E, t, n0=0, n1=None):
    """f32 W [k, n1 - n0] = diag(suh) H rot H diag(svh), blockwise 128 (columns n0:n1 on 128-boundaries)."""
    tr = t["trellis"].cuda()
    k, n = tr.shape[0] * 16, tr.shape[1] * 16
    n1 = n if n1 is None else n1
    rot = torch.empty((k, n), dtype=torch.float16, device="cuda")
    E.ext().reconstruct(rot, tr, E.k_of(tr), False, True)
    H = sylvester128("cuda")
    w = rot[:, n0:n1].float()
    del rot
    w = (H @ w.reshape(k // 128, 128, -1)).reshape(k, -1) * t["suh"].cuda().float()[:, None]
    w = (w.reshape(-1, 128) @ H).reshape(k, -1) * t["svh"].cuda().float()[None, n0:n1]
    return w


def load(layer, t, shard_id=None):
    for f, w in t.items():
        p = getattr(layer, f)
        if shard_id is None:
            p.weight_loader(p, w)
        else:
            p.weight_loader(p, w, shard_id)


def finish(layer):
    layer.quant_method.process_weights_after_loading(layer)
    return layer


def tp_patch(rank, size):
    """Build layers as rank `rank` of `size` (linear.py / vocab_parallel_embedding.py read the TP group at __init__)."""
    import vllm.model_executor.layers.linear as L
    import vllm.model_executor.layers.vocab_parallel_embedding as V
    ps = [mock.patch.object(m, f, (lambda v: (lambda *a, **k: v))(val))
          for m in (L, V) for f, val in (("get_tensor_model_parallel_rank", rank),
                                         ("get_tensor_model_parallel_world_size", size)) if hasattr(m, f)]
    return ps


class TP:
    def __init__(self, rank, size):
        self.ps = tp_patch(rank, size)

    def __enter__(self):
        for p in self.ps:
            p.__enter__()

    def __exit__(self, *a):
        for p in reversed(self.ps):
            p.__exit__(*a)


def X(m, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((m, k), generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)


# ------------------------------------------------------------------------------------------------------ tests
def t_had(E):
    print("[had] Sylvester H vs ext.had_r_128", flush=True)
    a = torch.randn(4, 256, device="cuda").half()
    b = torch.empty_like(a)
    E.ext().had_r_128(a, b, None, None, 1.0)
    H = sylvester128("cuda")
    ref = (a.float().reshape(-1, 128) @ H).reshape(4, 256)
    check("had_r_128 == x @ H128 (Sylvester, 1/sqrt(128))", rel(b, ref) < 2e-3, f"rel {rel(b, ref):.2e}")


def t_reconstruct(E, shapes):
    print("[reconstruct] reconstruct_f16 vs f32 reference", flush=True)
    for (k, n, K) in shapes:
        t = make_exl3(k, n, K, seed=k + n + K)
        tr, suh, svh = t["trellis"].cuda(), t["suh"].cuda(), t["svh"].cuda()
        w = E.reconstruct_f16(tr, suh, svh, K, True)
        wf = E.Exl3Part(tr, suh, svh, K, False, True, 0, n).dequant()
        r = ref_weight(E, t)
        check(f"reconstruct {k}->{n} K{K}", w.shape == (k, n) and rel(w, r) < 1e-3 and rel(wf, r) < 3e-3,
              f"rel {rel(w, r):.2e} (fused kernel {rel(wf, r):.2e}) w rms {r.pow(2).mean().sqrt().item():.3e}")


def t_layers(E, cfg, shapes, Ms):
    from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
    print("[layers] Replicated / Column / Row at TP1", flush=True)
    for (k, n, K) in shapes:
        t = make_exl3(k, n, K, seed=k * 3 + n + K)
        r = ref_weight(E, t)
        for cls in (ReplicatedLinear, ColumnParallelLinear, RowParallelLinear):
            layer = finish(_load_new(cls, k, n, cfg, t))
            errs = []
            for m in Ms:
                x = X(m, k, m)
                y = layer.quant_method.apply(layer, x)
                errs.append(rel(y, x.float() @ r))
                assert y.dtype == torch.bfloat16 and y.shape == (m, n)
            check(f"{cls.__name__} {k}->{n} K{K} M={Ms}", max(errs) < TOL, " ".join(f"{e:.1e}" for e in errs))
            del layer


def _load_new(cls, k, n, cfg, t, shard_id=None, **kw):
    layer = cls(k, n, bias=False, quant_config=cfg, params_dtype=BF, prefix=f"t.{cls.__name__}", **kw)
    load(layer, t, shard_id)
    return layer


def t_tp8(E, cfg, Ms):
    from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
    print("[tp8] column concat / row partial sums over 8 simulated ranks", flush=True)
    for cls, (k, n, K) in ((ColumnParallelLinear, (1280, 32768, 5)), (RowParallelLinear, (8192, 5120, 5)),
                           (ColumnParallelLinear, (6144, 25600, 4))):
        t = make_exl3(k, n, K, seed=n + 7)
        r = ref_weight(E, t)
        xs = {m: X(m, k, 100 + m) for m in Ms}
        outs = {m: [] for m in Ms}
        for rank in range(8):
            with TP(rank, 8):
                layer = finish(_load_new(cls, k, n, cfg, t))
            assert layer.tp_rank == rank and layer.tp_size == 8
            for m in Ms:
                x = xs[m] if cls is ColumnParallelLinear else xs[m][:, rank * k // 8:(rank + 1) * k // 8].contiguous()
                outs[m].append(layer.quant_method.apply(layer, x).float())
            del layer
        errs = []
        for m in Ms:
            y = torch.cat(outs[m], 1) if cls is ColumnParallelLinear else torch.stack(outs[m]).sum(0)
            errs.append(rel(y, xs[m].float() @ r))
        check(f"TP8 {cls.__name__} {k}->{n} K{K}", max(errs) < TOL, " ".join(f"{e:.1e}" for e in errs))


def t_stacked(E, cfg, Ms):
    from vllm.model_executor.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear
    print("[stacked] fused_wqa_wkv / wo_a slices / shared expert (dense at TP8)", flush=True)
    # fused_wqa_wkv: 5120 -> [1280 (K5), 512 (K6)], disable_tp (replicated), separate suh per shard
    ta, tb = make_exl3(5120, 1280, 5, seed=11), make_exl3(5120, 512, 6, seed=12)
    assert not torch.equal(ta["suh"], tb["suh"])
    layer = MergedColumnParallelLinear(5120, [1280, 512], bias=False, quant_config=cfg, params_dtype=BF,
                                       prefix="t.fused_wqa_wkv", disable_tp=True)
    load(layer, tb, 1)
    load(layer, ta, 0)
    finish(layer)
    r = torch.cat([ref_weight(E, ta), ref_weight(E, tb)], 1)
    errs = [rel(layer.quant_method.apply(layer, X(m, 5120, m)), X(m, 5120, m).float() @ r) for m in Ms]
    check("fused_wqa_wkv two shards (K5 + K6)", len(layer.exl3_parts) == 2 and max(errs) < TOL,
          " ".join(f"{e:.1e}" for e in errs))
    fused = [layer]
    # the common case (39 of 40 layers, compressor): equal K -> one exl3_mgemm for both shards
    tc = make_exl3(5120, 512, 5, seed=13)
    layer = MergedColumnParallelLinear(5120, [1280, 512], bias=False, quant_config=cfg, params_dtype=BF,
                                       prefix="t.fused_wqa_wkv_k5", disable_tp=True)
    load(layer, ta, 0)
    load(layer, tc, 1)
    finish(layer)
    r = torch.cat([ref_weight(E, ta), ref_weight(E, tc)], 1)
    errs = [rel(layer.quant_method.apply(layer, X(m, 5120, m)), X(m, 5120, m).float() @ r) for m in Ms]
    check("fused_wqa_wkv two shards (K5 + K5, multi-matrix GEMV)", layer.exl3_mg is not None and max(errs) < TOL,
          " ".join(f"{e:.1e}" for e in errs))
    fused.append(layer)

    # wo_a: plain column 4096 -> 8192 from eight 4096 -> 1024 slices delivered by index
    sl = [make_exl3(4096, 1024, 5, seed=40 + g) for g in range(8)]

    def wo_a():
        return ColumnParallelLinear(4096, 8192, bias=False, quant_config=cfg, params_dtype=BF, prefix="t.wo_a")
    rs = [ref_weight(E, s) for s in sl]
    layer = wo_a()
    for g in range(8):
        load(layer, sl[g], g)
    finish(layer)
    x = X(4, 4096, 3)
    check("wo_a TP1 eight slices", len(layer.exl3_parts) == 8 and rel(layer.quant_method.apply(layer, x),
                                                                       x.float() @ torch.cat(rs, 1)) < TOL)
    ok = True
    for rank in (0, 5, 7):
        with TP(rank, 8):
            layer = wo_a()
        for g in range(8):
            load(layer, sl[g], g)
        finish(layer)
        e = rel(layer.quant_method.apply(layer, x), x.float() @ rs[rank])
        ok &= len(layer.exl3_parts) == 1 and e < TOL
    check("wo_a TP8 rank r keeps slice r", ok)

    # shared expert at TP8: gate_up [2304, 2304] merged, down 2304 -> 5120 row; 288 a rank -> dequantized at load
    tg, tu, td = make_exl3(5120, 2304, 5, 21), make_exl3(5120, 2304, 5, 22), make_exl3(2304, 5120, 5, 23)
    rg, ru, rd = ref_weight(E, tg), ref_weight(E, tu), ref_weight(E, td)
    x = X(8, 5120, 9)
    hsum, ok = 0, True
    for rank in range(8):
        with TP(rank, 8):
            gu = MergedColumnParallelLinear(5120, [2304, 2304], bias=False, quant_config=cfg, params_dtype=BF,
                                            prefix="t.gate_up_proj")
            dn = RowParallelLinear(2304, 5120, bias=False, quant_config=cfg, params_dtype=BF, prefix="t.down_proj")
        load(gu, tg, 0)
        load(gu, tu, 1)
        load(dn, td)
        finish(gu)
        finish(dn)
        ok &= gu.exl3_dense and dn.exl3_dense
        ok &= tuple(gu.weight.shape) == (576, 5120) and tuple(dn.weight.shape) == (5120, 288)
        s = slice(rank * 288, (rank + 1) * 288)
        y = gu.quant_method.apply(gu, x).float()
        ok &= rel(y, x.float() @ torch.cat([rg[:, s], ru[:, s]], 1)) < TOL
        h = X(8, 288, 50 + rank)
        hsum = hsum + dn.quant_method.apply(dn, h).float()
        ok &= rel(dn.quant_method.apply(dn, h), h.float() @ rd[s]) < TOL
        if rank == 0:
            hall = h
        else:
            hall = torch.cat([hall, h], 1)
    check("shared expert TP8 (dense): gate_up slices + down partial sums", ok and rel(hsum, hall.float() @ rd) < TOL)
    return fused


def t_lm_head(E, cfg):
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    print("[lm_head] 5120 -> 129280 K6 dequantized at load, TP8 ranks 0/3/7", flush=True)
    t = make_exl3(5120, 129280, 6, seed=77)
    ok = True
    for rank in (0, 3, 7):
        with TP(rank, 8), torch.device("cuda"):            # vLLM builds the model under the target device
            head = ParallelLMHead(129280, 5120, params_dtype=BF, quant_config=cfg, prefix="lm_head")
        assert isinstance(head.quant_method, E.Exl3LMHeadMethod)
        load(head, t)
        head.quant_method.process_weights_after_loading(head)
        s0, s1 = head.shard_indices.org_vocab_start_index, head.shard_indices.org_vocab_end_index
        b0, b1 = s0 // 128 * 128, min(-(-s1 // 128) * 128, 129280)
        r = ref_weight(E, t, b0, b1)[:, s0 - b0:s1 - b0].t()
        w = head.weight
        e = rel(w[:s1 - s0], r)
        pad_ok = bool((w[s1 - s0:] == 0).all())
        ok &= w.dtype == torch.bfloat16 and w.shape[0] == head.num_embeddings_per_partition and e < 5e-3 and pad_ok
        h = X(2, 5120, 1)
        lg = head.quant_method.apply(head, h)
        ok &= rel(lg[:, :s1 - s0], h.float() @ r.t()) < TOL
        print(f"    rank {rank}: rows {s0}:{s1} slab {b0}:{b1} rel {e:.1e} weight {tuple(w.shape)}", flush=True)
        del head
    check("lm_head TP8 shards == reference rows, padding zero, stock apply", ok)


def t_graph(E, cfg, fused):
    """Capture straight after load (no eager call first): process_weights_after_loading's warm-up must have tuned every
    exl3_gemm bucket, else the capture would run the autotuner (timed launches + host sync) inside the graph."""
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    print("[graph] HIP graph capture of apply (shape never run before this layer's load)", flush=True)
    t = make_exl3(2048, 24576, 4, seed=5)
    with TP(2, 8):
        fresh = finish(_load_new(ColumnParallelLinear, 2048, 24576, cfg, t))
    for layer, k in [(fresh, 2048)] + [(f, 5120) for f in fused]:
        ok = True
        for m in list(range(1, 9)) + [16, 24, 32, 33, 64, 144, 512, 1100]:
            xs = X(m, k, 1000 + m)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                yg = layer.quant_method.apply(layer, xs)
            for seed in (1, 2):
                xs.copy_(X(m, k, 2000 + 10 * m + seed))
                g.replay()
                torch.cuda.synchronize()
                ok &= torch.equal(yg, layer.quant_method.apply(layer, xs))
        check(f"graph replay == eager (bit-exact) {layer.prefix} M=1..8,16,24,32,33,64,144,512,1100", ok)


def t_graph_unwarmed(E):
    """Negative control (informational): why apply never sends 9+ rows to one exl3_gemm -- a single 32-row call on a
    never-tuned shape takes the autotuned WMMA GEMM, whose first call host-syncs inside the capture."""
    t = make_exl3(3072, 1536, 4, seed=6)
    p = E.Exl3Part(t["trellis"].cuda(), t["suh"].cuda(), t["svh"].cuda(), 4, False, True, 0, 1536)
    a = X(32, 3072, 1).half()
    ah = torch.empty_like(a)
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            y = torch.empty((32, 1536), dtype=torch.float32, device="cuda")
            E.ext().exl3_gemm(a, p.trellis, y, p.suh, ah, p.svh, -1, False, True, 0)
        g.replay()
        torch.cuda.synchronize()
        e = rel(y, a.float() @ ref_weight(E, t))
        print(f"    cold capture at M=32 went through; replay rel {e:.1e}", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"    cold capture at M=32 failed: {type(ex).__name__}: {str(ex)[:200]}", flush=True)


def bench(fn, it=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(it):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / it


def bench_graph(fn, calls=10, it=20):
    """us a call with `calls` calls captured in one HIP graph (the GPU cost a graphed decode step sees)."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(it):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    del g
    return a.elapsed_time(b) * 1000 / (it * calls)


def t_timing(E, shapes):
    """apply's paths under a HIP graph: GEMV in 8-row chunks (f32 C) at M <= 64 and the prefill path at M > 8 (plain
    rotated-basis reconstruct + Hadamards, 1024-row matmul chunks), us a call."""
    print("[timing] graph, us a call: gemv8 (8-row GEMV chunks) | recon (plain reconstruct + had + chunked mm)",
          flush=True)
    for (k, n, K) in shapes:
        t = make_exl3(k, n, K, seed=1)
        p = E.Exl3Part(t["trellis"].cuda(), t["suh"].cuda(), t["svh"].cuda(), K, False, True, 0, n)
        cells = []
        for m in (1, 8, 16, 24, 32, 64, 4096):
            a = X(m, k, m).half()
            ah = torch.empty_like(a)
            g = bench_graph(lambda: p.gemm(a, ah)) if m <= 64 else float("nan")
            r = bench_graph(lambda: p.recon_mm(a), calls=4 if m > 64 else 10) if m > 8 else float("nan")
            cells.append(f"M={m} {g:.1f}|{r:.0f}")
        gbs = k * n * K / 8 / (float(cells[0].split()[1].split("|")[0]) * 1e-6) / 1e9
        print(f"    {k:5d}->{n:5d} K{K}: " + "  ".join(cells) + f"  (M=1 {gbs:.0f} GB/s trellis)", flush=True)


def t_fused(E, shapes):
    """Prefill crossover plain vs fused reconstruct (both with 1024-row matmul chunks), graph, us a call."""
    print("[fused] graph, us a call: plain | fused", flush=True)
    keep = E.FUSED_M, E.FUSED_M_WIDE
    for (k, n, K) in shapes:
        t = make_exl3(k, n, K, seed=1)
        p = E.Exl3Part(t["trellis"].cuda(), t["suh"].cuda(), t["svh"].cuda(), K, False, True, 0, n)
        cells = []
        for m in (128, 256, 384, 512, 1024, 2048, 4096):
            a = X(m, k, m).half()
            E.FUSED_M = E.FUSED_M_WIDE = 1 << 30
            pl = bench_graph(lambda: p.recon_mm(a), calls=2, it=10)
            E.FUSED_M = E.FUSED_M_WIDE = 0
            fu = bench_graph(lambda: p.recon_mm(a), calls=2, it=10)
            cells.append(f"M={m} {pl:.0f}|{fu:.0f}")
        print(f"    {k:5d}->{n:5d} K{K}: " + "  ".join(cells), flush=True)
    E.FUSED_M, E.FUSED_M_WIDE = keep


def t_streams(E):
    """exllamav3's RDNA multi-row GEMV republishes its pointers (B, C, A_had, svh) and arrival counters through ONE
    per-device parameter block (rocm/quant/exl3_gemv_multirow_rdna.cu, Exl3GemvMrParams), so two EXL3 GEMVs in flight on
    two streams can read each other's pointers. Informational: count wrong results under deliberate overlap."""
    print("[streams] EXL3 GEMVs on two streams at once", flush=True)
    ta, tb = make_exl3(5120, 1280, 5, 1), make_exl3(1280, 4096, 5, 2)
    pa = E.Exl3Part(ta["trellis"].cuda(), ta["suh"].cuda(), ta["svh"].cuda(), 5, False, True, 0, 1280)
    pb = E.Exl3Part(tb["trellis"].cuda(), tb["suh"].cuda(), tb["svh"].cuda(), 5, False, True, 0, 4096)
    res = {}
    for m in (1, 4):
        a, b = X(m, 5120, 1).half(), X(m, 1280, 2).half()
        ra, rb = pa.gemm(a, torch.empty_like(a)), pb.gemm(b, torch.empty_like(b))
        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
        bad = tot = 0
        for _ in range(100):
            s1.wait_stream(torch.cuda.current_stream())
            s2.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s1):
                ya = [pa.gemm(a, torch.empty_like(a)) for _ in range(16)]
            with torch.cuda.stream(s2):
                yb = [pb.gemm(b, torch.empty_like(b)) for _ in range(16)]
            torch.cuda.synchronize()
            bad += sum(not torch.equal(y, ra) for y in ya) + sum(not torch.equal(y, rb) for y in yb)
            tot += 32
        res[m] = (bad, tot)
        print(f"    M={m}: {bad} of {tot} concurrent results differ from the serial result", flush=True)
    return res


def t_dummy(E, cfg):
    """R9K_EXL3_DUMMY=1: layers that loaded nothing get random matrices of their real geometry (no assert)."""
    from vllm.model_executor.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear
    print("[dummy] R9K_EXL3_DUMMY=1 with no loads", flush=True)
    os.environ["R9K_EXL3_DUMMY"] = "1"
    try:
        ok = True
        with TP(3, 8):
            ls = [ColumnParallelLinear(1280, 32768, bias=False, quant_config=cfg, params_dtype=BF, prefix="d.wq_b"),
                  MergedColumnParallelLinear(5120, [1280, 512], bias=False, quant_config=cfg, params_dtype=BF,
                                             prefix="d.fused_wqa_wkv", disable_tp=True),
                  RowParallelLinear(8192, 5120, bias=False, quant_config=cfg, params_dtype=BF, prefix="d.wo_b"),
                  MergedColumnParallelLinear(5120, [2304, 2304], bias=False, quant_config=cfg, params_dtype=BF,
                                             prefix="d.gate_up"),
                  RowParallelLinear(2304, 5120, bias=False, quant_config=cfg, params_dtype=BF, prefix="d.down")]
        for layer in ls:
            finish(layer)
            for m in (1, 24):
                y = layer.quant_method.apply(layer, X(m, layer.exl3_k, m))
                ok &= tuple(y.shape) == (m, layer.exl3_n) and bool(torch.isfinite(y).all())
        ok &= len(ls[1].exl3_parts) == 2 and ls[3].exl3_dense and ls[4].exl3_dense
    finally:
        del os.environ["R9K_EXL3_DUMMY"]
    layer = ColumnParallelLinear(1280, 4096, bias=False, quant_config=cfg, params_dtype=BF, prefix="d.real")
    try:
        finish(layer)
        ok = False
    except AssertionError:
        pass
    check("dummy mode: wq_b / fused_wqa_wkv / wo_b / dense shared expert finite, right shapes; no env -> assert", ok)


def main():
    E = init_vllm()
    cfg = E.Exl3Config.from_config({"quant_method": "exl3", "version": "1.6.2", "avg_bits": 2.9, "head_bits": 6,
                                    "codebook": "mul1", "out_scales": "always"})
    shapes = [(5120, 1280, 5), (1280, 32768, 5), (8192, 5120, 5), (5120, 2304, 3)]
    Ms = [1, 8, 13, 32, 64, 1100, 4096]
    print(f"device {torch.cuda.get_device_name()}  RECON_M {E.RECON_M}", flush=True)
    t0 = time.time()
    t_had(E)
    t_reconstruct(E, shapes + [(512, 128, 8), (6144, 25600, 4)])
    t_layers(E, cfg, shapes, Ms)
    t_tp8(E, cfg, Ms)
    fused = t_stacked(E, cfg, Ms)
    t_lm_head(E, cfg)
    t_dummy(E, cfg)
    t_graph(E, cfg, fused)
    if not QUICK:
        t_timing(E, [(5120, 1280, 5), (1280, 4096, 5), (1024, 5120, 5), (4096, 1024, 5), (5120, 512, 5)])
        t_fused(E, [(5120, 1280, 5), (5120, 2304, 3), (2304, 5120, 3), (1280, 4096, 5), (1280, 32768, 5),
                    (8192, 5120, 5)])
        t_streams(E)
        print("[graph-cold] negative control", flush=True)
        t_graph_unwarmed(E)
    print(f"done in {time.time() - t0:.0f} s; {len(FAIL)} failed: {FAIL}", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    main()


# ------------------------------------------------------------------------------------------------------ numbers
# VM100 card 6, r9700/vllm:exl3, 2026-11-02, after K's thresholds (RECON_M 32, 8-row GEMV chunks, FUSED_M 1024 /
# FUSED_M_WIDE 384, 1024-row matmul chunks); 29 checks pass, 13 s:
#   reconstruct_f16 (f32 rotations) vs reference: 2.1e-4 relative; fused reconstruct_had_slice kernel: 9.1e-4.
#   apply (bf16 output) vs x @ W in f32, every layer type / shape, M = 1, 8, 13, 32 (GEMV chunks), 64, 1100 (plain
#     reconstruct), 4096 (fused): 1.7e-3 / 1.7e-3 / 1.9e-3 -- the bf16 output rounding dominates (K measures 2-3e-4
#     on the f32/f16 C before the cast). Same for TP8 column concat / row partial sums, fused_wqa_wkv (K5 + K6),
#     wo_a slices, TP8 shared expert (dense). lm_head rows (bf16): 1.7e-3.
#   HIP graph: capture straight after load replays bit-identical to eager at M = 1..8, 16, 24, 32, 33, 64, 144, 512,
#     1100. Warm-up = one eager exl3_gemm per part at 1..8 rows (asserting the GEMV took it: tags 90-92). Negative
#     control: one 32-row exl3_gemm on a cold shape (the autotuned GEMM) invalidates the capture.
#   Two streams: 0-36 of 3200 concurrent GEMV results wrong (varies by run) -- shared per-device GEMV parameter block.
#   graph, us a call: 8-row GEMV chunks (f32 C) | plain reconstruct + had_r_128 + chunked matmul
#                    M=1    M=8    M=16       M=24       M=32       M=64        M=4096
#     5120->1280 K5  14.3   21.4   41.7 | 53  61.3 | 51  79.4 | 51  157 | 52    - | 628
#     1280->4096 K5  12.2   18.2   35.6 | 39  51.8 | 38  67.9 | 38  134 | 68    - | 450
#     1024->5120 K5  12.1   18.0   35.4 | 35  51.2 | 34  67.6 | 34  133 | 35    - | 473
#     4096->1024 K5  11.4   16.9   33.3 | 44  47.4 | 42  63.0 | 42  125 | 69    - | 572
#     5120->512  K5   9.6   14.8   29.1 | 43  41.1 | 44  54.6 | 44  109 | 44    - | 249
#     (M=1: 170-290 GB/s of trellis, cache-assisted.) On these <= 4 MB TP8 shards reconstruct already wins at 24
#     rows; RECON_M stays 32 (owner's default; R9K_EXL3_RECON_M=16 is the per-shape optimum here).
#   prefill crossover, graph, us a call, plain | fused (both with 1024-row matmul chunks):
#                     M=256      M=384      M=512      M=1024       M=2048       M=4096
#     5120->1280 K5   69 | 82    86 | 96    105 | 112  189 | 177    393 | 326    844 | 639
#     5120->2304 K3  103 | 136  123 | 152   162 | 191  298 | 316    654 | 619   1417 | 1225
#     2304->5120 K3  104 | 134  133 | 163   172 | 205  309 | 339    661 | 661   1445 | 1268
#     1280->4096 K5   99 | 108   83 | 90    101 | 107  138 | 127    294 | 236    644 | 483
#     1280->32768 K5 448 | 455  606 | 525   815 | 657  2073 | 1044  5591 | 1911 16155 | 3527
#     8192->5120 K5  377 | 481  446 | 542   539 | 663  1179 | 1192  2532 | 2052  5395 | 3961
#     -> fused from 1024 rows (even at 1024, fused at 2048+), from 384 rows on 16384+ wide outputs (had_r_128 over the
#     wide output is the cost). 5120->1280 at 4096 rows: 628 us now vs 1604 unchunked / fused before.
#   Phase 2 (2026-11-02, card 3): static input / scratch / C buffers per layer, one output cast, exl3_mgemm for stacked
#     layers with equal K. Launches per apply and graph us per apply, before -> after:
#       Replicated 5120->1280      M=1 4 -> 4 (22.9 -> 22.9 us)   M=6 4 -> 4 (29.7 -> 29.8)
#       Column wq_b TP8 1280->4096 M=1 4 -> 4 (21.0 -> 20.9)      M=6 4 -> 4 (26.2 -> 26.3)
#       fused_wqa_wkv K5+K5        M=1 7 -> 4 (~35 -> 27.4)       M=6 7 -> 5 (~46 -> 38.0)
#       fused_wqa_wkv K5+K6 (L39)  M=1 7 -> 6 (35.5 -> 33.1)      M=6 7 -> 7 (46.1 -> 46.4)
