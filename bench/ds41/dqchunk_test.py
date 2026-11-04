#!/usr/bin/env python3
"""dqchunk_test.py: single-card check of the reconstruct tier's row chunking (quant/exl3_moe.py _expert_dq, package E,
v0.4.2) against the unchunked v0.4.1 code, on synthetic EXL3 experts (random trellis, sign / scale vectors as the
kernel gates use). Bitwise comparison of the expert output at 1 / 1024 / 4096 / 12288 rows for K = 2 / 3 / 4 (both the
deterministic out= path and the atomic index_add path) plus 4,095 / 4,096 / 5,000 / 8,192 / 12,289 (piece edges), the
hipBLAS premise itself (every row of one M-row hgemm vs 4,096-row pieces, M up to 24,576, both shapes), one full apply()
above FUSED_ROWS with one heavy expert and random routing (old module vs new), a HIP-graph replay of the pieced call vs
eager, and timing. Needs the old file as
r9700_vllm/quant/exl3_moe_old.py (v0.4.1's exl3_moe.py) in the test tree. Run inside the serving image on one card:
  HIP_VISIBLE_DEVICES=7 PYTHONPATH=/opt/r9700 R9K_EXL3_MOE_DECODE=mgemv python3 bench/ds41/dqchunk_test.py
Peak VRAM ~0.8 GiB (the unchunked 12,288-row reference)."""
import math, os, sys, time, types
os.environ.setdefault("R9K_EXL3_MOE_DECODE", "mgemv")
import torch
from r9700_vllm.quant import exl3_moe as NEW, exl3_moe_old as OLD
from r9700_vllm.quant.exl3 import ext

DEV = torch.device("cuda:0")
H, I, LIMIT = 5120, 2304, 10.0
X = ext()


def synth(k, n, K, seed):
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    tr = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * K), dtype=torch.int32, device=DEV, generator=g).to(torch.int16)
    suh = (torch.randint(0, 2, (k,), device=DEV, generator=g) * 2 - 1).half()
    svh = ((torch.randint(0, 2, (n,), device=DEV, generator=g) * 2 - 1).float()
           * (0.5 + torch.rand((n,), device=DEV, generator=g)) / math.sqrt(k)).half()
    return tr, suh, svh


def make(mod, K, E, topk, C):
    """A layer + method of module mod as process_weights_after_loading leaves them (C: fused-tier concurrency)."""
    layer = torch.nn.Module()
    layer.layer_name = "model.layers.5.ffn.experts"
    for r, (k, n) in enumerate(((H, I), (H, I), (I, H))):
        ws = [synth(k, n, K, seed=1000 * K + 3 * e + r) for e in range(E)]
        for f, i in (("trellis", 0), ("suh", 1), ("svh", 2)):
            t = torch.stack([w[i] for w in ws])
            layer.register_buffer(f"exl3_{mod.ROLES[r]}_{f}", t, persistent=False)
            step = t[0].numel() * t.element_size()
            layer.register_buffer(f"exl3_{mod.ROLES[r]}_{f}_ptrs",
                                  t.data_ptr() + torch.arange(E, dtype=torch.int64, device=DEV) * step, persistent=False)
    layer._exl3 = types.SimpleNamespace(E=E, H=H, I=I, device=DEV)
    layer.exl3_g2l = torch.arange(E, dtype=torch.int64, device=DEV)
    layer._exl3_r9k = False
    m = object.__new__(mod.Exl3MoEMethod)
    m.moe = types.SimpleNamespace(experts_per_token=topk, num_experts=E)
    m.K, m.mul1, m.mcg, m.act_limit = [K] * 3, True, False, LIMIT
    R = mod.FUSED_ROWS
    m.fused = tuple(torch.empty((C, R, d), dtype=torch.float16, device=DEV) for d in (H, H, I, I))
    m.dq_temp = torch.empty((H * I,), dtype=torch.float16, device=DEV)
    return layer, m


def old_expert(mo, layer, e, x16, tok, out):
    mo._expert_dq(layer, e, x16.index_select(0, tok), out)


def same(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def premise():
    """hgemm: every row of the M-row product == the same rows from DQ_ROWS-row pieces (last piece overlapping)."""
    ok, P = True, NEW.DQ_ROWS
    for k, n in ((H, I), (I, H)):
        w = (torch.randn((k, n), device=DEV) * 0.02).half()
        a = torch.randn((24576, k), device=DEV).half()
        res = []
        for M in (4096, 4097, 5000, 6000, 7777, 8191, 8192, 10000, 12288, 12289, 16384, 20000, 24576):
            full = torch.empty((M, n), dtype=torch.float32, device=DEV); X.hgemm(a[:M], w, full)
            pc = torch.empty((P, n), dtype=torch.float32, device=DEV)
            good = True
            for r0 in list(range(0, M - P, P)) + [M - P]:
                X.hgemm(a[r0:r0 + P], w, pc)
                good &= torch.equal(pc.view(torch.int32), full[r0:r0 + P].view(torch.int32))
            res.append(f"{M}:{'=' if good else 'DIFF'}"); ok &= good
            del full, pc
        print(f"premise hgemm {k}x{n}, M-row product vs {P}-row pieces: " + " ".join(res))
        del w, a; torch.cuda.empty_cache()
    return ok


def main():
    torch.manual_seed(0)
    print(f"device {torch.cuda.get_device_name()}  DQ_ROWS {NEW.DQ_ROWS}")
    ok = premise()
    for K in (2, 3, 4):
        layer, mn = make(NEW, K, 2, 2, 2)
        mo = object.__new__(OLD.Exl3MoEMethod); mo.__dict__.update(mn.__dict__)
        for c in (1, 1024, 4095, 4096, 5000, 8192, 12288, 12289):
            x16 = (torch.randn((c, H), device=DEV) * 0.5).half()
            tok = torch.randperm(c, device=DEV)
            ref = torch.empty((c, H), dtype=torch.float32, device=DEV)
            old_expert(mo, layer, 1, x16, tok, ref); torch.cuda.synchronize()
            ref_h = ref.cpu(); del ref; torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
            new = torch.full((c, H), float("nan"), dtype=torch.float32, device=DEV)
            mn._expert_dq(layer, 1, x16, tok, new); torch.cuda.synchronize()
            trans = (torch.cuda.max_memory_allocated() - base - new.numel() * 4) / 2**20
            eq = same(new.cpu(), ref_h)
            # atomic path: acc.index_add_(tok, y * w), unique tok -> one add a row
            w = torch.rand((c,), device=DEV).half()
            a_old = torch.zeros((c, H), dtype=torch.float32, device=DEV)
            y = torch.empty((c, H), dtype=torch.float32, device=DEV)
            old_expert(mo, layer, 1, x16, tok, y); a_old.index_add_(0, tok, y * w.float().unsqueeze(1)); del y
            a_new = torch.zeros_like(a_old)
            mn._expert_dq(layer, 1, x16, tok, None, w, a_new)
            eqa = same(a_new, a_old)
            ok &= eq and eqa
            print(f"K={K} rows {c:6d}: deterministic {'bit-identical' if eq else 'DIFFERS'}, atomic "
                  f"{'bit-identical' if eqa else 'DIFFERS'}; chunked transient {trans:.1f} MiB")
            del a_old, a_new, new, x16, tok, ref_h; torch.cuda.empty_cache()
        # graph replay of the chunked call (fixed shapes) == eager
        c = 9000
        x16 = torch.randn((c, H), device=DEV).half(); tok = torch.randperm(c, device=DEV)
        eag = torch.empty((c, H), dtype=torch.float32, device=DEV); mn._expert_dq(layer, 0, x16, tok, eag)
        gout = torch.empty_like(eag)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            mn._expert_dq(layer, 0, x16, tok, gout)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            mn._expert_dq(layer, 0, x16, tok, gout)
        gout.zero_(); g.replay(); torch.cuda.synchronize()
        eqg = same(gout, eag); ok &= eqg
        print(f"K={K} graph replay ({c} rows, 3 pieces) vs eager: {'bit-identical' if eqg else 'DIFFERS'}")
        del g, x16, tok, eag, gout; torch.cuda.empty_cache()
        # timing (median of 7) old vs new
        for c in (2048, 4096, 12288):
            x16 = torch.randn((c, H), device=DEV).half(); tok = torch.arange(c, device=DEV)
            out = torch.empty((c, H), dtype=torch.float32, device=DEV)
            def tm(fn):
                ts = []
                for _ in range(8):
                    torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                return sorted(ts[1:])[3] * 1e3
            to = tm(lambda: old_expert(mo, layer, 0, x16, tok, out)); tn = tm(lambda: mn._expert_dq(layer, 0, x16, tok, out))
            print(f"K={K} rows {c:6d}: unchunked {to:7.2f} ms, chunked {tn:7.2f} ms")
            del x16, tok, out; torch.cuda.empty_cache()
        del layer, mn, mo; torch.cuda.empty_cache()

    # full apply above FUSED_ROWS: old module vs new, one expert heavy (all tokens pick expert 2) + random second pick
    for K in (3,):
        T, E, topk = 5000, 4, 2
        ln, mn = make(NEW, K, E, topk, X.exl3_moe_max_concurrency(0))
        mo = object.__new__(OLD.Exl3MoEMethod); mo.__dict__.update(mn.__dict__)
        x = (torch.randn((T, H), device=DEV) * 0.5).bfloat16()
        second = torch.randint(0, E - 1, (T,), device=DEV); second = second + (second >= 2).long()
        ids = torch.stack([torch.full_like(second, 2), second], 1).to(torch.int32)
        w = torch.rand((T, topk), device=DEV, dtype=torch.float32)
        yo = mo.apply(ln, x, w, ids); yn = mn.apply(ln, x, w, ids); torch.cuda.synchronize()
        eq = torch.equal(yo, yn); ok &= eq
        print(f"apply T={T} K={K} (expert 2: {T} rows, others ~{T // 3}): old vs new {'bit-identical' if eq else 'DIFFERS'}"
              f" (max |d| {(yo.float() - yn.float()).abs().max().item():.3g})")
    print("ALL BIT-IDENTICAL" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


main()
