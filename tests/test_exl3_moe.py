"""Exl3MoEMethod (quant/exl3_moe.py) on one gfx1201 card: a stock RoutedExperts layer as rank 0 of EP 8 (experts 0-47 of
384 local), synthetic EXL3 experts loaded through vLLM's own expert params mapping, vs a per-expert f32 reference.

Synthetic tensors as package K's kernel gates: random int16 trellis (every bitstream is a valid trellis code), suh =
random signs, svh = random signs x U(0.5, 1.5) / sqrt(in), mul1 codebook. Reference: W = diag(suh) H Wrot H diag(svh)
with Wrot = ext.reconstruct (exact decode) and H the 128-point block Hadamard, in f32 (quant/exl3.py dequant_f32);
act = min(silu(g), L) * clamp(u, -L, L) (exllamav3's act_limit, L = swiglu_limit 10); out = sum_k w_k down(act).

Run (VM100, card 5, wheel installed into the dev image for the run):
    ~/ds41/exl3/gpu-run.sh 5 bash -c 'pip install -q --no-deps /ds41/exl3/dist/*.whl && python3 tests/test_exl3_moe.py'
"""
import math
import os
import sys
import time

import torch

import exllamav3_ext as ext
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

from r9700_vllm.quant import exl3_moe as M
from r9700_vllm.quant.exl3 import Exl3Config, dequant_f32

DEV = torch.device("cuda:0")
G, EP, TOPK, H, I, LIMIT = 384, 8, 6, 5120, 2304, 10.0
E = G // EP
torch.backends.cuda.matmul.allow_tf32 = False


def make_layer(K: int, seed: int, real: str | None = None):
    pc = FusedMoEParallelConfig(tp_size=1, pcp_size=1, dp_size=1, ep_size=EP, tp_rank=0, pcp_rank=0, dp_rank=0,
                                ep_rank=0, sp_size=1, use_ep=True, all2all_backend="allgather_reducescatter",
                                enable_eplb=False)
    emm = ExpertMapManager(max_num_batched_tokens=8192, top_k=TOPK, global_num_experts=G, num_redundant_experts=0,
                           num_expert_group=None, moe_parallel_config=pc, placement_strategy="linear",
                           enable_eplb=False)
    cfg = FusedMoEConfig(num_experts=G, experts_per_token=TOPK, hidden_dim=H, intermediate_size=I, intermediate_pad=0,
                         num_local_experts=emm.local_num_experts, num_logical_experts=G, activation=MoEActivation.SILU,
                         device=DEV, routing_method=RoutingMethodType.DeepseekV4, moe_parallel_config=pc,
                         in_dtype=torch.bfloat16, swiglu_limit=LIMIT)
    qc = Exl3Config.from_config({"quant_method": "exl3", "version": "1.6.2", "avg_bits": 2.9, "head_bits": 6,
                                 "codebook": "mul1"})
    qc.get_quant_method = lambda layer, prefix: M.Exl3MoEMethod(layer.moe_config, qc)
    with set_current_vllm_config(VllmConfig()), torch.device(DEV):
        layer = RoutedExperts(f"layers.{seed}.ffn.experts", torch.bfloat16, cfg, qc, emm, ckpt_gate_proj_name="w1",
                              ckpt_down_proj_name="w2", ckpt_up_proj_name="w3", swiglu_limit=LIMIT)
    assert isinstance(layer.quant_method, M.Exl3MoEMethod) and layer.local_num_experts == E

    # checkpoint tensors exactly as the pack names them, through the stock mapping (deepseek_v41 load_weights path)
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    mapping = RoutedExperts.build_expert_params_mapping("w1", "w2", "w3", num_experts=G, routed_experts_prefix="")
    params = dict(layer.named_parameters())
    ref = {}
    src = None
    if real:
        from safetensors import safe_open
        src = safe_open(real, "pt")
        names = list(src.keys())
        pre = names[0].split(".ffn.")[0] + ".ffn.experts."
        nameset = set(names)
        have = sorted({int(n[len(pre):].split(".")[0]) for n in names if n.startswith(pre)})
        have = [c for c in have if all(f"{pre}{c}.{ck}.{f}" in nameset for ck in ("w1", "w2", "w3")
                                       for f in ("trellis", "suh", "svh"))]
        assert have, f"{real}: no complete expert"
        print(f"  {real}: {len(have)} complete experts (reused cyclically for the 48 local + 2 non-local slots)")
    for e in list(range(E)) + [E, G - 1]:            # two non-local experts: the loader must refuse them
        for ck, (k, n) in (("w1", (H, I)), ("w3", (H, I)), ("w2", (I, H))):
            if src is not None:                      # pack expert have[i % n] stands in for global expert i
                ce = have[e % len(have)]
                t, suh, svh = (src.get_tensor(f"{pre}{ce}.{ck}.{f}").to(DEV) for f in ("trellis", "suh", "svh"))
                assert t.shape[-1] == 16 * K, (real, t.shape, K)
            else:
                t = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * K), dtype=torch.int32, device=DEV,
                                  generator=g).to(torch.int16)
                suh = (torch.randint(0, 2, (k,), device=DEV, generator=g) * 2 - 1).half()
                svh = ((torch.randint(0, 2, (n,), device=DEV, generator=g) * 2 - 1).float()
                       * (0.5 + torch.rand((n,), device=DEV, generator=g)) / math.sqrt(k)).half()
            tensors = {"trellis": t.cpu(), "suh": suh.cpu(), "svh": svh.cpu(), "mul1": torch.tensor(-2082680531,
                                                                                                    dtype=torch.int32)}
            for field, w in tensors.items():
                name = f"experts.{e}.{ck}.{field}"
                hits = [mp for mp in mapping if mp[1] in name]
                ok = False
                for pname, wname, eid, sid in hits:
                    pn = name.replace(wname, pname).removeprefix("experts.")
                    p = params[pn]
                    ok |= bool(p.weight_loader(p, w, pn, shard_id=sid, expert_id=eid, return_success=True))
                assert ok == (e < E), (name, ok)
            if e < E:
                ref[e, ck] = weight_f32(t, suh, svh)
    layer.quant_method.process_weights_after_loading(layer)
    own = sum(b.numel() * b.element_size() for n, b in layer.named_buffers() if n.startswith("exl3_"))
    shared = sum(t.numel() * t.element_size() for t in M._shared.values())
    print(f"  K={K}: layer expert buffers {own / 2**20:.1f} MiB (trellis + suh/svh + pointer tables), "
          f"process-wide shared scratch {shared / 2**20:.1f} MiB")
    return layer, ref


def weight_f32(trellis, suh, svh):
    return dequant_f32(trellis, suh, svh, trellis.shape[-1] // 16, True)


def routing(T: int, all_local, g: torch.Generator):
    """all_local: True = picks among this rank's experts 0-47, False = among all 384, "none" = never this rank."""
    pool = E if all_local is True else G - E if all_local == "none" else G
    ids = torch.rand((T, pool), device=DEV, generator=g).topk(TOPK, dim=1).indices.to(torch.int32)
    if all_local == "none":
        ids += E
    w = torch.rand((T, TOPK), device=DEV, generator=g) + 0.05
    return (w / w.sum(1, keepdim=True) * 1.5).float(), ids


def reference(x, w, ids, ref):
    xf = x.float()
    out = torch.zeros((x.shape[0], H), dtype=torch.float32, device=DEV)
    for e in range(E):
        rows, slot = (ids == e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        xe = xf[rows]
        gt = torch.nn.functional.silu(xe @ ref[e, "w1"]).clamp(max=LIMIT)
        u = (xe @ ref[e, "w3"]).clamp(-LIMIT, LIMIT)
        out.index_add_(0, rows, ((gt * u) @ ref[e, "w2"]) * w[rows, slot].unsqueeze(1))
    return out


def errs(y, r):
    d = (y.float() - r).abs()
    rn = r.norm().item()
    return d.max().item(), d.max().item() / max(r.abs().max().item(), 1e-9), (d.norm().item() / max(rn, 1e-9))


def check_numerics(layer, ref, K, Ms=(1, 2, 3, 4, 8, 16, 21, 64, 512, 4096), tol=1e-2):
    ok = True
    g = torch.Generator(device=DEV)
    g.manual_seed(100 + K)
    for T in Ms:
        for all_local in ((False, True, "none") if T <= 21 else (False, True)):
            for scale in ((1.0, 6.0) if T == 64 else (1.0,)):
                w, ids = routing(T, all_local, g)
                x = (torch.randn((T, H), device=DEV, generator=g) * scale).to(torch.bfloat16)
                y = layer.quant_method.apply(layer, x, w, ids, None, None)
                r = reference(x, w, ids, ref)
                ma, mr, rr = errs(y, r)
                local = int((ids < E).sum())
                heavy = int((torch.bincount(ids.flatten().long(), minlength=G)[:E] > (M.RECON_ROWS if T > M.FUSED_ROWS else 1 << 30)).sum())
                good = rr < tol and bool(torch.isfinite(y).all())
                ok &= good
                print(f"  K={K} M={T:5d} {'all-local' if all_local is True else 'none-loc ' if all_local else 'random   '} x*{scale:<3g} local picks {local:6d} "
                      f"heavy {heavy:2d} | maxabs {ma:.3e} max/|ref| {mr:.3e} rms rel {rr:.3e} "
                      f"{'ok' if good else '<-- FAIL'}")
    return ok


def check_graphs(layer, K, tag, Ms=range(1, 9)):
    ok = True
    g = torch.Generator(device=DEV)
    g.manual_seed(200 + K)
    for T in Ms:
        w, ids = routing(T, False, g)
        ids[0, 0] = 3                               # at least one local pick
        x = torch.randn((T, H), device=DEV, generator=g).to(torch.bfloat16)
        sx, sw, si = x.clone(), w.clone(), ids.clone()
        for _ in range(2):
            layer.quant_method.apply(layer, sx, sw, si, None, None)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sy = layer.quant_method.apply(layer, sx, sw, si, None, None)
        diffs = []
        for trial in range(3):
            w2, ids2 = routing(T, trial == 2, g)
            x2 = torch.randn((T, H), device=DEV, generator=g).to(torch.bfloat16)
            sx.copy_(x2)
            sw.copy_(w2)
            si.copy_(ids2)
            graph.replay()
            eager = layer.quant_method.apply(layer, x2, w2, ids2, None, None)
            torch.cuda.synchronize()
            diffs.append((sy.float() - eager.float()).abs().max().item())
        good = max(diffs) == 0.0
        ok &= good
        print(f"  K={K} {tag} graph M={T}: replay vs eager max diff {max(diffs):.3e} {'ok' if good else '<-- FAIL'}")
    return ok


def bench(layer, K, iters=200):
    g = torch.Generator(device=DEV)
    g.manual_seed(300 + K)
    rows = []
    for tier, dec, atomic in (("mgemv       ", "mgemv", False), ("coop        ", "coop", False),
                              ("fused det   ", "fused", False), ("fused atomic", "fused", True)):
        M.DECODE, M.ATOMIC = dec, atomic
        for all_local in (False, True):
            for T in (1, 2, 4, 8, 12, 16, 21):
                if dec == "coop" and T > 8:
                    continue
                w, ids = routing(T, all_local, g)
                if not all_local:
                    ids[0, 0] = 5
                x = torch.randn((T, H), device=DEV, generator=g).to(torch.bfloat16)
                f = lambda: layer.quant_method.apply(layer, x, w, ids, None, None)
                for _ in range(10):
                    f()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    f()
                torch.cuda.synchronize()
                eager = (time.perf_counter() - t0) / iters * 1e6
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    f()
                for _ in range(10):
                    graph.replay()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    graph.replay()
                torch.cuda.synchronize()
                gr = (time.perf_counter() - t0) / iters * 1e6
                n_exp = int((torch.unique(ids[ids < E])).numel())
                mb = n_exp * 3 * H * I * K / 8 / 1e6
                rows.append((tier, all_local, T, n_exp, eager, gr, mb / gr * 1e6 / 1e3))
                print(f"  K={K} {tier} {'all-local' if all_local else 'random   '} M={T}: "
                      f"{n_exp:2d} local experts, eager {eager:7.1f} us, graph {gr:7.1f} us "
                      f"({mb / gr * 1e6 / 1e3:6.1f} GB/s of trellis)")
    M.DECODE, M.ATOMIC = "mgemv", False
    for atomic in (False, True):
        M.ATOMIC = atomic
        for T in (64, 512, 2048, 4096, 8192):
            w, ids = routing(T, False, g)
            x = torch.randn((T, H), device=DEV, generator=g).to(torch.bfloat16)
            f = lambda: layer.quant_method.apply(layer, x, w, ids, None, None)
            for _ in range(3):
                f()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                f()
            torch.cuda.synchronize()
            us = (time.perf_counter() - t0) / 10 * 1e6
            cnt = torch.bincount(ids.flatten().long(), minlength=G)[:E]
            print(f"  K={K} prefill {'atomic' if atomic else 'det   '} random M={T:5d}: {int(cnt.sum()):6d} local picks, "
                  f"{int((cnt > M.RECON_ROWS).sum()) if T > M.FUSED_ROWS else 0} heavy, max {int(cnt.max())} rows | eager {us:9.1f} us "
                  f"({us / T:6.1f} us/token)")
    M.ATOMIC = False
    return rows


if __name__ == "__main__":
    print("device", torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).gcnArchName,
          "| g_get_num_sms", ext.g_get_num_sms(0), "| exl3_moe_max_concurrency", ext.exl3_moe_max_concurrency(0),
          "| fused rows", M.FUSED_ROWS, "| decode", M.DECODE, M.DECODE_ROWS)
    allok = True
    for K, seed in ((3, 1), (2, 18)) if os.environ.get("EXL3_ONLY_REAL") != "1" else ():
        layer, ref = make_layer(K, seed)
        print(f"-- numerics K={K}, default tiers (decode {M.DECODE}) (expert buffers {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated)")
        allok &= check_numerics(layer, ref, K)
        for dec in ("coop", "fused"):
            M.DECODE = dec
            print(f"-- numerics K={K}, decode tier {dec} (R9K_EXL3_MOE_DECODE={dec})")
            allok &= check_numerics(layer, ref, K, Ms=(1, 2, 3, 4, 8, 16))
        M.ATOMIC = True
        print(f"-- numerics K={K}, fused tier, atomic accumulation")
        allok &= check_numerics(layer, ref, K)
        M.ATOMIC = False
        print(f"-- HIP graph capture K={K}")
        allok &= check_graphs(layer, K, "fused")
        M.DECODE = "coop"
        allok &= check_graphs(layer, K, "coop ")
        M.DECODE = "mgemv"
        allok &= check_graphs(layer, K, "mgemv", Ms=list(range(1, 9)) + [12, 16, 21])
        print(f"-- timing K={K} (us a call, 48 local experts)")
        bench(layer, K)
        del layer, ref
        torch.cuda.empty_cache()
    real_dir = os.environ.get("EXL3_REAL_DIR", "/ds41/exl3/kerneltest/real")
    for K, f in ((3, "l0-sample.safetensors"), (2, "l18-sample.safetensors")):
        path = os.path.join(real_dir, f)
        if not os.path.exists(path):
            print(f"-- real experts: {path} missing, skipped")
            continue
        layer, ref = make_layer(K, 0 if K == 3 else 18, real=path)
        print(f"-- numerics on real pack experts ({f}, K={K}), default tiers")
        allok &= check_numerics(layer, ref, K, Ms=(1, 4, 8, 16, 64, 512, 4096))
        allok &= check_graphs(layer, K, "mgemv", Ms=(1, 4, 8, 16))
        del layer, ref
        torch.cuda.empty_cache()
    print("ALL OK" if allok else "FAILURES")
    sys.exit(0 if allok else 1)

# Results, VM100 card 5 (R9700 gfx1201, SCLK offset -500 MHz as set on the box), r9700/vllm:dev + exllamav3 1.6.6 wheel,
# 2026-11-02, ALL OK (241 checks synthetic + real):
#   g_get_num_sms 32, exl3_moe_max_concurrency 8. Device memory: one layer's experts 609.5 MiB at K=3, 407.0 MiB at K=2
#   (trellis + suh/svh + pointer tables); process-wide shared scratch 146.6 MiB (fused temp 116 MiB, dq 22.5 MiB, decode).
#   Numerics (bf16 out vs f32 reference), every tier, K=3 and K=2, synthetic and real pack experts (layer 0 / layer 18
#   samples): rms rel 1.82e-3 .. 1.91e-3 (bf16 output rounding is ~1.1e-3 of it), max/|ref| <= 3.9e-3, including x*6
#   (act_limit clipping), all-local, none-local (exact zeros) and M=4096 all-local with every expert in the reconstruct
#   tier. HIP graph capture: replay == eager bitwise for mgemv M=1..8,12,16,21, coop M=1..8, fused M=1..8.
#   Graph replay, us a call (48 local experts, K=3):
#     tier    routing     M=1    M=2    M=4    M=8    M=16
#     mgemv   random     76.8  141.6  140.1  376.1  857.7   (1/2/1/5/6 local experts)
#     mgemv   all-local 193.9  382.5  688.3 1333.5 2804.0   (6/10/20/32/40 local experts; 411 GB/s at M=1)
#     coop    random    201.9  399.4  498.4  718.4      -
#     coop    all-local 714.2 1263.3 2039.0 3422.5      -
#     fused   random    454.3  451.8  672.8  472.1 1064.6
#     fused   all-local 707.5 1085.3 1720.4 2992.4 3697.9
#   K=2: mgemv random M=1 77.8, all-local M=1 184.7, M=8 1289.4 us.
#   Prefill, eager, random 6-of-384 routing, deterministic accumulation (K=3): M=64 2.69 ms, 512 4.32 ms, 2048 8.05 ms,
#   4096 14.3 ms, 8192 23.1 ms (22 experts > RECON_ROWS=128 in the reconstruct tier; 27.0 ms with the fused tier alone),
#   16384 29.5 ms (50.3 ms fused alone). Atomic accumulation: within +-10%, no consistent gain.
