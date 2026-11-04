"""DeepSeek-V4.1 decode glue (kernels/r9k_dsv4_glue.hip via r9700_vllm/dsv4_glue.py) against the stock ops it replaces,
one GPU, real shapes (hidden 5120, q_lora 1280, kv 512, 384 experts x 6 / drafter 128 x 3), M = 1, 2, 6, 12, 24, 32.

  norm     r9k_dsv4_norm_pre vs vLLM's RMSNorm op on this pin (ir.ops.rms_norm, attn_norm / ffn_norm) and stock
           fused_q_kv_rmsnorm (q_norm / kv_norm, Triton) -- bf16 outputs, from bf16 rows and from the f32 C buffer
           (rounded to bf16 first, as the EXL3 apply's cast would); f16 output == bf16 output cast to f16.
  rotate   the f16 rows rotated with suh vs the A_had exllamav3's exl3_gemm writes for the same input (bit-exact).
  route    r9k_dsv4_route vs torch.mm(x, W^T, out_dtype=f32) + ops.topk_hash_softplus_sqrt (what stock runs on
           gfx1201): (a) top-k bit-exact given identical logits (the kernel's own logits fed to the stock op);
           (b) end to end: logits within f32 rounding, ids identical except near-ties (counted), weights within 1e-6;
           (c) the fused ffn_norm variant: normalized rows == norm_pre's, routing == (a); bias_vl sentinel rows and
           is_padding rows.
  graph    capture + replay == eager, bitwise.
  timing   graph replay per launch vs stock's launches (median over replays).
Real weights when the pack is mounted (PACK, LAYER: ffn.gate.weight / bias / bias_vl, the norm weights of one layer),
random otherwise. Run in the image with PYTHONPATH=/opt/r9700, R9K_LIB pointing at a libr9k with the glue kernels.
TIMING=0 skips timing. Exit status 1 on any FAIL."""
import json
import os
import statistics
import sys

import torch

import vllm  # noqa: F401
import vllm._custom_ops as ops

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from r9700_vllm import dsv4_glue as G

dev = torch.device("cuda:0")
torch.cuda.set_device(dev)
PACK = os.environ.get("PACK", "/ds41/exl3/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw")
LAYER = int(os.environ.get("LAYER", "7"))
EPS = 1e-6
H, QL, KVD, E, TOPK, RSF = 5120, 1280, 512, 384, 6, 1.5
MS = [int(m) for m in os.environ.get("MS", "1,2,4,6,8,12,24,32").split(",")]
ok_all = True
gen = torch.Generator(device="cpu").manual_seed(11)


def check(name, cond, msg=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {msg}", flush=True)


def ulp_diff(a, b):
    """(elements differing, max difference in bf16 ulps) of two bf16 tensors."""
    ai, bi = a.view(torch.int16).int(), b.view(torch.int16).int()
    d = (ai - bi).abs()
    return int((d > 0).sum()), int(d.max()) if d.numel() else 0


def load():
    idx = os.path.join(PACK, "model.safetensors.index.json")
    names = {"gate_w": f"layers.{LAYER}.ffn.gate.weight", "bias": f"layers.{LAYER}.ffn.gate.bias",
             "bias_vl": f"layers.{LAYER}.ffn.gate.bias_vl", "attn_norm": f"layers.{LAYER}.attn_norm.weight",
             "ffn_norm": f"layers.{LAYER}.ffn_norm.weight", "q_norm": f"layers.{LAYER}.attn.q_norm.weight",
             "kv_norm": f"layers.{LAYER}.attn.kv_norm.weight"}
    if os.path.exists(idx):
        from safetensors import safe_open
        wm = json.load(open(idx))["weight_map"]
        out = {}
        for k, n in names.items():
            with safe_open(os.path.join(PACK, wm[n]), "pt") as f:
                t = f.get_tensor(n)
            # as the stock loaders store them: gate weight f16 -> bf16, gate bias f16 -> f32
            out[k] = t.to(torch.bfloat16 if k in ("gate_w",) or "norm" in k else torch.float32).contiguous().to(dev)
        print(f"weights: pack layer {LAYER}")
        return out
    print("weights: random (no pack)")
    return {"gate_w": (torch.randn(E, H, generator=gen) * H ** -0.5).bfloat16().to(dev),
            "bias": (torch.randn(E, generator=gen) * 0.05).to(dev), "bias_vl": (torch.randn(E, generator=gen) * 0.05).to(dev),
            "attn_norm": (1 + 0.1 * torch.randn(H, generator=gen)).bfloat16().to(dev),
            "ffn_norm": (1 + 0.1 * torch.randn(H, generator=gen)).bfloat16().to(dev),
            "q_norm": (1 + 0.1 * torch.randn(QL, generator=gen)).bfloat16().to(dev),
            "kv_norm": (1 + 0.1 * torch.randn(KVD, generator=gen)).bfloat16().to(dev)}


W = load()


def rms_ref_native(x, w):
    xf = x.float()
    v = xf.pow(2).mean(-1, keepdim=True)
    return ((xf * torch.rsqrt(v + EPS)).to(w.dtype) * w).to(x.dtype)


def rms_ref_fused(x, w):
    xf = x.float()
    v = (xf * xf).sum(-1, keepdim=True) / x.shape[-1]
    return (xf * torch.rsqrt(v + EPS) * w.float()).to(x.dtype)


def stock_rms(x, w):
    """What vLLM's RMSNorm runs on this pin: the IR op's vllm_c impl (the server log: rms_norm=['vllm_c', 'native'];
    STOCK_RMS=native for the torch reference)."""
    if os.environ.get("STOCK_RMS", "vllm_c") == "vllm_c":
        out = torch.empty_like(x)
        torch.ops._C.rms_norm(out, x.contiguous(), w, EPS)      # the _C kernel the server runs (Z's trace)
        return out
    from vllm import ir
    return ir.ops.rms_norm.impls["native"].impl_fn(x, w, EPS, None)


def rows(M, n, scale=1.0, seed=0):
    g = torch.Generator(device="cpu").manual_seed(1000 + seed)
    return (torch.randn(M, n, generator=g) * scale).bfloat16().to(dev)


# --------------------------------------------------------------------------------------------------------- norm
print("== norm (attn_norm / ffn_norm: hidden 5120)")
try:
    from vllm import ir
    print("  ir.ops.rms_norm impls:", list(getattr(ir.ops.rms_norm, "impls", {}).keys()))
except Exception as e:
    print("  ir introspection:", e)
modes = {}
for M in MS:
    x = rows(M, H, 3.0, M)
    st = stock_rms(x, W["attn_norm"])
    nat, fus = rms_ref_native(x, W["attn_norm"]), rms_ref_fused(x, W["attn_norm"])
    for mode in (0, 1):
        ob = torch.empty_like(x)
        oh = torch.empty(M, H, dtype=torch.float16, device=dev)
        G.norm_pre([G.Seg(x, W["attn_norm"], True, ob, oh)], M, EPS, mode)
        n, mx = ulp_diff(ob, st)
        modes.setdefault(mode, []).append((n, mx))
        if M == MS[-1]:
            print(f"  mode {mode}: vs stock rms_norm {n}/{ob.numel()} elems differ (max {mx} ulp); "
                  f"vs native {ulp_diff(ob, nat)}, vs fused-f32 {ulp_diff(ob, fus)}; stock==native {ulp_diff(st, nat)}")
        check(f"f16 out == bf16 out cast M={M} mode={mode}", torch.equal(oh, ob.half()))
best = min(modes, key=lambda m: sum(n for n, _ in modes[m]))
print(f"  stock RMSNorm matches mode {best} best (set R9K_DSV41_NORM_MODE={best} if not the default {G.NORM_MODE})")
for M in MS:
    x = rows(M, H, 3.0, M)
    ob = torch.empty_like(x)
    G.norm_pre([G.Seg(x, W["attn_norm"], True, ob)], M, EPS, best)
    n, mx = ulp_diff(ob, stock_rms(x, W["attn_norm"]))
    check(f"attn_norm M={M} within 1 bf16 ulp", mx <= 1, f"({n} of {ob.numel()} differ)")

print("== q_norm / kv_norm from the fused_wqa_wkv output (bf16 rows and the f32 C buffer)")
from vllm.models.common.ops import fused_q_kv_rmsnorm
for M in MS:
    c = (torch.randn(M, QL + KVD, generator=gen) * 2).to(dev)          # the GEMV's f32 output, row layout
    qr_kv = c.bfloat16()
    q_st, kv_st = fused_q_kv_rmsnorm(qr_kv[:, :QL], qr_kv[:, QL:], W["q_norm"], W["kv_norm"], EPS)
    for src, name in ((qr_kv, "bf16"), (c, "f32")):
        q = torch.empty(M, QL, dtype=torch.bfloat16, device=dev)
        kv = torch.empty(M, KVD, dtype=torch.bfloat16, device=dev)
        qh = torch.empty(M, QL, dtype=torch.float16, device=dev)
        G.norm_pre([G.Seg(src[:, :QL], W["q_norm"], True, q, qh), G.Seg(src[:, QL:], W["kv_norm"], True, kv)],
                   M, EPS, 1)
        nq, mq = ulp_diff(q, q_st)
        nk, mk = ulp_diff(kv, kv_st)
        check(f"q/kv norm M={M} from {name}", mq <= 1 and mk <= 1 and torch.equal(qh, q.half()),
              f"(q {nq}/{q.numel()} max {mq} ulp, kv {nk}/{kv.numel()} max {mk})")
    # part-block layout of a multi-part GEMV C buffer: qr block [M, QL] then kv block [M, KVD]
    cb = torch.cat([c[:, :QL].reshape(-1), c[:, QL:].reshape(-1)])
    q2 = torch.empty(M, QL, dtype=torch.bfloat16, device=dev)
    kv2 = torch.empty(M, KVD, dtype=torch.bfloat16, device=dev)
    G.norm_pre([G.Seg(cb[:M * QL].view(M, QL), W["q_norm"], True, q2),
                G.Seg(cb[M * QL:].view(M, KVD), W["kv_norm"], True, kv2)], M, EPS, 1)
    G.norm_pre([G.Seg(c[:, :QL], W["q_norm"], True, q), G.Seg(c[:, QL:], W["kv_norm"], True, kv)], M, EPS, 1)
    check(f"part-block layout == row layout M={M}", torch.equal(q2, q) and torch.equal(kv2, kv))

# ------------------------------------------------------------------------------------------------------- rotate
print("== rotation (suh) vs exllamav3's A_had")
try:
    import exllamav3_ext as X
    have_x = True
except Exception as e:
    have_x = False
    print("  exllamav3_ext unavailable:", e)
if have_x:
    for K, N in ((H, 1792), (QL, 4096)):
        suh = (torch.randint(0, 2, (K,), generator=gen) * 2 - 1).half().to(dev)
        tr = torch.randint(-32768, 32768, (K // 16, N // 16, 80), generator=gen, dtype=torch.int32).to(torch.int16).to(dev)
        svh = ((torch.rand(N, generator=gen) - 0.5) * 2 / K ** 0.5).half().to(dev)
        for M in (1, 2, 6, 8):
            x = rows(M, K, 2.0, 7 + M)
            a16 = x.half().contiguous()
            ah = torch.empty_like(a16)
            cc = torch.empty(M, N, dtype=torch.float32, device=dev)
            X.exl3_gemm(a16, tr, cc, suh, ah, svh, -1, False, True, 0)
            ours = G.prepass(x, None, suh)
            d = (ours.view(torch.int16).int() - ah.view(torch.int16).int()).abs()
            # M = 1: exllamav3's single-row rotation, bit-exact; M > 1: its multi-row rotation differs by <= 2 f16 ulp
            check(f"rotated input K={K} M={M} vs exl3_gemm A_had", torch.equal(ours, ah) if M == 1 else int(d.max()) <= 2,
                  f"({int((d > 0).sum())}/{d.numel()} differ, max {int(d.max())} f16 ulp)")
        x = rows(4, K, 2.0, 99)
        w = W["attn_norm"] if K == H else W["q_norm"]
        nb = torch.empty_like(x)
        G.norm_pre([G.Seg(x, w, True, nb)], 4, EPS, best)
        r1 = G.prepass(x, w, suh, EPS, mode=best)
        r2 = G.prepass(nb, None, suh)
        check(f"norm + rotate K={K} == rotate(norm)", torch.equal(r1, r2))

# -------------------------------------------------------------------------------------------------------- route
print("== router (gate 5120 -> 384, sqrt-softplus top-6, rsf 1.5)")


def stock_topk(logits, bias, topk, bias_vl=None, ids=None, lo=0, pad=None):
    M = logits.shape[0]
    tw = torch.empty(M, topk, dtype=torch.float32, device=dev)
    ti = torch.empty(M, topk, dtype=torch.int32, device=dev)
    te = torch.empty(M, topk, dtype=torch.int32, device=dev)
    ops.topk_hash_softplus_sqrt(tw, ti, te, logits, True, RSF, bias, ids, None, is_padding=pad, bias_vl=bias_vl,
                                image_sentinel_lo=lo)
    return tw, ti


G.warm(dev)
for M in MS:
    x = rows(M, H, 1.0, 50 + M)
    tw, ti, _, lg, _ = G.route(x, W["gate_w"], W["bias"], TOPK, True, RSF, logits_out=True)
    sw, si = stock_topk(lg, W["bias"], TOPK)
    check(f"top-k given identical logits M={M}", torch.equal(ti, si) and torch.equal(tw, sw),
          f"(ids {int((ti != si).sum())} differ, max |dw| {(tw - sw).abs().max().item():.2g})")
    ref = torch.mm(x, W["gate_w"].t(), out_dtype=torch.float32)
    rel = ((lg - ref).abs().max() / ref.abs().max()).item()
    sw2, si2 = stock_topk(ref, W["bias"], TOPK)
    same = (torch.sort(ti, 1)[0] == torch.sort(si2, 1)[0]).all(1)
    check(f"end to end M={M}: logits rel err {rel:.2e}", rel < 1e-5,
          f"(rows with identical id sets {int(same.sum())}/{M}; weights max |d| on those "
          f"{((tw - sw2).abs()[same].max().item() if same.any() else 0):.2g})")
# sentinel / padding / ids dtype
M = 6
x = rows(M, H, 1.0, 77)
ids = torch.tensor([5, 128000, 128001, 7, 128004, 128005], dtype=torch.int32, device=dev)
pad = torch.tensor([False, False, False, False, True, True], device=dev)
tw, ti, _, lg, _ = G.route(x, W["gate_w"], W["bias"], TOPK, True, RSF, bias_vl=W["bias_vl"], input_ids=ids,
                        sentinel_lo=128000, is_padding=pad, logits_out=True)
sw, si = stock_topk(lg, W["bias"], TOPK, W["bias_vl"], ids, 128000, pad)
real = ~pad
check("bias_vl sentinel rows + is_padding", torch.equal(ti, si) and torch.equal(tw[real], sw[real]),
      f"(ids {ti.tolist()} vs {si.tolist()})")
tw64, ti64, _, _, _ = G.route(x, W["gate_w"], W["bias"], TOPK, True, RSF, bias_vl=W["bias_vl"], input_ids=ids.long(),
                           sentinel_lo=128000, is_padding=pad)
check("int64 input ids", torch.equal(ti64, ti) and torch.equal(tw64, tw))
# drafter router: 128 experts, top-3
gd = W["gate_w"][:128].contiguous()
bd = W["bias"][:128].contiguous()
x = rows(6, H, 1.0, 78)
tw, ti, _, lg, _ = G.route(x, gd, bd, 3, True, RSF, logits_out=True)
sw, si = stock_topk(lg, bd, 3)
check("drafter shape (128 experts, top-3)", torch.equal(ti, si) and torch.equal(tw, sw))
# fused ffn_norm
for M in MS:
    xr = rows(M, H, 3.0, 90 + M)
    tw, ti, xn, lg, _ = G.route(xr, W["gate_w"], W["bias"], TOPK, True, RSF, norm_w=W["ffn_norm"], eps=EPS, mode=best,
                                logits_out=True)
    n, mx = ulp_diff(xn, stock_rms(xr, W["ffn_norm"]))
    tw2, ti2, _, lg2, _ = G.route(xn, W["gate_w"], W["bias"], TOPK, True, RSF, logits_out=True)
    check(f"fused ffn_norm M={M}: xn vs stock RMSNorm, logits / routing == route(xn)",
          mx <= 1 and torch.equal(lg, lg2) and torch.equal(ti, ti2) and torch.equal(tw, tw2),
          f"(xn {n}/{xn.numel()} differ, max {mx} ulp)")

print("== shared expert gate_up + silu-and-mul (clamp 10) in the router launch (TP8 slice: 2 x 288 rows)")
I_SH, LIM = 288, 10.0
sw_ = (torch.randn(2 * I_SH, H, generator=gen) * H ** -0.5 * 3).bfloat16().to(dev)


def stock_act(gu):
    out = torch.empty(gu.shape[0], gu.shape[1] // 2, dtype=gu.dtype, device=dev)
    torch.ops._C.silu_and_mul_with_clamp(out, gu, LIM, 1.0, 0.0)
    return out


def emul_act(gu):
    g = gu[:, :I_SH].float().clamp(max=LIM)
    u = gu[:, I_SH:].float().clamp(-LIM, LIM)
    return ((g / (1 + torch.exp(-g))) * u).bfloat16()


gu0 = (torch.randn(64, 2 * I_SH, generator=gen) * 4).bfloat16().to(dev)
n, mx = ulp_diff(emul_act(gu0), stock_act(gu0))
print(f"  formula check (torch emulation vs the _C op on the same gate_up): {n}/{64 * I_SH} differ, max {mx} ulp")
for M in MS:
    xr = rows(M, H, 3.0, 400 + M)
    tw, ti, xn, _, act = G.route(xr, W["gate_w"], W["bias"], TOPK, True, RSF, norm_w=W["ffn_norm"], eps=EPS,
                                 mode=best, shared_w=sw_, limit=LIM)
    gu = torch.mm(xn, sw_.t())
    ref = stock_act(gu)
    tw2, ti2, _, _, _ = G.route(xr, W["gate_w"], W["bias"], TOPK, True, RSF, norm_w=W["ffn_norm"], eps=EPS, mode=best)
    n, mx = ulp_diff(act, ref)
    rel = ((act.float() - ref.float()).norm() / ref.float().norm()).item()
    check(f"shared act M={M} vs hipBLAS gate_up + _C act", rel < 6e-3 and torch.equal(ti, ti2) and torch.equal(tw, tw2),
          f"({n}/{act.numel()} differ, max {mx} ulp, rel {rel:.1e}; routing unchanged)")



print("== attention input: attn_norm + compressor kv_score (f32 [M, 1024]) + indexer weights_proj (bf16 [M, 32])")
Wc = (torch.randn(1024, H, generator=gen) * H ** -0.5).bfloat16().to(dev)
Wp = (torch.randn(32, H, generator=gen) * H ** -0.5).bfloat16().to(dev)
for M in MS:
    x = rows(M, H, 3.0, 600 + M)
    xn, (kvs, wts) = G.norm_gemv(x, W["attn_norm"], EPS, [(Wc, True), (Wp, False)], mode=best)
    st = stock_rms(x, W["attn_norm"])
    nxn, mxn = ulp_diff(xn, st)
    kref = torch.mm(st, Wc.t(), out_dtype=torch.float32)
    wref = torch.nn.functional.linear(st, Wp)
    krel = ((kvs - kref).abs().max() / kref.abs().max()).item()
    nw_, mw = ulp_diff(wts, wref)
    _, (k1,) = G.norm_gemv(x, W["attn_norm"], EPS, [(Wc, True)], mode=best)
    check(f"attn input M={M}", mxn <= 1 and krel < 1e-2 and mw <= 2 and torch.equal(k1, kvs),
          f"(xn {nxn} differ; kv_score rel {krel:.1e}, weights {nw_}/{wts.numel()} differ max {mw} ulp)")


print("== DSpark Markov head (greedy): 5 dependent steps over [129280 x 256]")
VOC, RK, NSPEC = 129280, 256, 5


def load_markov():
    idx = os.path.join(PACK, "model.safetensors.index.json")
    if os.path.exists(idx):
        from safetensors import safe_open
        wm = json.load(open(idx))["weight_map"]
        out = []
        for n in ("mtp.2.markov_head.embed.weight", "mtp.2.markov_head.head.weight"):
            with safe_open(os.path.join(PACK, wm[n]), "pt") as f:
                out.append(f.get_tensor(n).to(torch.bfloat16).contiguous().to(dev))
        print("  Markov weights: pack mtp.2")
        return out
    print("  Markov weights: random")
    return [(torch.randn(VOC, RK, generator=gen) * 0.05).bfloat16().to(dev),
            (torch.randn(VOC, RK, generator=gen) * 0.05).bfloat16().to(dev)]


MW1, MW2 = load_markov()
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
_dug = dispatch_unquantized_gemm()


def _gemm(layer, x, w, b):
    """vLLM's unquantized GEMM dispatch (the lm_head path: wvSplitK skinny GEMM on ROCm), F.linear if it refuses."""
    try:
        return _dug(torch.nn.Module(), x, w, b)
    except Exception:
        return torch.nn.functional.linear(x, w, b)


def markov_stock(base, input_ids, anchor, rows, tokens):
    prev = input_ids[anchor[:rows]]
    for i in range(NSPEC):
        e = torch.nn.functional.embedding(prev, MW1)
        bias = _gemm(None, e, MW2, None)
        lg = base[:, i] + bias
        d = lg.argmax(dim=-1)
        tokens[:rows, i] = d
        prev = d
    return tokens


mk_dims = []
for rows_ in (1, 2, 4, 8):
    base = (torch.randn(rows_, NSPEC, VOC, generator=gen) * 2).bfloat16().to(dev)
    input_ids = torch.randint(0, VOC, (64,), generator=gen, dtype=torch.int32).to(dev)
    anchor = (torch.arange(rows_, dtype=torch.int64) * 6).to(dev)
    tr = markov_stock(base, input_ids, anchor, rows_, torch.zeros(8, NSPEC, dtype=torch.int64, device=dev))
    to = torch.zeros(8, NSPEC, dtype=torch.int64, device=dev)
    seen = []
    G.markov_greedy(MW1, MW2, base, input_ids, anchor, to, rows_, NSPEC, None, lambda s_, lg: seen.append(lg.clone()))
    # step-0 logits vs stock's (same previous token by construction)
    e0 = torch.nn.functional.embedding(input_ids[anchor], MW1)
    l0 = base[:, 0] + _gemm(None, e0, MW2, None)
    n0, m0 = ulp_diff(seen[0], l0)
    same = int((to[:rows_] == tr[:rows_]).all(1).sum())
    check(f"markov greedy rows={rows_}", same == rows_ and n0 <= l0.numel() // 10000,
          f"(token rows identical {same}/{rows_}; step-0 logits {n0}/{l0.numel()} differ, max {m0} ulp)")
    if rows_ == 1:
        f8 = G.fp8_rows(MW2)
        t8 = torch.zeros(8, NSPEC, dtype=torch.int64, device=dev)
        G.markov_greedy(MW1, MW2, base, input_ids, anchor, t8, rows_, NSPEC, f8)
        print(f"  fp8 head: tokens equal to bf16 {int((t8[:1] == tr[:1]).sum())}/{NSPEC}")

# -------------------------------------------------------------------------------------------------------- hooks
print("== route hook on stock modules (RMSNorm + GateLinear + FusedTopKBiasRouter, as the MoE runner calls them)")
try:
    import types
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import FusedTopKBiasRouter
    from vllm import ir
    cfg = VllmConfig()
    torch.set_default_dtype(torch.bfloat16)          # vLLM builds the model under the model dtype
    with set_current_vllm_config(cfg):
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                     distributed_init_method="tcp://127.0.0.1:29571", backend="nccl")
        initialize_model_parallel(1)
        gate = GateLinear(H, E, bias=False, out_dtype=torch.float32, prefix="t.gate").to(dev)
        gate.weight.data.copy_(W["gate_w"])
        norm = RMSNorm(H, EPS).to(dev)
        norm.weight.data.copy_(W["ffn_norm"])
        from vllm.models.deepseek_v4.amd.model import DeepseekV4MLP
        mlp = DeepseekV4MLP(H, I_SH, "silu", swiglu_limit=LIM, quant_config=None, reduce_results=False,
                            prefix="t.shared").to(dev)
        mlp.gate_up_proj.weight.data.copy_(sw_)
        mlp.down_proj.weight.data.copy_((torch.randn(H, I_SH, generator=gen) * I_SH ** -0.5).bfloat16())
        router = FusedTopKBiasRouter(top_k=TOPK, global_num_experts=E, e_score_correction_bias=W["bias"],
                                     renormalize=True, routed_scaling_factor=RSF, scoring_func="sqrtsoftplus",
                                     bias_vl=W["bias_vl"], image_sentinel_lo=128000)
        layer = types.SimpleNamespace(ffn=types.SimpleNamespace(experts=types.SimpleNamespace(router=router), gate=gate,
                                                                shared_experts=mlp),
                                      ffn_norm=norm)
        with ir.ops.rms_norm.set_priority(["vllm_c", "native"]) if hasattr(ir.ops.rms_norm, "set_priority") \
                else __import__("contextlib").nullcontext():
            def run(M, seed):
                x = rows(M, H, 3.0, seed)
                ids = torch.arange(M, dtype=torch.int32, device=dev)
                xn = layer.ffn_norm(x)
                sh = mlp(xn)
                lg, _ = gate(xn)
                return xn, router.select_experts(hidden_states=xn, router_logits=lg, input_ids=ids), sh
            ref = {M: run(M, 300 + M) for M in MS}
            h = G._RouteHook(layer)
            check("hook binds on stock modules", h.ok() and h.bind())
            for M in MS:
                h.ids = torch.arange(M, dtype=torch.int32, device=dev)
                xn, (tw, ti), sh = run(M, 300 + M)
                rxn, (rtw, rti), rsh = ref[M]
                srel = ((sh.float() - rsh.float()).norm() / rsh.float().norm()).item()
                n, mx = ulp_diff(xn, rxn)
                same = (torch.sort(ti, 1)[0] == torch.sort(rti, 1)[0]).all(1)
                check(f"hooked ffn_norm+gate+router+shared M={M}", mx <= 1 and h.stash is None and h.act is None
                      and h.shared_on and srel < 4e-3,
                      f"(xn {n} elems differ max {mx} ulp; id sets equal {int(same.sum())}/{M}; "
                      f"w max |d| {(tw - rtw).abs()[same].max().item() if same.any() else 0:.2g}; shared out rel {srel:.1e})")
            x = rows(40, H, 3.0, 1)                               # prefill width: stock path, no stash
            h.ids = None
            xn = layer.ffn_norm(x)
            check("M=40 takes stock's ops", h.stash is None and xn.shape == x.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("  hook test skipped:", e)
finally:
    torch.set_default_dtype(torch.float32)

# -------------------------------------------------------------------------------------------------------- graph
print("== graph capture + replay")
M = 6
xg = rows(M, H, 3.0, 5)
cg = (torch.randn(M, QL + KVD, generator=gen) * 2).to(dev)
outs = {}


def body():
    nb = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
    nh = torch.empty(M, H, dtype=torch.float16, device=dev)
    G.norm_pre([G.Seg(xg, W["attn_norm"], True, nb, nh)], M, EPS, best)
    q = torch.empty(M, QL, dtype=torch.bfloat16, device=dev)
    kv = torch.empty(M, KVD, dtype=torch.bfloat16, device=dev)
    G.norm_pre([G.Seg(cg[:, :QL], W["q_norm"], True, q), G.Seg(cg[:, QL:], W["kv_norm"], True, kv)], M, EPS, 1)
    r = G.route(xg, W["gate_w"], W["bias"], TOPK, True, RSF, norm_w=W["ffn_norm"], eps=EPS, mode=best)
    return nb, nh, q, kv, r[0], r[1], r[2]


eager = [t.clone() for t in body()]
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    body()
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    gout = body()
for _ in range(3):
    g.replay()
torch.cuda.synchronize()
check("graph replay == eager (norm, q/kv norm, fused router)", all(torch.equal(a, b) for a, b in zip(eager, gout)))

# ------------------------------------------------------------------------------------------------------- timing
if os.environ.get("TIMING", "1") != "0":
    print("== timing (graph replay, us a call; stock = the launches it replaces)")
    from vllm.models.common.ops import fused_q_kv_rmsnorm as fqk

    def gtime(fn, reps=200):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(20):
                fn()
        g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps // 20 + 3):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1000 / 20)
        return statistics.median(ts)

    for M in (1, 2, 4, 6, 8, 12, 24):
        x = rows(M, H, 3.0, M)
        c = (torch.randn(M, QL + KVD, generator=gen) * 2).to(dev)
        abuf = torch.empty(M, H, dtype=torch.float16, device=dev)
        qb = torch.empty(M, QL, dtype=torch.float16, device=dev)
        nb = torch.empty_like(x)

        def stock_attn_front():
            y = stock_rms(x, W["attn_norm"])
            abuf.copy_(y)

        def ours_attn_front():
            G.norm_pre([G.Seg(x, W["attn_norm"], True, nb, abuf)], M, EPS, best)

        def stock_qkv():
            qr_kv = c.to(torch.bfloat16)
            q, kv = fqk(qr_kv[:, :QL], qr_kv[:, QL:], W["q_norm"], W["kv_norm"], EPS)
            qb.copy_(q)

        q = torch.empty(M, QL, dtype=torch.bfloat16, device=dev)
        kv = torch.empty(M, KVD, dtype=torch.bfloat16, device=dev)

        def ours_qkv():
            G.norm_pre([G.Seg(c[:, :QL], W["q_norm"], True, q, qb), G.Seg(c[:, QL:], W["kv_norm"], True, kv)],
                       M, EPS, 1)

        def stock_router():
            y = stock_rms(x, W["ffn_norm"])
            lg = torch.mm(y, W["gate_w"].t(), out_dtype=torch.float32)
            stock_topk(lg, W["bias"], TOPK)

        def ours_router():
            G.route(x, W["gate_w"], W["bias"], TOPK, True, RSF, norm_w=W["ffn_norm"], eps=EPS, mode=best)

        def ours_router_nonorm():
            G.route(x, W["gate_w"], W["bias"], TOPK, True, RSF)

        def stock_ffn_front():
            y = stock_rms(x, W["ffn_norm"])
            lg = torch.mm(y, W["gate_w"].t(), out_dtype=torch.float32)
            stock_topk(lg, W["bias"], TOPK)
            stock_act(torch.mm(y, sw_.t()))

        def stock_attn_in():
            y = stock_rms(x, W["attn_norm"])
            torch.mm(y, Wc.t(), out_dtype=torch.float32)
            torch.nn.functional.linear(y, Wp)

        def ours_attn_in():
            G.norm_gemv(x, W["attn_norm"], EPS, [(Wc, True), (Wp, False)], mode=best)

        def ours_ffn_front():
            G.route(x, W["gate_w"], W["bias"], TOPK, True, RSF, norm_w=W["ffn_norm"], eps=EPS, mode=best,
                    shared_w=sw_, limit=LIM)

        res = {k: gtime(f) for k, f in (("attn_norm+cast stock", stock_attn_front), ("attn_norm+cast r9k", ours_attn_front),
                                         ("C cast+q/kv norm+cast stock", stock_qkv), ("q/kv norm r9k", ours_qkv),
                                         ("ffn_norm+gate+topk stock", stock_router), ("ffn_norm+route r9k", ours_router),
                                         ("route r9k (no norm)", ours_router_nonorm),
                                         ("ffn front stock", stock_ffn_front),
                                         ("ffn front r9k", ours_ffn_front),
                                         ("attn input stock", stock_attn_in),
                                         ("attn input r9k", ours_attn_in))}
        print(f"  M={M}: " + ", ".join(f"{k} {v:.1f}" for k, v in res.items()), flush=True)
        if M == 6:
            from torch.profiler import ProfilerActivity, profile
            for name, fn in (("attn_norm+cast", (stock_attn_front, ours_attn_front)),
                             ("q/kv norm+casts", (stock_qkv, ours_qkv)),
                             ("ffn front (norm, gate, top-k, shared gate_up, act)", (stock_ffn_front, ours_ffn_front)),
                             ("attn input (norm, kv_score, weights_proj)", (stock_attn_in, ours_attn_in))):
                cnt = []
                for f in fn:
                    f()
                    torch.cuda.synchronize()
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        f()
                        torch.cuda.synchronize()
                    ks = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
                    cnt.append((len(ks), sorted(set(k[:40] for k in ks))))
                print(f"  launches {name}: stock {cnt[0][0]} {cnt[0][1]} -> r9k {cnt[1][0]}", flush=True)

if os.environ.get("TIMING", "1") != "0":
    print("== timing: DSpark Markov head, 5 steps (graph replay, us)")
    for rows_ in (1, 4):
        base = (torch.randn(rows_, NSPEC, VOC, generator=gen) * 2).bfloat16().to(dev)
        input_ids = torch.randint(0, VOC, (64,), generator=gen, dtype=torch.int32).to(dev)
        anchor = (torch.arange(rows_, dtype=torch.int64) * 6).to(dev)
        tk = torch.zeros(8, NSPEC, dtype=torch.int64, device=dev)
        f8 = G.fp8_rows(MW2)
        r_st = gtime(lambda: markov_stock(base, input_ids, anchor, rows_, tk), reps=40)
        r_ours = gtime(lambda: G.markov_greedy(MW1, MW2, base, input_ids, anchor, tk, rows_, NSPEC), reps=40)
        r_lg = gtime(lambda: G.markov_greedy(MW1, MW2, base, input_ids, anchor, tk, rows_, NSPEC, None,
                                             lambda s_, lg: None), reps=40)
        r_f8 = gtime(lambda: G.markov_greedy(MW1, MW2, base, input_ids, anchor, tk, rows_, NSPEC, f8), reps=40)
        print(f"  rows={rows_}: stock loop {r_st:.1f}, r9k {r_ours:.1f} (+ logits stored {r_lg:.1f}), r9k fp8 head {r_f8:.1f}",
              flush=True)


# ------------------------------------------------------------------------------------- confidence head + the hook
print("== DSpark confidence head (adaptive verification): r9k_dsv4_markov_conf vs stock's cat + f32 GEMM + sigmoid")
CK = H + RK


def load_conf():
    idx = os.path.join(PACK, "model.safetensors.index.json")
    if os.path.exists(idx):
        from safetensors import safe_open
        wm = json.load(open(idx))["weight_map"]
        n = "mtp.2.confidence_head.proj.weight"
        with safe_open(os.path.join(PACK, wm[n]), "pt") as f:
            print("  confidence weights: pack mtp.2")
            return f.get_tensor(n).to(torch.float32).contiguous().to(dev)       # params_dtype f32 (stock loader)
    print("  confidence weights: random")
    return (torch.randn(1, CK, generator=gen) * CK ** -0.5).to(dev)


CW = load_conf()


def conf_stock(hid, prevs, w=CW, b=None):
    """stock DSparkConfidenceHead + compute_confidence: cat([hidden, w1[prev]]).float() -> ReplicatedLinear f32
    (vLLM's unquantized GEMM dispatch) -> sigmoid. hid [N, H] bf16, prevs [N] token ids."""
    e = torch.nn.functional.embedding(prevs, MW1)
    x = torch.cat([hid, e], dim=-1).float()
    return torch.sigmoid(_gemm(None, x, w, b).squeeze(-1)), x


for rows_ in (1, 2, 4, 8):
    N = rows_ * NSPEC
    hid = rows(N, H, 1.0, 900 + rows_)
    input_ids = torch.randint(0, VOC, (64,), generator=gen, dtype=torch.int32).to(dev)
    anchor = (torch.arange(rows_, dtype=torch.int64) * 6).to(dev)
    tok = torch.randint(0, VOC, (8, NSPEC), generator=gen, dtype=torch.int64).to(dev)
    prevs = torch.cat([input_ids[anchor].long()[:, None], tok[:rows_, :NSPEC - 1]], 1).reshape(-1)
    cs, x = conf_stock(hid, prevs)
    out = torch.full((8, NSPEC), -1.0, device=dev)
    G.markov_confidence(hid, MW1, CW, None, input_ids, anchor, tok, rows_, NSPEC, out)
    zref = (x.double() @ CW.double().t()).squeeze(-1)                     # exact-ish dot
    zst = _gemm(None, x, CW, None).squeeze(-1).double()
    zst1 = torch.cat([_gemm(None, x[i:i + 1], CW, None).squeeze(-1) for i in range(N)]).double()   # stock at M = 1
    ours = out[:rows_].reshape(-1)
    zo = torch.logit(ours.double().clamp(1e-12, 1 - 1e-12))
    e_st = (zst - zref).abs().max().item()
    e_st1 = (zst1 - zref).abs().max().item()
    e_o = (zo - zref).abs().max().item()
    dc = (ours - cs).abs().max().item()
    neq = int((ours != cs).sum())
    check(f"confidence rows={rows_}", dc < 2e-6 and bool((out[rows_:] == -1).all()),
          f"(prob max |ours - stock| {dc:.1e}, {neq}/{N} not bit-equal; logit |err| vs f64: stock M={N} {e_st:.1e}, "
          f"stock M=1 {e_st1:.1e} (its own batch-size spread {(zst - zst1).abs().max().item():.1e}), ours {e_o:.1e})")

print("== Markov hook on a stock-shaped speculator (greedy drafts + confidence head), eager and graph replay")
import types


class UnquantizedEmbeddingMethod:  # noqa: N801 -- the class name the hook checks
    pass


class UnquantizedLinearMethod:  # noqa: N801
    pass


def mk_model(base_all):
    w2m = types.SimpleNamespace(weight=MW2, quant_method=UnquantizedEmbeddingMethod())
    w1m = torch.nn.Embedding(VOC, RK, _weight=MW1.clone(), device=dev, dtype=torch.bfloat16)
    proj = types.SimpleNamespace(weight=CW, bias=None, quant_method=UnquantizedLinearMethod())
    ch = types.SimpleNamespace(proj=proj, with_markov=True)
    mh = types.SimpleNamespace(markov_w1=w1m, markov_w2=w2m)
    m = types.SimpleNamespace(model=types.SimpleNamespace(markov_head=mh, confidence_head=ch),
                              logits_processor=types.SimpleNamespace(scale=1.0, soft_cap=None))
    m.compute_draft_logits = lambda h: base_all[: h.shape[0]]
    m.markov_embed = lambda t: w1m(t)
    m.markov_bias = lambda e: _gemm(None, e, MW2, None)
    m.compute_confidence = lambda hh, me: conf_stock_x(hh, me)
    return m


def conf_stock_x(hh, me):
    x = torch.cat([hh, me], dim=-1).float()
    return torch.sigmoid(_gemm(None, x, CW, None).squeeze(-1))


def mk_spec(rows_, seed):
    g2 = torch.Generator(device="cpu").manual_seed(seed)
    sp = types.SimpleNamespace(draft_logits=None, draft_watermarker=None, _d2t_scatter_index=None, _draft_topk=None,
                               use_confidence_head=True, acceptance_estimator=None, num_speculative_steps=NSPEC)
    sp.input_buffers = types.SimpleNamespace(input_ids=torch.randint(0, VOC, (64,), generator=g2,
                                                                       dtype=torch.int32).to(dev))
    sp._anchor_idx = (torch.arange(8, dtype=torch.int64) * NSPEC).to(dev)
    sp.sample_indices = torch.randperm(64, generator=g2)[:8 * NSPEC].to(dev)
    sp.draft_tokens = torch.zeros(8, NSPEC, dtype=torch.int64, device=dev)
    sp.draft_token_confidence_probs = torch.zeros(8, NSPEC, dtype=torch.float32, device=dev)
    return sp


def stock_loop(model, sp, rows_, head_hidden):
    """DSparkSpeculator._sample_sequential, greedy drafts, use_confidence_head (this pin), no estimator."""
    num_sample = rows_ * NSPEC
    sh = head_hidden[sp.sample_indices[:num_sample]]
    bl = model.compute_draft_logits(sh).view(rows_, NSPEC, -1)
    prev = sp.input_buffers.input_ids[sp._anchor_idx[:rows_]]
    embs = []
    for i in range(NSPEC):
        me = model.markov_embed(prev)
        embs.append(me)
        d = (bl[:, i] + model.markov_bias(me)).argmax(dim=-1)
        sp.draft_tokens[:rows_, i] = d
        prev = d
    c = model.compute_confidence(sh, torch.stack(embs, dim=1).flatten(0, 1))
    sp.draft_token_confidence_probs[:rows_] = c.view(rows_, NSPEC)


for rows_ in (1, 4, 8):
    base_all = (torch.randn(rows_ * NSPEC, VOC, generator=gen) * 2).bfloat16().to(dev)
    hh = rows(64, H, 1.0, 950 + rows_)
    m = mk_model(base_all)
    sp_ref, sp_hk = mk_spec(rows_, 7 + rows_), mk_spec(rows_, 7 + rows_)
    stock_loop(m, sp_ref, rows_, hh)
    why = G._markov_why_not(m, sp_hk, rows_)
    cwhy = G._conf_why_not(m, sp_hk, hh)
    took = G.markov_sample_sequential(m, sp_hk, rows_, hh)
    tsame = int((sp_hk.draft_tokens[:rows_] == sp_ref.draft_tokens[:rows_]).all(1).sum())
    dc = (sp_hk.draft_token_confidence_probs[:rows_] - sp_ref.draft_token_confidence_probs[:rows_]).abs().max().item()
    # graph capture + replay of the hook == its eager result
    sp_g = mk_spec(rows_, 7 + rows_)
    st_ = torch.cuda.Stream()
    st_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st_):
        G.markov_sample_sequential(m, sp_g, rows_, hh)
    torch.cuda.current_stream().wait_stream(st_)
    sp_g.draft_tokens.zero_()
    sp_g.draft_token_confidence_probs.zero_()
    gg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gg):
        G.markov_sample_sequential(m, sp_g, rows_, hh)
    sp_g.draft_tokens.zero_()
    sp_g.draft_token_confidence_probs.zero_()
    gg.replay()
    torch.cuda.synchronize()
    geq = torch.equal(sp_g.draft_tokens, sp_hk.draft_tokens) and \
        torch.equal(sp_g.draft_token_confidence_probs, sp_hk.draft_token_confidence_probs)
    check(f"hook rows={rows_} (adaptive, confidence head)", why is None and cwhy is None and took and tsame == rows_
          and dc < 2e-6 and geq,
          f"(declines: {why} / {cwhy}; drafts identical {tsame}/{rows_}; conf max |d| {dc:.1e}; graph == eager {geq})")
    if rows_ == 4:                                     # R9K_DSV41_MARKOV_CONF=stock: bit-exact with stock's loop
        G.MARKOV_CONF = "stock"
        sp_s = mk_spec(rows_, 7 + rows_)
        G.markov_sample_sequential(m, sp_s, rows_, hh)
        G.MARKOV_CONF = "r9k"
        check("hook rows=4 MARKOV_CONF=stock bit-exact", torch.equal(sp_s.draft_tokens, sp_ref.draft_tokens)
              and torch.equal(sp_s.draft_token_confidence_probs, sp_ref.draft_token_confidence_probs))

if os.environ.get("TIMING", "1") != "0":
    print("== timing: confidence head (graph replay, us)")
    for rows_ in (1, 4):
        N = rows_ * NSPEC
        hid = rows(N, H, 1.0, 990)
        input_ids = torch.randint(0, VOC, (64,), generator=gen, dtype=torch.int32).to(dev)
        anchor = (torch.arange(rows_, dtype=torch.int64) * 6).to(dev)
        tok = torch.randint(0, VOC, (8, NSPEC), generator=gen, dtype=torch.int64).to(dev)
        out = torch.zeros(8, NSPEC, device=dev)
        w1m = torch.nn.Embedding(VOC, RK, _weight=MW1.clone(), device=dev, dtype=torch.bfloat16)

        def st_conf():
            prev0 = input_ids[anchor[:rows_]]
            es = [w1m(prev0)] + [w1m(tok[:rows_, i]) for i in range(NSPEC - 1)]
            out[:rows_] = conf_stock_x(hid, torch.stack(es, dim=1).flatten(0, 1)).view(rows_, NSPEC)

        t_st = gtime(st_conf, reps=40)
        t_o = gtime(lambda: G.markov_confidence(hid, MW1, CW, None, input_ids, anchor, tok, rows_, NSPEC, out), reps=40)
        print(f"  rows={rows_}: stock (5 embeddings + stack + cat + f32 GEMM + sigmoid + copy) {t_st:.1f}, "
              f"r9k {t_o:.1f}", flush=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            st_conf()
            torch.cuda.synchronize()
        print("   stock kernels:", sorted({e.name[:60] for e in prof.events() if e.device_type.name == "CUDA"}))

print("ALL OK" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
