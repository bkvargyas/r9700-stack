"""DeepSeek-V4.1 hyper-connection seams on RDNA4: hc_dsv4.py / kernels/r9k_mhc.hip vs vLLM's torch reference
(model_executor/kernels/mhc/torch.py: mhc_post_torch + mhc_pre_delayed_torch) and vs stock's TileLang ops
(mhc_post_tilelang + mhc_pre_delayed_tilelang, what RDNA4 runs today), at M = 1, 2, 6, 24, 32 with the real shapes
(hc_mult 4, hidden 5120, fn [24, 20480]) and the attn and ffn parameter sets of one pack layer (random ones when the
pack is absent). Cases: the seam (post folded into the pre, the common call), the pre without a post (engram
layers / pipeline entry), the layer-0 broadcast (x [T, 5120] with the folded fn), the standalone post. Checks: new
residual and layer input bit-exact vs torch, gates within f32 tolerance, batch invariance (row 0 alone vs in a batch
of 32, bitwise), graph capture + replay bitwise vs eager. Then graph-captured timing per launch (median of replays):
our seam vs stock's three launches. Run in the ROCm image with PYTHONPATH=/opt/r9700 on one GPU (< 1 GiB).
TIMING=0 skips timing; PACK=<dir> (default the EXL3 pack under /ds41), LAYER=<n> (default 7)."""
import importlib, json, os, statistics, sys
import torch
import vllm  # noqa: F401
import vllm.model_executor.kernels.mhc  # noqa: F401  (registers the stock ops)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from r9700_vllm import hc_dsv4 as D

TR = importlib.import_module("vllm.model_executor.kernels.mhc.torch")
dev = torch.device("cuda:0")
HC, H, K = 4, 5120, 4 * 5120
RMS_EPS, HC_EPS, POST_MULT, ITERS = 1e-20, 1e-6, 2.0, 20
EPS_ARGS = (RMS_EPS, HC_EPS, HC_EPS, POST_MULT, ITERS)
PACK = os.environ.get("PACK", "/ds41/exl3/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw")
LAYER = int(os.environ.get("LAYER", "7"))
ok_all = True
g = torch.Generator(device="cpu").manual_seed(5)


def check(name, cond, msg=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {msg}", flush=True)


def load_params():
    """{attn, ffn}: (fn [24, 20480], scale [3], base [24]) of pack layer LAYER, else random at the pack's scale."""
    idx = os.path.join(PACK, "model.safetensors.index.json")
    out = {}
    if os.path.exists(idx):
        from safetensors import safe_open
        wm = json.load(open(idx))["weight_map"]
        for part in ("attn", "ffn"):
            names = [f"layers.{LAYER}.hc_{part}_{s}" for s in ("fn", "scale", "base")]
            ts = []
            for n in names:
                with safe_open(os.path.join(PACK, wm[n]), "pt") as f:
                    ts.append(f.get_tensor(n).float().contiguous().to(dev))
            out[part] = tuple(ts)
        print(f"params: pack layer {LAYER} ({PACK})")
        return out
    for part in ("attn", "ffn"):
        out[part] = ((torch.randn(24, K, generator=g) * K ** -0.5).to(dev),
                     (torch.rand(3, generator=g) + 0.2).to(dev), (torch.randn(24, generator=g) * 0.3).to(dev))
    print("params: random (no pack index found)")
    return out


def inputs(M, seed):
    gg = torch.Generator(device="cpu").manual_seed(seed)
    res = (torch.randn(M, HC, H, generator=gg) * 0.8).to(torch.bfloat16).to(dev)
    so = (torch.randn(M, H, generator=gg) * 0.5).to(torch.bfloat16).to(dev)
    post = (torch.rand(M, HC, 1, generator=gg) * POST_MULT).to(dev)
    c = torch.softmax(torch.randn(M, HC, HC, generator=gg) * 2, -1)
    for _ in range(10):
        c = c / c.sum(-2, keepdim=True)
        c = c / c.sum(-1, keepdim=True)
    pre = (torch.sigmoid(torch.randn(M, HC, generator=gg)) + HC_EPS).to(dev)
    return res, so, post, c.contiguous().to(dev), pre


def ref_seam(res, so, post, comb, fn, sc, base, pre):
    r = TR.mhc_post_torch(so, res, post, comb) if so is not None else res
    p, cm, li, nx = TR.mhc_pre_delayed_torch(r, fn, sc, base, *EPS_ARGS, pre_mix=pre)
    return r, p, cm, li, nx


def tl_seam(res, so, post, comb, fn, sc, base, pre):
    r = torch.ops.vllm.mhc_post_tilelang(so, res, post, comb) if so is not None else res
    return (r, *torch.ops.vllm.mhc_pre_delayed_tilelang(r, fn, sc, base, *EPS_ARGS, pre, None, None, 1e-6))


def ours(res, so, post, comb, fn, sc, base, pre):
    if so is None:
        return (res, *D.mhc_pre(res, fn, sc, base, *EPS_ARGS, pre))
    return D.mhc_post_pre(res, so, post, comb, fn, sc, base, *EPS_ARGS, pre)


def ulp_bf16(a, b):
    ai, bi = a.view(torch.int16).int(), b.view(torch.int16).int()
    return int((ai - bi).abs().max()) if a.numel() else 0


def gate_err(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(((a - b).abs() / (b.abs() + 1e-6)).max())


def close_bf16(a, b):
    """within one bf16 rounding of each other: |a - b| <= 2^-7 |b| + 2^-9 mean|b| (the absolute term covers sums
    that cancel to near zero, where an input's 1-ulp difference is many ulp of the result); (ok, n_diff, max)."""
    d = (a.float() - b.float()).abs()
    ok = bool((d <= b.float().abs() * 2 ** -7 + b.float().abs().mean() * 2 ** -9).all())
    return ok, int((a != b).sum()), float(d.max())


def compare(tag, got, ref, exact=True):
    """exact: residual' / layer input bitwise (vs torch); else within one bf16 rounding (vs TileLang, which rounds
    its post and collapse differently from torch)."""
    r, p, cm, li, nx = got
    rr, rp, rcm, rli, rnx = ref
    for name, a, b in (("residual'", r, rr), ("layer_input", li, rli)):
        if exact:
            check(f"{tag} {name}", torch.equal(a, b), f"max ulp {ulp_bf16(a, b)}")
        else:
            ok, n, mx = close_bf16(a, b)
            check(f"{tag} {name}", ok, f"{n} of {a.numel()} differ (max |d| {mx:.2e})")
    e = max(gate_err(p, rp), gate_err(cm, rcm), gate_err(nx, rnx))
    check(f"{tag} gates", e < 1e-3 and p.shape == rp.shape and cm.shape == rcm.shape and nx.shape == rnx.shape,
          f"max rel err post/comb/pre {gate_err(p, rp):.1e} / {gate_err(cm, rcm):.1e} / {gate_err(nx, rnx):.1e}")


def graph_us(f, reps=20, inner=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(inner):
            f()
    gr.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        gr.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / inner)
    return statistics.median(ts)


def main():
    assert D.available(), "libr9k has no r9k_mhc_seam (rebuild kernels/build.sh)"
    params = load_params()
    for M in (1, 2, 6, 24, 32):
        print(f"M={M}")
        for part, (fn, sc, base) in params.items():
            res, so, post, comb, pre = inputs(M, 100 + M)
            got = ours(res, so, post, comb, fn, sc, base, pre)
            compare(f"seam/{part} vs torch", got, ref_seam(res, so, post, comb, fn, sc, base, pre))
            compare(f"seam/{part} vs tilelang", got, tl_seam(res, so, post, comb, fn, sc, base, pre), exact=False)
            got = ours(res, None, None, None, fn, sc, base, pre)
            compare(f"pre/{part} vs torch", got, ref_seam(res, None, None, None, fn, sc, base, pre))
        fn, sc, base = params["attn"]
        res, so, post, comb, pre = inputs(M, 200 + M)
        fnb = fn.view(24, HC, H).sum(1).contiguous()
        x = so
        resx = x.unsqueeze(1).expand(-1, HC, -1).contiguous()
        p, cm, li, nx = D.mhc_pre(resx, fnb, sc, base, *EPS_ARGS, None, x)
        rp, rcm, rli, rnx = TR.mhc_pre_delayed_torch(resx, fnb, sc, base, *EPS_ARGS, pre_mix=None, x=x)
        compare("broadcast vs torch", (resx, p, cm, li, nx), (resx, rp, rcm, rli, rnx))
        out = D.mhc_post(so, res, post, comb)
        ref = TR.mhc_post_torch(so, res, post, comb)
        tl = torch.ops.vllm.mhc_post_tilelang(so, res, post, comb)
        check("post vs torch", torch.equal(out, ref), f"max ulp {ulp_bf16(out, ref)}")
        ok, n, mx = close_bf16(out, tl)
        check("post vs tilelang", ok, f"{n} of {out.numel()} differ (max |d| {mx:.2e})")

    print("batch invariance + graph replay")
    fn, sc, base = params["ffn"]
    res, so, post, comb, pre = inputs(32, 7)
    full = ours(res, so, post, comb, fn, sc, base, pre)
    for M in (1, 6, 24):
        part = ours(res[:M].contiguous(), so[:M].contiguous(), post[:M].contiguous(), comb[:M].contiguous(), fn, sc,
                    base, pre[:M].contiguous())
        check(f"rows 0..{M - 1} alone == in a batch of 32", all(torch.equal(a, b[:M]) for a, b in zip(part, full)))
    for M in (1, 24):
        args = [t[:M].contiguous() for t in (res, so, post, comb)] + [fn, sc, base, pre[:M].contiguous()]
        eager = ours(*args)
        outs = {}
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            ours(*args)
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            outs["a"] = ours(*args)
            outs["b"] = ours(outs["a"][0], *args[1:7], outs["a"][4])          # chained: the next seam on its output
        for rep in range(3):
            gr.replay()
            torch.cuda.synchronize()
            ok = all(torch.equal(a, b) for a, b in zip(outs["a"], eager))
            chained = ours(eager[0], *args[1:7], eager[4])
            ok &= all(torch.equal(a, b) for a, b in zip(outs["b"], chained))
            check(f"graph replay {rep} M={M} (two chained seams) == eager", ok)

    print("hook: MHCPreDelayedOp / MHCPostOp after deepseek_v41_rdna4._install_mhc, eager and torch.compile'd")
    from r9700_vllm.models import deepseek_v41_rdna4 as R
    from vllm.model_executor.layers import mhc as ML
    check("install", R._install_mhc() and ML.MHCPreDelayedOp.forward_hip.__r9k_patch__ == "dsv41_mhc")
    from vllm.config import VllmConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig()):
        pre_op, post_op = ML.MHCPreDelayedOp(), ML.MHCPostOp()
    fn, sc, base = params["attn"]

    def layer(res, so, post, comb, pre):            # an attention seam, then a standalone post (the model's shapes)
        r, p, cm, li, nx = pre_op(res, fn, sc, base, *EPS_ARGS, pre_mix=pre, sublayer_out=so, post_layer_mix=post,
                                  comb_res_mix=comb)
        return r, p, cm, li, nx, post_op(so, r, p, cm)

    compiled = torch.compile(layer, dynamic=True, fullgraph=True)
    for M in (1, 6, 24, 40):                          # 40: wider than the kernels, stock's ops inside the same op
        res, so, post, comb, pre = inputs(M, 400 + M)
        want = list(ours(res, so, post, comb, fn, sc, base, pre))
        want.append(D.mhc_post(so, want[0], want[1], want[2]))
        if M > D.max_m():
            want = list(tl_seam(res, so, post, comb, fn, sc, base, pre))
            want.append(torch.ops.vllm.mhc_post_tilelang(so, want[0], want[1], want[2]))
        for name, f in (("eager", layer), ("compiled", compiled)):
            got = f(res, so, post, comb, pre)
            check(f"{name} op M={M} == direct call", all(torch.equal(a, b) for a, b in zip(got, want)))

    if os.environ.get("TIMING", "1") != "0":
        print("timing (graph replay, us a call, median)")
        print(f"{'M':>4} {'stock seam (3 launches)':>24} {'r9k seam':>9} {'stock post':>11} {'r9k post':>9} "
              f"{'stock pre (2)':>14} {'r9k pre':>8}")
        for M in (1, 2, 6, 24, 32):
            fn, sc, base = params["attn"]
            res, so, post, comb, pre = inputs(M, 300 + M)
            t_stock = graph_us(lambda: tl_seam(res, so, post, comb, fn, sc, base, pre))
            t_ours = graph_us(lambda: D.mhc_post_pre(res, so, post, comb, fn, sc, base, *EPS_ARGS, pre))
            t_sp = graph_us(lambda: torch.ops.vllm.mhc_post_tilelang(so, res, post, comb))
            t_op = graph_us(lambda: D.mhc_post(so, res, post, comb))
            t_spre = graph_us(lambda: torch.ops.vllm.mhc_pre_delayed_tilelang(res, fn, sc, base, *EPS_ARGS, pre, None,
                                                                              None, 1e-6))
            t_opre = graph_us(lambda: D.mhc_pre(res, fn, sc, base, *EPS_ARGS, pre))
            print(f"{M:>4} {t_stock:>24.1f} {t_ours:>9.1f} {t_sp:>11.1f} {t_op:>9.1f} {t_spre:>14.1f} {t_opre:>8.1f}",
                  flush=True)
    print(f"peak memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    print("ALL OK" if ok_all else "SOME CHECKS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
