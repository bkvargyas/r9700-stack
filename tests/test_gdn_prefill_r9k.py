"""Our short-prefill GDN core (kernels/r9k_gdn.hip r9k_gdn_seq, r9700_vllm/models/gdn.py seq_core / _prefill): the
delta rule token by token + the gated RMS norm in one launch, in place of stock's chunked composition
(fused_post_conv_prep + chunk_gated_delta_rule + state gather/scatter + layer_norm_fwd).

Single GPU:  python3 tests/test_gdn_prefill_r9k.py            (BENCH=0 skips the timing table)

The chunked form is the same mathematics in a different order of operations (per-64-token WY solves on bf16 q/k), so
the two do not agree bit for bit; the check is against the definition instead:
  A. prefill sequences vs an fp64 token-by-token reference: output equal except where fp32 summation order flips a
     bf16 rounding (<= 3e-3 of the elements, each small), final state within 1e-4 relative (fp32 state) or by the
     flip rule (bf16 state). Stock's own distance to the same reference is printed next to ours, and ours must not
     be further from the reference than stock is. Sequences without an initial state start from zero whatever the
     slot holds; a sequence whose slot is 0 comes out zero and stores nothing; rows past the batch are zeroed.
  B. rowmap: b/a/z/out addressed through a row permutation give bit-identical results to the compact run and leave
     every other row of out untouched.
  C. spec-decode sequences (acc given) are bit-identical to r9k_gdn_decode_mtp, output and state.
  D. the conv in front of it (r9k_gdn_conv) vs stock's causal_conv1d_fn: output and conv state bit-identical, for
     either state layout, with and without initial state, sequences shorter than the state, a state wider than
     the conv's history (speculative decoding keeps its window in the same tensor), and through a rowmap.
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule  # noqa: E402
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import fused_post_conv_prep  # noqa: E402
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd  # noqa: E402

from r9700_vllm.models import gdn as G  # noqa: E402

dev = torch.device("cuda")
D = 128
EPS = 1e-6
g = torch.Generator(device="cpu").manual_seed(0)
bad = 0


def rnd(*shape, scale=1.0, dtype=torch.bfloat16):
    return (torch.randn(shape, generator=g) * scale).to(dtype).to(dev)


def make_case(lens, HK=8, HV=24, hasinit=None, state_dtype=torch.float32, tail=0, skip=(), nblocks=None):
    N = len(lens)
    nblocks = nblocks or max(32, N + 8)
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)
    L = int(cu[-1])
    rows = L + tail
    idx = (torch.randperm(nblocks - 1, generator=g)[:N] + 1).to(torch.int32)
    for n in skip:
        idx[n] = 0
    hi = torch.tensor(hasinit if hasinit is not None else [True] * N, dtype=torch.bool)
    QKV = (2 * HK + HV) * D
    return dict(N=N, HK=HK, HV=HV, QKV=QKV, L=L, rows=rows, cu=cu.to(dev), idx=idx.to(dev), hasinit=hi.to(dev),
                mixed=rnd(L, QKV), z=rnd(rows, HV * D), ba=rnd(rows, 2 * HV),
                A_log=rnd(HV, scale=0.5, dtype=torch.float32), dt_bias=rnd(HV, dtype=torch.float32),
                w=(1 + rnd(D, scale=0.1)).float(), state=rnd(nblocks, HV, D, D, scale=0.3, dtype=state_dtype))


def gate(zf, act):
    return torch.sigmoid(zf) if act == "sigmoid" else zf * torch.sigmoid(zf)


def reference(c, state, act):
    """fp64, token by token; state updated in place (rounded to its dtype once, at the end of each sequence)."""
    HK, HV, L = c["HK"], c["HV"], c["L"]
    m = c["mixed"].double()
    q = m[:, : HK * D].view(L, HK, D)
    k = m[:, HK * D: 2 * HK * D].view(L, HK, D)
    v = m[:, 2 * HK * D:].view(L, HV, D)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * D ** -0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q, k = q.repeat_interleave(HV // HK, dim=1), k.repeat_interleave(HV // HK, dim=1)       # [L, HV, D]
    b, a = c["ba"].double().chunk(2, dim=-1)
    gdec = torch.exp(-torch.exp(c["A_log"].double()) * torch.nn.functional.softplus(a + c["dt_bias"].double()))
    beta = torch.sigmoid(b)
    z = c["z"].double().view(-1, HV, D)
    out = torch.zeros((c["rows"], HV, D), dtype=torch.bfloat16, device=dev)
    for n in range(c["N"]):
        lo, hi, i0 = int(c["cu"][n]), int(c["cu"][n + 1]), int(c["idx"][n])
        if i0 <= 0 or hi <= lo:
            continue
        h = state[i0].double() if bool(c["hasinit"][n]) else torch.zeros((HV, D, D), dtype=torch.float64, device=dev)
        for t in range(lo, hi):
            h = h * gdec[t][:, None, None]
            d = (h * k[t][:, None, :]).sum(-1)
            vv = (v[t] - d) * beta[t][:, None]
            h = h + vv[:, :, None] * k[t][:, None, :]
            o = (h * q[t][:, None, :]).sum(-1).to(torch.bfloat16).double()
            y = o * torch.rsqrt((o * o).mean(-1, keepdim=True) + EPS) * c["w"].double() * gate(z[t], act)
            out[t] = y.to(torch.bfloat16)
        state[i0] = h.to(state.dtype)
    return out.view(c["rows"], HV * D)


def stock(c, state, act):
    """vLLM's prefill composition on the same buffers."""
    HK, HV, L, rows = c["HK"], c["HV"], c["L"], c["rows"]
    b, a = c["ba"][:L].chunk(2, dim=-1)
    q, k, v, gg, beta = fused_post_conv_prep(conv_output=c["mixed"], a=a.contiguous(), b=b.contiguous(),
                                             A_log=c["A_log"], dt_bias=c["dt_bias"], num_k_heads=HK, head_k_dim=D,
                                             head_v_dim=D, apply_l2norm=True, output_g_exp=False)
    ii = c["idx"].long()
    init = state[ii]
    init[~c["hasinit"], ...] = 0
    o, final = chunk_gated_delta_rule(q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0), g=gg.unsqueeze(0),
                                      beta=beta.unsqueeze(0), initial_state=init, output_final_state=True,
                                      cu_seqlens=c["cu"], use_qk_l2norm_in_kernel=False)
    state[ii] = final.to(state.dtype)
    core = torch.zeros((rows, HV, D), dtype=torch.bfloat16, device=dev)
    core[:L] = o.squeeze(0)
    out, _, _ = layer_norm_fwd(core.view(rows * HV, D), c["w"].to(torch.bfloat16), None, EPS,
                               z=c["z"].reshape(rows * HV, D), group_size=D, norm_before_gate=True, is_rms_norm=True,
                               activation=act)
    return out.view(rows, HV * D)


def ours(c, state, act, out=None, rowmap=None, ba=None, z=None):
    out = torch.full((c["rows"], c["HV"] * D), 7.0, dtype=torch.bfloat16, device=dev) if out is None else out
    b, a = (c["ba"] if ba is None else ba).chunk(2, dim=-1)
    G.seq_core(c["mixed"], b, a, c["A_log"], c["dt_bias"], c["w"], c["z"] if z is None else z, out, state, c["idx"],
               c["cu"], c["HK"], c["HV"], D ** -0.5, EPS, hasinit=c["hasinit"], rowmap=rowmap, zero_from=c["L"],
               gate_sigmoid=act == "sigmoid")
    return out


def flips(o, s):
    diff = (o != s)
    nf = diff.sum().item()
    tol = torch.maximum(s.float().abs() * 2 ** -6, torch.full_like(s.float(), 1e-2 * s.float().abs().mean().item()))
    small = ((o.float() - s.float()).abs() <= tol) | ~diff
    return bool(torch.isfinite(o.float()).all()) and nf <= max(1, int(o.numel() * 3e-3)) and bool(small.all()), nf


def rel(x, y):
    return ((x.double() - y.double()).norm() / y.double().norm().clamp_min(1e-30)).item()


def check(name, lens, act="silu", with_stock=True, **kw):
    global bad
    skip = kw.get("skip", ())
    c = make_case(lens, **kw)
    r_state, o_state, s_state = c["state"].clone(), c["state"].clone(), c["state"].clone()
    with torch.no_grad():
        r = reference(c, r_state, act)
        o = ours(c, o_state, act)
        s = stock(c, s_state, act) if with_stock and not skip and min(lens) > 0 else None
    torch.cuda.synchronize()
    ok_o, nf = flips(o, r)
    if c["state"].dtype == torch.float32:
        e = ((o_state - r_state).abs() / r_state.abs().clamp_min(1e-3)).max().item()
        ok_s, msg_s = e <= 1e-4, f"state rel {e:.1e}"
    else:
        ok_s, n2 = flips(o_state, r_state)
        msg_s = f"state flips {n2}"
    ok = ok_o and ok_s
    L = c["L"]
    for n in skip:                                   # slot 0: zero rows, nothing stored (state[0] untouched)
        lo, hi = int(c["cu"][n]), int(c["cu"][n + 1])
        ok &= not o[lo:hi].any().item()
    ok &= bool((o_state[0] == c["state"][0]).all())
    if c["rows"] > L:
        ok &= not o[L:].any().item()
    untouched = [i for i in range(c["state"].shape[0]) if i not in set(c["idx"].tolist())]
    ok &= bool((o_state[untouched] == c["state"][untouched]).all())
    msg = f"out flips {nf}/{o.numel()}  {msg_s}  |ours-ref| {rel(o[:L], r[:L]):.1e}"
    if s is not None:
        es, eo = rel(s[:L], r[:L]), rel(o[:L], r[:L])
        ss = rel(s_state[c["idx"].long()], r_state[c["idx"].long()])
        ok &= eo <= es + 1e-6
        msg += f"  |stock-ref| {es:.1e} (state {ss:.1e})"
    bad += not ok
    print(f"  {name:<46} {act:<7} {msg}  {'ok' if ok else 'FAIL'}")


def check_rowmap(name, lens, spec=False, **kw):
    """b/a/z/out through a row permutation == the compact run, bit for bit; other rows of out untouched."""
    global bad
    if spec:
        c, extra = make_spec(lens, kw.pop("accs"))
    else:
        c, extra = make_case(lens, **kw), {}
    L, rows = c["L"], c["L"] + 9
    perm = torch.randperm(rows, generator=g)[:L].to(dev)                  # original row of each compact row
    ba_o = rnd(rows, 2 * c["HV"]); z_o = rnd(rows, c["HV"] * D)
    ba_o[perm] = c["ba"][:L]; z_o[perm] = c["z"][:L]
    st1, st2 = c["state"].clone(), c["state"].clone()
    out_o = torch.full((rows, c["HV"] * D), 7.0, dtype=torch.bfloat16, device=dev)
    b, a = c["ba"].chunk(2, dim=-1); bo, ao = ba_o.chunk(2, dim=-1)
    out_c = torch.full((c["rows"], c["HV"] * D), 7.0, dtype=torch.bfloat16, device=dev)
    args = (c["A_log"], c["dt_bias"], c["w"])
    tail = (c["idx"], c["cu"], c["HK"], c["HV"], D ** -0.5, EPS)
    k1 = dict(acc=extra["acc"]) if spec else dict(hasinit=c["hasinit"])
    G.seq_core(c["mixed"], b, a, *args, c["z"], out_c, st1, *tail, **k1)
    G.seq_core(c["mixed"], bo, ao, *args, z_o, out_o, st2, *tail, rowmap=perm, **k1)
    torch.cuda.synchronize()
    mask = torch.ones(rows, dtype=torch.bool, device=dev); mask[perm] = False
    ok = bool((out_o[perm] == out_c[:L]).all()) and bool((st1 == st2).all()) and bool((out_o[mask] == 7.0).all())
    bad += not ok
    print(f"  {name:<46} rowmap == compact, {int(mask.sum())} other rows untouched  {'ok' if ok else 'FAIL'}")


def make_spec(lens, accs, HK=4, HV=12, S=8, state_dtype=torch.float32):
    N = len(lens)
    c = make_case(lens, HK=HK, HV=HV, state_dtype=state_dtype, nblocks=max(64, N * S + 8))
    idx = torch.randperm(c["state"].shape[0] - 1, generator=g)[: N * S].reshape(N, S).to(torch.int32) + 1
    c["idx"] = idx.to(dev)
    return c, dict(acc=torch.tensor(accs, dtype=torch.int32, device=dev))


def check_spec(name, lens, accs, state_dtype=torch.float32, act="silu"):
    global bad
    c, e = make_spec(lens, accs, state_dtype=state_dtype)
    st1, st2 = c["state"].clone(), c["state"].clone()
    o1 = torch.empty((c["rows"], c["HV"] * D), dtype=torch.bfloat16, device=dev)
    o2 = torch.empty_like(o1)
    b, a = c["ba"].chunk(2, dim=-1)
    gs = act == "sigmoid"
    G.decode_mtp(c["mixed"], b, a, c["A_log"], c["dt_bias"], c["w"], c["z"], o1, st1, c["idx"], c["cu"], e["acc"],
                 c["HK"], c["HV"], D ** -0.5, EPS, gate_sigmoid=gs)
    G.seq_core(c["mixed"], b, a, c["A_log"], c["dt_bias"], c["w"], c["z"], o2, st2, c["idx"], c["cu"], c["HK"],
               c["HV"], D ** -0.5, EPS, acc=e["acc"], gate_sigmoid=gs)
    torch.cuda.synchronize()
    ok = bool((o1 == o2).all()) and bool((st1 == st2).all()) and bool((st1 != c["state"]).any())
    bad += not ok
    print(f"  {name:<46} {act:<7} == r9k_gdn_decode_mtp (out and state)  {'ok' if ok else 'FAIL'}")


def check_conv(name, lens, hasinit=None, C=5120, layout="DS", rowmap=False, skip=(), width=3):
    global bad
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    N = len(lens)
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device=dev)
    L = int(cu[-1])
    lines = N + 5
    table = (torch.randperm(lines - 1, generator=g)[:N] + 1).to(torch.int32)
    for n in skip:
        table[n] = 0
    table = torch.stack([table, torch.zeros_like(table)], 1).to(dev)      # idx is a strided column, as in serving
    idx = table[:, 0]
    hi = torch.tensor(hasinit if hasinit is not None else [True] * N, dtype=torch.bool, device=dev)
    wt = rnd(C, 4, scale=0.5)
    xc = rnd(L, C + 64)                                                    # compact rows; C is a prefix of the row
    st = rnd(lines, C, width) if layout == "DS" else rnd(lines, width, C)   # width > 3: the spec-decode window
    s1, s2 = st.clone(), st.clone()
    v1 = s1 if layout == "DS" else s1.transpose(-1, -2)
    v2 = s2 if layout == "DS" else s2.transpose(-1, -2)
    ref = causal_conv1d_fn(xc[:, :C].transpose(0, 1), wt, None, activation="silu", conv_states=v1,
                           has_initial_state=hi, cache_indices=idx, query_start_loc=cu).transpose(0, 1)
    if rowmap:
        rows = L + 7
        perm = torch.randperm(rows, generator=g)[:L].to(dev)
        xo = rnd(rows, C + 64)
        xo[perm] = xc
        o = G.conv_prefill(xo, C, L, wt, v2, idx, cu, hasinit=hi, rowmap=perm)
    else:
        o = G.conv_prefill(xc, C, L, wt, v2, idx, cu, hasinit=hi)
    torch.cuda.synchronize()
    keep = torch.ones(L, dtype=torch.bool, device=dev)
    for n in skip:                                       # stock leaves these rows uninitialised; ours are zero
        keep[int(cu[n]): int(cu[n + 1])] = False
    nd = int((o[keep] != ref[keep]).sum())
    ns = int((s1 != s2).sum())
    ok = nd == 0 and ns == 0 and bool((s2 != st).any()) and not o[~keep].any().item() and ref.is_contiguous()
    bad += not ok
    print(f"  {name:<46} out diffs {nd}/{o.numel()}  state diffs {ns}  {'ok' if ok else 'FAIL'}")


def wall_ms(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


def bench(lens, HK=8, HV=24):
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    c = make_case(lens, HK=HK, HV=HV)
    st = c["state"].clone()
    out = torch.empty((c["rows"], HV * D), dtype=torch.bfloat16, device=dev)
    C = c["QKV"]
    wt, cs = rnd(C, 4, scale=0.5), rnd(c["state"].shape[0], C, 3)
    with torch.no_grad():
        s = wall_ms(lambda: stock(c, st, "silu"))
        o = wall_ms(lambda: ours(c, st, "silu", out=out))
        sc = wall_ms(lambda: causal_conv1d_fn(c["mixed"].transpose(0, 1), wt, None, activation="silu", conv_states=cs,
                                              has_initial_state=c["hasinit"], cache_indices=c["idx"],
                                              query_start_loc=c["cu"]))
        oc = wall_ms(lambda: G.conv_prefill(c["mixed"], C, c["L"], wt, cs, c["idx"], c["cu"], hasinit=c["hasinit"]))
    print(f"  {len(lens)} x {lens[0]:<5} tokens  Hv {HV}:  core stock {s:6.3f} ms  ours {o:6.3f} ms   "
          f"conv stock {sc:6.3f} ms  ours {oc:6.3f} ms   (eager call, wall)")


def main():
    if not G.seq_available():
        print("libr9k.so has no r9k_gdn_seq: rebuild kernels/")
        sys.exit(1)
    print("A. prefill sequences vs the fp64 token-by-token reference")
    check("1 seq, 45 tokens, no initial state", [45], hasinit=[False])
    check("1 seq, 45 tokens, initial state", [45])
    check("1 seq, 1 token", [1])
    check("1 seq, 200 tokens (4 chunks in stock)", [200], hasinit=[False])
    check("3 seqs [70, 9, 130], init [0, 1, 0]", [70, 9, 130], hasinit=[False, True, False])
    check("8 seqs x 33", [33] * 8, hasinit=[False, True] * 4)
    check("Hk 4 Hv 12 (TP4), 2 seqs [64, 65]", [64, 65], HK=4, HV=12, hasinit=[False, False])
    check("bf16 state, 2 seqs [50, 3]", [50, 3], state_dtype=torch.bfloat16, hasinit=[False, True])
    check("sigmoid gate (Flash-Next), [45, 20]", [45, 20], act="sigmoid", hasinit=[False, True])
    check("tail padding rows 5", [45, 7], tail=5)
    check("seq 1 has slot 0", [20, 12, 9], skip=(1,))
    check("an empty sequence in the middle", [20, 0, 9], with_stock=False)
    check("512 tokens", [512], hasinit=[False])
    print("B. rowmap")
    check_rowmap("prefill, 3 seqs [40, 5, 17]", [40, 5, 17], hasinit=[False, True, False])
    check_rowmap("spec, 4 seqs x 8", [8] * 4, spec=True, accs=[1, 3, 8, 5])
    print("C. spec-decode sequences")
    check_spec("3 seqs [8, 2, 1], accepted [3, 2, 1]", [8, 2, 1], [3, 2, 1])
    check_spec("16 seqs x 8, mixed accepted", [8] * 16, [1, 2, 3, 4, 5, 6, 7, 8] * 2)
    check_spec("bf16 state, 4 seqs", [4, 4, 4, 4], [4, 3, 2, 1], state_dtype=torch.bfloat16)
    check_spec("sigmoid gate, 4 seqs", [4, 4, 4, 4], [4, 3, 2, 1], act="sigmoid")
    print("D. the conv")
    check_conv("1 seq, 45 tokens, no initial state", [45], hasinit=[False])
    check_conv("1 seq, 45 tokens, initial state", [45])
    check_conv("lengths [1, 2, 3, 4], init [1, 0, 1, 0]", [1, 2, 3, 4], hasinit=[True, False, True, False])
    check_conv("lengths [2, 1], initial state", [2, 1])
    check_conv("3 seqs [70, 9, 130], other state layout", [70, 9, 130], hasinit=[False, True, False], layout="SD")
    check_conv("rowmap, 3 seqs [40, 5, 17]", [40, 5, 17], hasinit=[False, True, False], rowmap=True)
    check_conv("seq 1 has slot 0", [20, 12, 9], skip=(1,))
    check_conv("C 2560 (TP4), 512 tokens", [512], C=2560, hasinit=[False])
    check_conv("state 10 wide (7 draft tokens), [45, 2, 9]", [45, 2, 9], hasinit=[False, True, True], width=10)
    check_conv("state 6 wide, other layout, rowmap", [33, 1], hasinit=[True, True], width=6, layout="SD", rowmap=True)
    print(f"correctness: {'PASS' if bad == 0 else 'FAIL'}")
    if os.environ.get("BENCH", "1") != "0":
        for T in (16, 45, 128, 256, 512, 1024, 2048):
            bench([T])
        bench([128] * 4)
        bench([512] * 4)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
