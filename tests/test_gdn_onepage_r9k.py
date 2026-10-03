"""One state page per request under speculative decoding (kernels/r9k_gdn.hip r9k_gdn_spec_verify /
r9k_gdn_spec_commit, r9700_vllm/models/gdn.py spec_verify / spec_commit) against the slot design it replaces
(r9k_gdn_decode_mtp: one state page per candidate token, the next step starting from the accepted token's page).

Single GPU:  python3 tests/test_gdn_onepage_r9k.py            (BENCH=0 skips the timing table)

A. Multi-step simulation, N sequences, K steps, 1..S candidates per step, a random number accepted (1..T):
   slot design = decode_mtp into the request's slot row (the accepted slot read next step);
   one page = verify, which first replays the previous step's accepted tokens from the record, then runs the
   candidates and writes the new record. Every step's outputs, and the page after each replay against the slot the
   slot design reads: bit-identical. Also the step after a "prefill" (no record), 1-candidate steps, both state
   dtypes, the sigmoid gate, a row map, and several layers.
B. Conv window shift: stock causal_conv1d_update on (window, num_accepted = k) and on (window shifted left by
   k - 1, num_accepted = 1) must give the same conv'd rows and the same new window -- that is what the align-mode
   commit does before vLLM resets num_accepted to 1.
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from r9700_vllm.models import gdn as G  # noqa: E402

dev = torch.device("cuda")
D = 128
EPS = 1e-6
g = torch.Generator(device="cpu").manual_seed(0)
bad = 0


def rnd(*shape, scale=1.0, dtype=torch.bfloat16):
    return (torch.randn(shape, generator=g) * scale).to(dtype).to(dev)


def simulate(name, N, K, HK=4, HV=12, S=8, state_dtype=torch.float32, act="silu", layers=1, rowmap=False,
             fixed_T=None):
    global bad
    QKV = (2 * HK + HV) * D
    A_log = [rnd(HV, scale=0.5, dtype=torch.float32) for _ in range(layers)]
    dt_bias = [rnd(HV, dtype=torch.float32) for _ in range(layers)]
    w = (1 + rnd(D, scale=0.1)).float()
    # slot design: per request S slots (row n of idx); one page: page n+1 of `page`
    idx_slots = torch.arange(1, N * S + 1, dtype=torch.int32).reshape(N, S).to(dev)
    page_idx = torch.arange(1, N + 1, dtype=torch.int32).to(dev)
    init = [rnd(N, HV, D, D, scale=0.3, dtype=state_dtype) for _ in range(layers)]
    slot_state = [rnd(N * S + 1, HV, D, D, scale=0.3, dtype=state_dtype) for _ in range(layers)]
    page_state = [rnd(N + 1, HV, D, D, scale=0.3, dtype=state_dtype) for _ in range(layers)]
    for li in range(layers):
        slot_state[li][idx_slots[:, 0].long()] = init[li]          # "prefill" state: slot 0 / the page
        page_state[li][page_idx.long()] = init[li]
    rec_qkv = [torch.zeros((N + 1, 2, S, QKV), dtype=torch.bfloat16, device=dev) for _ in range(layers)]
    rec_ab = [torch.zeros((N + 1, 2, S, 2 * HV), dtype=torch.bfloat16, device=dev) for _ in range(layers)]
    flags = [torch.zeros((N + 1, 2), dtype=torch.int32, device=dev) for _ in range(layers)]
    acc = torch.ones(N, dtype=torch.int32, device=dev)                  # after the prefill
    gs = act == "sigmoid"
    ok = True
    mism = 0
    prev_ns = None
    for step in range(K + 1):                        # one extra step so the last replay is checked too
        T = [1 if step == K else (int(fixed_T) if fixed_T else int(torch.randint(1, S + 1, (1,), generator=g)))
             for _ in range(N)]
        cu = torch.tensor([0] + torch.tensor(T).cumsum(0).tolist(), dtype=torch.int32, device=dev)
        L = int(cu[-1])
        rows = L + (5 if rowmap else 0)
        mixed = rnd(L, QKV)
        ba = rnd(rows, 2 * HV)
        z = rnd(rows, HV * D)
        perm = torch.randperm(rows, generator=g)[:L].to(dev) if rowmap else None
        b, a = ba.chunk(2, dim=-1)
        out_slot = torch.full((rows, HV * D), 7.0, dtype=torch.bfloat16, device=dev)
        out_page = torch.full((rows, HV * D), 7.0, dtype=torch.bfloat16, device=dev)
        if rowmap:
            ba_c = ba[perm]; z_c = z[perm]
            b_c, a_c = ba_c.chunk(2, dim=-1)
        for li in range(layers):
            # one page first: its replay must reproduce the slot the slot design is about to read (and overwrite)
            if rowmap:
                G.spec_verify(mixed, b, a, A_log[li], dt_bias[li], w, z, out_page, page_state[li], rec_qkv[li],
                              rec_ab[li], flags[li], page_idx, cu, acc, HK, HV, D ** -0.5, EPS, rowmap=perm,
                              gate_sigmoid=gs)
            else:
                G.spec_verify(mixed, b, a, A_log[li], dt_bias[li], w, z, out_page, page_state[li], rec_qkv[li],
                              rec_ab[li], flags[li], page_idx, cu, acc, HK, HV, D ** -0.5, EPS, zero_from=L,
                              gate_sigmoid=gs)
            torch.cuda.synchronize()
            want = slot_state[li][idx_slots[torch.arange(N), (acc - 1).long()].long()]
            got = page_state[li][page_idx.long()]
            if not bool((want == got).all()):
                ok = False
            if rowmap:
                oc = torch.full((L, HV * D), 7.0, dtype=torch.bfloat16, device=dev)
                G.decode_mtp(mixed, b_c, a_c, A_log[li], dt_bias[li], w, z_c, oc, slot_state[li], idx_slots, cu, acc,
                             HK, HV, D ** -0.5, EPS, gate_sigmoid=gs)
                out_slot[perm] = oc
            else:
                G.decode_mtp(mixed, b, a, A_log[li], dt_bias[li], w, z, out_slot, slot_state[li], idx_slots, cu, acc,
                             HK, HV, D ** -0.5, EPS, gate_sigmoid=gs)
            torch.cuda.synchronize()
            sel = perm if rowmap else slice(0, L)
            if not bool((out_slot[sel] == out_page[sel]).all()):
                ok = False; mism += int((out_slot[sel] != out_page[sel]).sum())
            if not rowmap and not bool((out_page[L:] == 0).all()):
                ok = False
            fl = flags[li][page_idx.long()]
            if not bool((fl[:, 0] & 0xff == torch.tensor(T, device=dev, dtype=torch.int32)).all()) \
                    or not bool((fl[:, 1] == 0).all()):
                ok = False
        # the sampler accepts 1..T[n]
        ns = torch.tensor([int(torch.randint(1, T[n] + 1, (1,), generator=g)) for n in range(N)], dtype=torch.int32,
                          device=dev)
        acc = ns
        prev_ns = ns
    bad += not ok
    print(f"  {name:<58} {'ok' if ok else 'FAIL'}" + ("" if ok else f" (output mismatches {mism})"))


def lockstep(name, N, K, HK=16, HV=48, S=8, state_dtype=torch.float32):
    """N identical requests in one batch (same rows, same acceptance every step, as identical prompts at
    temperature 0 give) against the same request alone: one page and the slot design must both be batch-invariant,
    and must agree with each other. This is the serving pattern that showed divergence on 2026-10-03."""
    global bad
    QKV = (2 * HK + HV) * D
    A_log = rnd(HV, scale=0.5, dtype=torch.float32); dt_bias = rnd(HV, dtype=torch.float32)
    w = (1 + rnd(D, scale=0.1)).float()
    init = rnd(1, HV, D, D, scale=0.3, dtype=state_dtype)
    def setup(n):
        page_idx = torch.arange(1, n + 1, dtype=torch.int32).to(dev)
        idx_slots = torch.arange(1, n * S + 1, dtype=torch.int32).reshape(n, S).to(dev)
        ps = torch.zeros((n + 1, HV, D, D), dtype=state_dtype, device=dev); ps[1:] = init
        ss = torch.zeros((n * S + 1, HV, D, D), dtype=state_dtype, device=dev); ss[idx_slots[:, 0].long()] = init
        return dict(page_idx=page_idx, idx_slots=idx_slots, ps=ps, ss=ss,
                    rq=torch.zeros((n + 1, 2, S, QKV), dtype=torch.bfloat16, device=dev),
                    rab=torch.zeros((n + 1, 2, S, 2 * HV), dtype=torch.bfloat16, device=dev),
                    fl=torch.zeros((n + 1, 2), dtype=torch.int32, device=dev),
                    acc=torch.ones(n, dtype=torch.int32, device=dev))
    runs = {1: setup(1), N: setup(N)}
    ok = True; mism = {"page_batch_vs_lone": 0, "slot_batch_vs_lone": 0, "page_vs_slot": 0}
    for step in range(K):
        T = int(torch.randint(1, S + 1, (1,), generator=g))
        row_qkv = rnd(T, QKV); row_ba = rnd(T, 2 * HV); row_z = rnd(T, HV * D)
        ns = int(torch.randint(1, T + 1, (1,), generator=g))
        outs = {}
        for n, R in runs.items():
            cu = torch.arange(0, n * T + 1, T, dtype=torch.int32, device=dev)
            mixed = row_qkv.repeat(n, 1); ba = row_ba.repeat(n, 1); z = row_z.repeat(n, 1)
            b, a = ba.chunk(2, dim=-1)
            op = torch.full((n * T, HV * D), 7.0, dtype=torch.bfloat16, device=dev)
            os_ = torch.full((n * T, HV * D), 7.0, dtype=torch.bfloat16, device=dev)
            G.spec_verify(mixed, b, a, A_log, dt_bias, w, z, op, R["ps"], R["rq"], R["rab"], R["fl"], R["page_idx"],
                          cu, R["acc"], HK, HV, D ** -0.5, EPS, zero_from=n * T)
            G.decode_mtp(mixed, b, a, A_log, dt_bias, w, z, os_, R["ss"], R["idx_slots"], cu, R["acc"], HK, HV,
                         D ** -0.5, EPS)
            torch.cuda.synchronize()
            outs[n] = (op.view(n, T, -1), os_.view(n, T, -1))
            R["acc"] = torch.full((n,), ns, dtype=torch.int32, device=dev)
        lp, ls = outs[1]; bp, bs = outs[N]
        for i in range(N):
            mism["page_batch_vs_lone"] += int((bp[i] != lp[0]).sum())
            mism["slot_batch_vs_lone"] += int((bs[i] != ls[0]).sum())
            mism["page_vs_slot"] += int((bp[i] != bs[i]).sum())
        # the pages of all N copies must hold the same state as the lone page
        if not bool((runs[N]["ps"][1:] == runs[1]["ps"][1]).all()):
            ok = False; mism["page_state_vs_lone"] = mism.get("page_state_vs_lone", 0) + 1
    if any(v for v in mism.values()):
        ok = False
    bad += not ok
    print(f"  {name:<58} {'ok' if ok else 'FAIL'}" + ("" if ok else f" {mism}"))


def check_conv_shift(name, C=2560, S=8, k=5):
    """stock update on (W, acc=k) == stock update on (shift(W, k-1), acc=1): rows and the new window."""
    global bad
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    N, T = 3, S
    Lw = 3 + S
    wt = rnd(C, 4, scale=0.5)
    x = rnd(N * T, C)
    cu = torch.arange(0, N * T + 1, T, dtype=torch.int32, device=dev)
    accs = torch.tensor([k, 1, S], dtype=torch.int32, device=dev)
    W = rnd(N + 1, C, Lw)
    idx = torch.arange(1, N + 1, dtype=torch.int32, device=dev)
    W1 = W.clone()
    y1 = causal_conv1d_update(x.clone(), W1, wt, None, "silu", conv_state_indices=idx, num_accepted_tokens=accs,
                              query_start_loc=cu, max_query_len=T, validate_data=False)
    W2 = W.clone()
    for n in range(N):                                # the commit's shift: left by acc - 1, in place
        sh = int(accs[n]) - 1
        if sh:
            W2[n + 1, :, : Lw - sh] = W[n + 1, :, sh:].clone()
    y2 = causal_conv1d_update(x.clone(), W2, wt, None, "silu", conv_state_indices=idx,
                              num_accepted_tokens=torch.ones_like(accs), query_start_loc=cu, max_query_len=T,
                              validate_data=False)
    torch.cuda.synchronize()
    # the new windows: compare only the columns stock defines (the tail beyond may differ in don't-care columns)
    ok = bool((y1 == y2).all()) and bool((W1[1:] == W2[1:]).all())
    if not ok:
        print("    rows equal:", bool((y1 == y2).all()), " windows equal:", bool((W1[1:] == W2[1:]).all()),
              " differing window cols:", sorted(set((W1[1:] != W2[1:]).nonzero()[:, 2].tolist()))[:12])
    bad += not ok
    print(f"  {name:<58} {'ok' if ok else 'FAIL'}")


def check_commit_shift(name, C=2560, S=8, HK=4, HV=12):
    """the commit kernel's own window shift == the Python shift above (per channel, left by ns - 1)."""
    global bad
    N = 3
    Lw = 3 + S
    conv = rnd(N + 1, C, Lw)
    want = conv.clone()
    ns = torch.tensor([5, 1, S], dtype=torch.int32, device=dev)
    for n in range(N):
        sh = int(ns[n]) - 1
        if sh:
            want[n + 1, :, : Lw - sh] = conv[n + 1, :, sh:].clone()
    QKV = (2 * HK + HV) * D
    assert QKV == C
    st = rnd(N + 1, HV, D, D, scale=0.3, dtype=torch.float32)
    rq = torch.zeros((N + 1, S, QKV), dtype=torch.bfloat16, device=dev)
    rab = torch.zeros((N + 1, S, 2 * HV), dtype=torch.bfloat16, device=dev)
    page_idx = torch.arange(1, N + 1, dtype=torch.int32, device=dev)
    G.spec_commit([(st, rq, rab, conv, rnd(HV, dtype=torch.float32), rnd(HV, dtype=torch.float32))], page_idx, ns,
                  HK, HV, conv_shift=True)
    torch.cuda.synchronize()
    ok = bool((conv[:, :, : Lw - 1] == want[:, :, : Lw - 1]).all())     # the last column is a don't-care after a shift
    for n in range(N):
        sh = int(ns[n]) - 1
        ok &= bool((conv[n + 1, :, : Lw - sh] == want[n + 1, :, : Lw - sh]).all())
    bad += not ok
    print(f"  {name:<58} {'ok' if ok else 'FAIL'}")


def bench_verify(N, HK=8, HV=24, S=8):
    """one layer's verify with a live record (replay of S tokens + S candidates) vs the slot kernel (S candidates)."""
    QKV = (2 * HK + HV) * D
    L = N * S
    cu = torch.arange(0, L + 1, S, dtype=torch.int32, device=dev)
    mixed, ba, z = rnd(L, QKV), rnd(L, 2 * HV), rnd(L, HV * D)
    b, a = ba.chunk(2, dim=-1)
    A, dtb, w = rnd(HV, scale=0.5, dtype=torch.float32), rnd(HV, dtype=torch.float32), (1 + rnd(D, scale=0.1)).float()
    st = rnd(N * S + 1, HV, D, D, scale=0.3)
    idx = torch.arange(1, N * S + 1, dtype=torch.int32).reshape(N, S).to(dev)
    pidx = idx[:, 0].contiguous()
    rq = rnd(N * S + 1, 2, S, QKV); rab = rnd(N * S + 1, 2, S, 2 * HV)
    fl = torch.zeros((N * S + 1, 2), dtype=torch.int32, device=dev)
    acc = torch.full((N,), S, dtype=torch.int32, device=dev)
    out = torch.empty((L, HV * D), dtype=torch.bfloat16, device=dev)
    def run_page(live):
        fl[:, 0] = S if live else 0
        G.spec_verify(mixed, b, a, A, dtb, w, z, out, st, rq, rab, fl, pidx, cu, acc, HK, HV, D ** -0.5, EPS)
    def run_slot():
        G.decode_mtp(mixed, b, a, A, dtb, w, z, out, st, idx, cu, acc, HK, HV, D ** -0.5, EPS)
    for name, f in (("slot kernel (decode_mtp)", run_slot), ("one page, no record", lambda: run_page(False)),
                    ("one page, replay 8 then 8", lambda: run_page(True))):
        for _ in range(3): f()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(20): f()
        torch.cuda.synchronize()
        print(f"  verify N {N:<3} Hv {HV} S {S}: {name:<28} {(time.perf_counter() - t) / 20 * 1e6:7.1f} us")


def bench(N, HK=8, HV=24, S=8, layers=5):
    QKV = (2 * HK + HV) * D
    st = [rnd(N + 1, HV, D, D, scale=0.3) for _ in range(layers)]
    rq = [rnd(N + 1, S, QKV) for _ in range(layers)]
    rab = [rnd(N + 1, S, 2 * HV) for _ in range(layers)]
    conv = [rnd(N + 1, QKV, 3 + S) for _ in range(layers)]
    A = [rnd(HV, scale=0.5, dtype=torch.float32) for _ in range(layers)]
    dtb = [rnd(HV, dtype=torch.float32) for _ in range(layers)]
    idx = torch.arange(1, N + 1, dtype=torch.int32, device=dev)
    ns = torch.full((N,), S, dtype=torch.int32, device=dev)
    lay = [(st[i], rq[i], rab[i], conv[i], A[i], dtb[i]) for i in range(layers)]
    for shift in (False, True):
        for _ in range(3):
            G.spec_commit(lay, idx, ns, HK, HV, conv_shift=shift)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(20):
            G.spec_commit(lay, idx, ns, HK, HV, conv_shift=shift)
        torch.cuda.synchronize()
        print(f"  commit N {N:<3} {layers} layers, {S} tokens each, bf16 state, shift={int(shift)}: "
              f"{(time.perf_counter() - t) / 20 * 1e6:7.1f} us per launch")


def main():
    if not G.onepage_available():
        print("libr9k.so has no r9k_gdn_spec_verify: rebuild kernels/")
        sys.exit(1)
    print("A. verify + commit vs the slot design")
    simulate("1 seq, 6 steps", 1, 6)
    simulate("4 seqs, 8 steps", 4, 8)
    simulate("16 seqs, 5 steps, 27B-TP2 heads (8/24)", 16, 5, HK=8, HV=24)
    simulate("4 seqs, 1-token steps", 4, 4, fixed_T=1)
    simulate("4 seqs, always 8 tokens", 4, 4, fixed_T=8)
    simulate("bf16 state, 4 seqs, 6 steps", 4, 6, state_dtype=torch.bfloat16)
    simulate("sigmoid gate, 4 seqs, 4 steps", 4, 4, act="sigmoid")
    simulate("3 layers, 4 seqs, 5 steps", 4, 5, layers=3)
    simulate("row map, 4 seqs, 5 steps", 4, 5, rowmap=True)
    lockstep("lockstep: 4 identical seqs vs lone, 27B-TP1 heads (16/48)", 4, 12)
    lockstep("lockstep: 8 identical seqs vs lone, 27B-TP1 heads", 8, 12)
    lockstep("lockstep: 8 identical seqs vs lone, 27B-TP2 heads (8/24)", 8, 12, HK=8, HV=24)
    lockstep("lockstep: 4 identical seqs, bf16 state", 4, 8, state_dtype=torch.bfloat16)
    print("B. the conv window shift")
    check_conv_shift("stock update: (W, acc=k) == (shift(W, k-1), acc=1)")
    check_commit_shift("commit kernel shift == the Python shift")
    print(f"correctness: {'PASS' if bad == 0 else 'FAIL'}")
    if os.environ.get("BENCH", "1") != "0":
        for N in (1, 8, 16):
            bench_verify(N)
        bench(16)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
