"""Our fused GDN MTP decode core (kernels/r9k_gdn.hip r9k_gdn_decode_mtp, r9700_vllm/models/gdn.py) vs the stock
composition it replaces: fused_sigmoid_gating_delta_rule_update (Triton) on q/k/v made contiguous, output copied into
a zeroed core buffer, then layer_norm_fwd (RMS, norm before a silu gate).

Single GPU:  python3 tests/test_gdn_decode_r9k.py            (BENCH=0 skips the timing table)

Flash-Next at TP=4: Hk=4, Hv=12, D=128, S=4 spec slots per sequence; silu and sigmoid output gates (Flash-Next
uses sigmoid). Inputs are views of the merged in_proj row
(mixed_qkv = row[:2560], z = row[2560:]) and of ba (b = ba[:, :Hv], a = ba[:, Hv:]) as in serving. The output must
match stock except where our fp32 summation order flips a bf16 rounding (counted, <= 3e-3 of the elements, each
within 2 bf16 ulps or 1% of the tensor's mean magnitude -- the norm re-rounds an already flipped o); the fp32 state
within 1e-4 relative, the bf16 state by the same flip rule. Sequences whose initial state index is 0 (padding) must
come out zero (stock leaves their rows uninitialised, so they are excluded from the comparison), as must rows past
the last sequence. The timing table is only meaningful on an idle GPU.
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd  # noqa: E402

from r9700_vllm.models import gdn as G  # noqa: E402

dev = torch.device("cuda")
HK, HV, D, S = 4, 12, 128, 4
QKV = (2 * HK + HV) * D
ROW = QKV + HV * D                  # merged in_proj row: [q k v | z]
EPS = 1e-6
g = torch.Generator(device="cpu").manual_seed(0)
bad = 0


def rnd(*shape, scale=1.0, dtype=torch.bfloat16):
    return (torch.randn(shape, generator=g) * scale).to(dtype).to(dev)


def make_case(lens, accs, nblocks=None, state_dtype=torch.float32, tail=0, skip=()):
    """lens[n] tokens per sequence (<= S), accs[n] accepted tokens (1..S); skip: sequences whose initial slot is 0."""
    N = len(lens)
    nblocks = nblocks or max(64, N * S + 8)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
    rows = int(cu[-1]) + tail
    idx = torch.randperm(nblocks - 1, generator=g)[: N * S].reshape(N, S).to(torch.int32) + 1
    for n in skip:
        idx[n, accs[n] - 1] = 0
    acc = torch.tensor(accs, dtype=torch.int32)
    qkvz = rnd(rows, ROW)
    ba = rnd(rows, 2 * HV)
    A_log = rnd(HV, scale=0.5, dtype=torch.float32)
    dt_bias = rnd(HV, dtype=torch.float32)
    w = (1 + rnd(D, scale=0.1)).float()             # the norm weight is a bf16 parameter; both sides see its value
    state = rnd(nblocks, HV, D, D, scale=0.3, dtype=state_dtype)
    return dict(N=N, rows=rows, n_act=int(cu[-1]), cu=cu.to(dev), idx=idx.to(dev), acc=acc.to(dev), qkvz=qkvz,
                ba=ba, A_log=A_log, dt_bias=dt_bias, w=w, state=state)


ACT = "silu"


def stock(c, state):
    """the stock spec-decode composition on views of the same buffers (state updated in place)."""
    rows, N = c["rows"], c["N"]
    mixed = c["qkvz"][:, :QKV]
    z = c["qkvz"][:, QKV:]
    b, a = c["ba"].chunk(2, dim=-1)
    q, k, v = torch.split(mixed, [HK * D, HK * D, HV * D], dim=-1)
    fused = torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)], dim=0)      # rearrange_mixed_qkv
    q = fused[: rows * HK * D].view(1, rows, HK, D)
    k = fused[rows * HK * D: 2 * rows * HK * D].view(1, rows, HK, D)
    v = fused[2 * rows * HK * D:].view(1, rows, HV, D)
    o, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=c["A_log"], a=a.contiguous(), b=b.contiguous(), dt_bias=c["dt_bias"], q=q, k=k, v=v,
        initial_state=state, inplace_final_state=True, cu_seqlens=c["cu"], ssm_state_indices=c["idx"],
        num_accepted_tokens=c["acc"], use_qk_l2norm_in_kernel=True)
    core = torch.zeros((rows, HV, D), dtype=torch.bfloat16, device=dev)
    n_act = c["n_act"]                        # a host int: stock reads num_actual_tokens from the metadata
    core[:n_act] = o.squeeze(0)[:n_act]
    out, _, _ = layer_norm_fwd(core.view(rows * HV, D), c["w"].to(torch.bfloat16), None, EPS,
                               z=z.reshape(rows * HV, D), group_size=D, norm_before_gate=True, is_rms_norm=True,
                               activation=ACT)
    return out.view(rows, HV * D)


def ours(c, state):
    out = torch.empty((c["rows"], HV * D), dtype=torch.bfloat16, device=dev)
    b, a = c["ba"].chunk(2, dim=-1)
    G.decode_mtp(c["qkvz"][:, :QKV], b, a, c["A_log"], c["dt_bias"], c["w"], c["qkvz"][:, QKV:], out, state, c["idx"],
                 c["cu"], c["acc"], HK, HV, D ** -0.5, EPS, gate_sigmoid=ACT == "sigmoid")
    return out


def flips_ok(o, s, name):
    diff = (o != s)
    nf = diff.sum().item()
    tol = torch.maximum(s.float().abs() * 2 ** -6, torch.full_like(s.float(), 1e-2 * s.float().abs().mean().item()))
    small = ((o.float() - s.float()).abs() <= tol) | ~diff
    ok = bool(torch.isfinite(o).all()) and nf <= max(1, int(o.numel() * 3e-3)) and bool(small.all())
    return ok, f"{name} flips {nf}/{o.numel()}" + ("" if small.all() else " (large!)")


def check(name, lens, accs, state_dtype=torch.float32, tail=0, skip=(), act="silu"):
    global bad, ACT
    ACT = act
    c = make_case(lens, accs, state_dtype=state_dtype, tail=tail, skip=skip)
    s_state, o_state = c["state"].clone(), c["state"].clone()
    with torch.no_grad():
        s = stock(c, s_state)
        o = ours(c, o_state)
    torch.cuda.synchronize()
    keep = torch.ones(c["rows"], dtype=torch.bool, device=dev)
    for n in skip:                                  # stock never writes these rows (Triton returns early)
        keep[int(c["cu"][n].item()): int(c["cu"][n + 1].item())] = False
    ok_o, msg_o = flips_ok(o[keep], s[keep], "out")
    if state_dtype == torch.float32:
        rel = ((o_state - s_state).abs() / s_state.abs().clamp_min(1e-3)).max().item()
        ok_s, msg_s = rel <= 1e-4, f"state rel {rel:.1e}"
    else:
        ok_s, msg_s = flips_ok(o_state, s_state, "state")
    touched = (o_state != c["state"]).any().item()
    zero_ok = True
    for n in skip:
        lo, hi = int(c["cu"][n].item()), int(c["cu"][n + 1].item())
        zero_ok &= not o[lo:hi].any().item()
    if tail:
        zero_ok &= not o[c["n_act"]:].any().item()
    ok = ok_o and ok_s and touched and zero_ok
    bad += not ok
    print(f"  {name:<44} {act:<7} {msg_o:<22} {msg_s:<18} {'ok' if ok else 'FAIL'}"
          + ("" if zero_ok else " (skipped/tail rows not zero)"))


def graph_us(fn, reps=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, stream=st):
            for _ in range(reps):
                fn()
    torch.cuda.synchronize()
    gr.replay()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(5):
        gr.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / 5 / reps * 1e6


def bench(N, state_dtype):
    c = make_case([S] * N, [S] * N, state_dtype=state_dtype)
    st = c["state"].clone()
    with torch.no_grad():
        for name, f in (("stock (glue + Triton + norm)", lambda: stock(c, st)),
                        ("ours (one launch)", lambda: ours(c, st))):
            print(f"  N {N:<3} state {str(state_dtype)[6:]:<8} {name:<30} {graph_us(f):7.1f} us (graph replay)")


def main():
    if not G.available():
        print("libr9k.so has no r9k_gdn_decode_mtp: rebuild kernels/")
        sys.exit(1)
    check("1 seq, 4 tokens, accepted 4", [4], [4])
    check("1 seq, 4 tokens, accepted 1", [4], [1])
    check("3 seqs [4, 2, 1], accepted [3, 2, 1]", [4, 2, 1], [3, 2, 1])
    check("16 seqs x 4, mixed accepted", [4] * 16, [1, 2, 3, 4] * 4)
    check("16 seqs, tail padding rows 3", [4] * 16, [4] * 16, tail=3)
    check("8 seqs, seq 2 and 5 padded (slot 0)", [4] * 8, [2] * 8, skip=(2, 5))
    check("bf16 state, 4 seqs", [4, 4, 4, 4], [4, 3, 2, 1], state_dtype=torch.bfloat16)
    check("bf16 state, 1 seq 1 token", [1], [1], state_dtype=torch.bfloat16)
    check("64 seqs x 4", [4] * 64, [4] * 64)
    check("sigmoid gate (Flash-Next), 4 seqs", [4, 4, 4, 4], [4, 3, 2, 1], act="sigmoid")
    check("sigmoid gate, 16 seqs, bf16 state", [4] * 16, [1, 2, 3, 4] * 4, state_dtype=torch.bfloat16, act="sigmoid")
    check("sigmoid gate, 1 seq 1 token", [1], [1], act="sigmoid")
    print(f"correctness: {'PASS' if bad == 0 else 'FAIL'}")
    if os.environ.get("BENCH", "1") != "0":
        for N in (1, 4, 16):
            bench(N, torch.float32)
        bench(16, torch.bfloat16)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
