"""DeepSeek-V4.1 sparse-MLA decode on RDNA4: our split-K launcher (attn/dsv4_decode.py) vs stock's un-tuned-architecture
fallback (`_sparse_attn_decode_ragged_kernel`) + the caller's standalone inverse RoPE (`rocm_inverse_rope_rows_`), and vs a
torch fp32 reference, on synthetic fp8_ds_mla caches with ragged SWA / top-k segments built the way
`rocm_sparse_attn_decode` receives them (q [T, 8, 512] bf16, SWA cache [blocks, 32, 584] uint8, compressed cache
[blocks, 64|128, 584], attn_sink [8]); the bf16 single-segment launcher vs stock's bf16 split-K. Then graph-captured
timing (median of replays) of both fp8 paths. Run in the ROCm image with PYTHONPATH=/opt/r9700 on one GPU.
TIMING=0 skips timing; QUICK=1 runs fewer correctness cases."""
import os, sys, statistics, torch, vllm  # noqa: F401
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as O
from r9700_vllm.attn import dsv4_decode as D

STOCK_BF16 = O.rocm_sparse_attn_decode_bf16          # before the hook section replaces the module's names

dev = torch.device("cuda:0")
H, NOPE, ROPE, HD = 8, 448, 64, 512
SCALE = HD ** -0.5
P = 70000
g = torch.Generator(device="cpu").manual_seed(11)
ok_all = True


def make_cache(nblocks, bs):
    """uint8 [nblocks, bs, 584] fp8_ds_mla pages; returns (cache, dequantized bf16 rows [nblocks*bs, 512])."""
    x = torch.randn(nblocks * bs, NOPE, generator=g) * 0.6
    sc = torch.randint(122, 130, (nblocks * bs, NOPE // 64), generator=g, dtype=torch.int32)
    fp8 = (x / torch.exp2(sc.float() - 127).repeat_interleave(64, 1)).to(torch.float8_e4m3fn)
    rope = (torch.randn(nblocks * bs, ROPE, generator=g) * 0.6).to(torch.bfloat16)
    page = torch.zeros(nblocks, bs * 584, dtype=torch.uint8)
    data = page[:, : bs * 576].view(nblocks, bs, 576)
    data[:, :, :NOPE] = fp8.view(torch.uint8).view(nblocks, bs, NOPE)
    data[:, :, NOPE:] = rope.view(torch.uint8).view(nblocks, bs, ROPE * 2)
    scales = page[:, bs * 576:].view(nblocks, bs, 8)
    scales[:, :, : NOPE // 64] = sc.to(torch.uint8).view(nblocks, bs, NOPE // 64)
    deq = torch.cat([fp8.float().to(torch.bfloat16).float() * torch.exp2(sc.float() - 127).repeat_interleave(64, 1),
                     rope.float()], 1)
    return page.view(nblocks, bs, 584).to(dev), deq.to(torch.bfloat16).to(dev)


def ragged(lens, nrows):
    idx = [torch.randperm(nrows, generator=g)[:n] for n in lens]
    ind = torch.cat(idx).to(torch.int32) if idx else torch.zeros(0, dtype=torch.int32)
    ptr = torch.zeros(len(lens) + 1, dtype=torch.int32)
    ptr[1:] = torch.cumsum(torch.tensor(lens, dtype=torch.int32), 0)
    return ind.to(dev), ptr.to(dev)


def cos_sin():
    inv = 1.0 / (10000 ** (torch.arange(0, ROPE, 2).float() / ROPE))
    f = torch.outer(torch.arange(P).float(), inv)
    return torch.cat([f.cos(), f.sin()], 1).to(dev)


CS = cos_sin()


def torch_ref(q, rows_list, sink, pos):
    H = q.shape[1]
    """fp32 reference: per query softmax over its keys (+ sink), then the inverse GPT-J RoPE on the last 64 dims."""
    out = torch.zeros(q.shape, dtype=torch.float32, device=dev)
    for t, rows in enumerate(rows_list):
        qf = q[t].float()
        if rows.shape[0] == 0 and sink is None:
            continue
        s = (qf @ rows.float().T) * SCALE if rows.shape[0] else torch.zeros(H, 0, device=dev)
        if sink is not None:
            s = torch.cat([s, sink[:, None].float()], 1)
        p = torch.softmax(s, 1)
        o = p[:, : rows.shape[0]] @ rows.float() if rows.shape[0] else torch.zeros(H, HD, device=dev)
        if pos is not None:
            c, sn = CS[pos[t], : ROPE // 2], CS[pos[t], ROPE // 2:]
            a, b = o[:, NOPE::2].clone(), o[:, NOPE + 1::2].clone()
            o[:, NOPE::2], o[:, NOPE + 1::2] = a * c + b * sn, b * c - a * sn
        out[t] = o
    return out


def stock_fallback(q, sc, si, sp, ec, ei, ep, sink, pos):
    out = O._rocm_sparse_attn_decode_ragged_triton(q, sc, si, sp, SCALE, sink, NOPE, ROPE, ec, ei, ep)
    if pos is not None:
        O.rocm_inverse_rope_rows_(out, pos, CS, ROPE)
    return out


def ours(q, sc, si, sp, ec, ei, ep, sink, pos):
    return D.sparse_decode_fp8(q, sc, si, sp, SCALE, sink, NOPE, ROPE, ec, ei, ep,
                               inv_rope_positions=pos, inv_rope_cos_sin_cache=CS if pos is not None else None)


def rel(a, b):
    """max |a - b| over max |b| (bitwise agreement is not expected: the splits change where the online softmax
    rounds p to bf16, and the fused inverse RoPE rotates before the bf16 store rather than after)."""
    return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-30)


def close_bf16(a, b, ulps=2):
    """elementwise within `ulps` bf16 ulps of the larger magnitude (floor 2^-20)."""
    m = torch.maximum(a.abs(), b.abs()).clamp_min(2.0 ** -20)
    return bool(((a - b).abs() <= ulps * torch.exp2(torch.floor(torch.log2(m)) - 7)).all())


SWA_BS = 32
swa_cache, swa_rows = make_cache(160, SWA_BS)
ext = {bs: make_cache(12800 // bs, bs) for bs in (64, 128)}

print(f"device {torch.cuda.get_device_name()} WGPs {torch.cuda.get_device_properties(0).multi_processor_count}")
print("== fp8_ds_mla, fused inverse RoPE vs stock fallback + standalone inverse RoPE ==")
cases = []
for T in (1, 6, 24):
    for (lo_s, hi_s, lo_e, hi_e) in ((128, 128, 512, 512), (1, 128, 0, 512), (1, 16, 0, 40), (1, 1, 0, 0), (100, 128, 1, 3)):
        for bs in (64, 128):
            cases.append((T, lo_s, hi_s, lo_e, hi_e, bs, True, True))
for T in (1, 6, 24):
    cases.append((T, 1, 128, 0, 0, 64, False, True))       # swa_only (no compressed segment)
    cases.append((T, 1, 128, 0, 512, 128, True, False))    # no sink
if os.environ.get("QUICK") == "1":
    cases = cases[::4]
for (T, lo_s, hi_s, lo_e, hi_e, bs, has_extra, has_sink) in cases:
    q = (torch.randn(T, H, HD, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    sink = (torch.randn(H, generator=g) * 2).to(dev) if has_sink else None
    pos = torch.randint(0, P, (T,), generator=g).to(dev)
    sl = torch.randint(lo_s, hi_s + 1, (T,), generator=g).tolist()
    si, sp = ragged(sl, swa_cache.shape[0] * SWA_BS)
    if has_extra:
        ec, erows = ext[bs]
        el = torch.randint(lo_e, hi_e + 1, (T,), generator=g).tolist()
        ei, ep = ragged(el, ec.shape[0] * bs)
    else:
        ec = ei = ep = erows = None; el = [0] * T
    rows = []
    for t in range(T):
        r = swa_rows[si[sp[t]:sp[t + 1]].long()]
        if has_extra:
            r = torch.cat([r, erows[ei[ep[t]:ep[t + 1]].long()]])
        rows.append(r)
    ref = torch_ref(q, rows, sink, pos)
    a = stock_fallback(q, swa_cache, si, sp, ec, ei, ep, sink, pos).float()
    b = ours(q, swa_cache, si, sp, ec, ei, ep, sink, pos).float()
    e_ab, e_ar, e_br = rel(b, a), rel(a, ref), rel(b, ref)
    ok = bool(torch.isfinite(b).all()) and e_br <= 1.5 * e_ar + 1e-4
    ok_all &= ok
    print(f"T={T:2d} swa {lo_s:3d}-{hi_s:3d} topk {lo_e:3d}-{hi_e:3d} bs{bs:3d} extra={int(has_extra)} sink={int(has_sink)} "
          f"splits={D._splits(T, 1, sum(sl) / T, sum(el) / T)}  ours-vs-stock {e_ab:.1e}  stock-vs-ref {e_ar:.1e}  "
          f"ours-vs-ref {e_br:.1e}  {'ok' if ok else 'FAIL'}")

print("== wrapper contract (rocm_sparse_attn_decode): rotated-row count, dense indices, non-direct output ==")
T = 6
q = (torch.randn(T, H, HD, generator=g) * 0.5).to(torch.bfloat16).to(dev)
sink = (torch.randn(H, generator=g) * 2).to(dev)
pos = torch.randint(0, P, (T,), generator=g).to(dev)
si, sp = ragged([128] * T, swa_cache.shape[0] * SWA_BS)
ec, _ = ext[64]
ei, ep = ragged([512] * T, ec.shape[0] * 64)
want = stock_fallback(q, swa_cache, si, sp, ec, ei, ep, sink, pos)
out = torch.empty_like(q)
n = D.rocm_sparse_attn_decode(q, ec, swa_cache, False, None, None, si.view(T, 128), torch.full((T,), 128, dtype=torch.int32, device=dev),
                              None, None, ei, ep, sink, SCALE, HD, NOPE, ROPE, out,
                              inv_rope_positions=pos, inv_rope_cos_sin_cache=CS)
e = rel(out.float(), want.float())
out0 = torch.empty_like(q)
n0 = D.rocm_sparse_attn_decode(q, ec, swa_cache, False, None, None, None, None, si, sp, ei, ep, sink, SCALE, HD, NOPE, ROPE, out0)
O.rocm_inverse_rope_rows_(out0, pos, CS, ROPE)
same = rel(out0.float(), out.float())
ok = n == T and n0 == 0 and e < 1e-2 and same < 1e-2
ok_all &= ok
print(f"rotated={n} (want {T}), dense SWA indices: vs stock {e:.1e}; without positions rotated={n0}, + standalone "
      f"inverse RoPE vs the fused rows {same:.1e}  {'ok' if ok else 'FAIL'}")

print("== hook: models/deepseek_v41_rdna4.patch() routes the stock names (and the model module's import) ==")
from r9700_vllm.models import deepseek_v41_rdna4 as M
M.patch()
import vllm.models.deepseek_v41.amd.rocm as R
ok = (getattr(O.rocm_sparse_attn_decode, "__r9k_patch__", None) == "dsv41_splitk_decode"
      and R.rocm_sparse_attn_decode is O.rocm_sparse_attn_decode
      and getattr(O.rocm_sparse_attn_decode_bf16, "__r9k_patch__", None) == "dsv41_splitk_decode")
out = torch.empty_like(q)
n = R.rocm_sparse_attn_decode(q=q, kv_cache=ec, swa_k_cache=swa_cache, swa_only=False, topk_indices=None, topk_lens=None,
                              swa_indices=si.view(T, 128), swa_lens=None, swa_ragged_indices=si, swa_ragged_indptr=sp,
                              topk_ragged_indices=ei, topk_ragged_indptr=ep, attn_sink=sink, scale=SCALE, head_dim=HD,
                              nope_head_dim=NOPE, rope_head_dim=ROPE, output=out, inv_rope_positions=pos,
                              inv_rope_cos_sin_cache=CS)
ok = ok and n == T and rel(out.float(), want.float()) < 1e-2
ok_all &= ok
print(f"installed: {O.rocm_sparse_attn_decode.__r9k_patch__}, via the model module rotated={n}  {'ok' if ok else 'FAIL'}")

print("== bf16 KV (single segment), our split-K vs stock bf16 split-K ==")
kvb = (torch.randn(20000, 1, HD, generator=g) * 0.6).to(torch.bfloat16).to(dev)
for T, L in ((1, 512), (6, 2048), (24, 700), (4, 37)):
    q = (torch.randn(T, 16, HD, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    sink = (torch.randn(16, generator=g) * 2).to(dev)
    ki, kp = ragged(torch.randint(max(1, L // 2), L + 1, (T,), generator=g).tolist(), kvb.shape[0])
    a = torch.empty(T, 16, HD, dtype=torch.bfloat16, device=dev); b = torch.empty_like(a)
    STOCK_BF16(q, kvb, SCALE, HD, NOPE, ROPE, sink, a, ki, kp, 4)
    D.rocm_sparse_attn_decode_bf16(q, kvb, SCALE, HD, NOPE, ROPE, sink, b, ki, kp, 4)
    rows = [kvb[ki[kp[t]:kp[t + 1]].long(), 0] for t in range(T)]
    ref = torch_ref(q, rows, sink, None)
    e_ab, e_ar, e_br = rel(b.float(), a.float()), rel(a.float(), ref), rel(b.float(), ref)
    ok = torch.isfinite(b).all().item() and e_br <= 1.5 * e_ar + 1e-4
    ok_all &= ok
    print(f"T={T:2d} len<={L:4d} ours-vs-stock {e_ab:.1e}  stock-vs-ref {e_ar:.1e}  ours-vs-ref {e_br:.1e}  {'ok' if ok else 'FAIL'}")

print("ALL OK" if ok_all else "FAILURES")

if os.environ.get("TIMING", "1") == "1":
    def graph_us(fn, inner=20, reps=31):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for _ in range(inner):
                fn()
        import time
        t_end = time.perf_counter() + 0.3          # let the clocks ramp before measuring
        while time.perf_counter() < t_end:
            gr.replay(); torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); gr.replay(); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1000 / inner)
        return statistics.median(ts)

    print("== timing (graph-captured, median us a call = one layer's decode attention incl. inverse RoPE) ==")
    print(f"{'T':>3} {'swa':>4} {'topk':>5} {'stock fallback':>15} {'ours':>8} {'splits':>7} {'warps':>6}")
    ec, _ = ext[64]
    for T in (1, 6, 24):
        for S, K in ((128, 512), (128, 0), (64, 128), (16, 16)):
            q = (torch.randn(T, H, HD, generator=g) * 0.5).to(torch.bfloat16).to(dev)
            sink = (torch.randn(H, generator=g) * 2).to(dev)
            pos = torch.randint(0, P, (T,), generator=g).to(dev)
            si, sp = ragged([S] * T, swa_cache.shape[0] * SWA_BS)
            if K:
                ei, ep = ragged([K] * T, ec.shape[0] * 64); e_c = ec
            else:
                ei = ep = e_c = None
            t0 = graph_us(lambda: stock_fallback(q, swa_cache, si, sp, e_c, ei, ep, sink, pos))
            t1 = graph_us(lambda: ours(q, swa_cache, si, sp, e_c, ei, ep, sink, pos))
            print(f"{T:3d} {S:4d} {K:5d} {t0:15.1f} {t1:8.1f} {D._splits(T, 1, S, K):7d} {D._warps(T * D._splits(T, 1, S, K)):6d}")
