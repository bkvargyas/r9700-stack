"""libr4d paged attention (attn/triton3d.py prefill / mixed path) == stock unified attention on a K/V-packed LBHNC cache."""
import os, sys, time
os.environ.setdefault("R9K_PAGED_ATTN", "r4d")     # this test exercises the libr4d pair
import torch
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from r9700_vllm.attn.triton3d import _r4d_kernels, _plan

k = _r4d_kernels()
if k is None:
    print("no libr4d attention kernels"); sys.exit(1)
kernels, maxq, scratch_fn = k
prefill, decode = kernels["bf16"]
ok = True
torch.manual_seed(0)
dev, hd, hq, hkv, B = "cuda", 256, 12, 2, 16
for qlens, ctxs in [([4096], [4096]), ([2000], [9000]), ([300, 300, 8], [700, 1500, 900]), ([8, 8, 8, 8], [150, 400, 1200, 60]),
                    ([1], [33])]:
    nb = [(c + B - 1) // B for c in ctxs]
    nblocks = sum(nb) + 8
    kv = torch.randn(nblocks, hkv, B, 2 * hd, device=dev, dtype=torch.bfloat16)      # LBHNC per-layer view
    perm = torch.randperm(nblocks, device=dev)
    bt = torch.zeros(len(ctxs), max(nb), dtype=torch.int32, device=dev)
    o = 0
    for i, n in enumerate(nb):
        bt[i, :n] = perm[o:o + n].int(); o += n
    T = sum(qlens)
    q = torch.randn(T, hq, hd, device=dev, dtype=torch.bfloat16)
    qsl = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0)), dtype=torch.int32)
    sl = torch.tensor(ctxs, dtype=torch.int32, device=dev)
    kc, vc = kv.transpose(1, 2).split(hd, dim=-1)
    ref = torch.empty_like(q)
    unified_attention(q, kc, vc, ref, qsl.to(dev), max(qlens), sl, max(ctxs), hd ** -0.5, True, (-1, -1), bt, 0,
                      None, None, None)
    out = torch.empty_like(q)
    scratch = torch.empty(max(scratch_fn(len(ctxs), maxq, hq, hkv, hd, 32768, 0), 1), dtype=torch.uint8, device=dev)
    stream = torch.cuda.current_stream().cuda_stream
    q_row = hq * hd * 2
    def run():
        for first_req, nseq, q_len, first_tok in _plan(qsl, len(ctxs)):
            (decode if q_len <= maxq else prefill)(
                q.data_ptr() + first_tok * q_row, kv.data_ptr(), bt.data_ptr() + first_req * bt.shape[1] * 4,
                sl.data_ptr() + first_req * 4, out.data_ptr() + first_tok * q_row, 0, 0, scratch.data_ptr(), nseq, q_len,
                hq, hkv, hd, B, bt.shape[1], kv.stride(0), kv.stride(1), hd ** -0.5, 0, 32768, stream)
    run(); torch.cuda.synchronize()
    err = ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    good = err < 2e-2
    ok &= good
    tr = []
    for fn in (lambda: unified_attention(q, kc, vc, ref, qsl.to(dev), max(qlens), sl, max(ctxs), hd ** -0.5, True,
                                         (-1, -1), bt, 0, None, None, None), run):
        fn(); torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(5): fn()
        torch.cuda.synchronize(); tr.append((time.perf_counter() - t) / 5 * 1e3)
    print(f"  q={qlens} ctx={ctxs}: max rel {err:.2e} {'ok' if good else '<-- FAIL'}   unified {tr[0]:7.2f} ms  r4d {tr[1]:7.2f} ms")
# fp8 KV cache with per-tensor K/V scales (the 27B's served configuration) -> fp8 kernels + descale buffers
pf8, dc8 = kernels["fp8_e4m3"]
for qlens, ctxs in [([4096], [4096]), ([2000], [9000]), ([300, 8], [700, 900])]:
    nb = [(c + B - 1) // B for c in ctxs]
    nblocks = sum(nb) + 8
    ks, vs = 0.05, 0.03
    kvf = torch.randn(nblocks, hkv, B, 2 * hd, device=dev)
    kvf[..., :hd] /= ks
    kvf[..., hd:] /= vs
    kv8 = kvf.clamp(-448, 448).to(torch.float8_e4m3fn)
    perm = torch.randperm(nblocks, device=dev)
    bt = torch.zeros(len(ctxs), max(nb), dtype=torch.int32, device=dev)
    o = 0
    for i, n in enumerate(nb):
        bt[i, :n] = perm[o:o + n].int(); o += n
    T = sum(qlens)
    q = torch.randn(T, hq, hd, device=dev, dtype=torch.bfloat16)
    qsl = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0)), dtype=torch.int32)
    sl = torch.tensor(ctxs, dtype=torch.int32, device=dev)
    # dense fp32 torch reference (dequantized K/V, causal by absolute position)
    ref = torch.empty(T, hq, hd, device=dev)
    K = (kv8[..., :hd].float() * ks).permute(1, 0, 2, 3)
    V = (kv8[..., hd:].float() * vs).permute(1, 0, 2, 3)
    for i, (qlen_i, ctx_i) in enumerate(zip(qlens, ctxs)):
        blocks = bt[i, :nb[i]].long()
        Ki = K[:, blocks].reshape(hkv, -1, hd)[:, :ctx_i].repeat_interleave(hq // hkv, 0)
        Vi = V[:, blocks].reshape(hkv, -1, hd)[:, :ctx_i].repeat_interleave(hq // hkv, 0)
        t0 = int(qsl[i])
        qi = q[t0:t0 + qlen_i].float().permute(1, 0, 2)
        Sm = qi @ Ki.transpose(1, 2) * hd ** -0.5
        pq = torch.arange(ctx_i - qlen_i, ctx_i, device=dev)[:, None]
        Sm = Sm.masked_fill(torch.arange(ctx_i, device=dev)[None] > pq, float("-inf"))
        ref[t0:t0 + qlen_i] = (Sm.softmax(-1) @ Vi).permute(1, 0, 2)
    out = torch.empty_like(q)
    kb = torch.full((64 * hkv,), ks, device=dev)
    vb = torch.full((64 * hkv,), vs, device=dev)
    scratch = torch.empty(max(scratch_fn(len(ctxs), maxq, hq, hkv, hd, 32768, 0), 1), dtype=torch.uint8, device=dev)
    q_row = hq * hd * 2
    for first_req, nseq, q_len, first_tok in _plan(qsl, len(ctxs)):
        (dc8 if q_len <= maxq else pf8)(
            q.data_ptr() + first_tok * q_row, kv8.data_ptr(), bt.data_ptr() + first_req * bt.shape[1] * 4,
            sl.data_ptr() + first_req * 4, out.data_ptr() + first_tok * q_row, kb.data_ptr(), vb.data_ptr(),
            scratch.data_ptr(), nseq, q_len, hq, hkv, hd, B, bt.shape[1], kv8.stride(0), kv8.stride(1), hd ** -0.5, 0,
            32768, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    err = ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    good = err < 3e-2
    ok &= good
    print(f"  fp8 KV q={qlens} ctx={ctxs}: max rel {err:.2e} {'ok' if good else '<-- FAIL'}")
print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
