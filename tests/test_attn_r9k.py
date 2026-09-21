"""libr9k paged attention (kernels/r9k_attn.hip) == stock unified attention on a K/V-packed LBHNC cache.

Same shapes / comparison as test_attn_r4d.py plus chunked-prefill cases (q_len < seq_len with the chunk start not on a
block boundary), where causal-offset bugs hide. Prints GPU time per call against unified attention and, when
r4d.so is present, libr4d. R9K_ATTN_ITERS sets the timing iterations (default 10)."""
import os, sys, time
os.environ.setdefault("R9K_PAGED_ATTN", "r4d")     # only so _r4d_kernels() yields the libr4d timing column
import torch
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from r9700_vllm.attn.triton3d import _plan
from r9700_vllm.kernels.attn import prefill_kernels

ITERS = int(os.environ.get("R9K_ATTN_ITERS", "10"))
kern = prefill_kernels()
prefill = kern["bf16"]
try:
    from r9700_vllm.attn.triton3d import _r4d_kernels
    r4d = _r4d_kernels()
except Exception:
    r4d = None
ok = True
torch.manual_seed(0)
dev, hd, hq, hkv, B = "cuda", 256, 12, 2, 16


def gpu_ms(fn, iters=ITERS):
    fn(); torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


def make_case(qlens, ctxs, kv_dtype=torch.bfloat16):
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
    return kv, bt, q, qsl, sl


SHAPES = [([4096], [4096]), ([2000], [9000]), ([300, 300, 8], [700, 1500, 900]), ([8, 8, 8, 8], [150, 400, 1200, 60]),
          ([1], [33]),
          # chunked prefill: chunk starts off a block boundary, several sequences with equal / unequal q_len
          ([1000], [1037]), ([4096], [8195]), ([500, 500], [1700, 531]), ([17, 17, 3], [1000, 40, 19]),
          ([1024, 1024, 1024, 1024], [1024, 2048, 3072, 4096]), ([16], [16]), ([15], [4095])]
for qlens, ctxs in SHAPES:
    kv, bt, q, qsl, sl = make_case(qlens, ctxs)
    kc, vc = kv.transpose(1, 2).split(hd, dim=-1)
    ref = torch.empty_like(q)
    def run_ref():
        unified_attention(q, kc, vc, ref, qsl.to(dev), max(qlens), sl, max(ctxs), hd ** -0.5, True, (-1, -1), bt, 0,
                          None, None, None)
    run_ref()
    out = torch.full_like(q, float("nan"))
    stream = torch.cuda.current_stream().cuda_stream
    q_row = hq * hd * 2
    groups = _plan(qsl, len(ctxs))
    def run():
        for first_req, nseq, q_len, first_tok in groups:
            prefill(q.data_ptr() + first_tok * q_row, kv.data_ptr(), bt.data_ptr() + first_req * bt.shape[1] * 4,
                    sl.data_ptr() + first_req * 4, out.data_ptr() + first_tok * q_row, 0, 0, 0, nseq, q_len,
                    hq, hkv, hd, B, bt.shape[1], kv.stride(0), kv.stride(1), hd ** -0.5, 0, 32768, stream)
    run(); torch.cuda.synchronize()
    err = ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    good = err < 2e-2 and torch.isfinite(out.float()).all().item()
    ok &= good
    t_ref, t_r9k = gpu_ms(run_ref), gpu_ms(run)
    t_r4d = ""
    if r4d is not None:
        kernels, maxq, scratch_fn = r4d
        pf, dc = kernels["bf16"]
        scratch = torch.empty(max(scratch_fn(len(ctxs), maxq, hq, hkv, hd, 32768, 0), 1), dtype=torch.uint8, device=dev)
        out2 = torch.empty_like(q)
        def run_r4d():
            for first_req, nseq, q_len, first_tok in groups:
                (dc if q_len <= maxq else pf)(
                    q.data_ptr() + first_tok * q_row, kv.data_ptr(), bt.data_ptr() + first_req * bt.shape[1] * 4,
                    sl.data_ptr() + first_req * 4, out2.data_ptr() + first_tok * q_row, 0, 0, scratch.data_ptr(), nseq,
                    q_len, hq, hkv, hd, B, bt.shape[1], kv.stride(0), kv.stride(1), hd ** -0.5, 0, 32768, stream)
        t_r4d = f"  r4d {gpu_ms(run_r4d):7.3f} ms"
    print(f"  q={qlens} ctx={ctxs}: max rel {err:.2e} {'ok' if good else '<-- FAIL'}   unified {t_ref:7.3f} ms  "
          f"r9k {t_r9k:7.3f} ms{t_r4d}", flush=True)

pf8 = kern.get("fp8_e4m3")
if pf8 is None:
    print("  (no fp8 kernel in this build)")
else:
    for qlens, ctxs in [([4096], [4096]), ([2000], [9000]), ([300, 8], [700, 900]), ([1000], [1037])]:
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
        out = torch.full_like(q, float("nan"))
        kb = torch.full((64 * hkv,), ks, device=dev)
        vb = torch.full((64 * hkv,), vs, device=dev)
        q_row = hq * hd * 2
        stream = torch.cuda.current_stream().cuda_stream
        groups = _plan(qsl, len(ctxs))
        def run8():
            for first_req, nseq, q_len, first_tok in groups:
                pf8(q.data_ptr() + first_tok * q_row, kv8.data_ptr(), bt.data_ptr() + first_req * bt.shape[1] * 4,
                    sl.data_ptr() + first_req * 4, out.data_ptr() + first_tok * q_row, kb.data_ptr(), vb.data_ptr(),
                    0, nseq, q_len, hq, hkv, hd, B, bt.shape[1], kv8.stride(0), kv8.stride(1), hd ** -0.5, 0, 32768,
                    stream)
        run8(); torch.cuda.synchronize()
        err = ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
        good = err < 3e-2 and torch.isfinite(out.float()).all().item()
        ok &= good
        print(f"  fp8 KV q={qlens} ctx={ctxs}: max rel {err:.2e} {'ok' if good else '<-- FAIL'}   r9k {gpu_ms(run8):7.3f} ms",
              flush=True)
print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
