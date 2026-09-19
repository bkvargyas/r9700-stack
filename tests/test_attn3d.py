"""Split-KV (3D) unified attention with multi-token queries (attn/triton3d.py) == stock 2D path."""
import sys
import torch
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

ok = True
torch.manual_seed(0)
dev = "cuda"
for (seqs, qlen, hq, hkv, hd, blk, ctx) in [(1, 8, 12, 2, 256, 16, 173), (4, 8, 12, 2, 256, 16, 700),
                                             (2, 5, 32, 8, 128, 16, 64), (3, 8, 12, 2, 256, 16, 5)]:
    ctx_lens = [ctx + 37 * i for i in range(seqs)]
    nblk_per = [(c + blk - 1) // blk for c in ctx_lens]
    nblocks = sum(nblk_per) + 4
    k = torch.randn(nblocks, blk, hkv, hd, device=dev, dtype=torch.bfloat16)
    v = torch.randn(nblocks, blk, hkv, hd, device=dev, dtype=torch.bfloat16)
    perm = torch.randperm(nblocks, device=dev)
    bt = torch.zeros(seqs, max(nblk_per), dtype=torch.int32, device=dev)
    o = 0
    for i, n in enumerate(nblk_per):
        bt[i, :n] = perm[o:o + n]
        o += n
    qlens = [min(qlen, c) for c in ctx_lens]
    T = sum(qlens)
    q = torch.randn(T, hq, hd, device=dev, dtype=torch.bfloat16)
    cu = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0)), dtype=torch.int32, device=dev)
    sl = torch.tensor(ctx_lens, dtype=torch.int32, device=dev)
    common = dict(max_seqlen_k=max(ctx_lens), softmax_scale=hd ** -0.5, causal=True, window_size=(-1, -1),
                  block_table=bt, softcap=0, q_descale=None, k_descale=None, v_descale=None)
    o2 = torch.empty_like(q)
    unified_attention(q, k, v, o2, cu, max(qlens), sl, **common)            # stock 2D
    segs = 16
    so = torch.empty(T, hq, segs, triton_pad := 1 << (hd - 1).bit_length(), device=dev)
    sm = torch.empty(T, hq, segs, device=dev)
    se = torch.empty(T, hq, segs, device=dev)
    o3 = torch.empty_like(q)
    unified_attention(q, k, v, o3, cu, 1, sl, **common, seq_threshold_3D=seqs, num_par_softmax_segments=segs,
                      softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se)   # forced 3D
    err = ((o3.float() - o2.float()).abs().max() / o2.float().abs().max()).item()
    good = err < 2e-2
    ok &= good
    print(f"  seqs={seqs} q/seq={qlen} hq={hq} hkv={hkv} hd={hd} ctx~{ctx}: 3D vs 2D max rel {err:.2e} "
          f"{'ok' if good else '<-- FAIL'}")
print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
