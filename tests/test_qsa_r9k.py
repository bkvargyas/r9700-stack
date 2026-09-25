"""Our QSA sparse attention (kernels/r9k_qsa.hip, attn/qsa.py) vs vLLM's Triton qsa_sparse_paged_attention.

Single GPU:  python3 tests/test_qsa_r9k.py            (BENCH=0 skips the timing table)

Checks, on synthetic selections built exactly the way the indexer builds them (random distinct complete groups
below the row's position, expanded by the stock expand_qsa_block_indices_cuda, so the causal tail is included):
  * ours vs an fp32 torch reference over the listed tokens, and stock vs the same reference (so the tolerance
    is justified by what the stock kernel itself achieves)
  * prefill tiles (many rows of one request), a mixed batch (several requests, uneven lengths, padding rows
    with request -1), decode rows (1 row per request, forced splits + merge), MTP verify batches (4 rows/req)
  * GQA 6 / 1 kv head (TP=4) and GQA 12 (TP=2)
Then times prefill / decode shapes, ours vs stock.
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vllm.models.qwen4_exp.amd.ops.qsa import expand_qsa_block_indices_cuda, qsa_sparse_paged_attention  # noqa: E402

from r9700_vllm.attn import qsa as R  # noqa: E402

HD, PAGE, GS, TOPK = 256, 16, 4, 2048
dev = torch.device("cuda")


def make_case(seq_lens, q_lens, hq, hkv, seed=0, pad_rows=0):
    """One batch: per request a context length and a query length (the last q_len tokens are the queries)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    nreq = len(seq_lens)
    max_blocks = max((s + PAGE - 1) // PAGE for s in seq_lens)
    nblocks = sum((s + PAGE - 1) // PAGE for s in seq_lens) + 3
    perm = torch.randperm(nblocks - 1, generator=g) + 1        # block 0 never used: catches a zero-default bug
    bt = torch.full((nreq, max_blocks), -1, dtype=torch.int32)
    at = 0
    for r, s in enumerate(seq_lens):
        nb = (s + PAGE - 1) // PAGE
        bt[r, :nb] = perm[at:at + nb].to(torch.int32)
        at += nb
    kv = torch.randn((nblocks, hkv, PAGE, 2 * HD), generator=g).to(torch.bfloat16)
    key_cache, value_cache = kv.transpose(1, 2).split(HD, dim=-1)       # the stock views: [blocks, page, hkv, 256]
    rows = sum(q_lens)
    q = torch.randn((rows + pad_rows, hq, HD), generator=g).to(torch.bfloat16)
    t2r = torch.full((rows + pad_rows,), -1, dtype=torch.int32)
    qsl = torch.zeros(nreq + 1, dtype=torch.int32)
    pos = torch.empty(rows, dtype=torch.int32)
    at = 0
    for r, (s, ql) in enumerate(zip(seq_lens, q_lens)):
        t2r[at:at + ql] = r
        pos[at:at + ql] = torch.arange(s - ql, s)
        qsl[r + 1] = at + ql
        at += ql
    # indexer-style selection: random distinct complete groups below the position, -1 padded, then expanded
    block_topk = TOPK // GS
    blocks = torch.full((rows, block_topk), -1, dtype=torch.int32)
    for i in range(rows):
        vis = min(int(pos[i] + 1) // GS, block_topk)
        if vis:
            blocks[i, :vis] = torch.randperm(int(pos[i] + 1) // GS, generator=g)[:vis].to(torch.int32)
    sl = torch.tensor(seq_lens, dtype=torch.int32)
    d = lambda t: t.to(dev)
    idx = expand_qsa_block_indices_cuda(d(blocks), d(pos), d(sl), d(t2r[:rows]), GS, TOPK)
    idx = torch.cat([idx, torch.full((pad_rows, idx.shape[1]), -1, dtype=torch.int32, device=dev)])
    return dict(q=d(q), kc=d(key_cache), vc=d(value_cache), idx=idx, bt=d(bt), t2r=d(t2r), qsl=d(qsl), sl=d(sl),
                pos=pos, rows=rows, hq=hq, hkv=hkv)


def reference(c, row_ids):
    """fp32 softmax attention over each row's listed tokens (the stock semantics)."""
    out = torch.zeros((len(row_ids), c["hq"], HD), dtype=torch.float32, device=dev)
    gqa = c["hq"] // c["hkv"]
    for n, i in enumerate(row_ids):
        req = int(c["t2r"][i])
        if req < 0:
            continue
        toks = c["idx"][i]
        toks = toks[toks >= 0].long()
        phys = c["bt"][req][toks // PAGE].long()
        slot = toks % PAGE
        k = c["kc"][phys, slot].float()            # [n, hkv, 256]
        v = c["vc"][phys, slot].float()
        qf = c["q"][i].float()                     # [hq, 256]
        for h in range(c["hq"]):
            kh = h // gqa
            s = (k[:, kh] @ qf[h]) / 16.0
            p = torch.softmax(s, 0)
            out[n, h] = p @ v[:, kh]
    return out


def run_ours(c, nsplit=None):
    out = torch.zeros_like(c["q"])
    R.sparse_attention(c["q"], c["kc"], c["vc"], c["idx"], c["bt"], c["t2r"], c["qsl"], c["sl"], out, nsplit=nsplit)
    return out


def run_stock(c):
    out = torch.zeros_like(c["q"])
    n = c["rows"]
    qsa_sparse_paged_attention(c["q"][:n], c["kc"], c["vc"], c["idx"][:n], c["bt"], c["t2r"][:n], out[:n])
    return out


def check(name, c, nsplit=None, nref=48):
    ours, stock = run_ours(c, nsplit), run_stock(c)
    torch.cuda.synchronize()
    g = torch.Generator(device="cpu").manual_seed(1)
    ids = torch.randperm(c["rows"], generator=g)[:nref].tolist()
    if c["q"].shape[0] > c["rows"]:
        ids.append(c["q"].shape[0] - 1)                      # a padding row: must stay zero
    ref = reference(c, ids)
    eo = (ours[ids].float() - ref).abs()
    es = (stock[ids].float() - ref).abs()
    scale = ref.abs().amax(dim=-1, keepdim=True).clamp_min(1e-3)
    ro, rs = (eo / scale).max().item(), (es / scale).max().item()
    ok = ro <= max(3 * rs, 1.5e-2) and torch.isfinite(ours).all().item()
    print(f"  {name:<44} max rel err ours {ro:.2e}  stock {rs:.2e}  {'ok' if ok else 'FAIL'}")
    return ok


def bench(name, c, iters=20):
    for f, lab in ((run_stock, "stock"), (run_ours, "ours")):
        for _ in range(3):
            f(c)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            f(c)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t) / iters * 1e6
        print(f"  {name:<40} {lab:<6} {us:9.0f} us")


def main():
    if not R.available():
        print("libr9k.so has no r9k_qsa_attn: rebuild kernels/")
        sys.exit(1)
    bad = 0
    # TP=4 geometry: 6 q heads, 1 kv head
    bad += not check("prefill 1x(ctx 4096, q 4096) GQA6", make_case([4096], [4096], 6, 1, seed=1))
    bad += not check("prefill chunk 1x(ctx 12288, q 4096) GQA6", make_case([12288], [4096], 6, 1, seed=2))
    bad += not check("mixed 3 reqs + 5 pad rows GQA6",
                     make_case([300, 9000, 17], [300, 1500, 17], 6, 1, seed=3, pad_rows=5))
    bad += not check("decode 4 reqs x 1 row, splits 16 GQA6", make_case([8000, 3, 12345, 640], [1, 1, 1, 1], 6, 1, seed=4),
                     nsplit=16)
    bad += not check("decode 4 reqs x 1 row, splits auto GQA6", make_case([8000, 3, 12345, 640], [1, 1, 1, 1], 6, 1, seed=4))
    bad += not check("mtp verify 4 reqs x 4 rows GQA6", make_case([8000, 30, 12345, 640], [4, 4, 4, 4], 6, 1, seed=5))
    bad += not check("mtp verify 4 reqs x 4 rows splits 8 GQA6", make_case([8000, 30, 12345, 640], [4, 4, 4, 4], 6, 1, seed=5),
                     nsplit=8)
    bad += not check("tiny ctx (1..20) 6 reqs GQA6", make_case([1, 2, 3, 4, 5, 20], [1, 2, 3, 4, 5, 20], 6, 1, seed=6))
    # TP=2 geometry: 12 q heads
    bad += not check("prefill 1x(ctx 4096, q 4096) GQA12", make_case([4096], [4096], 12, 1, seed=7))
    bad += not check("decode 2 reqs x 1 row GQA12", make_case([5000, 777], [1, 1], 12, 1, seed=8))
    # 2 kv heads (GQA 6 each), exercises the kv-head stride
    bad += not check("prefill 2 kv heads GQA6", make_case([2048, 2048], [2048, 100], 12, 2, seed=9))
    print("correctness:", "PASS" if bad == 0 else f"FAIL ({bad})")

    if os.environ.get("BENCH", "1") == "1":
        print("timing (GQA 6, 1 kv head = TP4):")
        bench("prefill chunk ctx 12288 q 4096", make_case([12288], [4096], 6, 1, seed=11))
        bench("prefill ctx 4096 q 4096", make_case([4096], [4096], 6, 1, seed=12))
        bench("prefill 2x(ctx 2048 q 2048)", make_case([2048, 2048], [2048, 2048], 6, 1, seed=13))
        bench("decode 1 row ctx 8000", make_case([8000], [1], 6, 1, seed=14), iters=100)
        bench("decode 4 rows ctx ~8000", make_case([8000, 8100, 7900, 8200], [1, 1, 1, 1], 6, 1, seed=15), iters=100)
        bench("mtp 4 reqs x 4 rows ctx ~8000", make_case([8000, 8100, 7900, 8200], [4, 4, 4, 4], 6, 1, seed=16), iters=100)
        bench("mtp 16 reqs x 4 rows ctx ~8000", make_case([8000 + 37 * i for i in range(16)], [4] * 16, 6, 1, seed=17),
              iters=50)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
