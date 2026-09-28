"""Our QSA indexer scoring (kernels/r9k_qsa_score.hip, attn/qsa_score.py) vs vLLM's Triton qsa_mqa_paged and the
stock end-to-end qsa_select_paged_tokens.

Single GPU:  python3 tests/test_qsa_score.py            (BENCH=0 skips the timing table)

Checks, on synthetic compressed-key caches with a permuted page table (page 0 never used):
  * logits on every visible column vs an fp32 torch reference (relu(q.k) summed over heads / sqrt(128)), ours and
    stock against the same reference; visible_blocks equal to stock's
  * the selected token lists of the full pipeline (our scorer -> stock top_k_per_row_decode -> stock expansion)
    vs stock's; rows that differ must differ only in blocks whose reference score ties with the k-th best
  * prefill tiles (one request), chunked prefill (context longer than the chunk), a mixed batch with padding rows
    (request -1), decode rows, MTP verify rows, tiny contexts, odd page sizes, 1/2/8 heads
Then times prefill / decode shapes, ours vs stock.
"""
from __future__ import annotations

import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vllm.models.qwen4_exp.amd.ops.qsa import qsa_mqa_paged, qsa_select_paged_tokens  # noqa: E402

from r9700_vllm.attn import qsa_score as S  # noqa: E402

HD, CR, TOPK = 128, 4, 2048
dev = torch.device("cuda")


def make_case(seq_lens, q_lens, heads=4, page=16, seed=0, pad_rows=0):
    """Per request a context length (tokens) and a query length (the last q_len tokens are the queries).
    The compressed cache holds one row per CR tokens; the page table is over compressed rows."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    nreq = len(seq_lens)
    ncols = [(s + CR - 1) // CR for s in seq_lens]
    max_pages = max(max((c + page - 1) // page for c in ncols), 1)
    npages = sum((c + page - 1) // page for c in ncols) + 3
    perm = torch.randperm(npages - 1, generator=g) + 1
    bt = torch.full((nreq, max_pages), -1, dtype=torch.int32)
    at = 0
    for r, c in enumerate(ncols):
        nb = (c + page - 1) // page
        bt[r, :nb] = perm[at:at + nb].to(torch.int32)
        at += nb
    kc = torch.randn((npages, page, 1, HD), generator=g).to(torch.bfloat16)
    rows = sum(q_lens)
    q = torch.randn((rows + pad_rows, heads, HD), generator=g).to(torch.bfloat16)
    t2r = torch.full((rows + pad_rows,), -1, dtype=torch.int32)
    pos = torch.zeros(rows + pad_rows, dtype=torch.int32)
    at = 0
    for r, (s, ql) in enumerate(zip(seq_lens, q_lens)):
        t2r[at:at + ql] = r
        pos[at:at + ql] = torch.arange(s - ql, s)
        at += ql
    sl = torch.tensor(seq_lens, dtype=torch.int32)
    d = lambda t: t.to(dev)
    return dict(q=d(q), kc=d(kc), bt=d(bt), t2r=d(t2r), pos=d(pos), sl=d(sl), rows=rows + pad_rows, page=page,
                heads=heads, seq_lens=seq_lens)


def reference(c):
    """fp32 scores for every row over its visible columns: list of (visible, scores[visible])."""
    kc = c["kc"].float()
    q = c["q"].float()
    out = []
    for i in range(c["rows"]):
        req = int(c["t2r"][i])
        if req < 0:
            out.append((0, None))
            continue
        vis = min((int(c["pos"][i]) + 1) // CR, int(c["sl"][req]) // CR)
        if vis == 0:
            out.append((0, None))
            continue
        cols = torch.arange(vis, device=dev)
        phys = c["bt"][req][cols // c["page"]].long()
        keys = kc[phys, cols % c["page"], 0]                       # [vis, 128]
        s = torch.relu(keys @ q[i].T).sum(1) / math.sqrt(HD)       # [vis]
        out.append((vis, s))
    return out


def run_stock_score(c):
    return qsa_mqa_paged(c["q"], c["kc"], c["bt"], c["t2r"], c["pos"], c["sl"], CR)


def run_ours_score(c, nsplit=None):
    return S.score(c["q"], c["kc"], c["bt"], c["t2r"], c["pos"], c["sl"], CR, nsplit=nsplit)


def run_stock_select(c):
    return qsa_select_paged_tokens(c["q"], c["kc"], c["bt"], c["t2r"], c["pos"], c["sl"], TOPK, CR)


def run_ours_select(c):
    return S.select_paged_tokens(c["q"], c["kc"], c["bt"], c["t2r"], c["pos"], c["sl"], TOPK, CR)


def check(name, c, nsplit=None):
    ref = reference(c)
    lo, vo = run_ours_score(c, nsplit)
    ls, vs = run_stock_score(c)
    torch.cuda.synchronize()
    vis_ok = torch.equal(vo, vs)
    ro = rs = 0.0
    for i, (vis, s) in enumerate(ref):
        if vis == 0:
            continue
        scale = s.abs().max().clamp_min(1e-3)
        ro = max(ro, ((lo[i, :vis] - s).abs().max() / scale).item())
        rs = max(rs, ((ls[i, :vis] - s).abs().max() / scale).item())
        if not torch.isfinite(lo[i, :vis]).all():
            ro = float("inf")
    score_ok = ro <= max(3 * rs, 2e-3)
    # end to end: selected token lists
    so = run_ours_select(c)
    ss = run_stock_select(c)
    torch.cuda.synchronize()
    same = 0
    sel_ok = True
    block_topk = TOPK // CR
    for i, (vis, s) in enumerate(ref):
        a, b = so[i].sort().values, ss[i].sort().values
        if torch.equal(a, b):
            same += 1
            continue
        if vis == 0 or vis <= block_topk:
            sel_ok = False           # nothing to select among: lists must be identical
            continue
        # differing blocks must tie (within tolerance) with the k-th best reference score
        kth = s.topk(block_topk).values[-1]
        tol = 1e-3 * s.abs().max().clamp_min(1e-3)
        diff = torch.tensor(sorted(set(a.tolist()) ^ set(b.tolist())), device=dev)
        diff = diff[diff >= 0] // CR
        if diff.numel() and ((s[diff] - kth).abs() > tol).any():
            sel_ok = False
    ok = vis_ok and score_ok and sel_ok
    print(f"  {name:<46} score rel err ours {ro:.2e} stock {rs:.2e}  visible {'=' if vis_ok else '!='}  "
          f"select identical {same}/{c['rows']} {'ok' if ok else 'FAIL'}")
    return ok


def bench(name, c, iters=20):
    for f, lab in ((run_stock_score, "stock"), (run_ours_score, "ours"), (run_stock_select, "stock e2e"),
                   (run_ours_select, "ours e2e")):
        for _ in range(3):
            f(c)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            f(c)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t) / iters * 1e6
        print(f"  {name:<40} {lab:<10} {us:9.0f} us")


def main():
    if not S.available():
        print("libr9k.so has no r9k_qsa_score: rebuild kernels/")
        sys.exit(1)
    bad = 0
    bad += not check("prefill 1x(ctx 4096, q 4096)", make_case([4096], [4096], seed=1))
    bad += not check("prefill chunk 1x(ctx 12288, q 4096)", make_case([12288], [4096], seed=2))
    bad += not check("prefill chunk 1x(ctx 32768, q 4096)", make_case([32768], [4096], seed=3))
    bad += not check("prefill 1x(ctx 4096, q 4096) forced 5 splits", make_case([4096], [4096], seed=1), nsplit=5)
    bad += not check("mixed uneven tiles 5 reqs forced 3 splits",
                     make_case([33, 4097, 1000, 8191, 5], [33, 7, 1000, 9, 5], seed=5), nsplit=3)
    bad += not check("mixed 3 reqs + 5 pad rows", make_case([300, 9000, 17], [300, 1500, 17], seed=4, pad_rows=5))
    bad += not check("mixed uneven tiles 5 reqs", make_case([33, 4097, 1000, 8191, 5], [33, 7, 1000, 9, 5], seed=5))
    bad += not check("decode 4 reqs x 1 row", make_case([8000, 3, 12345, 640], [1, 1, 1, 1], seed=6))
    bad += not check("mtp verify 4 reqs x 4 rows", make_case([8000, 30, 12345, 640], [4, 4, 4, 4], seed=7))
    bad += not check("tiny ctx (1..20) 6 reqs", make_case([1, 2, 3, 4, 5, 20], [1, 2, 3, 4, 5, 20], seed=8))
    bad += not check("page 4, ctx 3000", make_case([3000, 777], [3000, 1], page=4, seed=9))
    bad += not check("page 64, ctx 9000", make_case([9000], [2048], page=64, seed=10))
    bad += not check("1 head", make_case([6000], [2048], heads=1, seed=11))
    bad += not check("2 heads", make_case([6000], [2048], heads=2, seed=12))
    bad += not check("8 heads", make_case([6000], [2048], heads=8, seed=13))
    print("correctness:", "PASS" if bad == 0 else f"FAIL ({bad})")

    if os.environ.get("BENCH", "1") == "1":
        print("timing (4 heads, page 16):")
        bench("prefill chunk ctx 4096 q 4096", make_case([4096], [4096], seed=21))
        bench("prefill chunk ctx 12288 q 4096", make_case([12288], [4096], seed=22))
        bench("prefill chunk ctx 32768 q 4096", make_case([32768], [4096], seed=23))
        bench("prefill ctx 2048 q 2048", make_case([2048], [2048], seed=24))
        bench("decode 1 row ctx 8000", make_case([8000], [1], seed=25), iters=100)
        bench("mtp 4 reqs x 4 rows ctx ~8000", make_case([8000, 8100, 7900, 8200], [4, 4, 4, 4], seed=26), iters=100)
        bench("mtp 16 reqs x 4 rows ctx ~8000", make_case([8000 + 37 * i for i in range(16)], [4] * 16, seed=27),
              iters=50)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
