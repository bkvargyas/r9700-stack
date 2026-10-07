"""r9k_moe_route (one-launch softmax top-k + align) against vLLM's topk_softmax + moe_align_block_size.

Ids must agree on every row whose selection has no near-tie (the kernels differ only in the summation order of
the row sums), weights to fp32 rounding, and the tables must be a valid moe_align_block_size layout: every real
row exactly once, each block's rows on that block's expert, experts ascending, padding = numel, ntpp = the sum
of the block-padded counts (= what the stock align reports)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from r9700_vllm.kernels import moe as KM                                     # noqa: E402
from r9700_vllm.moe import route                                            # noqa: E402
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size   # noqa: E402
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk          # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda")
E, TOPK = 512, 10
fails = 0


def check_tables(sorted_ids, expert_ids, ntpp, ids, blk, numel):
    n = int(ntpp.item())
    assert n % blk == 0, f"ntpp {n} not a multiple of {blk}"
    s = sorted_ids[:n].tolist()
    e = expert_ids[: n // blk].tolist()
    flat = ids.reshape(-1).tolist()
    seen = sorted(x for x in s if x != numel)
    assert seen == list(range(numel)), "not every routed row appears exactly once"
    last = -1
    for b in range(n // blk):
        assert e[b] >= last, "experts not ascending"
        last = e[b]
        for x in s[b * blk:(b + 1) * blk]:
            if x != numel:
                assert flat[x] == e[b], f"row {x} (expert {flat[x]}) in a block of expert {e[b]}"
    # block-padded count per expert == ntpp
    cnt = torch.bincount(ids.reshape(-1).to(torch.int64), minlength=max(E, int(ids.max()) + 1)).tolist()
    assert n == sum((c + blk - 1) // blk * blk for c in cnt), "ntpp != sum of padded counts"


for M in (1, 3, 4, 8, 16, 37, 64, 128, 256):
    for renorm in (True, False):
        for scale in (1.0, 4.0):
            x = (torch.randn(M, E, device=dev) * scale).to(torch.bfloat16)
            hidden = torch.empty((M, 8), device=dev, dtype=torch.bfloat16)
            numel = M * TOPK
            blk = KM.MOE_BLOCK * KM.pick_mt(numel, E)
            cap = route.capacity(numel, E, blk)
            sorted_ids = torch.empty(cap, dtype=torch.int32, device=dev)
            expert_ids = torch.empty(cap // blk, dtype=torch.int32, device=dev)
            ntpp = torch.zeros(1, dtype=torch.int32, device=dev)
            w, ids = route.moe_route(x, TOPK, renorm, blk, sorted_ids, expert_ids, ntpp, torch.zeros(route.scratch_ints(E), dtype=torch.int32, device=dev))
            torch.cuda.synchronize()
            w0, ids0, _ = fused_topk(hidden, x, TOPK, renorm)
            s0, e0, n0 = moe_align_block_size(ids0, blk, E)
            # ids: compare as sets per row; rows that differ must be near-ties in the stock probabilities
            p = torch.softmax(x.float(), dim=-1)
            diff_rows = 0
            for r in range(M):
                a, b = set(ids[r].tolist()), set(ids0[r].tolist())
                if a != b:
                    pa = p[r, sorted(a - b)].min().item() if a - b else 0
                    pb = p[r, sorted(b - a)].max().item() if b - a else 0
                    assert abs(pa - pb) < 1e-5, f"M={M} row {r}: ids differ beyond a near-tie ({pa} vs {pb})"
                    diff_rows += 1
            # weights: match by expert id (stock's order is by descending prob, ours too, but compare by id)
            wm = {}
            for r in range(M):
                d0 = dict(zip(ids0[r].tolist(), w0[r].tolist()))
                for k in range(TOPK):
                    i = ids[r, k].item()
                    if i in d0:
                        wm[(r, i)] = abs(w[r, k].item() - d0[i])
            maxd = max(wm.values()) if wm else 0.0
            assert maxd < 2e-6, f"M={M} renorm={renorm}: weight max abs diff {maxd}"
            check_tables(sorted_ids, expert_ids, ntpp, ids, blk, numel)
            assert int(ntpp.item()) == int(n0.item()), f"ntpp {ntpp.item()} vs stock {n0.item()}"
            print(f"M={M:3d} renorm={int(renorm)} scale={scale}: ids rows differing (near-ties) {diff_rows}, "
                  f"weight max|d| {maxd:.2e}, ntpp {ntpp.item()} blk {blk}: OK")

# degenerate rows (what a dummy / warm-up step can feed the router): NaN, +-inf, all-equal, and E not a
# multiple of 32 (padded lanes). Ids must stay valid and distinct, tables valid, no out-of-range writes.
for Ed in (512, 100, 48):
    M = 24
    x = (torch.randn(M, Ed, device=dev) * 2).to(torch.bfloat16)
    x[0] = float("nan"); x[1, :5] = float("nan"); x[2] = float("inf"); x[3] = float("-inf"); x[4] = 0.0
    x[5] = 1.0; x[6, ::2] = float("inf"); x[7] = -3e38
    tk = min(TOPK, Ed)
    numel = M * tk; blk = KM.MOE_BLOCK * KM.pick_mt(numel, Ed); cap = route.capacity(numel, Ed, blk)
    guard = 64
    s_ = torch.full((cap + guard,), -9, dtype=torch.int32, device=dev)
    e_ = torch.full((cap // blk + guard,), -9, dtype=torch.int32, device=dev)
    n_ = torch.zeros(1, dtype=torch.int32, device=dev)
    w, ids = route.moe_route(x, tk, True, blk, s_[:cap], e_[:cap // blk], n_, torch.zeros(route.scratch_ints(Ed), dtype=torch.int32, device=dev))
    torch.cuda.synchronize()
    assert int(ids.min()) >= 0 and int(ids.max()) < Ed, f"E={Ed}: id out of range"
    assert all(len(set(ids[r].tolist())) == tk for r in range(M)), f"E={Ed}: duplicate ids in a row"
    assert bool((s_[cap:] == -9).all()) and bool((e_[cap // blk:] == -9).all()), f"E={Ed}: wrote past the tables"
    assert bool(torch.isfinite(w).all()), f"E={Ed}: non-finite weight"
    check_tables(s_[:cap], e_[:cap // blk], n_, ids, blk, numel)
    print(f"degenerate rows, E={Ed}: ids valid, tables valid, weights finite: OK")
# the kernel refuses what it cannot do
cap = route.capacity(10, E, 16)
bufs = (torch.empty(cap, dtype=torch.int32, device=dev), torch.empty(cap // 16, dtype=torch.int32, device=dev),
        torch.zeros(1, dtype=torch.int32, device=dev))
try:
    route.moe_route(torch.randn(1, E, device=dev).to(torch.bfloat16), 17, True, 16, *bufs, torch.zeros(route.scratch_ints(E), dtype=torch.int32, device=dev))
    print("FAIL: topk 17 accepted"); fails += 1
except RuntimeError:
    print("topk > 16 refused: OK")
print("test_moe_route_r9k:", "FAIL" if fails else "PASS")
sys.exit(1 if fails else 0)
