#!/usr/bin/env python3
"""widths.py TRACE.json [DRAFT_NAMES.json]: the verify width the GPU actually ran in each step (adaptive verification trims
it on device below the scheduled k+1 tokens a stream). v0.4.1+: read off the r9k_exl3_gemv template's row count. v0.4.0: read off the shared-expert w13 GEMM of the forward replay: hipBLAS
picks wvSplitK_hf_sml_<.., N> for N <= 4 rows (N = rows) and a Cijk tile above; for > 4 rows the EXL3 GEMV launch count
a layer (one exl3_gemv launch per 8-row chunk) gives ceil(rows / 8). Prints the distribution of widths over the steps."""
import collections, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ledger
if len(sys.argv) > 2:
    ledger.DRAFT_NAMES.update(json.load(open(sys.argv[2])))
steps, phases, ks, rt = ledger.analyze(ledger.load(sys.argv[1]))
by = collections.defaultdict(list)
for k in ks:
    if k["phase"] == "forward":
        by[k["step"]].append(k)
dist = collections.Counter()
for s, kk in by.items():
    # v0.4.1+: the r9k_exl3_gemv launches carry the row count: k_gemv<bits, cb, M> (M = 1..8) or k_gemv_mma<bits, cb, RT, NB2>
    # (RT 16-row tiles, 9..48 rows); the most common one in the forward is the verify width (every attention linear sees it)
    g = collections.Counter()
    for k in kk:
        m = re.match(r"(?:\(anonymous namespace\)::)?k_gemv<\d+, \w+, (\d+)>", k["short"])
        if m:
            g[f"{m.group(1)} rows"] += 1; continue
        m = re.match(r"(?:\(anonymous namespace\)::)?k_gemv_mma<\d+, \w+, (\d+), \d+>", k["short"])
        if m:
            rt = int(m.group(1)); g[f"{max(9, 16 * rt - 15)}-{16 * rt} rows (k_gemv_mma, {rt} row tiles)"] += 1
    if g:
        dist[g.most_common(1)[0][0]] += 1
        continue
    # v0.4.0: the shared-expert w13 GEMM: hipBLAS picks wvSplitK_hf_sml_<.., N> for N <= 4 rows, a Cijk tile above
    w = [re.search(r", (\d+)>$", k["short"]) for k in kk if k["short"].startswith("wvSplitK_hf_sml_<__hip_bfloat16, 32, 2, 16")]
    if w and w[0]:
        dist[f"{w[0].group(1)} rows"] += 1
        continue
    n = sum(1 for k in kk if k["short"].startswith("exl3_gemv_mr_dot_multi"))
    rec = sum(1 for k in kk if k["short"].startswith("reconstruct_kernel"))
    dist[f"> 4 rows: {n / 40:.0f} GEMV chunks of 8 a layer" + (f", {rec} reconstructs" if rec else "")] += 1
tot = sum(dist.values())
for k, v in sorted(dist.items(), key=lambda kv: -kv[1]):
    print(f"  {v:4d} steps ({100 * v / tot:4.1f} %)  {k}")
