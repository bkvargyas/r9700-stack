#!/usr/bin/env python3
"""skew_traces.py TRACE_DIR [--json OUT]: expert-parallel rank skew from the eight per-rank torch-profiler traces of one
profile window (package Z2's ~/ds41/exl3/Z2/<tag>/s<N>/, GPU-only or CPU+GPU). Package H3, notes/exl3-ep-skew.md.

Per rank: the GPU kernels in time order. Every MoE layer call = k_gu + k_dn (the r9k expert tier) followed by the post-MoE
all-reduce; the all-reduce before its k_gu is the post-attention one. Calls are matched across ranks by the all-reduce's
END time (the eight ranks leave a one-kernel / split-path all-reduce together; the trace clocks are host time) and the
match is checked by kernel template. Layer labels: the drafter's layers are k_gu<4,..> (K = 4); the target's 40 layers
are counted from the first k_gu after a drafter call (K pattern 3 x 18, 2 x 5, 3 x 17, checked).

Reports, per MoE call and summed a step:
  tier_r    = k_gu + k_dn on rank r (the expert tier: ~ distinct local experts x 13.27 MB at its bandwidth)
  seg_r     = previous all-reduce end -> this all-reduce start on rank r (everything rank r computes before it arrives)
  ar_r      = the all-reduce kernel on rank r; min_r ar_r = the collective itself, ar_r - min = rank r waiting
  late      = argmax_r seg_r (the rank everybody waits for)
  skew      = max_r seg_r - mean_r seg_r (what perfect balance of the same work would save on that call)
  tier skew = max_r tier_r - mean_r tier_r
"""
import bisect, collections, glob, json, os, re, statistics as st, sys

AR = re.compile(r"r9k_ar8|ncclDevKernel|allreduce|cross_device")
GU = re.compile(r"k_gu<(\d+)")
DN = re.compile(r"k_dn<(\d+)")


def kernels(path):
    ev = json.load(open(path))["traceEvents"]
    ks = sorted(((e["ts"], e["dur"], e["name"]) for e in ev if e.get("cat") == "kernel" and "dur" in e), key=lambda x: x[0])
    del ev
    return ks


def calls(ks):
    """[(kind, ts, dur, name)] of the interesting kernels; kind in gu / dn / ar."""
    out = []
    for ts, d, n in ks:
        n2 = n.replace("(anonymous namespace)::", "")
        if GU.search(n2):
            out.append(("gu", ts, d, n2))
        elif DN.search(n2):
            out.append(("dn", ts, d, n2))
        elif AR.search(n2) and "ag8" not in n2:
            out.append(("ar", ts, d, n2))
    return out


def moe_records(cs):
    """One record per MoE call: k_gu, k_dn, the post-MoE all-reduce, the all-reduce before (post-attention)."""
    recs = []
    last_ar = None
    i = 0
    while i < len(cs):
        k = cs[i]
        if k[0] == "ar":
            last_ar = k
            i += 1
            continue
        if k[0] == "gu" and i + 1 < len(cs) and cs[i + 1][0] == "dn":
            j = i + 2
            while j < len(cs) and cs[j][0] != "ar":
                j += 1
            if j >= len(cs) or last_ar is None:
                break
            gu, dn, ar = k, cs[i + 1], cs[j]
            K = int(GU.search(gu[3]).group(1))
            recs.append({"K": K, "gu": gu[2], "dn": dn[2], "tier": gu[2] + dn[2], "ar": ar[2], "ar_ts": ar[1],
                         "ar_end": ar[1] + ar[2], "seg": ar[1] - (last_ar[1] + last_ar[2]),
                         "pre_ar": last_ar[2], "pre_ar_end": last_ar[1] + last_ar[2],
                         "gu_ts": gu[1], "tmpl": re.sub(r"\[clone.*", "", gu[3])[:40]})
            last_ar = ar
            i = j + 1
            continue
        i += 1
    return recs


def label(recs):
    """layer label: 'd0..d2' drafter (K=4), 't0..t39' target; None where the count does not fit."""
    n = None
    for r in recs:
        if r["K"] == 4:
            r["layer"] = None
            n = 0
        elif n is not None and n < 40:
            want = 2 if 18 <= n <= 22 else 3
            r["layer"] = n if r["K"] == want else None
            n += 1
        else:
            r["layer"] = None
    # drafter layers: index within the K=4 run
    run = 0
    for r in recs:
        if r["K"] == 4:
            r["dlayer"] = run
            run += 1
        else:
            r["dlayer"] = None
            run = 0
    return recs


def main():
    d = sys.argv[1]
    js = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    files = sorted(glob.glob(os.path.join(d, "*rank*.pt.trace.json")), key=lambda f: int(re.search(r"rank(\d+)", f).group(1)))
    R = [label(moe_records(calls(kernels(f)))) for f in files]
    nr = len(R)
    # match by post-MoE all-reduce end time, anchored on rank 0
    ends = [[r["ar_end"] for r in rr] for rr in R]
    groups = []
    for i, r0 in enumerate(R[0]):
        g = [r0]
        ok = True
        for k in range(1, nr):
            j = bisect.bisect_left(ends[k], r0["ar_end"])
            cand = [c for c in (j - 1, j) if 0 <= c < len(ends[k])]
            if not cand:
                ok = False; break
            c = min(cand, key=lambda c: abs(ends[k][c] - r0["ar_end"]))
            rk = R[k][c]
            if abs(rk["ar_end"] - r0["ar_end"]) > 20 or rk["K"] != r0["K"]:
                ok = False; break
            g.append(rk)
        if ok:
            groups.append(g)
    print(f"{d}: {nr} ranks, MoE calls a rank {[len(x) for x in R]}, matched {len(groups)} (of rank 0's {len(R[0])})")
    endsk = [max(r["ar_end"] for r in g) - min(r["ar_end"] for r in g) for g in groups]
    print(f"  all-reduce end spread across ranks: median {st.median(endsk):.1f} us, p95 {sorted(endsk)[int(.95*len(endsk))]:.1f}")
    out = {"dir": d, "ranks": nr, "calls": len(groups)}
    # per call quantities
    rows = []
    for g in groups:
        seg = [r["seg"] for r in g]
        tier = [r["tier"] for r in g]
        ar = [r["ar"] for r in g]
        pre = [r["pre_ar"] for r in g]
        late = max(range(nr), key=lambda k: seg[k])
        hot = max(range(nr), key=lambda k: tier[k])
        rows.append({"K": g[0]["K"], "layer": g[0]["layer"], "dlayer": g[0]["dlayer"], "seg": seg, "tier": tier, "ar": ar,
                     "pre": pre, "late": late, "hot": hot, "ar_min": min(ar),
                     "skew": max(seg) - st.mean(seg), "tskew": max(tier) - st.mean(tier),
                     "wait0": ar[0] - min(ar), "pre_wait0": pre[0] - min(pre)})
    tgt = [x for x in rows if x["K"] != 4 and x["layer"] is not None]
    dft = [x for x in rows if x["K"] == 4]
    steps = len([x for x in tgt if x["layer"] == 0]) or 1
    print(f"  target MoE calls {len(tgt)} ({steps} steps with layer 0), drafter {len(dft)}")

    def summ(xs, tag):
        if not xs:
            return {}
        n = len(xs)
        res = {
            "calls": n,
            "tier_mean_rank_us": [st.mean(x["tier"][k] for x in xs) for k in range(nr)],
            "tier_mean_over_ranks": st.mean(st.mean(x["tier"]) for x in xs),
            "tier_max_over_ranks": st.mean(max(x["tier"]) for x in xs),
            "seg_mean_over_ranks": st.mean(st.mean(x["seg"]) for x in xs),
            "seg_max_over_ranks": st.mean(max(x["seg"]) for x in xs),
            "ar_rank0": st.mean(x["ar"][0] for x in xs), "ar_min": st.mean(x["ar_min"] for x in xs),
            "ar_mean_rank_us": [st.mean(x["ar"][k] for x in xs) for k in range(nr)],
            "pre_ar_rank0": st.mean(x["pre"][0] for x in xs), "pre_ar_min": st.mean(min(x["pre"]) for x in xs),
            "skew_us_call": st.mean(x["skew"] for x in xs), "tier_skew_us_call": st.mean(x["tskew"] for x in xs),
            "late_rank_hist": collections.Counter(x["late"] for x in xs),
            "hot_rank_hist": collections.Counter(x["hot"] for x in xs),
            "late_is_hot": sum(1 for x in xs if x["late"] == x["hot"]) / n,
        }
        print(f"  -- {tag}: {n} calls")
        print("     tier (k_gu+k_dn) mean by rank us: " + " ".join(f"{v:6.1f}" for v in res["tier_mean_rank_us"]))
        print(f"     tier mean over ranks {res['tier_mean_over_ranks']:.1f}, max over ranks {res['tier_max_over_ranks']:.1f} "
              f"(tier skew {res['tier_skew_us_call']:.1f} us a call)")
        print(f"     segment before the post-MoE all-reduce: mean over ranks {res['seg_mean_over_ranks']:.1f}, max {res['seg_max_over_ranks']:.1f} "
              f"(skew {res['skew_us_call']:.1f} us a call)")
        print(f"     post-MoE all-reduce: rank0 {res['ar_rank0']:.1f}, min over ranks {res['ar_min']:.1f}; by rank "
              + " ".join(f"{v:6.1f}" for v in res["ar_mean_rank_us"]))
        print(f"     post-attention all-reduce: rank0 {res['pre_ar_rank0']:.1f}, min {res['pre_ar_min']:.1f}")
        print(f"     late rank (max segment) histogram {dict(sorted(res['late_rank_hist'].items()))}; hot rank (max tier) "
              f"{dict(sorted(res['hot_rank_hist'].items()))}; late == hot in {100*res['late_is_hot']:.0f} %")
        res["late_rank_hist"] = dict(res["late_rank_hist"]); res["hot_rank_hist"] = dict(res["hot_rank_hist"])
        return res

    out["target"] = summ(tgt, "target, all layers")
    out["drafter"] = summ(dft, "drafter")
    # per layer: hot rank persistence (is it the same rank for a layer step after step?)
    per = collections.defaultdict(list)
    for x in tgt:
        per[x["layer"]].append(x)
    print("  per target layer: tier mean / max over ranks us, the most frequent hot rank and its share, skew us")
    pl = {}
    for l in sorted(per):
        xs = per[l]
        hh = collections.Counter(x["hot"] for x in xs).most_common(1)[0]
        rm = [st.mean(x["tier"][k] for x in xs) for k in range(nr)]
        pl[l] = {"n": len(xs), "tier_mean": st.mean(st.mean(x["tier"]) for x in xs), "tier_max": st.mean(max(x["tier"]) for x in xs),
                 "hot_rank": hh[0], "hot_share": hh[1] / len(xs), "skew": st.mean(x["skew"] for x in xs),
                 "tier_by_rank": rm, "tier_skew": st.mean(x["tskew"] for x in xs),
                 "struct_tier_skew": max(rm) - st.mean(rm)}
        print(f"    L{l:2d} n={len(xs):3d} tier {pl[l]['tier_mean']:6.1f} / {pl[l]['tier_max']:6.1f}  hot r{hh[0]} {100*hh[1]/len(xs):3.0f} %  "
              f"seg skew {pl[l]['skew']:5.1f}  tier skew {pl[l]['tier_skew']:5.1f} (of which structural {pl[l]['struct_tier_skew']:5.1f})  by rank "
              + " ".join(f"{v:5.1f}" for v in rm))
    out["per_layer"] = pl
    # a step: sums over the 40 target layers (per-step means)
    S = steps
    tot = lambda key: sum(x[key] for x in tgt) / S
    struct = sum(v["struct_tier_skew"] * v["n"] for v in pl.values()) / S
    out["per_step"] = {
        "steps": S, "seg_skew_ms": tot("skew") / 1000, "tier_skew_ms": tot("tskew") / 1000,
        "structural_tier_skew_ms": struct / 1000,
        "post_moe_wait_rank0_ms": tot("wait0") / 1000, "post_attn_wait_rank0_ms": tot("pre_wait0") / 1000,
        "ar_min_ms": sum(x["ar_min"] for x in tgt) / S / 1000,
        "tier_mean_ms": sum(st.mean(x["tier"]) for x in tgt) / S / 1000,
        "tier_max_ms": sum(max(x["tier"]) for x in tgt) / S / 1000,
    }
    p = out["per_step"]
    print(f"  A STEP (40 target MoE calls, {S} steps): segment skew {p['seg_skew_ms']:.2f} ms, expert-tier skew {p['tier_skew_ms']:.2f} ms "
          f"(structural = per-layer rank means {p['structural_tier_skew_ms']:.2f} ms), rank-0 wait post-MoE {p['post_moe_wait_rank0_ms']:.2f} "
          f"+ post-attention {p['post_attn_wait_rank0_ms']:.2f} ms, collective itself (min) {p['ar_min_ms']:.2f} ms; "
          f"tier mean over ranks {p['tier_mean_ms']:.2f} ms vs max {p['tier_max_ms']:.2f}")
    if js:
        json.dump(out, open(js, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
