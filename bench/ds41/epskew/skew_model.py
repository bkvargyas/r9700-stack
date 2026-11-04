#!/usr/bin/env python3
"""skew_model.py: package H3's expert-parallel skew model (notes/exl3-ep-skew.md). numpy + torch (run inside r9700/vllm:dev,
CPU only: `docker run --rm -v ...:/w --entrypoint python3 r9700/vllm:dev /w/.../skew_model.py ...`).

  fit RING.pt TRACE_DIR OUT.json
      align the routing ring of a profile window (run_skew.py prof:N:W) with the eight rank traces of the same window, call
      by call per layer (offset chosen by the correlation between local distinct experts and k_gu + k_dn time), and fit
      the expert tier's time on a rank: tier = a + b * n + c * (s - n), n = distinct local experts, s = local slots, per
      (K, verify width T). Also checks the model: predicted vs measured sum over layers of max over ranks.
  hist OUT.json RING.pt [RING.pt ...]
      the routing distribution: picks a rank a step (mean, max / mean), the hottest experts' share, per-rank structural
      load, cross-workload stability of the per-expert frequencies.
Ring layout (bench/ds41/epskew/epskew_ext.py): ring [layers, N, 144] int16 global ids (-1 pad), ptr [layers], names, G."""
import json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_ring(path):
    d = torch.load(path)
    ring, ptr, N = d["ring"].numpy(), d["ptr"].numpy(), d["N"]
    layers = []
    for i, (nm, G) in enumerate(zip(d["names"], d["G"])):
        m = re.search(r"layers\.(\d+)\.", nm)
        li = int(m.group(1)) if m else -1
        p = int(ptr[i])
        if p <= N:
            rows = ring[i, :p]
        else:      # wrapped: oldest first
            k = p % N
            rows = np.concatenate([ring[i, k:], ring[i, :k]])
        layers.append({"name": nm, "layer": li, "G": int(G), "rows": rows, "steps": len(rows)})
    return layers


def width(rows):
    """slots used a step"""
    return (rows >= 0).sum(1)


def local_counts(rows, owner, R=8):
    """rows [steps, 144] global ids -> n [steps, R] distinct local experts, s [steps, R] local slots under owner[G] -> rank"""
    st = rows.shape[0]
    n = np.zeros((st, R), np.int32); s = np.zeros((st, R), np.int32)
    valid = rows >= 0
    own = np.where(valid, owner[np.clip(rows, 0, None)], -1)
    for r in range(R):
        s[:, r] = (own == r).sum(1)
    # distinct: sort ids per row, count first occurrences
    srt = np.sort(np.where(valid, rows, 32767), 1)
    first = np.ones_like(srt, bool); first[:, 1:] = srt[:, 1:] != srt[:, :-1]
    first &= srt != 32767
    ow2 = np.where(srt != 32767, owner[np.clip(srt, 0, len(owner) - 1)], -1)
    for r in range(R):
        n[:, r] = ((ow2 == r) & first).sum(1)
    return n, s


def linear_owner(G, R=8):
    return np.arange(G) // (G // R)


def cmd_fit(ring_path, tdir, out):
    import skew_traces as ST, glob
    files = sorted(glob.glob(os.path.join(tdir, "*rank*.pt.trace.json")), key=lambda f: int(re.search(r"rank(\d+)", f).group(1)))
    recs = [ST.label(ST.moe_records(ST.calls(ST.kernels(f)))) for f in files]
    L = {l["layer"]: l for l in load_ring(ring_path)}
    pts = []
    rep = {"layers": {}}
    for li in sorted(L):
        lay = L[li]
        if li < 40:
            tr = [[r for r in rr if r["layer"] == li] for rr in recs]
        else:
            tr = [[r for r in rr if r["K"] == 4 and r["dlayer"] == li - 40] for rr in recs]
        Q = min(len(t) for t in tr)
        if Q < 10:
            continue
        tier = np.array([[t[i]["tier"] for t in tr] for i in range(Q)])       # [Q, 8]
        K = tr[0][0]["K"]
        own = linear_owner(lay["G"])
        n, s = local_counts(lay["rows"], own)
        T = width(lay["rows"])
        P = len(n)
        best = None
        for o in range(0, max(P - Q, 0) + 1):
            a = n[o:o + Q].astype(float).ravel(); b = tier.ravel()
            if a.std() == 0:
                continue
            c = np.corrcoef(a, b)[0, 1]
            if best is None or c > best[0]:
                best = (c, o)
        if best is None:
            continue
        c, o = best
        rep["layers"][int(li)] = {"K": K, "corr": round(float(c), 3), "offset": o, "Q": Q, "P": P}
        for i in range(Q):
            for r in range(8):
                pts.append((K, int(T[o + i]), int(n[o + i, r]), int(s[o + i, r]), float(tier[i, r]), li, i))
    pts = np.array(pts, dtype=float)
    rep["points"] = len(pts)
    # fit per (K, T)
    fits = {}
    for K in sorted(set(pts[:, 0].astype(int))):
        for T in sorted(set(pts[pts[:, 0] == K][:, 1].astype(int))):
            m = (pts[:, 0] == K) & (pts[:, 1] == T)
            if m.sum() < 40:
                continue
            X = np.stack([np.ones(m.sum()), pts[m, 2], pts[m, 3] - pts[m, 2]], 1)
            coef, *_ = np.linalg.lstsq(X, pts[m, 4], rcond=None)
            pred = X @ coef
            r2 = 1 - ((pts[m, 4] - pred) ** 2).sum() / ((pts[m, 4] - pts[m, 4].mean()) ** 2).sum()
            tab = {}
            for nn in sorted(set(pts[m, 2].astype(int))):
                mm = m & (pts[:, 2] == nn)
                tab[int(nn)] = [round(float(pts[mm, 4].mean()), 1), int(mm.sum())]
            fits[f"{K},{T}"] = {"a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2]), "r2": float(r2), "n": int(m.sum()),
                                "by_n": tab}
            print(f"K={K} T={T:3d} slots: tier = {coef[0]:6.1f} + {coef[1]:5.1f} n + {coef[2]:5.1f} (s - n)   R2 {r2:.3f}  ({m.sum()} points)"
                  f"  mean by n: " + " ".join(f"{k}:{v[0]}" for k, v in list(tab.items())[:10]))
    rep["fits"] = fits
    # model check: sum over the calls of max over ranks, measured vs predicted from the fit
    meas = pred = 0.0
    for K, T, *_ in []:
        pass
    groups = {}
    for p in pts:
        groups.setdefault((int(p[5]), int(p[6])), []).append(p)
    for g in groups.values():
        key = f"{int(g[0][0])},{int(g[0][1])}"
        if key not in fits:
            continue
        f = fits[key]
        meas += max(p[4] for p in g)
        pred += max(f["a"] + f["b"] * p[2] + f["c"] * (p[3] - p[2]) for p in g)
    rep["check_sum_max_meas_us"] = meas; rep["check_sum_max_pred_us"] = pred
    print(f"check: sum over calls of max over ranks: measured {meas/1000:.1f} ms, model {pred/1000:.1f} ms ({len(groups)} calls)")
    print("alignment corr by layer: " + " ".join(f"{k}:{v['corr']}" for k, v in rep["layers"].items()))
    json.dump(rep, open(out, "w"), indent=1)


def cmd_hist(out, paths):
    rep = {}
    allfreq = {}
    for path in paths:
        tag = os.path.basename(path).replace(".pt", "")
        Ls = load_ring(path)
        r = {"layers": {}}
        tot_max = tot_mean = 0.0
        for lay in Ls:
            G = lay["G"]; rows = lay["rows"]
            if lay["steps"] < 10:
                continue
            own = linear_owner(G)
            n, s = local_counts(rows, own)
            ids = rows[rows >= 0]
            freq = np.bincount(ids, minlength=G).astype(float)
            allfreq.setdefault(lay["layer"], {})[tag] = freq
            share = np.sort(freq)[::-1] / freq.sum()
            rank_load = np.bincount(own, weights=freq, minlength=8) / freq.sum() * 8   # 1.0 = even
            T = width(rows)
            r["layers"][lay["layer"]] = {
                "G": G, "steps": lay["steps"], "slots_mean": float(T.mean()),
                "picks_rank_step_mean": float(s.mean()), "picks_max_over_mean": float(s.max(1).mean() / max(s.mean(), 1e-9)),
                "distinct_rank_step_mean": float(n.mean()), "distinct_max_mean": float(n.max(1).mean()),
                "top1_share": float(share[0]), "top8_share": float(share[:8].sum()), "top24_share": float(share[:24].sum()),
                "top48_share": float(share[:48].sum()), "unused_experts": int((freq == 0).sum()),
                "rank_load_rel": [round(float(x), 3) for x in rank_load]}
        rep[tag] = r
        tl = [v for k, v in r["layers"].items() if 0 <= k < 40]
        if tl:
            print(f"{tag}: {tl[0]['steps']} steps, slots a step {np.mean([v['slots_mean'] for v in tl]):.1f}; "
                  f"distinct local experts a rank a step mean {np.mean([v['distinct_rank_step_mean'] for v in tl]):.2f}, "
                  f"max over ranks {np.mean([v['distinct_max_mean'] for v in tl]):.2f}; picks max/mean {np.mean([v['picks_max_over_mean'] for v in tl]):.2f}; "
                  f"top-1 / 8 / 24 / 48 experts' share {np.mean([v['top1_share'] for v in tl]):.3f} / {np.mean([v['top8_share'] for v in tl]):.3f} / "
                  f"{np.mean([v['top24_share'] for v in tl]):.3f} / {np.mean([v['top48_share'] for v in tl]):.3f} (uniform 0.003 / 0.021 / 0.063 / 0.125); "
                  f"worst rank's structural load {np.mean([max(v['rank_load_rel']) for v in tl]):.2f} x mean")
    # stability: correlation of per-expert frequencies between workloads, per layer
    stab = {}
    tags = sorted({t for d in allfreq.values() for t in d})
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            cs = [np.corrcoef(d[a], d[b])[0, 1] for li, d in allfreq.items() if 0 <= li < 40 and a in d and b in d]
            if cs:
                stab[f"{a}~{b}"] = float(np.mean(cs))
    rep["freq_corr"] = stab
    print("per-expert frequency correlation between workloads (mean over target layers): " +
          ", ".join(f"{k} {v:.2f}" for k, v in stab.items()))
    json.dump(rep, open(out, "w"), indent=1)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "fit":
        cmd_fit(*sys.argv[2:5])
    elif c == "hist":
        cmd_hist(sys.argv[2], sys.argv[3:])
