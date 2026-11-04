#!/usr/bin/env python3
"""skew_place.py FIT.json OUT_MAP.json --train RING.pt[,RING.pt] --test RING.pt[,RING.pt] [--replicas R] [--iters N]
Package H3 (notes/exl3-ep-skew.md): placement / replica policies for the EXL3 expert-parallel tier, costed on recorded
routing (bench/ds41/epskew/epskew_ext.py rings) with the tier cost model of skew_model.py fit
(tier on a rank = a + b n + c (s - n) per (K, verify slots); a step pays sum over layers of max over ranks).
numpy + torch (run in r9700/vllm:dev, CPU only).

Policies (each: ms a step = mean over steps of sum over the 40 target (+ 3 drafter) layers of max over the 8 ranks, and
the same for the mean over ranks = the balanced floor of the same work):
  linear      stock placement, rank r owns experts [48 r, 48 r + 48)
  random      a random permutation (sanity)
  placed      load-time placement map: greedy by the train histogram, then swap local search on the train steps
              (sum over steps of max over ranks); written to OUT_MAP.json (R9K_EXL3_EP_MAP, r9700_vllm/moe/exl3_placement.py)
  +replicas   R extra copies a layer (R / 8 a rank: 8 = one a rank a layer = 40 x 12.66 MiB = 0.49 GiB a rank) of the train-hottest experts, on the ranks where they cost least; a step's copy is
              chosen on device by every rank alike (same routing on all ranks): greedy, least-loaded host
  eplb-hash   the same replicas, the copy chosen by stock EPLB's token hash (load-blind)
  floor       every rank at the mean (perfect balance of the same distinct experts; no placement reaches it)
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skew_model as SM

R = 8


class Cost:
    def __init__(self, fits):
        self.f = {tuple(int(x) for x in k.split(",")): v for k, v in fits.items()}

    def coef(self, K, T):
        ks = [k for k in self.f if k[0] == K] or [k for k in self.f if k[0] == 3]
        k = min(ks, key=lambda k: (abs(k[1] - T), k[1]))
        v = self.f[k]
        return v["a"], v["b"], v["c"]


def step_costs(rows, owner, K, cost, Rk=R):
    """per step: (max over ranks, mean over ranks) tier us"""
    n, s = SM.local_counts(rows, owner, Rk)
    T = SM.width(rows)
    out_max = np.zeros(len(rows)); out_mean = np.zeros(len(rows))
    for t in np.unique(T):
        m = T == t
        a, b, c = cost.coef(K, int(t))
        tier = a + b * n[m] + c * (s[m] - n[m])
        out_max[m] = tier.max(1); out_mean[m] = tier.mean(1)
    return out_max, out_mean


def occurrences(rows, G):
    """per expert: (steps array, count array) over the steps it appears in"""
    occ = [[] for _ in range(G)]
    st, sl = np.nonzero(rows >= 0)
    ids = rows[st, sl]
    key = st.astype(np.int64) * G + ids
    u, c = np.unique(key, return_counts=True)
    es, ss = u % G, u // G
    order = np.argsort(es, kind="stable")
    es, ss, c = es[order], ss[order], c[order]
    bounds = np.searchsorted(es, np.arange(G + 1))
    return [(ss[bounds[e]:bounds[e + 1]], c[bounds[e]:bounds[e + 1]]) for e in range(G)]


def optimize(rows, G, K, cost, freq, iters, seed=0):
    """load-time placement for one layer: owner[G] -> rank, 48 a rank (G / 8)."""
    rng = np.random.default_rng(seed)
    cap = G // R
    # greedy: hottest first onto the least-loaded rank with room
    owner = -np.ones(G, int); load = np.zeros(R); room = np.full(R, cap)
    for e in np.argsort(-freq, kind="stable"):
        r = min((r for r in range(R) if room[r] > 0), key=lambda r: load[r])
        owner[e] = r; load[r] += freq[e]; room[r] -= 1
    T = SM.width(rows)
    st = len(rows)
    ab = np.zeros((st, 3))
    for t in np.unique(T):
        ab[T == t] = cost.coef(K, int(t))
    b, c = ab[:, 1], ab[:, 2]
    occ = occurrences(rows, G)
    n, s = SM.local_counts(rows, owner)
    w = b[:, None] * n + c[:, None] * (s - n)          # variable part of the tier a rank a step
    cur = w.max(1)
    acc = 0
    hot = np.argsort(-freq)
    for it in range(iters):
        # a: an expert on the worst rank of a costly step; b: any expert on another rank
        if it % 2 == 0:
            i = rng.integers(st); r = int(np.argmax(w[i]))
            cand = [e for e in np.nonzero(rows[i] >= 0)[0]]
            ea = int(rows[i][rng.choice(cand)]) if cand else int(rng.integers(G))
            if owner[ea] != r:
                ea = int(rng.integers(G))
        else:
            ea = int(hot[rng.integers(min(G, 96))])
        eb = int(rng.integers(G))
        p, q = owner[ea], owner[eb]
        if p == q:
            continue
        sa, ca = occ[ea]; sb, cb = occ[eb]
        idx = np.union1d(sa, sb)
        if len(idx) == 0:
            owner[ea], owner[eb] = q, p
            continue
        wn = w[idx].copy()
        ia = np.searchsorted(idx, sa); ib = np.searchsorted(idx, sb)
        # expert a moves p -> q, b moves q -> p (distinct +1 per step present, slots + count)
        da = b[sa] + c[sa] * (ca - 1); db = b[sb] + c[sb] * (cb - 1)
        wn[ia, p] -= da; wn[ia, q] += da
        wn[ib, q] -= db; wn[ib, p] += db
        newmax = wn.max(1)
        delta = (newmax - cur[idx]).sum()
        if delta < -1e-9:
            w[idx] = wn; cur[idx] = newmax
            owner[ea], owner[eb] = q, p
            acc += 1
    return owner, acc


def replicas_for(owner, freq, rows, G, K, cost, nrep):
    """nrep extra copies: the hottest experts (by train frequency), each copy on the rank with the lowest train load
    that does not hold it yet. Returns hosts[e] = list of ranks."""
    hosts = [[int(owner[e])] for e in range(G)]
    def load():
        L = np.zeros(R)
        for e in range(G):
            for h in hosts[e]:
                L[h] += freq[e] / len(hosts[e])
        return L
    hot = [int(e) for e in np.argsort(-freq)]
    cap = -(-nrep // R); extra = np.zeros(R, int)    # extra copies a rank (12.66 MiB each at K = 3)
    k = 0
    while nrep > 0 and k < 4 * G:
        e = hot[k % G]
        k += 1
        cands = [r for r in range(R) if r not in hosts[e] and extra[r] < cap]
        if not cands:
            continue
        L = load()
        r = min(cands, key=lambda r: L[r])
        hosts[e].append(r); extra[r] += 1
        nrep -= 1
    return hosts


def replica_costs(rows, hosts, K, cost, mode):
    """per step max/mean tier with replicas: fixed experts on their only host; replicated ones (in slot order of first
    appearance) to the least-loaded host (mode greedy: by the current variable cost) or by stock EPLB's token hash."""
    st = len(rows)
    T = SM.width(rows)
    out_max = np.zeros(st); out_mean = np.zeros(st)
    multi = np.array([len(h) > 1 for h in hosts])
    single = np.array([h[0] for h in hosts])
    for i in range(st):
        ids = rows[i][rows[i] >= 0]
        a, b, c = cost.coef(K, int(T[i]))
        u, first, cnt = np.unique(ids, return_index=True, return_counts=True)
        n = np.zeros(R); s = np.zeros(R)
        fix = ~multi[u]
        np.add.at(n, single[u[fix]], 1); np.add.at(s, single[u[fix]], cnt[fix])
        rep = np.nonzero(~fix)[0]
        for j in rep[np.argsort(first[rep])]:
            hs = hosts[u[j]]
            if mode == "greedy":
                r = min(hs, key=lambda r: (b * n[r] + c * (s[r] - n[r]), r))
                n[r] += 1; s[r] += cnt[j]
            else:   # stock EPLB: each slot picks copy hash(token) % copies (load-blind), so one expert may run on several hosts
                used = {}
                for t in np.nonzero(rows[i] == u[j])[0] // 6:
                    r = hs[((int(t) * 2654435769) & 0xFFFFFFFF) % len(hs)]
                    used[r] = used.get(r, 0) + 1
                for r, k in used.items():
                    n[r] += 1; s[r] += k
        tier = a + b * n + c * (s - n)
        out_max[i] = tier.max(); out_mean[i] = tier.mean()
    return out_max, out_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fit"); ap.add_argument("out")
    ap.add_argument("--train", required=True); ap.add_argument("--test", action="append", default=[])
    ap.add_argument("--replicas", default="0,8,16,32"); ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--layers", default="all")
    A = ap.parse_args()
    cost = Cost(json.load(open(A.fit))["fits"])
    train = [SM.load_ring(p) for p in A.train.split(",")]
    tests = {os.path.basename(t).replace(".pt", ""): [SM.load_ring(p) for p in t.split(",")] for t in A.test}
    K_of = lambda li: 4 if li >= 40 else (2 if 18 <= li <= 22 else 3)
    lays = sorted({l["layer"] for l in train[0]})
    if A.layers != "all":
        lays = [int(x) for x in A.layers.split(",")]
    maps = {}
    owners = {}
    hostsets = {int(r): {} for r in A.replicas.split(",")}
    for li in lays:
        rows = np.concatenate([[l for l in tr if l["layer"] == li][0]["rows"] for tr in train])
        G = [l for l in train[0] if l["layer"] == li][0]["G"]
        freq = np.bincount(rows[rows >= 0], minlength=G).astype(float)
        own, acc = optimize(rows, G, K_of(li), cost, freq, A.iters, seed=li)
        owners[li] = own
        maps[str(li)] = {"G": G, "owner": [int(x) for x in own]}
        for rtot in hostsets:
            per = rtot if li < 40 else 0             # extra copies a target layer (over all ranks; rtot / 8 a rank)
            hostsets[rtot][li] = replicas_for(own, freq, rows, G, K_of(li), cost, per)
        print(f"layer {li}: placement swaps accepted {acc}", file=sys.stderr, flush=True)
    json.dump({"version": 1, "ranks": R, "layers": maps,
               "note": "owner[g] = EP rank holding global expert g; 48 a rank (16 for the drafter's 128); skew_place.py"},
              open(A.out, "w"))
    res = {}
    for tag, rings in [("train", train)] + list(tests.items()):
        for ri, ring in enumerate(rings):
            nm = f"{tag}[{ri}]"
            tot = {}
            for li in lays:
                L = [l for l in ring if l["layer"] == li]
                if not L:
                    continue
                rows, G = L[0]["rows"], L[0]["G"]
                K = K_of(li)
                rng = np.random.default_rng(1000 + li)
                pol = {"linear": SM.linear_owner(G), "random": rng.permutation(SM.linear_owner(G)), "placed": owners[li]}
                for p, own in pol.items():
                    mx, mn = step_costs(rows, own, K, cost)
                    tot.setdefault(p, [0, 0]); tot[p][0] += mx.mean(); tot[p][1] += mn.mean()
                    if p == "linear":
                        tot.setdefault("floor", [0, 0]); tot["floor"][0] += mn.mean(); tot["floor"][1] += mn.mean()
                for rtot, hs in hostsets.items():
                    if rtot == 0 or li >= 40:
                        continue
                    for mode in ("greedy", "hash"):
                        mx, mn = replica_costs(rows, hs[li], K, cost, mode)
                        key = f"placed+{rtot} replicas ({'device greedy' if mode == 'greedy' else 'eplb hash'})"
                        tot.setdefault(key, [0, 0]); tot[key][0] += mx.mean(); tot[key][1] += mn.mean()
                for rtot, hs in hostsets.items():
                    if li >= 40 and rtot:
                        for mode in ("greedy", "hash"):
                            key = f"placed+{rtot} replicas ({'device greedy' if mode == 'greedy' else 'eplb hash'})"
                            mx, mn = step_costs(rows, owners[li], K, cost)
                            tot.setdefault(key, [0, 0]); tot[key][0] += mx.mean(); tot[key][1] += mn.mean()
            res[nm] = {k: {"max_ms": v[0] / 1000, "mean_ms": v[1] / 1000} for k, v in tot.items()}
            print(f"== {nm} ({len(ring[0]['rows'])} steps, slots a step {SM.width(ring[0]['rows']).mean():.1f}): expert tier ms a step "
                  f"(sum over layers of max over ranks; the mean over ranks in brackets)")
            base = tot["linear"][0]
            for k, v in tot.items():
                print(f"   {k:40s} {v[0]/1000:6.2f} ({v[1]/1000:5.2f})   vs linear {(v[0]-base)/1000:+6.2f} ms")
    json.dump(res, open(A.out.replace(".json", "-eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
