#!/usr/bin/env python3
"""skew_noise.py FIT.json RING.pt [RING.pt ...]: the expert-tier skew a step that uniform random routing would leave with the
same verify widths (6 distinct experts a token, iid tokens): the per-step multinomial noise no placement removes. Package H3."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skew_model as SM, skew_place as SP

cost = SP.Cost(json.load(open(sys.argv[1]))["fits"])
for p in sys.argv[2:]:
    ring = SM.load_ring(p); rng = np.random.default_rng(0)
    tot = {"real linear": [0, 0], "uniform routing": [0, 0]}
    for L in ring:
        li, G, rows = L["layer"], L["G"], L["rows"]
        K = 4 if li >= 40 else (2 if 18 <= li <= 22 else 3)
        mx, mn = SP.step_costs(rows, SM.linear_owner(G), K, cost)
        tot["real linear"][0] += mx.mean(); tot["real linear"][1] += mn.mean()
        uni = -np.ones_like(rows)
        for i, w in enumerate(SM.width(rows)):
            for t in range(w // 6):
                uni[i, t * 6:t * 6 + 6] = rng.choice(G, 6, replace=False)
        mx, mn = SP.step_costs(uni, SM.linear_owner(G), K, cost)
        tot["uniform routing"][0] += mx.mean(); tot["uniform routing"][1] += mn.mean()
    print(f"{os.path.basename(p)} ({len(ring[0]['rows'])} steps, {SM.width(ring[0]['rows']).mean():.1f} slots): " + "; ".join(
        f"{k}: max {v[0]/1000:.2f} ms, mean {v[1]/1000:.2f} ms, skew {(v[0]-v[1])/1000:.2f} ms" for k, v in tot.items()))
