#!/usr/bin/env python3
"""gputl.py TRACE.json [label]: GPU-timeline wall split from a GPU-activity-only torch trace (launch G4: profiler activities
["CUDA"], so the CPU side runs without profiler overhead). Kernels of one hipGraphLaunch share a correlation id; a step is
one target-forward graph replay (the largest correlation groups, > 400 kernels) to the next. Per step: period, GPU busy
inside the forward replay / the drafter replay(s) / eager kernels (sampler, rejection sampler, drafter input prep), and idle
time by where it falls (before the forward replay = after the previous step's last kernel; between forward and drafter
replay; after the drafter replay)."""
import collections, json, os, re, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ledger import cat_of, short                     # the ledger's kernel categories
ev = json.load(open(sys.argv[1]))["traceEvents"]; lab = sys.argv[2] if len(sys.argv) > 2 else ""
DRAFT = set(json.load(open(sys.argv[3]))) if len(sys.argv) > 3 else set()   # E1's draft-names.json
k = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e], key=lambda e: e["ts"])
cnt = collections.Counter(e["args"].get("correlation") for e in k)
for e in k:
    n = cnt[e["args"].get("correlation")]
    e["grp"] = "fwd" if n > 400 else ("graph" if n > 20 else "eager")
# forward replay starts
starts = []; prev = None
for i, e in enumerate(k):
    c = e["args"].get("correlation")
    if e["grp"] == "fwd" and c != prev:
        starts.append(i)
    prev = c
rows = []
for a, b in zip(starts, starts[1:]):
    seg = k[a:b]; t0 = seg[0]["ts"]; t1 = k[b]["ts"]
    busy = collections.defaultdict(float); idle = collections.defaultdict(float); end = t0; n = collections.Counter()
    seen_fwd_end = False; seen_graph = False
    for e in seg:
        n[e["grp"]] += 1
        if e["ts"] > end + 1:
            where = "inside forward replay" if e["grp"] == "fwd" else ("before drafter replay" if e["grp"] == "graph" and not seen_graph else
                    "inside drafter replay" if e["grp"] == "graph" else "between eager kernels / after forward")
            idle[where] += e["ts"] - end
        if e["grp"] == "graph":
            seen_graph = True
        s0 = max(e["ts"], end); s1 = e["ts"] + e["dur"]
        if s1 > s0:
            busy[e["grp"]] += s1 - s0
        end = max(end, s1)
    idle["after last kernel, before next forward replay"] += max(0, t1 - end)
    cats = collections.defaultdict(float); cn = collections.Counter(); last = {}
    for e in seg:
        sn = short(e["name"])
        ph = "forward" if e["grp"] == "fwd" else "draft" if (e["grp"] == "graph" or sn in DRAFT) else "sample"
        c = cat_of(e["name"], ph)
        # the dequant-path rule of ledger.py (a BLAS kernel right after an EXL3 reconstruct / Hadamard)
        if c == "hipBLAS/hipBLASLt bf16 GEMM" and re.search(r"reconstruct|had_", last.get(ph, "")):
            c = "EXL3 dequant path: hipBLAS f16 GEMM"
        if not c.startswith("norms"):
            last[ph] = e["name"]
        key = ("drafter: " if ph == "draft" else "") + c
        cats[key] += e["dur"]; cn[key] += 1
    rows.append((t1 - t0, busy, idle, n, cats, cn))
m = lambda f: statistics.mean(f(r) for r in rows)
print(f"{lab}: {len(rows)} steps; period {m(lambda r: r[0])/1000:.2f} ms (median {statistics.median(r[0] for r in rows)/1000:.2f})")
for g in ("fwd", "graph", "eager"):
    print(f"  busy {g:6s} {m(lambda r: r[1].get(g, 0))/1000:7.2f} ms  launches {m(lambda r: r[3].get(g, 0)):7.1f}")
keys = sorted({kk for r in rows for kk in r[2]})
for kk in keys:
    print(f"  idle {kk:48s} {m(lambda r: r[2].get(kk, 0))/1000:7.2f} ms")
print(f"  total launches a step {m(lambda r: sum(r[3].values())):.0f}")
allk = sorted({kk for r in rows for kk in r[4]}, key=lambda kk: -m(lambda r: r[4].get(kk, 0)))
tot = m(lambda r: sum(r[4].values()))
print(f"  GPU time a step by category (sum of kernel durations {tot/1000:.2f} ms):")
for kk in allk:
    v = m(lambda r: r[4].get(kk, 0))
    print(f"    {v:9.1f} us {100*v/tot:5.1f}%  {m(lambda r: r[5].get(kk, 0)):7.1f} launches  {kk}")
nm = collections.defaultdict(float); nc = collections.Counter()
for (a, b) in zip(starts, starts[1:]):
    for e in k[a:b]:
        key = ("F " if e["grp"] == "fwd" else "D " if e["grp"] == "graph" else "E ") + short(e["name"])
        nm[key] += e["dur"]; nc[key] += 1
print("  top kernels (us a step, launches a step; F = forward replay, D = drafter replay, E = eager):")
for kk, v in sorted(nm.items(), key=lambda kv: -kv[1])[:40]:
    print(f"    {v/len(rows):8.1f} {nc[kk]/len(rows):6.1f}  {kk}")
