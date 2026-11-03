#!/usr/bin/env python3
"""classavg.py LIST.txt: per-position mean kernel duration over all layers of each layer class (layers.py segmentation) of
one step's launch list -- the condensed per-layer launch list of notes/exl3-ledger.md takes its graph-replay durations here."""
import collections, re, statistics, sys
rows = []
for line in open(sys.argv[1]):
    if line.startswith("#"):
        continue
    p = [x.strip() for x in line.split("|")]; h = p[0].split(None, 2)
    rows.append(dict(phase=h[1], cat=h[2] if len(h) > 2 else "", name=p[1], dur=float(p[3])))
for phase in ("forward", "draft"):
    rr = [r for r in rows if r["phase"] == phase]
    seams = [j for j, r in enumerate(rr) if r["name"].startswith("r9k_mhc_seam_kernel")][0::2]
    segs = [rr[s:(seams[k + 1] if k + 1 < len(seams) else len(rr))] for k, s in enumerate(seams)]
    cls = collections.defaultdict(list)
    for li, seg in enumerate(segs):
        cls[tuple(re.sub(r"<.*", "", r["name"]) for r in seg)].append((li, seg))
    for sig, members in cls.items():
        print(f"== {phase} layers {[li for li, _ in members]}: {len(sig)} launches")
        tot = 0
        for q in range(len(sig)):
            d = [seg[q]["dur"] for _, seg in members]; tot += statistics.mean(d)
            print(f"  {q:3d} {members[0][1][q]['cat'][:28]:28s} {members[0][1][q]['name'][:60]:60s} mean {statistics.mean(d):8.1f}  min {min(d):7.1f}  max {max(d):7.1f}")
        print(f"  total mean {tot:.1f} us")
