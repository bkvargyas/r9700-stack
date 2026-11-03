#!/usr/bin/env python3
"""layers.py LIST.txt [OUT.txt]: split a ledger.py --list launch list (one step) into decoder layers and the drafter.
The forward's layer l starts at its (2l)-th hyper-connection seam (r9k_mhc_seam_kernel: two seams a layer, the attention
pre and the FFN pre; the post update of each sublayer is folded into the next seam). Prints per layer: launches, GPU us
without / with collectives, and the layer's kernel-name signature class; then each distinct layer class in full."""
import collections, re, sys
rows = []
for line in open(sys.argv[1]):
    if line.startswith("#"):
        continue
    p = [x.strip() for x in line.split("|")]
    head = p[0].split(None, 2)
    rows.append({"i": int(head[0]), "phase": head[1], "cat": head[2] if len(head) > 2 else "", "name": p[1], "grid": p[2],
                 "dur": float(p[3]), "op": p[5] if len(p) > 5 else "", "dims": p[6] if len(p) > 6 else ""})
out = open(sys.argv[2], "w") if len(sys.argv) > 2 else sys.stdout
for phase in ("forward", "draft"):
    rr = [r for r in rows if r["phase"] == phase]
    seams = [j for j, r in enumerate(rr) if r["name"].startswith("r9k_mhc_seam_kernel")]
    starts = seams[0::2]
    segs = [("pre", rr[:starts[0]] if starts else rr)]
    for n, s in enumerate(starts):
        e = starts[n + 1] if n + 1 < len(starts) else len(rr)
        segs.append((f"L{n}", rr[s:e]))
    # the tail after the last layer (final norm / head ...) is inside the last segment; split it at the final mhc_post
    classes = collections.OrderedDict()
    print(f"== {phase}: {len(rr)} launches, {len(starts)} layers", file=out)
    for lab, seg in segs:
        sig = tuple(re.sub(r"<.*", "", r["name"]) for r in seg if r["cat"] != "collective")
        cid = classes.setdefault(sig, len(classes))
        coll = sum(r["dur"] for r in seg if r["cat"] == "collective")
        tot = sum(r["dur"] for r in seg)
        ncoll = sum(1 for r in seg if r["cat"] == "collective")
        print(f"  {lab:4s} {len(seg):4d} launches ({ncoll} collectives)  {tot - coll:8.1f} us + collectives {coll:8.1f} us  class {cid}", file=out)
    for sig, cid in classes.items():
        ex = next(seg for lab, seg in segs if tuple(re.sub(r"<.*", "", r["name"]) for r in seg if r["cat"] != "collective") == sig)
        print(f"\n-- {phase} class {cid}: {len(ex)} launches", file=out)
        for r in ex:
            print(f"   {r['cat'][:26]:26s} {r['name'][:62]:62s} {r['grid']:28s} {r['dur']:8.1f}  {r['op'][:34]:34s} {r['dims'][:70]}", file=out)
