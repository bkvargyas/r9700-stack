import json, glob, os, sys
sys.path.insert(0, os.path.expanduser("~/BetterBench"))
from betterbench.report import combined_score, concurrency_rows, prefill_rows
D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/tp4v2")
def load(n):
    p = f"{D}/{n}.json"
    return json.load(open(p)) if os.path.exists(p) else None
for n in ("tp4-full", "tp2a-full", "tp2b-full"):
    r = load(n)
    if not r: continue
    print(f"== {n}  endpoint {r['env']['endpoint']}")
    c = combined_score(r)
    if c: print(f"  decode combined {c['decode']:.1f} tok/s  step p50 {c['update_p50']:.2f} ms  ttft p50 {c['ttft_p50']:.0f} ms")
    pf = [f"{x['target_depth']//1000}k {x['pp_med']:.0f}" for x in prefill_rows(r) if not x["skipped"]]
    if pf: print("  prefill tok/s  " + "  ".join(pf))
    cr = concurrency_rows(r)
    if cr: print("  concurrency agg tok/s  " + "  ".join(f"c{x['level']} {x['aggregate_tps']:.0f} (ttft {x['ttft_p50']:.0f}ms)" for x in cr))
rows = []
for L in (1, 2, 4, 8):
    a, b = load(f"dual-a-c{L}"), load(f"dual-b-c{L}")
    if not (a and b): continue
    ra, rb = concurrency_rows(a)[0], concurrency_rows(b)[0]
    rows.append(f"total c{2*L}: {ra['aggregate_tps'] + rb['aggregate_tps']:.0f} (A {ra['aggregate_tps']:.0f} + B {rb['aggregate_tps']:.0f}, ttft {ra['ttft_p50']:.0f}/{rb['ttft_p50']:.0f}ms)")
if rows: print("== dual TP2, both loaded\n  " + "\n  ".join(rows))
