#!/usr/bin/env python3
"""vramwatch.py SECONDS [interval_s] [OUT]: sample the driver's VRAM use (sysfs mem_info_vram_used / mem_info_vram_total of
every card) for SECONDS and print, per card, min / max used and min free in MiB; with OUT, also write the time series
(one line a sample). Run on VM100 next to a workload (setsid nohup ... &) and read the summary when it ends."""
import glob, sys, time

T = float(sys.argv[1]); DT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25; OUT = sys.argv[3] if len(sys.argv) > 3 else None
cards = sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"), key=lambda f: int(f.split("/")[4][4:]))
tot = [int(open(f.replace("_used", "_total")).read()) / 2**20 for f in cards]
lo = [1e18] * len(cards); hi = [0.0] * len(cards)
fo = open(OUT, "w") if OUT else None
t0 = time.time()
while time.time() - t0 < T:
    u = [int(open(f).read()) / 2**20 for f in cards]
    for i, x in enumerate(u):
        lo[i] = min(lo[i], x); hi[i] = max(hi[i], x)
    if fo:
        fo.write(f"{time.time() - t0:.2f} " + " ".join(f"{x:.0f}" for x in u) + "\n")
    time.sleep(DT)
names = [f.split("/")[4] for f in cards]
print("card      " + " ".join(f"{n:>7s}" for n in names))
print("min used  " + " ".join(f"{x:7.0f}" for x in lo))
print("max used  " + " ".join(f"{x:7.0f}" for x in hi))
print("min free  " + " ".join(f"{t - x:7.0f}" for t, x in zip(tot, hi)))
print(f"worst card: max used {max(hi):.0f} MiB = {max(hi) / 1024:.2f} GiB, min free {min(t - x for t, x in zip(tot, hi)) / 1024:.2f} GiB"
      f" (of {tot[0] / 1024:.2f} GiB)")
