#!/usr/bin/env python3
"""memprobe.py URL {mem|reset|inv} [--json FILE]: per-rank memory of a server launched with MEMPROBE=1
(serve/ds41-exl3.sh; worker extension bench/ds41/memprobe_ext.py over the stock dev endpoint POST /collective_rpc).
  mem    torch allocated / reserved (current + peak since the last reset), device used, graph pools; sysfs VRAM per card
  inv    mem + rank 0's inventory of device tensors by component (model, drafter, runner, plugin / vLLM globals)
  try [step max]  raw hipMalloc in step-MiB chunks until failure (freed again): the real allocatable headroom
  unreached  rank 0's live device tensors (gc) outside the inventory, by shape/dtype
  reset  reset the allocator peak counters on every rank (then run a workload, then `mem`)
Run on VM100 (reads /sys/class/drm/card*/device/mem_info_vram_used for the driver's view)."""
import glob, json, sys, urllib.request

URL = sys.argv[1].rstrip("/")
CMD = sys.argv[2] if len(sys.argv) > 2 else "mem"
JS = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None


def rpc(method, *args):
    body = {"method": method, "args": list(args), "timeout": 120}
    r = urllib.request.urlopen(urllib.request.Request(URL + "/collective_rpc", json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=180)
    return json.loads(r.read())["results"]


def sysfs():
    out = []
    for f in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_used")):
        try:
            out.append((f.split("/")[4], int(open(f).read()) / 2**20))
        except OSError:
            pass
    return out


if CMD == "try":
    for r in sorted(rpc("r9k_try_alloc", *(sys.argv[3:5] or ["64", "4096"])), key=lambda r: r["rank"]):
        print(r)
    print("sysfs vram_used MiB: " + " ".join(f"{c}={m:.0f}" for c, m in sysfs())); sys.exit(0)
if CMD == "unreached":
    r = min(rpc("r9k_unreached", "40"), key=lambda r: r["rank"])
    print(f"rank0: {r['unreached_mib']:.0f} MiB in {r['count']} device tensors the inventory walk did not reach")
    for k, m, n in r["top"]:
        print(f"  {m:9.1f} MiB  x{n:<4d} {k}")
    sys.exit(0)
if CMD == "reset":
    print(rpc("r9k_reset_peak")); sys.exit(0)
res = rpc("r9k_mem", "1" if CMD == "inv" else "0")
print(f"{'rank':>4} {'alloc':>8} {'allocPk':>8} {'resv':>8} {'resvPk':>8} {'inact':>7} {'devUsed':>8} {'devFree':>8} {'graphPools':>10} retries ooms")
for r in sorted(res, key=lambda r: r["rank"]):
    gp = sum(v["reserved_mib"] for k, v in r["pools"].items() if k not in ("(0, 0)", "error"))
    print(f"{r['rank']:>4} {r['allocated_mib']:8.0f} {r['allocated_peak_mib']:8.0f} {r['reserved_mib']:8.0f} "
          f"{r['reserved_peak_mib']:8.0f} {r['inactive_split_mib']:7.0f} {r['device_used_mib']:8.0f} "
          f"{r['device_free_mib']:8.0f} {gp:10.0f} {r['num_alloc_retries']:>7} {r['num_ooms']:>4}")
print("sysfs vram_used MiB: " + " ".join(f"{c}={m:.0f}" for c, m in sysfs()))
r0 = min(res, key=lambda r: r["rank"])
print("rank0 pools: " + ", ".join(f"{k}: {v['reserved_mib']:.0f} resv / {v['allocated_mib']:.0f} alloc ({v['segments']} seg)"
                                  for k, v in r0["pools"].items()))
if "inventory" in r0:
    inv = r0["inventory"]
    print(f"rank0 inventory: {inv['total_mib']:.0f} MiB in distinct device storages (excluding pinned-host Engram)")
    for c, m in inv["categories"].items():
        print(f"  {c:40s} {m:9.1f} MiB")
    print("  largest:")
    for m, p in inv["largest"]:
        print(f"    {m:9.1f}  {p[:150]}")
if JS:
    json.dump({"ranks": res, "sysfs": sysfs()}, open(JS, "w"), indent=1)
