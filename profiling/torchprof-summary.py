#!/usr/bin/env python3
"""Summarize a torch-profiler chrome trace (vLLM worker rank 0): GPU kernel time by name, per decode step."""
import collections, glob, json, re, sys
path = sorted(glob.glob(sys.argv[1] + "/*rank0*.json") or glob.glob(sys.argv[1] + "/*.json"))[-1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
ev = json.load(open(path))["traceEvents"]
k = collections.Counter(); n = collections.Counter()
for e in ev:
    if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e:
        name = re.sub(r"\(.*", "", e["name"])[:90]
        k[name] += e["dur"]; n[name] += 1
tot = sum(k.values())
print(f"{path}\nGPU busy {tot/1e3:.1f} ms total, {tot/1e3/steps:.1f} ms/step over {steps} steps")
for name, us in k.most_common(40):
    print(f"{100*us/tot:5.1f}% {us/1e3/steps:8.2f} ms/step {n[name]//max(1,steps):5d}/step  {name}")
