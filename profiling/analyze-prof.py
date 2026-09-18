#!/usr/bin/env python3
"""Summarize rocprofv3 kernel/memcpy traces inside the windows marked by profile-decode.sh.

usage: analyze-prof.py PROFDIR      (expects PROFDIR/windows.json and PROFDIR/rp/**/*_kernel_trace.csv)
Per window and per GPU agent: GPU-busy time (kernel intervals clipped to the window), top kernels, and time by
kernel family. Also H2D/D2H memcpy bytes. Timestamps: picks whichever clock (mono/boot) the trace overlaps.
"""
import csv, glob, json, os, re, sys
from collections import defaultdict

FAMILIES = [
    ("expert_moe", r"moe_lru|gemm_moe|moe_route|fused_moe|moe_align|topk_softmax|grouped_topk|expert"),
    ("qsa_sparse_attn", r"attn_sparse|qsa|indexer|sparse_score|topk_expand"),
    ("attention", r"attn|flash|paged|unified_attention|reshape_and_cache|kv_cache"),
    ("gdn_linear_attn", r"gdn|gated_delta|chunk_scan|kkt|recurrent|conv1d|causal_conv|fla_|chunk_"),
    ("ple", r"ple"),
    ("allreduce_comm", r"ar_oneshot|ar_twoshot|allreduce|all_reduce|rccl|nccl|ncclDev|all_gather|clav_ar|clav_ag"),
    ("dense_gemm", r"gemm|Cijk|mxfp4|fp8|scaled_mm|wmma|matmul|gemv|hipblaslt|w4a|w8a"),
    ("sampling_spec", r"sampl|argmax|softmax|topk|top_p|rejection|spec|mtp|penalt"),
    ("norm_elementwise", r"triton_|rms|norm|silu|gelu|rotary|rope|elementwise|vectorized|copy|fill|cat|index|scatter|gather|reduce|quant"),
]
FAM_RE = [(n, re.compile(p, re.I)) for n, p in FAMILIES]
def family(name):
    for n, r in FAM_RE:
        if r.search(name): return n
    return "other"

def short(name, n=90):
    name = re.sub(r"\(.*$", "", name)          # drop demangled arg lists
    return name if len(name) <= n else name[:n] + "…"

def main(d):
    W = json.load(open(os.path.join(d, "windows.json")))
    kfiles = glob.glob(os.path.join(d, "rp", "**", "*kernel_trace.csv"), recursive=True)
    mfiles = glob.glob(os.path.join(d, "rp", "**", "*memory_copy_trace.csv"), recursive=True)
    print(f"kernel trace files: {len(kfiles)}  memcpy files: {len(mfiles)}")
    # choose clock
    tmin, tmax = None, None
    for f in kfiles:
        with open(f) as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                s = int(row["Start_Timestamp"]); tmin = s if tmin is None else min(tmin, s); tmax = s if tmax is None else max(tmax, s)
    print(f"trace span ns {tmin}..{tmax}")
    clock = None
    for c in ("boot", "mono", "wall"):
        a, b = W["decode1"][0][c], W["prefill8k"][1][c]
        if tmin is not None and tmin <= b and a <= tmax: clock = c; break
    if clock is None: print("!! no clock overlaps the trace; window stamps:", {k: v[0] for k, v in W.items()}); return
    print(f"clock={clock}\n")
    wins = {k: (v[0][clock], v[1][clock], v[2]) for k, v in W.items()}
    busy = defaultdict(lambda: defaultdict(float))          # (win, agent) -> name -> ns
    cnt = defaultdict(lambda: defaultdict(int))
    for f in kfiles:
        with open(f) as fh:
            for row in csv.DictReader(fh):
                s, e = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
                for w, (a, b, _) in wins.items():
                    if e > a and s < b:
                        key = (w, row.get("Agent_Id", "?"))
                        busy[key][row["Kernel_Name"]] += min(e, b) - max(s, a); cnt[key][row["Kernel_Name"]] += 1
    mem = defaultdict(lambda: defaultdict(int))
    for f in mfiles:
        with open(f) as fh:
            for row in csv.DictReader(fh):
                s, e = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
                for w, (a, b, _) in wins.items():
                    if e > a and s < b:
                        mem[w][row.get("Direction", row.get("Operation", "?"))] += int(row.get("Bytes", 0) or 0)
    for w, (a, b, ntok) in wins.items():
        dur = (b - a) / 1e9
        print("=" * 100); print(f"WINDOW {w}: {dur:.2f}s, {ntok} tokens -> {ntok/dur:.1f} tok/s")
        for k in sorted(x for x in busy if x[0] == w):
            tot = sum(busy[k].values()); print(f"\n-- agent {k[1]}: GPU busy {tot/1e9:.2f}s = {100*tot/1e9/dur:.0f}% of window "
                                             f"({tot/1e6/max(ntok,1):.2f} ms busy per token)")
            fam = defaultdict(float)
            for n, t in busy[k].items(): fam[family(n)] += t
            print("   by family: " + ", ".join(f"{n} {100*t/tot:.1f}%" for n, t in sorted(fam.items(), key=lambda x: -x[1])))
            print(f"   {'%busy':>6} {'ms':>9} {'calls':>7}  kernel")
            for n, t in sorted(busy[k].items(), key=lambda x: -x[1])[:25]:
                print(f"   {100*t/tot:6.1f} {t/1e6:9.1f} {cnt[k][n]:7d}  [{family(n)}] {short(n)}")
        if mem[w]: print("\n   memcpy bytes: " + ", ".join(f"{d}: {v/1e9:.2f} GB" for d, v in mem[w].items()))
        print()

if __name__ == "__main__":
    main(sys.argv[1])
