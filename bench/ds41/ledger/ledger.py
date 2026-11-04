#!/usr/bin/env python3
"""ledger.py RANK0_TRACE.json [--ranks DIR] [--json OUT] [--list OUT.txt] [--tokens N]

Step ledger from a vLLM torch-profiler trace (one worker rank; kineto/roctracer GPU activity), package Z
(notes/exl3-ledger.md). Needs the launch's VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 annotations:
  steps    = the worker's "execute_context_C(c)_generation_G(g)" annotations (one a scheduler step; a step's window runs
             to the next one, so it holds the forward and the following sample_tokens: sampler, rejection, draft);
  phases   = stock's "gpu_model_runner: preprocess / forward / postprocess / sample / draft / bookkeep / ..." scopes;
  kernels  -> their launching runtime call (correlation id; a hipGraphLaunch owns every node of its graph) -> the step and
             the innermost phase scope around that call; eager traces with stacks / shapes also give the innermost
             cpu_op (+ Input Dims) and Python frame of each launch.
Kernels are binned by name into the ledger categories (CATS below). Only pure decode steps (context 0) count.
--ranks DIR: every rank's trace in DIR; collectives are matched call by call across ranks (same step, same ordinal) and the
minimum over the eight ranks is reported next to rank 0's duration (the rest is waiting for the slowest rank).
--list: the launch list of one median step (every kernel in order: phase, category, name, grid, block, dur, cpu op, shapes)."""
import argparse, bisect, collections, glob, json, os, re, statistics, sys

CATS = [  # (category, regex on the kernel name); first match wins
    ("collective", r"r9k_ar|r9k_ag|ncclDevKernel|ar8|allreduce|all_gather|cross_device"),
    # v0.4.1 package B (kernels/r9k_dsv4_glue.hip): the FFN front (ffn_norm + router gate GEMV + top-k + shared gate_up +
    # silu, <= 8 rows), the attention-input norm / pre-pass, the DSpark markov head
    ("B FFN front r9k_dsv4_sk (norm + router + top-k + shared w13 + silu)", r"r9k_dsv4_sk"),
    ("B attention input / norm pre-pass r9k_dsv4_norm", r"r9k_dsv4_norm"),
    ("B markov head r9k_dsv4_markov", r"r9k_dsv4_markov"),
    ("expert tier (r9k)", r"r9k_moe|k_gu<|k_gu$|k_dn<|k_dn$|k_groups|k_dq_tiles|moe_exl3"),
    ("expert tier (exllamav3 fused / gather)", r"exl3_moe"),
    # v0.4.1 package D (kernels/r9k_exl3_gemv.hip, anonymous namespace): k_gemv<bits, cb, M> (M = 1..8 rows) and
    # k_gemv_mma<bits, cb, RT, NB2> (9..48 rows); exllamav3's own path (exl3_gemv_mr_*, reconstruct, had_*) stays its own row
    ("EXL3 GEMV r9k_exl3_gemv", r"^k_gemv"),
    ("EXL3 GEMV", r"exl3_gemv|exl3_gemm|exl3_mgemm|had_r_128|had_hf_r_128|had_r|reconstruct_kernel|exl3_"),
    ("hipBLAS/hipBLASLt bf16 GEMM", r"^Cijk_|wvSplitK|hipblaslt|gemm|Gemm|LLMM|wvSpltK"),
    ("hc seams (r9k_mhc / stock mhc)", r"r9k_mhc|mhc_|hc_prenorm|_hc_head|hc_split|_post_update|sinkhorn|r9k_hc"),
    ("router top-k", r"topkGating|grouped_topk|moe_align|topk_softmax"),
    ("engram", r"engram|_hash_ids"),
    ("attention: decode (split-K partial + reduce)", r"_sparse_attn_decode|sparse_attn|flash_mla|mla_decode"),
    ("attention: indexer", r"mqa_logits|topKPerRow|indexer|_compute_topk_lens|_pack_global_topk|_count_valid_row|_compact_dense|candidate|_block_scores|_mask_candidates|_store_candidates|_dequantize_and_gather_k"),
    ("attention: KV write / RoPE / compressor", r"QNormRope|rope_quant|_inverse_rope|compress|_swa_|swa_indices|_ring_slot|_apply_write|_zero_kv"),
    ("norms / casts / elementwise glue", r"rms_norm|copy_kernel|elementwise|FillFunctor|act_and_mul|silu|CUDAFunctor|index_elementwise|scatter_gather|reduce_kernel|vectorized_gather|copyBuffer|fillBuffer|triton_poi|triton_per|triton_red|cat_|direct_copy|_to_copy|kernel\.kd$|rocprim|radixSort|bitonic|gatherTopK|sbtopk|warptopk"),
]
DRAFT_NAMES = set()
PHASES = ("preprocess", "forward", "postprocess", "sample", "draft", "bookkeep", "eplb", "ModelRunnerOutput")


def cat_of(name, phase):
    c = "everything else"
    nm = name.replace("(anonymous namespace)::", "").replace("void ", "")
    for cn, rx in CATS:
        if re.search(rx, nm):
            c = cn; break
    if phase == "sample" and c not in ("collective", "hipBLAS/hipBLASLt bf16 GEMM"):
        return "sampler / rejection sampler"
    return c


def short(n):
    n = n.replace("(anonymous namespace)::", "")
    n = re.sub(r"\(.*", "", n).replace("void ", "").replace(".kd", "")
    n = re.sub(r"\[clone.*", "", n)
    if n.startswith("Cijk_"):
        m = re.search(r"MT\d+x\d+x\d+", n); n = "Cijk_" + n[5:20] + ".." + (m.group(0) if m else "")
    return n[:110]


def load(path):
    return json.load(open(path))["traceEvents"]


def analyze(ev):
    ann = [e for e in ev if e.get("cat") == "user_annotation" and e.get("ph") == "X"]
    steps = sorted([e for e in ann if e["name"].startswith("execute_")], key=lambda e: e["ts"])
    tid = steps[0]["tid"] if steps else None
    phases = sorted([e for e in ann if e["name"].startswith("gpu_model_runner: ") and e["tid"] == tid], key=lambda e: e["ts"])
    rt = {}
    for e in ev:
        if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {}):
            rt[e["args"]["correlation"]] = e
    cpuops = sorted([e for e in ev if e.get("cat") == "cpu_op" and e.get("tid") == tid and "dur" in e], key=lambda e: e["ts"])
    pyf = sorted([e for e in ev if e.get("cat") == "python_function" and e.get("tid") == tid and "dur" in e], key=lambda e: e["ts"])
    kern = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e], key=lambda e: e["ts"])
    sts = [s["ts"] for s in steps]

    def innermost(lst, starts, t):
        # innermost interval containing t (lst sorted by ts; nested intervals)
        i = bisect.bisect_right(starts, t) - 1; best = None
        while i >= 0 and i > bisect.bisect_right(starts, t) - 400:
            e = lst[i]
            if e["ts"] <= t <= e["ts"] + e["dur"]:
                if best is None or e["dur"] < best["dur"]:
                    best = e
                    if lst is not phases:
                        return best
            i -= 1
        return best

    # Phases. This pin runs the DSpark drafter on model runner v2 (vllm/v1/worker/gpu), which has no profiler scopes: the
    # worker's execute_* annotation covers the target forward (execute_model); sample_tokens (sampler / rejection sampler /
    # the drafter's propose) runs after it, unannotated. So: forward = launched inside an execute_* annotation; draft =
    # launched under a Python frame of the speculator's propose (eager traces with stacks), or a hipGraphLaunch outside the
    # forward (graph traces: the drafter's captured graphs), or a kernel name the eager trace saw only under propose
    # (DRAFT_NAMES); everything else after the forward = "sample" (sampler, rejection sampler, spec bookkeeping).
    ex = [(s["ts"], s["ts"] + s["dur"]) for s in steps]
    exs = [a for a, b in ex]
    dr = sorted([(f["ts"], f["ts"] + f["dur"]) for f in pyf if re.search(r"\): propose$|speculator\.py\(\d+\): propose|dspark/.*: (propose|run_model|_run)", f["name"])])
    drs = [a for a, b in dr]
    def inside(iv, ivs, t):
        i = bisect.bisect_right(ivs, t) - 1
        while i >= 0 and i > bisect.bisect_right(ivs, t) - 50:
            if iv[i][0] <= t <= iv[i][1]:
                return True
            i -= 1
        return False
    cst = [c["ts"] for c in cpuops]; fst = [f["ts"] for f in pyf]
    out = []
    for k in kern:
        a = k.get("args", {}); r = rt.get(a.get("correlation"))
        t = r["ts"] if r else k["ts"]
        si = bisect.bisect_right(sts, t) - 1
        if si < 0 or si >= len(steps):
            continue
        sn = short(k["name"])
        g = bool(r and "GraphLaunch" in r["name"])
        if inside(ex, exs, t):
            phase = "forward"
        elif (dr and inside(dr, drs, t)) or (not dr and (g or sn in DRAFT_NAMES)):
            phase = "draft"
        else:
            phase = "sample"
        op = innermost(cpuops, cst, t) if (r and cpuops) else None
        fn = innermost(pyf, fst, t) if (r and pyf) else None
        out.append({"step": si, "name": k["name"], "short": sn, "ts": k["ts"], "dur": k["dur"], "phase": phase,
                    "graph": g, "launch": r["name"] if r else "?",
                    "grid": a.get("grid"), "block": a.get("block"), "bytes": a.get("bytes"),
                    "op": op["name"] if op else "", "dims": (op.get("args", {}).get("Input Dims") if op else None),
                    "py": fn["name"] if fn else "", "cat": cat_of(k["name"], phase)})
    # EXL3 dequant path (rows > R9K_EXL3_RECON_M): reconstruct / had_r_128 -> hipBLAS f16 GEMM -> had_r_128. A BLAS kernel whose
    # nearest preceding non-glue kernel (same step, same phase) is an EXL3 reconstruct / Hadamard belongs to the EXL3 row.
    last = {}
    for k in out:
        key = (k["step"], k["phase"])
        if k["cat"] == "hipBLAS/hipBLASLt bf16 GEMM" and re.search(r"reconstruct|had_", last.get(key, "")):
            k["cat"] = "EXL3 dequant path: hipBLAS f16 GEMM"
        if not k["cat"].startswith("norms"):
            last[key] = k["name"]
    return steps, phases, out, rt


def step_meta(s):
    m = re.search(r"context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)", s["name"])
    return tuple(int(x) for x in m.groups()) if m else (None,) * 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace"); ap.add_argument("--ranks"); ap.add_argument("--json"); ap.add_argument("--list")
    ap.add_argument("--tokens", type=int, default=0, help="only steps with this many generation tokens (0 = the most common)")
    ap.add_argument("--label", default="")
    ap.add_argument("--draft-names", help="json list of kernel short names launched only under the drafter (from an eager trace)")
    ap.add_argument("--dump-draft-names", help="write the kernel short names seen under the drafter's propose and never elsewhere")
    A = ap.parse_args()
    if A.draft_names:
        DRAFT_NAMES.update(json.load(open(A.draft_names)))
    ev = load(A.trace)
    steps, phases, ks, rt = analyze(ev)
    meta = [step_meta(s) for s in steps]
    if A.dump_draft_names:
        dn = {k["short"] for k in ks if k["phase"] == "draft"} - {k["short"] for k in ks if k["phase"] != "draft"}
        json.dump(sorted(dn), open(A.dump_draft_names, "w"), indent=0)
    dec = [i for i, m in enumerate(meta) if m[0] == 0 and m[2]]
    cnt = collections.Counter(meta[i][3] for i in dec)
    T = A.tokens or cnt.most_common(1)[0][0]
    sel = [i for i in dec if meta[i][3] == T and i + 1 < len(steps)]   # need the next step's start for the wall time
    print(f"{A.label} trace {os.path.basename(A.trace)}: {len(steps)} steps; decode step token counts {dict(cnt)}; using T={T}: {len(sel)} steps")
    bystep = collections.defaultdict(list)
    for k in ks:
        bystep[k["step"]].append(k)
    # per-step wall (CPU: start of this step's annotation to the next one's), GPU busy (union), launches
    walls, busy, spans, nlaunch, ngl = [], [], [], [], []
    catsum = collections.defaultdict(list); catn = collections.defaultdict(list)
    phwall = collections.defaultdict(list)
    for i in sel:
        kk = sorted(bystep[i], key=lambda k: k["ts"])
        w = steps[i + 1]["ts"] - steps[i]["ts"]; walls.append(w)
        u = 0; end = -1e30; st = None
        for k in kk:
            s0, s1 = k["ts"], k["ts"] + k["dur"]
            if s0 > end:
                u += s1 - s0; end = s1
            elif s1 > end:
                u += s1 - end; end = s1
        busy.append(u); nlaunch.append(len(kk))
        ngl.append(sum(1 for c, r in rt.items() if r["name"].endswith("GraphLaunch") and steps[i]["ts"] <= r["ts"] < steps[i + 1]["ts"] and r["tid"] == steps[i]["tid"]))
        agg = collections.defaultdict(float); n = collections.Counter()
        for k in kk:
            key = ("drafter: " if k["phase"] == "draft" else "") + k["cat"]
            agg[key] += k["dur"]; n[key] += 1
        for key in set(list(agg) + list(catsum)):
            catsum[key].append(agg.get(key, 0.0)); catn[key].append(n.get(key, 0))
        for p in phases:
            if steps[i]["ts"] <= p["ts"] < steps[i + 1]["ts"]:
                phwall[p["name"].split(": ", 1)[1]].append(p["dur"])
        # GPU time by phase
    med = lambda x: statistics.median(x) if x else 0
    mean = lambda x: sum(x) / len(x) if x else 0
    res = {"label": A.label, "trace": A.trace, "T": T, "steps": len(sel), "step_wall_us_median": med(walls), "step_wall_us_mean": mean(walls),
           "gpu_busy_us_mean": mean(busy), "launches_mean": mean(nlaunch), "graph_launches_mean": mean(ngl),
           "categories": {k: {"us": mean(v), "launches": mean(catn[k])} for k, v in catsum.items()},
           "phase_cpu_us_mean": {k: mean(v) * len(v) / max(1, len(sel)) for k, v in phwall.items()}}
    # GPU timeline split: period = first kernel of step i+1 - first kernel of step i; busy = union of kernel intervals;
    # idle gaps (> 1 us) attributed to the phase of the kernel that ends the gap, and to graph vs eager kernels.
    per, gb, ge, gaps = [], [], [], collections.defaultdict(list)
    for i in sel:
        kk = sorted(bystep[i], key=lambda k: k["ts"]); kn = bystep.get(i + 1)
        if not kk or not kn:
            continue
        t0 = kk[0]["ts"]; t1 = min(k["ts"] for k in kn); per.append(t1 - t0)
        end = t0; g = collections.defaultdict(float); bg = be = 0.0
        for k in kk:
            if k["ts"] > end + 1:
                g[("draft" if k["phase"] == "draft" else k["phase"]) + (" graph" if k["graph"] else " eager")] += k["ts"] - end
            a = max(k["ts"], end); b = k["ts"] + k["dur"]
            if b > a:
                if k["graph"]:
                    bg += b - a
                else:
                    be += b - a
            end = max(end, b)
        if t1 > end:
            g["tail (to the next step's first kernel)"] += t1 - end
        gb.append(bg); ge.append(be)
        for key in set(list(g) + list(gaps)):
            gaps[key].append(g.get(key, 0.0))
    res["gpu_timeline"] = {"period_us": mean(per), "busy_graph_us": mean(gb), "busy_eager_us": mean(ge),
                           "idle_before_us": {k: mean(v) for k, v in gaps.items()}}
    # GPU time by phase
    phg = collections.defaultdict(float); phn = collections.Counter()
    for i in sel:
        for k in bystep[i]:
            phg[k["phase"]] += k["dur"]; phn[k["phase"]] += 1
    res["phase_gpu_us_mean"] = {p: phg[p] / len(sel) for p in phg}
    res["phase_launches_mean"] = {p: phn[p] / len(sel) for p in phn}
    # per-kernel-name table (per step)
    nm = collections.defaultdict(lambda: [0.0, 0])
    for i in sel:
        for k in bystep[i]:
            key = ("D " if k["phase"] == "draft" else "S " if k["phase"] == "sample" else "M ") + k["short"]
            nm[key][0] += k["dur"]; nm[key][1] += 1
    res["kernels"] = sorted([[k, v[0] / len(sel), v[1] / len(sel)] for k, v in nm.items()], key=lambda x: -x[1])
    # collectives across ranks
    if A.ranks:
        others = [f for f in sorted(glob.glob(os.path.join(A.ranks, "*rank*.pt.trace.json"))) if os.path.abspath(f) != os.path.abspath(A.trace)]
        def colls(kslist, stepsl):
            d = collections.defaultdict(list)
            for k in kslist:
                if k["cat"] == "collective":
                    d[k["step"]].append(k)
            return {s: sorted(v, key=lambda k: k["ts"]) for s, v in d.items()}
        c0 = colls(ks, steps)
        # align steps by order of decode steps with the same metadata (ranks run the same schedule)
        mins = {}
        for f in others:
            s2, p2, k2, _ = analyze(load(f))
            c2 = colls(k2, s2)
            # step index alignment: rank traces start at the same /start_profile, but may differ by a step; align by name sequence
            names0 = [s["name"] for s in steps]; names2 = [s["name"] for s in s2]
            off = 0
            for o in range(-3, 4):
                ok = sum(1 for i in range(len(names0)) if 0 <= i + o < len(names2) and names0[i] == names2[i + o])
                if ok > off:
                    off, best = ok, o
            for i in sel:
                j = i + best
                if i in c0 and j in c2 and len(c0[i]) == len(c2[j]):
                    for q, (a, b) in enumerate(zip(c0[i], c2[j])):
                        mins.setdefault((i, q), [a["dur"]]).append(b["dur"])
        tot0 = collections.defaultdict(float); totmin = collections.defaultdict(float); nn = collections.Counter()
        for (i, q), v in mins.items():
            k = c0[i][q]; key = ("drafter " if k["phase"] == "draft" else "") + k["short"] + f" grid={k['grid']}"
            tot0[key] += v[0]; totmin[key] += min(v); nn[key] += 1
        res["collectives_vs_ranks"] = {k: {"rank0_us_per_step": tot0[k] / len(sel), "min_over_ranks_us_per_step": totmin[k] / len(sel),
                                           "calls_per_step": nn[k] / len(sel), "ranks": len(others) + 1} for k in tot0}
    # report
    print(f"T={T}: step wall median {res['step_wall_us_median']/1000:.2f} ms (mean {res['step_wall_us_mean']/1000:.2f}), GPU busy {res['gpu_busy_us_mean']/1000:.2f} ms, "
          f"{res['launches_mean']:.0f} kernels a step, {res['graph_launches_mean']:.1f} hipGraphLaunch a step")
    tot = sum(v["us"] for v in res["categories"].values())
    for k, v in sorted(res["categories"].items(), key=lambda kv: -kv[1]["us"]):
        print(f"  {v['us']:9.1f} us {100*v['us']/tot:5.1f}%  {v['launches']:7.1f} launches  {k}")
    tl = res["gpu_timeline"]
    print(f"  GPU timeline a step: period {tl['period_us']/1000:.2f} ms = busy in graph replays {tl['busy_graph_us']/1000:.2f} + busy eager {tl['busy_eager_us']/1000:.2f} + idle {sum(tl['idle_before_us'].values())/1000:.2f} ms;"
          " idle before: " + ", ".join(f"{k} {v/1000:.2f}" for k, v in sorted(tl['idle_before_us'].items(), key=lambda kv: -kv[1])))
    print("  phases (GPU us / launches / CPU-scope us):", {p: (round(res['phase_gpu_us_mean'][p]), round(res['phase_launches_mean'][p]), round(res['phase_cpu_us_mean'].get(p, 0))) for p in res["phase_gpu_us_mean"]})
    if "collectives_vs_ranks" in res:
        print("  collectives (rank0 us/step, min over ranks us/step, calls/step):")
        for k, v in sorted(res["collectives_vs_ranks"].items(), key=lambda kv: -kv[1]["rank0_us_per_step"]):
            print(f"    {v['rank0_us_per_step']:8.1f} {v['min_over_ranks_us_per_step']:8.1f} {v['calls_per_step']:6.1f}  {k}")
    print("  collectives by kernel (us a step, launches a step, us a call):")
    for k, us, n in res["kernels"]:
        if re.search(CATS[0][1], k):
            print(f"    {us:8.1f} {n:6.1f} {us / n if n else 0:8.1f}  {k}")
    print("  top kernels (us a step, launches a step):")
    for k, us, n in res["kernels"][:45]:
        print(f"    {us:8.1f} {n:6.1f}  {k}")
    if A.json:
        json.dump(res, open(A.json, "w"), indent=1)
    if A.list and sel:
        i = sorted(sel, key=lambda i: walls[sel.index(i)])[len(sel) // 2]
        with open(A.list, "w") as f:
            f.write(f"# launch list, {A.label} step {i} ({steps[i]['name']}), {len(bystep[i])} kernels; columns: idx phase category | kernel | grid block | dur_us | launch | cpu op | input dims | python frame\n")
            for q, k in enumerate(sorted(bystep[i], key=lambda k: k["ts"])):
                f.write(f"{q:4d} {k['phase']:10s} {k['cat'][:28]:28s} | {k['short']:70s} | {k['grid']} {k['block']} | {k['dur']:7.1f} | {k['launch']} | {k['op'][:40]} | {json.dumps(k['dims'])[:160] if k['dims'] else ''} | {k['py'][:90]}\n")
        print(f"  launch list of step {i} -> {A.list}")


if __name__ == "__main__":
    main()
