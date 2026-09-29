#!/usr/bin/env python3
"""conc-8 aggregate by workload mix on one server: 8 copies of one prompt type vs the mixed probe set.
usage: mix.py [URL] [-v] [--only=code,mixed4]
Separates prompt diversity (distinct experts streamed per step with offloaded experts) from prompt type."""
import json, sys, time, threading, urllib.request, random, statistics as st
B = next((a for a in sys.argv[1:] if a.startswith("http")), "http://localhost:8080")
P = {
 "code": "Write a Python class implementing an LRU cache with type hints, docstrings and unit tests.",
 "prose": "Write a long, vivid short story about a lighthouse keeper during a storm.",
 "json": "Output a JSON array of 20 fictional employees with fields id, name, department, salary, skills.",
 "math": "Solve step by step: a train leaves at 3pm at 80 km/h, another at 4pm at 100 km/h. When does the second catch up? Then generalize.",
 "harness": "Explain concept number 0: describe tensor parallelism in depth.",
}
def req(p, n=256):
    b = {"model": "Qwen3.8", "messages": [{"role": "user", "content": p}], "max_tokens": n, "temperature": 0,
         "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    return json.loads(urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions",
        json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=900).read())["usage"]["completion_tokens"]
def spec():
    d = a = 0.0
    for l in urllib.request.urlopen(B + "/metrics").read().decode().splitlines():
        if l.startswith("vllm:spec_decode_num_drafts_total"): d += float(l.split()[-1])
        if l.startswith("vllm:spec_decode_num_accepted_tokens_total"): a += float(l.split()[-1])
    return d, a
def agg(prompts):
    res = []; th = [threading.Thread(target=lambda p=p: res.append(req(p))) for p in prompts]
    d0, a0 = spec(); t = time.time(); [x.start() for x in th]; [x.join() for x in th]; w = time.time() - t
    d1, a1 = spec()
    return sum(res) / w, 1 + (a1 - a0) / max(d1 - d0, 1), w / max(d1 - d0, 1) * 1e3 * len(prompts)
def nonce(): return f"[req {random.getrandbits(48):012x}] "
req("hi", 16)
W = {k: [nonce() + v + f" (#{i})" for i in range(8)] for k, v in P.items()}
W["mixed4"] = [nonce() + list(P.values())[i % 4] + f" (#{i})" for i in range(8)]
W["single-code"] = [nonce() + P["code"]]; W["single-prose"] = [nonce() + P["prose"]]
only = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--only=")), None)
if only:                        # e.g. --only=code,mixed4 for a fast A/B
    W = {k: v for k, v in W.items() if k in only}
order = list(W) * 3
for k in order[:len(W)]: agg(W[k][:1])          # warm each type once
R = {}
for k in order:
    t0 = time.strftime("%H:%M:%S")
    R.setdefault(k, []).append(agg([nonce() + p.split("] ", 1)[1] for p in W[k]]))
    if "-v" in sys.argv:        # wall-clock window of each run, to line up with the server's expert cache stats
        print(f"  run {k:13s} {t0} .. {time.strftime('%H:%M:%S')}  {R[k][-1][0]:6.1f} tok/s", flush=True)
for k in W:
    xs = R[k]; t = [x[0] for x in xs]
    print(f"{k:13s} n={len(W[k])}  agg {st.median(t):6.1f} tok/s [{min(t):.1f}..{max(t):.1f}]  tok/step {st.median([x[1] for x in xs]):.2f}  ms/step {st.median([x[2] for x in xs]):.1f}")
