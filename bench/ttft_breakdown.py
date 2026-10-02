#!/usr/bin/env python3
"""Where does a short prompt's time to first token go? For each of six prompt lengths: N sequential requests with
max_tokens=1 (so the whole request is the first step), the client's median, and the server's own histograms over
the same requests (sum/count deltas of /metrics): time in queue, prefill, inference, and engine iterations per
request. Then the shortest prompt at 1..17 output tokens, which gives the cost of the steps after the first.

A first step that costs several decode steps for a 45-token prompt is fixed overhead, not compute: profile it
(serve.sh PROF=1 PROFSTACK=true, POST /start_profile, a few such requests, /stop_profile). That is how the GDN
prefill core's ~60 ms on the 27B was found (PROGRESS.md, 2026-10-02).

usage: ttft_breakdown.py [URL] [LABEL]        (default http://localhost:8080; thinking off, temperature 0)
env ITEMS=0,2,6,... picks the prompt lengths (about 45 + 20 tokens per item)."""
import json, os, sys, time, urllib.request
B = next((a for a in sys.argv[1:] if a.startswith("http")), "http://localhost:8080")
lab = next((a for a in sys.argv[1:] if not a.startswith("http")), "diag")
N = 15
ITEMS = [int(v) for v in os.environ.get("ITEMS", "0,2,6,14,30,60").split(",")]   # ~45 + 20 tokens per item
H = ("time_to_first_token_seconds", "request_queue_time_seconds", "request_prefill_time_seconds",
     "request_inference_time_seconds", "request_decode_time_seconds", "e2e_request_latency_seconds")
def met():
    m = {}
    for l in urllib.request.urlopen(B + "/metrics").read().decode().splitlines():
        if not l.startswith("vllm:"): continue
        name = l.split("{")[0].split(" ")[0][5:]
        for h in H:
            if name in (h + "_sum", h + "_count"): m[name] = m.get(name, 0.0) + float(l.split()[-1])
        if name in ("iteration_tokens_total_count", "prompt_tokens_total", "generation_tokens_total",
                    "spec_decode_num_drafts_total"):
            m[name] = m.get(name, 0.0) + float(l.split()[-1])
    return m
def ask(items, nonce, max_tokens=1):
    p = f"[{nonce}] Summarize in one sentence: " + " ".join(
        f"Item {i}: the quick brown fox jumps over the lazy dog near river bank {i}." for i in range(items))
    b = {"model": "Qwen3.8", "messages": [{"role": "user", "content": p}], "max_tokens": max_tokens, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    u = json.loads(urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", json.dumps(b).encode(),
        {"Content-Type": "application/json"}), timeout=600).read())["usage"]
    return (time.time() - t) * 1e3, u["prompt_tokens"]
for it in (0, 6, 30): ask(it, "warm")
print(f"[{lab}] {'tokens':>6s} {'client':>7s} {'ttft':>6s} {'queue':>6s} {'prefill':>8s} {'infer':>6s} {'decode':>7s} {'e2e':>6s} {'iters/req':>9s} {'drafts/req':>10s}   (ms, means over {N})")
for items in ITEMS:
    a = met(); ws = []; toks = 0
    for r in range(N):
        w, toks = ask(items, f"{items}-{r}-{time.time_ns()}"); ws.append(w)
        time.sleep(0.05)
    time.sleep(0.3); b = met()
    d = {k: b.get(k, 0) - a.get(k, 0) for k in b}
    mean = lambda h: 1e3 * d.get(h + "_sum", 0) / max(d.get(h + "_count", 0), 1)
    ws.sort()
    print(f"[{lab}] {toks:6d} {ws[len(ws) // 2]:7.0f} {mean(H[0]):6.0f} {mean(H[1]):6.1f} {mean(H[2]):8.1f} {mean(H[3]):6.0f} "
          f"{mean(H[4]):7.1f} {mean(H[5]):6.0f} {d.get('iteration_tokens_total_count', 0) / N:9.2f} "
          f"{d.get('spec_decode_num_drafts_total', 0) / N:10.2f}")

print(f"[{lab}] shortest prompt, by max_tokens:  " + "  ".join(
    (lambda a, ws, b: f"{mt}:{sorted(ws)[len(ws) // 2]:.0f}ms/{(b.get('iteration_tokens_total_count', 0) - a.get('iteration_tokens_total_count', 0)) / N:.1f}it")(
        met(), [ask(0, f"mt{mt}-{r}-{time.time_ns()}", mt)[0] for r in range(N)], (time.sleep(0.3), met())[1])
    for mt in (1, 2, 3, 5, 9, 17)))
