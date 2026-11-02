#!/usr/bin/env python3
"""Timing probe for a (possibly dummy-weight) server: decode tok/s single stream, N-stream aggregate, prefill TTFT.
usage: tprobe.py URL MODEL [streams] [max_tokens]"""
import json, sys, time, threading, random, urllib.request
URL, MODEL = sys.argv[1].rstrip("/") + "/v1/chat/completions", sys.argv[2]
NS, MT = int(sys.argv[3]) if len(sys.argv) > 3 else 4, int(sys.argv[4]) if len(sys.argv) > 4 else 128
def req(prompt, max_tokens, out, i=0):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "ignore_eos": True, "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time(); first = None; n = 0
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900) as r:
        for line in r:
            if not line.startswith(b"data:") or b"[DONE]" in line: continue
            d = json.loads(line[5:])
            if d.get("usage") and d["usage"].get("completion_tokens"):
                n = d["usage"]["completion_tokens"]            # tokens, not chunks: a spec-decode step emits several a chunk
            if not d.get("choices"): continue
            dl = d["choices"][0].get("delta", {}) if d["choices"] else {}
            if (dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning")) and first is None:
                first = time.time()
    out[i] = (first - t0 if first else None, time.time() - (first or t0), n)
o = {}; req("Write a long story about a lighthouse keeper.", MT, o)
ttft, dt, n = o[0]; print(f"single stream: {n} tokens in {dt:.2f}s = {n/dt:.1f} tok/s (ttft {ttft:.2f}s)")
o = {}; th = [threading.Thread(target=req, args=(f"Story number {i} about the sea.", MT, o, i)) for i in range(NS)]
t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - t0
tot = sum(v[2] for v in o.values()); print(f"{NS} streams: {tot} tokens in {wall:.2f}s = {tot/wall:.1f} tok/s aggregate, {tot/NS/max(v[1] for v in o.values()):.1f} tok/s per stream")
random.seed(1); words = " ".join(random.choice(["alpha","river","stone","cloud","north","delta"]) + str(random.randint(0, 999)) for _ in range(1500))
for rep in range(2):
    o = {}; t0 = time.time(); req(words + f" Summarize {rep}.", 1, o); ttft, dt, n = o[0]; print(f"prefill pass {rep}: ttft {ttft:.2f}s")
body = {"model": MODEL, "messages": [{"role": "user", "content": words}], "max_tokens": 1}
r = json.loads(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900).read())
pt = r["usage"]["prompt_tokens"]; print(f"prefill: {pt} tokens, ttft {ttft:.2f}s = {pt/ttft:.0f} tok/s")
