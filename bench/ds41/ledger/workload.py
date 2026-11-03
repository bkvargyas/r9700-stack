#!/usr/bin/env python3
"""workload.py URL MODEL STREAMS WINDOW_S OUT.json [profile=1] [prompt_set=prose]

Fixed decode workload for the step ledger (notes/exl3-ledger.md). Starts STREAMS greedy streaming requests (prose, thinking
off, ignore_eos, 4000 tokens, distinct prompts so the streams do not decode in lockstep), waits until every stream is past
its prefill (>= 48 completion tokens), then
  A. timing window (no profiler): WINDOW_S seconds; per-stream completion tokens from continuous usage stats, and the
     server's /metrics spec-decode counters (drafts = verify steps x streams, accepted tokens) differenced over the window;
  B. profile window (profile=1): POST /start_profile, WINDOW_S seconds, POST /stop_profile (same counters around it);
then aborts the requests (closes the streams). Writes OUT.json: per window wall s, tokens a stream, tok/s, drafts,
accepted, MAL, verify steps (= drafts / streams), step wall ms = window / steps, and the tok/s that (accepted + drafts)
/ streams / window predicts."""
import json, re, sys, threading, time, urllib.request

B = sys.argv[1].rstrip("/"); MODEL = sys.argv[2]; NS = int(sys.argv[3]); WIN = float(sys.argv[4]); OUT = sys.argv[5]
PROF = (sys.argv[6] if len(sys.argv) > 6 else "1") == "1"
TOPICS = ["lighthouses", "windmills", "canals", "paper", "railways", "glassmaking", "clocks", "bridges"]
PROMPTS = [f"Write a 3000-word essay about the history of {t}. Use varied vocabulary and no repetition." for t in TOPICS]

samples = {i: [] for i in range(NS)}       # (t, completion_tokens)
stop = threading.Event()


def stream(i):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}], "max_tokens": 4000,
            "temperature": 0.0, "ignore_eos": True, "stream": True, "chat_template_kwargs": {"thinking": False},
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    r = urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=1800)
    for line in r:
        if stop.is_set():
            break
        if not line.startswith(b"data:") or b"[DONE]" in line:
            continue
        d = json.loads(line[5:])
        u = d.get("usage") or {}
        if u.get("completion_tokens"):
            samples[i].append((time.time(), u["completion_tokens"]))
    r.close()


def metrics():
    txt = urllib.request.urlopen(B + "/metrics", timeout=30).read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith("#") or "_created" in line:
            continue
        m = re.match(r'(vllm:spec_decode_[a-z_]+?)(?:_total)?(\{[^}]*\})?\s+([0-9.e+-]+)$', line)
        if m and "position" not in (m.group(2) or ""):
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


def toks_at(i, t):
    s = [c for (ts, c) in samples[i] if ts <= t]
    return s[-1] if s else 0


def window(name, pre=None, post=None):
    if pre:
        pre()
    m0 = metrics(); t0 = time.time()
    time.sleep(WIN)
    t1 = time.time(); m1 = metrics()
    tp = time.time()
    if post:
        post()
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    toks = [toks_at(i, t1) - toks_at(i, t0) for i in range(NS)]
    drafts = d.get("vllm:spec_decode_num_drafts", 0); acc = d.get("vllm:spec_decode_num_accepted_tokens", 0)
    steps = drafts / NS if NS else 0
    w = t1 - t0
    r = {"window": name, "wall_s": round(w, 3), "tokens_per_stream": toks, "tok_s_per_stream": [round(x / w, 1) for x in toks],
         "tok_s_aggregate": round(sum(toks) / w, 1), "drafts": drafts, "accepted": acc,
         "draft_tokens": d.get("vllm:spec_decode_num_draft_tokens", 0),
         "mal": round(1 + acc / drafts, 3) if drafts else None, "verify_steps": steps,
         "step_wall_ms": round(1000 * w / steps, 2) if steps else None,
         "predicted_tok_s_per_stream": round((acc + drafts) / NS / w, 1) if NS else None,
         "stop_profile_s": round(time.time() - tp, 1)}
    print(json.dumps(r), flush=True)
    return r


th = [threading.Thread(target=stream, args=(i,), daemon=True) for i in range(NS)]
for t in th:
    t.start(); time.sleep(0.3)
for _ in range(600):
    if all(samples[i] and samples[i][-1][1] >= 48 for i in range(NS)):
        break
    time.sleep(0.5)
time.sleep(1.0)
res = {"streams": NS, "window_s": WIN, "windows": [window("timing")]}
if PROF:
    post = lambda: urllib.request.urlopen(urllib.request.Request(B + "/stop_profile", b"", method="POST"), timeout=1800).read()
    pre = lambda: urllib.request.urlopen(urllib.request.Request(B + "/start_profile", b"", method="POST"), timeout=600).read()
    res["windows"].append(window("profile", pre, post))
stop.set()
json.dump(res, open(OUT, "w"), indent=1)
