#!/usr/bin/env python3
"""spectune.py URL MODEL LABEL [reps] [prose,code,gsm]: draft-length table row set for one server configuration.
Workloads (greedy, natural stop): prose = ask3's lighthouse essay (thinking off, 700 tokens), code = ask3's LRU module
(thinking off, 700), gsm = GSM8K test questions 0-15 as gsm8k_ds41.py asks them (default template = thinking on, 2048;
1 stream: questions 0-3 back to back, 4 streams: 16 questions dealt round-robin); each at 1 stream (the original prompt) and 4 streams
(the original + three siblings, so the streams do not decode in lockstep). Decode tok/s is usage-based: per stream = completion tokens over
first-token-to-end summed over its requests (a spec step emits several tokens a chunk); aggregate = all tokens over the
wall time (includes prefills). Acceptance comes from the server's /metrics spec-decode
counters, differenced around each run: mean acceptance length = 1 + accepted / drafts, per-position acceptance rates.
Prints one line per (workload, streams, rep) and a JSON line (prefix RESULT) for the notes table."""
import os, json, re, sys, threading, time, urllib.request

B = sys.argv[1].rstrip("/"); MODEL = sys.argv[2]; LABEL = sys.argv[3]; REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 2
U = B + "/v1/chat/completions"
W = {
    "prose": (False, 700, ["Write a 500-word essay about the history of lighthouses. Use varied vocabulary and no repetition.",
                           "Write a 500-word essay about the history of windmills. Use varied vocabulary and no repetition.",
                           "Write a 500-word essay about the history of canals. Use varied vocabulary and no repetition.",
                           "Write a 500-word essay about the history of paper. Use varied vocabulary and no repetition."]),
    "code": (False, 700, ["Write a complete Python module implementing an LRU cache class with get/put, type hints and a docstring, plus three unit tests.",
                          "Write a complete Python module implementing an LFU cache class with get/put, type hints and a docstring, plus three unit tests.",
                          "Write a complete Python module implementing a fixed-size ring buffer class with push/pop, type hints and a docstring, plus three unit tests.",
                          "Write a complete Python module implementing a trie class with insert/search/starts_with, type hints and a docstring, plus three unit tests."]),
    "gsm": (None, 2048, None),   # GSM8K test questions 0-15, gsm8k_ds41.py's prompt and default (thinking) template
}


def gsm_prompts():
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    rows = urllib.request.urlopen(url, timeout=60).read().decode().splitlines()[:16]
    return [json.loads(r)["question"] + "\nGive the final numeric answer." for r in rows]


W["gsm"] = (None, 2048, gsm_prompts())


def metrics():
    txt = urllib.request.urlopen(B + "/metrics", timeout=30).read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith("#") or "_created" in line:
            continue
        m = re.match(r'(vllm:spec_decode_[a-z_]+?)(?:_total)?(\{[^}]*\})?\s+([0-9.e+-]+)$', line)
        if not m:
            continue
        name, lab, v = m.group(1), m.group(2) or "", float(m.group(3))
        pos = re.search(r'position="(\d+)"', lab)
        key = name + (f"[{pos.group(1)}]" if pos else "")
        out[key] = out.get(key, 0.0) + v
    return out


def req(prompt, think, n, res, i):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}}
    if think is not None:
        body["chat_template_kwargs"] = {"thinking": think}
    t0 = time.time(); first = None; ntok = 0
    with urllib.request.urlopen(urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data:") or b"[DONE]" in line:
                continue
            d = json.loads(line[5:])
            if d.get("usage") and d["usage"].get("completion_tokens"):
                ntok = d["usage"]["completion_tokens"]
            if d.get("choices") and first is None:
                dl = d["choices"][0].get("delta") or {}
                if dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning"):
                    first = time.time()
    res[i] = (first or t0, time.time(), ntok)


def stream(prompts, think, n, res, i):
    """one client: its prompts back to back; (decode seconds summed, tokens)"""
    acc = [0.0, 0]
    for j, p in enumerate(prompts):
        r = {}; req(p, think, n, r, 0); acc[0] += r[0][1] - r[0][0]; acc[1] += r[0][2]
    res[i] = acc


def run(wl, ns):
    think, n, prompts = W[wl]
    m0 = metrics(); res = {}
    # prose / code: one prompt a stream; gsm: 16 questions dealt round-robin over the streams (1 stream: questions 0-3)
    mine = [prompts[i::ns] if wl == "gsm" else [prompts[i % len(prompts)]] for i in range(ns)]   # 8 streams reuse prompts
    if wl == "gsm" and ns == 1:
        mine = [prompts[:4]]
    th = [threading.Thread(target=stream, args=(mine[i], think, n, res, i)) for i in range(ns)]
    t0 = time.time()
    [t.start() for t in th]; [t.join() for t in th]
    m1 = metrics()
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    wall = time.time() - t0; tot = sum(v[1] for v in res.values())
    per = [v[1] / v[0] for v in res.values() if v[0] > 0]
    drafts = d.get("vllm:spec_decode_num_drafts", 0); acc = d.get("vllm:spec_decode_num_accepted_tokens", 0)
    dtoks = d.get("vllm:spec_decode_num_draft_tokens", 0)
    pp = [round(d[k] / drafts, 2) for k in sorted((k for k in d if k.startswith("vllm:spec_decode_num_accepted_tokens_per_pos")),
                                                   key=lambda k: int(k.split("[")[1][:-1]))] if drafts else []
    return {"label": LABEL, "wl": wl, "streams": ns, "tokens": tot, "agg_tok_s": round(tot / wall, 1),
            "per_stream_tok_s": round(sum(per) / len(per), 1), "mal": round(1 + acc / drafts, 2) if drafts else None,
            "draft_acc_rate": round(acc / dtoks, 3) if dtoks else None, "per_pos": pp}


for wl in (sys.argv[5].split(",") if len(sys.argv) > 5 else ("prose", "code", "gsm")):
    for ns in [int(v) for v in os.environ.get("STREAMS", "1,4").split(",")]:   # STREAMS=1,4,8 for the NSEQ-8 recipe
        for rep in range(REPS):
            r = run(wl, ns); r["rep"] = rep
            print(f"{LABEL:14s} {wl:5s} x{ns}  {r['tokens']:5d} tok  agg {r['agg_tok_s']:6.1f} tok/s  per-stream {r['per_stream_tok_s']:6.1f}"
                  f"  MAL {r['mal']}  acc {r['draft_acc_rate']}  per-pos {r['per_pos']}", flush=True)
            print("RESULT " + json.dumps(r), flush=True)
