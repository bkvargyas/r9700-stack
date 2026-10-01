#!/usr/bin/env python3
"""Prefill throughput by KIND of prompt, for configurations where the experts are not all in VRAM.

With offloaded experts a prefill chunk costs what it has to fetch, and that depends on how many distinct experts
the prompt routes to. A synthetic filler built from a small vocabulary (BetterBench's prefill sweep shuffles ~75
words) routes narrowly; real text does not. This measures both on the same server.

usage: prefill_kinds.py [URL] [--text FILE] [--depths 8000,24000] [--reps 3] [-v]
  kinds: bb    = BetterBench's filler (one paragraph's words, shuffled per request)
         docs  = a random slice of FILE (default: this repo's PROGRESS.md), i.e. real mixed prose, numbers, code
"""
import json, os, random, sys, time, urllib.request, statistics as st

B = next((a for a in sys.argv[1:] if a.startswith("http")), "http://localhost:8080")
def opt(name, default):
    return next((sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a == name), default)
TEXT = opt("--text", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "PROGRESS.md"))
DEPTHS = [int(x) for x in opt("--depths", "8000,24000").split(",")]
REPS = int(opt("--reps", "3"))
V = "-v" in sys.argv

PARA = ("In distributed systems the tension between consistency, availability, and partition tolerance shapes almost "
        "every design decision. A service that prioritizes strong consistency may reject writes during a network "
        "split, while an available-first design accepts them and reconciles later. Caches, replication logs, quorums, "
        "and vector clocks are the everyday tools used to navigate these trade-offs, and the right choice depends on "
        "the workload, the cost of a stale read, and how users perceive latency. ")
WORDS = PARA.split()
DOC = open(TEXT, errors="replace").read()

def body(kind, chars):
    if kind == "bb":
        return " ".join(random.choices(WORDS, k=chars // 6 + 64))[:chars]
    o = random.randrange(0, max(1, len(DOC) - chars))
    return (DOC + DOC)[o:o + chars]

def ask(kind, chars):
    """-> (prompt_tokens, ttft_s) for one request; 2 output tokens, streamed."""
    msg = f"[{random.getrandbits(48):012x}] Read the following context, then reply with the single word: ack.\n\n" \
          f"{body(kind, chars)}\n\nReply with one word: ack."
    b = {"model": "Qwen3.8", "messages": [{"role": "user", "content": msg}], "max_tokens": 2, "temperature": 0,
         "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter(); first = None; pt = 0
    with urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", json.dumps(b).encode(),
                                                       {"Content-Type": "application/json"}), timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                pt = ev["usage"]["prompt_tokens"]
            ch = ev.get("choices") or []
            if ch and first is None and ((ch[0].get("delta") or {}).get("content") or (ch[0].get("delta") or {}).get("reasoning_content")):
                first = time.perf_counter()
    return pt, (first or time.perf_counter()) - t0

cpt = {}
for k in ("bb", "docs"):                    # chars per token of each kind, from the server's own count
    pt, _ = ask(k, 8000)
    cpt[k] = 8000 / max(pt - 40, 1)
R = {}
for _ in range(REPS):
    for d in DEPTHS:
        for k in ("bb", "docs"):
            t0 = time.strftime("%H:%M:%S")
            pt, ttft = ask(k, int(d * cpt[k]))
            R.setdefault((k, d), []).append((pt, pt / ttft))
            if V:
                print(f"  run {k + '-' + str(d):13s} {t0} .. {time.strftime('%H:%M:%S')}  {pt / ttft:6.1f} tok/s", flush=True)
for (k, d), xs in R.items():
    t = [x[1] for x in xs]
    print(f"{k:5s} ~{d:6d} tok (actual {int(st.median([x[0] for x in xs])):6d}, {cpt[k]:.2f} chars/tok)  "
          f"prefill {st.median(t):6.0f} tok/s [{min(t):.0f}..{max(t):.0f}]")
