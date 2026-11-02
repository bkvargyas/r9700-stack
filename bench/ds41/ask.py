#!/usr/bin/env python3
"""ask.py URL MODEL [max_tokens] [prompt...]: correctness prompts; prints reasoning (truncated) and content."""
import json, sys, time, urllib.request
U = sys.argv[1].rstrip("/") + "/v1/chat/completions"
MODEL = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
DEFAULT = [
    ("What is the capital of France? Answer with one word.", 64),
    ("Compute 17 * 23 and explain briefly.", 300),
    ("Write a Python function that returns the n-th Fibonacci number iteratively.", 400),
    ("A train leaves at 3pm at 80 km/h, another at 4pm at 100 km/h on the same track. When does the second catch up? Be brief.", 500),
]
def ask(prompt, n):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0.0}
    t = time.time()
    req = urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    dt = time.time() - t
    m = r["choices"][0]["message"]
    u = r["usage"]
    reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
    ct = u["completion_tokens"]
    print(f"--- [{dt:.1f}s, {ct} tok, {ct / dt:.1f} tok/s, finish={r['choices'][0].get('finish_reason')}] {prompt[:70]!r}")
    print(f"    reasoning: {reasoning[:400]!r}")
    print(f"    content:   {str(m.get('content'))[:600]!r}", flush=True)
prompts = [(" ".join(sys.argv[4:]), N)] if len(sys.argv) > 4 else DEFAULT
for p, n in prompts:
    ask(p, n if len(sys.argv) <= 4 else N)
