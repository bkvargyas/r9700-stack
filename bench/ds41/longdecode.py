#!/usr/bin/env python3
"""longdecode.py URL MODEL ctx_tokens [n_tokens]: decode tok/s after a long prefill (thinking off, ignore_eos)."""
import json, random, sys, time, urllib.request
U = sys.argv[1].rstrip("/") + "/v1/chat/completions"; MODEL = sys.argv[2]; CTX = int(sys.argv[3]); N = int(sys.argv[4]) if len(sys.argv) > 4 else 128
random.seed(CTX)
S = ["The weather was mild and the market was busy.", "Several boats were moored along the quay.", "A dog barked somewhere down the lane.",
     "The bakery sold out of rye bread by noon.", "Children played near the old fountain.", "The clock tower struck the hour twice."]
doc = " ".join(random.choice(S) for _ in range(max(CTX // 9, 10)))
body = {"model": MODEL, "messages": [{"role": "user", "content": doc + "\n\nWrite a long story that continues this scene."}],
        "max_tokens": N, "temperature": 0.0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False}}
t0 = time.time(); first = None; n = 0
with urllib.request.urlopen(urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800) as r:
    for line in r:
        if not line.startswith(b"data:") or b"[DONE]" in line: continue
        d = json.loads(line[5:])
        if d.get("usage") and d["usage"].get("completion_tokens"): n = d["usage"]["completion_tokens"]; pt = d["usage"]["prompt_tokens"]
        if d.get("choices") and first is None and (d["choices"][0].get("delta") or {}).get("content"): first = time.time()
dt = time.time() - first
print(f"ctx {pt} tokens: ttft {first - t0:.2f}s, then {n} tokens in {dt:.2f}s = {n / dt:.1f} tok/s decode")
