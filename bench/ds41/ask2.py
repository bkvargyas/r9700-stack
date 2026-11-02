#!/usr/bin/env python3
"""ask2.py URL MODEL: sampling / thinking variants of one prompt, to see whether the repetition is greedy-specific."""
import json, sys, time, urllib.request
U = sys.argv[1].rstrip("/") + "/v1/chat/completions"; MODEL = sys.argv[2]
def ask(label, prompt, n, **kw):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": n}
    body.update(kw)
    t = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900).read())
    m = r["choices"][0]["message"]; u = r["usage"]
    reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
    print(f"--- {label} [{u['completion_tokens']} tok, finish={r['choices'][0].get('finish_reason')}]")
    print(f"    reasoning: {reasoning[:300]!r}")
    print(f"    content:   {str(m.get('content'))[:300]!r}", flush=True)
P = "What is the capital of France? Answer with one word."
ask("greedy, thinking off", P, 40, temperature=0.0, chat_template_kwargs={"thinking": False})
ask("temp 0.6 top_p 0.95, thinking off", P, 40, temperature=0.6, top_p=0.95, chat_template_kwargs={"thinking": False})
ask("temp 1.0 top_p 0.95, thinking on", P, 120, temperature=1.0, top_p=0.95)
ask("greedy, thinking off, longer", "List the first ten prime numbers separated by commas.", 60, temperature=0.0, chat_template_kwargs={"thinking": False})
ask("greedy, thinking off, 200 tok", "Write three sentences about the ocean.", 200, temperature=0.0, chat_template_kwargs={"thinking": False})
