#!/usr/bin/env python3
"""divergence.py URL MODEL [n_tokens]: decode-vs-prefill agreement. Greedy-decode N tokens from a raw prompt (completions
API, token ids in/out), then for each prefix recompute the next-token distribution by a fresh prefill (prefix caching must
be off) and compare: top-1 agreement and the logprob the prefill assigns to the decode-chosen token."""
import json, sys, urllib.request
U = sys.argv[1].rstrip("/") + "/v1/completions"; MODEL = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 24
def post(body):
    body.update({"model": MODEL, "temperature": 0.0, "logprobs": 5, "return_tokens_as_token_ids": True})
    return json.loads(urllib.request.urlopen(urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900).read())["choices"][0]
def tid(s): return int(s.split(":")[1])
for prompt in ["The history of lighthouses begins in antiquity, when",
               "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number iteratively.\"\"\"\n",
               "Q: Compute 17 * 23 and explain briefly.\nA:"]:
    d = post({"prompt": prompt, "max_tokens": N})
    toks = [tid(t) for t in d["logprobs"]["tokens"]]
    dlp = d["logprobs"]["token_logprobs"]
    # prompt token ids: ask the server to echo the prompt once
    e = post({"prompt": prompt, "max_tokens": 1, "echo": True})
    ptoks = [tid(t) for t in e["logprobs"]["tokens"]][:-1]
    agree, dlps = 0, []
    rows = []
    for k in range(N):
        p = post({"prompt": ptoks + toks[:k], "max_tokens": 1})
        ptop = tid(p["logprobs"]["tokens"][0])
        top5 = {tid(t): lp for t, lp in p["logprobs"]["top_logprobs"][0].items()}
        plp = top5.get(toks[k])
        agree += ptop == toks[k]
        if plp is not None: dlps.append(abs(plp - dlp[k]))
        rows.append(f"{k:2d} dec={toks[k]:6d}({dlp[k]:6.2f}) pre={ptop:6d}({p['logprobs']['token_logprobs'][0]:6.2f}) {'ok' if ptop == toks[k] else 'DIFF'}{'' if plp is not None else ' (decode token not in prefill top5)'}")
    print(f"=== {prompt[:50]!r}: top-1 agreement {agree}/{N}; mean |dlogprob| of the decode token {sum(dlps)/max(len(dlps),1):.3f} over {len(dlps)}")
    print("    decode text: " + repr(d["text"][:160]))
    for r in rows:
        if "DIFF" in r or "not in" in r: print("    " + r)
