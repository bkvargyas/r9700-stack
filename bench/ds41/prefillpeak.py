#!/usr/bin/env python3
"""prefillpeak.py URL MODEL ctx_tokens [streams]: send `streams` concurrent prompts of ~ctx_tokens random filler (thinking off,
4 output tokens) -- a prefill-only load whose chunks are max-num-batched-tokens wide -- and report TTFT. Pair with
memprobe.py reset / mem (allocator peak) and vramwatch.py (driver peak)."""
import json, random, sys, threading, time, urllib.request
U = sys.argv[1].rstrip("/") + "/v1/chat/completions"; MODEL = sys.argv[2]; CTX = int(sys.argv[3]); NS = int(sys.argv[4]) if len(sys.argv) > 4 else 1
def one(i, out):
    random.seed(CTX * 7 + i + int(time.time()))
    words = " ".join(random.choice(["alpha", "river", "stone", "cloud", "north", "delta"]) + str(random.randint(0, 999)) for _ in range(int(CTX / 2.6)))
    body = {"model": MODEL, "messages": [{"role": "user", "content": words + "\nSay ok."}], "max_tokens": 4, "temperature": 0,
            "chat_template_kwargs": {"thinking": False}}
    t = time.time(); r = json.loads(urllib.request.urlopen(urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800).read())
    out[i] = (r["usage"]["prompt_tokens"], time.time() - t)
o = {}; th = [threading.Thread(target=one, args=(i, o)) for i in range(NS)]; t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]
for i in sorted(o): print(f"stream {i}: {o[i][0]} prompt tokens, {o[i][1]:.2f} s")
print(f"wall {time.time() - t0:.2f} s, {sum(v[0] for v in o.values()) / (time.time() - t0):.0f} prompt tok/s")
