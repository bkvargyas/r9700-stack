#!/usr/bin/env python3
"""warm.py PORT MODEL: the launcher's WARM=1 requests (decode, one 2.5k-token prefill, decode) so the Triton JITs and the
drafter's first-step warm-up land before the measured windows (serve/ds41-exl3.sh WARM block, as a separate step)."""
import json, sys, urllib.request
port, model = sys.argv[1], sys.argv[2]
words = " ".join(f"w{i}" for i in range(1500))
for n, m in ((8, 96), (2500, 4), (8, 96), (8, 96)):
    body = {"model": model, "messages": [{"role": "user", "content": words[:n * 2] + " Say ok."}], "max_tokens": m,
            "ignore_eos": True, "chat_template_kwargs": {"thinking": False}}
    urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", json.dumps(body).encode(),
                                                  {"Content-Type": "application/json"}), timeout=600).read()
print("warm-up done")
