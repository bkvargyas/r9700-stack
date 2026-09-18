#!/usr/bin/env python3
"""Prefill throughput with unique random prompts (no prefix-cache hits). usage: prefill.py [words ...]"""
import json, random, sys, time, urllib.request
BASE = "http://localhost:8080/v1/chat/completions"
for w in [int(x) for x in (sys.argv[1:] or ["120", "500", "2000", "8000"])]:
    p = " ".join(f"{random.choice(['alpha','river','stone','cloud','north'])}{random.randint(0, 999)}" for _ in range(w))
    body = {"model": "Qwen3.8", "messages": [{"role": "user", "content": p + " Summarize."}], "max_tokens": 1,
            "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(BASE, json.dumps(body).encode(),
                                          {"Content-Type": "application/json"}), timeout=900).read())
    dt = time.time() - t; n = r["usage"]["prompt_tokens"]
    print(f"prefill {n:6d} tok: {n/dt:7.0f} tok/s ({dt:.2f}s)", flush=True)
