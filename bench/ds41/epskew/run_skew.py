#!/usr/bin/env python3
"""run_skew.py URL OUTDIR PHASE [PHASE ...]: client for package H3's routing recorder (bench/ds41/epskew/epskew_ext.py,
worker extension of a server launched by run_skew.sh). OUTDIR is the host directory mounted at /h3out in the container.
Phases:
  prof:NS:W     NS greedy prose streams (workload.py's essays); once all are past prefill: reset the ring, POST
                /start_profile, W seconds, /stop_profile, dump -> OUTDIR/prof-s<NS>.pt (the steps of the profile window,
                for skew_fit.py next to the rank traces)
  hist:WL:NS    reset, run workload WL (prose | code | mixed) at NS streams to completion, dump -> OUTDIR/hist-<WL>-s<NS>.pt
  maps          every rank's local expert list per layer -> OUTDIR/maps.json (checks a placement map took)"""
import json, sys, threading, time, urllib.request

B = sys.argv[1].rstrip("/"); OUT = sys.argv[2]; MODEL = "DeepSeek-V4.1"
TOPICS = ["lighthouses", "windmills", "canals", "paper", "railways", "glassmaking", "clocks", "bridges"]
ESSAYS = [f"Write a 3000-word essay about the history of {t}. Use varied vocabulary and no repetition." for t in TOPICS]
PROSE = [f"Write a 700-word essay about the history of {t}. Use varied vocabulary and no repetition." for t in
         TOPICS + ["printing", "maps", "salt", "tea"]]
CODE = ["Write a complete Python module implementing an LRU cache class with get/put, type hints and a docstring, plus three unit tests.",
        "Write a complete Python module implementing an LFU cache class with get/put, type hints and a docstring, plus three unit tests.",
        "Write a complete Python module implementing a fixed-size ring buffer class with push/pop, type hints and a docstring, plus three unit tests.",
        "Write a complete Python module implementing a trie class with insert/search/starts_with, type hints and a docstring, plus three unit tests.",
        "Write a C function that parses an IPv4 address string into a uint32_t with full error checking, and a small test main.",
        "Write a Rust struct for a binary min-heap with push, pop and peek, with doc comments and unit tests.",
        "Write a JavaScript module implementing a debounce and a throttle function with JSDoc and Jest tests.",
        "Write a SQL schema for a library (books, authors, loans, members) with indexes, plus five example queries.",
        "Write a Python asyncio TCP echo server with graceful shutdown and a client test.",
        "Write a Go HTTP handler that serves JSON from an in-memory key-value store with GET/PUT/DELETE and tests.",
        "Write a bash script that rotates log files in a directory, keeping the last seven, with argument parsing.",
        "Write a Python implementation of Dijkstra's algorithm over an adjacency dict, with type hints and tests."]


def post(path, body=None, timeout=600):
    req = urllib.request.Request(B + path, json.dumps(body or {}).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def rpc(method, *args):
    r = json.loads(post("/collective_rpc", {"method": method, "args": list(args), "timeout": 300}, 400))
    return r["results"]


def chat(prompt, max_tokens, stream=False):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.0,
            "chat_template_kwargs": {"thinking": False}}
    if stream:
        body.update(stream=True, ignore_eos=True, stream_options={"include_usage": True, "continuous_usage_stats": True})
    return urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(),
                                                        {"Content-Type": "application/json"}), timeout=1800)


def prof(ns, win):
    toks = [0] * ns
    stop = threading.Event()

    def run(i):
        r = chat(ESSAYS[i % len(ESSAYS)], 4000, stream=True)
        for line in r:
            if stop.is_set():
                break
            if line.startswith(b"data: {"):
                u = json.loads(line[6:]).get("usage")
                if u:
                    toks[i] = u.get("completion_tokens", 0)
        r.close()
    th = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(ns)]
    [t.start() for t in th]
    while min(toks) < 48:
        time.sleep(0.2)
    rpc("r9k_ep_reset")
    t0 = time.time(); k0 = list(toks)
    post("/start_profile")
    time.sleep(win)
    post("/stop_profile", timeout=900)
    t1 = time.time(); k1 = list(toks)
    res = rpc("r9k_ep_dump", f"/h3out/prof-s{ns}.pt")
    stop.set()
    print(f"prof s{ns}: {t1 - t0:.1f} s, tokens {[b - a for a, b in zip(k0, k1)]}, dump {[r for r in res if r.get('path')]}", flush=True)


def hist(wl, ns):
    prompts = {"prose": PROSE, "code": CODE, "mixed": PROSE[:6] + CODE[:6]}[wl]
    rpc("r9k_ep_reset")
    t0 = time.time(); q = list(prompts); lock = threading.Lock(); n = [0]

    def worker():
        while True:
            with lock:
                if not q:
                    return
                p = q.pop(0)
            d = json.loads(chat(p, 900).read())
            with lock:
                n[0] += d["usage"]["completion_tokens"]
    th = [threading.Thread(target=worker) for _ in range(ns)]
    [t.start() for t in th]
    [t.join() for t in th]
    res = rpc("r9k_ep_dump", f"/h3out/hist-{wl}-s{ns}.pt")
    ptr = [r for r in res if r.get("path")][0]["ptr"]
    print(f"hist {wl} s{ns}: {len(prompts)} requests, {n[0]} tokens, {time.time() - t0:.0f} s, steps a layer {max(ptr)}"
          f"{' (ring wrapped: last N kept)' if max(ptr) > 4096 else ''}", flush=True)


for ph in sys.argv[3:]:
    a = ph.split(":")
    if a[0] == "prof":
        prof(int(a[1]), float(a[2]))
    elif a[0] == "hist":
        hist(a[1], int(a[2]))
    elif a[0] == "maps":
        json.dump(rpc("r9k_ep_maps"), open(f"{OUT}/maps.json", "w"))
        print("maps written", flush=True)
