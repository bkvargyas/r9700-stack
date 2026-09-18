#!/usr/bin/env python3
"""Repeated-run benchmark for an OpenAI-compatible vLLM endpoint.

Single-shot numbers on this stack moved +-5-10% with compile-cache state, prefix-cache hits and run order, so:
  * every prompt carries a random nonce up front (no prefix-cache hits, prefill really prefills);
  * single-stream uses streaming: TTFT and decode rate are measured separately (decode = tokens after the first
    over the time after the first; MTP chunks are counted by usage, not by chunk);
  * each metric is repeated REPS times and reported as median [min..max] with the coefficient of variation;
  * results append as JSON lines (one per metric) tagged with the config label, for `summarize`.

usage:
  harness.py run LABEL [--base URL] [--reps 5] [--out results.jsonl] [--quick]
  harness.py summarize results.jsonl [...]      # median-of-medians per label/metric, A/B deltas vs first label
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.request

MODEL = "Qwen3.8"
ESSAY = "Write a detailed 300-word essay about the history of computing."


def _nonce() -> str:
    return f"[req {random.getrandbits(48):012x}] "


def _post(base, body, timeout=600):
    return urllib.request.urlopen(urllib.request.Request(
        base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}),
        timeout=timeout)


def _body(prompt, max_tokens, stream=False):
    b = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
         "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True}
    if stream:
        b |= {"stream": True, "stream_options": {"include_usage": True}}
    return b


def stream_one(base, prompt, max_tokens):
    """-> (ttft_s, decode_tok_s, completion_tokens, prompt_tokens)"""
    t0 = time.perf_counter()
    t_first = t_last = None
    usage = {}
    with _post(base, _body(prompt, max_tokens, stream=True)) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            ch = ev.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                now = time.perf_counter()
                t_first = t_first or now
                t_last = now
    ct = usage.get("completion_tokens", 0)
    dec = (ct - 1) / (t_last - t_first) if t_first and t_last > t_first and ct > 1 else float("nan")
    return (t_first or t0) - t0, dec, ct, usage.get("prompt_tokens", 0)


def one(base, prompt, max_tokens):
    t0 = time.perf_counter()
    u = json.loads(_post(base, _body(prompt, max_tokens)).read())["usage"]
    return u["completion_tokens"], u["prompt_tokens"], time.perf_counter() - t0


def aggregate(base, conc, max_tokens=256):
    res, lock = [], threading.Lock()

    def w(i):
        ct, _, _ = one(base, _nonce() + f"Explain concept number {i}: describe tensor parallelism in depth.",
                       max_tokens)
        with lock:
            res.append(ct)
    t0 = time.perf_counter()
    ts = [threading.Thread(target=w, args=(i,)) for i in range(conc)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return sum(res) / (time.perf_counter() - t0)


def prefill(base, words):
    prompt = _nonce() + " ".join(f"w{random.randint(0, 10**6)}" for _ in range(words)) + "\nSummarize in 3 words."
    ttft, _, _, pt = stream_one(base, prompt, 2)
    return pt / ttft


def _stats(xs):
    xs = [x for x in xs if x == x]
    if not xs:
        return {"median": float("nan"), "min": float("nan"), "max": float("nan"), "cv": float("nan"), "n": 0}
    m = statistics.median(xs)
    cv = statistics.pstdev(xs) / statistics.mean(xs) if len(xs) > 1 else 0.0
    return {"median": m, "min": min(xs), "max": max(xs), "cv": cv, "n": len(xs), "samples": xs}


def run(a):
    base, reps = a.base, a.reps
    one(base, "hi", 4)                                                  # connection + graph warm
    metrics = {}
    sd, ttft = [], []
    for _ in range(reps):
        t, d, _, _ = stream_one(base, _nonce() + ESSAY, 300)
        ttft.append(t * 1000)
        sd.append(d)
    metrics["single_decode_tok_s"] = sd
    metrics["single_ttft_ms"] = ttft
    for c in ((8,) if a.quick else (4, 8, 16)):
        metrics[f"agg{c}_tok_s"] = [aggregate(base, c) for _ in range(max(2, reps // 2 + 1))]
    for w in ((2000,) if a.quick else (2000, 8000)):
        metrics[f"prefill{w}w_tok_s"] = [prefill(base, w) for _ in range(max(2, reps // 2 + 1))]
    with open(a.out, "a") as f:
        for k, xs in metrics.items():
            s = _stats(xs)
            f.write(json.dumps({"label": a.label, "metric": k, "t": time.time(), **s}) + "\n")
            print(f"[{a.label}] {k:22s} {s['median']:9.1f}  [{s['min']:.1f}..{s['max']:.1f}]  cv {s['cv']*100:4.1f}%"
                  f"  n={s['n']}")


def summarize(paths):
    rows = [json.loads(l) for p in paths for l in open(p) if l.strip()]
    labels = list(dict.fromkeys(r["label"] for r in rows))
    metrics = list(dict.fromkeys(r["metric"] for r in rows))
    by = {}
    for r in rows:
        by.setdefault((r["label"], r["metric"]), []).append(r["median"])
    ref = labels[0]
    print(f"{'metric':22s}" + "".join(f"{l[:22]:>24s}" for l in labels))
    for m in metrics:
        line = f"{m:22s}"
        r0 = by.get((ref, m))
        for l in labels:
            xs = by.get((l, m))
            if not xs:
                line += f"{'-':>24s}"
                continue
            med = statistics.median(xs)
            rng = (max(xs) - min(xs)) / med * 100 if len(xs) > 1 and med else 0.0
            cell = f"{med:.1f} ±{rng / 2:.0f}% ({len(xs)})"
            if l != ref and r0:
                cell += f" {(med / statistics.median(r0) - 1) * 100:+.1f}%"
            line += f"{cell:>24s}"
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("label")
    r.add_argument("--base", default="http://localhost:8080")
    r.add_argument("--reps", type=int, default=5)
    r.add_argument("--out", default="results.jsonl")
    r.add_argument("--quick", action="store_true")
    s = sp.add_parser("summarize")
    s.add_argument("paths", nargs="+")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else summarize(a.paths)
