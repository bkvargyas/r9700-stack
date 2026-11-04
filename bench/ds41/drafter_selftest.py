#!/usr/bin/env python3
"""drafter_selftest.py URL MODEL [--prose-min 1.3] [--code-min 3.0]: post-READY self-test of the DSpark drafter.

Sends one greedy prose request and one greedy code request (~200 completion tokens each, thinking off, one stream) and
reads the server's speculative-decoding counters from /metrics around each request: mean acceptance length
MAL = 1 + accepted / drafts. Exits 0 when MAL >= --prose-min on prose and >= --code-min on code; exits 1 with a clear
message otherwise (a drafter that accepts nothing runs at MAL ~1.0 while the target's text stays sane, so a launch can
look healthy without this check: notes/exl3-drafter-heisenbug.md). Exit 2: the server could not be queried.
Healthy DeepSeek-V4.1 EXL3 + DSpark k=5 numbers: prose 1.7-2.0, code 4.1-4.5 (notes/exl3-serving.md)."""
import argparse, json, re, sys, urllib.request

PROSE = "Write a 500-word essay about the history of lighthouses. Use varied vocabulary and no repetition."
CODE = ("Write a complete Python module implementing an LRU cache class with get/put, type hints and a docstring, "
        "plus three unit tests.")


def metrics(base):
    txt = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    acc = drafts = 0.0
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r'vllm:spec_decode_num_(accepted_tokens|drafts)(?:_total)?(\{[^}]*\})?\s+([0-9.e+-]+)$', line)
        if m:
            if m.group(1) == "drafts":
                drafts += float(m.group(3))
            else:
                acc += float(m.group(3))
    return acc, drafts


def ask(base, model, prompt, n):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0.0,
            "chat_template_kwargs": {"thinking": False}}
    r = urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=900)
    return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("--prose-min", type=float, default=1.3)
    ap.add_argument("--code-min", type=float, default=3.0)
    ap.add_argument("--tokens", type=int, default=200)
    a = ap.parse_args()
    base = a.url.rstrip("/")
    bad = []
    try:
        for name, prompt, lo in (("prose", PROSE, a.prose_min), ("code", CODE, a.code_min)):
            a0, d0 = metrics(base)
            r = ask(base, a.model, prompt, a.tokens)
            a1, d1 = metrics(base)
            drafts = d1 - d0
            mal = 1.0 + (a1 - a0) / drafts if drafts > 0 else float("nan")
            ntok = r.get("usage", {}).get("completion_tokens")
            ok = drafts > 0 and mal >= lo
            print(f"drafter_selftest {name}: {ntok} tokens, {int(drafts)} drafts, MAL {mal:.2f} (min {lo}) "
                  f"{'ok' if ok else 'FAIL'}", flush=True)
            if not ok:
                bad.append(f"{name} MAL {mal:.2f} < {lo}" if drafts > 0 else f"{name}: no drafts counted (drafter off?)")
    except Exception as e:  # noqa: BLE001
        print(f"drafter_selftest: cannot query {base}: {e}", file=sys.stderr)
        sys.exit(2)
    if bad:
        print("drafter_selftest FAILED: the DSpark drafter is not accepting drafts (" + "; ".join(bad) +
              "). The target may still answer correctly; speculation is broken. See notes/exl3-drafter-heisenbug.md.",
              file=sys.stderr)
        sys.exit(1)
    print("drafter_selftest OK")


if __name__ == "__main__":
    main()
