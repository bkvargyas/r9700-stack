#!/usr/bin/env python3
"""Repeat the 8-way sanity check under the conditions that broke it once (2026-10-02): right after long prefills.

Each round: PRE long-context requests (a real-text slice and/or the BetterBench filler, 2 output tokens), then CONC
concurrent "What is N times 3?" questions. STRICT: the answer must be the number and nothing else. Prints every bad
answer in full. Exit status 1 if any round had a bad answer.

usage: sanity_stress.py [URL] [--rounds 20] [--sanity-reps 1] [--conc 8] [--pre docs,bb | --pre ""] [--depth 24000] [--text FILE]
"""
import json, os, random, re, sys, threading, time, urllib.request

B = next((a for a in sys.argv[1:] if a.startswith("http")), "http://localhost:8080")
def opt(name, default):
    return next((sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a == name), default)
ROUNDS, CONC, DEPTH = int(opt("--rounds", "20")), int(opt("--conc", "8")), int(opt("--depth", "24000"))
REPS = int(opt("--sanity-reps", "1"))        # 8-way sanity checks per round
PRE = [k for k in opt("--pre", "docs,bb").split(",") if k]
TEXT = opt("--text", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "PROGRESS.md"))
DOC = open(TEXT, errors="replace").read() if "docs" in PRE else ""
WORDS = ("In distributed systems the tension between consistency, availability, and partition tolerance shapes almost "
         "every design decision. A service that prioritizes strong consistency may reject writes during a network "
         "split, while an available-first design accepts them and reconciles later.").split()

def post(content, max_tokens, temperature=0):
    b = {"model": "Qwen3.8", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
         "temperature": temperature, "chat_template_kwargs": {"enable_thinking": False}}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", json.dumps(b).encode(),
                                          {"Content-Type": "application/json"}), timeout=900).read())
    return r["choices"][0]["message"]["content"] or "", r["usage"]

def pre(kind, chars):
    if kind == "bb":
        body = " ".join(random.choices(WORDS, k=chars // 6 + 64))[:chars]
    else:
        o = random.randrange(0, max(1, len(DOC) - chars)); body = (DOC + DOC)[o:o + chars]
    return post(f"[{random.getrandbits(48):012x}] Read the following context, then reply with the single word: ack.\n\n"
                f"{body}\n\nReply with one word: ack.", 2)[1]["prompt_tokens"]

cpt = {}
for k in set(PRE):
    cpt[k] = 6000 / max(pre(k, 6000) - 40, 1)
bad_rounds, bad_total, t0 = 0, 0, time.time()
for r in range(ROUNDS):
    depths = [pre(k, int(DEPTH * cpt[k])) for k in PRE]
    bad = {}
    for rep in range(REPS):
        out = {}
        base = random.randint(11, 60)
        def q(i):
            out[i] = post(f"What is {base + i} times 3? Reply with just the number.", 16)[0].strip()
        ts = [threading.Thread(target=q, args=(i,)) for i in range(CONC)]
        [t.start() for t in ts]; [t.join() for t in ts]
        bad.update({f"{rep}.{i}": out[i] for i in range(CONC) if not re.fullmatch(rf"{(base + i) * 3}\.?", out[i])})
    bad_total += len(bad); bad_rounds += bool(bad)
    print(f"round {r + 1:2d}  pre {depths}  sanity {CONC * REPS - len(bad)}/{CONC * REPS}" + ("" if not bad else "  BAD: " + json.dumps(bad)), flush=True)
print(f"sanity_stress: {ROUNDS} rounds, {bad_rounds} with a bad answer, {bad_total} bad answers of {ROUNDS * CONC * REPS}  ({time.time() - t0:.0f} s)")
sys.exit(1 if bad_total else 0)
