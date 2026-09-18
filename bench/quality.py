#!/usr/bin/env python3
"""Fast quality gate for an OpenAI-compatible server: GSM8K subset (no thinking) + needle-in-haystack.

usage: quality.py [N_GSM8K=100] [CONC=8] [BASE=http://localhost:8080] [MODEL=Qwen3.8]
Prints accuracy and needle hits; exit 0. Deterministic (temperature 0) so two stacks can be compared.
"""
import concurrent.futures as cf
import json
import random
import re
import sys
import time
import urllib.request

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 8
BASE = (sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8080") + "/v1/chat/completions"
MODEL = sys.argv[4] if len(sys.argv) > 4 else "Qwen3.8"


def chat(content, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return r["choices"][0]["message"].get("content") or "", r["usage"]


def last_number(t):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", (t or "").replace(",", ""))
    return nums[-1].rstrip(".") if nums else None


def gsm8k():
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    rows = [json.loads(x) for x in urllib.request.urlopen(url, timeout=60).read().decode().splitlines()]
    rows = rows[:N]

    def one(d):
        gt = d["answer"].split("####")[-1].strip().replace(",", "")
        out, u = chat(d["question"] + "\nSolve step by step, then give the final numeric answer on the last line.", 768)
        got = last_number(out)
        try:
            ok = got is not None and abs(float(got) - float(gt)) < 1e-6
        except ValueError:
            ok = False
        return ok, u["completion_tokens"]

    t = time.time()
    with cf.ThreadPoolExecutor(CONC) as ex:
        res = list(ex.map(one, rows))
    acc = sum(r[0] for r in res) / len(res)
    toks = sum(r[1] for r in res)
    print(f"GSM8K no-think: {acc*100:.1f}% ({sum(r[0] for r in res)}/{len(res)}), {toks} tokens in {time.time()-t:.0f}s")


def needle():
    random.seed(0)
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    hits = 0
    for ctx_words in (3000, 12000, 24000):
        key = random.randint(100000, 999999)
        words = (filler * (ctx_words // 18 + 1)).split()[:ctx_words]
        pos = random.randint(ctx_words // 5, 4 * ctx_words // 5)
        words.insert(pos, f"The secret passcode is {key}.")
        out, u = chat(" ".join(words) + "\n\nWhat is the secret passcode? Reply with the number only.", 16)
        ok = str(key) in out
        hits += ok
        print(f"needle ~{u['prompt_tokens']} tok: {'hit' if ok else 'MISS'} ({out.strip()[:30]!r})")
    print(f"needle: {hits}/3")


if __name__ == "__main__":
    gsm8k()
    needle()
