#!/usr/bin/env python3
"""An eval that can actually detect a small regression, unlike GSM8K-500.

    eval.py selftest [N] [CONC]        # is the stack reproducible at all? run this FIRST
    eval.py run NAME [N] [CONC]        # record per-question outputs to ~/.r9keval/NAME.json
    eval.py compare A B                # paired comparison of two runs

Why this exists: `gsm8k.py` already samples at temperature 0, yet six runs scattered 95.6-97.4% with no
consistent ordering. So that scatter was never sampling noise -- it is the stack itself being non-reproducible,
and the prime suspect is dynamic batching: at concurrency > 1 the batch composition varies run to run, which
changes GEMM shapes and reduction order, which changes numerics, which flips tokens. `selftest` measures that
directly, because there is no point building statistics on top of an unquantified noise floor.

What makes this more sensitive than an accuracy number:

  * **Output agreement.** Under greedy decoding two numerically equivalent configs should emit the SAME tokens.
    Divergence rate is far more sensitive than accuracy -- a lossy change can flip several percent of tokens
    while accuracy barely moves, because most flips do not change the final answer. This is the metric that can
    see a 6-bit all-reduce or an fp8 KV cache.
  * **Paired testing.** Both configs answer the same questions, so we compare per-question outcomes and apply
    McNemar's test to the discordant pairs. Vastly more powerful than comparing two independent accuracies:
    detecting a 1% shift in an unpaired proportion near 97% needs ~4,500 questions per arm, which is more than
    the whole GSM8K test set.
  * **The full test set**, not a 500-question slice.
"""
import concurrent.futures as cf
import hashlib
import json
import math
import os
import pathlib
import re
import sys
import urllib.request

BASE = os.environ.get("EVAL_BASE", "http://localhost:8080") + "/v1/chat/completions"
MODEL = os.environ.get("EVAL_MODEL", "Qwen3.8")
OUT = pathlib.Path(os.environ.get("EVAL_DIR", os.path.expanduser("~/.r9keval")))
CACHE = pathlib.Path(os.path.expanduser("~/.cache/r9keval"))


def load_gsm8k():
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "gsm8k_test.jsonl"
    if not f.exists():
        url = ("https://raw.githubusercontent.com/openai/grade-school-math/master/"
               "grade_school_math/data/test.jsonl")
        f.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    out = []
    for line in f.read_text().splitlines():
        d = json.loads(line)
        gt = d["answer"].split("####")[-1].strip().replace(",", "")
        out.append((d["question"], gt))
    return out


def last_number(text):
    if not text:
        return None
    nums = re.findall(r"-?\$?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1].replace("$", "").rstrip(".") if nums else None


THINK = os.environ.get("EVAL_THINK") == "1"


def ask(q):
    """Greedy. EVAL_THINK=1 turns on long chain-of-thought.

    That mode matters for numerics: a lossy change perturbs every layer of every forward, and a short answer
    gives that perturbation almost no chance to compound. A long reasoning chain is where a small per-call
    error turns into a different conclusion, so a null result on short answers is the WEAKEST possible
    evidence that a numeric change is safe."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": q + "\nGive the final numeric answer."}],
            "max_tokens": 4096 if THINK else 1024, "temperature": 0.0, "top_p": 1.0, "seed": 1234,
            "chat_template_kwargs": ({"enable_thinking": True, "reasoning_effort": "medium"} if THINK
                                     else {"enable_thinking": False})}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return r["choices"][0]["message"]["content"] or ""


def run(n, conc):
    data = load_gsm8k()[:n]
    res = [None] * len(data)

    def work(i):
        q, gt = data[i]
        err = None
        try:
            txt = ask(q)
        except Exception as e:
            txt, err = "", f"{type(e).__name__}"
        got = last_number(txt)
        ok = got is not None and abs(float(got) - float(gt)) < 1e-6 if _num(got) and _num(gt) else False
        return i, {"q_sha": hashlib.sha1(q.encode()).hexdigest()[:12], "gt": gt, "got": got, "ok": bool(ok),
                   "out_sha": hashlib.sha1(txt.encode()).hexdigest()[:16], "ntok_approx": len(txt) // 4,
                   "err": err, "text": txt}

    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        for i, r in ex.map(work, range(len(data))):
            res[i] = r
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(data)}", flush=True)
    return res


def _num(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def nerr(res):
    return sum(1 for r in res if r.get("err"))


def check(res, name="run"):
    """A request that never reached the server is not a wrong answer. Scoring it as one turns a dead server
    into a confident 0.00% and a p=0.000 'regression' -- which is exactly what happened on 2026-09-22."""
    e = nerr(res)
    if e:
        kinds = {}
        for r in res:
            if r.get("err"):
                kinds[r["err"]] = kinds.get(r["err"], 0) + 1
        raise SystemExit(f"REFUSING to score {name}: {e}/{len(res)} requests failed ({kinds}). "
                         f"The server was not healthy; this is not a quality result. Fix the run and repeat.")


def acc(res):
    return sum(r["ok"] for r in res) / max(len(res), 1)


def agreement(a, b):
    """Fraction of questions where the two runs produced byte-identical output."""
    n = min(len(a), len(b))
    same = sum(1 for i in range(n) if a[i]["out_sha"] == b[i]["out_sha"])
    return same / max(n, 1), n


def mcnemar(a, b):
    """Paired test on accuracy. Returns (a_only_right, b_only_right, two-sided p)."""
    n01 = sum(1 for x, y in zip(a, b) if x["ok"] and not y["ok"])
    n10 = sum(1 for x, y in zip(a, b) if y["ok"] and not x["ok"])
    m = n01 + n10
    if m == 0:
        return n01, n10, 1.0
    # exact binomial two-sided
    k = min(n01, n10)
    p = 2 * sum(math.comb(m, i) for i in range(k + 1)) / (2 ** m)
    return n01, n10, min(p, 1.0)


def save(name, res, meta):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps({"meta": meta, "res": res}))
    print(f"wrote {OUT / (name + '.json')}")


def load(name):
    return json.loads((OUT / f"{name}.json").read_text())


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"

    if cmd == "selftest":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        print(f"Self-consistency: the SAME config twice. Any disagreement is the stack's own noise floor,\n"
              f"and sets the smallest regression this eval can ever detect. n={n}\n")
        for conc in (1, 8):
            a = run(n, conc)
            check(a, f"selftest conc={conc} run 1")
            b = run(n, conc)
            check(b, f"selftest conc={conc} run 2")
            ag, m = agreement(a, b)
            n01, n10, p = mcnemar(a, b)
            print(f"  conc={conc}: acc {acc(a)*100:.2f}% vs {acc(b)*100:.2f}%   "
                  f"identical output {ag*100:.1f}% of {m}   flips {n01}/{n10}")
            if ag < 1.0:
                print(f"    -> NOT reproducible at conc={conc}. {(1-ag)*100:.1f}% of answers differ between "
                      f"two runs of the same build.")
            else:
                print(f"    -> bit-reproducible at conc={conc}: a real difference of ANY size is detectable.")
        return

    if cmd == "run":
        name = sys.argv[2]
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 1319
        conc = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        res = run(n, conc)
        check(res, name)
        save(name, res, {"n": n, "conc": conc, "base": BASE, "model": MODEL, "think": THINK,
                         "env": {k: v for k, v in os.environ.items() if k.startswith("R9K_")}})
        print(f"{name}: acc {acc(res)*100:.2f}%  ({sum(r['ok'] for r in res)}/{len(res)})")
        return

    if cmd == "compare":
        A, B = load(sys.argv[2]), load(sys.argv[3])
        a, b = A["res"], B["res"]
        check(a, sys.argv[2])
        check(b, sys.argv[3])
        ag, m = agreement(a, b)
        n01, n10, p = mcnemar(a, b)
        print(f"{sys.argv[2]}  acc {acc(a)*100:.2f}%   n={len(a)}  conc={A['meta']['conc']}")
        print(f"{sys.argv[3]}  acc {acc(b)*100:.2f}%   n={len(b)}  conc={B['meta']['conc']}")
        print(f"\n  identical output on {ag*100:.1f}% of {m} questions"
              f"   <- the sensitive metric; accuracy can hide numeric drift")
        print(f"  paired accuracy: {sys.argv[2]}-only-right {n01}, {sys.argv[3]}-only-right {n10}, "
              f"McNemar p={p:.3f}")
        verdict = ("no detectable difference" if p > 0.05 else
                   f"DIFFERENT (p={p:.3f}) -- {sys.argv[3] if n01 > n10 else sys.argv[2]} is worse")
        print(f"  verdict: {verdict}")
        if ag < 1.0:
            print(f"\n  NOTE: {(1-ag)*100:.1f}% of outputs differ. Compare that against the selftest noise "
                  f"floor at the same concurrency before reading anything into it.")
        for k in ("env",):
            da = {kk: vv for kk, vv in A["meta"].get(k, {}).items()}
            db = {kk: vv for kk, vv in B["meta"].get(k, {}).items()}
            diff = {kk for kk in set(da) | set(db) if da.get(kk) != db.get(kk)}
            if diff:
                print(f"\n  config differences: " + ", ".join(
                    f"{kk}: {da.get(kk, '-')} -> {db.get(kk, '-')}" for kk in sorted(diff)))
        return

    print(__doc__)
    sys.exit(2)


if __name__ == "__main__":
    main()
