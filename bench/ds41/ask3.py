#!/usr/bin/env python3
"""ask3.py URL MODEL: length / context discriminators with thinking off."""
import json, sys, time, urllib.request, random
U = sys.argv[1].rstrip("/") + "/v1/chat/completions"; MODEL = sys.argv[2]
def ask(label, prompt, n, **kw):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0.0,
            "chat_template_kwargs": {"thinking": False}}
    body.update(kw)
    t = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900).read())
    m = r["choices"][0]["message"]; u = r["usage"]; dt = time.time() - t
    c = str(m.get("content"))
    print(f"--- {label} [{u['prompt_tokens']} prompt, {u['completion_tokens']} tok, finish={r['choices'][0].get('finish_reason')}, {dt:.1f}s]")
    print("    head: " + repr(c[:260]))
    print("    tail: " + repr(c[-300:]), flush=True)
ask("long essay", "Write a 500-word essay about the history of lighthouses. Use varied vocabulary and no repetition.", 700)
ask("code", "Write a complete Python module implementing an LRU cache class with get/put, type hints and a docstring, plus three unit tests.", 700)
random.seed(7)
facts = ["The red fox lives in the north meadow.", "Clara keeps her spare key under the blue flowerpot.", "The train to Oslo leaves at 6:40.",
         "Grandfather's violin is stored in the attic trunk.", "The library closes at nine on Thursdays."]
filler = " ".join(random.choice(["The weather was mild and the market was busy.", "Several boats were moored along the quay.", "A dog barked somewhere down the lane.", "The bakery sold out of rye bread by noon.", "Children played near the old fountain."]) for _ in range(180))
doc = filler[:len(filler)//2] + " " + facts[1] + " " + filler[len(filler)//2:] + " " + facts[2]
ask("needle ~1.5k ctx", doc + "\n\nQuestion: where does Clara keep her spare key, and when does the train to Oslo leave? Answer briefly.", 80)
ask("needle ~3k ctx", doc + " " + filler + "\n\nQuestion: where does Clara keep her spare key? One sentence.", 60)
