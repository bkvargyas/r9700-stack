#!/usr/bin/env python3
"""needle.py URL MODEL ctx_tokens [depth_fracs]: retrieval of planted facts from a long filler prompt (thinking off).
Filler is randomized sentences (~9 tokens each); facts planted at the given depth fractions; one question per fact."""
import json, random, sys, time, urllib.request
U = sys.argv[1].rstrip("/") + "/v1/chat/completions"; MODEL = sys.argv[2]; CTX = int(sys.argv[3])
DEPTHS = [float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0.1, 0.5, 0.9]
random.seed(CTX)
S = ["The weather was mild and the market was busy.", "Several boats were moored along the quay.", "A dog barked somewhere down the lane.",
     "The bakery sold out of rye bread by noon.", "Children played near the old fountain.", "The clock tower struck the hour twice.",
     "A courier delivered parcels to the inn.", "The ferry schedule changed for the season."]
FACTS = [("The secret password for the archive is 'violet-kestrel-42'.", "What is the secret password for the archive?", "violet-kestrel-42"),
         ("Marta's bicycle is painted bright orange and parked behind the chapel.", "What color is Marta's bicycle and where is it parked?", "orange"),
         ("The quarterly meeting was moved to Thursday the 17th at 14:30.", "When was the quarterly meeting moved to?", "17"),
         ("Professor Ilves keeps the lab key inside a hollow book titled 'Tides'.", "Where does Professor Ilves keep the lab key?", "Tides")]
n_sent = max(CTX // 9, 10)
sents = [random.choice(S) for _ in range(n_sent)]
qa = []
for i, d in enumerate(DEPTHS):
    f, q, a = FACTS[i % len(FACTS)]
    sents.insert(int(d * len(sents)), f); qa.append((q, a))
doc = " ".join(sents)
ok = 0
for q, a in qa:
    body = {"model": MODEL, "messages": [{"role": "user", "content": doc + "\n\nQuestion: " + q + " Answer in one short sentence."}],
            "max_tokens": 60, "temperature": 0.0, "chat_template_kwargs": {"thinking": False}}
    t = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800).read())
    dt = time.time() - t; c = str(r["choices"][0]["message"].get("content")); hit = a.lower() in c.lower(); ok += hit
    print(f"[{r['usage']['prompt_tokens']} prompt tok, {dt:.1f}s] {'OK  ' if hit else 'MISS'} q={q[:45]!r} -> {c[:120]!r}", flush=True)
print(f"needle ctx~{CTX}: {ok}/{len(qa)} retrieved")
