#!/usr/bin/env python3
"""Long-context soak: CONC clients for SECONDS, each sending a prompt of a random depth in [LO, HI] tokens cut from a
real text and asking for 256 tokens. Fills the KV pool while prefill chunks and decode steps overlap, which is the
memory peak a short benchmark never reaches. Exit status 1 on any error that is not a context-length rejection.

usage: soak.py [URL] [--text FILE] [--conc 8] [--seconds 900] [--lo 18000] [--hi 30000]
"""
import json, os, random, sys, threading, time, urllib.request, urllib.error, statistics as st

B = next((a for a in sys.argv[1:] if a.startswith("http")), "http://localhost:8080")
def opt(name, default):
    return next((sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a == name), default)
TEXT = opt("--text", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "PROGRESS.md"))
CONC, SECS = int(opt("--conc", "8")), int(opt("--seconds", "900"))
LO, HI = int(opt("--lo", "18000")), int(opt("--hi", "30000"))
DOC = open(TEXT, errors="replace").read()

def post(chars, max_tokens):
    o = random.randrange(0, max(1, len(DOC) - chars))
    msg = f"[{random.getrandbits(48):012x}] Summarize the following notes in three paragraphs.\n\n{(DOC + DOC)[o:o + chars]}"
    b = {"model": "Qwen3.8", "messages": [{"role": "user", "content": msg}], "max_tokens": max_tokens,
         "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}}
    return json.loads(urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", json.dumps(b).encode(),
                                             {"Content-Type": "application/json"}), timeout=1800).read())["usage"]

u = post(8000, 4)
cpt = 8000 / max(u["prompt_tokens"] - 30, 1)
ok, rej, err, lock, t_end = [], [], [], threading.Lock(), time.time() + SECS
def worker():
    while time.time() < t_end:
        d = random.randint(LO, HI); t0 = time.time()
        try:
            u = post(int(d * cpt), 256)
            with lock: ok.append((u["prompt_tokens"], u["completion_tokens"], time.time() - t0))
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            with lock: (rej if e.code == 400 and "context" in body.lower() else err).append(f"HTTP {e.code}: {body}")
        except Exception as e:
            with lock: err.append(repr(e)[:300])
            time.sleep(2)
t0 = time.time(); ts = [threading.Thread(target=worker) for _ in range(CONC)]
[t.start() for t in ts]; [t.join() for t in ts]; w = time.time() - t0
pt = [x[0] for x in ok]
print(f"soak {CONC} clients {w:.0f} s: {len(ok)} ok, {len(rej)} context-length rejections, {len(err)} errors; "
      f"prompts {min(pt) if pt else 0}..{max(pt) if pt else 0} tok (median {int(st.median(pt)) if pt else 0}), "
      f"{sum(pt) / w:.0f} prompt tok/s, {sum(x[1] for x in ok) / w:.1f} output tok/s, "
      f"median latency {st.median([x[2] for x in ok]) if ok else 0:.1f} s")
for e in err[:5]: print("  ERROR", e)
sys.exit(1 if err else 0)
