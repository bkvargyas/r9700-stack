#!/usr/bin/env python3
import json, re, sys, gzip, io, urllib.request, subprocess, tempfile, os, time
import concurrent.futures as cf
BASE="http://localhost:8080/v1/chat/completions"; MODEL="Qwen3.8"
TAG=sys.argv[1] if len(sys.argv)>1 else "run"
N=int(sys.argv[2]) if len(sys.argv)>2 else 164
CONC=8
def load():
    url="https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    raw=urllib.request.urlopen(url,timeout=60).read()
    return [json.loads(l) for l in gzip.decompress(raw).decode().splitlines()]
def gen(prompt):
    msg=("Complete the following Python function. Reply with the complete function "
         "in a single ```python code block, and nothing else.\n\n"+prompt)
    body={"model":MODEL,"messages":[{"role":"user","content":msg}],"max_tokens":3072,
          "temperature":0.0,"chat_template_kwargs":{"enable_thinking":True,"reasoning_effort":"medium"}}
    req=urllib.request.Request(BASE,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    r=json.loads(urllib.request.urlopen(req,timeout=300).read())
    m=r["choices"][0]["message"]; return (m.get("content") or "") or (m.get("reasoning") or "")
def extract(text, entry):
    blocks=re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    for b in reversed(blocks):
        if f"def {entry}" in b: return b
    if blocks: return blocks[-1]
    i=text.find(f"def {entry}")
    return text[i:] if i>=0 else text
def run_one(item):
    code=extract(gen(item["prompt"]), item["entry_point"])
    prog=code+"\n"+item["test"]+f"\ncheck({item['entry_point']})\n"
    try:
        with tempfile.NamedTemporaryFile("w",suffix=".py",delete=False) as f:
            f.write(prog); path=f.name
        p=subprocess.run([sys.executable,path],capture_output=True,timeout=20,
                         env={**os.environ,"OPENBLAS_NUM_THREADS":"1"})
        ok=(p.returncode==0)
    except Exception:
        ok=False
    finally:
        try: os.unlink(path)
        except: pass
    return item["task_id"], ok
data=load()[:N]
print(f"[{TAG}] HumanEval: {len(data)} problems, greedy, thinking=on", flush=True)
passed=0; done=0; t0=time.time(); fails=[]
with cf.ThreadPoolExecutor(max_workers=CONC) as ex:
    for tid,ok in ex.map(run_one, data):
        done+=1; passed+=ok
        if not ok: fails.append(tid)
        if done%20==0: print(f"  {done}/{len(data)}  pass@1={100*passed/done:.1f}%", flush=True)
print(f"\n[{TAG}] HumanEval pass@1: {passed}/{len(data)} = {100*passed/len(data):.2f}%  | {(time.time()-t0)/60:.1f} min", flush=True)
print(f"[{TAG}] failed: {','.join(sorted(fails))}", flush=True)
