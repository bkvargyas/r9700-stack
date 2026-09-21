#!/usr/bin/env python3
import os
import json, re, sys, time, urllib.request, concurrent.futures as cf
BASE=os.environ.get("PROD_BASE", "http://localhost:8000") + "/v1/chat/completions"; MODEL="Qwen3.8"
N=int(sys.argv[1]) if len(sys.argv)>1 else 200
CONC=int(sys.argv[2]) if len(sys.argv)>2 else 8
def load_gsm8k():
    url="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    data=urllib.request.urlopen(url,timeout=60).read().decode().splitlines()
    out=[]
    for line in data:
        d=json.loads(line); gt=d["answer"].split("####")[-1].strip().replace(",","")
        out.append((d["question"], gt))
    return out
def last_number(text):
    if not text: return None
    nums=re.findall(r"-?\$?\d[\d,]*\.?\d*", text.replace(",",""))
    if not nums: return None
    return nums[-1].replace("$","").rstrip(".")
def ask(q):
    body={"model":MODEL,"messages":[{"role":"user","content":q+"\nGive the final numeric answer."}],
          "max_tokens":4096,"temperature":0.0,"chat_template_kwargs":{"enable_thinking":True,"reasoning_effort":"medium"}}
    req=urllib.request.Request(BASE,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    r=json.loads(urllib.request.urlopen(req,timeout=300).read())
    m=r["choices"][0]["message"]; c=m.get("content") or ""; rz=m.get("reasoning") or ""
    fin=r["choices"][0].get("finish_reason")
    return last_number(c) or last_number(rz), fin
def norm(x):
    try: return float(x)
    except: return None
data=load_gsm8k()[:N]
print(f"GSM8K: {len(data)} questions, conc={CONC}, thinking=on", flush=True)
correct=0; done=0; truncated=0; t0=time.time()
def work(item):
    q,gt=item
    try:
        pred,fin=ask(q); ok = norm(pred) is not None and norm(gt) is not None and abs(norm(pred)-norm(gt))<1e-4
        return ok, fin=="length"
    except Exception as e:
        return False, False
with cf.ThreadPoolExecutor(max_workers=CONC) as ex:
    for ok,trunc in ex.map(work, data):
        done+=1; correct+=ok; truncated+=trunc
        if done%20==0: print(f"  {done}/{len(data)}  acc={100*correct/done:.1f}%  (trunc {truncated})", flush=True)
dt=time.time()-t0
print(f"\nGSM8K accuracy: {correct}/{len(data)} = {100*correct/len(data):.2f}%  | truncated(length) {truncated} | {dt/60:.1f} min", flush=True)
