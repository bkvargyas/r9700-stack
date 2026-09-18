#!/usr/bin/env python3
import sys, time, json, urllib.request, threading, statistics
BASE="http://localhost:8080/v1/chat/completions"; MODEL="Qwen3.8"
def call(prompt, max_tokens, no_think=False):
    body={"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0.0,"stream":False}
    if no_think: body["chat_template_kwargs"]={"enable_thinking":False}
    data=json.dumps(body).encode()
    t0=time.time()
    req=urllib.request.Request(BASE,data=data,headers={"Content-Type":"application/json"})
    r=json.loads(urllib.request.urlopen(req,timeout=300).read())
    dt=time.time()-t0
    u=r["usage"]; return u["completion_tokens"], u["prompt_tokens"], dt
def single_decode(n=3):
    xs=[]
    for _ in range(n):
        ct,pt,dt=call("Write a detailed 300-word essay about the history of computing.",300,no_think=True)
        xs.append(ct/dt)
    return statistics.mean(xs)
def aggregate(conc=8, max_tokens=200):
    res=[]; lock=threading.Lock()
    def worker(i):
        ct,pt,dt=call(f"Explain concept number {i}: describe tensor parallelism in depth.",max_tokens,no_think=True)
        with lock: res.append((ct,dt))
    t0=time.time(); ts=[threading.Thread(target=worker,args=(i,)) for i in range(conc)]
    [t.start() for t in ts]; [t.join() for t in ts]; wall=time.time()-t0
    total_tok=sum(c for c,_ in res)
    return total_tok/wall, len(res)
def prefill(ctx_words):
    prompt=("data "*ctx_words).strip()+"\nSummarize the above in 3 words."
    ct,pt,dt=call(prompt,8,no_think=True)
    return pt/dt, pt
if __name__=="__main__":
    tag=sys.argv[1] if len(sys.argv)>1 else "run"
    print(f"[{tag}] warming up..."); call("hi",5,no_think=True)
    sd=single_decode(); print(f"[{tag}] single-stream decode: {sd:.1f} tok/s")
    for c in (4,8,16):
        agg,n=aggregate(c); print(f"[{tag}] aggregate @{c} concurrent: {agg:.1f} tok/s ({n} ok)")
    for w in (500,2000,8000):
        pf,pt=prefill(w); print(f"[{tag}] prefill ~{pt} tok: {pf:.0f} tok/s")
