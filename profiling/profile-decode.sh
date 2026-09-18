#!/bin/bash
# Kernel-level profile of the standing Flash-Next config under rocprofv3 (sees kernels inside HIP graph replays).
# Marks three windows (single-stream decode, 8-concurrent decode, 8k prefill) with CLOCK_MONOTONIC/BOOTTIME
# stamps, then stops the server so rocprofv3 flushes, then runs analyze.py over the traces.
set -u
P=$HOME/fn-prof; sudo rm -rf $P/rp; mkdir -p $P
WRAP=rocprof XENV="VLLM_DISABLE_COMPILE_CACHE=1" ~/serve-flashnext-gptq.sh
docker update --restart no vllmflashnext >/dev/null
for i in $(seq 1 180); do sleep 10; curl -sf -m 3 localhost:8080/health >/dev/null && break
  docker ps -q -f name=vllmflashnext | grep -q . || { echo PROFILE_SERVER_DIED; exit 1; }; done
echo "healthy after ~$((i*10))s"
python3 - <<'EOF'
import json,time,urllib.request,threading
U="http://localhost:8080/v1/chat/completions"
def req(prompt,n,think=False):
    b={"model":"Qwen3.8","messages":[{"role":"user","content":prompt}],"max_tokens":n,"temperature":0,"ignore_eos":True,
       "chat_template_kwargs":{"enable_thinking":think}}
    return json.loads(urllib.request.urlopen(urllib.request.Request(U,json.dumps(b).encode(),{"Content-Type":"application/json"}),timeout=900).read())["usage"]
def stamp(): return {"mono":time.clock_gettime_ns(time.CLOCK_MONOTONIC),"boot":time.clock_gettime_ns(time.CLOCK_BOOTTIME),"wall":time.time_ns()}
W={}
req("Say hi.",64); req("Write a haiku about GPUs.",128)                       # warm
s=stamp(); u=req("Write a long essay about the history of the printing press.",600); W["decode1"]=[s,stamp(),u["completion_tokens"]]
th=[]; res=[]
s=stamp()
for k in range(8):
    t=threading.Thread(target=lambda k=k: res.append(req(f"Essay #{k}: write at length about the history of bridges.",300))); t.start(); th.append(t)
[t.join() for t in th]; W["decode8"]=[s,stamp(),sum(r["completion_tokens"] for r in res)]
long=" ".join(["The quick brown fox jumps over the lazy dog near the river bank."]*620)
s=stamp(); u=req("Summarize in one sentence: "+long,8); W["prefill8k"]=[s,stamp(),u["prompt_tokens"]]
json.dump(W,open("/home/devops/fn-prof/windows.json","w"),indent=1)
for k,(a,b,n) in W.items(): print(k, f"{(b['mono']-a['mono'])/1e9:.2f}s", n, "tokens")
EOF
# rocprofv3 is PID 1 and does NOT forward SIGTERM to its child (it just waits for children), so `docker stop`
# ends in SIGKILL and the trace is lost. Stop the vLLM server itself; rocprofv3 then finalizes and exits.
echo "stopping vLLM inside the container (rocprofv3 flushes on child exit)..."
docker exec vllmflashnext bash -c 'pkill -INT -f "bin/vllm serve" || pkill -INT -f "vllm serve"'
for i in $(seq 1 180); do docker ps -q -f name=vllmflashnext | grep -q . || break; sleep 10; done
docker ps -q -f name=vllmflashnext | grep -q . && { echo "container still up after 30 min; killing"; docker kill vllmflashnext; }
sleep 5; sudo chown -R $USER $P; ls -la $P/rp | head; du -sh $P/rp
python3 ~/analyze-prof.py $P > $P/report.txt 2>&1; cat $P/report.txt
echo PROFILE_DONE
