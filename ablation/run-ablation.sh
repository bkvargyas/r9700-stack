#!/bin/bash
# Kernel ablation for tcclaviger/vllm:dev Flash-Next GPTQ (TP2, P2P, auto expert offload, cudagraph [4]).
# Each variant = standing config + XENV toggles that disable one private kernel family (falls back to
# the image's generic Triton/torch path). Compile cache disabled so a cached graph can't mask a toggle.
# Usage: run-ablation.sh [tag ...]   (default: all).  Results: ~/abl/<tag>/, summary ~/abl/summary.tsv
set -u
OUT=$HOME/abl; mkdir -p $OUT
BASEENV="VLLM_DISABLE_COMPILE_CACHE=1"
declare -A V=(
  [base]=""
  [no_clav_attn]="CLAV_ATTN=0"
  [no_gdn_hip]="NO_AMD_GDN_HIP=1"
  [no_clav_helpers]="CLAV_HC=0 CLAV_CONV1D=0 CLAV_PLECONV=0 CLAV_SILU_QUANT=0 CLAV_MEMCPY=0 CLAV_STATE_COPY=0 CLAV_RESHAPE_CACHE=0"
  [no_fp8hip]="VLLM_DISABLE_FP8HIP=1"
  [no_rdna4_fp8]="VLLM_DISABLE_RDNA4_FP8_KERNEL=1"
  [no_r4d_mxfp4]="VLLM_DISABLE_R4D_MXFP4=1"
  [no_r4d_qsa]="R4D_QSA=0 R4D_QSA_PREP=0"
  [no_r4d_ple]="R4D_PLE=0"
  [no_r4d_select]="R4D_SELECT=0"
  [no_ar_quant]="R4D_AR_QUANT=0"
  [no_all_clav]="DISABLE_ALL_CLAV=1"
)
ORDER=(base no_clav_attn no_gdn_hip no_clav_helpers no_fp8hip no_rdna4_fp8 no_r4d_qsa no_r4d_ple no_r4d_select no_ar_quant no_r4d_mxfp4 no_all_clav)
[ $# -gt 0 ] && ORDER=("$@")
[ -f $OUT/summary.tsv ] || printf "tag\tstatus\tstartup_s\tchecks\tsingle\tagg4\tagg8\tagg16\tpf523\tpf2k\tpf8k\txenv\n" > $OUT/summary.tsv

check() { # known-answer prompts, temp 0, no thinking
python3 - <<'EOF'
import json,urllib.request
Q=[("What is 347*29? Reply with only the number.","10063"),
   ("What is the capital of Australia? One word.","Canberra"),
   ("Spell the word strawberry backwards. Reply with only the result.","yrrebwarts"),
   ("Sort these numbers ascending, comma separated, nothing else: 42, 7, 19, 3, 88","3, 7, 19, 42, 88"),
   ("Write a Python function fib(n) returning the n-th Fibonacci number iteratively. Code only.","a, b = b, a + b"),
   ("How many r's are in 'strawberry'? Reply with only the number.","3")]
ok=0
for q,a in Q:
    b={"model":"Qwen3.8","messages":[{"role":"user","content":q}],"max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
    try:
        r=json.loads(urllib.request.urlopen(urllib.request.Request("http://localhost:8080/v1/chat/completions",json.dumps(b).encode(),{"Content-Type":"application/json"}),timeout=120).read())
        c=r["choices"][0]["message"]["content"] or ""
    except Exception as e: c=f"ERR {e}"
    hit=a.replace(" ","").lower() in c.replace(" ","").lower(); ok+=hit
    print(("PASS" if hit else "FAIL"), repr(q[:40]), "->", repr(c[:160]))
print(f"CHECKS {ok}/{len(Q)}")
EOF
}

for tag in "${ORDER[@]}"; do
  D=$OUT/$tag; rm -rf $D; mkdir -p $D
  XENV="$BASEENV ${V[$tag]}"; echo "=== $(date +%T) $tag :: $XENV" | tee -a $OUT/run.log
  XENV="$XENV" ~/serve-flashnext-gptq.sh >> $OUT/run.log 2>&1
  docker update --restart no vllmflashnext >/dev/null   # a crash = fail, not a loop
  t0=$(date +%s); status=TIMEOUT
  for i in $(seq 1 150); do sleep 10
    curl -sf -m 3 localhost:8080/health >/dev/null && { status=OK; break; }
    docker ps -q -f name=vllmflashnext | grep -q . || { status=DIED; break; }
  done
  st=$(( $(date +%s)-t0 ))
  if [ $status = OK ]; then
    check > $D/check.txt 2>&1
    timeout 1500 python3 ~/bench.py $tag > $D/bench.txt 2>&1
  fi
  docker logs vllmflashnext > $D/log.txt 2>&1; gzip -f $D/log.txt
  num() { grep -m1 "$1" $D/bench.txt 2>/dev/null | grep -oE "[0-9.]+ tok/s" | head -1 | cut -d" " -f1; }
  chk=$(grep -oE "CHECKS [0-9]+/[0-9]+" $D/check.txt 2>/dev/null | cut -d" " -f2)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $tag $status $st "${chk:--}" "$(num single-stream)" "$(num '@4 ')" "$(num '@8 ')" "$(num '@16 ')" "$(num '~523')" "$(num '~2023')" "$(num '~8023')" "${V[$tag]:-none}" | tee -a $OUT/summary.tsv
done
echo "=== $(date +%T) ABLATION_DONE" | tee -a $OUT/run.log
