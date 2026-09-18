#!/bin/bash
# A/B: P2P on vs off, identical flags (MEM=60, default cudagraph ladder)
run() { tag=$1; shift
  env "$@" MEM=60 CG=default ~/serve-flashnext-gptq.sh
  t0=$(date +%s)
  for i in $(seq 1 90); do sleep 10
    curl -sf -m 3 localhost:8080/health >/dev/null && { echo "[$tag] HEALTHY after $(( $(date +%s)-t0 ))s"; break; }
    docker ps -q -f name=vllmflashnext | grep -q . || { echo "[$tag] DIED"; docker logs --tail 40 vllmflashnext; return; }
    docker logs vllmflashnext 2>&1 | grep -q "present state" && { echo "[$tag] ILLEGAL STATE"; return; }
  done
  docker logs vllmflashnext 2>&1 | grep -E "KV cache|resident|swap slots|Capturing|CUDAGraph memory|r4d|AR_QUANT" | cut -c1-220 | tail -12
  timeout 1500 python3 ~/bench.py $tag 2>&1 | tail -10
}
#run gptq-p2p-mem60 P2P=1
run gptq-nop2p-mem60 NOP2P=1
echo AB_DONE
