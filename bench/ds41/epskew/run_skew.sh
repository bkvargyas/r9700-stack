#!/bin/bash
# run_skew.sh TAG PHASE [PHASE ...]   (VM100; package H3, notes/exl3-ep-skew.md)
# One launch of the DeepSeek EXL3 server "ds41" from a clean tree (H3TREE, default ~/ds41/exl3/H3/t = git archive v0.4.1 +
# package H3's files) with the routing recorder (bench/ds41/epskew/epskew_ext.py as the worker extension r9k_epskew, dev
# endpoint on) and the GPU-only torch profiler configured (started only by a prof phase); then run_skew.py's phases;
# output under ~/ds41/exl3/H3/TAG/ (prof-s<N>.pt + prof-s<N>/ rank traces, hist-*.pt, launch.log, run.log).
# KEEPLOCK=1: keep ~/gpu-exclusive (and the offset) for a following H3 launch (an A/B pair in one window).
# Caller env passes through to the launcher (ADAPTIVE, NSEQ, R9K_* knobs via DOCKER_ARGS_EXTRA="-e K=V ...").
# GPU-PROTOCOL: waits until the box is free, holds ~/gpu-exclusive, offset -500 before and 0 after, stops its server.
TAG=$1; shift
H=$HOME/ds41/exl3/H3; T=${H3TREE:-$H/t}; D=$H/$TAG; mkdir -p $D/prof
log() { echo "[$(date -u "+%H:%M:%S")] $*" | tee -a $D/run.log; }
busy() { sudo docker ps --format "{{.Names}}" | grep -qE "^(ds41|vllmstock)$" && return 0; [ -e ~/gpu-exclusive ] && return 0; return 1; }
mine() { grep -q "^H3 " ~/gpu-exclusive 2>/dev/null && ! sudo docker ps --format "{{.Names}}" | grep -qE "^(ds41|vllmstock)$"; }
while busy && ! mine; do sleep 60; done
echo "H3 EP-skew $TAG $(date -u +%H:%M)" > ~/gpu-exclusive
log "lock taken"
bash ~/gpu-offset.sh -500 | tee -a $D/run.log
PCFG="--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"/prof\",\"torch_profiler_activities\":[\"CUDA\"],\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":false,\"torch_profiler_use_gzip\":false,\"torch_profiler_dump_cuda_time_total\":false,\"ignore_frontend\":true}"
WEXT=; [ "${RECORD:-1}" = 1 ] && WEXT="--worker-extension-cls r9k_epskew.EpSkew"
cd $T || exit 1
t0=$(date +%s)
log "launch $TAG: ADAPTIVE=${ADAPTIVE-true} NSEQ=${NSEQ-4} RECORD=${RECORD-1} extra docker args: ${DOCKER_ARGS_EXTRA-}"
REPO=$T VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 NAME=ds41 WARM=0 \
  DOCKER_ARGS="-v $D/prof:/prof -v $D:/h3out -e VLLM_SERVER_DEV_MODE=1 -v $T/bench/ds41/epskew/epskew_ext.py:/usr/local/lib/python3.12/dist-packages/r9k_epskew.py:ro ${DOCKER_ARGS_EXTRA-}" \
  EXTRA="$PCFG $WEXT ${EXTRA-}" bash serve/ds41-exl3.sh 2>&1 | tail -3 | tee -a $D/run.log
for i in $(seq 1 150); do
  sudo docker logs ds41 2>&1 | grep -qa "Application startup complete" && break
  sudo docker ps --format "{{.Names}}" | grep -q "^ds41$" || break
  sleep 6
done
sudo docker logs ds41 > $D/launch.log 2>&1
if grep -qa "Application startup complete" $D/launch.log; then
  python3 $T/bench/ds41/ledger/warm.py 8080 DeepSeek-V4.1 2>&1 | tail -1 | tee -a $D/run.log
  log "ready + warm after $(( $(date +%s) - t0 )) s"
  grep -aE "Model loading took|GPU KV cache size|EP_MAP|exl3 placement" $D/launch.log | grep -avE "Worker_TP[1-7]|VllmWorker-[1-7]" | tail -6 | cut -c1-240 | tee -a $D/run.log
  for ph in "$@"; do
    case $ph in
      prof:*) ns=$(echo $ph | cut -d: -f2)
              python3 $T/bench/ds41/epskew/run_skew.py http://localhost:8080 $D $ph 2>&1 | tee -a $D/run.log
              sleep 5; mkdir -p $D/prof-s$ns
              for i in $(seq 1 60); do a=$(du -sb $D/prof | cut -f1); sleep 3; b=$(du -sb $D/prof | cut -f1); [ "$a" = "$b" ] && break; done
              sudo mv $D/prof/* $D/prof-s$ns/ 2>/dev/null ;;
      spectune:*) python3 $T/bench/ds41/spectune.py http://localhost:8080 DeepSeek-V4.1 $TAG $(echo $ph | cut -d: -f2) $(echo $ph | cut -d: -f3) 2>&1 | tee -a $D/spectune.log | grep -aE "^(prose|code)|RESULT" | cut -c1-260 | tee -a $D/run.log ;;
      div)    python3 $T/bench/ds41/divergence.py http://localhost:8080 DeepSeek-V4.1 24 2>&1 | tee -a $D/div.log | grep -a "===" | tee -a $D/run.log ;;
      gsm:*)  python3 $T/bench/ds41/gsm8k_ds41.py $(echo $ph | cut -d: -f2) 8 DeepSeek-V4.1 2>&1 | tee -a $D/gsm.log | tail -2 | tee -a $D/run.log ;;
      work:*) ns=$(echo $ph | cut -d: -f2); python3 $T/bench/ds41/ledger/workload.py http://localhost:8080 DeepSeek-V4.1 $ns 6 $D/work-s$ns-$(date +%s).json 0 2>&1 | tail -3 | tee -a $D/run.log ;;
      *)      python3 $T/bench/ds41/epskew/run_skew.py http://localhost:8080 $D $ph 2>&1 | tee -a $D/run.log ;;
    esac
  done
else
  log "NOT READY"; grep -aE "Error|Traceback" $D/launch.log | tail -8 | tee -a $D/run.log
fi
sudo docker logs ds41 > $D/server.log 2>&1
sudo docker rm -f ds41 >/dev/null 2>&1
sudo chown -R devops $D
if [ "${KEEPLOCK:-0}" = 1 ]; then log "done $TAG; server stopped, lock KEPT for the next H3 launch"; exit 0; fi
bash ~/gpu-offset.sh 0 | tee -a $D/run.log
grep -q "^H3 " ~/gpu-exclusive 2>/dev/null && rm -f ~/gpu-exclusive
log "done $TAG; server stopped, offset 0, lock removed"
