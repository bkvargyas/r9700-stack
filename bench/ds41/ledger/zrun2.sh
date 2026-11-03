#!/bin/bash
# zrun.sh TAG PROFCFG STREAMS:WINDOW [STREAMS:WINDOW ...]   (VM100; package Z, notes/exl3-ledger.md)
# One ledger launch: relaunch the DeepSeek server "ds41" from the pristine v0.4.0 copy ~/r9700-exl3-Z with the caller's env
# (EAGER, NSEQ, ADAPTIVE, KVMEM, ...), torch profiler on (PROFCFG = graph | eager: eager adds Python stacks + shapes |
# cuda: GPU activity only, for the wall-clock split without the profiler's CPU overhead),
# VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 (stock's "gpu_model_runner: forward / sample / draft / ..." annotations), WARM=1;
# then for each STREAMS:WINDOW run workload.py (timing window, then profile window) and file the rank traces under
# ~/ds41/exl3/Z/TAG/sN/. Holds ~/gpu-exclusive (created if absent). Leaves the server up (the caller stops it).
TAG=$1; PC=$2; shift 2
Z=~/ds41/exl3/Z; D=$Z/$TAG; mkdir -p $D/prof
log() { echo "[$(date -u "+%H:%M:%S")] $*" | tee -a $D/run.log; }
if [ -e ~/gpu-exclusive ] && ! grep -q "^Z " ~/gpu-exclusive; then log "lock held by someone else: $(cat ~/gpu-exclusive)"; exit 1; fi
[ -e ~/gpu-exclusive ] || echo "Z step-ledger profiling $(date -u +%H:%M)" > ~/gpu-exclusive
bash ~/gpu-offset.sh -500 | tee -a $D/run.log
if [ "$PC" = eager ]; then ST=true; SH=true; else ST=false; SH=false; fi
ACT=; [ "$PC" = cuda ] && ACT='"torch_profiler_activities":["CUDA"],'   # GPU activity only: no CPU-side profiler overhead
PCFG="--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"/prof\",$ACT\"torch_profiler_with_stack\":$ST,\"torch_profiler_record_shapes\":$SH,\"torch_profiler_use_gzip\":false,\"torch_profiler_dump_cuda_time_total\":false,\"ignore_frontend\":true}"
sudo docker rm -f ds41 >/dev/null 2>&1
log "launch $TAG: EAGER=${EAGER-0} NSEQ=${NSEQ-4} ADAPTIVE=${ADAPTIVE-true} KVMEM=${KVMEM-1.7} DSPARK_K=${DSPARK_K-5} prof=$PC"
cd ~/r9700-exl3-Z || exit 1
t0=$(date +%s)
VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 KVMEM=${KVMEM-1.7} NAME=ds41 WARM=0 DOCKER_ARGS="-v $D/prof:/prof" EXTRA="$PCFG ${EXTRA-}" \
  bash serve/ds41-exl3.sh 2>&1 | tail -3 | tee -a $D/run.log
for i in $(seq 1 150); do
  sudo docker logs ds41 2>&1 | grep -qa "Application startup complete" && break
  sudo docker ps --format "{{.Names}}" | grep -q "^ds41$" || break
  sleep 6
done
sudo docker logs ds41 > $D/launch.log 2>&1
grep -qa "Application startup complete" $D/launch.log || { log "NOT READY"; grep -aE "Error|Traceback" $D/launch.log | tail -5 | tee -a $D/run.log; sudo docker rm -f ds41 >/dev/null 2>&1; exit 1; }
python3 $Z/tools/warm.py 8080 DeepSeek-V4.1 2>&1 | tail -1 | tee -a $D/run.log
log "ready + warm after $(( $(date +%s) - t0 )) s"
grep -aE "registered:|Model loading took|GPU KV cache size|Graph capturing finished|mhc|capture" $D/launch.log | grep -avE "Worker_TP[1-7]|VllmWorker-[1-7]" | tail -8 | cut -c1-220 | tee -a $D/run.log
for w in "$@"; do
  ns=${w%%:*}; win=${w##*:}
  log "workload streams=$ns window=$win"
  python3 $Z/tools/workload.py http://localhost:8080 DeepSeek-V4.1 $ns $win $D/s$ns.json 1 2>&1 | tee -a $D/run.log
  sleep 5; mkdir -p $D/s$ns
  # wait for the trace writers to finish (sizes stable)
  for i in $(seq 1 60); do a=$(du -sb $D/prof | cut -f1); sleep 3; b=$(du -sb $D/prof | cut -f1); [ "$a" = "$b" ] && break; done
  sudo mv $D/prof/* $D/s$ns/ 2>/dev/null; sudo chown -R devops $D/s$ns
  ls -la $D/s$ns | tail -n +2 | awk '{s+=$5} END {print NR" files, "s/1e6" MB"}' | tee -a $D/run.log
done
log "done $TAG"
