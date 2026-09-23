#!/bin/bash
# TP=4 (all four cards, RCCL all-reduce) vs two TP=2 servers (one per PLX switch).
# LAUNCH=27b.sh (Qwen3.8-27B-NVFP4, default) or flashnext.sh; OUT=results dir; TP4_ARGS / TP4_FALLBACK = extra launcher
# args for TP=4 (Flash-Next: try all experts in VRAM, OFFLOAD_GB=0, fall back to offload if it does not start).
# 2026-09-23. Phases run one after another; only phase 4 loads two servers at once, which IS the dual deployment.
set -u
R=~/r9700-build/repo; BB=~/bb-venv/bin/betterbench; LAUNCH=${LAUNCH:-27b.sh}; OUT=${OUT:-~/tp4v2}; mkdir -p $OUT
TP4_ARGS=${TP4_ARGS:-}; TP4_FALLBACK=${TP4_FALLBACK:-}; SKIP_TP4=${SKIP_TP4:-0}   # 1 = only the TP=2 phases
ONLY_TP4=${ONLY_TP4:-0}   # 1 = only the TP=4 phase
trap 'docker rm -f tp4 tp2a tp2b >/dev/null 2>&1' EXIT
docker ps --format "{{.Names}}" | grep -q . && { echo BUSY; exit 1; }
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $OUT/progress.log; }
up(){ # name port -> wait until serving
  for i in $(seq 1 120); do curl -sf localhost:$2/v1/models >/dev/null && { log "$1 READY"; python3 $R/bench/warmup.py http://localhost:$2 >/dev/null 2>&1; return 0; }
    docker ps --format "{{.Names}}" | grep -qx $1 || { log "$1 DIED"; docker logs $1 > $OUT/$1-died.log 2>&1; return 1; }; sleep 10; done
  log "$1 TIMEOUT"; return 1; }
bb(){ # port runname config [phase flags]
  local port=$1 name=$2 cfg=$3; shift 3
  $BB run --endpoint http://localhost:$port/v1 --model Qwen3.8 --name $name --no-update-check --no-html \
    --config $cfg --out $OUT/$name.json "$@" > $OUT/$name.log 2>&1; log "bb $name exit $?"; }
cfg(){ echo "{\"concurrency_levels\": [$1]}" > $OUT/cfg-$2.json; echo $OUT/cfg-$2.json; }
serve(){ # name gpus tp port nseq [launcher args...] (27b.sh sets NSEQ itself, so it goes in as an argument)
  local n=$1 g=$2 t=$3 p=$4 q=$5; shift 5
  NAME=$n GPUS=$g TP=$t PORT=$p bash $R/serve/$LAUNCH NSEQ=$q "$@" > $OUT/$n-launch.log 2>&1; }

if [ "$SKIP_TP4" != 1 ]; then
log "phase 1: TP=4"
log "launcher $LAUNCH, TP=4 args: ${TP4_ARGS:-none}"
if ! { serve tp4 0,1,2,3 4 8080 16 $TP4_ARGS && up tp4 8080; } && [ -n "$TP4_FALLBACK" ]; then
  log "TP=4 retry with fallback args: $TP4_FALLBACK"; mv $OUT/tp4-died.log $OUT/tp4-died-first.log 2>/dev/null
  docker rm -f tp4 >/dev/null 2>&1; serve tp4 0,1,2,3 4 8080 16 $TP4_FALLBACK && up tp4 8080
fi
docker ps --format "{{.Names}}" | grep -qx tp4 && {
  docker logs tp4 2>&1 | grep -iE "all-reduce|P2P" | sort -u | head -5 > $OUT/tp4-ar.txt
  bb 8080 tp4-full "$(cfg 1,2,4,8,16 tp4)"; }
docker logs tp4 > $OUT/tp4-server.log 2>&1; docker rm -f tp4 >/dev/null 2>&1

fi

[ "$ONLY_TP4" = 1 ] && { log "TP4VS2 DONE (TP=4 only)"; exit 0; }
log "phase 2+3: TP=2 on each switch, benchmarked alone"
serve tp2a 0,1 2 8080 8 && up tp2a 8080 && serve tp2b 2,3 2 8081 8 && up tp2b 8081 || exit 1
for s in tp2a:8080 tp2b:8081; do
  docker logs ${s%:*} 2>&1 | grep -iE "all-reduce|P2P" | sort -u | head -5 > $OUT/${s%:*}-ar.txt
done
bb 8080 tp2a-full "$(cfg 1,2,4,8 tp2)"
bb 8081 tp2b-full "$(cfg 1,2,4,8 tp2)"

log "phase 4: both TP=2 loaded together, same level on each (total = 2x)"
for L in 1 2 4 8; do
  c=$(cfg $L dual$L)
  bb 8080 dual-a-c$L $c --concurrency & bb 8081 dual-b-c$L $c --concurrency & wait
done
docker logs tp2a > $OUT/tp2a-server.log 2>&1; docker logs tp2b > $OUT/tp2b-server.log 2>&1
log "TP4VS2 DONE"
