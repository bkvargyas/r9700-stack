#!/bin/bash
# TP=4 (all four cards, RCCL all-reduce) vs two TP=2 servers (one per PLX switch), Qwen3.8-27B-NVFP4, serve/27b.sh.
# 2026-09-23. Phases run one after another; only phase 4 loads two servers at once, which IS the dual deployment.
set -u
R=~/r9700-build/repo; BB=~/bb-venv/bin/betterbench; OUT=~/tp4v2; mkdir -p $OUT
trap 'docker rm -f tp4 tp2a tp2b >/dev/null 2>&1' EXIT
docker ps --format "{{.Names}}" | grep -q . && { echo BUSY; exit 1; }
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $OUT/progress.log; }
up(){ # name port -> wait until serving
  for i in $(seq 1 120); do curl -sf localhost:$2/v1/models >/dev/null && { log "$1 READY"; python3 $R/bench/warmup.py http://localhost:$2 >/dev/null 2>&1; return 0; }
    docker ps --format "{{.Names}}" | grep -qx $1 || { log "$1 DIED"; docker logs --tail 60 $1 > $OUT/$1-died.log 2>&1; return 1; }; sleep 10; done
  log "$1 TIMEOUT"; return 1; }
bb(){ # port runname config [phase flags]
  local port=$1 name=$2 cfg=$3; shift 3
  $BB run --endpoint http://localhost:$port/v1 --model Qwen3.8 --name $name --no-update-check --no-html \
    --config $cfg --out $OUT/$name.json "$@" > $OUT/$name.log 2>&1; log "bb $name exit $?"; }
cfg(){ echo "{\"concurrency_levels\": [$1]}" > $OUT/cfg-$2.json; echo $OUT/cfg-$2.json; }
serve(){ # name gpus tp port nseq
  NAME=$1 GPUS=$2 TP=$3 PORT=$4 bash $R/serve/27b.sh NSEQ=$5 > $OUT/$1-launch.log 2>&1; }

log "phase 1: TP=4"
serve tp4 0,1,2,3 4 8080 16 && up tp4 8080 && {
  docker logs tp4 2>&1 | grep -iE "all-reduce|P2P" | sort -u | head -5 > $OUT/tp4-ar.txt
  bb 8080 tp4-full "$(cfg 1,2,4,8,16 tp4)"; }
docker logs tp4 > $OUT/tp4-server.log 2>&1; docker rm -f tp4 >/dev/null 2>&1

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
