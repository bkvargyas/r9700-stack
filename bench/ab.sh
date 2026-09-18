#!/bin/bash
# Interleaved A/B(/C...) benchmark: for each round, restart the server once per config (order rotates each round
# so drift and thermal/host-memory state do not favour one config), warm up, run harness.py, append JSON lines.
# usage: ROUNDS=3 REPS=5 bench/ab.sh "base:" "stage:R9K_STAGE_COLD=1" "nbt8k:NBT=8192 R9K_EXPERT_CACHE_SLOTS=240"
#   each config is "label:VAR=val VAR=val" -- env passed to serve/serve.sh (R9K_* knobs, NBT, MTP, OVERLAYS, ...).
# Output: bench/results/ab-<timestamp>.jsonl + a summary table (first config is the reference).
HERE=$(dirname "$(realpath "$0")")
ROUNDS=${ROUNDS:-3}; REPS=${REPS:-5}; BASE=${BASE:-http://localhost:8080}
OUT=${OUT:-$HERE/results/ab-$(date +%Y%m%d-%H%M%S).jsonl}; mkdir -p "$(dirname "$OUT")"
CFGS=("$@"); [ ${#CFGS[@]} -ge 1 ] || { sed -n 2,6p "$0"; exit 1; }
wait_ready() {
  for _ in $(seq 1 180); do
    curl -sf "$BASE/v1/models" >/dev/null && return 0
    docker ps --format '{{.Names}}' | grep -q '^vllmstock$' || { docker logs vllmstock 2>&1 | tail -20; return 1; }
    sleep 10
  done; return 1
}
for r in $(seq 0 $((ROUNDS - 1))); do
  n=${#CFGS[@]}
  for i in $(seq 0 $((n - 1))); do
    c=${CFGS[$(( (i + r) % n ))]}; label=${c%%:*}; envs=${c#*:}
    echo "=== round $((r + 1))/$ROUNDS  $label  ($envs)"
    env $envs bash "$HERE/../serve/serve.sh" >/dev/null || { echo "start failed: $label"; continue; }
    wait_ready || { echo "not ready: $label"; continue; }
    python3 "$HERE/warmup.py" "$BASE" >/dev/null
    python3 "$HERE/harness.py" run "$label" --base "$BASE" --reps "$REPS" --out "$OUT"
  done
done
echo "=== summary ($OUT)"
python3 "$HERE/harness.py" summarize "$OUT"
