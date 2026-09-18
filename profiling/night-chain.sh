#!/bin/bash
# After the ablation finishes: kernel profile, then restore the standing serve config.
until grep -q ABLATION_DONE ~/abl/run.log 2>/dev/null; do sleep 30; done
~/profile-decode.sh > ~/fn-prof.log 2>&1
~/serve-flashnext-gptq.sh > ~/restore.log 2>&1
for i in $(seq 1 60); do sleep 10; curl -sf -m 3 localhost:8080/health >/dev/null && { echo "RESTORED healthy $(date +%T)" >> ~/restore.log; break; }; done
echo CHAIN_DONE >> ~/restore.log
