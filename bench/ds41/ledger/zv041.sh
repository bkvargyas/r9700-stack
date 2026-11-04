#!/bin/bash
# zv041.sh: package Z2's launch sequence on VM100 -- the step ledger of v0.4.1 (git archive v0.4.1 -> ~/r9700-exl3-Z2,
# traces under ~/ds41/exl3/Z2/<tag>/s<streams>/). Waits for the box (GPU-PROTOCOL: no ds41 / vllmstock container, no
# ~/gpu-exclusive), holds the lock for the whole sequence, then E1 (eager + stacks / shapes, ADAPTIVE=false: names, call
# sites, the launch list), G1 (defaults, CPU+GPU profiler: phases, cross-rank collective minima), G3 (ADAPTIVE=false: the
# true 6- and 24-row verify), G4 (defaults, GPU-only: the headline table and the wall split). Ends with the server
# stopped, the clock offset back to 0 and the lock removed.
export ZTREE=$HOME/r9700-exl3-Z2 ZOUT=$HOME/ds41/exl3/Z2 ZLOCK=Z2
T=$ZOUT/tools
busy() { sudo docker ps --format "{{.Names}}" | grep -qE "^(ds41|vllmstock)$" && return 0; [ -e ~/gpu-exclusive ] && return 0; return 1; }
while busy; do sleep 60; done
echo "Z2 step-ledger profiling v0.4.1 (E1 G1 G3 G4) $(date -u +%H:%M)" > ~/gpu-exclusive
echo "[$(date -u +%H:%M:%S)] lock taken"
for L in "G4 cuda ADAPTIVE=true" "G3 graph ADAPTIVE=false" "G1 graph ADAPTIVE=true" "E1 eager ADAPTIVE=false"; do
  set -- $L
  if [ $2 = eager ]; then env EAGER=1 $3 NSEQ=4 bash $T/zrun2.sh $1 $2 1:6 4:6; else env EAGER=0 $3 NSEQ=4 bash $T/zrun2.sh $1 $2 1:3 4:3; fi
done
sudo docker rm -f ds41 >/dev/null 2>&1
bash ~/gpu-offset.sh 0
grep -q "^Z2 " ~/gpu-exclusive 2>/dev/null && rm -f ~/gpu-exclusive
echo "[$(date -u +%H:%M:%S)] zv041 done; server stopped, offset 0, lock removed"
