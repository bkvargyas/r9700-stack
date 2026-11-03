#!/bin/bash
# zrest.sh: wait for the box (GPU-PROTOCOL: no ds41 / vllmstock container, no ~/gpu-exclusive), take the lock, then
# G4 (defaults, GPU-only trace for the wall split) and G2b (NSEQ 8 at KVMEM 1.0, the 48-token shape); cleanup.
T=~/ds41/exl3/Z/tools
while sudo docker ps --format "{{.Names}}" | grep -qE "^(ds41|vllmstock)$" || [ -e ~/gpu-exclusive ]; do sleep 15; done
echo "Z step-ledger profiling (G4, G2b) $(date -u +%H:%M)" > ~/gpu-exclusive
echo "[$(date -u +%H:%M:%S)] lock taken"
EAGER=0 ADAPTIVE=true NSEQ=4 bash $T/zrun2.sh G4 cuda 1:3 4:3
EAGER=0 ADAPTIVE=true NSEQ=8 KVMEM=1.0 bash $T/zrun2.sh G2b graph 8:3
sudo docker rm -f ds41 >/dev/null 2>&1
bash ~/gpu-offset.sh 0
grep -q "^Z " ~/gpu-exclusive 2>/dev/null && rm -f ~/gpu-exclusive
echo "[$(date -u +%H:%M:%S)] zrest done; server stopped, lock removed"
