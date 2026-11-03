#!/bin/bash
# zall.sh: the package-Z launch sequence on VM100 (each launch = zrun.sh; traces under ~/ds41/exl3/Z/<tag>/s<streams>/).
# E1 eager + stacks/shapes (per-kernel attribution, launch list), G1 graphs (the ledger numbers), G3 graphs ADAPTIVE=false,
# G2 graphs NSEQ 8 (the 48-token shape). Ends with the server stopped, the lock removed and the clock offset back to 0.
T=~/ds41/exl3/Z/tools
# adaptive verification refuses eager mode (needs full graphs): the eager attribution launch runs ADAPTIVE=false
EAGER=1 ADAPTIVE=false NSEQ=4 bash $T/zrun.sh E1 eager 1:6 4:6
EAGER=0 ADAPTIVE=true NSEQ=4 bash $T/zrun.sh G1 graph 1:3 4:3
EAGER=0 ADAPTIVE=false NSEQ=4 bash $T/zrun.sh G3 graph 1:3 4:3
EAGER=0 ADAPTIVE=true NSEQ=8 bash $T/zrun.sh G2 graph 8:3 1:3
sudo docker rm -f ds41 >/dev/null 2>&1
bash ~/gpu-offset.sh 0
grep -q "^Z " ~/gpu-exclusive 2>/dev/null && rm -f ~/gpu-exclusive
echo "[$(date -u +%H:%M:%S)] zall done; server stopped, lock removed"
