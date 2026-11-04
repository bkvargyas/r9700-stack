#!/bin/bash
# zana.sh DIR TAG...: the ledger analyses of package Z / Z2 (CPU only), outputs next to the traces under DIR/<tag>/.
# E*: ledger.py with stacks (dumps draft-names.json), --list, layers, mklist (the launch list); G4: gputl.py (GPU-only
# traces); other G*: ledger.py --ranks (cross-rank collective minima), --list, layers, classavg, widths.
D=$1; shift; T=$(dirname "$(realpath "$0")")
r0() { ls $1/*rank0.*.pt.trace.json 2>/dev/null | head -1; }
DN=$D/E1/draft-names.json; [ -f $DN ] || DN=$HOME/ds41/exl3/Z/E1/draft-names.json
for tag in "$@"; do for s in s1 s4; do
  x=$D/$tag/$s; f=$(r0 $x); [ -n "$f" ] || continue
  case $tag in
    E*) python3 $T/ledger.py $f --label $tag-$s --json $x-ledger.json --list $x-list.txt --dump-draft-names $D/$tag/draft-names-$s.json > $x-ledger.txt
        [ $s = s1 ] && cp $D/$tag/draft-names-s1.json $D/$tag/draft-names.json
        python3 $T/layers.py $x-list.txt $x-layers.txt
        [ $s = s1 ] && python3 $T/mklist.py $x-list.txt $D/$tag/launch-list-T6.txt ;;
    G4) python3 $T/gputl.py $f $tag-$s $DN > $x-gputl.txt
        python3 $T/widths.py $f $DN > $x-widths.txt ;;
    *)  python3 $T/ledger.py $f --label $tag-$s --ranks $x --json $x-ledger.json --list $x-list.txt --draft-names $DN > $x-ledger.txt
        python3 $T/layers.py $x-list.txt $x-layers.txt; python3 $T/classavg.py $x-list.txt > $x-classavg.txt
        python3 $T/widths.py $f $DN > $x-widths.txt ;;
  esac
  echo "[$(date -u +%H:%M:%S)] analysed $tag/$s"
done; done
