#!/bin/bash
export PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/lib/llvm/bin:$PATH
cd /tmp
for so in "$@"; do
  echo "=== $so"
  rm -rf x; mkdir x; cp "$so" x/l.so
  (cd x && llvm-objdump --offloading l.so >/dev/null 2>&1)
  for img in x/l.so.*gfx1201*; do
    [ -e "$img" ] || { echo "(no gfx1201 image)"; continue; }
    llvm-readelf --notes "$img" | python3 -c '
import sys,re
t=sys.stdin.read()
for blk in re.split(r"\n\s+- \.args:",t)[1:]:
  if "hidden_hostcall_buffer" in blk:
    m=re.search(r"\.name:\s+(\S+)",blk); print(" ", (m.group(1) if m else "?")[:120])'
  done
done
