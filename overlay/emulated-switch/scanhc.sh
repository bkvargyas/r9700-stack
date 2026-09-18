#!/bin/bash
# Scan shared objects for device code objects whose kernels take hidden_hostcall_buffer.
export PATH=/opt/rocm/lib/llvm/bin:$PATH
roots="${@:-/opt/vllm/lib/python3.14/site-packages /app /opt/rocm/core-10.0/lib}"
tmp=$(mktemp -d)
find $roots -name "*.so*" -type f -size +100k 2>/dev/null | while read -r so; do
  readelf -S "$so" 2>/dev/null | grep -q hip_fatbin || continue
  d=$tmp/x; rm -rf $d; mkdir -p $d; cp "$so" $d/l.so
  ( cd $d && llvm-objdump --offloading l.so >/dev/null 2>&1 )
  for img in $d/l.so.*gfx1201*; do
    [ -e "$img" ] || continue
    n=$(llvm-readelf --notes "$img" 2>/dev/null | python3 -c '
import sys,re
t=sys.stdin.read()
ks=[]
for blk in re.split(r"\n\s+- \.args:",t)[1:]:
  if "hidden_hostcall_buffer" in blk:
    m=re.search(r"\.name:\s+(\S+)",blk); ks.append(m.group(1) if m else "?")
print(len(ks), " ".join(ks[:4]))')
    [ "${n%% *}" != "0" ] && echo "HOSTCALL $so :: $n"
  done
  nobits=$(readelf -S "$so" | grep -A1 hip_fatbin | grep -c NOBITS)
  [ "$nobits" -gt 0 ] && echo "KPACK(opaque) $so"
done
rm -rf $tmp
echo SCAN_DONE
