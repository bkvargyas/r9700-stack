#!/bin/bash
# Cap every R9700 in the guest (default 225 W). Waits for amdgpu to expose all of them.
CAP=${1:-225000000}
want=$(lspci -d 1002: | grep -c "R9700")
for i in $(seq 1 60); do
  files=$(ls /sys/class/drm/card*/device/hwmon/hwmon*/power1_cap 2>/dev/null)
  [ "$(echo "$files" | grep -c .)" -ge "$want" ] && break
  sleep 2
done
for f in $files; do
  echo "$CAP" > "$f" || echo "write failed: $f"
  echo "$f -> $(cat "$f") (min $(cat ${f}_min 2>/dev/null) max $(cat ${f}_max 2>/dev/null))"
done
