#!/usr/bin/env bash
# verify-nohostcall.sh — acceptance test for a rebuilt RCCL.
#
# Usage: ./verify-nohostcall.sh /path/to/librccl.so.1[.0]
#
# PASS requires all of:
#   hidden_hostcall_buffer == 0   (the whole point)
#   __assert_fail          == 0   (device asserts compiled out)
#   __ockl_fprintf         == 0   (no device printf / COLLTRACE)
#   ncclCommDump exported         (torch >= 2.x refuses to load RCCL without it)
#
# Reference: AMD's own shipped ROCm 7.1.1 RCCL scores 0 / 0 / 0.

set -uo pipefail
LIB="${1:-}"
[ -z "$LIB" ] && { echo "usage: $0 /path/to/librccl.so.1"; exit 2; }
[ -f "$LIB" ] || { echo "not a file: $LIB"; exit 2; }

RE=$(command -v llvm-readelf || true)
OD=$(command -v llvm-objdump || true)
if [ -z "$RE" ] || [ -z "$OD" ]; then
  echo "need llvm-readelf and llvm-objdump on PATH (ROCm's llvm/bin)."
  echo "e.g. export PATH=\$ROCM_PATH/lib/llvm/bin:\$PATH"
  exit 2
fi

echo "inspecting: $LIB"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cp "$LIB" "$tmp/lib.so"
( cd "$tmp" && "$OD" --offloading lib.so >/dev/null 2>&1 || true )

fail=0
found=0
for dev in "$tmp"/lib.so.*gfx*; do
  [ -e "$dev" ] || continue
  found=1
  arch=$(basename "$dev" | sed 's/.*\.//')
  # Without checking the reader's own status, a readelf that never ran leaves
  # notes empty, all three counts come back 0, and this prints PASS for a library
  # it never looked at. This is the acceptance test; it has to fail closed.
  if ! notes=$("$RE" --notes "$dev" 2>/dev/null); then
    printf '  [%s] llvm-readelf could not read this device image\n' "$(basename "$dev" | sed 's/.*\.//')"
    fail=$((fail + 1))
    continue
  fi
  hc=$(printf '%s' "$notes" | grep -ic hidden_hostcall_buffer || true)
  af=$(printf '%s' "$notes" | grep -ic __assert_fail || true)
  fp=$(printf '%s' "$notes" | grep -ic __ockl_fprintf || true)
  printf '  [%s] hidden_hostcall_buffer=%-4s __assert_fail=%-4s __ockl_fprintf=%-4s' "$arch" "$hc" "$af" "$fp"
  if [ "$hc" -eq 0 ] && [ "$af" -eq 0 ] && [ "$fp" -eq 0 ]; then
    echo "  PASS"
  else
    echo "  FAIL"
    fail=1
  fi
done

if [ "$found" -eq 0 ]; then
  # Some ROCm distributions (notably the pip `_rocm_sdk_libraries` wheels used by
  # recent rocm/vllm images) do NOT embed device code in the .so. There the
  # .hip_fatbin section is NOBITS and the real device image lives in a separate
  # per-architecture container: <sdk>/.kpack/rccl_lib_gfx<arch>.kpack (magic "KPAK").
  # That format is opaque to llvm tooling, so static inspection is not possible.
  nobits=$("$RE" -S "$LIB" 2>/dev/null | grep -i 'hip_fatbin' | grep -ci NOBITS || true)
  kdir=""
  for c in "$(dirname "$LIB")/../.kpack" "$(dirname "$LIB")/.kpack"; do
    [ -d "$c" ] && kdir=$(cd "$c" && pwd) && break
  done
  if [ "${nobits:-0}" -gt 0 ] || [ -n "$kdir" ]; then
    echo "  no device image in the library: this install uses the split 'kpack' layout."
    [ -n "$kdir" ] && echo "  device code lives in: $kdir/rccl_lib_gfx<arch>.kpack"
    echo
    echo "RESULT: NOT APPLICABLE — static inspection cannot work on this layout."
    echo "        The .kpack container is opaque to llvm-readelf/objdump."
    echo "        Use the RUNTIME probe instead, which needs no library inspection:"
    echo "          hipcc --offload-arch=<your gfx> -O2 diagnose/hipgate3.cpp -o hipgate3 && ./hipgate3"
    echo "        (This script's main job is verifying a library YOU built, which"
    echo "         does produce a classic fat binary and inspects fine.)"
    exit 2
  fi
  echo "  no device image extracted — is this really a bundled RCCL?"
  exit 2
fi

cd=$("$OD" -T "$LIB" 2>/dev/null | grep -c ncclCommDump || true)
printf '  ncclCommDump exported: %s' "$cd"
if [ "$cd" -ge 1 ]; then echo "  PASS"; else echo "  FAIL (torch will refuse to load this)"; fail=1; fi

echo
if [ "$fail" -eq 0 ]; then
  echo "RESULT: PASS — this library will dispatch on a platform without PCIe atomics."
  exit 0
else
  echo "RESULT: FAIL — see docs/root-cause.md §5. NDEBUG must reach the DEVICE pass;"
  echo "        RCCL's device compile bypasses CMAKE_CXX_FLAGS_RELEASE, so use"
  echo "        add_compile_definitions(NDEBUG) right after project(rccl CXX)."
  exit 1
fi
