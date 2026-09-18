#!/usr/bin/env bash
# Build libr9k.so (our gfx1201 kernels) with the image's hipcc. NDEBUG and no device asserts/printf,
# so no kernel requests hidden_hostcall_buffer (P2P-safe behind switches that drop PCIe atomics).
set -euo pipefail
cd "$(dirname "$0")"
GFX_ARCH=${GFX_ARCH:-gfx1201}
OUT=${OUT:-libr9k.so}
HIPCC=${HIPCC:-hipcc}
SRCS=(r9k_moe_mxfp4a8.hip)
$HIPCC -O3 -std=c++17 -fPIC -DNDEBUG --offload-arch="$GFX_ARCH" -ffp-contract=off -shared "${SRCS[@]}" -o "$OUT"
if grep -q -a hidden_hostcall_buffer "$OUT"; then echo "ERROR: $OUT contains hostcall kernels" >&2; exit 1; fi
for sym in r9k_moe_mxfp4a8 r9k_quant_rows_fp8 r9k_moe_block; do
  nm -D "$OUT" | grep -q " T $sym$" || { echo "MISSING EXPORT: $sym" >&2; exit 1; }
done
echo "built $(ls -la "$OUT")"
