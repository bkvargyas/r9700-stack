#!/usr/bin/env bash
# Build libr9k.so (our gfx1201 kernels) with the image's hipcc. NDEBUG and no device asserts/printf,
# so no kernel requests hidden_hostcall_buffer (P2P-safe behind switches that drop PCIe atomics).
set -euo pipefail
cd "$(dirname "$0")"
GFX_ARCH=${GFX_ARCH:-gfx1201}
OUT=${OUT:-libr9k.so}
HIPCC=${HIPCC:-hipcc}
SRCS=(r9k_moe_mxfp4a8.hip r9k_ple.hip third_party/davetha/r4d_lru.hip)
$HIPCC -O3 -std=c++17 -fPIC -DNDEBUG --offload-arch="$GFX_ARCH" -ffp-contract=off -shared "${SRCS[@]}" -o "$OUT"
if grep -q -a hidden_hostcall_buffer "$OUT"; then echo "ERROR: $OUT contains hostcall kernels" >&2; exit 1; fi
for sym in r9k_moe_mxfp4a8 r9k_quant_rows_fp8 r9k_silu_mul_quant_fp8 r9k_moe_block r9k_ple_gather_int6 r4d_lru_manage r4d_lru_gather r4d_lru_fused; do
  nm -D "$OUT" | grep -q " T $sym$" || { echo "MISSING EXPORT: $sym" >&2; exit 1; }
done
echo "built $(ls -la "$OUT")"
