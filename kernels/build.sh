#!/usr/bin/env bash
# Build libr9k.so (our gfx1201 kernels) with the image's hipcc. NDEBUG and no device asserts/printf,
# so no kernel requests hidden_hostcall_buffer (P2P-safe behind switches that drop PCIe atomics).
set -euo pipefail
cd "$(dirname "$0")"
GFX_ARCH=${GFX_ARCH:-gfx1201}
OUT=${OUT:-libr9k.so}
HIPCC=${HIPCC:-hipcc}
SRCS=(r9k_moe_mxfp4a8.hip r9k_ple.hip r9k_gemm_fp8.hip r9k_ar.hip r9k_ar_wht.hip r9k_attn.hip r9k_qsa.hip r9k_qsa_score.hip r9k_transpose.hip r9k_hc.hip r9k_ar4.hip r9k_router.hip r9k_norm_rope.hip r9k_gdn.hip third_party/davetha/r4d_lru.hip)
$HIPCC -O3 -std=c++17 -fPIC -DNDEBUG --offload-arch="$GFX_ARCH" -ffp-contract=off -shared ${HIPCFLAGS:-} "${SRCS[@]}" -o "$OUT"
if grep -q -a hidden_hostcall_buffer "$OUT"; then echo "ERROR: $OUT contains hostcall kernels" >&2; exit 1; fi
for sym in r9k_moe_mxfp4a8 r9k_moe_nvfp4a8 r9k_moe_4bit_prefill r9k_moe_prefill_bm r9k_moe_prefill_bk r9k_moe_4bit_prefill_at r9k_moe_atiled_bm r9k_quant_rows_fp8 r9k_quant_rows_fp8_tiled r9k_silu_mul_quant_fp8 r9k_moe_block r9k_fold_supported r9k_ple_gather_int6 r9k_gemm_fp8 r9k_gemm_fp8_block r9k_quant_group128_fp8 r4d_lru_manage r4d_lru_gather r4d_lru_fused r9k_ar_oneshot_2rank r9k_ar_oneshot_nrank r9k_ar_twoshot_nrank r9k_ar_max_ranks r9k_ar_max_blocks r9k_ar_ipc_alloc r9k_ar_ipc_open r9k_ar_ipc_handle_size r9k_wht_pack r9k_wht_reduce r9k_wht_reduce_at r9k_ar_push_2rank r9k_wht_group r9k_wht_group_bytes r9k_wht_bytes_for r9k_attn_prefill_paged r9k_attn_prefill_paged_fp8 r9k_qsa_bitmap r9k_qsa_attn r9k_qsa_merge r9k_qsa_score r9k_transpose16 r9k_hc_combine_norm r9k_hc_gate_mix r9k_ar4_pack_push r9k_ar4_reduce_bf16 r9k_ar4_reduce_pack_push r9k_ar4_decode r9k_ar4_push2 r9k_ar4_pack r9k_ar4_reduce_pack r9k_ar4_flag r9k_ar4_dma r9k_ar4_packed_bytes r9k_ar4_cbytes r9k_ar4_group r9k_ar4_max_blocks r9k_router_gemm r9k_router_max_m r9k_gemma_norm_rope r9k_hc_up_mix r9k_gdn_decode_mtp r9k_gdn_seq r9k_gdn_conv r9k_gdn_spec_verify r9k_gdn_spec_commit r9k_quant_rows_fp8_gate; do
  nm -D "$OUT" | grep -q " T $sym$" || { echo "MISSING EXPORT: $sym" >&2; exit 1; }
done
echo "built $(ls -la "$OUT")"
