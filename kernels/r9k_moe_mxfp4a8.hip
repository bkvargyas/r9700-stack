// SPDX-License-Identifier: LicenseRef-libr4d-derived
// Partly derived from libr4d (no upstream licence; used with permission). NOT Apache-2.0 -- see NOTICE.

// r9k_moe_mxfp4a8.hip -- grouped (MoE) OCP-MXFP4-weight x FP8-activation GEMM for gfx1201 (RDNA4).
//
// Derived from libr4d's r4d_gemm_mxfp4a8_nt_m64 (codeberg.org/StillDeadcode/libr4d, commit 5dc6302;
// LICENSE PENDING -- libr4d ships no license file yet). Same fragment-order weight layout, same
// folded-exponent e2m1->e4m3 unpack (four v_perm_b32 per eight weights), same fp8 WMMA and in-block
// split-K reduced through LDS. What is new is the MoE addressing:
//
//   * one workgroup row (blockIdx.y) = one 16-row block of the routing-sorted token list produced by
//     vLLM's moe_align_block_size(block_size=16): sorted_ids[16*by .. +16] are flat (token*topk+j)
//     indices, expert_ids[by] is the expert (or cache slot) all 16 rows go to;
//   * the A row of flat index r is r / a_row_div (a_row_div = topk for the gate_up GEMM, whose input
//     is the per-token hidden state; 1 for the down GEMM, whose input is per (token, j));
//   * blocks at or past num_tokens_post_padded, and blocks whose expert id is < 0 (not on this rank),
//     exit immediately -- the grid is sized from the host-known capacity so the launch is
//     cudagraph-safe while the device decides how much of it is live;
//   * padding rows (sorted id >= numel) are clamped to the block's first row for the load and dropped
//     at the store, keeping the inner loop branch-free as in the dense kernel;
//   * the epilogue can fold in the router weight topk_w[r] (down GEMM), so the caller only has to sum
//     the top-k rows per token afterwards.
//
// Weight per expert e (expert-major):
//   Wp  [E][N/16][K/16][32] uint32   fragment order (mxfp4_layout.permute_w per expert)
//   Ws  e*ws_estride + [K/32][N]  uint8   E8M0 block exponents, transposed to K-block-major
//   Wref e*wref_estride + [N]     uint8   per-row reference exponent = max over K blocks
// (the Python side packs Ws and Wref into one [E][K/32+1][N] tensor: both strides (K/32+1)*N.)
// Activations: A [Ma][K] e4m3 (per-row dynamic scale As[Ma] fp32). Output C [numel][N] bf16.
// Shape rules: N % 16 == 0, K % (SK*32) == 0 (K=320 -> SK in {1,2,5,10}).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <cstdint>
#include <cstdlib>

#define WAVE32 32
#define GROUP 32
#define MOE_BLOCK 16

typedef int   v2i __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));

__device__ __constant__ unsigned int kMag[16][2] = {
  {0x3c383000u, 0x4c484440u}, {0x34302800u, 0x44403c38u},
  {0x2c282000u, 0x3c383430u}, {0x24201800u, 0x34302c28u},
  {0x1c181000u, 0x2c282420u}, {0x14100800u, 0x24201c18u},
  {0x0c080400u, 0x1c181410u}, {0x06040200u, 0x14100c08u},
  {0x03020100u, 0x0c080604u}, {0x01010000u, 0x06040302u},
  {0x01000000u, 0x03020101u}, {0x00000000u, 0x01010100u},
  {0x00000000u, 0x01000000u}, {0x00000000u, 0x00000000u},
  {0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u},
};

__device__ __forceinline__ v2i mxfp4_unpack8(unsigned int wv, int dsh) {
  const unsigned int t0 = kMag[dsh][0], t1 = kMag[dsh][1];
  const unsigned int ev = wv & 0x0F0F0F0Fu;
  const unsigned int od = (wv >> 4) & 0x0F0F0F0Fu;
  const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u) | ((ev & 0x08080808u) << 4);
  const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u) | ((od & 0x08080808u) << 4);
  v2i out;
  out[0] = (int)__builtin_amdgcn_perm(bo, be, 0x05010400u);
  out[1] = (int)__builtin_amdgcn_perm(bo, be, 0x07030602u);
  return out;
}

// OCP e4m3fn byte -> fp32 (NVFP4 per-16 scales; always finite, NaN 0x7f never produced by quantizers).
__device__ __forceinline__ float e4m3_to_f32(unsigned int b) {
  const unsigned int em = b & 0x7Fu;
  const float mag = em >= 0x08u ? __uint_as_float((em + (120u << 3)) << 20) : (float)em * 0x1p-9f;
  return (b & 0x80u) ? -mag : mag;
}

// NV = false: OCP MXFP4 (E8M0 per 32 = two WMMA K steps). Ws = E8M0 [K/32][N] per expert; Wref (the per-row max
//             exponent) is read only by the FOLD variant (mxfp4_unpack8_fold); otherwise the scale is applied in fp32.
// NV = true:  NVFP4 (e4m3 scale per 16 = exactly one WMMA K step). Ws = e4m3 [K/16][N] per expert, Wref = fp32
//             per-row global multiplier [N] per expert (wref_estride in floats), applied in the epilogue.
// Both: the WMMA runs on the unscaled e2m1 values (kMag[0]) and the group's 16x16 partial is scaled into the
// accumulator -- in the WMMA C layout each lane holds 8 rows of ONE output column n, so the per-(n, group) scale
// is one scalar per lane: 8 FMAs per group, exact weight math.
// Loads: every global access is a uniform (SGPR) base plus a 32-bit per-lane offset, so the compiler emits
// saddr-form global_load with one v_add per address instead of 64-bit VALU adds with carries (the old loop spent
// ~25 of its ~200 instructions per K step on those plus the s_wait_alu they trigger). Requires every per-expert
// tensor to be < 4 GB, which holds for any single expert / dense weight here.
template <typename T>
__device__ __forceinline__ T ld_u32off(const T* __restrict__ base, unsigned off) {
  return *(const T*)((const unsigned char*)base + (size_t)off);
}
// Nontemporal load for streams read once per call. Only the MT == 1 (decode) path uses it: there every weight and
// scale byte is read by exactly one workgroup. At MT > 1 (prefill / wide batches) several routing blocks of the same
// expert re-read its weights, and the cache hit is worth more than the stream isolation (Flash-Next prefill 2000
// -> 1525 tok/s with NT on every path). R9K_NT=0 disables it everywhere.
#ifndef R9K_NT
#define R9K_NT 1
#endif
template <bool NTL, typename T>
__device__ __forceinline__ T ld_nt_u32off(const T* __restrict__ base, unsigned off) {
  if constexpr (NTL && R9K_NT)
    return __builtin_nontemporal_load((const T*)((const unsigned char*)base + (size_t)off));
  else
    return *(const T*)((const unsigned char*)base + (size_t)off);
}

// Folded-exponent unpack (FOLD paths): the e2m1 code is mapped to e4m3(value * 2^-dsh), dsh = per-row reference
// exponent (Wref, the row's max block exponent) minus the block's E8M0, so the WMMA consumes the block scale for free
// and only 2^(Wref - 127) is left for the epilogue. Exact for dsh <= 8 (kMag rows 0..8 reach e4m3's subnormals with
// no rounding), rounded for 9..12 (the e2m1 mantissa bits fall off the bottom of the subnormal range), flushed to zero
// for dsh >= 13 (kMag rows 13..15 are all zero). What can fold is decided per weight tensor on the host
// (kernels/moe.py:fold_stats / fold_ok), the kernel just clamps dsh to 0..15.
// The 16-row table lives in two VGPRs per wave, entry d in lane d (and d + 16), and is looked up with a lane
// permute: no dependent memory access per unpack. The old folded path's global kMag[dsh] load after every scale byte
// was what made it lose to fp32 group scaling in pass 1.
__device__ __forceinline__ unsigned int lane_select16(unsigned int tab, int d) {
#if __has_builtin(__builtin_amdgcn_permlane16_var)
  return __builtin_amdgcn_permlane16_var(tab, tab, (unsigned)d, false, false);
#else
  return (unsigned)__builtin_amdgcn_ds_bpermute(d << 2, (int)tab);
#endif
}
struct FoldTab { unsigned int t0, t1; };
__device__ __forceinline__ FoldTab fold_tab_load() {
  const int l = (threadIdx.x % WAVE32) & 15;
  return FoldTab{kMag[l][0], kMag[l][1]};
}
__device__ __forceinline__ int fold_dsh(int ref, int blk) {
  const int d = ref - blk;
  return d < 0 ? 0 : (d > 15 ? 15 : d);
}
__device__ __forceinline__ v2i mxfp4_unpack8_fold(unsigned int wv, FoldTab tab, int dsh) {
  const unsigned int t0 = lane_select16(tab.t0, dsh), t1 = lane_select16(tab.t1, dsh);
  const unsigned int ev = wv & 0x0F0F0F0Fu;
  const unsigned int od = (wv >> 4) & 0x0F0F0F0Fu;
  const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u) | ((ev & 0x08080808u) << 4);
  const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u) | ((od & 0x08080808u) << 4);
  v2i out;
  out[0] = (int)__builtin_amdgcn_perm(bo, be, 0x05010400u);
  out[1] = (int)__builtin_amdgcn_perm(bo, be, 0x07030602u);
  return out;
}
// 1 / 2 / 4 scale bytes of consecutive columns as one load (the column offset is a multiple of the width).
template <int W>
__device__ __forceinline__ unsigned int ld_scale_bytes(const unsigned char* __restrict__ Ws, unsigned off) {
  if constexpr (W == 4) return ld_u32off(( const unsigned int*)Ws, off);
  else if constexpr (W == 2) return ld_u32off((const unsigned short*)Ws, off);
  else return ld_u32off(Ws, off);
}

// A-tile staging for MT > 1: the BLK x K activation slab of the block is read from L2 once per workgroup per
// DBK-column slab into padded LDS (row stride DBK + 8 bytes: conflict-free ds_read_b64 for the 16 rows x 2
// K-halves of a WMMA A fragment) instead of once per wave from global memory. Register-prefetched one slab ahead.
// Workgroup barrier that only drains LDS traffic: __syncthreads() lowers to a workgroup release fence that also
// waits for every outstanding global load (s_wait_loadcnt 0x0), which would serialize the slab prefetch (the
// weight loads for slab s+1 are in flight across this barrier by design). Only LDS stores have to be visible.
__device__ __forceinline__ void lds_barrier() {
  asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory");
}

#define LDSA_DBK 64
#define LDSA_STR (LDSA_DBK + 8)
#define LDSA_NCH 4          // max 16-byte chunks per thread per slab (launcher: needs WV*32 threads >= BLK)

// FOLD (MXFP4 only): folded-exponent unpack (see mxfp4_unpack8_fold), no per-group FMAs, 2^(Wref - 127) in the
// epilogue. Selected per weight tensor by the host (MT bit 4).
template <int MT, int NPW, bool NV, bool LDSA, bool FOLD>
__global__ void __launch_bounds__(LDSA ? 256 : 1024) r9k_moe_mxfp4a8_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned int* __restrict__ Wp_all, const unsigned char* __restrict__ Ws_all,
    const unsigned char* __restrict__ Wref_all, __hip_bfloat16* __restrict__ C,
    const int* __restrict__ sorted_ids, const int* __restrict__ expert_ids,
    const int* __restrict__ ntpp, const float* __restrict__ topk_w,
    long ws_estride, long wref_estride, int numel, int a_row_div, int K, int N, int WV, int SK) {
  // One workgroup row = one routing block of BLK = 16*MT sorted rows (moe_align_block_size(block=16*MT)),
  // processed as MT WMMA M-tiles that share every weight fragment (MT x fewer weight reads per row).
  constexpr int BLK = 16 * MT;
  static_assert(!(LDSA && MT == 1), "LDS A staging is for MT > 1");
  static_assert(!(NV && FOLD), "folding is for E8M0 (MXFP4) scales");
  const int by = blockIdx.y;
  if (by * BLK >= ntpp[0]) return;
  const int e = expert_ids[by];
  if (e < 0) return;

  extern __shared__ float red[];
  unsigned char* const sA = (unsigned char*)red;
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int Ks = K / SK, kbase = ksplit * Ks;
  const int nlane = lane & 15, mhalf = (lane >> 4) * 8;
  const int ksteps = K / 16;

  const unsigned int* __restrict__ Wp = Wp_all + (size_t)e * (N / 16) * ksteps * WAVE32;
  const unsigned char* __restrict__ Ws = Ws_all + (size_t)e * ws_estride;
  const float* __restrict__ Wg = NV ? (const float*)Wref_all + (size_t)e * wref_estride : nullptr;
  const unsigned char* __restrict__ Wref = (!NV && FOLD) ? Wref_all + (size_t)e * wref_estride : nullptr;

  // This lane's A row in each M tile: the sorted flat index, clamped to the block's first row (always valid
  // for a live block) for padding entries.
  const int first = sorted_ids[by * BLK];
  if (first < 0 || first >= numel) return;          // fully padded block: nothing valid to read or write
  unsigned aoff[MT];                                // byte offset of the lane's A row (+ K half) in A
  if constexpr (!LDSA) {
#pragma unroll
    for (int i = 0; i < MT; i++) {
      int flat = sorted_ids[by * BLK + i * 16 + nlane];
      if (flat >= numel) flat = first;
      aoff[i] = (unsigned)(flat / a_row_div) * (unsigned)K + mhalf;
    }
  }

  // LDS staging bookkeeping (LDSA): this thread's 16-byte chunks of the SK x BLK x DBK slab.
  const int nthreads = WV * SK * WAVE32;
  unsigned sa_src[LDSA_NCH];                        // A byte offset of the chunk at slab 0
  // chunks per thread = SK*BLK*CPR / nthreads = BLK/(8*WV): exact by the launcher's gate, SGPR-uniform (WV is a
  // kernel argument) so the staging loops below are scalar-predicated, not exec-masked.
  const int nch = LDSA ? (BLK / 8) / WV : 0;
  if constexpr (LDSA) {
    constexpr int CPR = LDSA_DBK / 16;              // chunks per row per slab
    const int per_split = BLK * CPR;
#pragma unroll
    for (int q = 0; q < LDSA_NCH; q++) {
      if (q < nch) {
        const int c = tid + q * nthreads;
        const int g = c / per_split, rem = c - g * per_split, r = rem / CPR, c16 = rem - r * CPR;
        int flat = sorted_ids[by * BLK + r];
        if (flat >= numel) flat = first;
        sa_src[q] = (unsigned)(flat / a_row_div) * (unsigned)K + (unsigned)(g * Ks + c16 * 16);
      }
    }
  }

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  const int ntmax = N / 16 - 1;
  unsigned woff[NPW], soff[NPW];                    // byte offsets: weight fragment stream, scale column
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    const int t = nt0 + j;
    const int wt = t < ntmax ? t : ntmax;
    woff[j] = ((unsigned)wt * (unsigned)ksteps * WAVE32 + lane) * 4u;
    soff[j] = (unsigned)(wt * 16 + nlane);
  }
  // FOLD: the lane's column reference exponents (one byte per N tile, hoisted) and the wave's kMag table.
  int wref[NPW];
  FoldTab ftab;
  if constexpr (FOLD) {
    ftab = fold_tab_load();
#pragma unroll
    for (int j = 0; j < NPW; j++) wref[j] = Wref[soff[j]];
  }

  // Weights stream through once: nontemporal (TH_LOAD_NT) loads keep the stream from displacing A, the scales
  // and the split-K partials in L0/L2/MALL. Scale groups: NVFP4 = 16 K (one WMMA), MXFP4 = 32 K (two). The
  // unscaled e2m1 products of a group accumulate in a temporary and are scaled into acc once per group (8 FMAs
  // per lane), so no per-lane exponent table is needed: the old folded MXFP4 path issued a dependent kMag[dsh]
  // global load after every Ws byte (two serial memory latencies per K step) and flushed blocks >2^-9 below the
  // row max; the fp32 group scale is exact.
  // The scales are streamed nontemporally too: they are read exactly once per step like the weights, and a
  // cached scale stream flatters the rotating-copy benchmark (5 x 5.6 MB of NVFP4 scales stay in the 64 MB MALL:
  // 82 us "612 GB/s" that is really 89 us) without helping the served model.
  // Written as a macro, not a lambda: the lambda form makes the compiler wait for each K step's loads before
  // issuing the next step's (measured 90 -> 102 us on NVFP4 17408x5120 @M=8).
  constexpr int KG = NV ? 16 : GROUP, SPG = KG / 16;
#define R9K_GROUP(KGV, AF, WVV)                                                                               \
  {                                                                                                          \
    const int g_ = (KGV) / KG;                                                                               \
    _Pragma("unroll")                                                                                        \
    for (int j = 0; j < NPW; j++) {                                                                          \
      const unsigned int sb = ld_nt_u32off<(MT == 1)>(Ws, (unsigned)g_ * (unsigned)N + soff[j]);                        \
      if constexpr (FOLD) {                                                                                  \
        const int dsh_ = fold_dsh(wref[j], (int)sb);                                                         \
        _Pragma("unroll")                                                                                    \
        for (int sidx = 0; sidx < SPG; sidx++) {                                                             \
          const v2i wf = mxfp4_unpack8_fold(WVV[j][sidx], ftab, dsh_);                                       \
          _Pragma("unroll")                                                                                  \
          for (int i = 0; i < MT; i++)                                                                       \
            acc[j][i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(AF[sidx][i], wf, acc[j][i]);    \
        }                                                                                                    \
      } else {                                                                                               \
        v8f t[MT];                                                                                           \
        _Pragma("unroll")                                                                                    \
        for (int i = 0; i < MT; i++) t[i] = (v8f)(0.f);                                                      \
        _Pragma("unroll")                                                                                    \
        for (int sidx = 0; sidx < SPG; sidx++) {                                                             \
          const v2i wf = mxfp4_unpack8(WVV[j][sidx], 0);                                                     \
          _Pragma("unroll")                                                                                  \
          for (int i = 0; i < MT; i++)                                                                       \
            t[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(AF[sidx][i], wf, t[i]);              \
        }                                                                                                    \
        const float sc = NV ? __builtin_amdgcn_cvt_f32_fp8((int)sb, 0) : __uint_as_float(sb << 23);          \
        _Pragma("unroll")                                                                                    \
        for (int i = 0; i < MT; i++)                                                                         \
          _Pragma("unroll")                                                                                  \
          for (int q = 0; q < 8; q++) acc[j][i][q] = fmaf(t[i][q], sc, acc[j][i][q]);   /* -ffp-contract=off */ \
      }                                                                                                      \
    }                                                                                                        \
  }

  if constexpr (!LDSA) {
    // Direct path (MT == 1 decode, or K not a multiple of SK*DBK): A fragments straight from global (L0/L2
    // resident at decode M). The loop is unrolled so 4 K steps' weight loads are in flight per wave (each load
    // is one coalesced 128 B wave request; memory-level parallelism has to come from several in flight).
    constexpr int KU = MT == 1 ? 4 / SPG : 1;
#pragma unroll KU
    for (int k = 0; k < Ks; k += KG) {
      const int kg = kbase + k;
      unsigned int wv[NPW][SPG];
      v2i af[SPG][MT];
#pragma unroll
      for (int sidx = 0; sidx < SPG; sidx++) {
#pragma unroll
        for (int j = 0; j < NPW; j++)
          wv[j][sidx] = ld_nt_u32off<(MT == 1)>(Wp, woff[j] + (unsigned)(kg / 16 + sidx) * (WAVE32 * 4));
#pragma unroll
        for (int i = 0; i < MT; i++) af[sidx][i] = ld_u32off((const v2i*)A, aoff[i] + (unsigned)(kg + sidx * 16));
      }
      R9K_GROUP(kg, af, wv)
    }
  } else {
    constexpr int KSPS = LDSA_DBK / 16;             // K steps per slab
    constexpr int CPR = LDSA_DBK / 16;              // 16 B chunks per row per slab
    const int nslabs = Ks / LDSA_DBK;
    uint4 pre[LDSA_NCH];
    // Weight fragments for a slab are prefetched one slab ahead (KSPS x NPW registers), so the DRAM stream is
    // never waiting on the barrier + staging of the slab that consumes it.
    unsigned int wv[NPW][KSPS];
    const unsigned char* const myA = sA + (size_t)ksplit * BLK * LDSA_STR + nlane * LDSA_STR + mhalf;
#pragma unroll
    for (int q = 0; q < LDSA_NCH; q++)
      if (q < nch) pre[q] = ld_u32off((const uint4*)A, sa_src[q]);
#pragma unroll
    for (int sidx = 0; sidx < KSPS; sidx++)
#pragma unroll
      for (int j = 0; j < NPW; j++)
        wv[j][sidx] = ld_nt_u32off<(MT == 1)>(Wp, woff[j] + (unsigned)(kbase / 16 + sidx) * (WAVE32 * 4));
    for (int slab = 0; slab < nslabs; slab++) {
#pragma unroll
      for (int q = 0; q < LDSA_NCH; q++) {
        if (q < nch) {
          const int c = tid + q * nthreads;
          const int g = c / (BLK * CPR), rem = c - g * (BLK * CPR), r = rem / CPR, c16 = rem - r * CPR;
          *(uint4*)(sA + (g * BLK + r) * LDSA_STR + c16 * 16) = pre[q];
        }
      }
      lds_barrier();
      const bool more = slab + 1 < nslabs;
      if (more) {
#pragma unroll
        for (int q = 0; q < LDSA_NCH; q++)
          if (q < nch) pre[q] = ld_u32off((const uint4*)A, sa_src[q] + (unsigned)(slab + 1) * LDSA_DBK);
      }
      unsigned int wcur[NPW][KSPS];
#pragma unroll
      for (int sidx = 0; sidx < KSPS; sidx++)
#pragma unroll
        for (int j = 0; j < NPW; j++) wcur[j][sidx] = wv[j][sidx];
      if (more) {
        const int ks0 = (kbase + (slab + 1) * LDSA_DBK) / 16;
#pragma unroll
        for (int sidx = 0; sidx < KSPS; sidx++)
#pragma unroll
          for (int j = 0; j < NPW; j++)
            wv[j][sidx] = ld_nt_u32off<(MT == 1)>(Wp, woff[j] + (unsigned)(ks0 + sidx) * (WAVE32 * 4));
      }
      const int kg0 = kbase + slab * LDSA_DBK;
#pragma unroll
      for (int k = 0; k < LDSA_DBK; k += KG) {
        v2i af[SPG][MT];
        unsigned int wg[NPW][SPG];
#pragma unroll
        for (int sidx = 0; sidx < SPG; sidx++) {
#pragma unroll
          for (int i = 0; i < MT; i++) af[sidx][i] = *(const v2i*)(myA + i * 16 * LDSA_STR + k + sidx * 16);
#pragma unroll
          for (int j = 0; j < NPW; j++) wg[j][sidx] = wcur[j][k / 16 + sidx];
        }
        R9K_GROUP(kg0 + k, af, wg)
      }
      lds_barrier();
    }
  }
#undef R9K_GROUP

  // Reduce the SK partial sums in LDS, one M tile at a time (buffer stays WV*NPW*SK*256 floats).
  const int ncols = WV * NPW;
#pragma unroll
  for (int i = 0; i < MT; i++) {
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int slot = ((col_local * NPW + j) * SK + ksplit) * 256;
#pragma unroll
      for (int q = 0; q < 8; q++) red[slot + lane * 8 + q] = acc[j][i][q];
    }
    __syncthreads();
    for (int t = tid; t < ncols * 256; t += blockDim.x) {
      const int col = t >> 8, off = t & 255;
      const int n = (blockIdx.x * ncols + col) * 16 + ((off >> 3) & 15);
      const int mrow = i * 16 + ((off >> 7) << 3) + (off & 7);   // row within the BLK-row block
      if (n >= N) continue;
      const int fr = sorted_ids[by * BLK + mrow];
      if (fr >= numel) continue;                                  // padding row
      float v = 0.f;
      for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
      float sc = (NV ? Wg[n] : (FOLD ? __uint_as_float((unsigned)Wref[n] << 23) : 1.f)) * As[fr / a_row_div];
      if (topk_w) sc *= topk_w[fr];
      C[(size_t)fr * N + n] = (__hip_bfloat16)(v * sc);
    }
  }
}

// Per-row dynamic fp8 (e4m3fn, max 448) quantisation of a bf16 [M][K] matrix: one workgroup per row.
__global__ void r9k_quant_rows_fp8_kernel(const __hip_bfloat16* __restrict__ x, unsigned char* __restrict__ q,
                                          float* __restrict__ s, int K, int ld) {
  const int row = blockIdx.x;
  const __hip_bfloat16* xr = x + (size_t)row * ld;
  __shared__ float wmax[32];
  float m = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) m = fmaxf(m, fabsf((float)xr[k]));
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
  if ((threadIdx.x & 31) == 0) wmax[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < (blockDim.x >> 5) ? wmax[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
    if (threadIdx.x == 0) wmax[0] = v;
  }
  __syncthreads();
  const float amax = fmaxf(wmax[0], 1e-12f);
  const float inv = 448.f / amax;
  if (threadIdx.x == 0) s[row] = amax / 448.f;
  unsigned char* qr = q + (size_t)row * K;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    __hip_fp8_e4m3 f((float)xr[k] * inv);
    qr[k] = *reinterpret_cast<unsigned char*>(&f);
  }
}

// silu(gate) * up + per-row dynamic fp8 quant, fused: gu [M][2I] bf16 (gate | up halves, vLLM's gate_up
// layout) -> q [M][I] e4m3 + s [M]. One workgroup per row; the activation is computed twice (max pass +
// store pass) instead of staging I floats, which is cheaper than LDS at I = 320.
__global__ void r9k_silu_mul_quant_fp8_kernel(const __hip_bfloat16* __restrict__ gu, unsigned char* __restrict__ q,
                                              float* __restrict__ s, int I) {
  const int row = blockIdx.x;
  const __hip_bfloat16* g = gu + (size_t)row * 2 * I;
  const __hip_bfloat16* u = g + I;
  __shared__ float wmax[32];
  float m = 0.f;
  for (int k = threadIdx.x; k < I; k += blockDim.x) {
    const float a = (float)g[k];
    const float v = a / (1.f + __expf(-a)) * (float)u[k];
    m = fmaxf(m, fabsf(v));
  }
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
  if ((threadIdx.x & 31) == 0) wmax[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < (blockDim.x >> 5) ? wmax[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
    if (threadIdx.x == 0) wmax[0] = v;
  }
  __syncthreads();
  const float amax = fmaxf(wmax[0], 1e-12f);
  const float inv = 448.f / amax;
  if (threadIdx.x == 0) s[row] = amax / 448.f;
  unsigned char* qr = q + (size_t)row * I;
  for (int k = threadIdx.x; k < I; k += blockDim.x) {
    const float a = (float)g[k];
    const float v = a / (1.f + __expf(-a)) * (float)u[k];
    __hip_fp8_e4m3 f(v * inv);
    qr[k] = *reinterpret_cast<unsigned char*>(&f);
  }
}

extern "C" int r9k_silu_mul_quant_fp8(long gu, long q, long s, int M, int I, long stream) {
  if (M <= 0) return 0;
  hipLaunchKernelGGL(r9k_silu_mul_quant_fp8_kernel, dim3(M), dim3(128), 0, (hipStream_t)stream,
                     (const __hip_bfloat16*)gu, (unsigned char*)q, (float*)s, I);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

extern "C" int r9k_moe_block(void) { return MOE_BLOCK; }   // rows per M tile; routing block = 16*MT
extern "C" int r9k_fold_supported(void) { return 1; }       // folded-exponent MXFP4 (prefill nv=2, decode MT|16)

static int moe_launch(bool nv, long a, long as, long wq, long ws, long wref, long c, long sorted_ids,
                      long expert_ids, long ntpp, long topk_w, long ws_estride, long wref_estride, int max_blocks,
                      int numel, int a_row_div, int K, int N, int WV, int SK, int NPW, int MT, long stream) {
  if (N % 16) return -1;
  if (K % (SK * (nv ? 16 : GROUP))) return -2;
  if (WV * SK * WAVE32 > 1024) return -3;
  const int ncols = WV * NPW;
  size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  // MT bit 3 = stage A through LDS (a tuned per-config choice; R9K_LDSA=0/1 overrides for experiments). Legal
  // when the per-wave K range is whole DBK slabs, the block is <= 8 waves (launch bound 256 -> 256 VGPRs) and
  // the BLK x DBK slab splits evenly over the block's threads.
  bool ldsa = (MT & 8) != 0;
  // MT bit 4 = folded-exponent MXFP4 (ignored for NVFP4: its e4m3 scales cannot fold).
  const bool fold = !nv && (MT & 16) != 0;
  MT &= 7;
  static const int ldsa_env = [] { const char* v = getenv("R9K_LDSA"); return v ? (v[0] == '0' ? 0 : 1) : -1; }();
  if (ldsa_env >= 0) ldsa = ldsa_env != 0;
  // ...and each thread's share of the slab (2*MT/WV 16-byte chunks) must fit its LDSA_NCH staging registers --
  // without this check WV=1, MT=4 staged half the tile and read uninitialized LDS (NaN at serving batch 64).
  ldsa = ldsa && MT > 1 && K % (SK * LDSA_DBK) == 0 && WV * SK <= 8 && (2 * MT) % WV == 0 && (2 * MT) / WV <= LDSA_NCH;
  if (ldsa) { const size_t s2 = (size_t)SK * 16 * MT * LDSA_STR; if (s2 > smem) smem = s2; }
  if (smem > 64 * 1024) return -4;
  dim3 grid((N / 16 + ncols - 1) / ncols, max_blocks), block(WV * SK * WAVE32);
  hipStream_t st = (hipStream_t)stream;
#define L(MTv, NP, NVv, LD, FO) hipLaunchKernelGGL((r9k_moe_mxfp4a8_kernel<MTv, NP, NVv, LD, FO>), grid, block, smem, st, \
      (const unsigned char*)a, (const float*)as, (const unsigned int*)wq, (const unsigned char*)ws, \
      (const unsigned char*)wref, (__hip_bfloat16*)c, (const int*)sorted_ids,                        \
      (const int*)expert_ids, (const int*)ntpp, (const float*)topk_w, ws_estride, wref_estride, numel,  \
      a_row_div, K, N, WV, SK)
#define LN(MTv, NP, LD) do { if (nv) L(MTv, NP, true, LD, false); else if (fold) L(MTv, NP, false, LD, true); \
                             else L(MTv, NP, false, LD, false); } while (0)
#define LM(MTv, LD) switch (NPW) { case 1: LN(MTv, 1, LD); break; case 2: LN(MTv, 2, LD); break; \
                                   case 4: LN(MTv, 4, LD); break; default: return -5; }
  switch (MT) {
    case 1: LM(1, false); break;
    case 2: if (ldsa) LM(2, true) else LM(2, false); break;
    case 4: if (ldsa) LM(4, true) else LM(4, false); break;
    default: return -7;
  }
#undef LM
#undef LN
#undef L
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

// Returns 0 on success, negative on a shape error (the Python side raises).
extern "C" int r9k_moe_mxfp4a8(long a, long as, long wq, long ws, long wref, long c,
                               long sorted_ids, long expert_ids, long ntpp, long topk_w,
                               long ws_estride, long wref_estride, int max_blocks, int numel, int a_row_div, int K, int N,
                               int WV, int SK, int NPW, int MT, long stream) {
  return moe_launch(false, a, as, wq, ws, wref, c, sorted_ids, expert_ids, ntpp, topk_w, ws_estride, wref_estride,
                    max_blocks, numel, a_row_div, K, N, WV, SK, NPW, MT, stream);
}

// NVFP4 weights: ws = e4m3 [E][K/16][N] (ws_estride bytes), wref = fp32 per-row global multiplier [E][N]
// (wref_estride floats). Shape rule K % (SK*16) == 0.
extern "C" int r9k_moe_nvfp4a8(long a, long as, long wq, long ws, long wg, long c,
                               long sorted_ids, long expert_ids, long ntpp, long topk_w,
                               long ws_estride, long wg_estride, int max_blocks, int numel, int a_row_div, int K, int N,
                               int WV, int SK, int NPW, int MT, long stream) {
  return moe_launch(true, a, as, wq, ws, wg, c, sorted_ids, expert_ids, ntpp, topk_w, ws_estride, wg_estride,
                    max_blocks, numel, a_row_div, K, N, WV, SK, NPW, MT, stream);
}

// ---------------------------------------------------------------- prefill path (large M)
// The decode-shaped kernel above computes a 16*MT (<= 64) x 16*WV*NPW tile per workgroup: at prefill M every
// weight fragment is re-read M/64 times and the A slab is re-read once per 64-column block, and the per-wave
// inner loop carries MT A-fragment loads per NPW WMMAs. This path is a classic LDS-tiled GEMM: one workgroup =
// one routing block of BM = WM*TM*16 rows (up to 256) x BN = WN*TN*16 columns, WM x WN waves each owning a
// TM x TN grid of 16x16 accumulators, K in slabs of BK (32 or 64, single- or double-buffered) staged through LDS:
//   * A rows (gathered through sorted_ids like the decode kernel, so routed MoE blocks work unchanged) as
//     e4m3 bytes, row stride BK + 8 (conflict-free ds_read_b64 of a 16x16 fragment);
//   * weights unpacked ONCE per workgroup from the 4-bit fragment-order stream to e4m3 B fragments, kept in
//     fragment order in LDS (lane l of tile (nt, ks) at slot ((nt*4 + ks)*32 + l): a wave's B read is 256
//     contiguous bytes, no padding needed), so the e2m1 -> e4m3 v_perm work is done by NT threads per slab
//     instead of by every wave for every fragment;
//   * the per-(n, K-group) scales for the slab (BN x 64/KG bytes) in LDS as well.
// The next slab's global loads are issued right after the barrier that publishes the current one, so they are
// in flight during the slab's WMMAs (register cost: ACH uint4 + WSL uint32 + SCH bytes per thread).
// Numerics are those of the decode kernel: unscaled e2m1 WMMA partials per scale group, scaled into the fp32
// accumulator with explicit fmaf (exact weight math, no folded / flushed exponents; FOLD = the folded variant), epilogue
// bf16(acc * (Wg[n] *) As[row] (* topk_w[row])). Split-K is not used (K is walked whole by one workgroup).
#ifndef R9K_PF_ABL
#define R9K_PF_ABL 0      // timing ablations (never in production builds): 1 no reloads, 2 no WMMA, 4 no barriers
#endif
#ifndef R9K_AT_RESCALE
#define R9K_AT_RESCALE 0  // EXPERIMENT (A-tiled, FOLD = false only; not a production build): accumulator rescaling
                          // instead of partial + fmaf per scale group -- see r9k_moe_4bit_prefill_at_kernel
#endif

// BK: K per LDS slab. DB: double-buffered slabs -- the next slab is written to the other buffer right after this
// slab's WMMAs and ONE barrier per slab publishes it (single-buffered slabs need a second barrier before the
// buffer may be overwritten). BK=32 double-buffered costs the same LDS as BK=64 single-buffered; BK=64
// double-buffered (2 x 27 KB on the 256x128 tile, two workgroups per 128 KB WGP) halves the barriers per K and is
// the steady-state best on every served dense shape at M >= 512 (+4-5%).
// FOLD (MXFP4 only): the slab's weights are unpacked with the block exponent folded into the e4m3 bytes
// (mxfp4_unpack8_fold), so the WMMAs accumulate straight into acc -- no scale staging, no per-group FMAs -- and the
// epilogue applies 2^(Wref[n] - 127). Each thread's WSL weight slots are WSL consecutive lanes of one fragment, i.e.
// WSL consecutive columns at one K step, so it needs WSL reference-exponent bytes (hoisted) and one WSL-byte scale
// load per slab, prefetched with the weights.
template <int TM, int TN, int WM, int WN, bool NV, int BK, bool DB, bool FOLD>
__global__ void __launch_bounds__(WM * WN * WAVE32) r9k_moe_4bit_prefill_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned int* __restrict__ Wp_all, const unsigned char* __restrict__ Ws_all,
    const unsigned char* __restrict__ Wref_all, __hip_bfloat16* __restrict__ C,
    const int* __restrict__ sorted_ids, const int* __restrict__ expert_ids,
    const int* __restrict__ ntpp, const float* __restrict__ topk_w,
    long ws_estride, long wref_estride, int numel, int a_row_div, int K, int N) {
  constexpr int BM = WM * TM * 16, BN = WN * TN * 16, NT = WM * WN * WAVE32;
  constexpr int ASTR = BK + 8;                              // padded A row stride: conflict-free fragment reads
  constexpr int KSPS = BK / 16;                             // WMMA K steps per slab
  constexpr int KG = NV ? 16 : GROUP, SPG = KG / 16, G = BK / KG;   // scale groups per slab
  static_assert(G >= 1, "slab must hold a whole scale group");
  static_assert(!(NV && FOLD), "folding is for E8M0 (MXFP4) scales");
  constexpr int ACH = BM * (BK / 16) / NT;                  // 16 B A chunks per thread per slab
  constexpr int WSL = BN * (BK / 8) / NT;                   // uint32 weight slots per thread per slab
  constexpr int SCH = FOLD ? 1 : (BN * G + NT - 1) / NT;    // scale bytes per thread per slab (guarded; FOLD: none)
  static_assert(ACH * NT == BM * (BK / 16) && ACH >= 1 && ACH <= 8, "A slab must split evenly");
  static_assert(WSL * NT == BN * (BK / 8) && (WSL == 1 || WSL == 2 || WSL == 4), "W slab must split evenly");
  constexpr int SA_BYTES = BM * ASTR, SW_BYTES = BN * BK, SS_BYTES = FOLD ? 0 : ((BN * G + 15) & ~15);
  constexpr int BUF_BYTES = SA_BYTES + SW_BYTES + SS_BYTES, NBUF = DB ? 2 : 1;
  extern __shared__ __align__(16) unsigned char pf_smem[];
  int* const sRow = (int*)(pf_smem + NBUF * BUF_BYTES);

  // Workgroups are dispatched x-fastest: all column tiles of routing block 0, then of block 1, ... so a block's A
  // rows stay in L2 while the weight streams from MALL / DRAM once per routing block. A grouped swizzle (gm
  // consecutive blocks walking the column tiles together) was measured at gm = 2/4/8: no change within noise at
  // M = 2048 / 4096 on either format, so the plain order stays.
  const int bx = blockIdx.x, by = blockIdx.y;
  if (by * BM >= ntpp[0]) return;
  const int e = expert_ids[by];
  if (e < 0) return;
  const int first = sorted_ids[by * BM];
  if (first < 0 || first >= numel) return;

  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int wm = wave / WN, wn = wave % WN;
  const int nlane = lane & 15, mhalf = (lane >> 4) * 8;
  const int ksteps = K / 16, ntmax = N / 16 - 1;
  const int nt0 = bx * (BN / 16);
  const unsigned int* __restrict__ Wp = Wp_all + (size_t)e * (N / 16) * ksteps * WAVE32;
  const unsigned char* __restrict__ Ws = Ws_all + (size_t)e * ws_estride;
  const float* __restrict__ Wg = NV ? (const float*)Wref_all + (size_t)e * wref_estride : nullptr;
  const unsigned char* __restrict__ Wref = (!NV && FOLD) ? Wref_all + (size_t)e * wref_estride : nullptr;

  for (int r = tid; r < BM; r += NT) {
    const int fl = sorted_ids[by * BM + r];
    sRow[r] = (fl < 0 || fl >= numel) ? -1 : fl;            // -1: padding row (loaded as `first`, never stored)
  }
  __syncthreads();

  // Staging geometry, hoisted: A chunk c = tid + q*NT -> (row c / (BK/16), 16 B column c % (BK/16)).
  unsigned a_src[ACH];
  int a_dst[ACH];
#pragma unroll
  for (int q = 0; q < ACH; q++) {
    const int c = tid + q * NT, r = c / (BK / 16), c16 = c % (BK / 16);
    const int fl = sRow[r];
    a_src[q] = (unsigned)((fl < 0 ? first : fl) / a_row_div) * (unsigned)K + (unsigned)(c16 * 16);
    a_dst[q] = r * ASTR + c16 * 16;
  }
  // Weight slots s = tid*WSL + q (contiguous, never straddling a 32-lane fragment): slot -> (n tile, K step, lane).
  unsigned w_src, w_dst, ws_own = 0;                        // ws_own: FOLD scale bytes of the slots' columns, slab 0
  int wref_own[WSL];
  FoldTab ftab;
  {
    const int s0 = tid * WSL, lane_ = s0 & 31, rest = s0 >> 5, ks = rest % KSPS, ntl = rest / KSPS;
    const int ntg = nt0 + ntl < ntmax ? nt0 + ntl : ntmax;   // clamped: garbage columns are never stored
    w_src = (unsigned)(((ntg * ksteps + ks) * WAVE32 + lane_) * 4);
    w_dst = (unsigned)(SA_BYTES + ((ntl * KSPS + ks) * WAVE32 + lane_) * 8);
    if constexpr (FOLD) {
      const int col0 = ntg * 16 + (lane_ & 15);              // lane_ % WSL == 0: the WSL columns are aligned
      ws_own = (unsigned)((ks / SPG) * N + col0);
      _Pragma("unroll")
      for (int q = 0; q < WSL; q++) wref_own[q] = Wref[col0 + q];
      ftab = fold_tab_load();
    }
  }
  // Scale bytes t = tid + it*NT -> (group t / BN, local column t % BN), Ws is [K/KG][N] per expert.
  unsigned s_src[SCH];
#pragma unroll
  for (int it = 0; it < SCH; it++) {
    const int t = tid + it * NT, g = t / BN, nl = t % BN;
    const int ntg = nt0 + nl / 16 < ntmax ? nt0 + nl / 16 : ntmax;
    s_src[it] = (unsigned)(g * N + ntg * 16 + (nl & 15));
  }

  v8f acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++) acc[i][j] = (v8f)(0.f);

  uint4 pa[ACH];
  unsigned int pw[WSL];
  unsigned char ps[SCH];
  unsigned int psw = 0;                                     // FOLD: the WSL scale bytes of this thread's slots
  const int nslabs = K / BK;
#define PF_LOAD(SLAB)                                                                                   \
  {                                                                                                     \
    const int slab_ = (SLAB);                                                                           \
    _Pragma("unroll")                                                                                   \
    for (int q = 0; q < ACH; q++) pa[q] = ld_u32off((const uint4*)A, a_src[q] + (unsigned)(slab_ * BK)); \
    if constexpr (WSL == 4) {                                                                           \
      const uint4 v = ld_u32off((const uint4*)Wp, w_src + (unsigned)(slab_ * KSPS * WAVE32 * 4));      \
      pw[0] = v.x; pw[1] = v.y; pw[2] = v.z; pw[3] = v.w;                                               \
    } else if constexpr (WSL == 2) {                                                                    \
      const uint2 v = ld_u32off((const uint2*)Wp, w_src + (unsigned)(slab_ * KSPS * WAVE32 * 4));      \
      pw[0] = v.x; pw[1] = v.y;                                                                         \
    } else {                                                                                            \
      pw[0] = ld_u32off(Wp, w_src + (unsigned)(slab_ * KSPS * WAVE32 * 4));                            \
    }                                                                                                   \
    if constexpr (FOLD) {                                                                               \
      psw = ld_scale_bytes<WSL>(Ws, ws_own + (unsigned)(slab_ * G) * (unsigned)N);                      \
    } else {                                                                                            \
      _Pragma("unroll")                                                                                 \
      for (int it = 0; it < SCH; it++)                                                                  \
        if (tid + it * NT < BN * G) ps[it] = ld_u32off(Ws, s_src[it] + (unsigned)(slab_ * G) * (unsigned)N); \
    }                                                                                                   \
  }
#define PF_STORE(BUF)                                                                                   \
  {                                                                                                     \
    unsigned char* const sb_ = pf_smem + (BUF) * BUF_BYTES;                                             \
    _Pragma("unroll")                                                                                   \
    for (int q = 0; q < ACH; q++) *(uint4*)(sb_ + a_dst[q]) = pa[q];                                    \
    if constexpr (FOLD) {                                                                               \
      _Pragma("unroll")                                                                                 \
      for (int q = 0; q < WSL; q++)                                                                     \
        *(v2i*)(sb_ + w_dst + q * 8) =                                                                  \
            mxfp4_unpack8_fold(pw[q], ftab, fold_dsh(wref_own[q], (int)((psw >> (8 * q)) & 0xFFu)));   \
    } else {                                                                                            \
      _Pragma("unroll")                                                                                 \
      for (int q = 0; q < WSL; q++) *(v2i*)(sb_ + w_dst + q * 8) = mxfp4_unpack8(pw[q], 0);             \
      _Pragma("unroll")                                                                                 \
      for (int it = 0; it < SCH; it++)                                                                  \
        if (tid + it * NT < BN * G) sb_[SA_BYTES + SW_BYTES + tid + it * NT] = ps[it];                  \
    }                                                                                                   \
  }
  // Fragments (A: TM x SPG, B: TN x SPG, scales: TN bytes) are read from LDS one scale group ahead of the WMMAs
  // that consume them: left to the compiler every ds_load pair was followed by its s_wait_dscnt (12 exposed LDS
  // latencies per slab against 32 WMMAs; 3156 -> 2785 us on NVFP4 17408x5120 @M=2048 with the 256x128 tile).
#define PF_FRAG_LOAD(OFF, GV, AF, WF, SB)                                                                \
    {                                                                                                    \
      _Pragma("unroll")                                                                                  \
      for (int sidx = 0; sidx < SPG; sidx++) {                                                           \
        _Pragma("unroll")                                                                                \
        for (int i = 0; i < TM; i++)                                                                     \
          AF[sidx][i] = *(const v2i*)(myA + (OFF) + i * 16 * ASTR + ((GV) * SPG + sidx) * 16);           \
        _Pragma("unroll")                                                                                \
        for (int j = 0; j < TN; j++)                                                                     \
          WF[sidx][j] = *(const v2i*)(myW + (OFF) + (j * KSPS + (GV) * SPG + sidx) * (WAVE32 * 8));      \
      }                                                                                                  \
      if constexpr (!FOLD) {                                                                             \
        _Pragma("unroll")                                                                                \
        for (int j = 0; j < TN; j++) SB[j] = myS[(OFF) + (GV) * BN + j * 16];                            \
      }                                                                                                  \
    }
  const unsigned char* const myA = pf_smem + (wm * TM * 16 + nlane) * ASTR + mhalf;
  const unsigned char* const myW = pf_smem + SA_BYTES + (wn * TN * KSPS * WAVE32 + lane) * 8;
  const unsigned char* const myS = pf_smem + SA_BYTES + SW_BYTES + wn * TN * 16 + nlane;

  // One scale group's WMMAs for output columns J0 <= j < J1: unscaled e2m1 x e4m3 partials, scaled into acc.
#define PF_COMPUTE(J0, J1, AF, WF, SB)                                                                   \
    {                                                                                                    \
      _Pragma("unroll")                                                                                  \
      for (int j = (J0); j < (J1); j++) {                                                                \
        if constexpr (FOLD) {                                                                            \
          _Pragma("unroll")                                                                              \
          for (int sidx = 0; sidx < SPG; sidx++) {                                                       \
            _Pragma("unroll")                                                                            \
            for (int i = 0; i < TM; i++) {                                                               \
              if (R9K_PF_ABL & 2) { acc[i][j][0] += __int_as_float(AF[sidx][i][0] ^ WF[sidx][j][1]); continue; } \
              acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(AF[sidx][i], WF[sidx][j], acc[i][j]); \
            }                                                                                            \
          }                                                                                              \
          continue;                                                                                      \
        }                                                                                                \
        v8f t[TM];                                                                                       \
        _Pragma("unroll")                                                                                \
        for (int i = 0; i < TM; i++) t[i] = (v8f)(0.f);                                                  \
        _Pragma("unroll")                                                                                \
        for (int sidx = 0; sidx < SPG; sidx++) {                                                         \
          _Pragma("unroll")                                                                              \
          for (int i = 0; i < TM; i++) {                                                                 \
            if (R9K_PF_ABL & 2) { t[i][0] += __int_as_float(AF[sidx][i][0] ^ WF[sidx][j][1]); continue; } \
            t[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(AF[sidx][i], WF[sidx][j], t[i]);  \
          }                                                                                              \
        }                                                                                                \
        const float sc = NV ? __builtin_amdgcn_cvt_f32_fp8((int)SB[j], 0) : __uint_as_float(SB[j] << 23); \
        _Pragma("unroll")                                                                                \
        for (int i = 0; i < TM; i++)                                                                     \
          _Pragma("unroll")                                                                              \
          for (int q = 0; q < 8; q++) acc[i][j][q] = fmaf(t[i][q], sc, acc[i][j][q]);   /* -ffp-contract=off */ \
      }                                                                                                  \
    }
#define PF_ROTATE()                                                                                      \
    {                                                                                                    \
      _Pragma("unroll")                                                                                  \
      for (int sidx = 0; sidx < SPG; sidx++) {                                                           \
        _Pragma("unroll")                                                                                \
        for (int i = 0; i < TM; i++) afc[sidx][i] = afn[sidx][i];                                        \
        _Pragma("unroll")                                                                                \
        for (int j = 0; j < TN; j++) wfc[sidx][j] = wfn[sidx][j];                                        \
      }                                                                                                  \
      _Pragma("unroll")                                                                                  \
      for (int j = 0; j < TN; j++) sbc[j] = sbn[j];                                                      \
    }
  v2i afc[SPG][TM], wfc[SPG][TN], afn[SPG][TM], wfn[SPG][TN];
  unsigned int sbc[TN], sbn[TN];
  if constexpr (!DB) {
    // Single-buffered slabs: store s; barrier; issue global loads of s+1; WMMAs of s; barrier (everyone is done
    // reading before the overwrite). Fragments are read one scale group ahead of their WMMAs.
    PF_LOAD(0)
    for (int slab = 0; slab < nslabs; slab++) {
      PF_STORE(0)
      if (!(R9K_PF_ABL & 4)) lds_barrier();
      const bool more = slab + 1 < nslabs;
      if (!(R9K_PF_ABL & 1) && more) PF_LOAD(slab + 1)
      PF_FRAG_LOAD(0, 0, afc, wfc, sbc)
#pragma unroll
      for (int g = 0; g < G; g++) {
        if (g + 1 < G) PF_FRAG_LOAD(0, g + 1, afn, wfn, sbn)
        PF_COMPUTE(0, TN, afc, wfc, sbc)
        if (g + 1 < G) PF_ROTATE()
      }
      if (!(R9K_PF_ABL & 4)) lds_barrier();
    }
  } else {
    // Double-buffered slabs: store slab s into buffer s&1; ONE barrier publishes it (the buffer it overwrites was
    // read in slab s-1, and every wave passed this barrier only after finishing slab s-1); issue the global loads
    // of s+1; WMMAs of s with the fragments of group g+1 read under group g's WMMAs (sched_barrier keeps the
    // prefetch above them: the scheduler otherwise sinks each ds_load pair next to its s_wait_dscnt; NVFP4
    // 17408x5120 @M=2048 3127 -> 2983 us on the 256x128 tile). What is exposed per slab is the barrier plus the
    // first group's fragment latency, so BK=64 (4 NVFP4 / 2 MXFP4 groups per slab, LDS 2 x 27 KB on 256x128)
    // halves that cost per K. A split-barrier (s_barrier_signal / s_barrier_wait) software pipeline was measured
    // and rejected: keeping the compiler from hoisting the signal above the group's WMMAs needs the accumulators
    // as asm operands, which costs ~60 VGPRs (208, one workgroup per WGP) and ends up slower than this loop.
    PF_LOAD(0)
    for (int slab = 0; slab < nslabs; slab++) {
      const int buf = slab & 1;
      const int off = buf * BUF_BYTES;
      PF_STORE(buf)
      if (!(R9K_PF_ABL & 4)) lds_barrier();
      const bool more = slab + 1 < nslabs;
      if (!(R9K_PF_ABL & 1) && more) PF_LOAD(slab + 1)
      PF_FRAG_LOAD(off, 0, afc, wfc, sbc)
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int g = 0; g < G; g++) {
        if (g + 1 < G) {
          PF_FRAG_LOAD(off, g + 1, afn, wfn, sbn)
          __builtin_amdgcn_sched_barrier(0);
        }
        PF_COMPUTE(0, TN, afc, wfc, sbc)
        if (g + 1 < G) PF_ROTATE()
      }
    }
  }
#undef PF_ROTATE
#undef PF_COMPUTE
#undef PF_FRAG_LOAD
#undef PF_STORE
#undef PF_LOAD

  // Epilogue: lane holds rows mhalf..+8 of column nlane in every 16x16 tile; rows scatter through sRow.
#pragma unroll
  for (int j = 0; j < TN; j++) {
    const int n = (nt0 + wn * TN + j) * 16 + nlane;
    if (n >= N) continue;
    const float wg = NV ? Wg[n] : (FOLD ? __uint_as_float((unsigned)Wref[n] << 23) : 1.f);
#pragma unroll
    for (int i = 0; i < TM; i++) {
#pragma unroll
      for (int q = 0; q < 8; q++) {
        const int fr = sRow[wm * TM * 16 + i * 16 + mhalf + q];
        if (fr < 0) continue;
        float sc = wg * As[fr / a_row_div];
        if (topk_w) sc *= topk_w[fr];
        C[(size_t)fr * N + n] = (__hip_bfloat16)(acc[i][j][q] * sc);
      }
    }
  }
}

// Tile table (cfg index -> TM, TN, WM, WN, BK, DB); BM = WM*TM*16 rows per routing block, BN = WN*TN*16 columns.
struct PfCfg { int TM, TN, WM, WN, BK, DB; };
static const PfCfg kPfCfgs[] = {
  {4, 2, 4, 2, 64, 0},   // 0: 256 x 64,  8 waves
  {4, 4, 4, 2, 64, 0},   // 1: 256 x 128, 8 waves
  {4, 2, 2, 4, 64, 0},   // 2: 128 x 128, 8 waves
  {4, 2, 2, 2, 64, 0},   // 3: 128 x 64,  4 waves
  {2, 2, 4, 2, 64, 0},   // 4: 128 x 64,  8 waves
  {4, 1, 4, 2, 64, 0},   // 5: 256 x 32,  8 waves
  {2, 2, 2, 2, 64, 0},   // 6: 64 x 64,   4 waves
  {4, 2, 4, 4, 64, 0},   // 7: 256 x 128, 16 waves
  {4, 2, 4, 4, 32, 1},   // 8: 256 x 128, 16 waves, BK=32 double-buffered
  {4, 4, 4, 2, 32, 1},   // 9: 256 x 128, 8 waves,  BK=32 double-buffered
  {4, 2, 2, 4, 32, 1},   // 10: 128 x 128, 8 waves, BK=32 double-buffered
  {2, 2, 2, 2, 32, 1},   // 11: 64 x 64,  4 waves,  BK=32 double-buffered
  {4, 2, 4, 4, 64, 1},   // 12: 256 x 128, 16 waves, BK=64 double-buffered (2 x 27 KB: two workgroups per WGP)
  {4, 2, 2, 4, 64, 1},   // 13: 128 x 128, 8 waves,  BK=64 double-buffered
  {2, 2, 2, 2, 64, 1},   // 14: 64 x 64,   4 waves,  BK=64 double-buffered
  {2, 2, 2, 4, 64, 1},   // 15: 64 x 128,  8 waves,  BK=64 double-buffered (routed gate_up: N=640 = 5 tiles)
  {4, 4, 4, 2, 64, 1},   // 16: 256 x 128, 8 waves,  BK=64 double-buffered
};
#define PF_NCFG ((int)(sizeof(kPfCfgs) / sizeof(kPfCfgs[0])))

extern "C" int r9k_moe_prefill_bm(int cfg) {     // rows per routing block of a prefill cfg (-1: no such cfg)
  if (cfg < 0 || cfg >= PF_NCFG) return -1;
  return kPfCfgs[cfg].WM * kPfCfgs[cfg].TM * 16;
}

// Prefill entry: same operands / routing tables as r9k_moe_mxfp4a8 / r9k_moe_nvfp4a8 (nv selects the scale
// format: 0 MXFP4, 1 NVFP4, 2 MXFP4 with the block exponents folded into the weights), routing block =
// r9k_moe_prefill_bm(cfg) rows, tile = cfg. Shape rules: N % 16 == 0, K % 64 == 0.
extern "C" int r9k_moe_4bit_prefill(int nv, long a, long as, long wq, long ws, long wref, long c,
                                    long sorted_ids, long expert_ids, long ntpp, long topk_w,
                                    long ws_estride, long wref_estride, int max_blocks, int numel, int a_row_div,
                                    int K, int N, int cfg, long stream) {
  if (N % 16) return -1;
  if (K % 64) return -2;
  if (cfg < 0 || cfg >= PF_NCFG) return -7;
  if (nv < 0 || nv > 2) return -8;
  hipStream_t st = (hipStream_t)stream;
#define PL(TMv, TNv, WMv, WNv, NVv, BKv, DBv, FOv)                                                         \
  {                                                                                                        \
    constexpr int BM = WMv * TMv * 16, BN = WNv * TNv * 16, NT = WMv * WNv * WAVE32;                        \
    constexpr int G = BKv / (NVv ? 16 : GROUP);                                                            \
    constexpr size_t ssb = FOv ? 0 : (((size_t)BN * G + 15) & ~(size_t)15);                                \
    constexpr size_t buf = (size_t)BM * (BKv + 8) + (size_t)BN * BKv + ssb;                                \
    const size_t smem = (DBv ? 2 : 1) * buf + (size_t)BM * 4;                                              \
    if (smem > 64 * 1024) return -4;                                                                       \
    dim3 grid((N / 16 + BN / 16 - 1) / (BN / 16), max_blocks), block(NT);                                  \
    hipLaunchKernelGGL((r9k_moe_4bit_prefill_kernel<TMv, TNv, WMv, WNv, NVv, BKv, DBv, FOv>), grid, block, smem, st, \
        (const unsigned char*)a, (const float*)as, (const unsigned int*)wq, (const unsigned char*)ws,       \
        (const unsigned char*)wref, (__hip_bfloat16*)c, (const int*)sorted_ids, (const int*)expert_ids,    \
        (const int*)ntpp, (const float*)topk_w, ws_estride, wref_estride, numel, a_row_div, K, N);         \
  }
#define PLN(TMv, TNv, WMv, WNv, BKv, DBv) \
  do { if (nv == 1) PL(TMv, TNv, WMv, WNv, true, BKv, DBv, false) \
       else if (nv == 2) PL(TMv, TNv, WMv, WNv, false, BKv, DBv, true) \
       else PL(TMv, TNv, WMv, WNv, false, BKv, DBv, false) } while (0)
  switch (cfg) {
    case 0: PLN(4, 2, 4, 2, 64, false); break;
    case 1: PLN(4, 4, 4, 2, 64, false); break;
    case 2: PLN(4, 2, 2, 4, 64, false); break;
    case 3: PLN(4, 2, 2, 2, 64, false); break;
    case 4: PLN(2, 2, 4, 2, 64, false); break;
    case 5: PLN(4, 1, 4, 2, 64, false); break;
    case 6: PLN(2, 2, 2, 2, 64, false); break;
    case 7: PLN(4, 2, 4, 4, 64, false); break;
    case 8: PLN(4, 2, 4, 4, 32, true); break;
    case 9: PLN(4, 4, 4, 2, 32, true); break;
    case 10: PLN(4, 2, 2, 4, 32, true); break;
    case 11: PLN(2, 2, 2, 2, 32, true); break;
    case 12: PLN(4, 2, 4, 4, 64, true); break;
    case 13: PLN(4, 2, 2, 4, 64, true); break;
    case 14: PLN(2, 2, 2, 2, 64, true); break;
    case 15: PLN(2, 2, 2, 4, 64, true); break;
    case 16: PLN(4, 4, 4, 2, 64, true); break;
    default: return -7;
  }
#undef PLN
#undef PL
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

// ---------------------------------------------------------------- A-tiled prefill kernel (fragment-tiled activations)
// Same tile family, W staging (unpack through LDS, register-prefetched a slab ahead, one LDS-only barrier per
// slab), scale handling (FOLD) and epilogue as the prefill kernel above, but the activation comes in WMMA-FRAGMENT-TILED layout
// (r9k_quant_rows_fp8_tiled): fragment (mt, ks) of the 16-row block mt and K step ks is 256 contiguous bytes at
// (mt * K/16 + ks) * 256, lane l holding A[mt*16 + l%16][ks*16 + (l/16)*8 .. +8] -- exactly the bytes the WMMA
// wants in the lane's register. So a wave reads each of its TM x (BK/16) A fragments with ONE coalesced 256-byte
// global_load_b64 straight into the af[] register the WMMA consumes: no A slab in LDS, no per-fragment ds_read,
// no staging stores, and the LDS footprint drops from 2 x 26 KB (256x128 tile) to 2 x 8 KB, so more workgroups
// fit a CU. The price is that the WN waves sharing an M row read the same A fragments WN times from L2 (the LDS
// version read them once per workgroup), which is why the default tiles use WN = 2.
// The tiled buffer is indexed by SORTED POSITION (row p of routing block by = position by*BM + r), never through
// sorted_ids: dense callers pass identity tables so position = row; a routed caller has to write the tiled buffer
// in sorted order itself. Rows past the buffer's last 16-row block (mt_count) clamp to it (tile-granular, never
// predicated: they only reach accumulators the epilogue drops through sRow).
// FOLD (nv = 2): the slab's weights are unpacked with the block exponent folded into the e4m3 bytes and the epilogue
// applies 2^(Wref[n] - 127), exactly as in the folded LDS-A kernel. FOLD = false (nv = 0): the fp32-per-group
// variant of the LDS-A kernel -- the slab's E8M0 bytes (BN x G) are staged in LDS behind the weight slab (SS_BYTES,
// SCH bytes per thread, prefetched with the weights), each scale group's WMMAs go to a zeroed partial that is scaled
// into acc with explicit fmaf, and the epilogue applies As[row] (* topk_w) only. Same accumulation order as the
// corresponding LDS-A variant, so the outputs are bit-identical to it either way. The scale slab is BN*G bytes
// (256 B on the 256x128 tile) against the BM*BK = 16 KB A slab the tiled activation removes.
template <int TM, int TN, int WM, int WN, int BK, bool DB, bool FOLD>
__global__ void __launch_bounds__(WM * WN * WAVE32) r9k_moe_4bit_prefill_at_kernel(
    const unsigned char* __restrict__ AT, const float* __restrict__ As,
    const unsigned int* __restrict__ Wp_all, const unsigned char* __restrict__ Ws_all,
    const unsigned char* __restrict__ Wref_all, __hip_bfloat16* __restrict__ C,
    const int* __restrict__ sorted_ids, const int* __restrict__ expert_ids,
    const int* __restrict__ ntpp, const float* __restrict__ topk_w,
    long ws_estride, long wref_estride, int numel, int a_row_div, int mt_count, int K, int N) {
  constexpr int BM = WM * TM * 16, BN = WN * TN * 16, NT = WM * WN * WAVE32;
  constexpr int KSPS = BK / 16;                             // WMMA K steps per slab
  constexpr int SPG = GROUP / 16, G = BK / GROUP;           // K steps per scale group, groups per slab
  static_assert(G >= 1, "slab must hold a whole scale group");
  constexpr int WSL = BN * (BK / 8) / NT;                   // uint32 weight slots per thread per slab
  static_assert(WSL * NT == BN * (BK / 8) && (WSL == 1 || WSL == 2 || WSL == 4), "W slab must split evenly");
  constexpr int SCH = FOLD ? 1 : (BN * G + NT - 1) / NT;    // scale bytes per thread per slab (guarded; FOLD: none)
  constexpr int SW_BYTES = BN * BK, SS_BYTES = FOLD ? 0 : ((BN * G + 15) & ~15);
  constexpr int BUF_BYTES = SW_BYTES + SS_BYTES, NBUF = DB ? 2 : 1;
  extern __shared__ __align__(16) unsigned char pf_smem[];
  int* const sRow = (int*)(pf_smem + NBUF * BUF_BYTES);

  const int bx = blockIdx.x, by = blockIdx.y;
  if (by * BM >= ntpp[0]) return;
  const int e = expert_ids[by];
  if (e < 0) return;
  const int first = sorted_ids[by * BM];
  if (first < 0 || first >= numel) return;

  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int wm = wave / WN, wn = wave % WN;
  const int nlane = lane & 15, mhalf = (lane >> 4) * 8;
  const int ksteps = K / 16, ntmax = N / 16 - 1;
  const int nt0 = bx * (BN / 16);
  const unsigned int* __restrict__ Wp = Wp_all + (size_t)e * (N / 16) * ksteps * WAVE32;
  const unsigned char* __restrict__ Ws = Ws_all + (size_t)e * ws_estride;
  const unsigned char* __restrict__ Wref = FOLD ? Wref_all + (size_t)e * wref_estride : nullptr;

  for (int r = tid; r < BM; r += NT) {
    const int fl = sorted_ids[by * BM + r];
    sRow[r] = (fl < 0 || fl >= numel) ? -1 : fl;            // -1: padding row (computed, never stored)
  }
  __syncthreads();

  // A fragment bases: one wave-uniform (SGPR) byte pointer per M fragment, lane part an unsigned 32-bit offset
  // (saddr-form loads: no 64-bit VALU address chains). Clamped to the last block of the tiled buffer.
  const unsigned char* abase[TM];
#pragma unroll
  for (int i = 0; i < TM; i++) {
    int mt = by * (BM / 16) + wm * TM + i;
    mt = mt < mt_count - 1 ? mt : mt_count - 1;
    abase[i] = AT + (size_t)__builtin_amdgcn_readfirstlane(mt * ksteps * 256);
  }
  const unsigned aoff = (unsigned)lane * 8u;

  // Weight slots s = tid*WSL + q (contiguous, never straddling a 32-lane fragment): slot -> (n tile, K step, lane).
  unsigned w_src, w_dst, ws_own = 0;                        // ws_own: FOLD scale bytes of the slots' columns, slab 0
  int wref_own[WSL];
  FoldTab ftab;
  {
    const int s0 = tid * WSL, lane_ = s0 & 31, rest = s0 >> 5, ks = rest % KSPS, ntl = rest / KSPS;
    const int ntg = nt0 + ntl < ntmax ? nt0 + ntl : ntmax;   // clamped: garbage columns are never stored
    w_src = (unsigned)(((ntg * ksteps + ks) * WAVE32 + lane_) * 4);
    w_dst = (unsigned)(((ntl * KSPS + ks) * WAVE32 + lane_) * 8);
    if constexpr (FOLD) {
      const int col0 = ntg * 16 + (lane_ & 15);              // lane_ % WSL == 0: the WSL columns are aligned
      ws_own = (unsigned)((ks / SPG) * N + col0);
      _Pragma("unroll")
      for (int q = 0; q < WSL; q++) wref_own[q] = Wref[col0 + q];
      ftab = fold_tab_load();
    }
  }
  // Scale bytes t = tid + it*NT -> (group t / BN, local column t % BN), Ws is [K/32][N] per expert (!FOLD only).
  unsigned s_src[SCH];
#pragma unroll
  for (int it = 0; it < SCH; it++) {
    const int t = tid + it * NT, g = t / BN, nl = t % BN;
    const int ntg = nt0 + nl / 16 < ntmax ? nt0 + nl / 16 : ntmax;
    s_src[it] = (unsigned)(g * N + ntg * 16 + (nl & 15));
  }

  v8f acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++) acc[i][j] = (v8f)(0.f);

  unsigned int pw[WSL];
  unsigned int psw = 0;                                     // FOLD: the WSL scale bytes of this thread's slots
  unsigned char ps[SCH];                                    // !FOLD: this thread's scale bytes of the slab
  int se[TN];                                               // R9K_AT_RESCALE: acc frame exponent per column (E8M0)
#pragma unroll
  for (int j = 0; j < TN; j++) se[j] = 127;
  const int nslabs = K / BK;
#define AT_LOADW(SLAB)                                                                                  \
  {                                                                                                     \
    const int slab_ = (SLAB);                                                                           \
    if constexpr (WSL == 4) {                                                                           \
      const uint4 v = ld_u32off((const uint4*)Wp, w_src + (unsigned)(slab_ * KSPS * WAVE32 * 4));      \
      pw[0] = v.x; pw[1] = v.y; pw[2] = v.z; pw[3] = v.w;                                               \
    } else if constexpr (WSL == 2) {                                                                    \
      const uint2 v = ld_u32off((const uint2*)Wp, w_src + (unsigned)(slab_ * KSPS * WAVE32 * 4));      \
      pw[0] = v.x; pw[1] = v.y;                                                                         \
    } else {                                                                                            \
      pw[0] = ld_u32off(Wp, w_src + (unsigned)(slab_ * KSPS * WAVE32 * 4));                            \
    }                                                                                                   \
    if constexpr (FOLD) {                                                                               \
      psw = ld_scale_bytes<WSL>(Ws, ws_own + (unsigned)(slab_ * G) * (unsigned)N);                      \
    } else {                                                                                            \
      _Pragma("unroll")                                                                                 \
      for (int it = 0; it < SCH; it++)                                                                  \
        if (tid + it * NT < BN * G) ps[it] = ld_u32off(Ws, s_src[it] + (unsigned)(slab_ * G) * (unsigned)N); \
    }                                                                                                   \
  }
#define AT_STOREW(BUF)                                                                                  \
  {                                                                                                     \
    unsigned char* const sb_ = pf_smem + (BUF) * BUF_BYTES;                                             \
    if constexpr (FOLD) {                                                                               \
      _Pragma("unroll")                                                                                 \
      for (int q = 0; q < WSL; q++)                                                                     \
        *(v2i*)(sb_ + w_dst + q * 8) =                                                                  \
            mxfp4_unpack8_fold(pw[q], ftab, fold_dsh(wref_own[q], (int)((psw >> (8 * q)) & 0xFFu)));   \
    } else {                                                                                            \
      _Pragma("unroll")                                                                                 \
      for (int q = 0; q < WSL; q++) *(v2i*)(sb_ + w_dst + q * 8) = mxfp4_unpack8(pw[q], 0);             \
      _Pragma("unroll")                                                                                 \
      for (int it = 0; it < SCH; it++)                                                                  \
        if (tid + it * NT < BN * G) sb_[SW_BYTES + tid + it * NT] = ps[it];                             \
    }                                                                                                   \
  }
  // A fragments of a slab: TM x KSPS coalesced 256-byte wave reads, issued before the slab's W store + barrier so
  // their latency overlaps that work; consumed only by the WMMAs.
#define AT_LOADA(SLAB, AF)                                                                              \
  {                                                                                                     \
    const unsigned ko_ = aoff + (unsigned)((SLAB) * KSPS * 256);                                        \
    _Pragma("unroll")                                                                                   \
    for (int i = 0; i < TM; i++)                                                                        \
      _Pragma("unroll")                                                                                 \
      for (int st = 0; st < KSPS; st++) AF[i][st] = *(const v2i*)(abase[i] + ko_ + (unsigned)(st * 256)); \
  }
  const unsigned char* const myW = pf_smem + (wn * TN * KSPS * WAVE32 + lane) * 8;
  const unsigned char* const myS = pf_smem + SW_BYTES + wn * TN * 16 + nlane;
  // FOLD: the slab's KSPS x TN x TM WMMAs straight into acc. !FOLD: per scale group, unscaled e2m1 x e4m3 partials
  // over the group's SPG K steps, scaled into acc (the per-(n, group) E8M0 is one scalar per lane: 8 FMAs per
  // group per tile, exact weight math; -ffp-contract=off keeps the fmaf explicit).
#define AT_COMPUTE(OFF, AF)                                                                              \
  {                                                                                                      \
    v2i wf[KSPS][TN];                                                                                    \
    _Pragma("unroll")                                                                                    \
    for (int st = 0; st < KSPS; st++)                                                                    \
      _Pragma("unroll")                                                                                  \
      for (int j = 0; j < TN; j++) wf[st][j] = *(const v2i*)(myW + (OFF) + (j * KSPS + st) * (WAVE32 * 8)); \
    if constexpr (FOLD) {                                                                                \
      _Pragma("unroll")                                                                                  \
      for (int st = 0; st < KSPS; st++)                                                                  \
        _Pragma("unroll")                                                                                \
        for (int j = 0; j < TN; j++)                                                                     \
          _Pragma("unroll")                                                                              \
          for (int i = 0; i < TM; i++) {                                                                 \
            if (R9K_PF_ABL & 2) { acc[i][j][0] += __int_as_float(AF[i][st][0] ^ wf[st][j][1]); continue; } \
            acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(AF[i][st], wf[st][j], acc[i][j]); \
          }                                                                                              \
    } else if constexpr (R9K_AT_RESCALE) {                                                               \
      /* EXPERIMENT: acc[.][j] is kept in units of 2^(se[j] - 127), the exponent of the last group folded  \
         in; a new group multiplies acc by the (exact, power-of-two) ratio 2^(se - e_g) and the WMMAs      \
         accumulate straight into acc: 8 v_mul per tile per group instead of 8 v_mov + 8 v_fma. Exact      \
         weight math (no kMag, no flush; only fp32 accumulation rounding), but NOT bit-identical to the     \
         LDS-A kernel. |se - e_g| is assumed < 127 (a checkpoint's rows span ~16 binades; clamped anyway). */ \
      unsigned int sb[G][TN];                                                                            \
      _Pragma("unroll")                                                                                  \
      for (int g = 0; g < G; g++)                                                                        \
        _Pragma("unroll")                                                                                \
        for (int j = 0; j < TN; j++) sb[g][j] = myS[(OFF) + g * BN + j * 16];                            \
      _Pragma("unroll")                                                                                  \
      for (int g = 0; g < G; g++)                                                                        \
        _Pragma("unroll")                                                                                \
        for (int j = 0; j < TN; j++) {                                                                   \
          int d_ = se[j] - (int)sb[g][j];                                                                \
          d_ = d_ < -126 ? -126 : (d_ > 127 ? 127 : d_);                                                 \
          const float r_ = __uint_as_float((unsigned)(d_ + 127) << 23);                                  \
          se[j] = (int)sb[g][j];                                                                         \
          _Pragma("unroll")                                                                              \
          for (int i = 0; i < TM; i++) {                                                                 \
            _Pragma("unroll")                                                                            \
            for (int q = 0; q < 8; q++) acc[i][j][q] *= r_;                                              \
            _Pragma("unroll")                                                                            \
            for (int sidx = 0; sidx < SPG; sidx++) {                                                     \
              if (R9K_PF_ABL & 2) { acc[i][j][0] += __int_as_float(AF[i][g * SPG + sidx][0] ^ wf[g * SPG + sidx][j][1]); continue; } \
              acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(AF[i][g * SPG + sidx], wf[g * SPG + sidx][j], acc[i][j]); \
            }                                                                                            \
          }                                                                                              \
        }                                                                                                \
    } else {                                                                                             \
      unsigned int sb[G][TN];                                                                            \
      _Pragma("unroll")                                                                                  \
      for (int g = 0; g < G; g++)                                                                        \
        _Pragma("unroll")                                                                                \
        for (int j = 0; j < TN; j++) sb[g][j] = myS[(OFF) + g * BN + j * 16];                            \
      _Pragma("unroll")                                                                                  \
      for (int g = 0; g < G; g++)                                                                        \
        _Pragma("unroll")                                                                                \
        for (int j = 0; j < TN; j++) {                                                                   \
          v8f t[TM];                                                                                     \
          _Pragma("unroll")                                                                              \
          for (int i = 0; i < TM; i++) t[i] = (v8f)(0.f);                                                \
          _Pragma("unroll")                                                                              \
          for (int sidx = 0; sidx < SPG; sidx++)                                                         \
            _Pragma("unroll")                                                                            \
            for (int i = 0; i < TM; i++) {                                                               \
              if (R9K_PF_ABL & 2) { t[i][0] += __int_as_float(AF[i][g * SPG + sidx][0] ^ wf[g * SPG + sidx][j][1]); continue; } \
              t[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(AF[i][g * SPG + sidx], wf[g * SPG + sidx][j], t[i]); \
            }                                                                                            \
          const float sc = __uint_as_float(sb[g][j] << 23);                                              \
          _Pragma("unroll")                                                                              \
          for (int i = 0; i < TM; i++)                                                                   \
            _Pragma("unroll")                                                                            \
            for (int q = 0; q < 8; q++) acc[i][j][q] = fmaf(t[i][q], sc, acc[i][j][q]);                  \
        }                                                                                                \
    }                                                                                                    \
  }
  v2i af[TM][KSPS];
  AT_LOADW(0)
  if constexpr (!DB) {
    for (int slab = 0; slab < nslabs; slab++) {
      AT_LOADA(slab, af)
      AT_STOREW(0)
      if (!(R9K_PF_ABL & 4)) lds_barrier();
      if (!(R9K_PF_ABL & 1) && slab + 1 < nslabs) AT_LOADW(slab + 1)
      AT_COMPUTE(0, af)
      if (!(R9K_PF_ABL & 4)) lds_barrier();
    }
  } else {
    // One barrier per slab (as in the folded kernel): the store of slab s overwrites buffer s&1, last read during
    // slab s-2; every wave stores slab s only after passing barrier s-1, which it reaches only after finishing its
    // slab s-2 WMMAs, so no wave can still be reading that buffer.
    for (int slab = 0; slab < nslabs; slab++) {
      const int buf = slab & 1;
      AT_LOADA(slab, af)
      AT_STOREW(buf)
      if (!(R9K_PF_ABL & 4)) lds_barrier();
      if (!(R9K_PF_ABL & 1) && slab + 1 < nslabs) AT_LOADW(slab + 1)
      AT_COMPUTE(buf * BUF_BYTES, af)
    }
  }
#undef AT_COMPUTE
#undef AT_LOADA
#undef AT_STOREW
#undef AT_LOADW

  // Epilogue: lane holds rows mhalf..+8 of column nlane in every 16x16 tile; rows scatter through sRow.
#pragma unroll
  for (int j = 0; j < TN; j++) {
    const int n = (nt0 + wn * TN + j) * 16 + nlane;
    if (n >= N) continue;
    const float wg = FOLD ? __uint_as_float((unsigned)Wref[n] << 23)
                          : (R9K_AT_RESCALE ? __uint_as_float((unsigned)se[j] << 23) : 1.f);
#pragma unroll
    for (int i = 0; i < TM; i++) {
#pragma unroll
      for (int q = 0; q < 8; q++) {
        const int fr = sRow[wm * TM * 16 + i * 16 + mhalf + q];
        if (fr < 0) continue;
        float sc = wg * As[fr / a_row_div];
        if (topk_w) sc *= topk_w[fr];
        C[(size_t)fr * N + n] = (__hip_bfloat16)(acc[i][j][q] * sc);
      }
    }
  }
}

// Tile table of the A-tiled kernel (cfg index -> TM, TN, WM, WN, BK, DB); BM = WM*TM*16, BN = WN*TN*16.
static const PfCfg kAtCfgs[] = {
  {4, 2, 4, 2, 64, 1},    // 0: 256 x 64,  8 waves, BK=64
  {4, 4, 4, 2, 64, 1},    // 1: 256 x 128, 8 waves, BK=64
  {4, 2, 4, 2, 128, 1},   // 2: 256 x 64,  8 waves, BK=128 (half the barriers per K; 64 A VGPRs)
  {4, 2, 4, 4, 64, 1},    // 3: 256 x 128, 16 waves (the LDS kernel's cfg-12 geometry)
  {4, 2, 2, 4, 64, 1},    // 4: 128 x 128, 8 waves
  {2, 2, 4, 2, 64, 1},    // 5: 128 x 64,  8 waves (small M / grid fill)
  {4, 4, 4, 2, 64, 0},    // 6: 256 x 128, 8 waves, single-buffered (two barriers per slab)
  {4, 2, 4, 2, 64, 0},    // 7: 256 x 64,  8 waves, single-buffered
  {2, 4, 4, 2, 64, 1},    // 8: 128 x 128, 8 waves, TM=2 TN=4
};
#define AT_NCFG ((int)(sizeof(kAtCfgs) / sizeof(kAtCfgs[0])))

extern "C" int r9k_moe_atiled_bm(int cfg) {      // rows per routing block of an A-tiled cfg (-1: no such cfg)
  if (cfg < 0 || cfg >= AT_NCFG) return -1;
  return kAtCfgs[cfg].WM * kAtCfgs[cfg].TM * 16;
}

// A-tiled prefill entry: operands as r9k_moe_4bit_prefill with nv = 2 (fold = 1, folded MXFP4) or nv = 0 (fold = 0,
// fp32-per-group MXFP4; wref is then unused), `a` the fragment-tiled activation of mt_count 16-row blocks
// (r9k_quant_rows_fp8_tiled) in sorted-position order. K % BK == 0.
extern "C" int r9k_moe_4bit_prefill_at(int fold, long a, long as, long wq, long ws, long wref, long c,
                                       long sorted_ids, long expert_ids, long ntpp, long topk_w,
                                       long ws_estride, long wref_estride, int max_blocks, int numel, int a_row_div,
                                       int mt_count, int K, int N, int cfg, long stream) {
  if (N % 16) return -1;
  if (cfg < 0 || cfg >= AT_NCFG) return -7;
  if (K % kAtCfgs[cfg].BK) return -2;
  if (fold < 0 || fold > 1) return -8;
  if (mt_count <= 0) return -9;
  hipStream_t st = (hipStream_t)stream;
#define AL(TMv, TNv, WMv, WNv, BKv, DBv, FOv)                                                             \
  {                                                                                                        \
    constexpr int BM = WMv * TMv * 16, BN = WNv * TNv * 16, NT = WMv * WNv * WAVE32;                        \
    constexpr size_t ssb = FOv ? 0 : (((size_t)BN * (BKv / GROUP) + 15) & ~(size_t)15);                    \
    const size_t smem = (DBv ? 2 : 1) * ((size_t)BN * BKv + ssb) + (size_t)BM * 4;                         \
    dim3 grid((N / 16 + BN / 16 - 1) / (BN / 16), max_blocks), block(NT);                                  \
    hipLaunchKernelGGL((r9k_moe_4bit_prefill_at_kernel<TMv, TNv, WMv, WNv, BKv, DBv, FOv>), grid, block, smem, st, \
        (const unsigned char*)a, (const float*)as, (const unsigned int*)wq, (const unsigned char*)ws,       \
        (const unsigned char*)wref, (__hip_bfloat16*)c, (const int*)sorted_ids, (const int*)expert_ids,    \
        (const int*)ntpp, (const float*)topk_w, ws_estride, wref_estride, numel, a_row_div, mt_count, K, N); \
  }
#define ALF(TMv, TNv, WMv, WNv, BKv, DBv) \
  do { if (fold) AL(TMv, TNv, WMv, WNv, BKv, DBv, true) else AL(TMv, TNv, WMv, WNv, BKv, DBv, false) } while (0)
  switch (cfg) {
    case 0: ALF(4, 2, 4, 2, 64, true); break;
    case 1: ALF(4, 4, 4, 2, 64, true); break;
    case 2: ALF(4, 2, 4, 2, 128, true); break;
    case 3: ALF(4, 2, 4, 4, 64, true); break;
    case 4: ALF(4, 2, 2, 4, 64, true); break;
    case 5: ALF(2, 2, 4, 2, 64, true); break;
    case 6: ALF(4, 4, 4, 2, 64, false); break;
    case 7: ALF(4, 2, 4, 2, 64, false); break;
    case 8: ALF(2, 4, 4, 2, 64, true); break;
    default: return -7;
  }
#undef ALF
#undef AL
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

// Vectorized single-pass variant: each thread keeps its 16-byte chunks of the row in registers (one DRAM read),
// block max via wave shuffles + LDS, then hardware OCP-e4m3 conversion (v_cvt_pk_fp8_f32) and 8-byte stores.
// Needs K % 8 == 0, 16-byte aligned rows and K <= RQ_THREADS * RQ_VEC * 8. ~4x fewer instructions than the
// scalar two-pass kernel, which cost ~8 us per M<=8 call inside the decode graphs (266 calls per 27B step).
#define RQ_THREADS 512
#define RQ_VEC 5
__device__ __forceinline__ float bf16lo(unsigned int w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16hi(unsigned int w) { return __uint_as_float(w & 0xFFFF0000u); }

// TILED: the fp8 bytes go to the WMMA-fragment-tiled layout of r9k_moe_4bit_prefill_at (8-element group c of row
// r = uint2 slot ((r/16) * K/16 + c/2) * 32 + r%16 + 16 * (c%2)); same 8-byte store per group, only its address
// differs (K % 16 == 0). Rows the caller pads (M .. 16*ceil(M/16)) are never written: the GEMM computes on them
// and drops them.
template <bool TILED>
__global__ void __launch_bounds__(RQ_THREADS) r9k_quant_rows_fp8_v2_kernel(
    const __hip_bfloat16* __restrict__ x, unsigned char* __restrict__ q, float* __restrict__ s, int K, int ld) {
  const int row = blockIdx.x, tid = threadIdx.x, nvec = K / 8;
  const uint4* xr = (const uint4*)(x + (size_t)row * ld);
  uint4 v[RQ_VEC];
  float m = 0.f;
#pragma unroll
  for (int i = 0; i < RQ_VEC; i++) {
    const int c = tid + i * RQ_THREADS;
    v[i] = c < nvec ? xr[c] : make_uint4(0, 0, 0, 0);
    const unsigned int w[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
#pragma unroll
    for (int j = 0; j < 4; j++) m = fmaxf(m, fmaxf(fabsf(bf16lo(w[j])), fabsf(bf16hi(w[j]))));
  }
  __shared__ float wmax[RQ_THREADS / 32];
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
  if ((tid & 31) == 0) wmax[tid >> 5] = m;
  __syncthreads();
  if (tid < 32) {
    float t = tid < RQ_THREADS / 32 ? wmax[tid] : 0.f;
    for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor(t, o, 32));
    if (tid == 0) wmax[0] = t;
  }
  __syncthreads();
  const float amax = fmaxf(wmax[0], 1e-12f);
  const float inv = 448.f / amax;
  if (tid == 0) s[row] = amax / 448.f;
  uint2* qr = TILED ? (uint2*)q + (size_t)(row >> 4) * (K / 16) * 32 + (row & 15) : (uint2*)(q + (size_t)row * K);
#pragma unroll
  for (int i = 0; i < RQ_VEC; i++) {
    const int c = tid + i * RQ_THREADS;
    if (c >= nvec) break;
    int lo = __builtin_amdgcn_cvt_pk_fp8_f32(bf16lo(v[i].x) * inv, bf16hi(v[i].x) * inv, 0, false);
    lo = __builtin_amdgcn_cvt_pk_fp8_f32(bf16lo(v[i].y) * inv, bf16hi(v[i].y) * inv, lo, true);
    int hi = __builtin_amdgcn_cvt_pk_fp8_f32(bf16lo(v[i].z) * inv, bf16hi(v[i].z) * inv, 0, false);
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(bf16lo(v[i].w) * inv, bf16hi(v[i].w) * inv, hi, true);
    qr[TILED ? (c >> 1) * 32 + 16 * (c & 1) : c] = make_uint2((unsigned int)lo, (unsigned int)hi);
  }
}

extern "C" int r9k_quant_rows_fp8(long x, long q, long s, int M, int K, int ld, long stream) {
  if (M <= 0) return 0;
  if (K % 8 == 0 && ld % 8 == 0 && (x & 15) == 0 && (q & 7) == 0 && K <= RQ_THREADS * RQ_VEC * 8)
    hipLaunchKernelGGL(r9k_quant_rows_fp8_v2_kernel<false>, dim3(M), dim3(RQ_THREADS), 0, (hipStream_t)stream,
                       (const __hip_bfloat16*)x, (unsigned char*)q, (float*)s, K, ld);
  else
    hipLaunchKernelGGL(r9k_quant_rows_fp8_kernel, dim3(M), dim3(256), 0, (hipStream_t)stream,
                       (const __hip_bfloat16*)x, (unsigned char*)q, (float*)s, K, ld);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

// Fragment-tiled variant (see r9k_quant_rows_fp8_v2_kernel<true>): q must hold 16*ceil(M/16) * K bytes. Needs
// K % 16 == 0 and the vectorized kernel's limits (K <= 20480, 16-byte aligned rows); -3 otherwise.
extern "C" int r9k_quant_rows_fp8_tiled(long x, long q, long s, int M, int K, int ld, long stream) {
  if (M <= 0) return 0;
  if (!(K % 16 == 0 && ld % 8 == 0 && (x & 15) == 0 && (q & 7) == 0 && K <= RQ_THREADS * RQ_VEC * 8)) return -3;
  hipLaunchKernelGGL(r9k_quant_rows_fp8_v2_kernel<true>, dim3(M), dim3(RQ_THREADS), 0, (hipStream_t)stream,
                     (const __hip_bfloat16*)x, (unsigned char*)q, (float*)s, K, ld);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}
