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

// NV = false: OCP MXFP4 (E8M0 per 32, folded into the e4m3 exponent -- exact, no extra math).
// NV = true:  NVFP4 (e4m3 scale per 16 = exactly one WMMA K step): the WMMA runs on the unscaled e2m1 values
//             (kMag[0]) and its 16x16 partial is scaled into the accumulator -- in the WMMA C layout each lane
//             holds 8 rows of ONE output column n, so the per-(n, kstep) scale is one scalar per lane: 8 FMAs per
//             WMMA, exact NVFP4 weight math. Ws = e4m3 [K/16][N] per expert, Wref = fp32 per-row global
//             multiplier [N] per expert (wref_estride in floats), applied in the epilogue.
template <int MT, int NPW, bool NV>
__global__ void r9k_moe_mxfp4a8_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned int* __restrict__ Wp_all, const unsigned char* __restrict__ Ws_all,
    const unsigned char* __restrict__ Wref_all, __hip_bfloat16* __restrict__ C,
    const int* __restrict__ sorted_ids, const int* __restrict__ expert_ids,
    const int* __restrict__ ntpp, const float* __restrict__ topk_w,
    long ws_estride, long wref_estride, int numel, int a_row_div, int K, int N, int WV, int SK) {
  // One workgroup row = one routing block of BLK = 16*MT sorted rows (moe_align_block_size(block=16*MT)),
  // processed as MT WMMA M-tiles that share every weight fragment (MT x fewer weight reads per row).
  constexpr int BLK = 16 * MT;
  const int by = blockIdx.y;
  if (by * BLK >= ntpp[0]) return;
  const int e = expert_ids[by];
  if (e < 0) return;

  extern __shared__ float red[];
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int Ks = K / SK, kbase = ksplit * Ks;
  const int nlane = lane & 15, mhalf = (lane >> 4) * 8;
  const int ksteps = K / 16;

  const unsigned int* __restrict__ Wp = Wp_all + (size_t)e * (N / 16) * ksteps * WAVE32;
  const unsigned char* __restrict__ Ws = Ws_all + (size_t)e * ws_estride;
  const unsigned char* __restrict__ Wref = NV ? nullptr : Wref_all + (size_t)e * wref_estride;
  const float* __restrict__ Wg = NV ? (const float*)Wref_all + (size_t)e * wref_estride : nullptr;

  // This lane's A row in each M tile: the sorted flat index, clamped to the block's first row (always valid
  // for a live block) for padding entries.
  const int first = sorted_ids[by * BLK];
  if (first < 0 || first >= numel) return;          // fully padded block: nothing valid to read or write
  const unsigned char* __restrict__ arow[MT];
#pragma unroll
  for (int i = 0; i < MT; i++) {
    int flat = sorted_ids[by * BLK + i * 16 + nlane];
    if (flat >= numel) flat = first;
    arow[i] = A + (size_t)(flat / a_row_div) * K;
  }

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  const int ntmax = N / 16 - 1;
  int wtile[NPW], wrow[NPW];
  unsigned char wref[NPW];
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    const int t = nt0 + j;
    wtile[j] = t < ntmax ? t : ntmax;
    wrow[j] = wtile[j] * 16 + nlane;
    wref[j] = NV ? 0 : Wref[wrow[j]];
  }

  // MT == 1 (decode): K loop unrolled x4 so the next steps' weight loads issue before this step's WMMAs retire --
  // each load is one coalesced 128 B wave request, so memory-level parallelism has to come from several in flight
  // (+5-14% DRAM-fed on 27B / Flash-Next shapes). MT > 1: x1 (x4 spills the accumulator set: NVFP4 M=64
  // 396 -> 1792 us). A plain loop, not a lambda: the lambda form lost the NVFP4 gain (90 -> 102 us).
  constexpr int KU = MT == 1 ? 4 : 1;
#pragma unroll KU
  for (int k = 0; k < Ks; k += 16) {
    const int kg = kbase + k;
    const int kstep = kg / 16;
    const int blk = kg / GROUP;
    v2i af[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) {
      const unsigned int* pa = (const unsigned int*)(arow[i] + kg + mhalf);
      af[i][0] = (int)pa[0];
      af[i][1] = (int)pa[1];
    }
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const unsigned int wv = Wp[((size_t)wtile[j] * ksteps + kstep) * WAVE32 + lane];
      if constexpr (NV) {
        const v2i wf = mxfp4_unpack8(wv, 0);
        const float sc = e4m3_to_f32(Ws[(size_t)kstep * N + wrow[j]]);
#pragma unroll
        for (int i = 0; i < MT; i++) {
          const v8f t = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, (v8f)(0.f));
          acc[j][i] += t * sc;
        }
      } else {
        int dsh = (int)wref[j] - (int)Ws[(size_t)blk * N + wrow[j]];
        dsh = dsh < 0 ? 0 : (dsh > 15 ? 15 : dsh);
        const v2i wf = mxfp4_unpack8(wv, dsh);
#pragma unroll
        for (int i = 0; i < MT; i++)
          acc[j][i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, acc[j][i]);
      }
    }
  }

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
      float sc = (NV ? Wg[n] : __int_as_float((int)Wref[n] << 23)) * As[fr / a_row_div];
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

static int moe_launch(bool nv, long a, long as, long wq, long ws, long wref, long c, long sorted_ids,
                      long expert_ids, long ntpp, long topk_w, long ws_estride, long wref_estride, int max_blocks,
                      int numel, int a_row_div, int K, int N, int WV, int SK, int NPW, int MT, long stream) {
  if (N % 16) return -1;
  if (K % (SK * (nv ? 16 : GROUP))) return -2;
  if (WV * SK * WAVE32 > 1024) return -3;
  const int ncols = WV * NPW;
  const size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024) return -4;
  dim3 grid((N / 16 + ncols - 1) / ncols, max_blocks), block(WV * SK * WAVE32);
  hipStream_t st = (hipStream_t)stream;
#define L(MTv, NP, NVv) hipLaunchKernelGGL((r9k_moe_mxfp4a8_kernel<MTv, NP, NVv>), grid, block, smem, st, \
      (const unsigned char*)a, (const float*)as, (const unsigned int*)wq, (const unsigned char*)ws, \
      (const unsigned char*)wref, (__hip_bfloat16*)c, (const int*)sorted_ids,                        \
      (const int*)expert_ids, (const int*)ntpp, (const float*)topk_w, ws_estride, wref_estride, numel,  \
      a_row_div, K, N, WV, SK)
#define LN(MTv, NP) do { if (nv) L(MTv, NP, true); else L(MTv, NP, false); } while (0)
#define LM(MTv) switch (NPW) { case 1: LN(MTv, 1); break; case 2: LN(MTv, 2); break; case 4: LN(MTv, 4); break; \
                               default: return -5; }
  switch (MT) {
    case 1: LM(1); break;
    case 2: LM(2); break;
    case 4: LM(4); break;
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

// Vectorized single-pass variant: each thread keeps its 16-byte chunks of the row in registers (one DRAM read),
// block max via wave shuffles + LDS, then hardware OCP-e4m3 conversion (v_cvt_pk_fp8_f32) and 8-byte stores.
// Needs K % 8 == 0, 16-byte aligned rows and K <= RQ_THREADS * RQ_VEC * 8. ~4x fewer instructions than the
// scalar two-pass kernel, which cost ~8 us per M<=8 call inside the decode graphs (266 calls per 27B step).
#define RQ_THREADS 512
#define RQ_VEC 5
__device__ __forceinline__ float bf16lo(unsigned int w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16hi(unsigned int w) { return __uint_as_float(w & 0xFFFF0000u); }

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
  uint2* qr = (uint2*)(q + (size_t)row * K);
#pragma unroll
  for (int i = 0; i < RQ_VEC; i++) {
    const int c = tid + i * RQ_THREADS;
    if (c >= nvec) break;
    int lo = __builtin_amdgcn_cvt_pk_fp8_f32(bf16lo(v[i].x) * inv, bf16hi(v[i].x) * inv, 0, false);
    lo = __builtin_amdgcn_cvt_pk_fp8_f32(bf16lo(v[i].y) * inv, bf16hi(v[i].y) * inv, lo, true);
    int hi = __builtin_amdgcn_cvt_pk_fp8_f32(bf16lo(v[i].z) * inv, bf16hi(v[i].z) * inv, 0, false);
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(bf16lo(v[i].w) * inv, bf16hi(v[i].w) * inv, hi, true);
    qr[c] = make_uint2((unsigned int)lo, (unsigned int)hi);
  }
}

extern "C" int r9k_quant_rows_fp8(long x, long q, long s, int M, int K, int ld, long stream) {
  if (M <= 0) return 0;
  if (K % 8 == 0 && ld % 8 == 0 && (x & 15) == 0 && (q & 7) == 0 && K <= RQ_THREADS * RQ_VEC * 8)
    hipLaunchKernelGGL(r9k_quant_rows_fp8_v2_kernel, dim3(M), dim3(RQ_THREADS), 0, (hipStream_t)stream,
                       (const __hip_bfloat16*)x, (unsigned char*)q, (float*)s, K, ld);
  else
    hipLaunchKernelGGL(r9k_quant_rows_fp8_kernel, dim3(M), dim3(256), 0, (hipStream_t)stream,
                       (const __hip_bfloat16*)x, (unsigned char*)q, (float*)s, K, ld);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}
