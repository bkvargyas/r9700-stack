// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas

// r9k_gemm_fp8.hip -- dense FP8(e4m3) weight x FP8(e4m3) activation GEMM, per-row weight scale and per-row
// activation scale, bf16 out, for gfx1201 (RDNA4). C[M,N] = (A[M,K] * As[m]) @ (W[N,K] * Ws[n])^T.
//
// Skinny-M kernel for LM heads (vocab x hidden) and similar weight-bandwidth-bound shapes. Shares its shape
// with r9k_moe_mxfp4a8: the weight is pre-permuted into fragment order so a wave reads contiguous bytes, A is
// read straight from global (a lane's A fragment is 8 consecutive bytes of one row), split-K across the waves
// of a workgroup reduced in LDS, MT M-tiles per workgroup share each weight fragment. No unpack at all: an fp8
// weight fragment feeds v_wmma_f32_16x16x16_fp8_fp8 directly.
//
// Provenance note (2026-09-22): an earlier header credited this structure to libr4d. That was over-cautious and
// is corrected here. None of it is libr4d's to own, and none of it is present in this file by copying:
//   * fragment order is FORCED by the WMMA builtin -- v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12 requires lane l to
//     hold row (l & 15), k-bytes 8*(l >> 4)..+8, so pre-permuting the weight to match is the only way to read
//     it contiguously. Any correct implementation lands on the same bytes.
//   * split-K across a workgroup's waves reduced in LDS is a textbook GEMM technique, chosen here because it
//     needs no HBM partials or counters and so stays cudagraph-safe.
//   * MT M-tiles sharing a weight fragment is ordinary register tiling.
// This file contains no folded-exponent code (no kMag, no v_perm unpack, no Wref) -- it is fp8 x fp8 and has
// nothing to unpack. See notes/independence.md.
//
// Weight layout: Wp[((n/16) * (K/16) + (k/16)) * 32 + lane] is a uint2 (8 bytes) = row 16*(n/16) + (lane&15),
// k = 16*(k/16) + 8*(lane>>4) .. +8.  Shape rules: N % 16 == 0, K % (16*SK) == 0.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <cstdint>

#define WAVE32 32

typedef int   v2i __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));

template <int MT, int NPW>
__global__ void r9k_gemm_fp8_kernel(const unsigned char* __restrict__ A, const float* __restrict__ As,
                                    const v2i* __restrict__ Wp, const float* __restrict__ Ws,
                                    __hip_bfloat16* __restrict__ C, int M, int K, int N, int ldc, int WV, int SK) {
  extern __shared__ float red[];
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK, kbase = ksplit * Ks;
  const int nlane = lane & 15, mhalf = (lane >> 4) * 8;
  const int ksteps = K / 16;

  const unsigned char* arow[MT];
#pragma unroll
  for (int i = 0; i < MT; i++) {
    int r = (mt0 + i) * 16 + nlane;
    r = r < M - 1 ? r : M - 1;
    arow[i] = A + (size_t)r * K;
  }
  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  const int ntmax = N / 16 - 1;
  int wtile[NPW];
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    const int t = nt0 + j;
    wtile[j] = t < ntmax ? t : ntmax;
  }

  for (int k = 0; k < Ks; k += 16) {
    const int kg = kbase + k;
    const int kstep = kg / 16;
    v2i af[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) {
      const unsigned int* pa = (const unsigned int*)(arow[i] + kg + mhalf);
      af[i][0] = (int)pa[0];
      af[i][1] = (int)pa[1];
    }
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const v2i wf = Wp[((size_t)wtile[j] * ksteps + kstep) * WAVE32 + lane];
#pragma unroll
      for (int i = 0; i < MT; i++)
        acc[j][i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, acc[j][i]);
    }
  }

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
      const int m = (mt0 + i) * 16 + ((off >> 7) << 3) + (off & 7);
      if (n >= N || m >= M) continue;
      float v = 0.f;
      for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
      C[(size_t)m * ldc + n] = (__hip_bfloat16)(v * Ws[n] * As[m]);
    }
  }
}

// Block-scaled variant (DeepSeek/compressed-tensors "block" FP8): weight scales Bs[N/128][K/128] per 128x128 block,
// activation scales As[M][K/128] per token per 128-group. Each wave accumulates one 128-K group at a time into a
// temporary, then folds tmp * As[m][g] * Bs[n/128][g] into the running sum -- the same math as vLLM's Triton
// w8a8_block_scaled_mm, at split-K bandwidth. Requires K % (128*SK) == 0.
template <int MT, int NPW>
__global__ void r9k_gemm_fp8_block_kernel(const unsigned char* __restrict__ A, const float* __restrict__ As,
                                          const v2i* __restrict__ Wp, const float* __restrict__ Bs,
                                          __hip_bfloat16* __restrict__ C, int M, int K, int N, int ldc, int WV, int SK) {
  extern __shared__ float red[];
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK, kbase = ksplit * Ks;
  const int nlane = lane & 15, mhalf = (lane >> 4) * 8;
  const int ksteps = K / 16, kgroups = K / 128;

  const unsigned char* arow[MT];
  int mrow_e0[MT];
#pragma unroll
  for (int i = 0; i < MT; i++) {
    int r = (mt0 + i) * 16 + nlane;
    r = r < M - 1 ? r : M - 1;
    arow[i] = A + (size_t)r * K;
    mrow_e0[i] = (mt0 + i) * 16 + mhalf;             // accumulator rows this lane owns: mrow_e0 + e, e < 8
  }
  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  const int ntmax = N / 16 - 1;
  int wtile[NPW];
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    const int t = nt0 + j;
    wtile[j] = t < ntmax ? t : ntmax;
  }

  for (int kg0 = kbase; kg0 < kbase + Ks; kg0 += 128) {
    const int g = kg0 / 128;
    v8f tmp[NPW][MT];
#pragma unroll
    for (int j = 0; j < NPW; j++)
#pragma unroll
      for (int i = 0; i < MT; i++) tmp[j][i] = (v8f)(0.f);
#pragma unroll 2
    for (int k = 0; k < 128; k += 16) {
      const int kg = kg0 + k;
      const int kstep = kg / 16;
      v2i af[MT];
#pragma unroll
      for (int i = 0; i < MT; i++) {
        const unsigned int* pa = (const unsigned int*)(arow[i] + kg + mhalf);
        af[i][0] = (int)pa[0];
        af[i][1] = (int)pa[1];
      }
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const v2i wf = Wp[((size_t)wtile[j] * ksteps + kstep) * WAVE32 + lane];
#pragma unroll
        for (int i = 0; i < MT; i++)
          tmp[j][i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, tmp[j][i]);
      }
    }
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const float bs = Bs[(size_t)((wtile[j] * 16) / 128) * kgroups + g];
#pragma unroll
      for (int i = 0; i < MT; i++)
#pragma unroll
        for (int e = 0; e < 8; e++) {
          int m = mrow_e0[i] + e;
          m = m < M - 1 ? m : M - 1;
          acc[j][i][e] += tmp[j][i][e] * bs * As[(size_t)m * kgroups + g];
        }
    }
  }

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
      const int m = (mt0 + i) * 16 + ((off >> 7) << 3) + (off & 7);
      if (n >= N || m >= M) continue;
      float v = 0.f;
      for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
      C[(size_t)m * ldc + n] = (__hip_bfloat16)v;
    }
  }
}

// Per-token-group (128) dynamic fp8 quant: x [M][K] bf16 -> q [M][K] e4m3, s [M][K/128] fp32. One wave per group.
__global__ void r9k_quant_group128_fp8_kernel(const __hip_bfloat16* __restrict__ x, unsigned char* __restrict__ q,
                                              float* __restrict__ s, int M, int K, int ld) {
  const int groups = K / 128;
  const int gid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (gid >= M * groups) return;
  const int row = gid / groups, g = gid % groups, lane = threadIdx.x & 31;
  const __hip_bfloat16* xr = x + (size_t)row * ld + g * 128;
  float v[4], m = 0.f;
#pragma unroll
  for (int i = 0; i < 4; i++) { v[i] = (float)xr[lane * 4 + i]; m = fmaxf(m, fabsf(v[i])); }
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
  m = fmaxf(m, 1e-12f);
  const float inv = 448.f / m;
  if (lane == 0) s[(size_t)row * groups + g] = m / 448.f;
  unsigned char* qr = q + (size_t)row * K + g * 128;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    __hip_fp8_e4m3 f(v[i] * inv);
    qr[lane * 4 + i] = *reinterpret_cast<unsigned char*>(&f);
  }
}

extern "C" int r9k_quant_group128_fp8(long x, long q, long s, int M, int K, int ld, long stream) {
  if (M <= 0) return 0;
  if (K % 128) return -1;
  const long waves = (long)M * (K / 128);
  hipLaunchKernelGGL(r9k_quant_group128_fp8_kernel, dim3((unsigned)((waves + 7) / 8)), dim3(256), 0, (hipStream_t)stream,
                     (const __hip_bfloat16*)x, (unsigned char*)q, (float*)s, M, K, ld);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

extern "C" int r9k_gemm_fp8_block(long a, long as, long wq, long bs, long c, int M, int K, int N, int ldc, int WV,
                                  int SK, int NPW, int MT, long stream) {
  if (M <= 0) return 0;
  if (N % 16) return -1;
  if (K % (128 * SK)) return -2;
  if (WV * SK * WAVE32 > 1024) return -3;
  const int ncols = WV * NPW;
  const size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024) return -4;
  const int mtiles = (M + 15) / 16;
  dim3 grid((N / 16 + ncols - 1) / ncols, (mtiles + MT - 1) / MT), block(WV * SK * WAVE32);
  hipStream_t st = (hipStream_t)stream;
#define LB(MTv, NP) hipLaunchKernelGGL((r9k_gemm_fp8_block_kernel<MTv, NP>), grid, block, smem, st,              \
                                       (const unsigned char*)a, (const float*)as, (const v2i*)wq, (const float*)bs, \
                                       (__hip_bfloat16*)c, M, K, N, ldc, WV, SK)
#define LBM(MTv) switch (NPW) { case 1: LB(MTv, 1); break; case 2: LB(MTv, 2); break; case 4: LB(MTv, 4); break; \
                                default: return -5; }
  switch (MT) {
    case 1: LBM(1); break;
    case 2: LBM(2); break;
    case 4: LBM(4); break;
    default: return -7;
  }
#undef LBM
#undef LB
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

extern "C" int r9k_gemm_fp8(long a, long as, long wq, long ws, long c, int M, int K, int N, int ldc, int WV, int SK,
                            int NPW, int MT, long stream) {
  if (M <= 0) return 0;
  if (N % 16) return -1;
  if (K % (16 * SK)) return -2;
  if (WV * SK * WAVE32 > 1024) return -3;
  const int ncols = WV * NPW;
  const size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024) return -4;
  const int mtiles = (M + 15) / 16;
  dim3 grid((N / 16 + ncols - 1) / ncols, (mtiles + MT - 1) / MT), block(WV * SK * WAVE32);
  hipStream_t st = (hipStream_t)stream;
#define L(MTv, NP) hipLaunchKernelGGL((r9k_gemm_fp8_kernel<MTv, NP>), grid, block, smem, st, (const unsigned char*)a, \
                                      (const float*)as, (const v2i*)wq, (const float*)ws, (__hip_bfloat16*)c, M, K, N, \
                                      ldc, WV, SK)
#define LM(MTv) switch (NPW) { case 1: L(MTv, 1); break; case 2: L(MTv, 2); break; case 4: L(MTv, 4); break; \
                               default: return -5; }
  switch (MT) {
    case 1: LM(1); break;
    case 2: LM(2); break;
    case 4: LM(4); break;
    default: return -7;
  }
#undef LM
#undef L
  return hipGetLastError() == hipSuccess ? 0 : -6;
}
