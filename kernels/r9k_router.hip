// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas

// r9k_router.hip -- MoE router (gate) GEMM for gfx1201 at decode widths:
//   logits[M, E] = x[M, K] (bf16) . W[E, K]^T (bf16), fp32 accumulate, M <= 32; out bf16 (rounded once, as the
//   stock F.linear / hipBLASLt bf16 GEMM rounds) or fp32.
//
// vLLM's MoE runner runs the model's ReplicatedLinear gate (Flash-Next: K=2560, E=512, M = tokens x (MTP+1))
// through hipBLASLt, which picks a 16x16x32 tile: 19.6 us for a 2.6 MB weight (~130 GB/s), 51 calls per decode
// step. Weight-streaming shape: one wave per (expert, K split), lanes stride the row in 16 B chunks, every product
// is exact in fp32 (bf16 x bf16) and accumulated with fma; a wave shuffle reduces the lanes and LDS reduces the K
// splits. No device printf/assert.
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

typedef uint16_t v8u16 __attribute__((ext_vector_type(8)));
constexpr int WAVES = 8, NT = 32 * WAVES;
constexpr int MAXM = 32;

__device__ __forceinline__ float bf2f(uint16_t u) { return __uint_as_float((unsigned)u << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {    // round to nearest even (matches torch's bf16 conversion)
  unsigned u = __float_as_uint(f);
  if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0xffffu) ? 0x40u : 0u));
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
template <bool BF>
__device__ __forceinline__ void store_out(void* out, long i, float v) {
  if constexpr (BF) ((uint16_t*)out)[i] = f2bf(v);
  else ((float*)out)[i] = v;
}
// Epilogue for the hyper-connection down GEMM: columns < act_cols get vLLM's hc_silu on the bf16-rounded GEMM output
// (s = bf16(v) / hc; s * sigmoid(s)); the rest (injection logits, padding) are stored as the GEMM left them.
__device__ __forceinline__ float epilogue(float v, int e, int act_cols, float hc) {
  if (e >= act_cols) return v;
  const float s = bf2f(f2bf(v)) / hc;
  return s / (1.f + __expf(-s));
}
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  return v;
}

// MT: padded row count (template so acc[] stays in registers); SPLIT: waves sharing one expert's K range; BF: bf16
// output. Block = 8 waves = 8 / SPLIT experts. Row m >= M of x is clamped to the last real row (loaded, never stored).
template <int MT, int SPLIT, bool BF>
__global__ void __launch_bounds__(NT) r9k_router_gemm_kernel(
    const uint16_t* __restrict__ x, long sx, const uint16_t* __restrict__ w, long sw, void* __restrict__ out,
    long so, int M, int E, int K, int act_cols, float hc) {
  constexpr int EPB = WAVES / SPLIT;
  __shared__ float part[SPLIT > 1 ? SPLIT : 1][EPB][MAXM];
  const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
  const int el = wave / SPLIT, ks = wave % SPLIT;
  const int e = blockIdx.x * EPB + el;
  const bool live = e < E;
  const int Ks = K / SPLIT, k0 = ks * Ks, nch = Ks / 8;
  const uint16_t* wp = w + (long)(live ? e : 0) * sw + k0;
  const uint16_t* xp = x + k0;
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) acc[m] = 0.f;
  for (int c = lane; c < nch; c += 32) {
    const v8u16 wv = *reinterpret_cast<const v8u16*>(wp + c * 8);
    float wf[8];
#pragma unroll
    for (int i = 0; i < 8; i++) wf[i] = bf2f(wv[i]);
#pragma unroll
    for (int m = 0; m < MT; m++) {
      const int mr = m < M ? m : M - 1;
      const v8u16 xv = *reinterpret_cast<const v8u16*>(xp + (long)mr * sx + c * 8);
#pragma unroll
      for (int i = 0; i < 8; i++) acc[m] = fmaf(bf2f(xv[i]), wf[i], acc[m]);
    }
  }
#pragma unroll
  for (int m = 0; m < MT; m++) acc[m] = wave_sum(acc[m]);
  if constexpr (SPLIT == 1) {
    if (live && lane < M) {
      float v = 0.f;
#pragma unroll
      for (int m = 0; m < MT; m++) v = lane == m ? acc[m] : v;
      store_out<BF>(out, (long)lane * so + e, epilogue(v, e, act_cols, hc));
    }
  } else {
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < MT; m++) part[ks][el][m] = acc[m];
    }
    __syncthreads();
    if (ks == 0 && live && lane < M) {
      float s = 0.f;
#pragma unroll
      for (int q = 0; q < SPLIT; q++) s += part[q][el][lane];
      store_out<BF>(out, (long)lane * so + e, epilogue(s, e, act_cols, hc));
    }
  }
}

template <int MT, bool BF>
int launch_split(int split, const uint16_t* x, long sx, const uint16_t* w, long sw, void* out, long so, int M, int E,
                 int K, int act_cols, float hc, hipStream_t st) {
#define RL(S)                                                                                                 \
  do {                                                                                                        \
    const int epb = WAVES / (S);                                                                              \
    hipLaunchKernelGGL((r9k_router_gemm_kernel<MT, S, BF>), dim3((E + epb - 1) / epb), dim3(NT), 0, st, x, sx, \
                       w, sw, out, so, M, E, K, act_cols, hc);                                                \
  } while (0)
  switch (split) {
    case 1: RL(1); break;
    case 2: RL(2); break;
    case 4: RL(4); break;
    case 8: RL(8); break;
    default: return -5;
  }
#undef RL
  return 0;
}

template <bool BF>
int launch_m(int split, const uint16_t* x, long sx, const uint16_t* w, long sw, void* out, long so, int M, int E,
             int K, int ac, float hc, hipStream_t st) {
  if (M <= 1) return launch_split<1, BF>(split, x, sx, w, sw, out, so, M, E, K, ac, hc, st);
  if (M <= 2) return launch_split<2, BF>(split, x, sx, w, sw, out, so, M, E, K, ac, hc, st);
  if (M <= 4) return launch_split<4, BF>(split, x, sx, w, sw, out, so, M, E, K, ac, hc, st);
  if (M <= 8) return launch_split<8, BF>(split, x, sx, w, sw, out, so, M, E, K, ac, hc, st);
  if (M <= 16) return launch_split<16, BF>(split, x, sx, w, sw, out, so, M, E, K, ac, hc, st);
  return launch_split<32, BF>(split, x, sx, w, sw, out, so, M, E, K, ac, hc, st);
}

}  // namespace

// out[M, E] (row stride so elements; bf16 when out_bf16 else fp32) = x[M, K] bf16 (row stride sx elements)
// . w[E, K]^T bf16 (row stride sw). act_cols > 0: hc_silu(., hc) on the first act_cols output columns (the
// hyper-connection down GEMM). Rules: 1 <= M <= 32, K % (8 * split) == 0, split in {1, 2, 4, 8}, 16-byte aligned
// rows. Returns 0 or a negative shape code (the Python side raises).
extern "C" int r9k_router_gemm(long x, long sx, long w, long sw, long out, long so, int M, int E, int K, int split,
                               int out_bf16, int act_cols, float hc, long stream) {
  if (M < 1 || M > MAXM || E < 1 || K < 8) return -1;
  if (split < 1 || split > 8 || (split & (split - 1))) return -2;
  if (K % (8 * split)) return -3;
  if ((sx | sw) & 7) return -4;
  hipStream_t st = (hipStream_t)stream;
  const uint16_t* xp = (const uint16_t*)x;
  const uint16_t* wp = (const uint16_t*)w;
  if (act_cols > 0 && hc <= 0.f) return -7;
  const int rc = out_bf16 ? launch_m<true>(split, xp, sx, wp, sw, (void*)out, so, M, E, K, act_cols, hc, st)
                          : launch_m<false>(split, xp, sx, wp, sw, (void*)out, so, M, E, K, act_cols, hc, st);
  if (rc) return rc;
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

extern "C" int r9k_router_max_m(void) { return MAXM; }
