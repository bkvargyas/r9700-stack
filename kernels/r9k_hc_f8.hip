// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas

// r9k_hc_f8.hip -- the hyper-connection GEMMs at decode widths (M <= 8) on fp8 weights in permute_fp8's fragment
// order, i.e. the same tensors the prefill fp8 kernels read: no second copy of the weight, and half the bytes of
// the bf16 kernels (r9k_router.hip, r9k_hc.hip) that stream 1.29 GB a decode step at TP4.
//
// Fragment order [nt, ks, h, r, 8 bytes]: row n = nt*16 + r, k = ks*16 + h*8 + b -- the bf16 WMMA A-fragment's
// lane mapping (lane = h*16 + r), so a wave takes one 16-row tile and walks ks: every step is one 256 B contiguous
// read for the wave, each lane 8 fp8 of its row, converted to bf16 in registers (e4m3 is exact in bf16) and fed to
// v_wmma_f32_16x16x16_bf16 against the activations' B-fragment (lane = token m = lane & 15, k-half h): bf16 x bf16
// products exact in fp32, W8A16. The per-row fp32 scale is applied once at the end. Token tiles of 16, several
// per block (the converted weight fragment serves all of them); the LDS reduction caps the waves per tile.
//   r9k_hc_down_f8:   out[M, E] = hc_silu-on-act-cols( bf16( x[M, K] . W[E, K]^T ) )  -- SPLIT waves share a
//                     tile's K range (E = 336 is only 21 tiles; 16 splits put 336 waves on the 64 CUs)
//   r9k_hc_up_mix_f8: out[M, HD] = bf16( 1/4 sum_s sigmoid( bf16( lora[M, LR] . Wup[s*HD + d] ) ) * xn[M, s*HD+d] )
//                     on the hc4_interleave row order (r' = g*64 + s*16 + i): the four streams of 16 columns are
//                     the four tiles of one block, one wave each, and meet in LDS for the gated mean.
// No device printf/assert.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdlib.h>

namespace {

typedef __bf16 v8bf16 __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
constexpr int MAXM = 16;

__device__ __forceinline__ float bf2f(uint16_t u) { return __uint_as_float((unsigned)u << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {
  unsigned u = __float_as_uint(f);
  if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0xffffu) ? 0x40u : 0u));
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }

// this lane's 8 fp8 weights -> the bf16 A-fragment (the byte selector must be a literal)
__device__ __forceinline__ v8bf16 w8_to_bf16(const uint2 wv) {
  const int w0 = (int)wv.x, w1 = (int)wv.y;
  v8bf16 r;
  r[0] = (__bf16)__builtin_amdgcn_cvt_f32_fp8(w0, 0); r[1] = (__bf16)__builtin_amdgcn_cvt_f32_fp8(w0, 1);
  r[2] = (__bf16)__builtin_amdgcn_cvt_f32_fp8(w0, 2); r[3] = (__bf16)__builtin_amdgcn_cvt_f32_fp8(w0, 3);
  r[4] = (__bf16)__builtin_amdgcn_cvt_f32_fp8(w1, 0); r[5] = (__bf16)__builtin_amdgcn_cvt_f32_fp8(w1, 1);
  r[6] = (__bf16)__builtin_amdgcn_cvt_f32_fp8(w1, 2); r[7] = (__bf16)__builtin_amdgcn_cvt_f32_fp8(w1, 3);
  return r;
}

// acc[t][i] (+)= sum over ks in [ks0, ks1) of W[tile nt, row 8h + i, k] . x[m = t*16 + (lane & 15), k] for MT token
// tiles t (the WMMA C-fragment: lane holds rows 8*(lane >> 4) + i of column lane & 15). The weight fragment is
// converted once per k-step and reused for every token tile. Software-pipelined in chunks of U k-steps: chunk
// c+1's loads are in flight while chunk c's converts and WMMAs run (one load in flight per lane was ~40 dependent
// round trips a wave; the compiler did not hoist loads across the loop on its own).
template <int U, int MT>
struct Chunk {
  uint2 wv[U];
  v8bf16 xb[U][MT];
  __device__ __forceinline__ void load(const uint8_t* wp, const uint16_t* const* xp, int ks) {
#pragma unroll
    for (int u = 0; u < U; u++) wv[u] = *reinterpret_cast<const uint2*>(wp + (long)(ks + u) * 32 * 8);
#pragma unroll
    for (int u = 0; u < U; u++) {
#pragma unroll
      for (int t = 0; t < MT; t++) xb[u][t] = *reinterpret_cast<const v8bf16*>(xp[t] + (long)(ks + u) * 16);
    }
  }
  __device__ __forceinline__ void mma(v8f* acc) const {
#pragma unroll
    for (int u = 0; u < U; u++) {
      const v8bf16 a = w8_to_bf16(wv[u]);
#pragma unroll
      for (int t = 0; t < MT; t++) acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, xb[u][t], acc[t]);
    }
  }
};

// x rows: token tile t of this block starts at token t0 + 16 t; rows past M - 1 read row M - 1 (discarded later)
template <int MT>
__device__ __forceinline__ void frag_dot(const uint8_t* __restrict__ wq, long ksteps, int nt,
                                         const uint16_t* __restrict__ x, long sx, int M, int t0, int lane,
                                         int ks0, int ks1, v8f* acc) {
  constexpr int U = MT <= 2 ? 8 : (MT <= 4 ? 4 : 2);         // loads in flight per lane vs the register budget
  const int h = lane >> 4;
  const uint8_t* wp = wq + ((long)nt * ksteps * 32 + lane) * 8;      // + ks * 256
  const uint16_t* xp[MT];
#pragma unroll
  for (int t = 0; t < MT; t++) {
    int m = t0 + t * 16 + (lane & 15);
    m = m < M ? m : M - 1;
    xp[t] = x + (long)m * sx + h * 8;                                 // + ks * 16
  }
  const int nch = (ks1 - ks0) / U;
  Chunk<U, MT> A, B;
  if (nch > 0) A.load(wp, xp, ks0);
  int c = 0;
  for (; c + 1 < nch; c += 2) {
    B.load(wp, xp, ks0 + (c + 1) * U);
    A.mma(acc);
    if (c + 2 < nch) A.load(wp, xp, ks0 + (c + 2) * U);
    B.mma(acc);
  }
  if (c < nch) A.mma(acc);
  for (int ks = ks0 + nch * U; ks < ks1; ks++) {
    const v8bf16 a = w8_to_bf16(*reinterpret_cast<const uint2*>(wp + (long)ks * 32 * 8));
#pragma unroll
    for (int t = 0; t < MT; t++) {
      const v8bf16 xb = *reinterpret_cast<const v8bf16*>(xp[t] + (long)ks * 16);
      acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, xb, acc[t]);
    }
  }
}

__device__ __forceinline__ float hc_epilogue(float s, float sc, bool act, float hc) {
  float v = bf2f(f2bf(s * sc));                                 // the GEMM's bf16 output (stock rounds here)
  if (act) {
    const float a = v / hc;                                     // vLLM's hc_silu
    v = a / (1.f + __expf(-a));
  }
  return v;
}

// ------------------------------------------------------------------------------------------- down + hc_silu
// grid (E/16, KB, token groups of 16*MT), block 32*SPLIT: KB blocks x SPLIT waves share a tile's K range (E = 336
// is 21 tiles; the 3.4 MB must spread over the 64 CUs). Waves meet in LDS; KB > 1 (MT == 1 only) writes each
// block's partial to scratch[kb][m][e] and the last block to arrive sums them in kb order (deterministic, no
// float atomics), does the epilogue and resets the counter for the next launch.
template <int SPLIT, int MT>
__global__ void __launch_bounds__(32 * SPLIT) r9k_hc_down_f8_kernel(
    const uint16_t* __restrict__ x, long sx, const uint8_t* __restrict__ wq, const float* __restrict__ ws,
    uint16_t* __restrict__ out, long so, int M, int E, long ksteps, int act_cols, float hc,
    float* __restrict__ scratch, int* __restrict__ counters) {
  __shared__ float part[SPLIT][MT][16][16];                      // [wave][tile][e][m]
  __shared__ int last;
  const int t = threadIdx.x, lane = t & 31, wave = t >> 5, nt = blockIdx.x, kb = blockIdx.y, KB = gridDim.y;
  const int h = lane >> 4, m = lane & 15, t0 = blockIdx.z * 16 * MT;
  const int kpb = (int)(ksteps / KB), kps = kpb / SPLIT, ks0 = kb * kpb + wave * kps;
  v8f acc[MT];
#pragma unroll
  for (int i = 0; i < MT; i++) acc[i] = (v8f)(0.f);
  frag_dot<MT>(wq, ksteps, nt, x, sx, M, t0, lane, ks0, ks0 + kps, acc);
#pragma unroll
  for (int tt = 0; tt < MT; tt++) {
#pragma unroll
    for (int i = 0; i < 8; i++) part[wave][tt][8 * h + i][m] = acc[tt][i];
  }
  __syncthreads();
  if (wave != 0 && KB == 1) return;
  const int e0 = nt * 16 + 8 * h;
  if (KB == 1) {
#pragma unroll
    for (int tt = 0; tt < MT; tt++) {
      const int mg = t0 + tt * 16 + m;
      if (mg >= M) continue;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int e = e0 + i;
        if (e < E) {
          float v = 0.f;
#pragma unroll
          for (int q = 0; q < SPLIT; q++) v += part[q][tt][8 * h + i][m];
          out[(long)mg * so + e] = f2bf(hc_epilogue(v, ws[e], e < act_cols, hc));
        }
      }
    }
    return;
  }
  // KB > 1: MT == 1
  if (wave == 0 && m < M) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      if (e0 + i < E) {
        float v = 0.f;
#pragma unroll
        for (int q = 0; q < SPLIT; q++) v += part[q][0][8 * h + i][m];
        scratch[((long)kb * MAXM + m) * E + e0 + i] = v;
      }
    }
  }
  __threadfence();
  __syncthreads();
  if (t == 0) last = (atomicAdd(&counters[nt], 1) == KB - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  if (wave == 0 && m < M) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int e = e0 + i;
      if (e < E) {
        float v = 0.f;
        for (int q = 0; q < KB; q++) v += __builtin_nontemporal_load(&scratch[((long)q * MAXM + m) * E + e]);
        out[(long)m * so + e] = f2bf(hc_epilogue(v, ws[e], e < act_cols, hc));
      }
    }
  }
  if (t == 0) counters[nt] = 0;
}

// -------------------------------------------------------------------------------------- up GEMM + gated mean
// grid (HD/16, token groups of 16*MT), 4 waves = the four streams of one 16-column group (hc4_interleave: tile
// nt = g*4 + s holds rows r' = g*64 + s*16 + i, i.e. stream s, columns d = g*16 + i); the gated terms of one token
// tile at a time meet in LDS for the mean.
template <int MT>
__global__ void __launch_bounds__(128) r9k_hc_up_mix_f8_kernel(
    const uint16_t* __restrict__ lora, long s_l, const uint8_t* __restrict__ wq, const float* __restrict__ ws,
    const uint16_t* __restrict__ xn, long s_x, uint16_t* __restrict__ out, long s_o, int M, int HD, long ksteps) {
  __shared__ float term[4][16][16];                               // [stream][d][m]
  const int t = threadIdx.x, lane = t & 31, s = t >> 5, g = blockIdx.x;
  const int h = lane >> 4, m = lane & 15, nt = g * 4 + s, t0 = blockIdx.y * 16 * MT;
  v8f acc[MT];
#pragma unroll
  for (int i = 0; i < MT; i++) acc[i] = (v8f)(0.f);
  frag_dot<MT>(wq, ksteps, nt, lora, s_l, M, t0, lane, 0, (int)ksteps, acc);
  const int d0 = g * 16 + 8 * h;
#pragma unroll
  for (int tt = 0; tt < MT; tt++) {
    const int mg = t0 + tt * 16 + m;
    if (mg < M) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int d = d0 + i;
        if (d < HD) {
          const float gte = bf2f(f2bf(acc[tt][i] * ws[nt * 16 + 8 * h + i]));   // the gate, rounded as the GEMM output
          term[s][8 * h + i][m] = sigmoidf_(gte) * bf2f(xn[(long)mg * s_x + (long)s * HD + d]);
        }
      }
    }
    __syncthreads();
    if (s == 0 && mg < M) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int d = d0 + i;
        if (d < HD) {
          const float sum = ((term[0][8 * h + i][m] + term[1][8 * h + i][m]) + term[2][8 * h + i][m]) + term[3][8 * h + i][m];
          out[(long)mg * s_o + d] = f2bf(sum / 4.f);
        }
      }
    }
    __syncthreads();
  }
}

}  // namespace

// x [M, K] bf16 (row stride sx elements), wq = permute_fp8(e4m3 [E, K]), ws [E] fp32, out [M, E] bf16 (stride so).
// M up to 4096 (token tiles of 16, up to 8 a block; the weight tile is converted once and reused across them).
// split: waves sharing a tile's K range (1..32, capped by the token tiles; (K/16) % (kb*split) == 0); act_cols / hc
// as r9k_router_gemm.
// kb: blocks per tile along K (1..8); kb > 1 needs scratch [kb, 16, E] fp32 and counters [E/16] int32 (zero at
// first use; the kernel leaves them zero), both private to one stream.
extern "C" int r9k_hc_down_f8(long x, long sx, long wq, long ws, long out, long so, int M, int E, int K, int split,
                              int act_cols, float hc, int kb, long scratch, long counters, long stream) {
  if (M < 1 || M > 4096 || E < 1 || E % 16 || K < 16 || K % 16) return -1;
  if (split < 1 || split > 32 || (split & (split - 1)) || kb < 1 || kb > 8) return -2;
  if ((sx | so) & 7) return -3;
  if (act_cols > 0 && hc <= 0.f) return -4;
  if (kb > 1 && (!scratch || !counters)) return -5;
  if (kb > 1 && M > MAXM) return -8;                           // split-K across blocks is a one-tile path
  const int tiles = (M + 15) / 16;
  // token tiles per block (the converted weight fragment serves all of them; the weight tile is re-read per block
  // from L2): measured 2026-10-10 at 128 / 255 rows, down GEMM: 1 tile 36 / 64 us, 2 tiles 30 / 56, 4 tiles 32 / 75,
  // 8 tiles 48 / 110 (registers and the per-k-step x loads); 2 is the default, R9K_HC_F8_MT overrides
  static const int mt_env = getenv("R9K_HC_F8_MT") ? atoi(getenv("R9K_HC_F8_MT")) : 2;
  const int mt = mt_env > 1 ? (tiles >= 8 && mt_env >= 8 ? 8 : (tiles >= 4 && mt_env >= 4 ? 4 : (tiles >= 2 ? 2 : 1))) : 1;
  const int cap = mt >= 8 ? 4 : (mt >= 4 ? 8 : 16);            // waves per tile vs the LDS reduction (64 KB)
  if (mt > 1 && split > cap) split = cap;
  if ((K / 16) % (kb * split)) return -2;
  hipStream_t st = (hipStream_t)stream;
  const dim3 grid(E / 16, kb, (tiles + mt - 1) / mt), block(32 * split);
#define DL(S, T) hipLaunchKernelGGL((r9k_hc_down_f8_kernel<S, T>), grid, block, 0, st, (const uint16_t*)x, sx, \
                                    (const uint8_t*)wq, (const float*)ws, (uint16_t*)out, so, M, E, (long)(K / 16), act_cols, hc, \
                                    (float*)scratch, (int*)counters)
#define DS(T) switch (split) { case 1: DL(1, T); break; case 2: DL(2, T); break; case 4: DL(4, T); break; \
                               case 8: DL(8, T); break; case 16: DL(16, T); break; default: return -2; }
  switch (mt) {
    case 1: switch (split) { case 1: DL(1, 1); break; case 2: DL(2, 1); break; case 4: DL(4, 1); break; case 8: DL(8, 1); break;
                             case 16: DL(16, 1); break; case 32: DL(32, 1); break; default: return -2; } break;
    case 2: DS(2); break;
    case 4: switch (split) { case 1: DL(1, 4); break; case 2: DL(2, 4); break; case 4: DL(4, 4); break; case 8: DL(8, 4); break; default: return -2; } break;
    case 8: switch (split) { case 1: DL(1, 8); break; case 2: DL(2, 8); break; case 4: DL(4, 8); break; default: return -2; } break;
    default: return -2;
  }
#undef DS
#undef DL
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

// lora [M, LR] bf16, wq = permute_fp8(e4m3 hc4_interleave(Wup [4*HD, LR])), ws [4*HD] fp32 in that row order,
// xn [M, 4*HD] bf16, out [M, HD] bf16. HC == 4, HD % 16 == 0, LR % 16 == 0, M up to 4096.
extern "C" int r9k_hc_up_mix_f8(long lora, long s_l, long wq, long ws, long xn, long s_x, long out, long s_o,
                                int M, int LR, int HD, long stream) {
  if (M < 1 || M > 4096 || LR < 16 || LR % 16 || HD < 16 || HD % 16) return -1;
  if ((s_l | s_x | s_o) & 7) return -2;
  hipStream_t st = (hipStream_t)stream;
  const int tiles = (M + 15) / 16;
  static const int mt_env = getenv("R9K_HC_F8_MT") ? atoi(getenv("R9K_HC_F8_MT")) : 2;
  const int mt = mt_env > 1 ? (tiles >= 8 && mt_env >= 8 ? 8 : (tiles >= 4 && mt_env >= 4 ? 4 : (tiles >= 2 ? 2 : 1))) : 1;
  const dim3 grid(HD / 16, (tiles + mt - 1) / mt), block(128);
#define UL(T) hipLaunchKernelGGL((r9k_hc_up_mix_f8_kernel<T>), grid, block, 0, st, (const uint16_t*)lora, s_l, \
                                 (const uint8_t*)wq, (const float*)ws, (const uint16_t*)xn, s_x, (uint16_t*)out, s_o, M, HD, (long)(LR / 16))
  switch (mt) { case 1: UL(1); break; case 2: UL(2); break; case 4: UL(4); break; case 8: UL(8); break; default: return -2; }
#undef UL
  return hipGetLastError() == hipSuccess ? 0 : -6;
}
