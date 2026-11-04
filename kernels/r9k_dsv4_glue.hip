// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas
//
// r9k_dsv4_glue.hip -- DeepSeek-V4.1 decode-width glue on gfx1201 (wave32): launches folded around the EXL3 GEMVs and
// the MoE router (notes/exl3-plan.md Phase 4, package B; numbers in notes/exl3-glue.md).
//
// 1. r9k_dsv4_norm_pre -- RMSNorm (+ the bf16 -> f16 cast, + optionally exllamav3's input rotation) for up to two
//    row segments in one launch. Replaces, per decode layer: attn_norm + the EXL3 apply's bf16 -> f16 input copy
//    (fused_wqa_wkv); and the f32 -> bf16 output cast(s) of fused_wqa_wkv + stock's fused_q_kv_rmsnorm (q_norm /
//    kv_norm) + the f16 input copies of wq_b and indexer.wq_b. A segment reads bf16 rows, or f32 rows that it first
//    rounds to bf16 (the EXL3 GEMV's static C buffer, rounded exactly as the apply's `.to(bfloat16)` would), and writes
//    any of: normalized bf16 rows (strided), normalized f16 rows (the EXL3 GEMV input), or -- `suh` given -- the
//    f16 rows rotated as exllamav3's exl3_gemv_had_in_128 does (x * suh in f16, 128-point Hadamard in f32, * 1/sqrt(128),
//    RNE to f16): the interface for a GEMV that takes pre-rotated input.
// 2. r9k_dsv4_route -- the MoE router in one launch: [optional ffn_norm] + gate GEMV (bf16 x . bf16 W^T, f32 logits)
//    + vLLM's sqrt-softplus top-k with e_score_correction_bias / bias_vl (image sentinel rows) / is_padding,
//    renormalization and routed_scaling_factor. Replaces ffn_norm + the hipBLASLt gate GEMM + topk_hash_softplus_sqrt.
//    The top-k runs in the last block to arrive (ticket counter, self-resetting), one wave a row.
//
// Numerics (tests/test_dsv4_glue.py):
//  * RMSNorm: the row's sum of squares is reduced in one canonical order (one wave: lane l accumulates quads
//    l, l + 32, ... with fma, then an xor tree) in both kernels, so the two produce identical normalized rows; the
//    rounding points follow the stock op (mode 0: vLLM IR native, bf16(bf16(x * rrms) * w); mode 1: one cast of
//    x * rrms * w in f32 -- fused_q_kv_rmsnorm / csrc). The sum order differs from torch / Triton, so a row's rrms can
//    differ in its last ulp: outputs are bit-exact except a small fraction of elements one bf16 ulp apart.
//  * Router: logits are f32 dot products in a different order than hipBLASLt (~1e-6 relative); given the same
//    logits the top-k is bit-exact with csrc/moe/topk_softplus_sqrt_kernels.cu's generic kernel (threshold-20
//    softplus with __logf / __expf, sqrtf, NaN -> 0, choice = s + bias, ties to the lower id, weight =
//    (s + bias) - bias, sum in pick order, scale = rsf / sum (sum > 0, else 1), w * scale).
// Third-party code: had128 is exllamav3 v1.6.6's shuffle_had_f4x32 (MIT, Copyright (c) turboderp; via
// kernels/r9k_moe_exl3.hip). No device printf / assert; graph-safe (no host syncs, counters self-reset).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

namespace {

constexpr int NT = 256, NW = 8;
constexpr int MAXM = 32;            // rows a launch
constexpr int MAXE = 512;           // router experts
constexpr int MAXTOPK = 8;
constexpr float RS128 = 0.088388347648f;

typedef uint16_t v8u16 __attribute__((ext_vector_type(8)));
typedef uint16_t v4u16 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bf2f(uint16_t u) { return __uint_as_float((unsigned)u << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {    // round to nearest even (torch's conversion)
  unsigned u = __float_as_uint(f);
  if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0xffffu) ? 0x40u : 0u));
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }
__device__ __forceinline__ uint16_t f2h(float f) { return __half_as_ushort(__float2half_rn(f)); }
__device__ __forceinline__ float h2f(uint16_t u) { return __half2float(__ushort_as_half(u)); }

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  return v;
}

// EXLLAMAV3 (MIT): 128-point Hadamard across a wave, lane holds elements 4 lane .. 4 lane + 3, unnormalised
// (hadamard_inner.cuh shuffle_had_f4x32 after the in-register 4-point stage of exl3_gemv_had_in_128)
__device__ __forceinline__ void had128(float* h, const int lane) {
  float s0 = h[0] + h[1], d0 = h[0] - h[1], s1 = h[2] + h[3], d1 = h[2] - h[3];
  float h0 = s0 + s1, h1 = d0 + d1, h2_ = s0 - s1, h3 = d0 - d1;
#pragma unroll
  for (int i = 1; i < 32; i <<= 1) {
    uint32_t i0 = __float_as_uint(h0), i1 = __float_as_uint(h1), i2 = __float_as_uint(h2_), i3 = __float_as_uint(h3);
    uint64_t h01 = (uint64_t)i0 | (((uint64_t)i1) << 32);
    uint64_t h23 = (uint64_t)i2 | (((uint64_t)i3) << 32);
    uint64_t ph01 = __shfl_xor(h01, i, 32);
    uint64_t ph23 = __shfl_xor(h23, i, 32);
    float ph0 = __uint_as_float((uint32_t)(ph01 & 0xffffffff));
    float ph1 = __uint_as_float((uint32_t)(ph01 >> 32));
    float ph2 = __uint_as_float((uint32_t)(ph23 & 0xffffffff));
    float ph3 = __uint_as_float((uint32_t)(ph23 >> 32));
    int32_t sfm = -static_cast<int32_t>(lane & i) >> 31;
    i0 ^= sfm & 0x80000000; i1 ^= sfm & 0x80000000; i2 ^= sfm & 0x80000000; i3 ^= sfm & 0x80000000;
    h0 = __uint_as_float(i0) + ph0;
    h1 = __uint_as_float(i1) + ph1;
    h2_ = __uint_as_float(i2) + ph2;
    h3 = __uint_as_float(i3) + ph3;
  }
  h[0] = h0; h[1] = h1; h[2] = h2_; h[3] = h3;
}
// end of EXLLAMAV3 code

// four elements of a row as f32 (f32 rows are rounded to bf16 first: the value the bf16 tensor would hold)
__device__ __forceinline__ void ld4(const void* row, int in_f32, int k, float* v) {
  if (in_f32) {
    const float4 f = *reinterpret_cast<const float4*>((const float*)row + k);
    v[0] = rbf(f.x); v[1] = rbf(f.y); v[2] = rbf(f.z); v[3] = rbf(f.w);
  } else {
    const v4u16 u = *reinterpret_cast<const v4u16*>((const uint16_t*)row + k);
    v[0] = bf2f(u[0]); v[1] = bf2f(u[1]); v[2] = bf2f(u[2]); v[3] = bf2f(u[3]);
  }
}

// canonical row sum of squares (one wave): lane l takes quads l, l + 32, ...; fma in element order; xor tree
__device__ __forceinline__ float row_sumsq(const void* row, int in_f32, int n, int lane) {
  float s = 0.f;
  for (int q = lane; q < n / 4; q += 32) {
    float v[4];
    ld4(row, in_f32, 4 * q, v);
#pragma unroll
    for (int i = 0; i < 4; i++) s = fmaf(v[i], v[i], s);
  }
  return wave_sum(s);
}

__device__ __forceinline__ float rrms_of(float ss, int n, float eps, int mode) {
  const float var = mode == 0 ? ss * (1.f / (float)n) : ss / (float)n;
  return rsqrtf(var + eps);
}

// normalized element, bf16-valued (w: bf16 weight bits; has_w false = no weight)
__device__ __forceinline__ float norm_elem(float x, float rr, bool has_w, uint16_t w, int mode) {
  if (!has_w) return rbf(x * rr);
  if (mode == 0) return rbf(rbf(x * rr) * bf2f(w));
  return rbf(x * rr * bf2f(w));
}

// ------------------------------------------------------------------------------------------------- norm pre-pass
struct NormSeg {
  const void* in;          // [M, n] bf16 or f32, row stride in_stride elements
  long in_stride;
  int in_f32;
  int n;                   // multiple of 128, <= 16384
  const uint16_t* w;       // bf16 [n] or null (no normalization weight)
  int norm;                // 0 = copy / cast only (no RMSNorm)
  uint16_t* out_bf;        // bf16 [M, n] (row stride out_bf_stride) or null
  long out_bf_stride;
  uint16_t* out_h;         // f16 [M, n] (row stride out_h_stride) or null
  long out_h_stride;
  const uint16_t* suh;     // f16 [n] or null: out_h holds the rotated rows
};
struct NormArgs {
  NormSeg s[2];
  int M;
  float eps;
  int mode;
};

__global__ void __launch_bounds__(NT) r9k_dsv4_norm_kernel(NormArgs a) {
  const NormSeg& sg = a.s[blockIdx.y];
  const int m = blockIdx.x, t = threadIdx.x, lane = t & 31, w = t >> 5;
  const void* row = sg.in_f32 ? (const void*)((const float*)sg.in + (long)m * sg.in_stride)
                              : (const void*)((const uint16_t*)sg.in + (long)m * sg.in_stride);
  __shared__ float s_rr;
  float rr = 1.f;
  if (sg.norm) {
    if (w == 0) {
      const float ss = row_sumsq(row, sg.in_f32, sg.n, lane);
      if (lane == 0) s_rr = rrms_of(ss, sg.n, a.eps, a.mode);
    }
    __syncthreads();
    rr = s_rr;
  }
  const bool has_w = sg.w != nullptr;
  // quad q = t + 256 j: wave w, iteration j owns the 128-block w + 8 j (the rotation's unit)
  for (int q = t; q < sg.n / 4; q += NT) {
    const int k = 4 * q;
    float v[4];
    ld4(row, sg.in_f32, k, v);
    if (sg.norm) {
      v4u16 wu = has_w ? *reinterpret_cast<const v4u16*>(sg.w + k) : (v4u16)(0);
#pragma unroll
      for (int i = 0; i < 4; i++) v[i] = norm_elem(v[i], rr, has_w, wu[i], a.mode);
    }
    if (sg.out_bf) {
      v4u16 o;
#pragma unroll
      for (int i = 0; i < 4; i++) o[i] = f2bf(v[i]);
      *reinterpret_cast<v4u16*>(sg.out_bf + (long)m * sg.out_bf_stride + k) = o;
    }
    if (sg.out_h) {
      float h[4];
#pragma unroll
      for (int i = 0; i < 4; i++) h[i] = h2f(f2h(v[i]));        // the f16 input the GEMV would get
      if (sg.suh) {
        const v4u16 su = *reinterpret_cast<const v4u16*>(sg.suh + k);
#pragma unroll
        for (int i = 0; i < 4; i++) h[i] = h2f(f2h(h[i] * h2f(su[i])));   // __hmul2 (exact product, one rounding)
        had128(h, lane);
#pragma unroll
        for (int i = 0; i < 4; i++) h[i] *= RS128;
      }
      v4u16 o;
#pragma unroll
      for (int i = 0; i < 4; i++) o[i] = f2h(h[i]);
      *reinterpret_cast<v4u16*>(sg.out_h + (long)m * sg.out_h_stride + k) = o;
    }
  }
}

// ------------------------------------------------------------------------------------------------------- router
struct RouteArgs {
  const uint16_t* x;       // [M, K] bf16 (row stride sx): the normalized FFN input, or raw when nw
  long sx;
  const uint16_t* nw;      // ffn_norm weight bf16 [K], or null (x already normalized)
  float eps;
  int mode;
  uint16_t* xn;            // normalized rows out (bf16 [M, K] contiguous) when nw
  const uint16_t* w;       // gate weight bf16 [E, K]
  int E, K;
  const float* bias;       // e_score_correction_bias f32 [E]
  const float* bias_vl;    // f32 [E] or null
  const void* ids;         // input ids [M] (int32, or int64 when ids64), or null
  int ids64;
  int sentinel_lo;
  const uint8_t* pad;      // is_padding bool [M] or null
  int topk, renorm;
  float rsf;
  float* logits;           // scratch f32 [M, E] (also the logits output)
  float* w_out;            // [M, topk] f32
  int* id_out;             // [M, topk] int32
  int* ctr;                // one int, zero between launches
  int M;
  const uint16_t* sw;      // shared expert gate_up weight bf16 [2 I, K] (gate rows, then up rows), or null
  int I;
  float limit;             // swiglu clamp (inf: none)
  uint16_t* act;           // [M, I] bf16: silu(min(g, L)) * clamp(u, -L, L), g / u the bf16-rounded GEMM outputs
};

// rrms of the M rows, block-wide: thread t accumulates quads t, t + 256, ... of every row at once (the loads of all
// rows in flight), wave sums, then the eight waves in order.
template <int MT>
__device__ __forceinline__ void block_rrms(const uint16_t* x, long sx, int M, int K, float eps, int mode, float* s_rr,
                                           float (*red)[NW], int t) {
  const int lane = t & 31, w = t >> 5;
  float s[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) s[m] = 0.f;
  for (int q = t; q < K / 4; q += NT) {
#pragma unroll
    for (int m = 0; m < MT; m++) {
      if (m < M) {
        float v[4];
        ld4(x + (long)m * sx, 0, 4 * q, v);
#pragma unroll
        for (int i = 0; i < 4; i++) s[m] = fmaf(v[i], v[i], s[m]);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < MT; m++) {
    if (m < M) {
      const float r = wave_sum(s[m]);
      if (lane == 0) red[m][w] = r;
    }
  }
  __syncthreads();
  if (t < M) {
    float r = red[t][0];
#pragma unroll
    for (int i = 1; i < NW; i++) r += red[t][i];
    s_rr[t] = rrms_of(r, K, eps, mode);
  }
  __syncthreads();
}

typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
#ifdef R9K_SK_FMA
  c = fmaf(__uint_as_float(a << 16), __uint_as_float(b << 16), c);
  return fmaf(__uint_as_float(a & 0xffff0000u), __uint_as_float(b & 0xffff0000u), c);
#else
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2, a), __builtin_bit_cast(bf16x2, b), c, false);
#endif
}

// ----------------------------------------------------------------------------- split-K skinny GEMV (router / attn input)
// Grid (K / SK slices, groups). A group is 64 weight rows of one matrix: router experts, shared-expert column pairs
// (gate row j, up row I + j; 32 j a group), or plain rows. Block (s, g): the rows' rrms (when normalizing), the slice's
// normalized rows staged in LDS (block (s, 0) also stores them to xn), each wave 8 weight rows x all M rows (bf16 dot2,
// f32), partials to scratch; the last slice block of a group to arrive (ticket) sums the slices in order and runs the
// group's epilogue (router logits; silu-and-mul of the bf16-rounded gate / up; f32 or bf16 store); the last router group
// then runs the top-k. Counters self-reset.
constexpr int SK = 512, GR = 64, MAXG = 48;
struct SkMat { const uint16_t* w; int rows; int kind; void* out; };   // kind 0 router, 1 shared, 2 f32, 3 bf16
struct SkArgs {
  RouteArgs r;              // x, sx, nw, eps, mode, xn, K, M, the router's top-k fields, logits, ctr (route ticket)
  SkMat mat[3];
  int nmat;
  int g_mat[MAXG], g_row0[MAXG], g_slot[MAXG];
  int ngroups, route_groups, nslots;
  float* part;              // [slices][M][nslots]
  int* gctr;                // [MAXG]
};

template <int MT>
__global__ void __launch_bounds__(NT) r9k_dsv4_sk_kernel(SkArgs A) {
  const RouteArgs& a = A.r;
  __shared__ float s_rr[MAXM];
  __shared__ float red[MAXM][NW];
  extern __shared__ __attribute__((aligned(16))) uint16_t s_x[];   // [M][SK], sized at launch
  __shared__ int last;
  const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
  const int sl = blockIdx.x, g = blockIdx.y, nsl = gridDim.x;
  const bool nrm = a.nw != nullptr;
#ifndef R9K_SK_NO_RRMS
  if (nrm) block_rrms<MT>(a.x, a.sx, a.M, a.K, a.eps, a.mode, s_rr, red, t);
#else
  if (t < MAXM) s_rr[t] = 1.f;
  __syncthreads();
#endif
  const int k0 = sl * SK;
  for (int it = t; it < a.M * (SK / 4); it += NT) {
    const int m = it / (SK / 4), kk = 4 * (it % (SK / 4)), k = k0 + kk;
    const v4u16 xu = *reinterpret_cast<const v4u16*>(a.x + (long)m * a.sx + k);
    v4u16 o;
    if (nrm) {
      const v4u16 wu = *reinterpret_cast<const v4u16*>(a.nw + k);
#pragma unroll
      for (int i = 0; i < 4; i++) o[i] = f2bf(norm_elem(bf2f(xu[i]), s_rr[m], true, wu[i], a.mode));
      if (g == 0) *reinterpret_cast<v4u16*>(a.xn + (long)m * a.K + k) = o;
    } else {
      o = xu;
    }
    *reinterpret_cast<v4u16*>(s_x + m * SK + kk) = o;
  }
  __syncthreads();
  const SkMat& mt = A.mat[A.g_mat[g]];
  const int row0 = A.g_row0[g], slot0 = A.g_slot[g];
  // thread t: rows 4 rg .. 4 rg + 3 (rg = t / 16) x the slice's 16-byte chunks kq, kq + 16, kq + 32, kq + 48
  // (kq = t % 16): one staged-x load feeds four rows; lanes read consecutive chunks (no LDS bank conflicts, coalesced
  // weights); the 16 lanes of a row group are summed with four xor shuffles
  {
    float* pp = A.part + (long)sl * a.M * A.nslots;
    const int rg = t >> 4, kq = t & 15;
    uint4 wv[4][4];
    bool live[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int r = rg * 4 + i;
      long wrow;
      if (mt.kind == 1) {                           // shared: pair (j, gate | up)
        const int j = row0 + (r >> 1);
        live[i] = j < a.I;
        wrow = (r & 1) ? (long)a.I + j : j;
      } else {
        const int e = row0 + r;
        live[i] = e < mt.rows;
        wrow = e;
      }
      const uint4* wp = reinterpret_cast<const uint4*>(mt.w + (live[i] ? wrow : 0) * a.K + k0);
#pragma unroll
      for (int q = 0; q < 4; q++) wv[i][q] = wp[kq + 16 * q];
    }
#pragma unroll
    for (int m = 0; m < MT; m++) {
      if (m < a.M) {
        const uint4* xp = reinterpret_cast<const uint4*>(s_x + m * SK);
        float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int q = 0; q < 4; q++) {
          const uint4 xw = xp[kq + 16 * q];
#pragma unroll
          for (int i = 0; i < 4; i++)
            s4[i] = dot2(xw.w, wv[i][q].w, dot2(xw.z, wv[i][q].z, dot2(xw.y, wv[i][q].y, dot2(xw.x, wv[i][q].x, s4[i]))));
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
          float v = s4[i];                            // xor-sum within the 16-lane row (DPP row_xmask, gfx10+)
          v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x161, 0xf, 0xf, false));
          v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x162, 0xf, 0xf, false));
          v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x164, 0xf, 0xf, false));
          v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x168, 0xf, 0xf, false));
          if (kq == 0 && live[i]) pp[(long)m * A.nslots + slot0 + rg * 4 + i] = v;
        }
      }
    }
  }
#ifdef R9K_SK_NO_TAIL
  return;
#endif
  __threadfence();
  __syncthreads();
  if (t == 0) last = (atomicAdd(&A.gctr[g], 1) == nsl - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  // group epilogue: slices summed in order
  for (int it = t; it < a.M * GR; it += NT) {
    const int m = it / GR, r = it % GR;
    if (mt.kind == 1 && (r & 1)) continue;          // the gate entry of a pair does the pair
    const int nper = mt.kind == 1 ? 2 : 1;
    float v[2] = {0.f, 0.f};
    for (int q = 0; q < nper; q++) {
      const long o = (long)m * A.nslots + slot0 + r + q;
      float sm = A.part[o];
      for (int s2 = 1; s2 < nsl; s2++) sm += A.part[(long)s2 * a.M * A.nslots + o];
      v[q] = sm;
    }
    if (mt.kind == 0) {
      const int e = row0 + r;
      if (e < mt.rows) a.logits[(long)m * a.E + e] = v[0];
    } else if (mt.kind == 1) {
      const int j = row0 + (r >> 1);
      if (j < a.I) {
        const float gg = fminf(rbf(v[0]), a.limit);
        const float uu = fmaxf(fminf(rbf(v[1]), a.limit), -a.limit);
        a.act[(long)m * a.I + j] = f2bf((gg / (1.0f + expf(-gg))) * uu);
      }
    } else {
      const int e = row0 + r;
      if (e < mt.rows) {
        if (mt.kind == 2) ((float*)mt.out)[(long)m * mt.rows + e] = v[0];
        else ((uint16_t*)mt.out)[(long)m * mt.rows + e] = f2bf(v[0]);
      }
    }
  }
  if (t == 0) A.gctr[g] = 0;
  if (mt.kind != 0) return;
  // the last router group to finish runs the top-k
  __threadfence();
  __syncthreads();
  if (t == 0) last = (atomicAdd(a.ctr, 1) == A.route_groups - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  const int per = a.E / 32;                                  // experts a lane (E % 32 == 0)
  for (int m = wave; m < a.M; m += NW) {
    bool vl = false;
    if (a.bias_vl && a.sentinel_lo > 0 && a.ids) {
      const long tok = a.ids64 ? ((const long*)a.ids)[m] : (long)((const int*)a.ids)[m];
      vl = tok >= a.sentinel_lo && tok < a.sentinel_lo + 5;
    }
    const float* rb = vl ? a.bias_vl : a.bias;
    float cv[MAXE / 32];
#pragma unroll
    for (int i = 0; i < MAXE / 32; i++) {
      if (i < per) {
        const int ex = lane + 32 * i;
        float v = __builtin_nontemporal_load(a.logits + (long)m * a.E + ex);
        v = v > 20.f ? v : __logf(1.f + __expf(v));
        v = sqrtf(v);
        if (isnan(v)) v = 0.f;
        cv[i] = rb ? v + rb[ex] : v;
      }
    }
    float wsel[MAXTOPK];
    int isel[MAXTOPK];
    float sum = 0.f;
    for (int k = 0; k < a.topk; k++) {
      float bv = -INFINITY;
      int bi = 0x7fffffff;
#pragma unroll
      for (int i = 0; i < MAXE / 32; i++)
        if (i < per && cv[i] > bv) { bv = cv[i]; bi = lane + 32 * i; }   // ascending ids: first max wins ties
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor(bv, o, 32);
        const int oi = __shfl_xor(bi, o, 32);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      if ((bi & 31) == lane) {
#pragma unroll
        for (int i = 0; i < MAXE / 32; i++)
          if (i == (bi >> 5)) cv[i] = -10000.f;
      }
      float wv = bv;
      if (rb) wv -= rb[bi];
      if (a.renorm) sum += wv;
      wsel[k] = wv;
      isel[k] = bi;
    }
    float scale = a.rsf;
    if (a.renorm) scale /= (sum > 0.f ? sum : 1.f);
    const bool pd = a.pad && a.pad[m];
    if (lane == 0) {
      for (int k = 0; k < a.topk; k++) {
        a.w_out[(long)m * a.topk + k] = a.renorm ? wsel[k] * scale : wsel[k] * a.rsf;
        a.id_out[(long)m * a.topk + k] = pd ? -1 : isel[k];
      }
    }
  }
  if (t == 0) *a.ctr = 0;
}

// ------------------------------------------------------------------------------------------------- DSpark Markov head
// One launch a draft step (greedy drafts): the previous token per request (step 0: input_ids[anchor_idx[r]]; later: the
// first maximum of the previous step's block maxima -- block 0 also stores it into draft_tokens[r, step - 1]), its
// Markov embedding w1[prev] (bf16 [R]), the bias GEMV over the vocabulary (w2 [V, R] bf16, f32 sum rounded to bf16 as
// vLLM's GEMV output; or an fp8 e4m3 copy with a per-row f32 scale), + the bf16 base logits (one bf16 rounding of the f32
// sum, as a bf16 tensor add), optionally the logits stored (the adaptive-verification estimator reads them), and this
// block's first maximum + its token id per request. 256 vocab rows a block (8 waves x 32 rows, 8 lanes a row, R = 256:
// 32 elements a lane); the weights are read once for all requests. r9k_dsv4_markov_last: the last step's tokens.
constexpr int MK_BLOCK = 256, MK_MAXR = 8, MK_R = 256;

template <bool FP8>
__global__ void __launch_bounds__(NT) r9k_dsv4_markov_kernel(
    const float* __restrict__ prev_max, const int* __restrict__ prev_arg, float* __restrict__ out_max,
    int* __restrict__ out_arg, const int* __restrict__ input_ids, const long* __restrict__ anchor,
    long* __restrict__ tokens, long tok_stride, const uint16_t* __restrict__ w1, const void* __restrict__ w2,
    const float* __restrict__ w2_scale, const uint16_t* __restrict__ base, long base_stride,
    uint16_t* __restrict__ logits, long logits_stride, int V, int nb, int step, int rows) {
  __shared__ __attribute__((aligned(16))) float s_e[MK_MAXR][MK_R];
  __shared__ __attribute__((aligned(16))) uint16_t s_eb[MK_MAXR][MK_R];
  __shared__ float s_v[MK_MAXR][MK_BLOCK];
  __shared__ int s_prev[MK_MAXR];
  const int t = threadIdx.x, lane = t & 31, wave = t >> 5, b = blockIdx.x;
  if (wave < rows) {
    const int r = wave;
    int prev;
    if (step == 0) {
      prev = input_ids[anchor[r]];
    } else {
      float bv = -INFINITY;
      int bi = 0x7fffffff;
      for (int j = lane; j < nb; j += 32) {          // ascending blocks: the first max of a lane wins its ties
        const float v = prev_max[r * nb + j];
        if (v > bv) { bv = v; bi = j; }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor(bv, o, 32);
        const int oi = __shfl_xor(bi, o, 32);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      prev = prev_arg[r * nb + bi];
      if (b == 0 && lane == 0) tokens[(long)r * tok_stride + step - 1] = prev;
    }
    if (lane == 0) s_prev[r] = prev;
  }
  __syncthreads();
  for (int i = t; i < rows * MK_R; i += NT) {
    const int r = i / MK_R, k = i % MK_R;
    const uint16_t ev = w1[(long)s_prev[r] * MK_R + k];
    s_e[r][k] = bf2f(ev);
    s_eb[r][k] = ev;
  }
  __syncthreads();
  const int sub = lane & 7, rw = lane >> 3;          // 4 rows a wave per pass, 8 lanes a row, 32 elements a lane
  for (int pass = 0; pass < 8; pass++) {
    const int lr = wave * 32 + pass * 4 + rw;         // row within the block
    const int v = b * MK_BLOCK + lr;
    const bool live = v < V;
    float wf[32];
    int ki[32];                                       // the element index each weight value belongs to
    uint4 wq[4];
    const int vv = live ? v : 0;
    if constexpr (FP8) {
      // 4-byte words interleaved over the 8 lanes of a row: word q * 8 + sub = elements q * 32 + sub * 4 .. + 3
      const uint32_t* wp = reinterpret_cast<const uint32_t*>((const uint8_t*)w2 + (long)vv * MK_R);
      const float sc = w2_scale[vv];
#pragma unroll
      for (int q = 0; q < 8; q++) {
        const uint32_t u = wp[q * 8 + sub];
        wf[4 * q + 0] = __builtin_amdgcn_cvt_f32_fp8(u, 0) * sc;
        wf[4 * q + 1] = __builtin_amdgcn_cvt_f32_fp8(u, 1) * sc;
        wf[4 * q + 2] = __builtin_amdgcn_cvt_f32_fp8(u, 2) * sc;
        wf[4 * q + 3] = __builtin_amdgcn_cvt_f32_fp8(u, 3) * sc;
#pragma unroll
        for (int j = 0; j < 4; j++) ki[4 * q + j] = q * 32 + sub * 4 + j;
      }
    } else {
      // 16-byte chunks interleaved over the 8 lanes: chunk q * 8 + sub = elements q * 64 + sub * 8 .. + 7
      const uint4* wp = reinterpret_cast<const uint4*>((const uint16_t*)w2 + (long)vv * MK_R);
#pragma unroll
      for (int q = 0; q < 4; q++) wq[q] = wp[q * 8 + sub];
    }
    for (int r = 0; r < rows; r++) {
      float acc = 0.f;
      if constexpr (FP8) {
#pragma unroll
        for (int i = 0; i < 32; i += 4) {
          const float4 ev = *reinterpret_cast<const float4*>(&s_e[r][ki[i]]);
          acc = fmaf(wf[i], ev.x, acc);
          acc = fmaf(wf[i + 1], ev.y, acc);
          acc = fmaf(wf[i + 2], ev.z, acc);
          acc = fmaf(wf[i + 3], ev.w, acc);
        }
      } else {
#pragma unroll
        for (int q = 0; q < 4; q++) {
          const uint4 ew = *reinterpret_cast<const uint4*>(&s_eb[r][q * 64 + sub * 8]);
          acc = dot2(wq[q].w, ew.w, dot2(wq[q].z, ew.z, dot2(wq[q].y, ew.y, dot2(wq[q].x, ew.x, acc))));
        }
      }
      acc += __shfl_xor(acc, 1, 32);
      acc += __shfl_xor(acc, 2, 32);
      acc += __shfl_xor(acc, 4, 32);
      if (sub == 0) {
        float lg = -INFINITY;
        if (live) {
          lg = rbf(bf2f(base[(long)r * base_stride + v]) + rbf(acc));
          if (logits) logits[(long)r * logits_stride + v] = f2bf(lg);
        }
        s_v[r][lr] = lg;
      }
    }
  }
  __syncthreads();
  if (wave < rows) {
    const int r = wave;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int j = lane; j < MK_BLOCK; j += 32) {
      const float v = s_v[r][j];
      if (v > bv) { bv = v; bi = j; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor(bv, o, 32);
      const int oi = __shfl_xor(bi, o, 32);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) {
      out_max[r * nb + b] = bv;
      out_arg[r * nb + b] = b * MK_BLOCK + bi;
    }
  }
}

__global__ void r9k_dsv4_markov_last_kernel(const float* __restrict__ prev_max, const int* __restrict__ prev_arg,
                                            long* __restrict__ tokens, long tok_stride, int nb, int step) {
  const int r = blockIdx.x, lane = threadIdx.x;
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int j = lane; j < nb; j += 32) {
    const float v = prev_max[r * nb + j];
    if (v > bv) { bv = v; bi = j; }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor(bv, o, 32);
    const int oi = __shfl_xor(bi, o, 32);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  if (lane == 0) tokens[(long)r * tok_stride + step] = prev_arg[r * nb + bi];
}

// DSpark confidence head on the drafts (adaptive verification with the checkpoint's trained confidence head; stock:
// gather + cat + .float() + an f32 [K -> 1] GEMM + sigmoid + copy). One block per (request r, draft step i), b = r * n + i:
// x = [hidden[b] (bf16 [H]), w1[prev(r, i)] (bf16 [R])], conf[r, i] = sigmoid(f32(x) . proj (f32 [H + R]) + bias),
// prev(r, 0) = input_ids[anchor[r]], prev(r, i) = tokens[r, i - 1] (the drafts the Markov launches wrote). The dot is
// summed in f64 and rounded once to f32 (within half an f32 ulp of the exact dot; stock's f32 GEMM sits within its own
// summation order's error of the same value), the sigmoid in f32 as torch's: 1 / (1 + exp(-z)).
__global__ void __launch_bounds__(NT) r9k_dsv4_markov_conf_kernel(
    const uint16_t* __restrict__ hidden, long h_stride, int H, const uint16_t* __restrict__ w1, int R,
    const float* __restrict__ proj, const float* __restrict__ pbias, const int* __restrict__ input_ids,
    const long* __restrict__ anchor, const long* __restrict__ tokens, long tok_stride, float* __restrict__ conf,
    long conf_stride, int n_spec) {
  __shared__ double s_p[NT / 32];
  const int b = blockIdx.x, r = b / n_spec, i = b % n_spec, t = threadIdx.x, lane = t & 31, wave = t >> 5;
  const long prev = i == 0 ? (long)input_ids[anchor[r]] : tokens[(long)r * tok_stride + i - 1];
  const uint16_t* hx = hidden + (long)b * h_stride;
  const uint16_t* ex = w1 + prev * R;
  double acc = 0.0;
  for (int k = t; k < H; k += NT) acc = fma((double)bf2f(hx[k]), (double)proj[k], acc);
  for (int k = t; k < R; k += NT) acc = fma((double)bf2f(ex[k]), (double)proj[H + k], acc);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor(acc, o, 32);
  if (lane == 0) s_p[wave] = acc;
  __syncthreads();
  if (t == 0) {
    double z = 0.0;
    for (int w = 0; w < NT / 32; w++) z += s_p[w];
    float zf = (float)z;
    if (pbias) zf += pbias[0];
    conf[(long)r * conf_stride + i] = 1.f / (1.f + expf(-zf));
  }
}

}  // namespace

extern "C" int r9k_dsv4_glue_max_m() { return MAXM; }

// Two segments (seg1 n = 0: one). eps / mode as above. Rules: M in 1..MAXM (any M works for the norm; MAXM is the
// router's), n % 128 == 0, 16-byte aligned rows. Returns 0 or a negative shape code.
extern "C" int r9k_dsv4_norm_pre(long in0, long in0_stride, int in0_f32, int n0, long w0, int norm0, long ob0,
                                 long ob0_stride, long oh0, long oh0_stride, long suh0,
                                 long in1, long in1_stride, int in1_f32, int n1, long w1, int norm1, long ob1,
                                 long ob1_stride, long oh1, long oh1_stride, long suh1,
                                 int M, float eps, int mode, long stream) {
  if (M < 1 || n0 <= 0 || n0 % 128 || n1 < 0 || n1 % 128 || mode < 0 || mode > 1) return -1;
  NormArgs a{};
  a.s[0] = NormSeg{(const void*)in0, in0_stride, in0_f32, n0, (const uint16_t*)w0, norm0, (uint16_t*)ob0, ob0_stride,
                   (uint16_t*)oh0, oh0_stride, (const uint16_t*)suh0};
  a.s[1] = NormSeg{(const void*)in1, in1_stride, in1_f32, n1, (const uint16_t*)w1, norm1, (uint16_t*)ob1, ob1_stride,
                   (uint16_t*)oh1, oh1_stride, (const uint16_t*)suh1};
  a.M = M; a.eps = eps; a.mode = mode;
  dim3 grid(M, n1 > 0 ? 2 : 1);
  hipLaunchKernelGGL(r9k_dsv4_norm_kernel, grid, dim3(NT), 0, (hipStream_t)stream, a);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

extern "C" int r9k_dsv4_markov_blocks(int V) { return (V + MK_BLOCK - 1) / MK_BLOCK; }

// One greedy DSpark Markov step (see r9k_dsv4_markov_kernel). R must be 256, rows <= 8; w2 fp8 when w2_scale != 0.
// prev_* / out_* scratch: f32 / int32 [8, nb]. logits: bf16 [rows, V] (row stride) or 0.
extern "C" int r9k_dsv4_markov_step(long prev_max, long prev_arg, long out_max, long out_arg, long input_ids,
                                    long anchor, long tokens, long tok_stride, long w1, long w2, long w2_scale,
                                    long base, long base_stride, long logits, long logits_stride, int V, int R,
                                    int step, int rows, long stream) {
  if (R != MK_R || rows < 1 || rows > MK_MAXR || V < 1 || step < 0) return -1;
  const int nb = (V + MK_BLOCK - 1) / MK_BLOCK;
  hipStream_t st = (hipStream_t)stream;
  if (w2_scale)
    hipLaunchKernelGGL(r9k_dsv4_markov_kernel<true>, dim3(nb), dim3(NT), 0, st, (const float*)prev_max,
                       (const int*)prev_arg, (float*)out_max, (int*)out_arg, (const int*)input_ids, (const long*)anchor,
                       (long*)tokens, tok_stride, (const uint16_t*)w1, (const void*)w2, (const float*)w2_scale,
                       (const uint16_t*)base, base_stride, (uint16_t*)logits, logits_stride, V, nb, step, rows);
  else
    hipLaunchKernelGGL(r9k_dsv4_markov_kernel<false>, dim3(nb), dim3(NT), 0, st, (const float*)prev_max,
                       (const int*)prev_arg, (float*)out_max, (int*)out_arg, (const int*)input_ids, (const long*)anchor,
                       (long*)tokens, tok_stride, (const uint16_t*)w1, (const void*)w2, (const float*)nullptr,
                       (const uint16_t*)base, base_stride, (uint16_t*)logits, logits_stride, V, nb, step, rows);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

extern "C" int r9k_dsv4_markov_last(long prev_max, long prev_arg, long tokens, long tok_stride, int V, int step,
                                    int rows, long stream) {
  if (rows < 1 || rows > MK_MAXR) return -1;
  const int nb = (V + MK_BLOCK - 1) / MK_BLOCK;
  hipLaunchKernelGGL(r9k_dsv4_markov_last_kernel, dim3(rows), dim3(32), 0, (hipStream_t)stream, (const float*)prev_max,
                     (const int*)prev_arg, (long*)tokens, tok_stride, nb, step);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

// The confidence head over rows x n_spec drafts (r9k_dsv4_markov_conf_kernel); run after r9k_dsv4_markov_last.
// hidden bf16 [rows * n_spec, H] (row stride h_stride), w1 bf16 [V, R] (R = 0: no Markov input), proj f32 [H + R],
// pbias f32 [1] or 0, tokens int64 (tok_stride), conf f32 (conf_stride; unit step stride).
extern "C" int r9k_dsv4_markov_conf(long hidden, long h_stride, int H, long w1, int R, long proj, long pbias,
                                    long input_ids, long anchor, long tokens, long tok_stride, long conf,
                                    long conf_stride, int rows, int n_spec, long stream) {
  if (rows < 1 || n_spec < 1 || H < 1 || R < 0 || (R && !w1)) return -1;
  hipLaunchKernelGGL(r9k_dsv4_markov_conf_kernel, dim3(rows * n_spec), dim3(NT), 0, (hipStream_t)stream,
                     (const uint16_t*)hidden, h_stride, H, (const uint16_t*)w1, R, (const float*)proj,
                     (const float*)pbias, (const int*)input_ids, (const long*)anchor, (const long*)tokens, tok_stride,
                     (float*)conf, conf_stride, n_spec);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

// Split-K router / attention-input GEMV (r9k_dsv4_sk_kernel). mats: up to three (w, rows, kind, out); kind 0 = router
// (rows = E; logits + top-k as r9k_dsv4_route), 1 = shared expert gate_up (rows = 2 I; act), 2 / 3 = f32 / bf16 out.
// part: f32 scratch >= (K / 512) * M * slots (slots = sum of the matrices' rows rounded to 64); gctr: int32[48] zeroed.
extern "C" int r9k_dsv4_sk(long x, long sx, long nw, float eps, int mode, long xn, int K, int M,
                           long w0, int rows0, int kind0, long out0, long w1, int rows1, int kind1, long out1,
                           long w2, int rows2, int kind2, long out2,
                           long bias, long bias_vl, long ids, int ids64, int sentinel_lo, long pad, int topk, int renorm,
                           float rsf, long logits, long w_out, long id_out, long ctr, float limit, long act,
                           long part, long gctr, long stream) {
  if (M < 1 || M > MAXM || K % SK || (sx & 7) || (nw && !xn)) return -1;
  SkArgs A{};
  RouteArgs& r = A.r;
  r.x = (const uint16_t*)x; r.sx = sx; r.nw = (const uint16_t*)nw; r.eps = eps; r.mode = mode; r.xn = (uint16_t*)xn;
  r.K = K; r.M = M; r.bias = (const float*)bias; r.bias_vl = (const float*)bias_vl; r.ids = (const void*)ids;
  r.ids64 = ids64; r.sentinel_lo = sentinel_lo; r.pad = (const uint8_t*)pad; r.topk = topk; r.renorm = renorm;
  r.rsf = rsf; r.logits = (float*)logits; r.w_out = (float*)w_out; r.id_out = (int*)id_out; r.ctr = (int*)ctr;
  r.limit = limit; r.act = (uint16_t*)act;
  const long ws[3] = {w0, w1, w2};
  const int rs[3] = {rows0, rows1, rows2}, ks[3] = {kind0, kind1, kind2};
  const long os[3] = {out0, out1, out2};
  int ng = 0, slots = 0;
  for (int i = 0; i < 3; i++) {
    if (!ws[i] || rs[i] <= 0) continue;
    const int mi = A.nmat++;
    A.mat[mi] = SkMat{(const uint16_t*)ws[i], rs[i], ks[i], (void*)os[i]};
    if (ks[i] == 0) {
      if (rs[i] % 32 || rs[i] > MAXE || topk < 1 || topk > MAXTOPK || !logits || !ctr) return -2;
      r.E = rs[i];
    }
    if (ks[i] == 1) { if (rs[i] % 2 || !act) return -3; r.I = rs[i] / 2; }
    const int per = ks[i] == 1 ? GR / 2 : GR;          // matrix rows (or j's) a group
    const int n = ks[i] == 1 ? rs[i] / 2 : rs[i];
    for (int r0 = 0; r0 < n; r0 += per) {
      if (ng >= MAXG) return -4;
      A.g_mat[ng] = mi; A.g_row0[ng] = r0; A.g_slot[ng] = slots; slots += GR;
      if (ks[i] == 0) A.route_groups++;
      ng++;
    }
  }
  if (!ng) return -5;
  A.ngroups = ng; A.nslots = slots; A.part = (float*)part; A.gctr = (int*)gctr;
  dim3 grid(K / SK, ng);
  hipStream_t st = (hipStream_t)stream;
  const size_t smem = (size_t)M * SK * sizeof(uint16_t);
  if (M <= 1) hipLaunchKernelGGL(r9k_dsv4_sk_kernel<1>, grid, dim3(NT), smem, st, A);
  else if (M <= 2) hipLaunchKernelGGL(r9k_dsv4_sk_kernel<2>, grid, dim3(NT), smem, st, A);
  else if (M <= 4) hipLaunchKernelGGL(r9k_dsv4_sk_kernel<4>, grid, dim3(NT), smem, st, A);
  else if (M <= 8) hipLaunchKernelGGL(r9k_dsv4_sk_kernel<8>, grid, dim3(NT), smem, st, A);
  else if (M <= 16) hipLaunchKernelGGL(r9k_dsv4_sk_kernel<16>, grid, dim3(NT), smem, st, A);
  else hipLaunchKernelGGL(r9k_dsv4_sk_kernel<32>, grid, dim3(NT), smem, st, A);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

extern "C" long r9k_dsv4_sk_part_floats(int K, int M, int slots) { return (long)(K / SK) * M * slots; }
