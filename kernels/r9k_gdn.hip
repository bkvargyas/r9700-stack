// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas

// r9k_gdn.hip -- Gated DeltaNet MTP decode after the causal conv, in one launch per (sequence, value head):
// sigmoid gating + the recurrent delta rule over the sequence's draft tokens + RMSNormGated on the output.
// Stock vLLM has this as the CUDA op fused_gdn_decode_post_conv_mtp, which is not built for ROCm; the fallback is
// the Triton recurrence wrapped in ~9 glue launches per layer (b/a contiguous, zeros, q/k/v cat, output copy, gated
// norm) -- 36 layers x ~11 HIP-graph nodes at ~1.5 us of dispatch each on this ROCm.
// Math and rounding points follow that Triton composition (tests/test_gdn_decode_r9k.py counts the flips):
//   fused_sigmoid_gating_delta_rule_update: fp32 state math; per token
//     g = -exp(A_log) * softplus(a + dt_bias) (threshold 20), beta = sigmoid(b), q,k l2-normalised (eps 1e-6),
//     q *= scale, h *= exp(g), v -= h.k, v *= beta, h += v (x) k, o = h.q rounded to bf16,
//     h stored (state dtype) at state_indices[n, t] when > 0; the initial h comes from
//     state_indices[n, num_accepted[n] - 1], and a sequence whose initial index is <= 0 is skipped (output 0).
//   layer_norm_fwd (RMS, norm_before_gate): y = (bf16 o as fp32) * rsqrt(mean(o^2) + eps) * w * gate(z), gate = silu
//   (z * sigmoid(z)) or, gate_sigmoid != 0, sigmoid(z) (Flash-Next's output_gate_type).
// Block = 256 threads: V row v = tid / 2, K half tid & 1 (64 fp32 of h per thread); K reductions pair via shfl,
// the norm's V reduction goes through LDS. Row layout of mixed_qkv: [q: Hk x 128 | k: Hk x 128 | v: Hv x 128];
// value head hv reads q/k head hv / (Hv / Hk). Rows past the last sequence (padding) get zero output. No printf/assert.
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int NT = 256, KD = 128, VD = 128, KH = 64, WAVES = NT / 32;
typedef uint16_t v8u16 __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bf2f(uint16_t u) { return __uint_as_float((unsigned)u << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {    // round to nearest even (torch's bf16 conversion)
  unsigned u = __float_as_uint(f);
  if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0xffffu) ? 0x40u : 0u));
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }

// 64 bf16 at p (16-byte aligned) -> fp32
__device__ __forceinline__ void load64(const uint16_t* __restrict__ p, float* f) {
#pragma unroll
  for (int c = 0; c < 8; c++) {
    v8u16 x = *reinterpret_cast<const v8u16*>(p + c * 8);
#pragma unroll
    for (int i = 0; i < 8; i++) f[c * 8 + i] = bf2f(x[i]);
  }
}

template <bool STATE_F32>
__device__ __forceinline__ void load_state(const void* __restrict__ s, float* h) {
  if constexpr (STATE_F32) {
    const v4f* p = reinterpret_cast<const v4f*>(s);
#pragma unroll
    for (int c = 0; c < 16; c++) {
      v4f x = p[c];
      h[c * 4] = x[0]; h[c * 4 + 1] = x[1]; h[c * 4 + 2] = x[2]; h[c * 4 + 3] = x[3];
    }
  } else {
    load64(reinterpret_cast<const uint16_t*>(s), h);
  }
}

template <bool STATE_F32>
__device__ __forceinline__ void store_state(void* __restrict__ s, const float* h) {
  if constexpr (STATE_F32) {
    v4f* p = reinterpret_cast<v4f*>(s);
#pragma unroll
    for (int c = 0; c < 16; c++) {
      v4f x = {h[c * 4], h[c * 4 + 1], h[c * 4 + 2], h[c * 4 + 3]};
      p[c] = x;
    }
  } else {
    v8u16* p = reinterpret_cast<v8u16*>(s);
#pragma unroll
    for (int c = 0; c < 8; c++) {
      v8u16 x;
#pragma unroll
      for (int i = 0; i < 8; i++) x[i] = f2bf(h[c * 8 + i]);
      p[c] = x;
    }
  }
}

template <bool STATE_F32>
__global__ void __launch_bounds__(NT) r9k_gdn_decode_mtp_kernel(
    const uint16_t* __restrict__ qkv, long sqkv, const uint16_t* __restrict__ bp, long sb,
    const uint16_t* __restrict__ ap, long sa, const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    const float* __restrict__ w, const uint16_t* __restrict__ z, long sz, uint16_t* __restrict__ out, long sout,
    void* __restrict__ state, long sstate, const int* __restrict__ idx, long sidx, const int* __restrict__ cu,
    const int* __restrict__ acc, int N, int Hk, int Hv, int rows_total, float scale, float eps, int gate_sigmoid) {
  const int n = blockIdx.x / Hv, hv = blockIdx.x - n * Hv;
  const int tid = threadIdx.x, v = tid >> 1, kh = tid & 1, lane = tid & 31, wave = tid >> 5;
  __shared__ float lds_o[VD];
  __shared__ float lds_part[WAVES];
  const int bos = cu[n], T = cu[n + 1] - bos;
  if (n == 0) {                                   // padding rows past the last sequence: zero output
    const int tail = cu[N];
    for (int r = tail + (tid >> 7); r < rows_total; r += 2) out[(long)r * sout + hv * VD + (tid & 127)] = 0;
  }
  if (T <= 0) return;
  const int i0 = max(acc[n] - 1, 0);
  const int idx0 = idx[(long)n * sidx + i0];
  if (idx0 <= 0) {
    for (int t = 0; t < T; t++) out[(long)(bos + t) * sout + hv * VD + (tid & 127)] = 0;
    return;
  }
  const long elem = STATE_F32 ? 4 : 2;
  const long hoff = ((long)hv * VD + v) * KD + kh * KH;
  float h[KH];
  load_state<STATE_F32>((const char*)state + ((long)idx0 * sstate + hoff) * elem, h);
  const int hk = hv / (Hv / Hk);
  const float negA = -expf(A_log[hv]), dtb = dt_bias[hv], wv = w[v];
  const long qoff = (long)hk * KD + kh * KH, koff = (long)(Hk + hk) * KD + kh * KH;
  const long voff = (long)(2 * Hk) * KD + (long)hv * VD + v;
  for (int t = 0; t < T; t++) {
    const long row = bos + t;
    const uint16_t* rp = qkv + row * sqkv;
    float k[KH];
    load64(rp + koff, k);
    float sk = 0.f, sq = 0.f;
#pragma unroll
    for (int i = 0; i < KH; i++) sk += k[i] * k[i];
    {                                                // q's sum of squares; q itself is streamed again for h.q below
      const v8u16* qp = reinterpret_cast<const v8u16*>(rp + qoff);
#pragma unroll
      for (int c = 0; c < 8; c++) {
        v8u16 x = qp[c];
#pragma unroll
        for (int i = 0; i < 8; i++) { const float f = bf2f(x[i]); sq += f * f; }
      }
    }
    sk += __shfl_xor(sk, 1, 32);
    sq += __shfl_xor(sq, 1, 32);
    const float rk = rsqrtf(sk + 1e-6f), rq = rsqrtf(sq + 1e-6f);
#pragma unroll
    for (int i = 0; i < KH; i++) k[i] = k[i] * rk;
    const float x = bf2f(ap[row * sa + hv]) + dtb;
    const float sp = x <= 20.f ? logf(1.f + expf(x)) : x;
    const float eg = expf(negA * sp);
    const float beta = sigmoidf_(bf2f(bp[row * sb + hv]));
    float vv = bf2f(rp[voff]);
    float d = 0.f;
#pragma unroll
    for (int i = 0; i < KH; i++) { h[i] = h[i] * eg; d += h[i] * k[i]; }
    d += __shfl_xor(d, 1, 32);
    vv = (vv - d) * beta;
    float o = 0.f;
    {
      const v8u16* qp = reinterpret_cast<const v8u16*>(rp + qoff);
#pragma unroll
      for (int c = 0; c < 8; c++) {
        v8u16 x = qp[c];
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const int j = c * 8 + i;
          h[j] = h[j] + vv * k[j];
          o += h[j] * ((bf2f(x[i]) * rq) * scale);
        }
      }
    }
    o += __shfl_xor(o, 1, 32);
    const int idxt = idx[(long)n * sidx + t];
    if (idxt > 0) store_state<STATE_F32>((char*)state + ((long)idxt * sstate + hoff) * elem, h);
    if (kh == 0) lds_o[v] = bf2f(f2bf(o));            // the Triton path rounds o to bf16 before the norm
    __syncthreads();
    float p = tid < VD ? lds_o[tid] * lds_o[tid] : 0.f;
    p = wave_sum(p);
    if (lane == 0) lds_part[wave] = p;
    __syncthreads();
    const float var = (lds_part[0] + lds_part[1] + lds_part[2] + lds_part[3]) / (float)VD;
    const float rstd = rsqrtf(var + eps);
    if (kh == 0) {
      const float zf = bf2f(z[row * sz + hv * VD + v]);
      float y = (lds_o[v] * rstd) * wv;
      const float sg = sigmoidf_(zf);
      y = y * (gate_sigmoid ? sg : zf * sg);
      out[row * sout + hv * VD + v] = f2bf(y);
    }
    __syncthreads();                                 // lds_o / lds_part reused by the next token
  }
}

}  // namespace

// mixed_qkv [rows, (2 Hk + Hv) x 128] bf16 (row stride sqkv), b / a [rows, Hv] bf16 (strides sb / sa), A_log /
// dt_bias [Hv] fp32, w [128] fp32, z [rows, Hv x 128] bf16 (stride sz), out [rows, Hv x 128] bf16 (stride sout),
// state [blocks, Hv, 128, 128] fp32 (state_f32) or bf16 (row stride sstate), idx [N, S] int32 (stride sidx),
// cu [N + 1] int32, acc [N] int32. Strides in elements. Returns 0 or a negative code.
extern "C" int r9k_gdn_decode_mtp(long qkv, long sqkv, long b, long sb, long a, long sa, long A_log, long dt_bias,
                                  long w, long z, long sz, long out, long sout, long state, long sstate,
                                  int state_f32, long idx, long sidx, int S, long cu, long acc, int N, int Hk,
                                  int Hv, int rows_total, float scale, float eps, int gate_sigmoid, long stream) {
  if (Hk < 1 || Hv < Hk || Hv % Hk || S < 1 || S > 64 || N < 0 || rows_total < 0) return -1;
  if ((sqkv | sz | sout | sstate) & 7) return -2;
  if (!qkv || !b || !a || !A_log || !dt_bias || !w || !z || !out || !state || !idx || !cu || !acc) return -3;
  if (N == 0) return 0;
  hipStream_t st = (hipStream_t)stream;
  const dim3 grid(N * Hv), block(NT);
  if (state_f32)
    hipLaunchKernelGGL(r9k_gdn_decode_mtp_kernel<true>, grid, block, 0, st, (const uint16_t*)qkv, sqkv,
                       (const uint16_t*)b, sb, (const uint16_t*)a, sa, (const float*)A_log, (const float*)dt_bias,
                       (const float*)w, (const uint16_t*)z, sz, (uint16_t*)out, sout, (void*)state, sstate,
                       (const int*)idx, sidx, (const int*)cu, (const int*)acc, N, Hk, Hv, rows_total, scale, eps,
                       gate_sigmoid);
  else
    hipLaunchKernelGGL(r9k_gdn_decode_mtp_kernel<false>, grid, block, 0, st, (const uint16_t*)qkv, sqkv,
                       (const uint16_t*)b, sb, (const uint16_t*)a, sa, (const float*)A_log, (const float*)dt_bias,
                       (const float*)w, (const uint16_t*)z, sz, (uint16_t*)out, sout, (void*)state, sstate,
                       (const int*)idx, sidx, (const int*)cu, (const int*)acc, N, Hk, Hv, rows_total, scale, eps,
                       gate_sigmoid);
  return hipGetLastError() == hipSuccess ? 0 : -4;
}
