// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas

// r9k_norm_rope.hip -- the QSA indexer's per-head glue in one launch: Gemma RMSNorm (fp32 math, x * rsqrt(mean(x^2)
// + eps) * (1 + w), rounded once to bf16) optionally followed by neox-style partial rotary embedding on the first R2
// elements with bf16 cos/sin, at exactly vLLM's rounding points:
//   o1 = bf16(bf16(x1 * cos) - bf16(x2 * sin)),  o2 = bf16(bf16(x2 * cos) + bf16(x1 * sin))
// (ApplyRotaryEmb.forward_static on bf16 tensors after the norm's own bf16 rounding). vLLM's Triton MRoPE kernel
// (3-row positions) rounds the two bf16 products toward zero and only the sum to nearest -- measured against the
// kernel, tests/test_indexer_glue_r9k.py -- so prow == 3 truncates the products. Stock runs the norm as ir.ops.rms_norm's native path (fp32 weight, ~10 ATen launches) and the
// 1-D rope as ~9 more, twice per QSA layer at decode; in a HIP graph every node costs ~1.5 us of dispatch on this
// ROCm, so the launches were the cost.
// Positions: 1 row (plain rope) or 3 rows [T/H/W][T] (MRoPE): half-dim j takes its cos/sin from position row
// sel(j) -- interleaved: 1 if j % 3 == 1 and j <= 3*sec_h, 2 if j % 3 == 2 and j <= 3*sec_w, else 0; concatenated:
// 0 for j < sec_t, 1 for j < sec_t + sec_h, else 2 (vLLM's _triton_mrope_forward masks).
// One wave per (token, head) row; D % 8 == 0, D <= 256, R2 % 16 == 0, R2 <= D. Row r = token r / H, head r % H at
// x + token * sx + head * sh. cos_sin: [max_pos, R2] bf16 = [cos(R2/2) | sin(R2/2)]. No device printf/assert.
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

typedef uint16_t v8u16 __attribute__((ext_vector_type(8)));
constexpr int WAVES = 8, NT = 32 * WAVES;

__device__ __forceinline__ float bf2f(uint16_t u) { return __uint_as_float((unsigned)u << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {    // round to nearest even (matches torch's bf16 conversion)
  unsigned u = __float_as_uint(f);
  if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0xffffu) ? 0x40u : 0u));
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float bf_trunc(float f) {   // fp32 -> bf16 toward zero, back to fp32
  return __uint_as_float(__float_as_uint(f) & 0xffff0000u);
}
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  return v;
}

struct MRope {
  int prow;            // position rows: 1 or 3
  long pstride;        // elements between position rows
  long ptok;           // elements between consecutive tokens within a row (1 when contiguous)
  int sec_t, sec_h, sec_w;
  int interleaved;
  int trunc;           // products rounded toward zero (the Triton MRoPE path)
};

__device__ __forceinline__ int mrope_sel(const MRope& m, int j) {
  if (m.prow == 1) return 0;
  if (m.interleaved) {
    if (j % 3 == 1 && j <= 3 * m.sec_h) return 1;
    if (j % 3 == 2 && j <= 3 * m.sec_w) return 2;
    return 0;
  }
  return j < m.sec_t ? 0 : (j < m.sec_t + m.sec_h ? 1 : 2);
}

template <bool ROPE>
__global__ void __launch_bounds__(NT) r9k_gemma_norm_rope_kernel(
    const uint16_t* __restrict__ x, long sx, long sh, const uint16_t* __restrict__ w, uint16_t* __restrict__ out,
    long so, long soh, int rows, int H, int D, float eps, const uint16_t* __restrict__ cs,
    const long* __restrict__ pos, int R2, MRope mr) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int r = blockIdx.x * WAVES + wave;
  if (r >= rows) return;                                 // uniform per wave
  const int t = r / H, h = r - t * H;
  const uint16_t* xp = x + (long)t * sx + (long)h * sh;
  uint16_t* op = out + (long)t * so + (long)h * soh;
  const int nch = D / 8;
  const bool act = lane < nch;
  v8u16 xv = (v8u16)(0);
  if (act) xv = *reinterpret_cast<const v8u16*>(xp + lane * 8);
  float v[8], ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    v[i] = bf2f(xv[i]);
    ss = fmaf(v[i], v[i], ss);
  }
  ss = wave_sum(ss);
  const float rr = rsqrtf(ss / (float)D + eps);
  v8u16 wv = (v8u16)(0);
  if (act) wv = *reinterpret_cast<const v8u16*>(w + lane * 8);
  v8u16 yb;
#pragma unroll
  for (int i = 0; i < 8; i++) yb[i] = f2bf((v[i] * rr) * (bf2f(wv[i]) + 1.f));
  if constexpr (ROPE) {
    const int half = R2 / 2, hc = half / 8;
    float yf[8], pf[8];
#pragma unroll
    for (int i = 0; i < 8; i++) yf[i] = bf2f(yb[i]);
    const int partner = lane < hc ? lane + hc : lane - hc;
#pragma unroll
    for (int i = 0; i < 8; i++) pf[i] = __shfl(yf[i], partner < 32 ? partner : lane, 32);
    if (lane < 2 * hc) {
      const bool first = lane < hc;
      const int cj = (first ? lane : lane - hc) * 8;
      const uint16_t* cp[3];
#pragma unroll
      for (int q = 0; q < 3; q++) cp[q] = cs + (long)pos[(q < mr.prow ? q : 0) * mr.pstride + (long)t * mr.ptok] * R2;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int j = cj + i;
        const uint16_t* c = cp[mrope_sel(mr, j)];
        const float pa = yf[i] * bf2f(c[j]), pb = pf[i] * bf2f(c[half + j]);
        const float a = mr.trunc ? bf_trunc(pa) : bf2f(f2bf(pa));
        const float b = mr.trunc ? bf_trunc(pb) : bf2f(f2bf(pb));
        yb[i] = f2bf(first ? a - b : a + b);
      }
    }
  }
  if (act) *reinterpret_cast<v8u16*>(op + lane * 8) = yb;
}

}  // namespace

// out[rows of D] = norm(x)[, rope on the first R2 with cos_sin[pos[token]]]. Row r: token r / H, head r % H.
// Strides in elements. pos: int64 [prow, T] (prow 1 or 3, row stride pstride, token stride ptok -- serving hands the
// indexer a [T] column view of a [T, 3] buffer, ptok 3); sec_* / interleaved are the MRoPE section rule for prow == 3.
// cs / pos may be 0 when R2 == 0 (norm only). Returns 0 or a negative shape code.
extern "C" int r9k_gemma_norm_rope(long x, long sx, long sh, long w, long out, long so, long soh, int rows, int H,
                                   int D, float eps, long cs, long pos, int R2, int prow, long pstride, long ptok,
                                   int sec_t, int sec_h, int sec_w, int interleaved, long stream) {
  if (rows < 0 || H < 1 || D < 8 || D > 256 || D % 8) return -1;
  if (R2 < 0 || R2 > D || R2 % 16) return -2;
  if (R2 > 0 && (cs == 0 || pos == 0)) return -3;
  if ((sx | sh | so | soh) & 7) return -4;
  if (prow != 1 && prow != 3) return -5;
  if (R2 > 0 && ptok < 0) return -5;
  if (rows == 0) return 0;
  hipStream_t st = (hipStream_t)stream;
  const dim3 grid((rows + WAVES - 1) / WAVES), block(NT);
  const MRope mr{prow, pstride, ptok, sec_t, sec_h, sec_w, interleaved, prow == 3 ? 1 : 0};
  if (R2 > 0)
    hipLaunchKernelGGL(r9k_gemma_norm_rope_kernel<true>, grid, block, 0, st, (const uint16_t*)x, sx, sh,
                       (const uint16_t*)w, (uint16_t*)out, so, soh, rows, H, D, eps, (const uint16_t*)cs,
                       (const long*)pos, R2, mr);
  else
    hipLaunchKernelGGL(r9k_gemma_norm_rope_kernel<false>, grid, block, 0, st, (const uint16_t*)x, sx, sh,
                       (const uint16_t*)w, (uint16_t*)out, so, soh, rows, H, D, eps, (const uint16_t*)0,
                       (const long*)0, 0, mr);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}
