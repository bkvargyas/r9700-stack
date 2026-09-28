// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas

// Hyper-connection glue for gfx1201 (Qwen4Exp / Flash-Next): the two per-token elementwise kernels around every
// block, written for DRAM bandwidth. Independent implementations of the semantics of vLLM's Triton
// _hc_gate_mix_kernel / _hc_combine_norm_kernel (Apache-2.0); bf16 in/out, fp32 math, same rounding points.
//
// gate_mix:     y[n, i] = (1/HC) * sum_s sigmoid(g[n, s*HD + i]) * x[n, s*HD + i]            (i < HD = DIM / HC)
// combine_norm: per (n, s): inj = 2 * sigmoid(l[n, s] / HC)
//               out[n, s*HD + i] = bf16(res + block[n, i] * inj)                             (materialized, bf16)
//               rrms = rsqrt(mean_i(out^2) + eps);  y = out * rrms;  y += y * w[(s*HD +) i]  -> bf16
// One 128-thread workgroup per (row, stream) for combine_norm (the combine result kept in registers as bf16 between
// the two passes); one wave per (row, 256-column chunk) for gate_mix. HD % 8 == 0, HD <= 4096, HC <= 8. No device
// printf/assert.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

namespace {

typedef uint16_t v8u16 __attribute__((ext_vector_type(8)));
constexpr int WAVES = 4, NT = 32 * WAVES;
// chunks of 8 per lane held between the two passes of combine_norm, as packed bf16 (the combine result is bf16 by
// definition): 4 VGPRs per chunk. Two instantiations keep the register footprint (and so occupancy) tight for the
// served HD=2560 (10 chunks) while allowing HD <= 4096.
constexpr int MAXHD = 4096;
constexpr int MAXHC = 8;

__device__ __forceinline__ float bf2f(uint16_t u) { return __uint_as_float((unsigned)u << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {    // round to nearest even (matches torch's bf16 conversion)
  unsigned u = __float_as_uint(f);
  if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0xffffu) ? 0x40u : 0u));
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  return v;
}

// One 128-thread workgroup per (row, stream): thread t owns chunks t, t+128, ... of the 2560-element stream, so a
// stream's loads are spread over 4 waves (a single wave per stream was 12.5 us vs Triton's 3.7 in graph replay at
// decode widths: one wave issuing 10 dependent-latency chunks serially). Cross-wave sum of squares through LDS.
template <int MAXCH>
__global__ void __launch_bounds__(NT) r9k_hc_combine_norm_kernel(
    const uint16_t* __restrict__ res, long s_res, const uint16_t* __restrict__ blk, long s_blk,
    const uint16_t* __restrict__ inj, long s_inj, const uint16_t* __restrict__ w, int w_shared,
    uint16_t* __restrict__ out, long s_out, uint16_t* __restrict__ y, long s_y,
    int rows, int HC, int HD, float eps) {
  __shared__ float part[WAVES];
  const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
  const int item = blockIdx.x;
  const int n = item / HC, s = item - n * HC;
  const float g = 2.f * sigmoidf_(bf2f(inj[(long)n * s_inj + s]) / (float)HC);
  const uint16_t* rp = res + (long)n * s_res + (long)s * HD;
  const uint16_t* bp = blk + (long)n * s_blk;
  const uint16_t* wp = w + (w_shared ? 0 : (long)s * HD);
  uint16_t* op = out + (long)n * s_out + (long)s * HD;
  uint16_t* yp = y + (long)n * s_y + (long)s * HD;
  const int nch = HD / 8;                              // 16 B chunks per row segment
  v8u16 rv[MAXCH], bv[MAXCH];
#pragma unroll
  for (int c = 0; c < MAXCH; c++) {                    // issue every load before consuming any
    const int ch = t + NT * c;
    if (ch < nch) {
      rv[c] = *reinterpret_cast<const v8u16*>(rp + ch * 8);
      bv[c] = *reinterpret_cast<const v8u16*>(bp + ch * 8);
    }
  }
  v8u16 o[MAXCH];
  float ss = 0.f;
#pragma unroll
  for (int c = 0; c < MAXCH; c++) {
    const int ch = t + NT * c;
    if (ch < nch) {
      v8u16 ov;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const float v = fmaf(bf2f(bv[c][i]), g, bf2f(rv[c][i]));
        const uint16_t ob = f2bf(v);
        ov[i] = ob;
        const float of = bf2f(ob);
        ss = fmaf(of, of, ss);
      }
      o[c] = ov;
      *reinterpret_cast<v8u16*>(op + ch * 8) = ov;
    }
  }
  ss = wave_sum(ss);
  if (lane == 0) part[wave] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < WAVES; i++) tot += part[i];
  const float rrms = rsqrtf(tot / (float)HD + eps);
#pragma unroll
  for (int c = 0; c < MAXCH; c++) {
    const int ch = t + NT * c;
    if (ch < nch) {
      const v8u16 wv = *reinterpret_cast<const v8u16*>(wp + ch * 8);
      v8u16 yv;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const float yy = bf2f(o[c][i]) * rrms;
        yv[i] = f2bf(fmaf(yy, bf2f(wv[i]), yy));
      }
      *reinterpret_cast<v8u16*>(yp + ch * 8) = yv;
    }
  }
}

// grid (rows * nchunks + WAVES - 1) / WAVES; a wave handles 256 columns (32 lanes x 8) of one row: reads HC x 16 B
// of x and g per lane, writes 16 B.
__global__ void __launch_bounds__(NT) r9k_hc_gate_mix_kernel(
    const uint16_t* __restrict__ x, long s_x, const uint16_t* __restrict__ gt, long s_g,
    uint16_t* __restrict__ y, long s_y, int rows, int HC, int HD) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int nchunks = (HD + 255) / 256;
  const int item = blockIdx.x * WAVES + wave;
  if (item >= rows * nchunks) return;
  const int n = item / nchunks, c0 = (item - n * nchunks) * 256 + lane * 8;
  if (c0 >= HD) return;
  const uint16_t* xp = x + (long)n * s_x + c0;
  const uint16_t* gp = gt + (long)n * s_g + c0;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  v8u16 xv[MAXHC], gv[MAXHC];
#pragma unroll
  for (int s = 0; s < MAXHC; s++)                      // all HC streams' loads in flight before the math
    if (s < HC) {
      xv[s] = *reinterpret_cast<const v8u16*>(xp + (long)s * HD);
      gv[s] = *reinterpret_cast<const v8u16*>(gp + (long)s * HD);
    }
#pragma unroll
  for (int s = 0; s < MAXHC; s++)
    if (s < HC) {
#pragma unroll
      for (int i = 0; i < 8; i++) acc[i] = fmaf(sigmoidf_(bf2f(gv[s][i])), bf2f(xv[s][i]), acc[i]);
    }
  const float inv = 1.f / (float)HC;
  v8u16 yv;
#pragma unroll
  for (int i = 0; i < 8; i++) yv[i] = f2bf(acc[i] * inv);
  *reinterpret_cast<v8u16*>(y + (long)n * s_y + c0) = yv;
}

}  // namespace

// All tensors bf16, last dim contiguous, element strides. HD % 8 == 0, HD <= 4096. Returns 0, <0 unsupported, or hip error.
extern "C" int r9k_hc_combine_norm(long res, long s_res, long blk, long s_blk, long inj, long s_inj, long w, int w_shared,
                                   long out, long s_out, long y, long s_y, int rows, int HC, int HD, float eps,
                                   long stream) {
  if (HD % 8 || HD > MAXHD || HC < 1 || rows < 0) return -1;
  if (rows == 0) return 0;
  dim3 grid(rows * HC), block(NT);
  if (HD <= 3 * 8 * NT)
    r9k_hc_combine_norm_kernel<3><<<grid, block, 0, (hipStream_t)stream>>>(
        (const uint16_t*)res, s_res, (const uint16_t*)blk, s_blk, (const uint16_t*)inj, s_inj, (const uint16_t*)w,
        w_shared, (uint16_t*)out, s_out, (uint16_t*)y, s_y, rows, HC, HD, eps);
  else
    r9k_hc_combine_norm_kernel<4><<<grid, block, 0, (hipStream_t)stream>>>(
        (const uint16_t*)res, s_res, (const uint16_t*)blk, s_blk, (const uint16_t*)inj, s_inj, (const uint16_t*)w,
        w_shared, (uint16_t*)out, s_out, (uint16_t*)y, s_y, rows, HC, HD, eps);
  return (int)hipGetLastError();
}

extern "C" int r9k_hc_gate_mix(long x, long s_x, long g, long s_g, long y, long s_y, int rows, int HC, int HD,
                               long stream) {
  if (HD % 8 || HC < 1 || HC > MAXHC || rows < 0) return -1;
  if (rows == 0) return 0;
  const int nchunks = (HD + 255) / 256;
  dim3 grid((rows * nchunks + WAVES - 1) / WAVES), block(NT);
  r9k_hc_gate_mix_kernel<<<grid, block, 0, (hipStream_t)stream>>>((const uint16_t*)x, s_x, (const uint16_t*)g, s_g,
                                                                  (uint16_t*)y, s_y, rows, HC, HD);
  return (int)hipGetLastError();
}

// ---------------------------------------------------------------- decode mix: up GEMM + sigmoid + gated mean
// block_input[m, d] = bf16( (1/HC) * sum_s sigmoid(bf16(lora[m] . Wup[s*HD + d])) * xn[m, s*HD + d] ), HC == 4.
// Stock runs the up GEMM as a bf16 skinny GEMM (12 us for the 6.5 MB weight at 4 rows), then hc_gate_mix; here the
// weight rows of the four streams of one output dim meet in one wave: lane = khalf + 2 * (s + 4 * dl) covers four
// dims x four streams x two K halves, the K halves reduce with one shuffle, the gate is rounded to bf16 as the GEMM
// output was, and the streams add in stock's order 0..3 before the /HC. lora (M x LR bf16) is staged in LDS once
// per block. M <= 8, LR % 16 == 0, LR <= 512, HD % 4 == 0.
namespace {
constexpr int MIX_MAXM = 8, MIX_MAXLR = 512, MIX_NT = 128, MIX_WAVES = 4;

// LPR lanes per weight row (2 or 4): lane = kpart + LPR * (s + 4 * dl); LPR=4 reads 64 contiguous bytes per row
// per instruction (better coalescing, twice the waves) at the cost of one more shuffle in the K reduction.
template <int MT, int LPR>
__global__ void __launch_bounds__(MIX_NT) r9k_hc_up_mix_kernel(
    const uint16_t* __restrict__ lora, long s_l, const uint16_t* __restrict__ wup, long s_w,
    const uint16_t* __restrict__ xn, long s_x, uint16_t* __restrict__ out, long s_o, int M, int LR, int HD) {
  constexpr int DPW = 32 / LPR / 4;                                   // output dims per wave
  __shared__ __attribute__((aligned(16))) uint16_t sl[MIX_MAXM][MIX_MAXLR];
  const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
  const int nchl = LR / 8;
  for (int c = t; c < M * nchl; c += MIX_NT) {
    const int m = c / nchl, cc = c - m * nchl;
    *reinterpret_cast<v8u16*>(&sl[m][cc * 8]) = *reinterpret_cast<const v8u16*>(lora + (long)m * s_l + cc * 8);
  }
  __syncthreads();
  const int kpart = lane % LPR, s = (lane / LPR) & 3, dl = lane / (LPR * 4);
  const int d0 = (blockIdx.x * MIX_WAVES + wave) * DPW;
  const int d = min(d0 + dl, HD - 1);
  const int n = s * HD + d;
  const int LP = LR / LPR, nch = LP / 8;
  const uint16_t* wp = wup + (long)n * s_w + kpart * LP;
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; m++) acc[m] = 0.f;
  v8u16 wv[MIX_MAXLR / 16];
#pragma unroll
  for (int c = 0; c < MIX_MAXLR / 16; c++)
    if (c < nch) wv[c] = *reinterpret_cast<const v8u16*>(wp + c * 8);
#pragma unroll
  for (int c = 0; c < MIX_MAXLR / 16; c++) {
    if (c < nch) {
#pragma unroll
      for (int m = 0; m < MT; m++) {
        const v8u16 xv = *reinterpret_cast<const v8u16*>(&sl[m < M ? m : 0][kpart * LP + c * 8]);
#pragma unroll
        for (int i = 0; i < 8; i++) acc[m] = fmaf(bf2f(xv[i]), bf2f(wv[c][i]), acc[m]);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < MT; m++) {
    acc[m] += __shfl_xor(acc[m], 1, 32);                          // K parts
    if constexpr (LPR == 4) acc[m] += __shfl_xor(acc[m], 2, 32);
    const float g = bf2f(f2bf(acc[m]));                            // the GEMM's bf16 output
    const float term = sigmoidf_(g) * bf2f(xn[(long)(m < M ? m : 0) * s_x + n]);
    const float t1 = __shfl(term, lane + LPR, 32), t2 = __shfl(term, lane + 2 * LPR, 32),
                t3 = __shfl(term, lane + 3 * LPR, 32);
    const float sum = ((term + t1) + t2) + t3;                    // streams 0..3 in stock's order
    if (kpart == 0 && s == 0 && d0 + dl < HD && m < M) out[(long)m * s_o + d] = f2bf(sum / 4.f);
  }
}
}  // namespace

extern "C" int r9k_hc_up_mix(long lora, long s_l, long wup, long s_w, long xn, long s_x, long out, long s_o, int M,
                             int LR, int HC, int HD, int lpr, long stream) {
  if (HC != 4 || M < 1 || M > MIX_MAXM || LR < 16 || LR > MIX_MAXLR || LR % 16 || HD < 2 || HD % 2) return -1;
  if ((s_l | s_w | s_x | s_o) & 7) return -2;
  if (lpr != 2 && lpr != 4) return -3;
  if (LR % (8 * lpr)) return -4;
  const int dpw = 32 / lpr / 4;
  dim3 grid((HD / dpw + MIX_WAVES - 1) / MIX_WAVES), block(MIX_NT);
  hipStream_t st = (hipStream_t)stream;
#define MIXL(MTv, L)                                                                                              \
  r9k_hc_up_mix_kernel<MTv, L><<<grid, block, 0, st>>>((const uint16_t*)lora, s_l, (const uint16_t*)wup, s_w,   \
                                                       (const uint16_t*)xn, s_x, (uint16_t*)out, s_o, M, LR, HD)
#define MIXM(L) do { if (M <= 1) MIXL(1, L); else if (M <= 2) MIXL(2, L); else if (M <= 4) MIXL(4, L); else MIXL(8, L); } while (0)
  if (lpr == 2) MIXM(2); else MIXM(4);
#undef MIXM
#undef MIXL
  return (int)hipGetLastError();
}
