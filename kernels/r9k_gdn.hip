// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas

// r9k_gdn.hip -- Gated DeltaNet MTP decode after the causal conv, in one launch per (sequence, value head):
// sigmoid gating + the recurrent delta rule over the sequence's draft tokens + RMSNormGated on the output.
// Further down: the same recurrence for short prefills (r9k_gdn_seq) and the causal conv in front of it
// (r9k_gdn_conv) -- the first token of a short prompt.
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


// One sequence's tokens through the same recurrence, for the batches the decode kernel does not take: SHORT PREFILLS
// (SPEC false) and the spec-decode sequences that share a step with a prefill (SPEC true). Stock runs a prefill
// through fused_post_conv_prep + chunk_gated_delta_rule (six Triton kernels: the chunked WY form, built for long
// sequences) + state gather/scatter + the gated norm -- ~30 launches and ~1.3 ms of Python per layer whatever the
// length, 48 layers on the 27B: the fixed ~60 ms under every first token (2026-10-02 profile). Token by token it is
// one launch; the chunked form only wins once the sequence is long enough to pay for that overhead, so the caller
// sends batches up to R9K_GDN_PREFILL_MAX tokens here and leaves longer ones to stock.
//   SPEC false: the initial state is state[idx[n]] when hasinit[n] (null hasinit = always), else zero; only the
//     final state is stored, back at idx[n]. softplus in stock's prefill form (fused_post_conv_prep).
//   SPEC true: as r9k_gdn_decode_mtp_kernel (initial state idx[n, acc[n] - 1], state stored per token at idx[n, t]).
// qkv rows are in COMPACT order (cu indexes them); b, a, z, out rows are ORIGINAL batch rows: rowmap[compact row]
// (int64, as torch's argsort gives it) or the same row when rowmap is null. Rows zero_from .. rows_total - 1 of out
// are zeroed (padding past the batch). A sequence whose state index is <= 0 gets zero output and stores nothing.
template <bool STATE_F32, bool SPEC>
__global__ void __launch_bounds__(NT) r9k_gdn_seq_kernel(
    const uint16_t* __restrict__ qkv, long sqkv, const uint16_t* __restrict__ bp, long sb,
    const uint16_t* __restrict__ ap, long sa, const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    const float* __restrict__ w, const uint16_t* __restrict__ z, long sz, uint16_t* __restrict__ out, long sout,
    void* __restrict__ state, long sstate, const int* __restrict__ idx, long sidx, const int* __restrict__ cu,
    const int* __restrict__ acc, const uint8_t* __restrict__ hasinit, const long* __restrict__ rowmap,
    int* __restrict__ flags, long sflags, int N, int Hk, int Hv, int zero_from, int rows_total, float scale,
    float eps, int gate_sigmoid) {
  const int n = blockIdx.x / Hv, hv = blockIdx.x - n * Hv;
  const int tid = threadIdx.x, v = tid >> 1, kh = tid & 1, lane = tid & 31, wave = tid >> 5;
  __shared__ float lds_o[VD];
  __shared__ float lds_part[WAVES];
  if (n == 0)
    for (int r = zero_from + (tid >> 7); r < rows_total; r += 2) out[(long)r * sout + hv * VD + (tid & 127)] = 0;
  const int bos = cu[n], T = cu[n + 1] - bos;
  if (T <= 0) return;
  int idx0;
  if constexpr (SPEC) idx0 = idx[(long)n * sidx + max(acc[n] - 1, 0)];
  else idx0 = idx[(long)n * sidx];
  if (!SPEC && flags != nullptr && idx0 > 0 && tid == 0) {   // a prefill leaves no live record (one-page state)
    flags[(long)idx0 * sflags] = 0;
    flags[(long)idx0 * sflags + 1] = 0;
  }
  if (idx0 <= 0) {
    for (int t = 0; t < T; t++) {
      const long ro = rowmap ? rowmap[bos + t] : (long)(bos + t);
      out[ro * sout + hv * VD + (tid & 127)] = 0;
    }
    return;
  }
  const long elem = STATE_F32 ? 4 : 2;
  const long hoff = ((long)hv * VD + v) * KD + kh * KH;
  float h[KH];
  bool fresh = false;
  if constexpr (!SPEC) fresh = hasinit != nullptr && hasinit[n] == 0;
  if (fresh) {
#pragma unroll
    for (int i = 0; i < KH; i++) h[i] = 0.f;
  } else {
    load_state<STATE_F32>((const char*)state + ((long)idx0 * sstate + hoff) * elem, h);
  }
  const int hk = hv / (Hv / Hk);
  const float negA = -expf(A_log[hv]), dtb = dt_bias[hv], wv = w[v];
  const long qoff = (long)hk * KD + kh * KH, koff = (long)(Hk + hk) * KD + kh * KH;
  const long voff = (long)(2 * Hk) * KD + (long)hv * VD + v;
  for (int t = 0; t < T; t++) {
    const long rc = bos + t;
    const long ro = rowmap ? rowmap[rc] : rc;
    const uint16_t* rp = qkv + rc * sqkv;
    float k[KH];
    load64(rp + koff, k);
    float sk = 0.f, sq = 0.f;
#pragma unroll
    for (int i = 0; i < KH; i++) sk += k[i] * k[i];
    {
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
    const float x = bf2f(ap[ro * sa + hv]) + dtb;
    float sp;
    if constexpr (SPEC) sp = x <= 20.f ? logf(1.f + expf(x)) : x;
    else sp = x <= 20.f ? (x > 0.f ? x + logf(1.f + expf(-x)) : logf(1.f + expf(x))) : x;
    const float eg = expf(negA * sp);
    const float beta = sigmoidf_(bf2f(bp[ro * sb + hv]));
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
        v8u16 x8 = qp[c];
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const int j = c * 8 + i;
          h[j] = h[j] + vv * k[j];
          o += h[j] * ((bf2f(x8[i]) * rq) * scale);
        }
      }
    }
    o += __shfl_xor(o, 1, 32);
    if constexpr (SPEC) {
      const int idxt = idx[(long)n * sidx + t];
      if (idxt > 0) store_state<STATE_F32>((char*)state + ((long)idxt * sstate + hoff) * elem, h);
    }
    if (kh == 0) lds_o[v] = bf2f(f2bf(o));
    __syncthreads();
    float p = tid < VD ? lds_o[tid] * lds_o[tid] : 0.f;
    p = wave_sum(p);
    if (lane == 0) lds_part[wave] = p;
    __syncthreads();
    const float var = (lds_part[0] + lds_part[1] + lds_part[2] + lds_part[3]) / (float)VD;
    const float rstd = rsqrtf(var + eps);
    if (kh == 0) {
      const float zf = bf2f(z[ro * sz + hv * VD + v]);
      float y = (lds_o[v] * rstd) * wv;
      const float sg = sigmoidf_(zf);
      y = y * (gate_sigmoid ? sg : zf * sg);
      out[ro * sout + hv * VD + v] = f2bf(y);
    }
    __syncthreads();
  }
  if constexpr (!SPEC) store_state<STATE_F32>((char*)state + ((long)idx0 * sstate + hoff) * elem, h);
}


// The causal conv in front of r9k_gdn_seq for prefill sequences (width 4, depthwise, no bias, SiLU) with the
// conv-state update: stock's causal_conv1d_fn (one Triton launch, ~0.15 ms of launcher per layer on this ROCm) as one
// HIP launch that also reads the batch's own rows, so a step that mixes prefills with spec-decode sequences needs no
// gather. Bit-identical to the Triton kernel, whose arithmetic is: each tap product bf16 x bf16 -> bf16 by
// TRUNCATION (how this backend lowers the fp32 -> bf16 conversion inside the multiply; found by enumeration against
// its output, tests/test_gdn_prefill_r9k.py holds it), summed in fp32 from the oldest tap, acc / (1 + exp(-acc)),
// and the stored result rounded to nearest.
// One thread per channel walks its sequence with the last three inputs in registers: they start from the state
// (zero without an initial state) and are the new state at the end -- which is stock's rule in all three of its
// cases (>= 3 tokens: the last three inputs; fewer: the old state shifted, or zeros, then the inputs).
// A sequence whose state index is <= 0 (null block) gets zero output and stores nothing.
__device__ __forceinline__ float bf_trunc(float f) { return __uint_as_float(__float_as_uint(f) & 0xffff0000u); }

__global__ void __launch_bounds__(256) r9k_gdn_conv_kernel(
    const uint16_t* __restrict__ x, long sx, const uint16_t* __restrict__ wt, long sw, uint16_t* __restrict__ cs,
    long scs_seq, long scs_dim, long scs_tok, uint16_t* __restrict__ out, long so, const int* __restrict__ idx,
    long sidx, const uint8_t* __restrict__ hasinit, const int* __restrict__ cu, const long* __restrict__ rowmap,
    int C) {
  const int n = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
  if (c >= C) return;
  const int bos = cu[n], T = cu[n + 1] - bos;
  if (T <= 0) return;
  const int i0 = idx[(long)n * sidx];
  if (i0 <= 0) {
    for (int t = 0; t < T; t++) out[(long)(bos + t) * so + c] = 0;
    return;
  }
  uint16_t* sp = cs + (long)i0 * scs_seq + (long)c * scs_dim;
  float c0 = 0.f, c1 = 0.f, c2 = 0.f;
  if (hasinit == nullptr || hasinit[n] != 0) {
    c0 = bf2f(sp[0]);
    c1 = bf2f(sp[scs_tok]);
    c2 = bf2f(sp[2 * scs_tok]);
  }
  const uint16_t* wp = wt + (long)c * sw;
  const float w0 = bf2f(wp[0]), w1 = bf2f(wp[1]), w2 = bf2f(wp[2]), w3 = bf2f(wp[3]);
  for (int t = 0; t < T; t++) {
    const long ro = rowmap ? rowmap[bos + t] : (long)(bos + t);
    const float xv = bf2f(x[ro * sx + c]);
    float acc = bf_trunc(c0 * w0);
    acc += bf_trunc(c1 * w1);
    acc += bf_trunc(c2 * w2);
    acc += bf_trunc(xv * w3);
    c0 = c1; c1 = c2; c2 = xv;
    acc = acc / (1.f + expf(-acc));
    out[(long)(bos + t) * so + c] = f2bf(acc);
  }
  sp[0] = f2bf(c0);
  sp[scs_tok] = f2bf(c1);
  sp[2 * scs_tok] = f2bf(c2);
}


// ---------------------------------------------------------------------------------------------------------------
// One state page per request under speculative decoding (2026-10-03, PROGRESS.md "one state page per request").
// Stock vLLM gives every request 1 + num_spec state pages per GDN group and the decode kernel above stores the state
// after EACH candidate token in its own page; the next step starts from the page of the last accepted token. Here
// the page holds the state after the accepted tokens plus a RECORD of the last step's candidate inputs, and the
// verify kernel (r9k_gdn_spec_verify_kernel) does, per (sequence, value head):
//   1. replay: if the page's flag says a record of T_prev candidates is live, re-run the recurrence over the first
//      acc[n] of them (acc = the number the sampler accepted, which vLLM hands to the next step) from the base state
//      and store the result as the new base -- exactly the fp32 recurrence the decode kernel ran before storing the
//      accepted token's page, from the same rounded base on the same bf16 inputs: bit-identical;
//   2. the new 1 + spec candidates from that base, outputs through the gated norm, no state stored;
//   3. the new record (the conv'd q/k/v rows and b/a of every candidate) into the OTHER record slot (two slots:
//      sibling blocks of the same sequence may still be replaying from the live one), and, once every value head's
//      block of the sequence has finished (per-page atomic counter), the flag: T | parity << 8.
// Prefill paths write flag 0 (no record). The commit kernel below is the same replay as a separate launch (kept for
// a post-sampling commit variant; not used by the serving path).
// Record layout per page: qkv rows [2, S, (2 Hk + Hv) x 128] bf16 (the mixed_qkv row layout), ab rows
// [2, S, 2 Hv] bf16 (b = cols [0, Hv), a = cols [Hv, 2 Hv), as the layer's split_ba views them); flags [2] int32 =
// {T | parity << 8, done counter}. b/a/z/out rows go through rowmap (int64 original row of each compact row) or the
// same row.

template <bool STATE_F32>
__device__ __forceinline__ void gdn_replay(const uint16_t* __restrict__ rq, const uint16_t* __restrict__ rab, int T,
                                           int Hk, int Hv, int hv, int kh, long koff, long voff, long rowlen,
                                           float negA, float dtb, float* h) {
  for (int t = 0; t < T; t++) {
    const uint16_t* rp = rq + (long)t * rowlen;
    float k[KH];
    load64(rp + koff, k);
    float sk = 0.f;
#pragma unroll
    for (int i = 0; i < KH; i++) sk += k[i] * k[i];
    sk += __shfl_xor(sk, 1, 32);
    const float rk = rsqrtf(sk + 1e-6f);
#pragma unroll
    for (int i = 0; i < KH; i++) k[i] = k[i] * rk;
    const float x = bf2f(rab[(long)t * 2 * Hv + Hv + hv]) + dtb;
    const float sp = x <= 20.f ? logf(1.f + expf(x)) : x;
    const float eg = expf(negA * sp);
    const float beta = sigmoidf_(bf2f(rab[(long)t * 2 * Hv + hv]));
    float vv = bf2f(rp[voff]);
    float d = 0.f;
#pragma unroll
    for (int i = 0; i < KH; i++) { h[i] = h[i] * eg; d += h[i] * k[i]; }
    d += __shfl_xor(d, 1, 32);
    vv = (vv - d) * beta;
#pragma unroll
    for (int j = 0; j < KH; j++) h[j] = h[j] + vv * k[j];
  }
  (void)kh;
}

template <bool STATE_F32>
__global__ void __launch_bounds__(NT) r9k_gdn_spec_verify_kernel(
    const uint16_t* __restrict__ qkv, long sqkv, const uint16_t* __restrict__ bp, long sb,
    const uint16_t* __restrict__ ap, long sa, const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    const float* __restrict__ w, const uint16_t* __restrict__ z, long sz, uint16_t* __restrict__ out, long sout,
    void* __restrict__ state, long sstate, uint16_t* __restrict__ rec_qkv, long srec_qkv,
    uint16_t* __restrict__ rec_ab, long srec_ab, int* __restrict__ flags, long sflags, int S,
    const int* __restrict__ idx, long sidx, const int* __restrict__ cu, const int* __restrict__ acc,
    const long* __restrict__ rowmap, int N, int Hk, int Hv, int zero_from, int rows_total, float scale, float eps,
    int gate_sigmoid) {
  const int n = blockIdx.x / Hv, hv = blockIdx.x - n * Hv;
  const int tid = threadIdx.x, v = tid >> 1, kh = tid & 1, lane = tid & 31, wave = tid >> 5;
  __shared__ float lds_o[VD];
  __shared__ float lds_part[WAVES];
  if (n == 0) {                                      // zero_from < 0: from the batch's last token (cu[N], device)
    const int zf = zero_from < 0 ? cu[N] : zero_from;
    for (int r = zf + (tid >> 7); r < rows_total; r += 2) out[(long)r * sout + hv * VD + (tid & 127)] = 0;
  }
  const int bos = cu[n], T = cu[n + 1] - bos;
  if (T <= 0) return;
  const int idx0 = idx[(long)n * sidx];
  if (idx0 <= 0) {
    for (int t = 0; t < T; t++) {
      const long ro = rowmap ? rowmap[bos + t] : (long)(bos + t);
      out[ro * sout + hv * VD + (tid & 127)] = 0;
    }
    return;
  }
  const long elem = STATE_F32 ? 4 : 2;
  const long hoff = ((long)hv * VD + v) * KD + kh * KH;
  float h[KH];
  load_state<STATE_F32>((const char*)state + ((long)idx0 * sstate + hoff) * elem, h);
  const int gq = Hv / Hk, hk = hv / gq;
  const bool owner = (hv % gq) == 0;                 // one value head per key head copies that head's q/k rows
  const float negA = -expf(A_log[hv]), dtb = dt_bias[hv], wv = w[v];
  const long qoff = (long)hk * KD + kh * KH, koff = (long)(Hk + hk) * KD + kh * KH;
  const long voff = (long)(2 * Hk) * KD + (long)hv * VD + v;
  const long rowlen = (long)(2 * Hk + Hv) * KD;
  int* fl = flags + (long)idx0 * sflags;
  const int fv = fl[0];                              // read before this block's done-count below: race-free
  const int live = fv & 0xff, par = (fv >> 8) & 1;
  uint16_t* rq = rec_qkv + (long)idx0 * srec_qkv;
  uint16_t* rab = rec_ab + (long)idx0 * srec_ab;
  if (live > 0) {
    const int nr = min(acc[n], live);
    gdn_replay<STATE_F32>(rq + (long)par * S * rowlen, rab + (long)par * S * 2 * Hv, nr, Hk, Hv, hv, kh, koff,
                          voff, rowlen, negA, dtb, h);
    store_state<STATE_F32>((char*)state + ((long)idx0 * sstate + hoff) * elem, h);
    if constexpr (!STATE_F32) {                      // continue from what the page holds, as a fresh step would
#pragma unroll
      for (int i = 0; i < KH; i++) h[i] = bf2f(f2bf(h[i]));
    }
  }
  uint16_t* rqn = rq + (long)(par ^ 1) * S * rowlen;
  uint16_t* rabn = rab + (long)(par ^ 1) * S * 2 * Hv;
  const int Tn = min(T, S);
  for (int t = 0; t < Tn; t++) {
    const long rc = bos + t;
    const long ro = rowmap ? rowmap[rc] : rc;
    const uint16_t* rp = qkv + rc * sqkv;
    uint16_t* rr = rqn + (long)t * rowlen;
    if (kh == 0) rr[voff] = rp[voff];
    if (owner) {                                     // threads 0..127: this key head's q channels; 128..255: its k
      const long c = tid < KD ? (long)hk * KD + tid : (long)(Hk + hk) * KD + (tid - KD);
      rr[c] = rp[c];
    }
    if (tid == 0) {
      rabn[(long)t * 2 * Hv + hv] = bp[ro * sb + hv];
      rabn[(long)t * 2 * Hv + Hv + hv] = ap[ro * sa + hv];
    }
    float k[KH];
    load64(rp + koff, k);
    float sk = 0.f, sq = 0.f;
#pragma unroll
    for (int i = 0; i < KH; i++) sk += k[i] * k[i];
    {
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
    const float rk = rsqrtf(sk + 1e-6f), rq_ = rsqrtf(sq + 1e-6f);
#pragma unroll
    for (int i = 0; i < KH; i++) k[i] = k[i] * rk;
    const float x = bf2f(ap[ro * sa + hv]) + dtb;
    const float sp = x <= 20.f ? logf(1.f + expf(x)) : x;
    const float eg = expf(negA * sp);
    const float beta = sigmoidf_(bf2f(bp[ro * sb + hv]));
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
        v8u16 x8 = qp[c];
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const int j = c * 8 + i;
          h[j] = h[j] + vv * k[j];
          o += h[j] * ((bf2f(x8[i]) * rq_) * scale);
        }
      }
    }
    o += __shfl_xor(o, 1, 32);
    if (kh == 0) lds_o[v] = bf2f(f2bf(o));
    __syncthreads();
    float p = tid < VD ? lds_o[tid] * lds_o[tid] : 0.f;
    p = wave_sum(p);
    if (lane == 0) lds_part[wave] = p;
    __syncthreads();
    const float var = (lds_part[0] + lds_part[1] + lds_part[2] + lds_part[3]) / (float)VD;
    const float rstd = rsqrtf(var + eps);
    if (kh == 0) {
      const float zf = bf2f(z[ro * sz + hv * VD + v]);
      float y = (lds_o[v] * rstd) * wv;
      const float sg = sigmoidf_(zf);
      y = y * (gate_sigmoid ? sg : zf * sg);
      out[ro * sout + hv * VD + v] = f2bf(y);
    }
    __syncthreads();
  }
  // the flag: written by the last of this sequence's Hv blocks to finish, after every block has read the old one
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(fl + 1, 1) == Hv - 1) {
      fl[1] = 0;
      fl[0] = Tn | ((par ^ 1) << 8);
      __threadfence();
    }
  }
}

// Replay ns[n] recorded tokens on the base state of page idx[n] and store it; shift the conv window (C channels,
// L columns, the request's page at conv + idx * sconv) left by ns[n] - 1 when conv_shift != 0. One launch covers
// `layers` layers of one KV-cache group (same page index per request): per-layer pointers in the tables.
template <bool STATE_F32>
__global__ void __launch_bounds__(NT) r9k_gdn_spec_commit_kernel(
    const long* __restrict__ tab,      // per layer: state, rec_qkv, rec_ab, conv, A_log, dt_bias (6 entries)
    long sstate, long srec_qkv, long srec_ab, long sconv, long sconv_c, long sconv_l, int L, int C, int conv_shift,
    const int* __restrict__ idx, long sidx, const int* __restrict__ ns, int layers, int N, int Hk, int Hv) {
  const int layer = blockIdx.y;
  const int n = blockIdx.x / Hv, hv = blockIdx.x - n * Hv;
  const int tid = threadIdx.x, v = tid >> 1, kh = tid & 1;
  const int idx0 = idx[(long)n * sidx];
  const int T = ns[n];
  if (idx0 <= 0 || T <= 0 || layer >= layers) return;
  const long* e = tab + (long)layer * 6;
  void* state = (void*)e[0];
  const uint16_t* rq = (const uint16_t*)e[1] + (long)idx0 * srec_qkv;
  const uint16_t* rab = (const uint16_t*)e[2] + (long)idx0 * srec_ab;
  const float* A_log = (const float*)e[4];
  const float* dt_bias = (const float*)e[5];
  const long elem = STATE_F32 ? 4 : 2;
  const long hoff = ((long)hv * VD + v) * KD + kh * KH;
  float h[KH];
  load_state<STATE_F32>((const char*)state + ((long)idx0 * sstate + hoff) * elem, h);
  const int gq = Hv / Hk, hk = hv / gq;
  const float negA = -expf(A_log[hv]), dtb = dt_bias[hv];
  const long koff = (long)(Hk + hk) * KD + kh * KH;
  const long voff = (long)(2 * Hk) * KD + (long)hv * VD + v;
  const long rowlen = (long)(2 * Hk + Hv) * KD;
  for (int t = 0; t < T; t++) {
    const uint16_t* rp = rq + (long)t * rowlen;
    float k[KH];
    load64(rp + koff, k);
    float sk = 0.f;
#pragma unroll
    for (int i = 0; i < KH; i++) sk += k[i] * k[i];
    sk += __shfl_xor(sk, 1, 32);
    const float rk = rsqrtf(sk + 1e-6f);
#pragma unroll
    for (int i = 0; i < KH; i++) k[i] = k[i] * rk;
    const float x = bf2f(rab[(long)t * 2 * Hv + Hv + hv]) + dtb;
    const float sp = x <= 20.f ? logf(1.f + expf(x)) : x;
    const float eg = expf(negA * sp);
    const float beta = sigmoidf_(bf2f(rab[(long)t * 2 * Hv + hv]));
    float vv = bf2f(rp[voff]);
    float d = 0.f;
#pragma unroll
    for (int i = 0; i < KH; i++) { h[i] = h[i] * eg; d += h[i] * k[i]; }
    d += __shfl_xor(d, 1, 32);
    vv = (vv - d) * beta;
#pragma unroll
    for (int j = 0; j < KH; j++) h[j] = h[j] + vv * k[j];
  }
  store_state<STATE_F32>((char*)state + ((long)idx0 * sstate + hoff) * elem, h);
  if (conv_shift && T > 1) {
    // this block's channels: its v head (128), and (owner) its key head's q and k channels (2 x 128)
    uint16_t* cw = (uint16_t*)e[3] + (long)idx0 * sconv;
    const int sh = T - 1;
    const bool owner = (hv % gq) == 0;
    auto shift = [&](long c) {
      uint16_t* row = cw + c * sconv_c;
      for (int j = 0; j + sh < L; j++) row[(long)j * sconv_l] = row[(long)(j + sh) * sconv_l];
    };
    if (tid < VD) shift((long)(2 * Hk) * KD + (long)hv * VD + tid);
    if (owner) {
      if (tid < VD) shift((long)hk * KD + tid);
      else shift((long)(Hk + hk) * KD + (tid - VD));
    }
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

// The general form (r9k_gdn_seq_kernel above). spec != 0: idx [N, S] (row stride sidx), acc [N]; spec == 0: idx [N]
// (element stride sidx), hasinit [N] uint8 or 0. rowmap [cu[N]] int64 or 0. flags [pages, 2] int32 (stride sflags)
// or 0: prefill sequences (spec == 0) get their page's flag zeroed (no live one-page record). Rows zero_from ..
// rows_total - 1 of out are zeroed (pass zero_from >= rows_total for none). Strides in elements. Returns 0 or a
// negative code.
extern "C" int r9k_gdn_seq(long qkv, long sqkv, long b, long sb, long a, long sa, long A_log, long dt_bias, long w,
                           long z, long sz, long out, long sout, long state, long sstate, int state_f32, long idx,
                           long sidx, long cu, long acc, long hasinit, long rowmap, long flags, long sflags, int N,
                           int Hk, int Hv, int zero_from, int rows_total, float scale, float eps, int gate_sigmoid,
                           int spec, long stream) {
  if (Hk < 1 || Hv < Hk || Hv % Hk || N < 0 || rows_total < 0 || zero_from < 0 || sidx < 1) return -1;
  if ((sqkv | sz | sout | sstate) & 7) return -2;
  if (!qkv || !b || !a || !A_log || !dt_bias || !w || !z || !out || !state || !idx || !cu || (spec && !acc)) return -3;
  if (N == 0) return 0;
  hipStream_t st = (hipStream_t)stream;
  const dim3 grid(N * Hv), block(NT);
#define R9K_GDN_SEQ_ARGS                                                                                             \
  (const uint16_t*)qkv, sqkv, (const uint16_t*)b, sb, (const uint16_t*)a, sa, (const float*)A_log,                    \
      (const float*)dt_bias, (const float*)w, (const uint16_t*)z, sz, (uint16_t*)out, sout, (void*)state, sstate,     \
      (const int*)idx, sidx, (const int*)cu, (const int*)acc, (const uint8_t*)hasinit, (const long*)rowmap,            \
      (int*)flags, sflags, N, Hk, Hv, zero_from, rows_total, scale, eps, gate_sigmoid
  if (state_f32 && spec) hipLaunchKernelGGL((r9k_gdn_seq_kernel<true, true>), grid, block, 0, st, R9K_GDN_SEQ_ARGS);
  else if (state_f32) hipLaunchKernelGGL((r9k_gdn_seq_kernel<true, false>), grid, block, 0, st, R9K_GDN_SEQ_ARGS);
  else if (spec) hipLaunchKernelGGL((r9k_gdn_seq_kernel<false, true>), grid, block, 0, st, R9K_GDN_SEQ_ARGS);
  else hipLaunchKernelGGL((r9k_gdn_seq_kernel<false, false>), grid, block, 0, st, R9K_GDN_SEQ_ARGS);
#undef R9K_GDN_SEQ_ARGS
  return hipGetLastError() == hipSuccess ? 0 : -4;
}

// x [rows, >= C] bf16 (row stride sx; the first C columns are the conv input), w [C, 4] bf16 (row stride sw),
// conv state [lines, C, >= 3] bf16 addressed by the three strides (either layout; columns 0-2 are used),
// out [cu[N], C] bf16 (row stride so),
// idx [N] int32 (element stride sidx), hasinit [N] uint8 or 0, cu [N + 1] int32, rowmap [cu[N]] int64 or 0 (the x row
// of each out row). Strides in elements. Returns 0 or a negative code.
extern "C" int r9k_gdn_conv(long x, long sx, long w, long sw, long cs, long scs_seq, long scs_dim, long scs_tok,
                            long out, long so, long idx, long sidx, long hasinit, long cu, long rowmap, int N, int C,
                            long stream) {
  if (N < 0 || C < 1 || sidx < 1 || sw < 4) return -1;
  if (!x || !w || !cs || !out || !idx || !cu) return -3;
  if (N == 0) return 0;
  const dim3 grid((C + 255) / 256, N), block(256);
  hipLaunchKernelGGL(r9k_gdn_conv_kernel, grid, block, 0, (hipStream_t)stream, (const uint16_t*)x, sx,
                     (const uint16_t*)w, sw, (uint16_t*)cs, scs_seq, scs_dim, scs_tok, (uint16_t*)out, so,
                     (const int*)idx, sidx, (const uint8_t*)hasinit, (const int*)cu, (const long*)rowmap, C);
  return hipGetLastError() == hipSuccess ? 0 : -4;
}

// Verify step on one page per request: qkv rows COMPACT (cu), b/a/z/out via rowmap or same row; state page idx[n]
// (element stride sidx) replayed from the live record by acc[n] tokens and stored, then read for the candidates;
// records at page idx[n] of rec_qkv [pages, 2, S, rowlen] (page stride srec_qkv), rec_ab [pages, 2, S, 2 Hv]
// (srec_ab), flags [pages, 2] int32 (sflags). Rows zero_from.. of out zeroed.
extern "C" int r9k_gdn_spec_verify(long qkv, long sqkv, long b, long sb, long a, long sa, long A_log, long dt_bias,
                                   long w, long z, long sz, long out, long sout, long state, long sstate,
                                   int state_f32, long rec_qkv, long srec_qkv, long rec_ab, long srec_ab, long flags,
                                   long sflags, int S, long idx, long sidx, long cu, long acc, long rowmap, int N,
                                   int Hk, int Hv, int zero_from, int rows_total, float scale, float eps,
                                   int gate_sigmoid, long stream) {
  if (Hk < 1 || Hv < Hk || Hv % Hk || S < 1 || N < 0 || rows_total < 0 || sidx < 1 || sflags < 2) return -1;
  if ((sqkv | sz | sout | sstate) & 7) return -2;
  if (!qkv || !b || !a || !A_log || !dt_bias || !w || !z || !out || !state || !rec_qkv || !rec_ab || !flags || !idx
      || !cu || !acc)
    return -3;
  if (N == 0) return 0;
  hipStream_t st = (hipStream_t)stream;
  const dim3 grid(N * Hv), block(NT);
#define R9K_GDN_VERIFY_ARGS                                                                                          \
  (const uint16_t*)qkv, sqkv, (const uint16_t*)b, sb, (const uint16_t*)a, sa, (const float*)A_log,                    \
      (const float*)dt_bias, (const float*)w, (const uint16_t*)z, sz, (uint16_t*)out, sout, (void*)state,             \
      sstate, (uint16_t*)rec_qkv, srec_qkv, (uint16_t*)rec_ab, srec_ab, (int*)flags, sflags, S, (const int*)idx,  \
      sidx, (const int*)cu, (const int*)acc, (const long*)rowmap, N, Hk, Hv, zero_from, rows_total, scale, eps,      \
      gate_sigmoid
  if (state_f32) hipLaunchKernelGGL((r9k_gdn_spec_verify_kernel<true>), grid, block, 0, st, R9K_GDN_VERIFY_ARGS);
  else hipLaunchKernelGGL((r9k_gdn_spec_verify_kernel<false>), grid, block, 0, st, R9K_GDN_VERIFY_ARGS);
#undef R9K_GDN_VERIFY_ARGS
  return hipGetLastError() == hipSuccess ? 0 : -4;
}

// Commit: tab = device array of 6 longs per layer (state, rec_qkv, rec_ab, conv, A_log, dt_bias pointers); strides
// in elements (sstate / srec_* / sconv = per page; sconv_c / sconv_l = conv window channel / column strides, L
// columns, C channels); ns [N] int32 = tokens to replay per request; conv_shift != 0 shifts the window by ns - 1.
extern "C" int r9k_gdn_spec_commit(long tab, long sstate, int state_f32, long srec_qkv, long srec_ab, long sconv,
                                   long sconv_c, long sconv_l, int L, int C, int conv_shift, long idx, long sidx,
                                   long ns, int layers, int N, int Hk, int Hv, long stream) {
  if (Hk < 1 || Hv < Hk || Hv % Hk || N < 0 || layers < 1 || sidx < 1) return -1;
  if (!tab || !idx || !ns) return -3;
  if (N == 0) return 0;
  hipStream_t st = (hipStream_t)stream;
  const dim3 grid(N * Hv, layers), block(NT);
  if (state_f32)
    hipLaunchKernelGGL((r9k_gdn_spec_commit_kernel<true>), grid, block, 0, st, (const long*)tab, sstate, srec_qkv,
                       srec_ab, sconv, sconv_c, sconv_l, L, C, conv_shift, (const int*)idx, sidx, (const int*)ns,
                       layers, N, Hk, Hv);
  else
    hipLaunchKernelGGL((r9k_gdn_spec_commit_kernel<false>), grid, block, 0, st, (const long*)tab, sstate, srec_qkv,
                       srec_ab, sconv, sconv_c, sconv_l, L, C, conv_shift, (const int*)idx, sidx, (const int*)ns,
                       layers, N, Hk, Hv);
  return hipGetLastError() == hipSuccess ? 0 : -4;
}
