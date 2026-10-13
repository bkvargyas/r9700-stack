// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas
// r9k_sparse_mla.hip -- sparse (top-k gathered), rope-free MLA attention for gfx1201, bf16 or fp8 e4m3 KV pool.
//
// GLM-5.3-Flash's 11 sparse layers: every query token attends over the <= 2048 latent rows its indexer selected.
// q[t, h, :] . K[slot, :] over D = kv_lora_rank (512), softmax, and V is the SAME latent row (weight absorption
// happens outside: the caller passes the absorbed q and un-absorbs the output). Stock vLLM's ROCm path for an fp8
// pool is CDNA assembly; its Triton path is bf16-only; aiter's Triton kernel asserts gfx950. This is ours.
//
// Interface (one call serves prefill and decode): q [T, H, D] bf16; pool [slots, D] bf16 (uint16) or fp8 e4m3
// (uint8) with one scalar descale; indices: int32 global slot ids, ragged per token through indptr [T + 1], -1 =
// masked; out [T, H, D] bf16. Block = 256 threads = 8 waves of 32: wave = head (<= 8 heads per block, grid.z over
// head groups), lane = row of a 32-row K tile. The tile is staged in LDS as bf16 (fp8 converted on the way in,
// v_cvt_pk_f32_fp8 is exact) with a per-row 16-byte-chunk swizzle so the 32 lanes read 32 rows without bank
// conflicts; q[h] sits in LDS too and is read as a wave broadcast. Scores: v_dot2_f32_bf16 over the 512 pairs.
// Softmax: online, per head, running max in log2 units (same scheme as r9k_attn.hip). PV: lane owns dims
// [16 lane, 16 lane + 16), p_r broadcast by shuffle, 16 fma per row. The fp8 descale folds into the softmax scale
// for K and into the output normaliser for V (same rows, same scalar).
// Split-K for decode (one token, one CU otherwise): grid.y = nsplit row ranges write (m, l, acc[512]) partials to
// scratch [T, nsplit, HB, 2 + D] fp32 and r9k_sparse_mla_combine merges them. nsplit = 1 writes out directly.
// No device printf/assert (the build enforces hostcall-free objects).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cmath>

namespace {
typedef __bf16 v8bf16 __attribute__((ext_vector_type(8)));
typedef __bf16 v2bf16 __attribute__((ext_vector_type(2)));
typedef float v2f __attribute__((ext_vector_type(2)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef unsigned v4u __attribute__((ext_vector_type(4)));
typedef float v8f __attribute__((ext_vector_type(8)));

constexpr int D = 512;              // kv_lora_rank (rope-free: the whole QK width)
constexpr int HB = 8;               // heads per block = waves
constexpr int NT = 256;             // threads
constexpr int TR = 32;              // K rows per tile = lanes
constexpr int ROWB = D * 2;         // bytes per bf16 row in LDS (1024)
constexpr int CHUNKS = ROWB / 16;   // 16-byte chunks per row (64)
constexpr int LDS_Q = HB * ROWB;    // 8 KB
constexpr int LDS_K = TR * ROWB;    // 32 KB
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ int koff(int row, int chunk) { return row * ROWB + (((chunk + row) & (CHUNKS - 1)) << 4); }

__device__ __forceinline__ unsigned pack_hi16(float lo, float hi) {
  return (__float_as_uint(lo) >> 16) | (__float_as_uint(hi) & 0xFFFF0000u);
}
// fp8 e4m3 x8 -> bf16 x8 (exact: 3 mantissa bits fit)
__device__ __forceinline__ v8bf16 fp8x8_to_bf16(v2i w) {
  const v2f a = __builtin_amdgcn_cvt_pk_f32_fp8(w[0], false), b = __builtin_amdgcn_cvt_pk_f32_fp8(w[0], true);
  const v2f c = __builtin_amdgcn_cvt_pk_f32_fp8(w[1], false), d = __builtin_amdgcn_cvt_pk_f32_fp8(w[1], true);
  const v4u r = {pack_hi16(a[0], a[1]), pack_hi16(b[0], b[1]), pack_hi16(c[0], c[1]), pack_hi16(d[0], d[1])};
  return __builtin_bit_cast(v8bf16, r);
}
__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
  return v;
}
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  return v;
}

// Stage one 32-row tile: row r <- pool[slot[r]] (or zeros for slot < 0 / beyond the range). 256 threads, 16 B
// (bf16) or 8 B->16 B (fp8) per thread per pass.
template <typename KT>
__device__ __forceinline__ void stage_tile(unsigned char* lds, const KT* __restrict__ pool, long pool_stride_elems,
                                           const int* __restrict__ slots, int nrows, int tid) {
  if constexpr (sizeof(KT) == 2) {
    // 32 rows x 64 chunks = 2048 chunks of 16 B; 8 per thread
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int c = tid + i * NT, row = c >> 6, chunk = c & (CHUNKS - 1);
      const int slot = row < nrows ? slots[row] : -1;
      v4u v = {0u, 0u, 0u, 0u};
      if (slot >= 0) v = *reinterpret_cast<const v4u*>(pool + (long)slot * pool_stride_elems + chunk * 8);
      *reinterpret_cast<v4u*>(lds + koff(row, chunk)) = v;
    }
  } else {
    // 32 rows x 32 chunks of 16 fp8 (= 2 bf16 chunks) = 1024 units; 4 per thread
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int c = tid + i * NT, row = c >> 5, u = c & 31;      // u: 16-element unit -> bf16 chunks 2u, 2u+1
      const int slot = row < nrows ? slots[row] : -1;
      v4u raw = {0u, 0u, 0u, 0u};
      if (slot >= 0) raw = *reinterpret_cast<const v4u*>(pool + (long)slot * pool_stride_elems + u * 16);
      const v2i lo = {(int)raw[0], (int)raw[1]}, hi = {(int)raw[2], (int)raw[3]};
      *reinterpret_cast<v8bf16*>(lds + koff(row, 2 * u)) = fp8x8_to_bf16(lo);
      *reinterpret_cast<v8bf16*>(lds + koff(row, 2 * u + 1)) = fp8x8_to_bf16(hi);
    }
  }
}

// One block: token t, head group g (heads g*8 .. g*8+7), row range [split * chunk, ...) of the token's list.
template <typename KT, bool PARTIAL>
__global__ void __launch_bounds__(NT) sparse_mla_kernel(
    const uint16_t* __restrict__ q, const KT* __restrict__ pool, const int* __restrict__ indices,
    const int* __restrict__ indptr, uint16_t* __restrict__ out, float* __restrict__ scratch, int nheads,
    long q_stride_t, long q_stride_h, long pool_stride, long out_stride_t, long out_stride_h, float scale_log2,
    float out_mult, int nsplit) {
  extern __shared__ __align__(16) unsigned char lds[];
  const int t = blockIdx.x, split = blockIdx.y, g = blockIdx.z;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
  const int h = g * HB + wave;
  const bool hvalid = h < nheads;
  const int beg = indptr[t], end = indptr[t + 1];
  const int n = end - beg;
  const int chunk = (n + nsplit - 1) / nsplit;
  const int r0 = split * chunk, r1 = min(r0 + chunk, n);
  const int* slots = indices + beg;

  // q for this block's heads -> LDS (zeros for heads past nheads)
  {
    const int hh = tid >> 5, c = tid & 31;             // 8 heads x 32 threads x 2 chunks
    const int hq = g * HB + hh;
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const int ch = c * 2 + j;
      v4u v = {0u, 0u, 0u, 0u};
      if (hq < nheads) v = *reinterpret_cast<const v4u*>(q + (long)t * q_stride_t + (long)hq * q_stride_h + ch * 8);
      *reinterpret_cast<v4u*>(lds + hh * ROWB + ch * 16) = v;
    }
  }
  unsigned char* kt = lds + LDS_Q;
  const uint16_t* qrow = reinterpret_cast<const uint16_t*>(lds + wave * ROWB);

  float m = -INFINITY, l = 0.f;
  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; i++) acc[i] = 0.f;

  for (int base = r0; base < r1; base += TR) {
    const int nrows = min(TR, r1 - base);
    __syncthreads();                                   // previous tile fully consumed
    stage_tile<KT>(kt, pool, pool_stride, slots + base, nrows, tid);
    __syncthreads();
    // scores: lane = row, wave = head
    float s = -INFINITY;
    const bool rvalid = lane < nrows && slots[base + lane] >= 0;
    if (rvalid) {
      float d0 = 0.f, d1 = 0.f;
      const unsigned char* kr = kt + lane * ROWB;
#pragma unroll 8
      for (int c = 0; c < CHUNKS; c++) {
        const v8bf16 kv = *reinterpret_cast<const v8bf16*>(kr + (((c + lane) & (CHUNKS - 1)) << 4));
        const v8bf16 qv = *reinterpret_cast<const v8bf16*>(qrow + c * 8);
        const v2bf16 k0 = {kv[0], kv[1]}, k1 = {kv[2], kv[3]}, k2 = {kv[4], kv[5]}, k3 = {kv[6], kv[7]};
        const v2bf16 q0 = {qv[0], qv[1]}, q1 = {qv[2], qv[3]}, q2 = {qv[4], qv[5]}, q3 = {qv[6], qv[7]};
        d0 = __builtin_amdgcn_fdot2_f32_bf16(k0, q0, d0, false);
        d1 = __builtin_amdgcn_fdot2_f32_bf16(k1, q1, d1, false);
        d0 = __builtin_amdgcn_fdot2_f32_bf16(k2, q2, d0, false);
        d1 = __builtin_amdgcn_fdot2_f32_bf16(k3, q3, d1, false);
      }
      s = (d0 + d1) * scale_log2;                       // log2-domain score
    }
    // online softmax (per wave = head)
    const float tmax = wave_max(s);
    if (tmax > -INFINITY) {
      const float m_new = fmaxf(m, tmax);
      const float alpha = (m == -INFINITY) ? 0.f : __builtin_amdgcn_exp2f(m - m_new);
      const float p = rvalid ? __builtin_amdgcn_exp2f(s - m_new) : 0.f;
      l = l * alpha + wave_sum(p);
      m = m_new;
#pragma unroll
      for (int i = 0; i < 16; i++) acc[i] *= alpha;
      // PV: lane owns dims [16 lane, 16 lane + 16) = chunks 2 lane, 2 lane + 1 of every row
      for (int r = 0; r < nrows; r++) {
        const float pr = __shfl(p, r, 32);
        if (pr == 0.f) continue;
        const unsigned char* vr = kt + r * ROWB;
        const v8bf16 v0 = *reinterpret_cast<const v8bf16*>(vr + (((2 * lane + r) & (CHUNKS - 1)) << 4));
        const v8bf16 v1 = *reinterpret_cast<const v8bf16*>(vr + (((2 * lane + 1 + r) & (CHUNKS - 1)) << 4));
#pragma unroll
        for (int i = 0; i < 8; i++) {
          acc[i] = fmaf(pr, (float)v0[i], acc[i]);
          acc[8 + i] = fmaf(pr, (float)v1[i], acc[8 + i]);
        }
      }
    }
  }
  if (!hvalid) return;
  if constexpr (PARTIAL) {
    // scratch [T, nsplit, HB, 2 + D]: m, l, acc
    float* p = scratch + (((long)t * nsplit + split) * HB + wave) * (2 + D);
    if (lane == 0) { p[0] = m; p[1] = l; }
#pragma unroll
    for (int i = 0; i < 16; i++) p[2 + 16 * lane + i] = acc[i];
  } else {
    const float inv = l > 0.f ? out_mult / l : 0.f;
    uint16_t* o = out + (long)t * out_stride_t + (long)h * out_stride_h + 16 * lane;
    v8f a0, a1;
#pragma unroll
    for (int i = 0; i < 8; i++) { a0[i] = acc[i] * inv; a1[i] = acc[8 + i] * inv; }
    const v4u r0v = {pack_hi16(a0[0], a0[1]), pack_hi16(a0[2], a0[3]), pack_hi16(a0[4], a0[5]), pack_hi16(a0[6], a0[7])};
    const v4u r1v = {pack_hi16(a1[0], a1[1]), pack_hi16(a1[2], a1[3]), pack_hi16(a1[4], a1[5]), pack_hi16(a1[6], a1[7])};
    // round-to-nearest-even would need the add3 trick; truncation is <= 2^-8 relative and matches r9k_attn's choice
    *reinterpret_cast<v4u*>(o) = r0v;
    *reinterpret_cast<v4u*>(o + 8) = r1v;
  }
}

// combine: block (t, head group), wave = head, lane = 16 dims; nsplit partials
__global__ void __launch_bounds__(NT) sparse_mla_combine_kernel(
    const float* __restrict__ scratch, uint16_t* __restrict__ out, int nheads, long out_stride_t, long out_stride_h,
    float out_mult, int nsplit) {
  const int t = blockIdx.x, g = blockIdx.y;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
  const int h = g * HB + wave;
  if (h >= nheads) return;
  const float* base = scratch + ((long)t * nsplit * HB + wave) * (2 + D);
  float M = -INFINITY;
  for (int s = 0; s < nsplit; s++) M = fmaxf(M, base[(long)s * HB * (2 + D)]);
  float acc[16], L = 0.f;
#pragma unroll
  for (int i = 0; i < 16; i++) acc[i] = 0.f;
  if (M > -INFINITY) {
    for (int s = 0; s < nsplit; s++) {
      const float* p = base + (long)s * HB * (2 + D);
      const float w = p[0] > -INFINITY ? __builtin_amdgcn_exp2f(p[0] - M) : 0.f;
      L += w * p[1];
#pragma unroll
      for (int i = 0; i < 16; i++) acc[i] = fmaf(w, p[2 + 16 * lane + i], acc[i]);
    }
  }
  const float inv = L > 0.f ? out_mult / L : 0.f;
  uint16_t* o = out + (long)t * out_stride_t + (long)h * out_stride_h + 16 * lane;
  v8f a0, a1;
#pragma unroll
  for (int i = 0; i < 8; i++) { a0[i] = acc[i] * inv; a1[i] = acc[8 + i] * inv; }
  const v4u r0v = {pack_hi16(a0[0], a0[1]), pack_hi16(a0[2], a0[3]), pack_hi16(a0[4], a0[5]), pack_hi16(a0[6], a0[7])};
  const v4u r1v = {pack_hi16(a1[0], a1[1]), pack_hi16(a1[2], a1[3]), pack_hi16(a1[4], a1[5]), pack_hi16(a1[6], a1[7])};
  *reinterpret_cast<v4u*>(o) = r0v;
  *reinterpret_cast<v4u*>(o + 8) = r1v;
}
}  // namespace

// q [T, H, 512] bf16 (strides in elements), pool [slots, 512] (kv_fp8: uint8 e4m3, else bf16), indices int32
// ragged (indptr [T+1]), out [T, H, 512] bf16. descale: scalar K/V descale (1.0 for bf16). nsplit > 1 needs
// scratch of T * nsplit * 8 * ceil(H/8) ... per head-group handled by the grid: scratch bytes =
// T * nsplit * HB * (2 + 512) * 4 * ceil(H / 8). Returns 0 or <0 for bad geometry, else the hip error.
extern "C" long r9k_sparse_mla_scratch_bytes(int ntok, int nheads, int nsplit) {
  return (long)ntok * nsplit * HB * (2 + D) * 4 * ((nheads + HB - 1) / HB);
}
extern "C" int r9k_sparse_mla(long q_ptr, long pool_ptr, long indices_ptr, long indptr_ptr, long out_ptr,
                              long scratch_ptr, int kv_fp8, float descale, int ntok, int nheads, int d,
                              long q_stride_t, long q_stride_h, long pool_stride, long out_stride_t, long out_stride_h,
                              float scale, int nsplit, long stream) {
  if (d != D || ntok <= 0 || nheads <= 0 || nsplit <= 0) return -1;
  if (nsplit > 1 && !scratch_ptr) return -2;
  const int hg = (nheads + HB - 1) / HB;
  hipStream_t st = (hipStream_t)stream;
  const float scale_log2 = scale * descale * LOG2E;
  const float out_mult = descale;
  const dim3 grid(ntok, nsplit, hg), block(NT);
  const size_t lds = LDS_Q + LDS_K;
  // hg > 1 with split-K: the scratch layout indexes heads by (g * HB + wave) via the kernel's own addressing of
  // `wave`; combine reads the same. Keep hg == 1 for the split path (TP8 GLM has 8 heads per rank).
  if (nsplit > 1 && hg != 1) return -3;
  if (kv_fp8) {
    if (nsplit == 1)
      hipLaunchKernelGGL((sparse_mla_kernel<uint8_t, false>), grid, block, lds, st, (const uint16_t*)q_ptr,
                         (const uint8_t*)pool_ptr, (const int*)indices_ptr, (const int*)indptr_ptr, (uint16_t*)out_ptr,
                         (float*)scratch_ptr, nheads, q_stride_t, q_stride_h, pool_stride, out_stride_t, out_stride_h,
                         scale_log2, out_mult, nsplit);
    else
      hipLaunchKernelGGL((sparse_mla_kernel<uint8_t, true>), grid, block, lds, st, (const uint16_t*)q_ptr,
                         (const uint8_t*)pool_ptr, (const int*)indices_ptr, (const int*)indptr_ptr, (uint16_t*)out_ptr,
                         (float*)scratch_ptr, nheads, q_stride_t, q_stride_h, pool_stride, out_stride_t, out_stride_h,
                         scale_log2, out_mult, nsplit);
  } else {
    if (nsplit == 1)
      hipLaunchKernelGGL((sparse_mla_kernel<uint16_t, false>), grid, block, lds, st, (const uint16_t*)q_ptr,
                         (const uint16_t*)pool_ptr, (const int*)indices_ptr, (const int*)indptr_ptr, (uint16_t*)out_ptr,
                         (float*)scratch_ptr, nheads, q_stride_t, q_stride_h, pool_stride, out_stride_t, out_stride_h,
                         scale_log2, out_mult, nsplit);
    else
      hipLaunchKernelGGL((sparse_mla_kernel<uint16_t, true>), grid, block, lds, st, (const uint16_t*)q_ptr,
                         (const uint16_t*)pool_ptr, (const int*)indices_ptr, (const int*)indptr_ptr, (uint16_t*)out_ptr,
                         (float*)scratch_ptr, nheads, q_stride_t, q_stride_h, pool_stride, out_stride_t, out_stride_h,
                         scale_log2, out_mult, nsplit);
  }
  if (nsplit > 1)
    hipLaunchKernelGGL(sparse_mla_combine_kernel, dim3(ntok, hg), block, 0, st, (const float*)scratch_ptr,
                       (uint16_t*)out_ptr, nheads, out_stride_t, out_stride_h, out_mult, nsplit);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}
