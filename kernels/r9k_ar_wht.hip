// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas

// r9k_ar_wht.hip -- compressed 2-rank all-reduce payload: Walsh-Hadamard rotation + 6-bit group quantisation.
//
// Independent implementation (see notes/independence.md): the transform is the ordinary 64-point Walsh-Hadamard
// butterfly, and rotating a block before low-bit quantisation to spread outliers across the block is a published
// technique (QuIP, QuaRot and others use it for weights/activations; the same argument applies to an all-reduce
// payload). No libr4d source or binary was consulted.
//
// Why it helps: the two GPUs talk over PCIe 3 here, so a large all-reduce is link-bound -- the exact kernel in
// r9k_ar.hip reaches only ~1.1x RCCL at 2 MB. Shipping 6.25 bits per element instead of 16 cuts the bytes on the
// wire by 2.56x, which is the entire win at prefill sizes.
//
// Format, per 64-element group: rotate by H64 (orthonormal, so the same butterfly inverts it), take the group's
// max |value| after rotation, and store 64 signed 6-bit codes (48 bytes) followed by one bf16 scale (2 bytes)
// = 50 bytes = 6.25 bits/element. Codes are round-to-nearest, clamped to [-31, 31] so negation is exact.
//
// Bit-identical BETWEEN RANKS (not to the exact sum): each rank reduces the same pair of *packed* groups -- its
// own packed copy and the peer's -- so both compute the same fp32 sum from the same inputs. That is why the local
// payload is packed and read back from the packed buffer rather than summed from the original input. Lossy, so
// it is opt-in and gated on downstream quality; the caller keeps small (decode) messages on the exact kernel.
//
// Layout note: one 64-element group is held one element per lane across a 64-thread half-block (two wave32s), so
// butterfly strides 1..16 are wave shuffles and stride 32 crosses the wave pair through LDS. Every barrier is
// reached by every thread of the block -- the tail is handled by predicating memory access, never by returning
// early, because a divergent s_barrier hangs the workgroup.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

#define WHT_GROUP 64                 // elements per rotation/quantisation group
#define WHT_BITS 6                   // default; the kernels are templated on B (6 or 4)
#define WHT_CODES 48                 // 64 * 6 / 8 bytes of codes
#define WHT_BYTES 50                 // + sizeof(bf16) scale
#define WHT_MAXC 31.0f
// Per-width constants. 4-bit halves the codes to 32 bytes (34 with the scale) = 4.25 bits/elem against 6.25,
// i.e. 1.47x fewer bytes on the wire -- which is the only lever left once the link is saturated. The price is
// quantisation error: the step is amax/7 instead of amax/31, so ~4x the relative error per call.
template <int B> struct WhtW {
  static constexpr int codes = WHT_GROUP * B / 8;
  static constexpr int bytes = codes + 2;
  static constexpr float maxc = (float)((1 << (B - 1)) - 1);   // 31 for 6-bit, 7 for 4-bit; -max is exact
  static constexpr unsigned mask = (1u << B) - 1u;
};
#define WHT_WAVE 32
#define WHT_NT 256                   // threads per block = 4 groups
#define WHT_GPB (WHT_NT / WHT_GROUP)

typedef unsigned int u32;

__device__ __forceinline__ float wht_ld(const __hip_bfloat16 v) { return (float)v; }
__device__ __forceinline__ float wht_ld(const _Float16 v) { return (float)v; }
__device__ __forceinline__ void wht_st(__hip_bfloat16& d, float v) { d = (__hip_bfloat16)v; }
__device__ __forceinline__ void wht_st(_Float16& d, float v) { d = (_Float16)v; }

// Per-block LDS: one float per element (the butterfly exchange and the amax fold) plus one int per element
// (the code exchange during packing). Both are indexed by [group-in-block][lane].
struct WhtSmem {
  float f[WHT_GPB * WHT_GROUP];
  int   c[WHT_GPB * WHT_GROUP];
};

// 64-point orthonormal Walsh-Hadamard. Self-inverse with the 1/8 factor applied by the caller.
// All barriers are unconditional: callers must not branch around this.
__device__ __forceinline__ float wht64(float v, int lane, float* g) {
#pragma unroll
  for (int s = 1; s < WHT_WAVE; s <<= 1) {            // strides 1,2,4,8,16 stay inside one wave32
    const float o = __shfl_xor(v, s, WHT_WAVE);
    v = (lane & s) ? o - v : v + o;
  }
  g[lane] = v;                                        // stride 32 crosses the wave pair
  __syncthreads();
  const float o = g[lane ^ 32];
  __syncthreads();
  return (lane & 32) ? o - v : v + o;
}

// Max |.| over the 64 lanes of a group.
__device__ __forceinline__ float wht_amax(float v, int lane, float* g) {
  float a = fabsf(v);
#pragma unroll
  for (int s = 1; s < WHT_WAVE; s <<= 1) a = fmaxf(a, __shfl_xor(a, s, WHT_WAVE));
  g[lane] = a;                                        // every lane of a wave now holds that wave's max
  __syncthreads();
  const float both = fmaxf(g[lane & 31], g[(lane & 31) + 32]);
  __syncthreads();
  return both;
}

// ---------------------------------------------------------------- pack: [numel] -> ceil(numel/64) * 50 bytes
template <typename T, int B = WHT_BITS>
__global__ void __launch_bounds__(WHT_NT) r9k_wht_pack_kernel(
    const T* __restrict__ x, unsigned char* __restrict__ pk, long ngroups) {
  using W = WhtW<B>;
  __shared__ WhtSmem sm;
  const int tid = threadIdx.x, lane = tid & (WHT_GROUP - 1), gib = tid >> 6;
  const long g = (long)blockIdx.x * WHT_GPB + gib;
  const bool live = g < ngroups;
  float* const gf = sm.f + gib * WHT_GROUP;
  int* const gc = sm.c + gib * WHT_GROUP;

  const float in = live ? wht_ld(x[g * WHT_GROUP + lane]) : 0.f;
  const float v = wht64(in, lane, gf) * 0.125f;
  const float a = wht_amax(v, lane, gf);

  const float scale = a * (1.0f / W::maxc);
  const float inv = a > 0.f ? W::maxc / a : 0.f;
  int c = __float2int_rn(v * inv);
  const int cm = (int)W::maxc;
  c = c > cm ? cm : (c < -cm ? -cm : c);
  gc[lane] = c & (int)W::mask;                        // two's complement in B bits
  __syncthreads();

  // Byte b holds bits [8b, 8b+8) of the stream. For B=6, 8b mod 6 is always even (0/2/4); for B=4 it is always
  // 0. Either way a byte spans at most two codes, so each byte has exactly one owner and no atomics are needed.
  if (live && lane < W::codes) {
    const int bit = lane * 8, e0 = bit / B, off = bit - e0 * B;
    u32 w = (u32)gc[e0] >> off;
    if (e0 + 1 < WHT_GROUP) w |= (u32)gc[e0 + 1] << (B - off);
    if (B < 8 && e0 + 2 < WHT_GROUP && off + 2 * B < 8)      // B=4, off=0: a byte holds exactly two codes
      w |= (u32)gc[e0 + 2] << (2 * B - off);
    pk[g * W::bytes + lane] = (unsigned char)(w & 0xFF);
  }
  if (live && lane == 0) *(__hip_bfloat16*)(pk + g * W::bytes + W::codes) = (__hip_bfloat16)scale;
}

// ---------------------------------------------------------------- reduce: unpack both payloads, sum, invert
template <typename T, int B = WHT_BITS>
__global__ void __launch_bounds__(WHT_NT) r9k_wht_reduce_kernel(
    const unsigned char* __restrict__ mine, const unsigned char* __restrict__ peer,
    T* __restrict__ out, long ngroups) {
  using W = WhtW<B>;
  __shared__ WhtSmem sm;
  const int tid = threadIdx.x, lane = tid & (WHT_GROUP - 1), gib = tid >> 6;
  const long g = (long)blockIdx.x * WHT_GPB + gib;
  const bool live = g < ngroups;
  float* const gf = sm.f + gib * WHT_GROUP;

  float acc = 0.f;
  if (live) {
    const int bit = lane * B, by = bit >> 3, off = bit & 7;
#pragma unroll
    for (int src = 0; src < 2; src++) {
      const unsigned char* p = (src ? peer : mine) + g * W::bytes;
      const u32 w = (u32)p[by] | ((u32)p[by + 1] << 8);   // a code never spans more than two bytes
      int c = (int)((w >> off) & W::mask);
      if (c & (1 << (B - 1))) c -= (1 << B);              // sign-extend
      acc += (float)c * (float)*(const __hip_bfloat16*)(p + W::codes);
    }
  }
  const float v = wht64(acc, lane, gf) * 0.125f;
  if (live) wht_st(out[g * WHT_GROUP + lane], v);
}

extern "C" int r9k_wht_group() { return WHT_GROUP; }
extern "C" int r9k_wht_bits() { return WHT_BITS; }
extern "C" int r9k_wht_group_bytes() { return WHT_BYTES; }
// bytes per group at a given width (bits 0 or 6 -> 50, 4 -> 34)
extern "C" int r9k_wht_bytes_for(int bits) {
  if (bits == 0 || bits == 6) return WhtW<6>::bytes;
  if (bits == 4) return WhtW<4>::bytes;
  return -1;
}

static inline dim3 wht_grid(long ng) { return dim3((unsigned)((ng + WHT_GPB - 1) / WHT_GPB)); }

template <typename T, int B>
static int wht_pack_launch(long x, long pk, long ng, long stream) {
  hipLaunchKernelGGL((r9k_wht_pack_kernel<T, B>), wht_grid(ng), dim3(WHT_NT), 0, (hipStream_t)stream,
                     (const T*)x, (unsigned char*)pk, ng);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

template <typename T, int B>
static int wht_reduce_launch(long mine, long peer, long out, long ng, long stream) {
  hipLaunchKernelGGL((r9k_wht_reduce_kernel<T, B>), wht_grid(ng), dim3(WHT_NT), 0, (hipStream_t)stream,
                     (const unsigned char*)mine, (const unsigned char*)peer, (T*)out, ng);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

// Same reduce, but the peer payload's address is resolved ON THE DEVICE from the double-buffer slot that
// r9k_ar_push_2rank recorded. The host must not read that slot back -- it advances inside a replayed graph.
template <typename T, int B = WHT_BITS>
__global__ void __launch_bounds__(WHT_NT) r9k_wht_reduce_at_kernel(
    const unsigned char* __restrict__ mine, const unsigned char* __restrict__ scratch,
    const int* __restrict__ slot, long slot_stride, T* __restrict__ out, long ngroups) {
  using W = WhtW<B>;
  __shared__ WhtSmem sm;
  const int tid = threadIdx.x, lane = tid & (WHT_GROUP - 1), gib = tid >> 6;
  const long g = (long)blockIdx.x * WHT_GPB + gib;
  const bool live = g < ngroups;
  float* const gf = sm.f + gib * WHT_GROUP;
  const unsigned char* const peer = scratch + (long)(*slot) * slot_stride;

  float acc = 0.f;
  if (live) {
    const int bit = lane * B, by = bit >> 3, off = bit & 7;
#pragma unroll
    for (int src = 0; src < 2; src++) {
      const unsigned char* p = (src ? peer : mine) + g * W::bytes;
      const u32 w = (u32)p[by] | ((u32)p[by + 1] << 8);
      int c = (int)((w >> off) & W::mask);
      if (c & (1 << (B - 1))) c -= (1 << B);
      acc += (float)c * (float)*(const __hip_bfloat16*)(p + W::codes);
    }
  }
  const float v = wht64(acc, lane, gf) * 0.125f;
  if (live) wht_st(out[g * WHT_GROUP + lane], v);
}

template <typename T, int B>
static int wht_reduce_at_launch(long mine, long scratch, long slot, long slot_stride, long out, long ng,
                                long stream) {
  hipLaunchKernelGGL((r9k_wht_reduce_at_kernel<T, B>), wht_grid(ng), dim3(WHT_NT), 0, (hipStream_t)stream,
                     (const unsigned char*)mine, (const unsigned char*)scratch, (const int*)slot, slot_stride,
                     (T*)out, ng);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

extern "C" int r9k_wht_reduce_at(long mine, long scratch, long slot, long slot_stride, long out, long numel,
                                 int dtype, int bits, long stream) {
  if (numel % WHT_GROUP) return -1;
  const long ng = numel / WHT_GROUP;
  if (ng <= 0) return 0;
  const bool b6 = (bits == 0 || bits == 6);
  if (!b6 && bits != 4) return -8;
  if (dtype == 0)
    return b6 ? wht_reduce_at_launch<__hip_bfloat16, 6>(mine, scratch, slot, slot_stride, out, ng, stream)
              : wht_reduce_at_launch<__hip_bfloat16, 4>(mine, scratch, slot, slot_stride, out, ng, stream);
  if (dtype == 1)
    return b6 ? wht_reduce_at_launch<_Float16, 6>(mine, scratch, slot, slot_stride, out, ng, stream)
              : wht_reduce_at_launch<_Float16, 4>(mine, scratch, slot, slot_stride, out, ng, stream);
  return -4;
}

// Pack `numel` elements (numel % 64 == 0) of `x` into `pk`. dtype: 0 bf16, 1 fp16.
extern "C" int r9k_wht_pack(long x, long pk, long numel, int dtype, int bits, long stream) {
  if (numel % WHT_GROUP) return -1;
  const long ng = numel / WHT_GROUP;
  if (ng <= 0) return 0;
  const bool b6 = (bits == 0 || bits == 6);
  if (!b6 && bits != 4) return -8;
  if (dtype == 0)
    return b6 ? wht_pack_launch<__hip_bfloat16, 6>(x, pk, ng, stream)
              : wht_pack_launch<__hip_bfloat16, 4>(x, pk, ng, stream);
  if (dtype == 1)
    return b6 ? wht_pack_launch<_Float16, 6>(x, pk, ng, stream)
              : wht_pack_launch<_Float16, 4>(x, pk, ng, stream);
  return -4;
}

// Sum the two packed payloads and write `numel` elements to `out`.
extern "C" int r9k_wht_reduce(long mine, long peer, long out, long numel, int dtype, int bits, long stream) {
  if (numel % WHT_GROUP) return -1;
  const long ng = numel / WHT_GROUP;
  if (ng <= 0) return 0;
  const bool b6 = (bits == 0 || bits == 6);
  if (!b6 && bits != 4) return -8;
  if (dtype == 0)
    return b6 ? wht_reduce_launch<__hip_bfloat16, 6>(mine, peer, out, ng, stream)
              : wht_reduce_launch<__hip_bfloat16, 4>(mine, peer, out, ng, stream);
  if (dtype == 1)
    return b6 ? wht_reduce_launch<_Float16, 6>(mine, peer, out, ng, stream)
              : wht_reduce_launch<_Float16, 4>(mine, peer, out, ng, stream);
  return -4;
}
