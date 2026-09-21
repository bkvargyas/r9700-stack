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
#define WHT_BITS 6
#define WHT_CODES 48                 // 64 * 6 / 8 bytes of codes
#define WHT_BYTES 50                 // + sizeof(bf16) scale
#define WHT_MAXC 31.0f
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
template <typename T>
__global__ void __launch_bounds__(WHT_NT) r9k_wht_pack_kernel(
    const T* __restrict__ x, unsigned char* __restrict__ pk, long ngroups) {
  __shared__ WhtSmem sm;
  const int tid = threadIdx.x, lane = tid & (WHT_GROUP - 1), gib = tid >> 6;
  const long g = (long)blockIdx.x * WHT_GPB + gib;
  const bool live = g < ngroups;
  float* const gf = sm.f + gib * WHT_GROUP;
  int* const gc = sm.c + gib * WHT_GROUP;

  const float in = live ? wht_ld(x[g * WHT_GROUP + lane]) : 0.f;
  const float v = wht64(in, lane, gf) * 0.125f;
  const float a = wht_amax(v, lane, gf);

  const float scale = a * (1.0f / WHT_MAXC);
  const float inv = a > 0.f ? WHT_MAXC / a : 0.f;
  int c = __float2int_rn(v * inv);
  c = c > 31 ? 31 : (c < -31 ? -31 : c);
  gc[lane] = c & 0x3F;                                // 6-bit two's complement
  __syncthreads();

  // Byte b holds bits [8b, 8b+8) of the 6-bit stream. 8b mod 6 is always even (0, 2 or 4), so a byte spans at
  // most two codes -- no third term is needed and no atomics are required: each byte has exactly one owner.
  if (live && lane < WHT_CODES) {
    const int bit = lane * 8, e0 = bit / WHT_BITS, off = bit - e0 * WHT_BITS;
    u32 w = (u32)gc[e0] >> off;
    if (e0 + 1 < WHT_GROUP) w |= (u32)gc[e0 + 1] << (WHT_BITS - off);
    pk[g * WHT_BYTES + lane] = (unsigned char)(w & 0xFF);
  }
  if (live && lane == 0) *(__hip_bfloat16*)(pk + g * WHT_BYTES + WHT_CODES) = (__hip_bfloat16)scale;
}

// ---------------------------------------------------------------- reduce: unpack both payloads, sum, invert
template <typename T>
__global__ void __launch_bounds__(WHT_NT) r9k_wht_reduce_kernel(
    const unsigned char* __restrict__ mine, const unsigned char* __restrict__ peer,
    T* __restrict__ out, long ngroups) {
  __shared__ WhtSmem sm;
  const int tid = threadIdx.x, lane = tid & (WHT_GROUP - 1), gib = tid >> 6;
  const long g = (long)blockIdx.x * WHT_GPB + gib;
  const bool live = g < ngroups;
  float* const gf = sm.f + gib * WHT_GROUP;

  float acc = 0.f;
  if (live) {
    const int bit = lane * WHT_BITS, by = bit >> 3, off = bit & 7;
#pragma unroll
    for (int src = 0; src < 2; src++) {
      const unsigned char* p = (src ? peer : mine) + g * WHT_BYTES;
      const u32 w = (u32)p[by] | ((u32)p[by + 1] << 8);   // a code never spans more than two bytes
      int c = (int)((w >> off) & 0x3F);
      if (c & 0x20) c -= 64;                              // sign-extend
      acc += (float)c * (float)*(const __hip_bfloat16*)(p + WHT_CODES);
    }
  }
  const float v = wht64(acc, lane, gf) * 0.125f;
  if (live) wht_st(out[g * WHT_GROUP + lane], v);
}

extern "C" int r9k_wht_group() { return WHT_GROUP; }
extern "C" int r9k_wht_bits() { return WHT_BITS; }
extern "C" int r9k_wht_group_bytes() { return WHT_BYTES; }

static inline dim3 wht_grid(long ng) { return dim3((unsigned)((ng + WHT_GPB - 1) / WHT_GPB)); }

template <typename T>
static int wht_pack_launch(long x, long pk, long ng, long stream) {
  hipLaunchKernelGGL(r9k_wht_pack_kernel<T>, wht_grid(ng), dim3(WHT_NT), 0, (hipStream_t)stream,
                     (const T*)x, (unsigned char*)pk, ng);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

template <typename T>
static int wht_reduce_launch(long mine, long peer, long out, long ng, long stream) {
  hipLaunchKernelGGL(r9k_wht_reduce_kernel<T>, wht_grid(ng), dim3(WHT_NT), 0, (hipStream_t)stream,
                     (const unsigned char*)mine, (const unsigned char*)peer, (T*)out, ng);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

// Same reduce, but the peer payload's address is resolved ON THE DEVICE from the double-buffer slot that
// r9k_ar_push_2rank recorded. The host must not read that slot back -- it advances inside a replayed graph.
template <typename T>
__global__ void __launch_bounds__(WHT_NT) r9k_wht_reduce_at_kernel(
    const unsigned char* __restrict__ mine, const unsigned char* __restrict__ scratch,
    const int* __restrict__ slot, long slot_stride, T* __restrict__ out, long ngroups) {
  __shared__ WhtSmem sm;
  const int tid = threadIdx.x, lane = tid & (WHT_GROUP - 1), gib = tid >> 6;
  const long g = (long)blockIdx.x * WHT_GPB + gib;
  const bool live = g < ngroups;
  float* const gf = sm.f + gib * WHT_GROUP;
  const unsigned char* const peer = scratch + (long)(*slot) * slot_stride;

  float acc = 0.f;
  if (live) {
    const int bit = lane * WHT_BITS, by = bit >> 3, off = bit & 7;
#pragma unroll
    for (int src = 0; src < 2; src++) {
      const unsigned char* p = (src ? peer : mine) + g * WHT_BYTES;
      const u32 w = (u32)p[by] | ((u32)p[by + 1] << 8);
      int c = (int)((w >> off) & 0x3F);
      if (c & 0x20) c -= 64;
      acc += (float)c * (float)*(const __hip_bfloat16*)(p + WHT_CODES);
    }
  }
  const float v = wht64(acc, lane, gf) * 0.125f;
  if (live) wht_st(out[g * WHT_GROUP + lane], v);
}

template <typename T>
static int wht_reduce_at_launch(long mine, long scratch, long slot, long slot_stride, long out, long ng,
                                long stream) {
  hipLaunchKernelGGL(r9k_wht_reduce_at_kernel<T>, wht_grid(ng), dim3(WHT_NT), 0, (hipStream_t)stream,
                     (const unsigned char*)mine, (const unsigned char*)scratch, (const int*)slot, slot_stride,
                     (T*)out, ng);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

extern "C" int r9k_wht_reduce_at(long mine, long scratch, long slot, long slot_stride, long out, long numel,
                                 int dtype, long stream) {
  if (numel % WHT_GROUP) return -1;
  const long ng = numel / WHT_GROUP;
  if (ng <= 0) return 0;
  if (dtype == 0) return wht_reduce_at_launch<__hip_bfloat16>(mine, scratch, slot, slot_stride, out, ng, stream);
  if (dtype == 1) return wht_reduce_at_launch<_Float16>(mine, scratch, slot, slot_stride, out, ng, stream);
  return -4;
}

// Pack `numel` elements (numel % 64 == 0) of `x` into `pk`. dtype: 0 bf16, 1 fp16.
extern "C" int r9k_wht_pack(long x, long pk, long numel, int dtype, long stream) {
  if (numel % WHT_GROUP) return -1;
  const long ng = numel / WHT_GROUP;
  if (ng <= 0) return 0;
  if (dtype == 0) return wht_pack_launch<__hip_bfloat16>(x, pk, ng, stream);
  if (dtype == 1) return wht_pack_launch<_Float16>(x, pk, ng, stream);
  return -4;
}

// Sum the two packed payloads and write `numel` elements to `out`.
extern "C" int r9k_wht_reduce(long mine, long peer, long out, long numel, int dtype, long stream) {
  if (numel % WHT_GROUP) return -1;
  const long ng = numel / WHT_GROUP;
  if (ng <= 0) return 0;
  if (dtype == 0) return wht_reduce_launch<__hip_bfloat16>(mine, peer, out, ng, stream);
  if (dtype == 1) return wht_reduce_launch<_Float16>(mine, peer, out, ng, stream);
  return -4;
}
