// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas

// r9k_ar4.hip -- compressed, topology-aware all-reduce for four R9700s on two PCIe switches (prefill sizes).
//
// Independent implementation (see notes/independence.md and notes/ar4-plan.md). Algorithms used: hierarchical
// (2x2) reduce-scatter / all-gather, the 64-point Walsh-Hadamard rotation before low-bit quantisation (QuIP /
// QuaRot style, as in r9k_ar_wht.hip), and the release/acquire flag handshake of r9k_ar.hip. No third-party
// collective code was consulted.
//
// Why: the two PEX 8747 switches meet through ONE x16 Gen3 path per direction (~13.5 GB/s, tuning/ar_push_bench.py),
// so an exact all-reduce cannot beat RCCL's ring here (1.73 ms per 11 MB message, ~40% above the byte bound) and the
// exact all-to-all two-shot is 2.3x slower still. Only fewer bytes win. This moves N*c bytes per rank over its own
// switch link and N*c/2 per direction over the shared path, c = 0.27 (4-bit) or 0.39 (6-bit) of bf16.
//
// Ranks: a PAIR shares a switch (pairs given by the caller, e.g. {0,1} and {2,3}); each rank's CROSS partner is the
// rank with the same position in the other pair. Rank r with pair position h (0/1) and pair index s (0/1) owns
// final quarter q = 2h + s of the message (quarters are contiguous 64-element-group ranges).
//   P1  pack-push its partner's half to the pair partner; wait for ours          (intra, local link)
//   R1  own half + decoded received half -> pair partial (bf16), rotated-domain sum
//   P2  pack-push the cross partner's quarter of the partial; wait for ours       (cross, shared path)
//   R2  own quarter of the partial + received -> FINAL quarter, quantised straight from the rotated sum into a
//       local packed buffer AND pushed to the cross partner; wait for its final quarter                (cross)
//   D3  decode own packed quarter and the received one into out (owner uses its own packed bytes too)
//   P4  forward both packed quarters (= our half, packed) to the pair partner; wait for theirs         (intra)
//   D4  decode the received half into out
// Every rank decodes identical bytes for every quarter, so all outputs are bit-identical. Quantisation count per
// element: the pair partner's term once, the cross terms twice (their pair partial then the cross exchange), the
// final value once for the gather -- more than the 2-rank path's single quantisation; gate with the paired eval.
//
// Wire format per 64-element group: B-bit two's-complement codes of round(v * maxc / amax) after the orthonormal
// H64 (B=4: 32 bytes, B=6: 48 bytes), and a bf16 scale = amax / maxc. A packed message of G groups is
// [codes: G*cbytes, padded to 16][scales: G*2, padded to 16]. Element order inside a group: lane l of the packing
// wave holds elements l and l+32, so H64 stages 1..16 are wave shuffles and stage 32 is in-lane -- no LDS, no
// block barrier in the transform. Reductions add rotated-domain values (the transform is linear), so the
// reduce-scatter phases never invert it and the final quarter is quantised directly from the rotated sum.
//
// Block = 256 threads = 8 waves, 8 groups per wave per iteration => 64 groups (4096 elements, 8 KB bf16 in,
// 2176 / 3200 bytes out) staged in LDS and stored to the peer with 16-byte lanes. Handshake: per (phase, block)
// flag word, device-resident sequence counter (cudagraph-safe), RELEASE store / ACQUIRE poll at system scope
// (r9k_ar.hip), double-buffered receive areas by sequence parity.
// No device printf/assert (the build enforces hostcall-free objects).

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

namespace {

typedef unsigned int u32;
typedef unsigned char u8;

constexpr int GROUP = 64;                 // elements per rotation group
constexpr int NT = 256;                   // threads per block
constexpr int NWAVE = NT / 32;
constexpr int GPW = 8;                    // groups per wave per iteration
constexpr int GPB = NWAVE * GPW;          // 64 groups per block iteration
constexpr int MAX_BLOCKS = 256;           // per phase; flags sized accordingly

template <int B> struct Wire {
  static constexpr int cbytes = GROUP * B / 8;                    // 32 or 48 code bytes per group
  static constexpr float maxc = (float)((1 << (B - 1)) - 1);      // 7 or 31
  static constexpr u32 mask = (1u << B) - 1u;
};

__host__ __device__ constexpr long align16(long n) { return (n + 15) & ~15L; }
__host__ __device__ constexpr long codes_bytes(long ngroups, int cbytes) { return align16(ngroups * cbytes); }
__host__ __device__ constexpr long packed_bytes(long ngroups, int cbytes) {
  return codes_bytes(ngroups, cbytes) + align16(ngroups * 2);
}

__device__ __forceinline__ float bf2f(uint16_t v) { return __uint_as_float((u32)v << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {          // round to nearest even
  const u32 x = __float_as_uint(f);
  return (uint16_t)((x + 0x7FFFu + ((x >> 16) & 1u)) >> 16);
}

// ---------------------------------------------------------------- 64-point Walsh-Hadamard, two elements per lane
// Lane l holds elements l (a) and l+32 (b). Orthonormal with the 1/8 applied by the caller where it matters; the
// forward transform is its own inverse (H^2 = 64 I), so decode = same butterfly * (1/64) overall, i.e. *0.125 each way.
__device__ __forceinline__ void wht64(float& a, float& b, int lane) {
#pragma unroll
  for (int s = 1; s < 32; s <<= 1) {
    const float oa = __shfl_xor(a, s, 32), ob = __shfl_xor(b, s, 32);
    a = (lane & s) ? oa - a : a + oa;
    b = (lane & s) ? ob - b : b + ob;
  }
  const float t = a;
  a = t + b;
  b = t - b;
}

__device__ __forceinline__ float wave_amax(float a, float b) {
  float m = fmaxf(fabsf(a), fabsf(b));
#pragma unroll
  for (int s = 1; s < 32; s <<= 1) m = fmaxf(m, __shfl_xor(m, s, 32));
  return m;
}

// Quantise the lane's two rotated values against the group amax; returns the two B-bit codes.
template <int B>
__device__ __forceinline__ void quant2(float a, float b, float amax, u32& ca, u32& cb, float& scale) {
  using W = Wire<B>;
  scale = amax * (1.0f / W::maxc);
  const float inv = amax > 0.f ? W::maxc / amax : 0.f;
  const int cm = (int)W::maxc;
  int ia = __float2int_rn(a * inv), ib = __float2int_rn(b * inv);
  ia = ia > cm ? cm : (ia < -cm ? -cm : ia);
  ib = ib > cm ? cm : (ib < -cm ? -cm : ib);
  ca = (u32)ia & W::mask;
  cb = (u32)ib & W::mask;
}

// Write one group's codes (this lane's share) into LDS at `dst` (cbytes bytes for the group).
template <int B>
__device__ __forceinline__ void codes_to_lds(u32 ca, u32 cb, int lane, u8* dst) {
  if constexpr (B == 4) {
    // byte j holds elements 2j (low nibble) and 2j+1 (high): j < 16 -> two `a` codes, else two `b` codes
    const int src = (2 * lane) & 31;
    const u32 a0 = __shfl(ca, src, 32), a1 = __shfl(ca, src + 1, 32);
    const u32 b0 = __shfl(cb, src, 32), b1 = __shfl(cb, src + 1, 32);
    const u32 lo = lane < 16 ? a0 : b0, hi = lane < 16 ? a1 : b1;
    dst[lane] = (u8)(lo | (hi << 4));
  } else {
    // 16 triples of 3 bytes: triple t holds elements 4t..4t+3 (t < 8 -> `a` codes of lanes 4t.., else `b` codes of
    // lanes 4t-32..). Shuffle both code variables from the four source lanes, then select per destination lane.
    const int t = lane & 15;
    const int src = (4 * t) & 31;
    u32 a4[4], b4[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      a4[k] = __shfl(ca, src + k, 32);
      b4[k] = __shfl(cb, src + k, 32);
    }
    const u32 x0 = t < 8 ? (a4[0] | (a4[1] << 6) | (a4[2] << 12) | (a4[3] << 18))
                         : (b4[0] | (b4[1] << 6) | (b4[2] << 12) | (b4[3] << 18));
    if (lane < 16) {
      dst[3 * t] = (u8)(x0 & 0xFF);
      dst[3 * t + 1] = (u8)((x0 >> 8) & 0xFF);
      dst[3 * t + 2] = (u8)((x0 >> 16) & 0xFF);
    }
  }
}

// Read this lane's two codes (elements l, l+32) of one group from packed codes at `src`; sign-extended.
template <int B>
__device__ __forceinline__ void codes_from(const u8* src, int lane, int& ca, int& cb) {
  using W = Wire<B>;
  if constexpr (B == 4) {
    const u32 x = src[lane >> 1], y = src[16 + (lane >> 1)];
    ca = (int)((x >> (4 * (lane & 1))) & 0xF);
    cb = (int)((y >> (4 * (lane & 1))) & 0xF);
  } else {
    // element e lives in triple e/4 at bit offset 6*(e%4) of its 24-bit little-endian value
    const int ta = lane >> 2, tb = (lane + 32) >> 2;
    const u32 xa = (u32)src[3 * ta] | ((u32)src[3 * ta + 1] << 8) | ((u32)src[3 * ta + 2] << 16);
    const u32 xb = (u32)src[3 * tb] | ((u32)src[3 * tb + 1] << 8) | ((u32)src[3 * tb + 2] << 16);
    ca = (int)((xa >> (6 * (lane & 3))) & W::mask);
    cb = (int)((xb >> (6 * (lane & 3))) & W::mask);
  }
  if (ca & (1 << (B - 1))) ca -= 1 << B;
  if (cb & (1 << (B - 1))) cb -= 1 << B;
}

// ---------------------------------------------------------------- handshake (r9k_ar.hip semantics, drain 4 / acq 2)
__device__ __forceinline__ void publish(u32* flag, u32 val) {
  __hip_atomic_store(flag, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void wait_for(u32* flag, u32 target) {
  while (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < target) __builtin_amdgcn_s_sleep(1);
}

// Per-block sequence step: returns s (this call's sequence number); buffer parity = s & 1.
__device__ __forceinline__ int seq_step(int* seq, int b) {
  __shared__ int s_seq;
  if (threadIdx.x == 0) s_seq = seq[b]++;
  __syncthreads();
  return s_seq;
}

// Coalesced copy of `nbytes` (multiple of 16) from LDS to global, whole block.
__device__ __forceinline__ void lds_to_global16(const u8* lds, u8* g, int nbytes) {
  for (int i = threadIdx.x * 16; i < nbytes; i += NT * 16)
    *reinterpret_cast<uint4*>(g + i) = *reinterpret_cast<const uint4*>(lds + i);
}

// Block-level pack of up to GPB groups starting at group g0 of `x` (bf16, rotated + quantised here) into LDS
// [codes GPB*cbytes][scales GPB*2]. Returns nothing; every thread participates. Groups >= ngroups are skipped.
template <int B>
__device__ __forceinline__ void pack_block(const uint16_t* __restrict__ x, long g0, long ngroups, u8* lds_codes,
                                           uint16_t* lds_scales) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
#pragma unroll
  for (int i = 0; i < GPW; i++) {
    const int gl = wave * GPW + i;                      // group within the block
    const long g = g0 + gl;
    if (g >= ngroups) break;
    float a = bf2f(x[g * GROUP + lane]), b = bf2f(x[g * GROUP + 32 + lane]);
    wht64(a, b, lane);
    a *= 0.125f; b *= 0.125f;
    const float amax = wave_amax(a, b);
    u32 ca, cb; float scale;
    quant2<B>(a, b, amax, ca, cb, scale);
    codes_to_lds<B>(ca, cb, lane, lds_codes + gl * Wire<B>::cbytes);
    if (lane == 0) lds_scales[gl] = f2bf(scale);
  }
}

// ---------------------------------------------------------------- kernels
// K1 / P2: pack `ngroups` groups of `x` and push into the peer's receive area (packed layout), then handshake.
// Grid: nblocks; each block owns a contiguous range of block-iterations (GPB groups each).
template <int B>
__global__ void __launch_bounds__(NT) ar4_pack_push_kernel(
    const uint16_t* __restrict__ x, long ngroups, u8* __restrict__ peer_recv, long recv_stride /*bytes per parity*/,
    u32* __restrict__ peer_flags, u32* __restrict__ flags, int* __restrict__ seq) {
  __shared__ __attribute__((aligned(16))) u8 lds_codes[GPB * Wire<B>::cbytes];
  __shared__ __attribute__((aligned(16))) uint16_t lds_scales[GPB];
  const int b = blockIdx.x, nb = gridDim.x;
  const int s = seq_step(seq, b);
  u8* dst = peer_recv + (long)(s & 1) * recv_stride;
  u8* dst_scales = dst + codes_bytes(ngroups, Wire<B>::cbytes);
  const long iters = (ngroups + GPB - 1) / GPB;
  for (long it = b; it < iters; it += nb) {
    const long g0 = it * GPB;
    const int gn = (int)min((long)GPB, ngroups - g0);
    pack_block<B>(x, g0, ngroups, lds_codes, lds_scales);
    __syncthreads();
    // codes: gn * cbytes bytes; when gn == GPB this is a multiple of 16 (64*32 / 64*48); a short tail is padded
    // by the layout (codes_bytes aligns the whole region), so writing the full 16-byte-rounded span is safe as long
    // as it stays inside the region: the tail block is the last one, and the region has align16 slack.
    lds_to_global16(lds_codes, dst + g0 * Wire<B>::cbytes, (int)align16((long)gn * Wire<B>::cbytes));
    // scales: gn * 2 bytes; same argument for the 16-byte rounding (region has align16 slack; tail is last)
    lds_to_global16(reinterpret_cast<const u8*>(lds_scales), dst_scales + g0 * 2, (int)align16((long)gn * 2));
    __syncthreads();
  }
  // publish then wait: our writes to the peer must be visible before it consumes them, and vice versa
  __syncthreads();
  if (threadIdx.x == 0) {
    publish(&peer_flags[b], (u32)(s + 1));
    wait_for(&flags[b], (u32)(s + 1));
  }
  __syncthreads();
}

// R1: partial = own (bf16, rotated here) + received codes*scale, inverse-rotated back to bf16.
template <int B>
__global__ void __launch_bounds__(NT) ar4_reduce_bf16_kernel(
    const uint16_t* __restrict__ own, const u8* __restrict__ recv_base, long recv_stride, const int* __restrict__ seq_of,
    int seq_block, long ngroups, uint16_t* __restrict__ out) {
  // the receive parity is the one the matching push used: seq_of[seq_block] was incremented by that push, so the
  // parity of the LAST push is (seq_of[seq_block] - 1) & 1
  const int par = (seq_of[seq_block] - 1) & 1;
  const u8* recv = recv_base + (long)par * recv_stride;
  const u8* recv_scales = recv + codes_bytes(ngroups, Wire<B>::cbytes);
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  for (long g = (long)blockIdx.x * NWAVE + wave; g < ngroups; g += (long)gridDim.x * NWAVE) {
    float a = bf2f(own[g * GROUP + lane]), b = bf2f(own[g * GROUP + 32 + lane]);
    wht64(a, b, lane);
    a *= 0.125f; b *= 0.125f;
    int ca, cb;
    codes_from<B>(recv + g * Wire<B>::cbytes, lane, ca, cb);
    const float sc = bf2f(reinterpret_cast<const uint16_t*>(recv_scales)[g]);
    a += (float)ca * sc;
    b += (float)cb * sc;
    wht64(a, b, lane);                                  // inverse
    out[g * GROUP + lane] = f2bf(a * 0.125f);
    out[g * GROUP + 32 + lane] = f2bf(b * 0.125f);
  }
}

// R2 + P3: final quarter = own partial quarter (rotated) + received codes; quantised into `local_packed` (kept for
// our own decode) and pushed to the cross partner's receive area; handshake.
template <int B>
__global__ void __launch_bounds__(NT) ar4_reduce_pack_push_kernel(
    const uint16_t* __restrict__ own, const u8* __restrict__ recv_base, long recv_stride, const int* __restrict__ seq_of,
    int seq_block, long ngroups, u8* __restrict__ local_packed, u8* __restrict__ peer_recv, long peer_stride,
    u32* __restrict__ peer_flags, u32* __restrict__ flags, int* __restrict__ seq) {
  __shared__ __attribute__((aligned(16))) u8 lds_codes[GPB * Wire<B>::cbytes];
  __shared__ __attribute__((aligned(16))) uint16_t lds_scales[GPB];
  const int b = blockIdx.x, nb = gridDim.x;
  const int s = seq_step(seq, b);
  const int par = (seq_of[seq_block] - 1) & 1;
  const u8* recv = recv_base + (long)par * recv_stride;
  const u8* recv_scales = recv + codes_bytes(ngroups, Wire<B>::cbytes);
  u8* dst = peer_recv + (long)(s & 1) * peer_stride;
  const long cb_all = codes_bytes(ngroups, Wire<B>::cbytes);
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const long iters = (ngroups + GPB - 1) / GPB;
  for (long it = b; it < iters; it += nb) {
    const long g0 = it * GPB;
    const int gn = (int)min((long)GPB, ngroups - g0);
#pragma unroll
    for (int i = 0; i < GPW; i++) {
      const int gl = wave * GPW + i;
      const long g = g0 + gl;
      if (g >= ngroups) break;
      float a = bf2f(own[g * GROUP + lane]), bb = bf2f(own[g * GROUP + 32 + lane]);
      wht64(a, bb, lane);
      a *= 0.125f; bb *= 0.125f;
      int ca, cbb;
      codes_from<B>(recv + g * Wire<B>::cbytes, lane, ca, cbb);
      const float sc = bf2f(reinterpret_cast<const uint16_t*>(recv_scales)[g]);
      a += (float)ca * sc;
      bb += (float)cbb * sc;
      const float amax = wave_amax(a, bb);
      u32 qa, qb; float scale;
      quant2<B>(a, bb, amax, qa, qb, scale);
      codes_to_lds<B>(qa, qb, lane, lds_codes + gl * Wire<B>::cbytes);
      if (lane == 0) lds_scales[gl] = f2bf(scale);
    }
    __syncthreads();
    const int ncb = (int)align16((long)gn * Wire<B>::cbytes), nsb = (int)align16((long)gn * 2);
    lds_to_global16(lds_codes, local_packed + g0 * Wire<B>::cbytes, ncb);
    lds_to_global16(lds_codes, dst + g0 * Wire<B>::cbytes, ncb);
    lds_to_global16(reinterpret_cast<const u8*>(lds_scales), local_packed + cb_all + g0 * 2, nsb);
    lds_to_global16(reinterpret_cast<const u8*>(lds_scales), dst + cb_all + g0 * 2, nsb);
    __syncthreads();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    publish(&peer_flags[b], (u32)(s + 1));
    wait_for(&flags[b], (u32)(s + 1));
  }
  __syncthreads();
}

// D3 / D4 / own: decode `ngroups` packed groups at `packed` into out (bf16).
template <int B>
__global__ void __launch_bounds__(NT) ar4_decode_kernel(
    const u8* __restrict__ packed_base, long stride, const int* __restrict__ seq_of, int seq_block, int use_parity,
    long ngroups, uint16_t* __restrict__ out) {
  const u8* packed = packed_base + (use_parity ? (long)((seq_of[seq_block] - 1) & 1) * stride : 0L);
  const u8* scales = packed + codes_bytes(ngroups, Wire<B>::cbytes);
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  for (long g = (long)blockIdx.x * NWAVE + wave; g < ngroups; g += (long)gridDim.x * NWAVE) {
    int ca, cb;
    codes_from<B>(packed + g * Wire<B>::cbytes, lane, ca, cb);
    const float sc = bf2f(reinterpret_cast<const uint16_t*>(scales)[g]);
    float a = (float)ca * sc, b = (float)cb * sc;
    wht64(a, b, lane);
    out[g * GROUP + lane] = f2bf(a * 0.125f);
    out[g * GROUP + 32 + lane] = f2bf(b * 0.125f);
  }
}

// P4: forward our packed HALF to the pair partner as two quarter messages [q_lo][q_hi], each `qbytes` (multiple
// of 16): our own final quarter from `own` (plain buffer) and the cross partner's from our P3 receive area, whose
// parity is resolved on device from the P3 sequence counter; one handshake for the whole phase.
__global__ void __launch_bounds__(NT) ar4_push2_kernel(
    const u8* __restrict__ own, const u8* __restrict__ recv3_base, long recv3_stride, const int* __restrict__ seq3,
    long qbytes, int own_first, u8* __restrict__ peer_recv, long recv_stride, u32* __restrict__ peer_flags,
    u32* __restrict__ flags, int* __restrict__ seq) {
  const int b = blockIdx.x, nb = gridDim.x;
  const int s = seq_step(seq, b);
  const u8* recv3 = recv3_base + (long)((seq3[0] - 1) & 1) * recv3_stride;
  u8* dst = peer_recv + (long)(s & 1) * recv_stride;
  u8* d_own = dst + (own_first ? 0 : qbytes);
  u8* d_fwd = dst + (own_first ? qbytes : 0);
  const long n16 = qbytes / 16;
  for (long i = (long)b * NT + threadIdx.x; i < n16; i += (long)nb * NT) {
    reinterpret_cast<uint4*>(d_own)[i] = reinterpret_cast<const uint4*>(own)[i];
    reinterpret_cast<uint4*>(d_fwd)[i] = reinterpret_cast<const uint4*>(recv3)[i];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    publish(&peer_flags[b], (u32)(s + 1));
    wait_for(&flags[b], (u32)(s + 1));
  }
  __syncthreads();
}

}  // namespace

// ---------------------------------------------------------------- C entry points (pointers as integers)
extern "C" int r9k_ar4_group() { return GROUP; }
extern "C" int r9k_ar4_max_blocks() { return MAX_BLOCKS; }
extern "C" int r9k_ar4_cbytes(int bits) { return bits == 4 ? Wire<4>::cbytes : bits == 6 ? Wire<6>::cbytes : -1; }
// bytes of a packed message of `ngroups` groups (codes region + scales region, each 16-byte aligned)
extern "C" long r9k_ar4_packed_bytes(long ngroups, int bits) {
  const int cb = r9k_ar4_cbytes(bits);
  return cb < 0 ? -1 : packed_bytes(ngroups, cb);
}

#define AR4_CHECK \
  if (bits != 4 && bits != 6) return -8; \
  if (nblocks <= 0 || nblocks > MAX_BLOCKS) return -2; \
  if (ngroups <= 0) return 0;

extern "C" int r9k_ar4_pack_push(long x, long ngroups, long peer_recv, long recv_stride, long peer_flags, long flags,
                                 long seq, int bits, int nblocks, long stream) {
  AR4_CHECK
  hipStream_t st = (hipStream_t)stream;
  if (bits == 4)
    ar4_pack_push_kernel<4><<<nblocks, NT, 0, st>>>((const uint16_t*)x, ngroups, (u8*)peer_recv, recv_stride,
                                                    (u32*)peer_flags, (u32*)flags, (int*)seq);
  else
    ar4_pack_push_kernel<6><<<nblocks, NT, 0, st>>>((const uint16_t*)x, ngroups, (u8*)peer_recv, recv_stride,
                                                    (u32*)peer_flags, (u32*)flags, (int*)seq);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

extern "C" int r9k_ar4_reduce_bf16(long own, long recv_base, long recv_stride, long seq_of, int seq_block, long ngroups,
                                   long out, int bits, int nblocks, long stream) {
  AR4_CHECK
  hipStream_t st = (hipStream_t)stream;
  if (bits == 4)
    ar4_reduce_bf16_kernel<4><<<nblocks, NT, 0, st>>>((const uint16_t*)own, (const u8*)recv_base, recv_stride,
                                                      (const int*)seq_of, seq_block, ngroups, (uint16_t*)out);
  else
    ar4_reduce_bf16_kernel<6><<<nblocks, NT, 0, st>>>((const uint16_t*)own, (const u8*)recv_base, recv_stride,
                                                      (const int*)seq_of, seq_block, ngroups, (uint16_t*)out);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

extern "C" int r9k_ar4_reduce_pack_push(long own, long recv_base, long recv_stride, long seq_of, int seq_block,
                                        long ngroups, long local_packed, long peer_recv, long peer_stride,
                                        long peer_flags, long flags, long seq, int bits, int nblocks, long stream) {
  AR4_CHECK
  hipStream_t st = (hipStream_t)stream;
  if (bits == 4)
    ar4_reduce_pack_push_kernel<4><<<nblocks, NT, 0, st>>>(
        (const uint16_t*)own, (const u8*)recv_base, recv_stride, (const int*)seq_of, seq_block, ngroups,
        (u8*)local_packed, (u8*)peer_recv, peer_stride, (u32*)peer_flags, (u32*)flags, (int*)seq);
  else
    ar4_reduce_pack_push_kernel<6><<<nblocks, NT, 0, st>>>(
        (const uint16_t*)own, (const u8*)recv_base, recv_stride, (const int*)seq_of, seq_block, ngroups,
        (u8*)local_packed, (u8*)peer_recv, peer_stride, (u32*)peer_flags, (u32*)flags, (int*)seq);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

// use_parity=1: `packed_base` is a double-buffered receive area, parity from seq_of[seq_block]; 0: plain buffer.
extern "C" int r9k_ar4_decode(long packed_base, long stride, long seq_of, int seq_block, int use_parity, long ngroups,
                              long out, int bits, int nblocks, long stream) {
  AR4_CHECK
  hipStream_t st = (hipStream_t)stream;
  if (bits == 4)
    ar4_decode_kernel<4><<<nblocks, NT, 0, st>>>((const u8*)packed_base, stride, (const int*)seq_of, seq_block,
                                                 use_parity, ngroups, (uint16_t*)out);
  else
    ar4_decode_kernel<6><<<nblocks, NT, 0, st>>>((const u8*)packed_base, stride, (const int*)seq_of, seq_block,
                                                 use_parity, ngroups, (uint16_t*)out);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

extern "C" int r9k_ar4_push2(long own, long recv3_base, long recv3_stride, long seq3, long qbytes, int own_first,
                             long peer_recv, long recv_stride, long peer_flags, long flags, long seq, int nblocks,
                             long stream) {
  if (qbytes % 16) return -1;
  if (nblocks <= 0 || nblocks > MAX_BLOCKS) return -2;
  if (qbytes <= 0) return 0;
  ar4_push2_kernel<<<nblocks, NT, 0, (hipStream_t)stream>>>(
      (const u8*)own, (const u8*)recv3_base, recv3_stride, (const int*)seq3, qbytes, own_first, (u8*)peer_recv,
      recv_stride, (u32*)peer_flags, (u32*)flags, (int*)seq);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}
