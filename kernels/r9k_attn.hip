// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas

// Paged causal attention for gfx1201 (RDNA4, wave32): the prefill / mixed-batch path of attn/triton3d.py.
//
// Independent implementation of paged FlashAttention (online softmax with running row max / row sum, K/V read
// block by block through a block table), written from the public algorithm, the gfx12 ISA and vLLM's Apache-2.0
// triton_unified_attention as the reference for the interface semantics. No third-party attention code consulted.
//
// Geometry: head_size 256, GQA = num_q_heads / num_kv_heads (<= 8), block_size 16, causal, no sliding window,
// bf16 queries/output. KV cache is the LBHNC per-layer view kv[num_blocks, num_kv_heads, 16, 2*256] with K in
// [..., :256] and V in [..., 256:], element strides kv_stride0 / kv_stride1 (kv_stride2 == 512, kv_stride3 == 1).
// KV dtype bf16 or fp8 e4m3 (per-(seq, kv_head) fp32 K/V descales, null => 1.0).
//
// Mapping. One workgroup = one (sequence, kv head, tile of 16 query tokens); two waves per q head of the GQA
// group, each owning one half (128) of the head dim (2*GQA waves, 12 for GQA 6). Each KV block is staged once per
// workgroup into LDS (coalesced K loads, hardware-transposed V loads; see the LDS layout below), double-buffered
// and prefetched one block ahead. Per wave, per KV block:
//   S^T[key][tok] partial = K[16 x 128] . Q^T     8 WMMA 16x16x16 bf16, A = K fragment (ds_read_b128), B = Q (VGPR)
//   S^T = own partial + partner's partial          1 KB LDS exchange + workgroup barrier
//   online softmax on S^T                         lane l owns query row l%16, keys 8*(l/16)..+8: the row max is 8
//                                                 in-lane max + one permlanex16; rescale only when a row max grew by
//                                                 > 2^8 (wave-uniform ballot), so O is rarely touched
//   O^T[d half][tok] += V^T[128 x 16] . P^T       8 WMMA, A = V^T fragment (ds_read_b128 of the transposed stage),
//                                                 B = P^T = the S^T accumulator as-is
//   publish the next block's stage, barrier
// The WMMA D layout (lane l: column l%16, rows 8*(l/16)..+8) keeps each lane on one query token for S, P and O,
// so max / sum / rescale are per-lane scalars. Halving the head dim per wave keeps Q (32 VGPR) and O (64 VGPR)
// resident (a full-head wave needs 192 for Q + O and spills; measured). ~185 VGPRs, 44 KB LDS at GQA 6.
//
// Measured on gfx1201 (tests/test_attn_r9k.py, 12 q / 2 kv heads): 4096-token causal prefill 1.21 ms bf16 /
// 1.13 ms fp8 KV (libr4d 1.26 / 1.21, stock unified attention 5.9), 2000 queries over a 9000 context 2.49 / 2.14
// (libr4d 2.35 / 1.90). Where the time goes (ablation): the kernel is LDS-bandwidth-bound (~136 KB of LDS traffic
// per block-step against 1536 WMMA cycles at 128 B/clk per WGP), not compute- or DRAM-bound; direct-from-cache
// fragment loads were worse (16 cache lines per 512 B fragment saturate the vector-memory path).
//
// Causal mask: query token t of the q_len chunk sits at context position p = seq_len - q_len + t and attends keys
// <= p. Blocks entirely below the diagonal of the tile skip the mask; the (at most two) diagonal blocks mask per
// element. Rows past q_len in a partial tile are clamped to the last row and not stored.
// Tiles are launched heaviest-first (reverse tile index) for load balance under the causal triangle.
// No device printf/assert (the build enforces hostcall-free objects).
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

typedef __bf16 v8bf16 __attribute__((ext_vector_type(8)));
typedef short v8s __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef float v2f __attribute__((ext_vector_type(2)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef unsigned v4u __attribute__((ext_vector_type(4)));
typedef __bf16 v2bf16 __attribute__((ext_vector_type(2)));

constexpr int HD = 256;            // head size
constexpr int HH = HD / 2;         // head-dim half per wave
constexpr int BS = 16;             // KV block size (keys per block)
constexpr int ROW = 2 * HD;        // elements per (block, kv head, slot): K then V
constexpr int DT = HH / 16;        // 8 d-tiles of 16 per wave
constexpr int MAX_GQA = 8;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float RESCALE_THRESH = 8.f;
// Build-time options (measured on gfx1201, 4096-token causal prefill: together ~3% faster in bf16, neutral in fp8;
// off by default to keep the exchange exact):
#ifndef R9K_ATTN_XH
#define R9K_ATTN_XH 0       // exchange the S^T partials as fp16 (halves that LDS traffic; ~2^-11 relative on S)
#endif
#ifndef R9K_ATTN_NBUF
#define R9K_ATTN_NBUF 2     // staged KV buffers: 2 = two barriers per block, 3 = one barrier (needs XH for LDS space)
#endif
static_assert(R9K_ATTN_NBUF == 2 || (R9K_ATTN_NBUF == 3 && R9K_ATTN_XH), "3 buffers need the fp16 exchange");   // log2 domain: rescale O when the row max grows by more than 2^8

__device__ __forceinline__ float lane_swap16(float x) {
  // value from lane (l + 16) % 32: the other half of the wave that holds the same query row
  return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(x), __float_as_int(x), 0x76543210, 0xFEDCBA98,
                                                     false, false));
}

// Workgroup barrier that only drains LDS traffic (__syncthreads would also wait for the in-flight K/V prefetch).
__device__ __forceinline__ void lds_barrier() {
  asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory");
}

__device__ __forceinline__ v8bf16 ld_frag(const uint16_t* p) {
  return *reinterpret_cast<const v8bf16*>(p);
}

__device__ __forceinline__ v8bf16 ld_frag_tr(const uint16_t* p) {
  v8s r = __builtin_amdgcn_global_load_tr_b128_v8i16((v8s*)p);
  return __builtin_bit_cast(v8bf16, r);
}

// gfx1201 has no v_cvt_pk_bf16_f32: the generic fp32 -> bf16 conversion is a ~6-op software sequence with a NaN
// check per element. These two are the cheap forms needed here.
// Two fp32 -> packed bf16 pair by taking the high halves (v_perm_b32): exact when the values carry <= 7 mantissa
// bits, else truncation.
__device__ __forceinline__ unsigned pack_hi16(float lo, float hi) {
  return __builtin_amdgcn_perm(__float_as_uint(hi), __float_as_uint(lo), 0x07060302u);
}

// fp32 -> bf16 round-to-nearest-even for finite inputs (P in [0, 2^8]): bfe + add3 per element + pack
__device__ __forceinline__ v8bf16 f32x8_to_bf16_rne(v8f f) {
  unsigned u[8];
#pragma unroll
  for (int e = 0; e < 8; e++) {
    const unsigned x = __float_as_uint(f[e]);
    u[e] = x + 0x7FFFu + ((x >> 16) & 1u);
  }
  const v4u r = {pack_hi16(__uint_as_float(u[0]), __uint_as_float(u[1])), pack_hi16(__uint_as_float(u[2]), __uint_as_float(u[3])),
                 pack_hi16(__uint_as_float(u[4]), __uint_as_float(u[5])), pack_hi16(__uint_as_float(u[6]), __uint_as_float(u[7]))};
  return __builtin_bit_cast(v8bf16, r);
}

// fp32 -> bf16 by truncation (P in [0, 2^8]: <= 2^-8 relative, and the row sum is taken over the same truncated
// values, so the normalization is self-consistent)
__device__ __forceinline__ v8bf16 f32x8_to_bf16_trunc(v8f f) {
  const v4u r = {pack_hi16(f[0], f[1]), pack_hi16(f[2], f[3]), pack_hi16(f[4], f[5]), pack_hi16(f[6], f[7])};
  return __builtin_bit_cast(v8bf16, r);
}

// sum of 8 packed bf16 (v_dot2_f32_bf16 against ones)
__device__ __forceinline__ float bf16x8_sum(v8bf16 v, float acc) {
  const v2bf16 one = {(__bf16)1.f, (__bf16)1.f};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const v2bf16 pr = {v[2 * i], v[2 * i + 1]};
    acc = __builtin_amdgcn_fdot2_f32_bf16(pr, one, acc, false);
  }
  return acc;
}

// fp8 e4m3 KV: 8 fp8 -> 8 bf16. v_cvt_pk_f32_fp8 is exact and e4m3 has 3 mantissa bits, so truncating the fp32 to
// its high half is exact as well (zero / denormal / NaN handled by the hardware convert). The descale is folded
// into the softmax scale for K and into the output normalizer for V, so no per-element multiply here.
__device__ __forceinline__ v8bf16 fp8x8_to_bf16(v2i w) {
  const v2f a = __builtin_amdgcn_cvt_pk_f32_fp8(w[0], false), b = __builtin_amdgcn_cvt_pk_f32_fp8(w[0], true);
  const v2f c = __builtin_amdgcn_cvt_pk_f32_fp8(w[1], false), d = __builtin_amdgcn_cvt_pk_f32_fp8(w[1], true);
  const v4u r = {pack_hi16(a[0], a[1]), pack_hi16(b[0], b[1]), pack_hi16(c[0], c[1]), pack_hi16(d[0], d[1])};
  return __builtin_bit_cast(v8bf16, r);
}

// LDS staging of one KV block, shared by the 2*GQA waves of the workgroup (a fragment load straight from the cache
// touches 16 lines for 512 B, and 12 waves each re-reading the block made the vector-memory path the bottleneck;
// with staging each block is read once per workgroup, with coalesced K loads and hardware-transposed V loads).
//   K:   [16 keys][512 B] row-major, the 16 B chunk index rotated by the row ((chunk + row) & 31) so the 8 lanes
//        of a ds_read_b128 group (rows 0..7 at one chunk) hit 8 distinct bank slots.
//   V^T: [256 d][32 B] (16 keys of one d), chunk (key half) XOR-swizzled with ((d >> 2) & 1) for the same reason.
// A K fragment kk of head-dim half d0 and a V^T fragment dt are then one ds_read_b128 each, in WMMA layout.
constexpr int LDS_K = 16 * 512;                 // bytes per staged K block
constexpr int LDS_V = 256 * 32;                 // bytes per staged V^T block
constexpr int LDS_BUF = LDS_K + LDS_V;          // 16 KB per block, double-buffered

__device__ __forceinline__ int lds_k_off(int row, int chunk) { return row * 512 + (((chunk + row) & 31) << 4); }
__device__ __forceinline__ int lds_v_off(int d, int chunk) { return LDS_K + d * 32 + ((chunk ^ ((d >> 2) & 1)) << 4); }

// KV element type traits: bf16 (uint16_t) or fp8 e4m3 (uint8_t). Each provides the per-thread stage of one block
// (global -> registers) and its store into the LDS layout above. NT threads, NW waves.
template <typename T, int NW> struct KvT;
template <int NW> struct KvT<uint16_t, NW> {
  static constexpr int NT = NW * 32;
  static constexpr int NK = (512 + NT - 1) / NT;          // 16 B K chunks per thread (512 per block)
  static constexpr int NV = (16 + NW - 1) / NW;           // V d-tiles per wave (16 per block)
  struct Stage { v8bf16 k[NK]; v8bf16 v[NV]; };
  static __device__ __forceinline__ void load(Stage& st, const uint16_t* kbase, int t, int wave, int lane) {
#pragma unroll
    for (int i = 0; i < NK; i++) {
      const int c = t + i * NT;
      if (c < 512) st.k[i] = ld_frag(kbase + (c >> 5) * ROW + (c & 31) * 8);
    }
    const uint16_t* vb = kbase + HD + (8 * (lane >> 4) + (lane & 7)) * ROW + 8 * ((lane >> 3) & 1);
#pragma unroll
    for (int i = 0; i < NV; i++) {
      const int dt = wave + i * NW;
      if (dt < 16) st.v[i] = ld_frag_tr(vb + 16 * dt);
    }
  }
  static __device__ __forceinline__ void store(const Stage& st, unsigned char* buf, int t, int wave, int lane) {
#pragma unroll
    for (int i = 0; i < NK; i++) {
      const int c = t + i * NT;
      if (c < 512) *reinterpret_cast<v8bf16*>(buf + lds_k_off(c >> 5, c & 31)) = st.k[i];
    }
#pragma unroll
    for (int i = 0; i < NV; i++) {
      const int dt = wave + i * NW;
      if (dt < 16) *reinterpret_cast<v8bf16*>(buf + lds_v_off(16 * dt + (lane & 15), lane >> 4)) = st.v[i];
    }
  }
};
template <int NW> struct KvT<uint8_t, NW> {
  static constexpr int NT = NW * 32;
  static constexpr int NK = (256 + NT - 1) / NT;          // 16 B fp8 K chunks per thread (256 per block)
  static constexpr int NV = (16 + NW - 1) / NW;
  struct Stage { v4u k[NK]; v2i v[NV]; };
  static __device__ __forceinline__ void load(Stage& st, const uint8_t* kbase, int t, int wave, int lane) {
#pragma unroll
    for (int i = 0; i < NK; i++) {
      const int c = t + i * NT;
      if (c < 256) st.k[i] = *reinterpret_cast<const v4u*>(kbase + (c >> 4) * ROW + (c & 15) * 16);
    }
    const uint8_t* vb = kbase + HD + (8 * (lane >> 4) + 4 * ((lane >> 3) & 1) + (lane & 3)) * ROW + 8 * ((lane >> 2) & 1);
#pragma unroll
    for (int i = 0; i < NV; i++) {
      const int dt = wave + i * NW;
      if (dt < 16) st.v[i] = __builtin_amdgcn_global_load_tr_b64_v2i32((v2i*)(vb + 16 * dt));
    }
  }
  static __device__ __forceinline__ void store(const Stage& st, unsigned char* buf, int t, int wave, int lane) {
#pragma unroll
    for (int i = 0; i < NK; i++) {
      const int c = t + i * NT;
      if (c < 256) {
        const v2i lo = {(int)st.k[i][0], (int)st.k[i][1]}, hi = {(int)st.k[i][2], (int)st.k[i][3]};
        *reinterpret_cast<v8bf16*>(buf + lds_k_off(c >> 4, 2 * (c & 15))) = fp8x8_to_bf16(lo);
        *reinterpret_cast<v8bf16*>(buf + lds_k_off(c >> 4, 2 * (c & 15) + 1)) = fp8x8_to_bf16(hi);
      }
    }
#pragma unroll
    for (int i = 0; i < NV; i++) {
      const int dt = wave + i * NW;
      if (dt < 16) *reinterpret_cast<v8bf16*>(buf + lds_v_off(16 * dt + (lane & 15), lane >> 4)) = fp8x8_to_bf16(st.v[i]);
    }
  }
};

template <int GQA, typename KV>
__global__ void __launch_bounds__(GQA * 64) r9k_attn_prefill_kernel(
    const uint16_t* __restrict__ q, const KV* __restrict__ kv, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens, uint16_t* __restrict__ out, const float* __restrict__ k_descale,
    const float* __restrict__ v_descale, int q_len, int hq, int hkv, int max_blocks, long kv_stride0,
    long kv_stride1, float scale_log2) {
  constexpr int NW = 2 * GQA;
  typedef KvT<KV, NW> Kv;
  constexpr int XB = R9K_ATTN_XH ? 2 : 4;                           // bytes per exchanged S^T element
  constexpr int XSLOTS = R9K_ATTN_NBUF == 3 ? 2 : 1;                 // exchange double-buffered with one barrier
  __shared__ __attribute__((aligned(16))) unsigned char lds[R9K_ATTN_NBUF * LDS_BUF + XSLOTS * NW * 256 * XB];
  unsigned char* xs = lds + R9K_ATTN_NBUF * LDS_BUF;                // S^T partial exchange [slot][NW][8][32]
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int t = threadIdx.x;
  const int r = lane & 15;           // query row of this lane within the tile
  const int kh = lane >> 4;          // which 8 keys / which 8-element half of a 16-wide fragment
  const int kvh = blockIdx.y;
  const int seq = blockIdx.z;
  const int head = kvh * GQA + (wave >> 1);
  const int d0 = (wave & 1) * HH;    // this wave's head-dim half
  const int tile = gridDim.x - 1 - blockIdx.x;
  const int t0 = tile * 16;
  const int seq_len = seq_lens[seq];
  const int base_pos = seq_len - q_len;                  // context position of query token 0
  const int t_last = min(t0 + 15, q_len - 1);
  const int tq = min(t0 + r, q_len - 1);
  const int p = base_pos + tq;                           // this lane's query position
  const int nblk = (base_pos + t_last) / BS + 1;         // KV blocks this tile touches (workgroup-uniform)
  const int p_min = base_pos + t0;
  const int b_mask = p_min >= 15 ? (p_min - 15) / BS + 1 : 0;   // first block needing the causal mask
  if (k_descale) scale_log2 *= k_descale[seq * hkv + kvh];        // fp8: K descale folded into the softmax scale

  // Q fragments: lane holds Q[tq][d0 + 16*kk + 8*kh .. +8] as the WMMA B operand (column = query row)
  const uint16_t* qrow = q + ((long)(seq * q_len + tq) * hq + head) * HD + d0 + 8 * kh;
  v8bf16 qf[DT];
#pragma unroll
  for (int kk = 0; kk < DT; kk++) qf[kk] = ld_frag(qrow + 16 * kk);

  v8f o[DT];
#pragma unroll
  for (int dt = 0; dt < DT; dt++) o[dt] = (v8f)(0.f);
  float m = -INFINITY, lsum = 0.f;

  const int* bt = block_table + (long)seq * max_blocks;
  const long head_off = (long)kvh * kv_stride1;
  // LDS fragment offsets of this lane (swizzled): K chunk 2*kk + kh of row r, V^T row d0 + 16*dt + r, chunk kh
  // (the V^T swizzle bit ((d >> 2) & 1) does not depend on dt, so + 512 * dt is exact)
  int kfo[DT];
#pragma unroll
  for (int kk = 0; kk < DT; kk++) kfo[kk] = lds_k_off(r, (d0 >> 3) + 2 * kk + kh);
  const int vfo = lds_v_off(d0 + r, kh);

  typename Kv::Stage st;
  Kv::load(st, kv + (long)__builtin_amdgcn_readfirstlane(bt[0]) * kv_stride0 + head_off, t, wave, lane);
  Kv::store(st, lds, t, wave, lane);
  if (R9K_ATTN_NBUF == 3 && nblk > 1) {           // three buffers: block 1 is staged in the prologue too
    Kv::load(st, kv + (long)__builtin_amdgcn_readfirstlane(bt[1]) * kv_stride0 + head_off, t, wave, lane);
    Kv::store(st, lds + LDS_BUF, t, wave, lane);
  }
  lds_barrier();

  unsigned char* xw = xs + (wave * 256 + lane) * XB;
  const unsigned char* xr = xs + ((wave ^ 1) * 256 + lane) * XB;
  for (int b = 0; b < nblk; b++) {
    const unsigned char* buf = lds + (R9K_ATTN_NBUF == 3 ? b % 3 : (b & 1)) * LDS_BUF;
    // stage block b + NBUF - 1 (past the end: re-read the last block, harmless, keeps the loop branch-free)
    const int bn = min(b + R9K_ATTN_NBUF - 1, nblk - 1);
    Kv::load(st, kv + (long)__builtin_amdgcn_readfirstlane(bt[bn]) * kv_stride0 + head_off, t, wave, lane);
    // partial S^T over this wave's head-dim half
    v8f s = (v8f)(0.f);
#pragma unroll
    for (int kk = 0; kk < DT; kk++) {
      const v8bf16 kf = *reinterpret_cast<const v8bf16*>(buf + kfo[kk]);
      s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(kf, qf[kk], s);
    }
    if (R9K_ATTN_NBUF == 3) {
      // one barrier per block: buffer (b+2)%3 held block b-1, which every wave finished before the previous barrier
      Kv::store(st, lds + ((b + 2) % 3) * LDS_BUF, t, wave, lane);
    }
    // exchange partials with the partner wave (other head-dim half)
    const int xslot = (XSLOTS == 2 ? (b & 1) : 0) * NW * 256 * XB;
    {
      if (R9K_ATTN_XH) {
#pragma unroll
        for (int e = 0; e < 8; e++) *reinterpret_cast<_Float16*>(xw + xslot + e * 32 * XB) = (_Float16)s[e];
      } else {
#pragma unroll
        for (int e = 0; e < 8; e++) *reinterpret_cast<float*>(xw + xslot + e * 32 * XB) = s[e];
      }
      lds_barrier();
      if (R9K_ATTN_XH) {
#pragma unroll
        for (int e = 0; e < 8; e++) s[e] += (float)*reinterpret_cast<const _Float16*>(xr + xslot + e * 32 * XB);
      } else {
#pragma unroll
        for (int e = 0; e < 8; e++) s[e] += *reinterpret_cast<const float*>(xr + xslot + e * 32 * XB);
      }
    }
    // lane l: S^T[keys 8*kh + e][row r] -> raw scores of query row r; the softmax scale is applied inside the
    // exp argument (fma), so the running max m is kept in log2 units and mloc in raw units
    float mloc = -INFINITY;
    if (b >= b_mask) {
      const int key0 = b * BS + 8 * kh;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        s[e] = (key0 + e <= p) ? s[e] : -INFINITY;
        mloc = fmaxf(mloc, s[e]);
      }
    } else {
#pragma unroll
      for (int e = 0; e < 8; e++) mloc = fmaxf(mloc, s[e]);
    }
    mloc = fmaxf(mloc, lane_swap16(mloc));
    // lazy rescale: the reference max only moves when the true max exceeds it by RESCALE_THRESH (p <= 2^8)
    if (__builtin_amdgcn_ballot_w32(mloc * scale_log2 > m + RESCALE_THRESH)) {
      const float m_new = mloc * scale_log2 > m + RESCALE_THRESH ? mloc * scale_log2 : m;
      const float alpha = __builtin_amdgcn_exp2f(m - m_new);       // 0 on the first block (m = -inf), 1 if unchanged
      m = m_new;
      lsum *= alpha;
#pragma unroll
      for (int dt = 0; dt < DT; dt++)
#pragma unroll
        for (int e = 0; e < 8; e++) o[dt][e] *= alpha;
    }
    v8f pf;
#pragma unroll
    for (int e = 0; e < 8; e++) pf[e] = __builtin_amdgcn_exp2f(fmaf(s[e], scale_log2, -m));
    const v8bf16 pb = f32x8_to_bf16_trunc(pf);
    lsum = bf16x8_sum(pb, lsum);
    // O^T += V^T . P^T : A = V^T fragment from LDS, B = P^T (the S^T layout as produced)
#pragma unroll
    for (int dt = 0; dt < DT; dt++) {
      const v8bf16 vf = *reinterpret_cast<const v8bf16*>(buf + vfo + 512 * dt);
      o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vf, pb, o[dt]);
    }
    // two buffers: publish the staged next block; everyone has finished this block's K (before the first
    // barrier) and V (here)
    if (R9K_ATTN_NBUF == 2) {
      Kv::store(st, lds + ((b + 1) & 1) * LDS_BUF, t, wave, lane);
      lds_barrier();
    }
  }
  float inv = 1.f / (lsum + lane_swap16(lsum));
  if (v_descale) inv *= v_descale[seq * hkv + kvh];          // fp8: V descale on the normalized output
  if (t0 + r < q_len) {
    uint16_t* orow = out + ((long)(seq * q_len + tq) * hq + head) * HD + d0 + 8 * kh;
#pragma unroll
    for (int dt = 0; dt < DT; dt++) {
      v8f v = o[dt];
#pragma unroll
      for (int e = 0; e < 8; e++) v[e] *= inv;
      *reinterpret_cast<v8bf16*>(orow + 16 * dt) = f32x8_to_bf16_rne(v);
    }
  }
}

template <int GQA, typename KV>
int launch_prefill(const uint16_t* q, const KV* kv, const int* bt, const int* sl, uint16_t* out, const float* kd,
                   const float* vd, int nseq, int q_len, int hq, int hkv, int max_blocks, long s0, long s1, float scale,
                   hipStream_t st) {
  dim3 grid((q_len + 15) / 16, hkv, nseq), block(GQA * 64);
  r9k_attn_prefill_kernel<GQA, KV><<<grid, block, 0, st>>>(q, kv, bt, sl, out, kd, vd, q_len, hq, hkv, max_blocks, s0,
                                                           s1, scale * LOG2E);
  return (int)hipGetLastError();
}

template <typename KV>
int prefill_paged(long q_ptr, long kv_ptr, long block_table_ptr, long seq_lens_ptr, long out_ptr, long k_descale_ptr,
                  long v_descale_ptr, int nseq, int q_len, int num_q_heads, int num_kv_heads, int head_size,
                  int block_size, int max_blocks, long kv_stride0, long kv_stride1, float scale, int sliding_window,
                  long stream) {
  if (head_size != HD || block_size != BS || sliding_window != 0 || num_kv_heads <= 0) return -1;
  if (num_q_heads % num_kv_heads) return -2;
  const int gqa = num_q_heads / num_kv_heads;
  if (gqa < 1 || gqa > MAX_GQA) return -3;
  if (nseq <= 0 || q_len <= 0) return 0;
  const uint16_t* q = (const uint16_t*)q_ptr;
  const KV* kv = (const KV*)kv_ptr;
  const int* bt = (const int*)block_table_ptr;
  const int* sl = (const int*)seq_lens_ptr;
  uint16_t* out = (uint16_t*)out_ptr;
  const float* kd = (const float*)k_descale_ptr;
  const float* vd = (const float*)v_descale_ptr;
  hipStream_t st = (hipStream_t)stream;
#define R9K_ATTN_CASE(G) \
  case G: return launch_prefill<G, KV>(q, kv, bt, sl, out, kd, vd, nseq, q_len, num_q_heads, num_kv_heads, max_blocks, kv_stride0, kv_stride1, scale, st);
  switch (gqa) {
    R9K_ATTN_CASE(1) R9K_ATTN_CASE(2) R9K_ATTN_CASE(3) R9K_ATTN_CASE(4)
    R9K_ATTN_CASE(5) R9K_ATTN_CASE(6) R9K_ATTN_CASE(7)
    default: R9K_ATTN_CASE(8)
  }
#undef R9K_ATTN_CASE
}

}  // namespace

// Same positional interface as the libr4d prefill entries used by attn/triton3d.py (pointers as integers).
// sliding_window must be 0; scratch_ptr and max_ctx are unused (no scratch needed). Returns 0 on success, <0 for
// unsupported geometry, else the hip error code. k/v descale pointers may be 0 (scale 1.0) and are only read by
// the fp8 entry.
extern "C" int r9k_attn_prefill_paged(long q_ptr, long kv_ptr, long block_table_ptr, long seq_lens_ptr, long out_ptr,
                                      long k_descale_ptr, long v_descale_ptr, long scratch_ptr, int nseq, int q_len,
                                      int num_q_heads, int num_kv_heads, int head_size, int block_size, int max_blocks,
                                      long kv_stride0, long kv_stride1, float scale, int sliding_window, int max_ctx,
                                      long stream) {
  (void)scratch_ptr; (void)max_ctx;
  return prefill_paged<uint16_t>(q_ptr, kv_ptr, block_table_ptr, seq_lens_ptr, out_ptr, 0, 0, nseq, q_len, num_q_heads,
                                 num_kv_heads, head_size, block_size, max_blocks, kv_stride0, kv_stride1, scale,
                                 sliding_window, stream);
}

extern "C" int r9k_attn_prefill_paged_fp8(long q_ptr, long kv_ptr, long block_table_ptr, long seq_lens_ptr, long out_ptr,
                                          long k_descale_ptr, long v_descale_ptr, long scratch_ptr, int nseq, int q_len,
                                          int num_q_heads, int num_kv_heads, int head_size, int block_size,
                                          int max_blocks, long kv_stride0, long kv_stride1, float scale,
                                          int sliding_window, int max_ctx, long stream) {
  (void)scratch_ptr; (void)max_ctx;
  return prefill_paged<uint8_t>(q_ptr, kv_ptr, block_table_ptr, seq_lens_ptr, out_ptr, k_descale_ptr, v_descale_ptr,
                                nseq, q_len, num_q_heads, num_kv_heads, head_size, block_size, max_blocks, kv_stride0,
                                kv_stride1, scale, sliding_window, stream);
}
