// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas
//
// r9k_exl3_gemv.hip -- decode-width (M = 1..8 rows) EXL3 trellis GEMV for gfx1201 (wave32): several matrices a launch,
// the input rotation (x * suh, 128-point Hadamard) and the output rotation (128-point Hadamard, * svh) in-kernel, bf16 or
// f16 input and bf16 / f16 / f32 output with no separate casts.
//
// Third-party code: the trellis decode, the RDNA "tiles" dot core and the 128-point Hadamard warp shuffle below (marked
// EXLLAMAV3) are copied from exllamav3 v1.6.6 (https://github.com/turboderp-org/exllamav3, MIT, Copyright (c) turboderp):
// exllamav3_ext/rocm/quant/exl3_gemv_tiles_rdna.cuh, exllamav3_ext/quant/exl3_dq.cuh (fshift) and
// exllamav3_ext/quant/hadamard_inner.cuh (shuffle_had_f4x32), by way of kernels/r9k_moe_exl3.hip (package P1), whose
// adaptation (integer K only, codebooks mcg / mul1, vector-typed halves, per-row byte offsets) they keep verbatim. The
// copy lives in an anonymous namespace, so it links next to r9k_moe_exl3.hip's own. The decoded weights are bit-identical
// to ext.reconstruct (tests/test_exl3_gemv.py checks every K through r9k_exl3_gemv_dq).
//
// Rows: up to 16 a group (the row tile M, one decode per k-tile feeds every row); more rows = more groups.
//
// Problem (notes/exl3-plan.md, Phase 4 package D): the attention / dense EXL3 linears of a decode step are 1.6-6 MB a
// rank each; exllamav3's per-call path runs them at 150-300 GB/s (an ~8 us floor a call, a separate rotation kernel, the
// casts around it), and its split-K stays inside one workgroup, so a [4096 -> 1024] matrix gets 8 workgroups on a 64-CU part.
//
// Here, one launch serves a list of groups (matrix, input rows, output rows); grid = sum over groups of
// (n / 128 output tiles) x (ks input chunks of kc = k / ks). A block (4 waves x TPW 16-column tiles = 128 outputs):
//   1. rotates its m rows' input chunk into LDS: x (bf16 / f16) * suh, Hadamard over each 128-block, / sqrt(128), f16
//      (x_mode 2: the caller's f16 input is already rotated -- package B's pre-pass -- and is only copied);
//   2. streams the chunk's k-tiles of its 8 column tiles through the dot core (one trellis decode feeds all m rows,
//      f32 accumulation), writes the f32 partial [kq, row, col] to scratch;
//   3. the last of the ks blocks of an output tile to arrive (device-scope counter, reset by that block, so no memset
//      and no host sync) sums the partials in chunk order, applies the output Hadamard and svh and writes the rows in
//      the group's output dtype at the group's column offset / row stride.
// Every sum has a fixed order: deterministic, HIP-graph safe. Scratch is shared by every launch on a stream.
//
// Package F (v0.4.2, the fdot path's fixed latency; phase trace with -DR9K_GEMV_TRACE, tests/test_exl3_gemv.py
// ONLY=trace): the first R9K_GEMV_PI k-tiles of trellis and svh are loaded before the rotation and stay in flight
// across an LDS-only barrier (__syncthreads' fence waits for every outstanding global load on gfx12); the epilogue
// issues a task's partial loads together (unconditional, no branch between them) and, for m < 4, splits a row's
// chunks over the block's waves (slice sums in LDS, then summed in slice order). The WMMA path is package D's.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef float f4 __attribute__((ext_vector_type(4)));

namespace {

#ifndef R9K_GEMV_TPW
#define R9K_GEMV_TPW 2             // adjacent 16-column tiles a wave (shared input loads)
#endif
#ifndef R9K_GEMV_KW
#define R9K_GEMV_KW 1              // waves splitting a block's input chunk (1 = 4 waves a block, 2 = 8)
#endif
#ifndef R9K_GEMV_FSPLIT_M
#define R9K_GEMV_FSPLIT_M 4        // row tiles up to this take the split epilogue (above: package D's finish_rows)
#endif
#ifndef R9K_GEMV_FSPLIT
#define R9K_GEMV_FSPLIT 1          // epilogue: rows < 4 split their partial chunks over the block's waves
#endif
#ifndef R9K_GEMV_U
#define R9K_GEMV_U 0               // k-tiles whose loads go out together (0 = exllamav3's table)
#endif
#ifndef R9K_GEMV_PI
#define R9K_GEMV_PI 1              // fdot path: k-tiles a wave issues before the input rotation
#endif
#ifndef R9K_GEMV_QB
#define R9K_GEMV_QB 4              // epilogue: partial chunks a task keeps in flight
#endif
constexpr int TPW = R9K_GEMV_TPW;
constexpr bool FSPLIT = R9K_GEMV_FSPLIT;
constexpr int CW = 8 / TPW;                 // column waves: 128 outputs a block
constexpr int KW = R9K_GEMV_KW;
constexpr int WAVES = CW * KW;
constexpr int MAXG = 16;                    // groups a launch
constexpr int MAXM = 16;                    // rows a group, fdot path (the row tile: one trellis decode feeds them all)
constexpr int MAXM_MMA = 48;                // rows a group, WMMA path (three 16-row tiles)
constexpr float RS128 = 0.088388347648f;    // 1/sqrt(128)

struct Frag { h2 v[2]; };

__device__ __forceinline__ h2 as_h2(uint32_t u) { return __builtin_bit_cast(h2, u); }
__device__ __forceinline__ float fdot2(h2 a, h2 b, float c) { return __builtin_amdgcn_fdot2(a, b, c, false); }

// ---------------------------------------------------------------------------------------------------------------
// EXLLAMAV3 (MIT): trellis decode, quant/exl3_dq.cuh + rocm/quant/exl3_gemv_tiles_rdna.cuh
// ---------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t fshift(const uint32_t b, const uint32_t a, int shift)
{
    uint64_t merged = ((uint64_t) a << 32) | (uint64_t) b;
    return (uint32_t) (merged >> shift);
}

__device__ __forceinline__ int uni(int x) { return __builtin_amdgcn_readfirstlane(x); }

template <typename T>
__device__ __forceinline__ T* uni_ptr(T* p)
{
    uint64_t v = (uint64_t) p;
    uint32_t lo = __builtin_amdgcn_readfirstlane((uint32_t) v);
    uint32_t hi = __builtin_amdgcn_readfirstlane((uint32_t) (v >> 32));
    return (T*) (((uint64_t) hi << 32) | (uint64_t) lo);
}

// x * C (+ D) mod 2^32 for the 16-bit x in the low (HI = false) or high half of src: two full-rate ops
template <uint32_t C, uint32_t D, bool HI>
__device__ __forceinline__ uint32_t hmul16(uint32_t src)
{
    constexpr uint32_t k_hi = C & 0xffff0000u;
    constexpr uint32_t c_lo = C & 0xffffu;
    uint32_t r, p;
    if constexpr (D == 0)
    {
        if constexpr (HI) asm ("v_pk_mul_lo_u16 %0, %1, %2" : "=v"(r) : "v"(src), "s"(k_hi));
        else asm ("v_pk_mul_lo_u16 %0, %1, %2 op_sel_hi:[0,1]" : "=v"(r) : "v"(src), "s"(k_hi));
    }
    else
    {
        constexpr uint32_t d = D;
        if constexpr (HI) asm ("v_pk_mad_u16 %0, %1, %2, %3" : "=v"(r) : "v"(src), "s"(k_hi), "s"(d));
        else asm ("v_pk_mad_u16 %0, %1, %2, %3 op_sel_hi:[0,1,1]" : "=v"(r) : "v"(src), "s"(k_hi), "s"(d));
    }
    if constexpr (HI) asm ("v_mad_u32_u16 %0, %1, %2, %3 op_sel:[1,0,0,0]" : "=v"(p) : "v"(src), "s"(c_lo), "v"(r));
    else asm ("v_mad_u32_u16 %0, %1, %2, %3" : "=v"(p) : "v"(src), "s"(c_lo), "v"(r));
    return p;
}

template <int cb, bool HI>
__device__ __forceinline__ uint32_t hash16(uint32_t src)
{
    if constexpr (cb == 1) return hmul16<0xCBAC1FEDu, 0u, HI>(src);   // mcg
    else return hmul16<0x83DCD12Du, 0u, HI>(src);                     // mul1
}

// Two hashed words -> the decoded pair (lo = word 0), bit for bit codebook.cuh decode_3inst_2<cb>
template <int cb>
__device__ __forceinline__ h2 decode_pair(uint32_t p0, uint32_t p1)
{
    if constexpr (cb == 2)
    {
        uint32_t s = __builtin_amdgcn_sad_u8(p0, 0u, 0x64006400u);
        s = __builtin_amdgcn_sad_hi_u8(p1, 0u, s);
        return __builtin_elementwise_fma(as_h2(s), as_h2(0x1eee1eeeu), as_h2(0xc931c931u));
    }
    else
    {
        p0 = (p0 & 0x8fff8fffu) ^ 0x3b603b60u;
        p1 = (p1 & 0x8fff8fffu) ^ 0x3b603b60u;
        h2 x0 = as_h2(p0), x1 = as_h2(p1);
        h2 lo = {x0.x, x1.x}, hi = {x0.y, x1.y};
        return lo + hi;
    }
}

template <int cb, bool HI0, bool HI1>
__device__ __forceinline__ h2 dq_pair(uint32_t src0, uint32_t src1)
{
    return decode_pair<cb>(hash16<cb, HI0>(src0), hash16<cb, HI1>(src1));
}

template <int cb>
__device__ __forceinline__ h2 dq_pair_lo(uint32_t w0, uint32_t w1) { return dq_pair<cb, false, false>(w0, w1); }

// Per-lane load plan: the dwords of a 16x16 tile (8 * bits dwords) lane L reads to decode its 8 weights
template <int bits> struct RawCount { static constexpr int R = (bits <= 4) ? 2 : (bits == 7 ? 8 : 4); };

template <int bits>
struct LanePlan
{
    static constexpr int R = RawCount<bits>::R;
    uint32_t off[R];     // byte offsets within the tile; off[2i] = "a" (hi), off[2i+1] = "b" (lo)
    int sh[R / 2];
};

template <int bits>
__device__ __forceinline__ void dq4_idx(int t_offset, int& i0, int& i2, int& s2)
{
    int b0 = (t_offset + 257) * bits - 16;
    int b1 = b0 + 3 * bits;
    int b2 = b1 + 16;
    i0 = (b0 / 32) % (bits * 256 / 32);
    s2 = ((b2 - 1) / 32 + 1) * 32 - b2;
    i2 = ((b2 - 1) / 32) % (bits * 256 / 32);
}

template <int bits>
__device__ __forceinline__ void dq2_idx(int t_offset, int& i0, int& i2, int& s2)
{
    int b0 = (t_offset + 257) * bits - 16;
    int b1 = b0 + 1 * bits;
    int b2 = b1 + 16;
    i0 = (b0 / 32) % (bits * 256 / 32);
    s2 = ((b2 - 1) / 32 + 1) * 32 - b2;
    i2 = ((b2 - 1) / 32) % (bits * 256 / 32);
}

template <int bits>
__device__ __forceinline__ void dq8_idx(int t_offset, int& i0, int& i2, int& s2)
{
    int b1 = (t_offset + 257) * bits;
    int b0 = b1 - 16;
    int b2 = b1 + bits * 7;
    i0 = (b0 / 32) % (bits * 256 / 32);
    s2 = ((b2 - 1) / 32 + 1) * 32 - b2;
    i2 = ((b2 - 1) / 32) % (bits * 256 / 32);
}

template <int bits>
__device__ __forceinline__ LanePlan<bits> lane_plan(int lane)
{
    LanePlan<bits> p;
    const int t_offset = lane << 3;
    if constexpr (bits == 1)
    {
        int i1 = t_offset >> 5;
        p.off[1] = i1 * 4; p.off[0] = ((i1 + 7) & 7) * 4; p.sh[0] = (~t_offset) & 24;
    }
    else if constexpr (bits == 2)
    {
        int i1 = t_offset >> 4;
        p.off[1] = i1 * 4; p.off[0] = ((i1 + 15) & 15) * 4; p.sh[0] = ((~t_offset) & 8) << 1;
    }
    else if constexpr (bits == 4)
    {
        int i1 = t_offset >> 3;
        p.off[1] = i1 * 4; p.off[0] = ((i1 + 31) & 31) * 4; p.sh[0] = 20;
    }
    else if constexpr (bits == 3)
    {
        int i0, i2, s2;
        dq8_idx<bits>(t_offset, i0, i2, s2);
        p.off[0] = i0 * 4; p.off[1] = i2 * 4; p.sh[0] = s2;
    }
    else if constexpr (bits == 7)
    {
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int i = 0; i < 2; ++i)
            {
                int i0, i2, s2;
                dq2_idx<bits>(t_offset + 4 * h + 2 * i, i0, i2, s2);
                p.off[(h * 2 + i) * 2] = i0 * 4; p.off[(h * 2 + i) * 2 + 1] = i2 * 4; p.sh[h * 2 + i] = s2;
            }
    }
    else
    {
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            int i0, i2, s2;
            dq4_idx<bits>(t_offset + 4 * h, i0, i2, s2);
            p.off[h * 2] = i0 * 4; p.off[h * 2 + 1] = i2 * 4; p.sh[h] = s2;
        }
    }
    return p;
}

template <int cb>
__device__ __forceinline__ void dq8_2bits(uint32_t a, uint32_t b, int shift, Frag& f0, Frag& f1)
{
    b = __funnelshift_r(b, a, shift);
    f0.v[0] = dq_pair_lo<cb>(b >> 14, b >> 12);
    f0.v[1] = dq_pair_lo<cb>(b >> 10, b >> 8);
    f1.v[0] = dq_pair_lo<cb>(b >> 6,  b >> 4);
    f1.v[1] = dq_pair_lo<cb>(b >> 2,  b);
}

template <int cb>
__device__ __forceinline__ void dq8_1bit(uint32_t a, uint32_t b, int shift, Frag& f0, Frag& f1)
{
    b = __funnelshift_r(b, a, shift);
    f0.v[0] = dq_pair_lo<cb>(b >> 7, b >> 6);
    f0.v[1] = dq_pair_lo<cb>(b >> 5, b >> 4);
    f1.v[0] = dq_pair_lo<cb>(b >> 3, b >> 2);
    f1.v[1] = dq_pair_lo<cb>(b >> 1, b);
}

template <int cb>
__device__ __forceinline__ void dq8_4bits(uint32_t a, uint32_t b, Frag& f0, Frag& f1)
{
    uint32_t s = __funnelshift_r(b, a, 20);
    f0.v[0] = dq_pair<cb, false, false>(s >> 8, s >> 4);
    f0.v[1] = dq_pair<cb, false, true >(s,      b);
    f1.v[0] = dq_pair<cb, false, false>(b >> 12, b >> 8);
    f1.v[1] = dq_pair<cb, false, false>(b >> 4,  b);
}

template <int bits, int cb>
__device__ __forceinline__ void dq8_3bits(uint32_t a, uint32_t b, int s2, Frag& f0, Frag& f1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    w7 = fshift(b, a, s2);
    w6 = w7 >> bits;
    w5 = w6 >> bits;
    w4 = w5 >> bits;
    w3 = fshift(b, a, s2 + bits * 4);
    w2 = w3 >> bits;
    w1 = w2 >> bits;
    w0 = w1 >> bits;
    f0.v[0] = dq_pair_lo<cb>(w0, w1);
    f0.v[1] = dq_pair_lo<cb>(w2, w3);
    f1.v[0] = dq_pair_lo<cb>(w4, w5);
    f1.v[1] = dq_pair_lo<cb>(w6, w7);
}

template <int bits, int cb>
__device__ __forceinline__ void dq4_fast(uint32_t a, uint32_t b, int s2, Frag& f)
{
    uint32_t w3 = fshift(b, a, s2);
    uint32_t w2 = fshift(b, a, s2 + bits);
    uint32_t w1 = fshift(b, a, s2 + bits * 2);
    uint32_t w0 = fshift(b, a, s2 + bits * 3);
    f.v[0] = dq_pair_lo<cb>(w0, w1);
    f.v[1] = dq_pair_lo<cb>(w2, w3);
}

template <int bits, int cb>
__device__ __forceinline__ h2 dq2_fast(uint32_t a, uint32_t b, int s2)
{
    uint32_t w1 = fshift(b, a, s2);
    uint32_t w0 = fshift(b, a, s2 + bits);
    return dq_pair_lo<cb>(w0, w1);
}

// Lane L's 8 weights of a 16x16 (in x out) tile: f0 = (B[r0][cA], B[r0+1][cA]), (B[r0+8][cA], B[r0+9][cA]),
// f1 the same rows at cB = cA + 8; r0 = (L % 4) * 2, cA = (L / 8) * 2 + ((L >> 2) & 1)
template <int bits, int cb>
__device__ __forceinline__ void dq_tile(const uint32_t* raw, const LanePlan<bits>& lp, Frag& f0, Frag& f1)
{
    if constexpr (bits == 1) dq8_1bit<cb>(raw[0], raw[1], lp.sh[0], f0, f1);
    else if constexpr (bits == 2) dq8_2bits<cb>(raw[0], raw[1], lp.sh[0], f0, f1);
    else if constexpr (bits == 4) dq8_4bits<cb>(raw[0], raw[1], f0, f1);
    else if constexpr (bits == 3) dq8_3bits<bits, cb>(raw[0], raw[1], lp.sh[0], f0, f1);
    else if constexpr (bits == 7)
    {
        f0.v[0] = dq2_fast<bits, cb>(raw[0], raw[1], lp.sh[0]);
        f0.v[1] = dq2_fast<bits, cb>(raw[2], raw[3], lp.sh[1]);
        f1.v[0] = dq2_fast<bits, cb>(raw[4], raw[5], lp.sh[2]);
        f1.v[1] = dq2_fast<bits, cb>(raw[6], raw[7], lp.sh[3]);
    }
    else
    {
        dq4_fast<bits, cb>(raw[0], raw[1], lp.sh[0], f0);
        dq4_fast<bits, cb>(raw[2], raw[3], lp.sh[1], f1);
    }
}

typedef __attribute__((address_space(1))) const uint32_t g_u32;
typedef __attribute__((address_space(3))) const uint32_t l_u32;
typedef __attribute__((address_space(1))) const char g_char;
typedef __attribute__((address_space(3))) const char l_char;

__device__ __forceinline__ uint32_t ld_g(g_char* base, uint32_t off) { return *(g_u32*) (base + off); }
__device__ __forceinline__ uint32_t ld_l(const char* base, uint32_t off) { return *(l_u32*) ((l_char*) base + off); }

// keeps a loop-invariant lane offset opaque so the loads stay in the SGPR-base + VGPR-offset form
__device__ __forceinline__ void opaque(uint32_t& v) { asm volatile ("" : "+v"(v)); }

// One k-tile of the fdot core from registers: the M rows' A words from LDS (A_c already at this k-tile), the decode of
// the wave's T tiles, the f32 dots (the inner body of exllamav3's exl3_gemv_tiles_rdna step).
template <int bits, int cb, int M, int T>
__device__ __forceinline__ void dot_regs(const char* A_c, const uint32_t* a_row, const uint32_t a_off,
                                         const uint32_t (&raw)[T][LanePlan<bits>::R], const LanePlan<bits>& lp,
                                         float* accA, float* accB)
{
    uint32_t w01[M], w89[M];
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        const char* a = A_c + a_row[r];
        w01[r] = ld_l(a, a_off);
        w89[r] = ld_l(a + 16, a_off);
    }
    #pragma unroll
    for (int t = 0; t < T; ++t)
    {
        Frag f0, f1;
        dq_tile<bits, cb>(raw[t], lp, f0, f1);
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            const h2 a01 = as_h2(w01[r]), a89 = as_h2(w89[r]);
            float& aA = accA[t * M + r];
            float& aB = accB[t * M + r];
            aA = fdot2(a01, f0.v[0], aA);
            aA = fdot2(a89, f0.v[1], aA);
            aB = fdot2(a01, f1.v[0], aB);
            aB = fdot2(a89, f1.v[1], aB);
        }
    }
}


template <int bits, int cb, int M, bool A_LDS, int U, int T>
__device__ __forceinline__ void tiles_step(const char* A_c, const uint32_t* a_row, g_char* b_c, const uint32_t b_stride,
                                           uint32_t* off, uint32_t& a_off, const LanePlan<bits>& lp,
                                           float* accA, float* accB)
{
    constexpr int R = LanePlan<bits>::R;
    constexpr int tile_bytes = 32 * bits;
    #pragma unroll
    for (int i = 0; i < R; ++i) opaque(off[i]);
    opaque(a_off);

    uint32_t raw[U][T][R];
    uint32_t w01[U][M], w89[U][M];
    #pragma unroll
    for (int u = 0; u < U; ++u)
        #pragma unroll
        for (int t = 0; t < T; ++t)
            #pragma unroll
            for (int i = 0; i < R; ++i)
                raw[u][t][i] = ld_g(b_c + (size_t) u * b_stride + t * tile_bytes, off[i]);
    #pragma unroll
    for (int u = 0; u < U; ++u)
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            const char* a = A_c + a_row[r] + u * 32;
            if constexpr (A_LDS) { w01[u][r] = ld_l(a, a_off); w89[u][r] = ld_l(a + 16, a_off); }
            else { w01[u][r] = ld_g((g_char*) a, a_off); w89[u][r] = ld_g((g_char*) a + 16, a_off); }
        }
    #pragma unroll
    for (int u = 0; u < U; ++u)
        #pragma unroll
        for (int t = 0; t < T; ++t)
        {
            Frag f0, f1;
            dq_tile<bits, cb>(raw[u][t], lp, f0, f1);
            #pragma unroll
            for (int r = 0; r < M; ++r)
            {
                const h2 a01 = as_h2(w01[u][r]), a89 = as_h2(w89[u][r]);
                float& aA = accA[t * M + r];
                float& aB = accB[t * M + r];
                aA = fdot2(a01, f0.v[0], aA);
                aA = fdot2(a89, f0.v[1], aA);
                aB = fdot2(a01, f1.v[0], aB);
                aB = fdot2(a89, f1.v[1], aB);
            }
        }
}

// exllamav3 exl3_tiles_u_splitk: k-tiles whose loads go out together
template <int bits, int M, int T>
__device__ __forceinline__ constexpr int tiles_u()
{
    if constexpr (R9K_GEMV_U > 0) return M >= 8 ? 1 : R9K_GEMV_U;
    if constexpr (M >= 8) return 1;
    if constexpr (T >= 2) return (bits <= 3) ? 2 : 1;
    return (bits <= 3) ? 4 : (bits <= 6 ? 2 : 1);
}

// The core: M rows x T adjacent 16-column tiles over nk k16-tiles. A: rows at byte offsets a_row[r] from A (LDS or
// global, f16, rotated basis); B: the matrix's first tile of this k-range / column range, b_stride bytes between
// k-tiles. out[t * M + r] = row r, column lane of tile t (lanes 0-15). All arguments but lane are wave-uniform.
// PI > 0 (package F): the first PI k-tiles' words were loaded by the caller (pre; issued before the input rotation so
// their DRAM latency overlaps it) and are consumed first; the loop then starts at k-tile PI.
template <int bits, int cb, int M, bool A_LDS, int T, int PI = 0>
__device__ __forceinline__ void dot_tiles(const half* A, const uint32_t* a_row_in, const uint16_t* B, uint32_t b_stride,
                                          const int lane, const int nk, float* out,
                                          const uint32_t (*pre)[T][LanePlan<bits>::R] = nullptr)
{
    constexpr int U = tiles_u<bits, M, T>();
    A = uni_ptr(A);
    B = uni_ptr(B);
    const int nk_u = uni(nk);
    const uint32_t b_stride_u = (uint32_t) uni((int) b_stride);
    const LanePlan<bits> lp = lane_plan<bits>(lane);
    constexpr int R = LanePlan<bits>::R;
    uint32_t off[R];
    #pragma unroll
    for (int i = 0; i < R; ++i) off[i] = lp.off[i];
    uint32_t a_off = (uint32_t) (lane & 3) * 4;
    uint32_t a_row[M];
    #pragma unroll
    for (int r = 0; r < M; ++r) a_row[r] = (uint32_t) uni((int) a_row_in[r]);

    float accA[T * M], accB[T * M];
    #pragma unroll
    for (int r = 0; r < T * M; ++r) { accA[r] = 0.0f; accB[r] = 0.0f; }

    g_char* b_c = (g_char*) (const char*) B;
    const char* a_c = (const char*) A;
    int k = 0;
    #pragma unroll
    for (int p = 0; p < PI; ++p)
    {
        dot_regs<bits, cb, M, T>(a_c, a_row, a_off, pre[p], lp, accA, accB);
        b_c += b_stride_u;
        a_c += 32;
    }
    k = PI;
    if constexpr (U > 1)
    {
        for (; k + U <= nk_u; k += U)
        {
            tiles_step<bits, cb, M, A_LDS, U, T>(a_c, a_row, b_c, b_stride_u, off, a_off, lp, accA, accB);
            b_c += (size_t) U * b_stride_u;
            a_c += U * 32;
        }
    }
    for (; k < nk_u; k++)
    {
        tiles_step<bits, cb, M, A_LDS, 1, T>(a_c, a_row, b_c, b_stride_u, off, a_off, lp, accA, accB);
        b_c += b_stride_u;
        a_c += 32;
    }

    #pragma unroll
    for (int r = 0; r < T * M; ++r)
    {
        accA[r] += __shfl_xor(accA[r], 1, 32);
        accA[r] += __shfl_xor(accA[r], 2, 32);
        accB[r] += __shfl_xor(accB[r], 1, 32);
        accB[r] += __shfl_xor(accB[r], 2, 32);
        float vA = __shfl(accA[r], (lane & 7) * 4, 32);
        float vB = __shfl(accB[r], (lane & 7) * 4, 32);
        out[r] = (lane & 8) ? vB : vA;
    }
}

// 128-point Hadamard across a wave: lane holds elements 4 lane .. 4 lane + 3 (hadamard_inner.cuh shuffle_had_f4x32
// after the in-register 4-point stage), unnormalised
__device__ __forceinline__ void had128(float* h, const int lane)
{
    float s0 = h[0] + h[1], d0 = h[0] - h[1], s1 = h[2] + h[3], d1 = h[2] - h[3];
    float h0 = s0 + s1, h1 = d0 + d1, h2_ = s0 - s1, h3 = d0 - d1;
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1)
    {
        uint32_t i0 = __float_as_uint(h0), i1 = __float_as_uint(h1), i2 = __float_as_uint(h2_), i3 = __float_as_uint(h3);
        uint64_t h01 = (uint64_t) i0 | (((uint64_t) i1) << 32);
        uint64_t h23 = (uint64_t) i2 | (((uint64_t) i3) << 32);
        uint64_t ph01 = __shfl_xor(h01, i, 32);
        uint64_t ph23 = __shfl_xor(h23, i, 32);
        float ph0 = __uint_as_float((uint32_t) (ph01 & 0xffffffff));
        float ph1 = __uint_as_float((uint32_t) (ph01 >> 32));
        float ph2 = __uint_as_float((uint32_t) (ph23 & 0xffffffff));
        float ph3 = __uint_as_float((uint32_t) (ph23 >> 32));
        int32_t sfm = -static_cast<int32_t>(lane & i) >> 31;
        i0 ^= sfm & 0x80000000; i1 ^= sfm & 0x80000000; i2 ^= sfm & 0x80000000; i3 ^= sfm & 0x80000000;
        h0 = __uint_as_float(i0) + ph0;
        h1 = __uint_as_float(i1) + ph1;
        h2_ = __uint_as_float(i2) + ph2;
        h3 = __uint_as_float(i3) + ph3;
    }
    h[0] = h0; h[1] = h1; h[2] = h2_; h[3] = h3;
}

// ---------------------------------------------------------------------------------------------------------------
// end of EXLLAMAV3 code
// ---------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ void ld_h4(const half* p, float* v)
{
    uint2 u = *(const uint2*) p;
    h2 a = as_h2(u.x), b = as_h2(u.y);
    v[0] = (float) a.x; v[1] = (float) a.y; v[2] = (float) b.x; v[3] = (float) b.y;
}

__device__ __forceinline__ void st_h4(half* p, const float* v)
{
    h2 a = {(_Float16) v[0], (_Float16) v[1]}, b = {(_Float16) v[2], (_Float16) v[3]};
    *(uint2*) p = make_uint2(__builtin_bit_cast(uint32_t, a), __builtin_bit_cast(uint32_t, b));
}

__device__ __forceinline__ float bf16_f(uint32_t b) { return __uint_as_float(b << 16); }

__device__ __forceinline__ uint32_t f_bf16(float f)       // round to nearest even, NaN kept quiet (= torch)
{
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (u >> 16) | 0x40;
    u += 0x7fffu + ((u >> 16) & 1);
    return u >> 16;
}

// One matrix (or one 8-row chunk of one) of a launch. x [m, k] row stride xs elements; y [m, n] row stride ys elements
// (the group's first output column); part [ks, m, n] f32; ctr [n / 128] int32, zero between launches.
struct Grp
{
    const uint16_t* tr;       // trellis [k/16, n/16, 16 * bits]
    const half* suh;          // [k]
    const half* svh;          // [n]
    const void* x;
    void* y;
    float* part;
    int* ctr;
    int k, n, m, xs, ys, ks, kc, blk0, xm, ym;   // xm: 0 bf16, 1 f16, 2 f16 rotated; ym: 0 bf16, 1 f16, 2 f32
};

struct Args
{
    Grp g[MAXG];
    int ng;
    unsigned long long* tr;   // R9K_GEMV_TRACE builds: per-block phase stamps [block][TRS] (wall_clock64, 100 MHz)
};

// Phase trace (compile with -DR9K_GEMV_TRACE; r9k_exl3_gemv_trace sets the buffer): stamp i of this block by thread 0.
constexpr int TRS = 8;
#ifdef R9K_GEMV_TRACE
#define TR(i) do { if (a.tr && threadIdx.x == 0) a.tr[(size_t) blockIdx.x * TRS + (i)] = wall_clock64(); } while (0)
#else
#define TR(i) do { } while (0)
#endif

// Step 1 for rows 0..m-1 of the block's input chunk [k0, k0 + 128 segs): one wave a (row, 128-block), four tasks a
// wave in flight (their global loads go out together), into LDS rows of xl halves. Split in two so a kernel can issue
// its first trellis loads between the input loads of the first batch (rot_load) and their use (rot_store): the DRAM
// latency of the first tiles then overlaps the input's L2 round trip and the Hadamard instead of following them.
constexpr int B4 = 4;
struct RotBatch { float v[B4][4], s[B4][4]; };

__device__ __forceinline__ void rot_load(const Grp& g, RotBatch& rb, const int k0, const int segs, const int t0,
                                         const int lane, const int nwaves)
{
    const int ntask = g.m * segs;
    #pragma unroll
    for (int j = 0; j < B4; ++j)
    {
        const int task = t0 + j * nwaves;
        if (task >= ntask) break;
        const int r = task / segs, seg = task % segs;
        const int kk = k0 + seg * 128 + lane * 4;
        if (g.xm == 0)
        {
            const uint2 u = *(const uint2*) ((const uint16_t*) g.x + (size_t) r * g.xs + kk);
            rb.v[j][0] = bf16_f(u.x & 0xffff); rb.v[j][1] = bf16_f(u.x >> 16);
            rb.v[j][2] = bf16_f(u.y & 0xffff); rb.v[j][3] = bf16_f(u.y >> 16);
        }
        else ld_h4((const half*) g.x + (size_t) r * g.xs + kk, rb.v[j]);
        if (g.xm != 2) ld_h4(g.suh + kk, rb.s[j]);
    }
}

__device__ __forceinline__ void rot_store(const Grp& g, RotBatch& rb, half* sh_x, const int xl, const int segs,
                                          const int t0, const int lane, const int nwaves)
{
    const int ntask = g.m * segs;
    #pragma unroll
    for (int j = 0; j < B4; ++j)
    {
        const int task = t0 + j * nwaves;
        if (task >= ntask) break;
        const int r = task / segs, seg = task % segs;
        if (g.xm != 2)
        {
            #pragma unroll
            for (int i = 0; i < 4; ++i) rb.v[j][i] *= rb.s[j][i];
            had128(rb.v[j], lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) rb.v[j][i] *= RS128;
        }
        st_h4(sh_x + (size_t) r * xl + seg * 128 + lane * 4, rb.v[j]);
    }
}

// the batches after the first (only when m x segs > B4 x waves)
__device__ __forceinline__ void rotate_rest(const Grp& g, half* sh_x, const int xl, const int k0, const int segs,
                                            const int wave, const int lane, const int nwaves)
{
    for (int t0 = wave + nwaves * B4; t0 < g.m * segs; t0 += nwaves * B4)
    {
        RotBatch rb;
        rot_load(g, rb, k0, segs, t0, lane, nwaves);
        rot_store(g, rb, sh_x, xl, segs, t0, lane, nwaves);
    }
}

// Package D's step 1 (WMMA blocks): rows 0..m-1 of the block's input chunk [k0, k0 + 128 segs): one wave a (row, 128-block), four tasks a
// wave in flight (their global loads go out together), into LDS rows of xl halves.
__device__ __forceinline__ void rotate_in(const Grp& g, half* sh_x, const int xl, const int k0, const int segs,
                                          const int wave, const int lane, const int nwaves)
{
    constexpr int B4 = 4;
    const int ntask = g.m * segs;
    for (int t0 = wave; t0 < ntask; t0 += nwaves * B4)
    {
        float v[B4][4], s[B4][4];
        #pragma unroll
        for (int j = 0; j < B4; ++j)
        {
            const int task = t0 + j * nwaves;
            if (task >= ntask) break;
            const int r = task / segs, seg = task % segs;
            const int kk = k0 + seg * 128 + lane * 4;
            if (g.xm == 0)
            {
                const uint2 u = *(const uint2*) ((const uint16_t*) g.x + (size_t) r * g.xs + kk);
                v[j][0] = bf16_f(u.x & 0xffff); v[j][1] = bf16_f(u.x >> 16);
                v[j][2] = bf16_f(u.y & 0xffff); v[j][3] = bf16_f(u.y >> 16);
            }
            else ld_h4((const half*) g.x + (size_t) r * g.xs + kk, v[j]);
            if (g.xm != 2) ld_h4(g.suh + kk, s[j]);
        }
        #pragma unroll
        for (int j = 0; j < B4; ++j)
        {
            const int task = t0 + j * nwaves;
            if (task >= ntask) break;
            const int r = task / segs, seg = task % segs;
            if (g.xm != 2)
            {
                #pragma unroll
                for (int i = 0; i < 4; ++i) v[j][i] *= s[j][i];
                had128(v[j], lane);
                #pragma unroll
                for (int i = 0; i < 4; ++i) v[j][i] *= RS128;
            }
            st_h4(sh_x + (size_t) r * xl + seg * 128 + lane * 4, v[j]);
        }
    }
}

// Package D's epilogue, unchanged (WMMA blocks and fdot row tiles above R9K_GEMV_FSPLIT_M): for output tile nb, per row (rows wave, wave + nwaves, ...), the ks partials summed in
// chunk order, the output Hadamard, svh, the store in the group's dtype. A wave works on four rows at once and issues
// their partial loads four chunks at a time (a serial chain of ks L2 round trips per row was the tail of every launch).
__device__ __forceinline__ void finish_rows(const Grp& g, const int nb, const int wave, const int lane, const int nwaves)
{
    constexpr int RR = 4, QB = 4;
    const int m = g.m, n = g.n, ks = g.ks;
    const int col = nb * 128 + lane * 4;
    const size_t qs = (size_t) m * n;
    float sv[4];
    ld_h4(g.svh + col, sv);
    for (int r0 = wave; r0 < m; r0 += nwaves * RR)
    {
        float v[RR][4];
        #pragma unroll
        for (int i = 0; i < RR; ++i) v[i][0] = v[i][1] = v[i][2] = v[i][3] = 0.0f;
        for (int q0 = 0; q0 < ks; q0 += QB)
        {
            f4 p[RR][QB];
            #pragma unroll
            for (int i = 0; i < RR; ++i)
                #pragma unroll
                for (int j = 0; j < QB; ++j)
                    if (r0 + i * nwaves < m && q0 + j < ks)
                        p[i][j] = __builtin_nontemporal_load(
                            (const f4*) (g.part + (size_t) (r0 + i * nwaves) * n + col + (q0 + j) * qs));
            #pragma unroll
            for (int i = 0; i < RR; ++i)
                #pragma unroll
                for (int j = 0; j < QB; ++j)
                    if (r0 + i * nwaves < m && q0 + j < ks)
                        #pragma unroll
                        for (int e = 0; e < 4; ++e) v[i][e] += p[i][j][e];
        }
        #pragma unroll
        for (int i = 0; i < RR; ++i)
        {
            const int r = r0 + i * nwaves;
            if (r >= m) break;
            had128(v[i], lane);
            #pragma unroll
            for (int e = 0; e < 4; ++e) v[i][e] *= RS128 * sv[e];
            if (g.ym == 2) *(f4*) ((float*) g.y + (size_t) r * g.ys + col) = f4{v[i][0], v[i][1], v[i][2], v[i][3]};
            else if (g.ym == 1) st_h4((half*) g.y + (size_t) r * g.ys + col, v[i]);
            else
                *(uint2*) ((uint16_t*) g.y + (size_t) r * g.ys + col) =
                    make_uint2(f_bf16(v[i][0]) | (f_bf16(v[i][1]) << 16), f_bf16(v[i][2]) | (f_bf16(v[i][3]) << 16));
        }
    }
}

// A workgroup barrier for LDS only. __syncthreads() is a workgroup-scope release / acquire fence + s_barrier, and on
// gfx12 (WGP mode) that fence waits for every outstanding global load of the wave (s_wait_loadcnt 0): trellis words
// issued before the barrier would be waited for at it, which defeats issuing them early. Here only the LDS stores are
// waited for (s_wait_dscnt 0) before the barrier; global loads stay in flight across it (package F).
__device__ __forceinline__ void bar_lds()
{
    asm volatile ("s_wait_dscnt 0x0" ::: "memory");
    __builtin_amdgcn_s_barrier();
    asm volatile ("" ::: "memory");
}

// The last arriver's epilogue for output tile nb: the ks partials of each row summed in chunk order, the output
// Hadamard, svh, the store in the group's dtype. Every load of a wave's share goes out at once (QB chunks of two tasks),
// so the epilogue is one L2 round trip, not ks / 4 of them. With red (an LDS scratch of 8 x 128 floats; fdot blocks)
// and few rows, a row's chunks are cut into S contiguous slices summed by different waves, then the slices in order by
// the row's wave: m = 1 spreads its 20-40 partials over all waves instead of leaving three idle. The partition depends
// only on (m, ks, nwaves): deterministic.
__device__ __forceinline__ void finish(const Grp& g, const int nb, const int wave, const int lane, const int nwaves,
                                       float* red, unsigned long long* trp = nullptr, const uint2* svp = nullptr)
{
    constexpr int QB = R9K_GEMV_QB, TW = 2;  // chunks a task in flight, tasks a wave in flight
    const int m = g.m, n = g.n, ks = g.ks;
    const int col = nb * 128 + lane * 4;
    const size_t qs = (size_t) m * n;
    int S = 1;
    if (red && m * 2 <= nwaves * 2) { S = (nwaves * 2) / m; S = S < ks ? S : ks; }
    const int ntask = m * S;
    float sv[4];
    if (svp)          // svh loaded by the caller at kernel start (off the epilogue's dependent chain)
    {
        h2 a0 = as_h2(svp->x), a1 = as_h2(svp->y);
        sv[0] = (float) a0.x; sv[1] = (float) a0.y; sv[2] = (float) a1.x; sv[3] = (float) a1.y;
    }
    else ld_h4(g.svh + col, sv);
    for (int t0 = wave; t0 < ntask; t0 += nwaves * TW)
    {
        float v[TW][4];
        int q_lo[TW], q_hi[TW], row[TW];
        #pragma unroll
        for (int i = 0; i < TW; ++i)
        {
            const int t = t0 + i * nwaves;
            const int r = t / S, s = t % S;
            row[i] = t < ntask ? r : -1;
            q_lo[i] = s * ks / S;
            q_hi[i] = (s + 1) * ks / S;
            v[i][0] = v[i][1] = v[i][2] = v[i][3] = 0.0f;
        }
        int span = 0;
        #pragma unroll
        for (int i = 0; i < TW; ++i) span = (row[i] >= 0 && q_hi[i] - q_lo[i] > span) ? q_hi[i] - q_lo[i] : span;
        for (int q0 = 0; q0 < span; q0 += QB)
        {
            f4 p[TW][QB];
            // unconditional loads (an invalid slot reads chunk 0 of row 0 and is not added): no branch between them,
            // so they all go out before the first wait
            #pragma unroll
            for (int i = 0; i < TW; ++i)
                #pragma unroll
                for (int j = 0; j < QB; ++j)
                {
                    const bool ok = row[i] >= 0 && q_lo[i] + q0 + j < q_hi[i];
                    const size_t o = ok ? (size_t) row[i] * n + (size_t) (q_lo[i] + q0 + j) * qs : 0;
                    p[i][j] = __builtin_nontemporal_load((const f4*) (g.part + o + col));
                }
            #pragma unroll
            for (int i = 0; i < TW; ++i)
                #pragma unroll
                for (int j = 0; j < QB; ++j)
                {
                    const bool ok = row[i] >= 0 && q_lo[i] + q0 + j < q_hi[i];
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) v[i][e] += ok ? p[i][j][e] : 0.0f;
                }
        }
#ifdef R9K_GEMV_TRACE
        if (trp && threadIdx.x == 0 && t0 == 0) trp[7] = wall_clock64();     // first partials summed
#endif
        #pragma unroll
        for (int i = 0; i < TW; ++i)
        {
            if (row[i] < 0) break;
            if (S > 1) { *(f4*) (red + (t0 + i * nwaves) * 128 + lane * 4) = f4{v[i][0], v[i][1], v[i][2], v[i][3]}; continue; }
            const int r = row[i];
            had128(v[i], lane);
            #pragma unroll
            for (int e = 0; e < 4; ++e) v[i][e] *= RS128 * sv[e];
            if (g.ym == 2) *(f4*) ((float*) g.y + (size_t) r * g.ys + col) = f4{v[i][0], v[i][1], v[i][2], v[i][3]};
            else if (g.ym == 1) st_h4((half*) g.y + (size_t) r * g.ys + col, v[i]);
            else
                *(uint2*) ((uint16_t*) g.y + (size_t) r * g.ys + col) =
                    make_uint2(f_bf16(v[i][0]) | (f_bf16(v[i][1]) << 16), f_bf16(v[i][2]) | (f_bf16(v[i][3]) << 16));
        }
    }
    if (S == 1) return;
    bar_lds();
    for (int r = wave; r < m; r += nwaves)
    {
        float v[4];
        const f4 h = *(const f4*) (red + (r * S) * 128 + lane * 4);
        v[0] = h[0]; v[1] = h[1]; v[2] = h[2]; v[3] = h[3];
        for (int s = 1; s < S; ++s)
        {
            const f4 u = *(const f4*) (red + (r * S + s) * 128 + lane * 4);
            #pragma unroll
            for (int e = 0; e < 4; ++e) v[e] += u[e];
        }
        had128(v, lane);
        #pragma unroll
        for (int e = 0; e < 4; ++e) v[e] *= RS128 * sv[e];
        if (g.ym == 2) *(f4*) ((float*) g.y + (size_t) r * g.ys + col) = f4{v[0], v[1], v[2], v[3]};
        else if (g.ym == 1) st_h4((half*) g.y + (size_t) r * g.ys + col, v);
        else
            *(uint2*) ((uint16_t*) g.y + (size_t) r * g.ys + col) =
                make_uint2(f_bf16(v[0]) | (f_bf16(v[1]) << 16), f_bf16(v[2]) | (f_bf16(v[3]) << 16));
    }
}

template <int bits, int cb, int M>
__global__ __launch_bounds__(WAVES * 32) void k_gemv(const Args a)
{
    extern __shared__ __align__(16) char smem[];
    half* sh_x = (half*) smem;                      // [m][kc] rotated input chunk; later the epilogue's slice sums
    __shared__ int s_last;
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int b = blockIdx.x;
    int gi = 0;
    for (int i = 1; i < a.ng; ++i) gi = b >= a.g[i].blk0 ? i : gi;
    gi = __builtin_amdgcn_readfirstlane(gi);
    const Grp& g = a.g[gi];
    const int m = g.m, n = g.n, ks = g.ks, kc = g.kc;
    const int lb = b - g.blk0, nb = lb / ks, kq = lb % ks;
    const int k0 = kq * kc, segs = kc / 128;
    TR(0);

    // the dot's operands: column wave cw takes tiles nb*8 + cw*TPW .., k-wave kw the kw-th part of the chunk
    constexpr int T = TPW;
    constexpr int R = LanePlan<bits>::R;
    constexpr int PI = R9K_GEMV_PI < 8 / KW ? R9K_GEMV_PI : 8 / KW;   // k-tiles issued before the rotation (nk >= 8/KW)
    constexpr int tile_bytes = 32 * bits;
    constexpr int tile_u16 = 16 * bits;
    const int cw = wave % CW, kw = wave / CW;
    const int kcw = kc / KW;
    const int tile0 = nb * 8 + cw * T;
    const int n_tiles = n / 16;
    const uint16_t* B = g.tr + ((size_t) ((k0 + kw * kcw) / 16) * n_tiles + tile0) * tile_u16;

    // 1a. the first batch of input loads, svh (for the epilogue, if this block runs it), then the first PI k-tiles of
    // trellis: their DRAM latency overlaps the input's round trip and the Hadamard instead of following them
    RotBatch rb;
    rot_load(g, rb, k0, segs, wave, lane, WAVES);
    const uint2 svr = *(const uint2*) (g.svh + nb * 128 + lane * 4);
    uint32_t pre[PI > 0 ? PI : 1][T][R];
    if constexpr (PI > 0)
    {
        const LanePlan<bits> lp = lane_plan<bits>(lane);
        g_char* b_c = (g_char*) (const char*) uni_ptr(B);
        const uint32_t kstride = (uint32_t) uni((int) ((uint32_t) n_tiles * tile_bytes));
        #pragma unroll
        for (int p = 0; p < PI; ++p)
        {
            g_char* bq = uni_ptr(b_c + p * kstride);
            #pragma unroll
            for (int t = 0; t < T; ++t)
                #pragma unroll
                for (int i = 0; i < R; ++i) pre[p][t][i] = ld_g(bq + t * tile_bytes, lp.off[i]);
        }
    }
    // 1b. rotate into LDS
    rot_store(g, rb, sh_x, kc, segs, wave, lane, WAVES);
    rotate_rest(g, sh_x, kc, k0, segs, wave, lane, WAVES);
    bar_lds();
    TR(1);

    // 2. stream the chunk
    uint32_t a_row[M];
    #pragma unroll
    for (int r = 0; r < M; ++r) a_row[r] = (uint32_t) (((r < m ? r : m - 1) * kc + kw * kcw) * 2);
    float out[TPW * M];
    dot_tiles<bits, cb, M, true, TPW, PI>(sh_x, a_row, B, (uint32_t) n_tiles * tile_u16 * 2, lane, kcw / 16, out, pre);
    TR(2);

    if constexpr (KW > 1)
    {
        float* red = (float*) (smem + (size_t) M * kc * sizeof(half));   // [KW-1][CW][TPW*M][16]
        __syncthreads();
        if (kw > 0 && lane < 16)
            #pragma unroll
            for (int i = 0; i < TPW * M; ++i) red[(((kw - 1) * CW + cw) * TPW * M + i) * 16 + lane] = out[i];
        __syncthreads();
        if (kw == 0 && lane < 16)
            for (int w = 1; w < KW; ++w)
                #pragma unroll
                for (int i = 0; i < TPW * M; ++i) out[i] += red[(((w - 1) * CW + cw) * TPW * M + i) * 16 + lane];
    }
    if (kw == 0 && lane < 16)
        for (int r = 0; r < m; ++r)
            #pragma unroll
            for (int t = 0; t < TPW; ++t)
                g.part[((size_t) kq * m + r) * n + (tile0 + t) * 16 + lane] = out[t * M + r];

    // 3. last arriver of the output tile: chunk-ordered sum, output Hadamard, svh, store
    __threadfence();
    __syncthreads();
    TR(3);
    int* c = g.ctr + nb;
    if (threadIdx.x == 0) s_last = atomicAdd(c, 1) == ks - 1;
    __syncthreads();
    TR(4);
    if (!s_last) return;
    __threadfence();
    if (threadIdx.x == 0) *c = 0;
    TR(6);
    // M <= FSPLIT_M: the split epilogue (all waves on a row's chunks, svh preloaded); wider row tiles keep package D's
    // row-per-wave epilogue, which measured faster there (6-8 rows: 16.5 vs 17.3-18.2 us on wq_b)
    if constexpr (M <= R9K_GEMV_FSPLIT_M)
    {
#ifdef R9K_GEMV_TRACE
        finish(g, nb, wave, lane, WAVES, FSPLIT ? (float*) smem : nullptr, a.tr ? a.tr + (size_t) blockIdx.x * TRS : nullptr, &svr);
#else
        finish(g, nb, wave, lane, WAVES, FSPLIT ? (float*) smem : nullptr, nullptr, &svr);
#endif
    }
    else finish_rows(g, nb, wave, lane, WAVES);
    TR(5);
}

// Rows 9..48: the same blocks, but the dot runs on WMMA (v_wmma_f32_16x16x16_f16, gfx12 operand layout: A lane L =
// row L%16, k 8(L/16)..+7; B lane L = column L%16, the same k; D lane L = column L%16, rows 8(L/16)..+7). One trellis
// decode a 16x16 tile feeds up to RT = 3 row tiles; the decoded tile goes through a per-wave LDS transpose ([n][k], row
// stride 24 halves) into the B layout. The rotated input rows sit in LDS with a padded row stride (kc + 8 halves) so the
// 16 rows of an A fragment hit different banks. Rows past m are computed from whatever is in LDS and never stored.
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef float f8 __attribute__((ext_vector_type(8)));
constexpr int TB = 24;                      // transpose buffer row stride (halves)

template <int bits, int cb, int RT, int NB2>
__global__ __launch_bounds__(CW * NB2 * 32) void k_gemv_mma(const Args a)
{
    extern __shared__ __align__(16) char smem[];
    half* sh_x = (half*) smem;                      // [RT*16][kc + 8] rotated input chunk
    __shared__ __align__(16) _Float16 s_b[CW * NB2][TPW][16 * TB];
    __shared__ int s_last;
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int b = blockIdx.x;
    int gi = 0;
    for (int i = 1; i < a.ng; ++i) gi = b >= a.g[i].blk0 ? i : gi;
    gi = __builtin_amdgcn_readfirstlane(gi);
    const Grp& g = a.g[gi];
    const int m = g.m, n = g.n, ks = g.ks, kc = g.kc;
    const int lb = b - g.blk0, nbb = lb / ks, kq = lb % ks;   // nbb: NB2 output tiles of 128
    const int cw = wave % CW, tj = wave / CW, nb = nbb * NB2 + tj;
    const int k0 = kq * kc, segs = kc / 128, xl = kc + 8;

    rotate_in(g, sh_x, xl, k0, segs, wave, lane, CW * NB2);
    __syncthreads();

    constexpr int R = LanePlan<bits>::R;
    constexpr int tile_bytes = 32 * bits;
    const int n_tiles = n / 16;
    const int tile0 = nb * 8 + cw * TPW;
    const uint32_t kstride = (uint32_t) n_tiles * tile_bytes;
    g_char* bp = (g_char*) (const char*) uni_ptr(g.tr + ((size_t) (k0 / 16) * n_tiles + tile0) * 16 * bits);
    const LanePlan<bits> lp = lane_plan<bits>(lane);
    uint32_t off[R];
    #pragma unroll
    for (int i = 0; i < R; ++i) off[i] = lp.off[i];
    const int r0 = (lane & 3) * 2, cA = (lane / 8) * 2 + ((lane >> 2) & 1), cB = cA + 8;
    const int ln = lane % 16, lh = lane / 16;
    f8 acc[RT][TPW];
    #pragma unroll
    for (int rt = 0; rt < RT; ++rt)
        #pragma unroll
        for (int t = 0; t < TPW; ++t)
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[rt][t][i] = 0.0f;
    const int nk = kc / 16;
    uint32_t raw[TPW][R];
    #pragma unroll
    for (int t = 0; t < TPW; ++t)
        #pragma unroll
        for (int i = 0; i < R; ++i) raw[t][i] = ld_g(bp + t * tile_bytes, off[i]);
    for (int kt = 0; kt < nk; ++kt)
    {
        uint32_t nxt[TPW][R];
        if (kt + 1 < nk)
        {
            g_char* bq = bp + (size_t) (kt + 1) * kstride;
            #pragma unroll
            for (int t = 0; t < TPW; ++t)
                #pragma unroll
                for (int i = 0; i < R; ++i) nxt[t][i] = ld_g(bq + t * tile_bytes, off[i]);
        }
        h8 bf[TPW];
        #pragma unroll
        for (int t = 0; t < TPW; ++t)
        {
            Frag f0, f1;
            dq_tile<bits, cb>(raw[t], lp, f0, f1);
            uint32_t* w = (uint32_t*) &s_b[wave][t][0];
            w[(cA * TB + r0) / 2] = __builtin_bit_cast(uint32_t, f0.v[0]);
            w[(cA * TB + r0 + 8) / 2] = __builtin_bit_cast(uint32_t, f0.v[1]);
            w[(cB * TB + r0) / 2] = __builtin_bit_cast(uint32_t, f1.v[0]);
            w[(cB * TB + r0 + 8) / 2] = __builtin_bit_cast(uint32_t, f1.v[1]);
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        #pragma unroll
        for (int t = 0; t < TPW; ++t) bf[t] = *(const h8*) &s_b[wave][t][ln * TB + lh * 8];
        #pragma unroll
        for (int rt = 0; rt < RT; ++rt)
        {
            const h8 af = *(const h8*) (sh_x + (size_t) (rt * 16 + ln) * xl + kt * 16 + lh * 8);
            #pragma unroll
            for (int t = 0; t < TPW; ++t)
                acc[rt][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(af, bf[t], acc[rt][t]);
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        if (kt + 1 < nk)
            #pragma unroll
            for (int t = 0; t < TPW; ++t)
                #pragma unroll
                for (int i = 0; i < R; ++i) raw[t][i] = nxt[t][i];
    }
    #pragma unroll
    for (int rt = 0; rt < RT; ++rt)
        #pragma unroll
        for (int t = 0; t < TPW; ++t)
            #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                const int r = rt * 16 + lh * 8 + i;
                if (r < m) g.part[((size_t) kq * m + r) * n + (tile0 + t) * 16 + ln] = acc[rt][t][i];
            }

    __threadfence();
    __syncthreads();
    int* c = g.ctr + nbb;
    if (threadIdx.x == 0) s_last = atomicAdd(c, 1) == ks - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    if (threadIdx.x == 0) *c = 0;
    finish_rows(g, nb, cw, lane, CW);
}

template <int bits, int cb>
int launch_mma(int M, int nb2, const Args& a, int nblk, size_t sm, hipStream_t st)
{
    dim3 grid(nblk), block(CW * nb2 * 32);
#define LM(rt, n2) case rt * 8 + n2: hipLaunchKernelGGL((k_gemv_mma<bits, cb, rt, n2>), grid, block, sm, st, a); break;
    switch ((M + 15) / 16 * 8 + nb2)
    {
        LM(1, 1) LM(1, 2) LM(1, 4) LM(2, 1) LM(2, 2) LM(2, 4) LM(3, 1) LM(3, 2) LM(3, 4)
        default: return -1;
    }
#undef LM
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// Validation: decode a whole trellis [k/16, n/16, 16 bits] to f16 [k, n] (rotated basis, = ext.reconstruct)
template <int bits, int cb>
__global__ __launch_bounds__(256) void k_dq(const uint16_t* trellis, half* out, int k, int n)
{
    const int wave = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    const int nt = n / 16;
    if (wave >= (k / 16) * nt) return;
    const int kt = wave / nt, ntile = wave % nt;
    g_char* bp = (g_char*) (const char*) (trellis + (size_t) wave * 16 * bits);
    const LanePlan<bits> lp = lane_plan<bits>(lane);
    uint32_t raw[LanePlan<bits>::R];
    #pragma unroll
    for (int i = 0; i < LanePlan<bits>::R; ++i) raw[i] = ld_g(bp, lp.off[i]);
    Frag f0, f1;
    dq_tile<bits, cb>(raw, lp, f0, f1);
    const int r0 = (lane & 3) * 2, cA = (lane / 8) * 2 + ((lane >> 2) & 1), cB = cA + 8;
    _Float16* o = (_Float16*) out;
    auto at = [&](int r, int cc) -> _Float16& { return o[(size_t) (kt * 16 + r) * n + ntile * 16 + cc]; };
    at(r0, cA) = f0.v[0].x; at(r0 + 1, cA) = f0.v[0].y; at(r0 + 8, cA) = f0.v[1].x; at(r0 + 9, cA) = f0.v[1].y;
    at(r0, cB) = f1.v[0].x; at(r0 + 1, cB) = f1.v[0].y; at(r0 + 8, cB) = f1.v[1].x; at(r0 + 9, cB) = f1.v[1].y;
}

template <int bits, int cb>
int launch_m(int M, const Args& a, int nblk, size_t sm, hipStream_t st)
{
    dim3 grid(nblk), block(WAVES * 32);
    switch (M)
    {
#define L(mm) case mm: hipLaunchKernelGGL((k_gemv<bits, cb, mm>), grid, block, sm, st, a); break;
        L(1) L(2) L(3) L(4) L(5) L(6) L(7) L(8) L(9) L(10) L(11) L(12) L(13) L(14) L(15) L(16)
#undef L
        default: return -1;
    }
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

template <int cb>
int launch_k(int K, int M, int mma, const Args& a, int nblk, size_t sm, hipStream_t st)
{
#define LK(bits) case bits: return mma ? launch_mma<bits, cb>(M, mma, a, nblk, sm, st) : launch_m<bits, cb>(M, a, nblk, sm, st);
    switch (K)
    {
        LK(2) LK(3) LK(4) LK(5) LK(6)
        default: return -2;
    }
#undef LK
}

}  // namespace

static unsigned long long* s_trace = nullptr;
// R9K_GEMV_TRACE builds: per-block phase stamps into buf (u64 [blocks][8]) from the next launch on; 0 turns it off.
// Returns 0, or -1 in a build without the trace.
extern "C" int r9k_exl3_gemv_trace(long buf)
{
#ifdef R9K_GEMV_TRACE
    s_trace = (unsigned long long*) buf;
    return 0;
#else
    (void) buf;
    return -1;
#endif
}

extern "C" int r9k_exl3_gemv_max_groups(void) { return MAXG; }
extern "C" int r9k_exl3_gemv_max_rows(void) { return MAXM; }
extern "C" int r9k_exl3_gemv_max_rows_mma(void) { return MAXM_MMA; }
extern "C" int r9k_exl3_gemv_waves(void) { return WAVES; }

// One launch over ng groups (mma 0: the fdot row tile, M <= 16; mma 1 / 2 / 4: WMMA, M <= 48, blocks of mma x 128
// output columns sharing one input rotation; every group's n / 128 a multiple of it). desc: ng x 13 int64 {trellis, suh, svh, x, y, k, n, m, xs, ys, ks, xm, ym} (pointers are
// device addresses; xs / ys row strides in elements, multiples of 4; xm 0 bf16 / 1 f16 / 2 f16 rotated; ym 0 bf16 /
// 1 f16 / 2 f32). Every group's matrix has K bits and codebook cb (1 mcg, 2 mul1); M = the row tile (>= every m).
// part: f32 scratch of part_floats; ctr: int32 counters (ctr_ints), zero at allocation and left zero by every launch.
// Group g's partials take sum_{h<g} ks_h m_h n_h floats from the start of part, its counters sum_{h<g} n_h / 128.
// Returns 0, or -1 bad shape / count, -2 bad K / codebook, -3 scratch too small, -4 launch error.
extern "C" int r9k_exl3_gemv(const long* desc, int ng, int M, int K, int cb, int mma, long part, long part_floats,
                             long ctr, int ctr_ints, long stream)
{
    if (ng < 1 || ng > MAXG || M < 1 || M > (mma ? MAXM_MMA : MAXM) || (mma != 0 && mma != 1 && mma != 2 && mma != 4)) return -1;
    if (cb != 1 && cb != 2) return -2;
    Args a;
    a.ng = ng;
    a.tr = s_trace;
    long poff = 0;
    int coff = 0, nblk = 0;
    size_t sm = 0;
    for (int i = 0; i < ng; ++i)
    {
        const long* d = desc + 13 * i;
        Grp& g = a.g[i];
        g.tr = (const uint16_t*) d[0]; g.suh = (const half*) d[1]; g.svh = (const half*) d[2];
        g.x = (const void*) d[3]; g.y = (void*) d[4];
        g.k = (int) d[5]; g.n = (int) d[6]; g.m = (int) d[7]; g.xs = (int) d[8]; g.ys = (int) d[9]; g.ks = (int) d[10];
        g.xm = (int) d[11]; g.ym = (int) d[12];
        if (g.k <= 0 || g.n <= 0 || g.k % 128 || g.n % 128 || g.m < 1 || g.m > M || g.ks < 1) return -1;
        if (g.k % (128 * g.ks) || (g.k / g.ks) % (16 * KW) || g.xs % 4 || g.ys % 4) return -1;
        if (g.xm < 0 || g.xm > 2 || g.ym < 0 || g.ym > 2) return -1;
        g.kc = g.k / g.ks;
        g.blk0 = nblk;
        g.part = (float*) part + poff;
        g.ctr = (int*) ctr + coff;
        poff += (long) g.ks * g.m * g.n;
        coff += g.n / 128;
        if (mma && (g.n / 128) % mma) return -1;
        nblk += (g.n / 128 / (mma ? mma : 1)) * g.ks;
        size_t s = mma ? (size_t) (M + 15) / 16 * 16 * (g.kc + 8) * sizeof(half) : (size_t) M * g.kc * sizeof(half);
        if (s > sm) sm = s;
    }
    if (poff > part_floats || coff > ctr_ints) return -3;
    if (KW > 1 && !mma) sm += (size_t) (KW - 1) * CW * TPW * M * 16 * sizeof(float);
    if (!mma && sm < 8 * 128 * sizeof(float)) sm = 8 * 128 * sizeof(float);   // finish's slice sums alias sh_x
    // WMMA blocks also hold the static transpose buffers (s_b: CW * NB2 waves x TPW tiles x 16 x TB halves); the sum
    // must fit the 64 KiB of a workgroup (NB2 = 4 with a 49 KiB chunk was an invalid-allocation abort before package F)
    const size_t st_lds = mma ? (size_t) CW * mma * TPW * 16 * TB * sizeof(_Float16) + 64 : 64;
    if (sm + st_lds > 64 * 1024) return -1;
    hipStream_t st = (hipStream_t) stream;
    return cb == 2 ? launch_k<2>(K, M, mma, a, nblk, sm, st) : launch_k<1>(K, M, mma, a, nblk, sm, st);
}

extern "C" int r9k_exl3_gemv_dq(long trellis, long out, int k, int n, int K, int cb, long stream)
{
    const int waves = (k / 16) * (n / 16);
    dim3 grid((waves * 32 + 255) / 256), block(256);
    hipStream_t st = (hipStream_t) stream;
#define DQ(bits) \
    if (cb == 2) hipLaunchKernelGGL((k_dq<bits, 2>), grid, block, 0, st, (const uint16_t*) trellis, (half*) out, k, n); \
    else hipLaunchKernelGGL((k_dq<bits, 1>), grid, block, 0, st, (const uint16_t*) trellis, (half*) out, k, n);
    switch (K)
    {
        case 1: DQ(1) break; case 2: DQ(2) break; case 3: DQ(3) break; case 4: DQ(4) break;
        case 5: DQ(5) break; case 6: DQ(6) break; case 7: DQ(7) break; case 8: DQ(8) break;
        default: return -2;
    }
#undef DQ
    return hipGetLastError() == hipSuccess ? 0 : -4;
}
