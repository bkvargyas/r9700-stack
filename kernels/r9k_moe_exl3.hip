// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas
//
// r9k_moe_exl3.hip -- decode-time routed experts on EXL3 trellis weights (gfx1201, wave32), two launches a MoE layer.
//
// Third-party code: the trellis decode and the RDNA "tiles" dot core below (marked EXLLAMAV3) are copied from
// exllamav3 v1.6.6 (https://github.com/turboderp-org/exllamav3, MIT, Copyright (c) turboderp):
// exllamav3_ext/rocm/quant/exl3_gemv_tiles_rdna.cuh (lane plan, fast pair decoder, exl3_tiles_step, the tile loop),
// exllamav3_ext/quant/exl3_dq.cuh (fshift) and exllamav3_ext/quant/hadamard_inner.cuh (shuffle_had_f4x32).
// Changes: integer K only (no half-integer rates), codebooks mcg / mul1 only, vector-typed halves instead of half2
// unions, and the tile loop takes per-row byte offsets (rows are arbitrary slots, not a strided slab). The decoded
// weights are bit-identical to ext.reconstruct (tests/test_exl3_moe_r9k.py checks every K through r9k_exl3_dq_tiles).
//
// Problem (notes/exl3-plan.md, Phase 2 P1): at decode an EP rank holds 48 of 384 experts, and a step of T tokens x 6
// picks has S = 6T (token, pick) slots of which ~S/8 are local. exllamav3's exl3_mgemm tier costs 13 launches at T = 1
// and 26 at T > 1 (rotation kernels, casts, index ops) and walks a grid over all S slots.
//
// Here, per layer:
//   k_gu  grid (I/128 output segments, KS1 input chunks, P group lanes); block = 8 waves.
//         Every block derives the same slot groups from the routing on device: local slots (g2l[id] in [0, E)) are
//         grouped by expert in slot order, at most R slots a group, so an expert picked by several tokens is
//         decoded once for all of them. Block (nb, ks, p) walks groups p, p + P, ...: rotates its rows' input chunk
//         (x * suh, 128-point Hadamard; gate and up have their own suh) into LDS, waves 0-3 run gate and 4-7 up, two
//         16-column tiles each, writes f32 partials, and the last of the KS1 blocks of (group, segment) to arrive
//         sums them in chunk order, applies the output Hadamard + svh to gate and up, the activation
//         (min(silu(g), L) * clamp(u, -L, L), L = act_limit; exllamav3's), and the down projection's input rotation
//         (suh_down, Hadamard) for that 128-block, writing the slot's rotated f16 activation a_rot.
//   k_dn  grid (H/128, KS2, P); block = 4 waves, same groups. Partials, last arriver applies the output Hadamard +
//         svh into a per-slot f32 row; the last slot of a token to finish a segment sums the token's local slots
//         x routing weight in pick order and writes y[t] (bf16 or f32). Tokens with no local pick get zeros.
// Arrival counters are zero at allocation and reset by their last arriver, so no memset and no host sync: the
// launch is HIP-graph safe and deterministic (every sum has a fixed order; grouping does not change any value).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdlib>

typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef float f4 __attribute__((ext_vector_type(4)));

namespace {

#ifndef R9K_EXL3_TPW
#define R9K_EXL3_TPW 2             // adjacent 16-column tiles a wave (shared input loads)
#endif
#ifndef R9K_EXL3_U
#define R9K_EXL3_U 0               // k-tiles whose loads go out together (0 = exllamav3's table)
#endif
constexpr int SMAX = 256;          // slots a launch (T * topk)
constexpr int MMAX = 8;            // rows a group
constexpr int TPW = R9K_EXL3_TPW;
constexpr int GU_WAVES = 2 * 8 / TPW;   // a 128-column segment of gate and of up
constexpr int DN_WAVES = 8 / TPW;
constexpr float RS128 = 0.088388347648f;   // 1/sqrt(128)

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
    if constexpr (R9K_EXL3_U > 0) return M >= 8 ? 1 : R9K_EXL3_U;
    if constexpr (M >= 8) return 1;
    if constexpr (T >= 2) return (bits <= 3) ? 2 : 1;
    return (bits <= 3) ? 4 : (bits <= 6 ? 2 : 1);
}

// The core: M rows x T adjacent 16-column tiles over nk k16-tiles. A: rows at byte offsets a_row[r] from A (LDS or
// global, f16, rotated basis); B: the matrix's first tile of this k-range / column range, b_stride bytes between
// k-tiles. out[t * M + r] = row r, column lane of tile t (lanes 0-15). All arguments but lane are wave-uniform.
template <int bits, int cb, int M, bool A_LDS, int T>
__device__ __forceinline__ void dot_tiles(const half* A, const uint32_t* a_row_in, const uint16_t* B, uint32_t b_stride,
                                          const int lane, const int nk, float* out)
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

__device__ __forceinline__ float bf16_f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

__device__ __forceinline__ uint16_t f_bf16(float f)
{
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1);
    return (uint16_t) (u >> 16);
}

struct Args
{
    const void* x;            // [T, H] bf16 (x_f16 = 0) or f16
    const int* ids;           // [T * topk] global expert ids
    const float* w;           // [T * topk] routing weights
    const int64_t* g2l;       // [G] global -> local id (outside [0, E) = not on this rank)
    const int64_t* tab[9];    // pointer tables [E]: gate trellis/suh/svh, up ..., down ...
    float* part1;             // [S, KS1, 2, I]
    float* part2;             // [S, KS2, H]  (may alias part1: k_dn starts after k_gu and never reads part1)
    half* a_rot;              // [S, I]
    float* out_slot;          // [S, H]
    int* ctr;                 // S * (I/128) + S * (H/128) + T * (H/128), zero between launches
    void* y;                  // [T, H] bf16 or f32
    int T, topk, S, G, E, H, I, R, ks1, ks2, x_f16, y_f32;
    float act_limit;
};

struct Setup
{
    int loc[SMAX];            // local expert of slot, -1 = not here
    int rank[SMAX];           // occurrences of the same expert before this slot
    int lead[SMAX];           // group g -> leader slot
    int rows[MMAX];           // current group's slots
    int nl, ng, m, e, last;
};

// Groups from the routing, identical in every block of both kernels. Returns false (block-uniform) when this
// block's group lane has nothing to do.
__device__ __forceinline__ bool setup(Setup& st, const Args& a, const int p, const bool keep_empty)
{
    const int tid = threadIdx.x, nt = blockDim.x;
    if (tid == 0) st.nl = 0;
    __syncthreads();
    int nloc = 0;
    for (int i = tid; i < a.S; i += nt)
    {
        int id = a.ids[i];
        int e = (id >= 0 && id < a.G) ? (int) a.g2l[id] : -1;
        e = (e >= 0 && e < a.E) ? e : -1;
        st.loc[i] = e;
        nloc += e >= 0;
    }
    if (nloc) atomicAdd(&st.nl, nloc);
    __syncthreads();
    if (p >= st.nl && !keep_empty) return false;
    for (int i = tid; i < a.S; i += nt)
    {
        const int e = st.loc[i];
        int r = 0;
        if (e >= 0)
            for (int j = 0; j < i; ++j) r += st.loc[j] == e;
        st.rank[i] = r;
    }
    __syncthreads();
    if (tid < 32)
    {
        int cnt = 0;
        for (int base = 0; base < a.S; base += 32)
        {
            const int i = base + tid;
            const bool f = i < a.S && st.loc[i] >= 0 && st.rank[i] % a.R == 0;
            const uint32_t mask = (uint32_t) __ballot(f);
            const int pos = cnt + __popc(mask & ((1u << tid) - 1u));
            if (f) st.lead[pos] = i;
            cnt += __popc(mask);
        }
        if (tid == 0) st.ng = cnt;
    }
    __syncthreads();
    return true;
}

// Rows of group g (thread 0), published by the caller's barrier
__device__ __forceinline__ void group_rows(Setup& st, const Args& a, const int g)
{
    const int l0 = st.lead[g];
    const int e = st.loc[l0];
    const int r1 = st.rank[l0] + a.R;
    int m = 0;
    for (int i = l0; i < a.S && m < a.R; ++i)
        if (st.loc[i] == e && st.rank[i] < r1) st.rows[m++] = i;
    st.m = m;
    st.e = e;
}

// Per-row dot dispatch: the row tile M is the smallest of 1, 2, 4, 8 covering m (rows past m re-read row m - 1)
template <int bits, int cb, bool A_LDS, int MM>
__device__ __forceinline__ void dot_rows(const int m, const half* A, const uint32_t* a_row, const uint16_t* B,
                                         uint32_t b_stride, int lane, int nk, float* out)
{
    if (m <= 1 || MM == 1) dot_tiles<bits, cb, 1, A_LDS, TPW>(A, a_row, B, b_stride, lane, nk, out);
    else if constexpr (MM >= 2)
    {
        if (m <= 2) dot_tiles<bits, cb, 2, A_LDS, TPW>(A, a_row, B, b_stride, lane, nk, out);
        else if constexpr (MM >= 4)
        {
            if (m <= 4) dot_tiles<bits, cb, 4, A_LDS, TPW>(A, a_row, B, b_stride, lane, nk, out);
            else if constexpr (MM >= 8) dot_tiles<bits, cb, 8, A_LDS, TPW>(A, a_row, B, b_stride, lane, nk, out);
        }
    }
}

__device__ __forceinline__ int mtile(int m) { return m <= 1 ? 1 : m <= 2 ? 2 : m <= 4 ? 4 : 8; }

// gate + up, activation, down-input rotation
template <int bits, int cb, int MM>
__global__ __launch_bounds__(GU_WAVES * 32) void k_gu(const Args a)
{
    extern __shared__ __align__(16) half sh_x[];      // [2][R][kc] rotated input chunk, gate then up
    __shared__ Setup st;
    const int nb = blockIdx.x, ks = blockIdx.y, p = blockIdx.z;
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (!setup(st, a, p, false)) return;
    const int kc = a.H / a.ks1, k0 = ks * kc, segs = kc / 128;
    const int NB = a.I / 128, n_tiles = a.I / 16;
    int* ctr = a.ctr;
    for (int g = p; g < st.ng; g += gridDim.z)
    {
        if (threadIdx.x == 0) group_rows(st, a, g);
        __syncthreads();
        const int m = st.m, e = st.e;

        // rotated input chunk: x[token] * suh, Hadamard, f16 into LDS; one wave a (row, matrix, 128-block)
        for (int task = wave; task < m * 2 * segs; task += GU_WAVES)
        {
            const int r = task / (2 * segs), mat = (task / segs) & 1, seg = task % segs;
            const int tok = st.rows[r] / a.topk;
            const int k = k0 + seg * 128 + lane * 4;
            float v[4];
            if (a.x_f16) ld_h4((const half*) a.x + (size_t) tok * a.H + k, v);
            else
            {
                uint2 u = *(const uint2*) ((const uint16_t*) a.x + (size_t) tok * a.H + k);
                v[0] = bf16_f(u.x & 0xffff); v[1] = bf16_f(u.x >> 16); v[2] = bf16_f(u.y & 0xffff); v[3] = bf16_f(u.y >> 16);
            }
            float s[4];
            ld_h4((const half*) a.tab[mat * 3 + 1][e] + k, s);
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] *= s[i];
            had128(v, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] *= RS128;
            st_h4(sh_x + ((size_t) (mat * a.R + r) * kc) + seg * 128 + lane * 4, v);
        }
        __syncthreads();

        {
            const int mat = wave / (GU_WAVES / 2);
            const int tile0 = nb * 8 + (wave % (GU_WAVES / 2)) * TPW;
            constexpr int tile_u16 = 16 * bits;
            const uint16_t* B = (const uint16_t*) a.tab[mat * 3][e] + ((size_t) (k0 / 16) * n_tiles + tile0) * tile_u16;
            uint32_t a_row[MM];
            #pragma unroll
            for (int r = 0; r < MM; ++r) a_row[r] = (uint32_t) ((r < m ? r : m - 1) * kc * 2);
            float out[TPW * MM];
            dot_rows<bits, cb, true, MM>(m, sh_x + (size_t) mat * a.R * kc, a_row, B, (uint32_t) n_tiles * tile_u16 * 2,
                                         lane, kc / 16, out);
            const int Mt = mtile(m);
            if (lane < 16)
                for (int r = 0; r < m; ++r)
                    #pragma unroll
                    for (int t = 0; t < TPW; ++t)
                        a.part1[(((size_t) st.rows[r] * a.ks1 + ks) * 2 + mat) * a.I + (tile0 + t) * 16 + lane] =
                            out[t * Mt + r];
        }
        __threadfence();
        __syncthreads();
        int* c = ctr + st.lead[g] * NB + nb;
        if (threadIdx.x == 0) st.last = atomicAdd(c, 1) == a.ks1 - 1;
        __syncthreads();
        if (st.last)
        {
            __threadfence();
            if (threadIdx.x == 0) *c = 0;
            for (int r = wave; r < m; r += GU_WAVES)
            {
                const int s = st.rows[r];
                const int col = nb * 128 + lane * 4;
                float gv[4] = {0, 0, 0, 0}, uv[4] = {0, 0, 0, 0};
                for (int k = 0; k < a.ks1; ++k)
                {
                    const f4 pg = *(const f4*) (a.part1 + (((size_t) s * a.ks1 + k) * 2 + 0) * a.I + col);
                    const f4 pu = *(const f4*) (a.part1 + (((size_t) s * a.ks1 + k) * 2 + 1) * a.I + col);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) { gv[i] += pg[i]; uv[i] += pu[i]; }
                }
                had128(gv, lane);
                had128(uv, lane);
                float sg[4], su[4], sd[4];
                ld_h4((const half*) a.tab[2][e] + col, sg);
                ld_h4((const half*) a.tab[5][e] + col, su);
                ld_h4((const half*) a.tab[7][e] + col, sd);
                const float L = a.act_limit;
                float v[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    const float gg = gv[i] * RS128 * sg[i];
                    float uu = uv[i] * RS128 * su[i];
                    float act = gg / (1.0f + __expf(-gg));
                    if (L > 0.0f) { act = fminf(act, L); uu = fminf(fmaxf(uu, -L), L); }
                    v[i] = act * uu * sd[i];
                }
                had128(v, lane);
                #pragma unroll
                for (int i = 0; i < 4; ++i) v[i] *= RS128;
                st_h4(a.a_rot + (size_t) s * a.I + col, v);
            }
        }
        __syncthreads();
    }
}

// down projection, output rotation, per-token weighted sum
template <int bits, int cb, int MM>
__global__ __launch_bounds__(DN_WAVES * 32) void k_dn(const Args a)
{
    __shared__ Setup st;
    const int nb = blockIdx.x, ks = blockIdx.y, p = blockIdx.z;
    const int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    const bool zero_lane = ks == 0 && p == 0;
    if (!setup(st, a, p, zero_lane)) return;
    const int NB = a.H / 128, n_tiles = a.H / 16;
    const int kc = a.I / a.ks2, k0 = ks * kc;
    int* ctr2 = a.ctr + a.S * (a.I / 128);
    int* ctr3 = ctr2 + a.S * NB;
    const int col = nb * 128 + lane * 4;

    if (zero_lane)    // tokens without a local pick: zeros (one wave a token)
        for (int t = wave; t < a.T; t += DN_WAVES)
        {
            int n = 0;
            for (int k = 0; k < a.topk; ++k) n += st.loc[t * a.topk + k] >= 0;
            if (n) continue;
            if (a.y_f32) *(f4*) ((float*) a.y + (size_t) t * a.H + col) = f4{0, 0, 0, 0};
            else *(uint2*) ((uint16_t*) a.y + (size_t) t * a.H + col) = make_uint2(0, 0);
        }

    for (int g = p; g < st.ng; g += gridDim.z)
    {
        if (threadIdx.x == 0) group_rows(st, a, g);
        __syncthreads();
        const int m = st.m, e = st.e;
        {
            const int tile0 = nb * 8 + wave * TPW;
            constexpr int tile_u16 = 16 * bits;
            const uint16_t* B = (const uint16_t*) a.tab[6][e] + ((size_t) (k0 / 16) * n_tiles + tile0) * tile_u16;
            uint32_t a_row[MM];
            #pragma unroll
            for (int r = 0; r < MM; ++r) a_row[r] = (uint32_t) (((size_t) st.rows[r < m ? r : m - 1] * a.I + k0) * 2);
            float out[TPW * MM];
            dot_rows<bits, cb, false, MM>(m, a.a_rot, a_row, B, (uint32_t) n_tiles * tile_u16 * 2, lane, kc / 16, out);
            const int Mt = mtile(m);
            if (lane < 16)
                for (int r = 0; r < m; ++r)
                    #pragma unroll
                    for (int t = 0; t < TPW; ++t)
                        a.part2[((size_t) st.rows[r] * a.ks2 + ks) * a.H + (tile0 + t) * 16 + lane] = out[t * Mt + r];
        }
        __threadfence();
        __syncthreads();
        int* c = ctr2 + st.lead[g] * NB + nb;
        if (threadIdx.x == 0) st.last = atomicAdd(c, 1) == a.ks2 - 1;
        __syncthreads();
        if (st.last)
        {
            __threadfence();
            if (threadIdx.x == 0) *c = 0;
            for (int r = wave; r < m; r += DN_WAVES)
            {
                const int s = st.rows[r], t = s / a.topk;
                float v[4] = {0, 0, 0, 0};
                for (int k = 0; k < a.ks2; ++k)
                {
                    const f4 pp = *(const f4*) (a.part2 + ((size_t) s * a.ks2 + k) * a.H + col);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) v[i] += pp[i];
                }
                had128(v, lane);
                float sv[4];
                ld_h4((const half*) a.tab[8][e] + col, sv);
                #pragma unroll
                for (int i = 0; i < 4; ++i) v[i] *= RS128 * sv[i];
                *(f4*) (a.out_slot + (size_t) s * a.H + col) = f4{v[0], v[1], v[2], v[3]};
                __threadfence();
                int n = 0;
                for (int k = 0; k < a.topk; ++k) n += st.loc[t * a.topk + k] >= 0;
                int old = 0;
                int* c3 = ctr3 + t * NB + nb;
                if (lane == 0) old = atomicAdd(c3, 1);
                old = __shfl(old, 0, 32);
                if (old != n - 1) continue;
                __threadfence();
                if (lane == 0) *c3 = 0;
                float acc[4] = {0, 0, 0, 0};
                for (int k = 0; k < a.topk; ++k)
                {
                    const int s2 = t * a.topk + k;
                    if (st.loc[s2] < 0) continue;
                    const float wk = a.w[s2];
                    const f4 o = *(const f4*) (a.out_slot + (size_t) s2 * a.H + col);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) acc[i] += wk * o[i];
                }
                if (a.y_f32) *(f4*) ((float*) a.y + (size_t) t * a.H + col) = f4{acc[0], acc[1], acc[2], acc[3]};
                else
                {
                    uint16_t* yb = (uint16_t*) a.y + (size_t) t * a.H + col;
                    *(uint2*) yb = make_uint2((uint32_t) f_bf16(acc[0]) | ((uint32_t) f_bf16(acc[1]) << 16),
                                              (uint32_t) f_bf16(acc[2]) | ((uint32_t) f_bf16(acc[3]) << 16));
                }
            }
        }
        __syncthreads();
    }
}

// Debug / validation: decode a whole trellis [k/16, n/16, 16 bits] to f16 [k, n] (rotated basis, = ext.reconstruct)
template <int bits, int cb>
__global__ __launch_bounds__(256) void k_dq_tiles(const uint16_t* trellis, half* out, int k, int n)
{
    const int wave = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    const int nt = n / 16;
    if (wave >= (k / 16) * nt) return;
    const int kt = wave / nt, ntile = wave % nt;
    g_char* b = (g_char*) (const char*) (trellis + (size_t) wave * 16 * bits);
    const LanePlan<bits> lp = lane_plan<bits>(lane);
    uint32_t raw[LanePlan<bits>::R];
    #pragma unroll
    for (int i = 0; i < LanePlan<bits>::R; ++i) raw[i] = ld_g(b, lp.off[i]);
    Frag f0, f1;
    dq_tile<bits, cb>(raw, lp, f0, f1);
    const int r0 = (lane & 3) * 2, cA = (lane / 8) * 2 + ((lane >> 2) & 1), cB = cA + 8;
    _Float16* o = (_Float16*) out;
    auto at = [&](int r, int c) -> _Float16& { return o[(size_t) (kt * 16 + r) * n + ntile * 16 + c]; };
    at(r0, cA) = f0.v[0].x; at(r0 + 1, cA) = f0.v[0].y; at(r0 + 8, cA) = f0.v[1].x; at(r0 + 9, cA) = f0.v[1].y;
    at(r0, cB) = f1.v[0].x; at(r0 + 1, cB) = f1.v[0].y; at(r0 + 8, cB) = f1.v[1].x; at(r0 + 9, cB) = f1.v[1].y;
}

template <int bits, int cb, int MM>
int launch(const Args& a, int P, hipStream_t st)
{
    const int kc = a.H / a.ks1;
    const size_t smem = (size_t) 2 * a.R * kc * sizeof(half);
    if (smem > 48 * 1024) return -3;
#ifdef R9K_EXL3_DEBUG_ONLY
    const char* only = getenv("R9K_EXL3_ONLY");     // timing one kernel alone: "gu" or "dn"
    if (!only || only[0] != 'd')
#endif
    hipLaunchKernelGGL((k_gu<bits, cb, MM>), dim3(a.I / 128, a.ks1, P), dim3(GU_WAVES * 32), smem, st, a);
#ifdef R9K_EXL3_DEBUG_ONLY
    if (!only || only[0] != 'g')
#endif
    hipLaunchKernelGGL((k_dn<bits, cb, MM>), dim3(a.H / 128, a.ks2, P), dim3(DN_WAVES * 32), 0, st, a);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

template <int cb, int MM>
int launch_k(int K, const Args& a, int P, hipStream_t st)
{
    switch (K)
    {
        case 2: return launch<2, cb, MM>(a, P, st);
        case 3: return launch<3, cb, MM>(a, P, st);
        case 4: return launch<4, cb, MM>(a, P, st);
        case 5: return launch<5, cb, MM>(a, P, st);
        case 6: return launch<6, cb, MM>(a, P, st);
        default: return -2;
    }
}

}  // namespace

// Scratch the caller provides (bytes): part1 4*S*ks1*2*I, part2 4*S*ks2*H (one buffer may serve both), a_rot 2*S*I,
// out_slot 4*S*H, and
// r9k_moe_exl3_ctr_ints(T, topk, H, I) int32 counters, zeroed once at allocation (each launch leaves them zero).
extern "C" int r9k_moe_exl3_ctr_ints(int T, int topk, int H, int I)
{
    const int S = T * topk;
    return S * (I / 128) + S * (H / 128) + T * (H / 128);
}

extern "C" int r9k_moe_exl3_max_slots(void) { return SMAX; }

// One MoE layer at decode: y[T, H] = sum over this rank's (token, pick) slots of w * down(act(gate x, up x)).
// tabs: 9 int64 device tables of E pointers (gate trellis, suh, svh, up ..., down ...). K: trellis bits (2-6, the
// same for the three roles). cb: 1 mcg, 2 mul1. R: rows a group (1..8; 1 = no sharing). P: group lanes (grid z).
// ks1 / ks2: input chunks of gate|up (H / ks1 a multiple of 128) and down (I / ks2 a multiple of 128).
extern "C" int r9k_moe_exl3_decode(long x, int x_f16, long ids, long w, long g2l, int G, int E, int T, int topk,
                                   int H, int I, long tabs, int K, int cb, float act_limit,
                                   long part1, long part2, long a_rot, long out_slot, long ctr,
                                   long y, int y_f32, int R, int P, int ks1, int ks2, long stream)
{
    const int S = T * topk;
    if (T < 1 || S > SMAX || R < 1 || R > MMAX || P < 1 || ks1 < 1 || ks2 < 1) return -1;
    if (H % (128 * ks1) || I % (128 * ks2) || H % 128 || I % 128) return -1;
    if (cb != 1 && cb != 2) return -2;
    Args a;
    a.x = (const void*) x; a.ids = (const int*) ids; a.w = (const float*) w; a.g2l = (const int64_t*) g2l;
    for (int i = 0; i < 9; ++i) a.tab[i] = ((const int64_t* const*) tabs)[i];
    a.part1 = (float*) part1; a.part2 = (float*) part2; a.a_rot = (half*) a_rot; a.out_slot = (float*) out_slot;
    a.ctr = (int*) ctr; a.y = (void*) y;
    a.T = T; a.topk = topk; a.S = S; a.G = G; a.E = E; a.H = H; a.I = I; a.R = R; a.ks1 = ks1; a.ks2 = ks2;
    a.x_f16 = x_f16; a.y_f32 = y_f32; a.act_limit = act_limit;
    hipStream_t st = (hipStream_t) stream;
    constexpr int CB2 = 2;
    if (R == 1) return cb == 2 ? launch_k<CB2, 1>(K, a, P, st) : launch_k<1, 1>(K, a, P, st);
    if (R <= 4) return cb == 2 ? launch_k<CB2, 4>(K, a, P, st) : launch_k<1, 4>(K, a, P, st);
    return cb == 2 ? launch_k<CB2, 8>(K, a, P, st) : launch_k<1, 8>(K, a, P, st);
}

extern "C" int r9k_exl3_dq_tiles(long trellis, long out, int k, int n, int K, int cb, long stream)
{
    const int waves = (k / 16) * (n / 16);
    dim3 grid((waves * 32 + 255) / 256), block(256);
    hipStream_t st = (hipStream_t) stream;
#define DQ(bits) \
    if (cb == 2) hipLaunchKernelGGL((k_dq_tiles<bits, 2>), grid, block, 0, st, (const uint16_t*) trellis, (half*) out, k, n); \
    else hipLaunchKernelGGL((k_dq_tiles<bits, 1>), grid, block, 0, st, (const uint16_t*) trellis, (half*) out, k, n);
    switch (K)
    {
        case 1: DQ(1) break; case 2: DQ(2) break; case 3: DQ(3) break; case 4: DQ(4) break;
        case 5: DQ(5) break; case 6: DQ(6) break; case 7: DQ(7) break; case 8: DQ(8) break;
        default: return -2;
    }
#undef DQ
    return hipGetLastError() == hipSuccess ? 0 : -4;
}
