// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas

// DeepSeek-V4.1 manifold hyper-connections (mHC) at decode widths for gfx1201: independent implementations of the
// semantics of vLLM's torch reference (model_executor/kernels/mhc/torch.py: mhc_post_torch, mhc_pre_delayed_torch;
// Apache-2.0). Stock runs a seam as three TileLang launches (mhc_post, hc_prenorm_gemm, mhc_pre_big_fuse: 27 us at
// M=1 on an R9700, the GEMM on two workgroups); here a seam is one launch.
//
// seam (post folded into the delayed pre), per token n, streams i, j < 4, h < H, K = 4H (or H for the broadcast x):
//   r'[j,h] = bf16( fma-chain_i(comb_in[i,j] * r[i,h]) + post_in[j] * so[h] )      (no sublayer_out: r' = r)
//   li[h]   = bf16( ((p0 r'0 + p1 r'1) + p2 r'2) + p3 r'3 ),  p = pre_in            (no pre_in: li = r'0)
//   mixes   = (x . fn^T) * rsqrt(sum(x^2) * (1/K) + rms_eps),  x = r' flattened (or the broadcast x)
//   pre_out = sigmoid(mixes[0:4] * s0 + b) + pre_eps;  post_out = sigmoid(mixes[4:8] * s1 + b) * post_mult
//   comb    = sinkhorn(softmax(mixes[8:24] * s2 + b)) (iters - 1 row/column passes after the first column pass)
// Rounding points and the summation orders of torch's ops on this GPU (probed, 2026-11-02): einsum = an fma chain
// from i = 0; the collapse = separate products added in stream order; softmax sums (e0 + e2) + (e1 + e3) and
// divides; row sums (a + b) + (c + d); column sums in row order; sigmoid 1 / (1 + exp(-x)); mean = sum * (1/K).
// r' and li are bit-exact against the reference. The gates are not: the 24 K-long dot products and the sum of squares
// run in a different order than torch's matmul (as stock's TileLang GEMM does), and the sinkhorn divides by a refined
// reciprocal (ndiv); they agree with torch to ~1e-5 relative.
//
// Grid (H / 64 chunks, ceil(T / TG) token groups), 256 threads. A workgroup owns h in [64 kc, 64 kc + 64) of all four
// streams: wave w applies the post and collapse of token w of its group (2 h per lane, every r read once), the new
// streams go to LDS as the GEMV input; then wave w forms mixes 3w..3w+2 over the chunk's 256 K (its fn slice held in
// registers, read once for the whole group) and wave 0 the sum of squares; per-chunk partials go to scratch, and the
// last workgroup of the group to arrive (ticket counter, reset for the next launch) reduces them in a fixed order --
// 8 runs of chunks, then the runs in order: independent of T and of which workgroup is last, so results are
// batch-invariant and bitwise reproducible -- and runs the gates and the sinkhorn, 16 lanes per token.
// post (standalone MHCPostOp): one wave per (token, 64 h), the same arithmetic as the seam's first stage.
// No device printf/assert; graph-safe (no host syncs, the counters self-reset).
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int HC = 4, NMIX = 24, CH = 64, NT = 256, NW = 8, MAXT = 32, NV = NMIX + 1, NSL = 8, RLMAX = 10;
typedef float v4f __attribute__((ext_vector_type(4)));
constexpr int PV = 28, NQ = PV / 4;                  // partials a token: 24 mixes + sum of squares, padded to float4s

__device__ __forceinline__ float bf2f(uint16_t u) { return __uint_as_float((unsigned)u << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {    // round to nearest even (torch's conversion)
  unsigned u = __float_as_uint(f);
  if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0xffffu) ? 0x40u : 0u));
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
// x / d for the sinkhorn's normalizations (d ~ 1, never denormal): v_rcp plus one Newton step, then the product --
// within 1.5 ulp of IEEE division at a third of its dependent latency (20 iterations x 2 passes are the kernel's
// serial tail). Not bit-exact with torch's division; the gates already differ from it by the GEMV's order.
__device__ __forceinline__ float ndiv(float x, float d) {
  float y = __builtin_amdgcn_rcpf(d);
  y = fmaf(fmaf(-d, y, 1.f), y, y);
  return x * y;
}
// DPP lane moves (no LDS round trip): quad_perm 0x00-0xff, row_ror:n = 0x120 + n within each 16-lane row
template <int CTRL>
__device__ __forceinline__ float dpp(float v) {
  return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xf, 0xf, false));
}
template <int CTRL>
__device__ __forceinline__ int dppi(int v) { return __builtin_amdgcn_update_dpp(0, v, CTRL, 0xf, 0xf, false); }
// wave32 sum, wave-uniform result: quads, then rows of 16 by rotation (every lane of a row holds the same sum),
// then the other row's sum (permlanex16; a + b == b + a, so every lane holds the same total). Deterministic.
__device__ __forceinline__ float wave_sum(float v) {
  v += dpp<0xb1>(v);                                   // quad_perm [1,0,3,2]
  v += dpp<0x4e>(v);                                   // quad_perm [2,3,0,1]
  v += dpp<0x124>(v);                                  // row_ror:4
  v += dpp<0x128>(v);                                  // row_ror:8
  return v + __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210, 0xfedcba98, false, false));
}

#ifdef R9K_MHC_PROF                                     // phase timestamps for tuning (not in libr9k)
__device__ long g_prof[2048][8];
#define PROF(i) do { if (threadIdx.x == 0) g_prof[blockIdx.y * gridDim.x + blockIdx.x][i] = wall_clock64(); } while (0)
#else
#define PROF(i) do { } while (0)
#endif

struct SeamArgs {
  const uint16_t* res;     // [T, 4, H] bf16
  const uint16_t* so;      // [T, H] bf16 sublayer output, or null (no post)
  const float* post_in;    // [T, 4]
  const float* comb_in;    // [T, 4, 4]
  const uint16_t* xin;     // [T, H] bf16 broadcast projection input, or null (project r')
  const float* fn;         // [24, K]
  const float* scale;      // [3]
  const float* base;       // [24]
  const float* pre_in;     // [T, 4], or null (select stream 0)
  uint16_t* res_out;       // [T, 4, H] (when so)
  float* post_out;         // [T, 4]
  float* comb_out;         // [T, 16]
  uint16_t* li;            // [T, H]
  float* pre_out;          // [T, 4]
  float* part;             // [H / 64][MAXT][PV] scratch
  int* ctr;                // [MAXT] arrival counters, zero between launches
  int T, H, K, iters;
  float rms_eps, pre_eps, sk_eps, post_mult;
};

// r'[j] for one token at h, h+1 (two bf16 of each stream packed in a uint32), the post folded in when so != null.
__device__ __forceinline__ void post_pair(const SeamArgs& a, int n, int h, unsigned (&rv)[HC], unsigned (&nv)[HC]) {
  const uint16_t* rp = a.res + (long)n * HC * a.H + h;
#pragma unroll
  for (int i = 0; i < HC; i++) rv[i] = *reinterpret_cast<const unsigned*>(rp + (long)i * a.H);
  if (!a.so) {
#pragma unroll
    for (int j = 0; j < HC; j++) nv[j] = rv[j];
    return;
  }
  const unsigned sv = *reinterpret_cast<const unsigned*>(a.so + (long)n * a.H + h);
  const float* cm = a.comb_in + (long)n * HC * HC;
  const float* pm = a.post_in + (long)n * HC;
#pragma unroll
  for (int j = 0; j < HC; j++) {
    unsigned o = 0;
#pragma unroll
    for (int e = 0; e < 2; e++) {
      float m = cm[j] * bf2f((uint16_t)(rv[0] >> (16 * e)));
#pragma unroll
      for (int i = 1; i < HC; i++) m = fmaf(cm[i * HC + j], bf2f((uint16_t)(rv[i] >> (16 * e))), m);
      const float pt = pm[j] * bf2f((uint16_t)(sv >> (16 * e)));
      o |= (unsigned)f2bf(m + pt) << (16 * e);
    }
    nv[j] = o;
  }
}

__device__ __forceinline__ unsigned collapse_pair(const float* p, const unsigned (&nv)[HC]) {
  unsigned o = 0;
#pragma unroll
  for (int e = 0; e < 2; e++) {
    float acc = p[0] * bf2f((uint16_t)(nv[0] >> (16 * e)));
#pragma unroll
    for (int s = 1; s < HC; s++) {
      const float prod = p[s] * bf2f((uint16_t)(nv[s] >> (16 * e)));
      acc = acc + prod;
    }
    o |= (unsigned)f2bf(acc) << (16 * e);
  }
  return o;
}

template <int NS, int TG>
__global__ void __launch_bounds__(NT) r9k_mhc_seam_kernel(SeamArgs a) {
  constexpr int KL = NS * CH, PL = KL / 32;              // K elements of the chunk, per lane
  __shared__ float xs[TG][KL];
  __shared__ float runs[TG][NV][NSL];
  __shared__ float red[TG][NV];
  __shared__ int last;
  const int t = threadIdx.x, lane = t & 31, w = t >> 5;
  const int kc = blockIdx.x, nkc = gridDim.x, g = blockIdx.y;
  const int h0 = kc * CH, t0 = g * TG, nt = min(TG, a.T - t0);

  PROF(0);
  float f[3][PL];                                        // fn[3w + o][k(lane + 32 i)], in flight during the post
#pragma unroll
  for (int o = 0; o < 3; o++)
#pragma unroll
    for (int i = 0; i < PL; i++) {
      const int kl = lane + 32 * i, s = kl / CH, hl = kl - s * CH;
      f[o][i] = a.fn[(long)(3 * w + o) * a.K + (long)s * a.H + h0 + hl];
    }

  if (w < nt) {
    const int n = t0 + w, h = h0 + 2 * lane;
    unsigned rv[HC], nv[HC];
    post_pair(a, n, h, rv, nv);
    if (a.so) {
      uint16_t* op = a.res_out + (long)n * HC * a.H + h;
#pragma unroll
      for (int j = 0; j < HC; j++) *reinterpret_cast<unsigned*>(op + (long)j * a.H) = nv[j];
    }
    unsigned lv;
    if (a.pre_in) {
      float p[HC];
#pragma unroll
      for (int s = 0; s < HC; s++) p[s] = a.pre_in[(long)n * HC + s];
      lv = collapse_pair(p, nv);
    } else {
      lv = nv[0];
    }
    *reinterpret_cast<unsigned*>(a.li + (long)n * a.H + h) = lv;
    if constexpr (NS == HC) {
#pragma unroll
      for (int j = 0; j < HC; j++) {
        xs[w][j * CH + 2 * lane] = bf2f((uint16_t)nv[j]);
        xs[w][j * CH + 2 * lane + 1] = bf2f((uint16_t)(nv[j] >> 16));
      }
    } else {
      const unsigned xv = *reinterpret_cast<const unsigned*>(a.xin + (long)n * a.H + h);
      xs[w][2 * lane] = bf2f((uint16_t)xv);
      xs[w][2 * lane + 1] = bf2f((uint16_t)(xv >> 16));
    }
  }
  __syncthreads();
  PROF(1);

#pragma unroll
  for (int tt = 0; tt < TG; tt++) {
    if (tt < nt) {
      float acc[3] = {0.f, 0.f, 0.f}, sq = 0.f;
#pragma unroll
      for (int i = 0; i < PL; i++) {
        const float x = xs[tt][lane + 32 * i];
#pragma unroll
        for (int o = 0; o < 3; o++) acc[o] = fmaf(x, f[o][i], acc[o]);
        sq = fmaf(x, x, sq);
      }
      float* pp = a.part + ((long)kc * MAXT + t0 + tt) * PV;
#pragma unroll
      for (int o = 0; o < 3; o++) {
        const float v = wave_sum(acc[o]);
        if (lane == 0) pp[3 * w + o] = v;
      }
      if (w == 0) {
        const float v = wave_sum(sq);
        if (lane == 0) pp[NMIX] = v;
      }
    }
  }
  PROF(2);
  __threadfence();
  __syncthreads();
  PROF(3);
  if (t == 0) last = (atomicAdd(&a.ctr[g], 1) == nkc - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  PROF(4);

  // runs of rl consecutive chunks per (token, float4 of values): a thread's RLMAX float4 loads are all issued before
  // the adds, (token, quad) fastest across lanes (coalesced); 1 round for TG <= 4, 2 for TG = 8
  const int rl = (nkc + NSL - 1) / NSL, items = nt * NQ * NSL;
  for (int it = t; it < items; it += NT) {
    const int jq = it % (nt * NQ), sl = it / (nt * NQ), tt = jq / NQ, v0 = 4 * (jq % NQ), k0 = sl * rl;
    const v4f* pp = reinterpret_cast<const v4f*>(a.part + ((long)k0 * MAXT + t0) * PV) + jq;
    v4f ld[RLMAX];
#pragma unroll
    for (int u = 0; u < RLMAX; u++)
      ld[u] = (u < rl && k0 + u < nkc) ? __builtin_nontemporal_load(pp + (long)u * MAXT * NQ) : (v4f)(0.f);
    v4f sm = ld[0];
#pragma unroll
    for (int u = 1; u < RLMAX; u++)
      if (u < rl && k0 + u < nkc) sm += ld[u];
    runs[tt][v0][sl] = sm.x;
    if (v0 + 1 < NV) runs[tt][v0 + 1][sl] = sm.y;
    if (v0 + 2 < NV) runs[tt][v0 + 2][sl] = sm.z;
    if (v0 + 3 < NV) runs[tt][v0 + 3][sl] = sm.w;
  }
  __syncthreads();
  for (int it = t; it < nt * NV; it += NT) {
    const int v = it % NV, tt = it / NV;
    float s = runs[tt][v][0];
#pragma unroll
    for (int sl = 1; sl < NSL; sl++) s += runs[tt][v][sl];
    red[tt][v] = s;
  }
  __syncthreads();

  PROF(5);
  // gates + sinkhorn: 16 lanes per token (e = 4 r + c), two tokens a wave. (One lane per token holding the block in
  // registers was 3x slower: 640 serial divisions on one lane are issue-bound.)
  const int tt = t >> 4, e = t & 15, r = e >> 2;
  if (tt >= TG) return;
  const bool live = tt < nt;
  const int ts = live ? tt : 0;
  const int n = t0 + ts;
  const float rr = rsqrtf(red[ts][NMIX] * (1.f / (float)a.K) + a.rms_eps);
  const float s0 = a.scale[0], s1 = a.scale[1], s2 = a.scale[2];
  if (live && e < HC) {
    a.pre_out[(long)n * HC + e] = sigm(red[ts][e] * rr * s0 + a.base[e]) + a.pre_eps;
    a.post_out[(long)n * HC + e] = sigm(red[ts][HC + e] * rr * s1 + a.base[HC + e]) * a.post_mult;
  }
  // row r = quad r of the token's 16 lanes (quad_perm broadcasts); column partners by row_ror 4/8/12, whose source
  // rows are found by rotating r itself (direction-agnostic), to add the column in row order
  const int sr1 = dppi<0x124>(r), sr2 = dppi<0x128>(r);           // the third copy is the remaining row
  auto col_sum = [&](float x) {
    const float y1 = dpp<0x124>(x), y2 = dpp<0x128>(x), y3 = dpp<0x12c>(x);
    float cv[HC];
#pragma unroll
    for (int k = 0; k < HC; k++) cv[k] = r == k ? x : sr1 == k ? y1 : sr2 == k ? y2 : y3;
    return ((cv[0] + cv[1]) + cv[2]) + cv[3];
  };
  float x = red[ts][2 * HC + e] * rr * s2 + a.base[2 * HC + e];
  {
    const float v0 = dpp<0x00>(x), v1 = dpp<0x55>(x), v2 = dpp<0xaa>(x), v3 = dpp<0xff>(x);
    const float mx = fmaxf(fmaxf(v0, v1), fmaxf(v2, v3));
    const float ex = expf(x - mx);
    const float e0 = dpp<0x00>(ex), e1 = dpp<0x55>(ex), e2 = dpp<0xaa>(ex), e3 = dpp<0xff>(ex);
    x = ex / ((e0 + e2) + (e1 + e3)) + a.sk_eps;               // torch's softmax order
  }
  x = ndiv(x, col_sum(x) + a.sk_eps);
  for (int it = 1; it < a.iters; it++) {
    const float v0 = dpp<0x00>(x), v1 = dpp<0x55>(x), v2 = dpp<0xaa>(x), v3 = dpp<0xff>(x);
    x = ndiv(x, ((v0 + v1) + (v2 + v3)) + a.sk_eps);
    x = ndiv(x, col_sum(x) + a.sk_eps);
  }
  if (live) a.comb_out[(long)n * HC * HC + e] = x;
  if (t == 0) a.ctr[g] = 0;
  PROF(6);
}

// standalone post: grid (H / 256, T), wave w of a block owns 64 h of one token
__global__ void __launch_bounds__(NT) r9k_mhc_post_kernel(SeamArgs a) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int n = blockIdx.y, h = (blockIdx.x * NW + w) * CH + 2 * lane;
  if (h >= a.H) return;
  unsigned rv[HC], nv[HC];
  post_pair(a, n, h, rv, nv);
  uint16_t* op = a.res_out + (long)n * HC * a.H + h;
#pragma unroll
  for (int j = 0; j < HC; j++) *reinterpret_cast<unsigned*>(op + (long)j * a.H) = nv[j];
}

template <int NS>
int launch_seam(const SeamArgs& a, hipStream_t st) {
  const int tg = a.T <= 1 ? 1 : a.T <= 2 ? 2 : a.T <= 4 ? 4 : 8;
  dim3 grid(a.H / CH, (a.T + tg - 1) / tg), block(NT);
  switch (tg) {
    case 1: r9k_mhc_seam_kernel<NS, 1><<<grid, block, 0, st>>>(a); break;
    case 2: r9k_mhc_seam_kernel<NS, 2><<<grid, block, 0, st>>>(a); break;
    case 4: r9k_mhc_seam_kernel<NS, 4><<<grid, block, 0, st>>>(a); break;
    default: r9k_mhc_seam_kernel<NS, 8><<<grid, block, 0, st>>>(a); break;
  }
  return (int)hipGetLastError();
}

}  // namespace

#ifdef R9K_MHC_PROF
extern "C" int r9k_mhc_prof_copy(long dst) { return (int)hipMemcpyFromSymbol((void*)dst, HIP_SYMBOL(g_prof), sizeof(g_prof)); }
#endif
extern "C" int r9k_mhc_max_tokens() { return MAXT; }
// scratch the caller provides: part floats, counter ints (zeroed once)
extern "C" long r9k_mhc_scratch_floats(int H) { return (long)(H / CH) * MAXT * PV; }
extern "C" int r9k_mhc_scratch_ctrs() { return MAXT; }

// All tensors contiguous; so / xin / pre_in may be 0. K = 4H (project r') or H (xin). Returns 0, <0 unsupported,
// or a hip error.
extern "C" int r9k_mhc_seam(long res, long so, long post_in, long comb_in, long xin, long fn, long scale, long base,
                            long pre_in, long res_out, long post_out, long comb_out, long li, long pre_out,
                            long part, long ctr, int T, int H, int K, int iters, float rms_eps, float pre_eps,
                            float sk_eps, float post_mult, long stream) {
  if (T < 0 || T > MAXT || H % CH || H <= 0 || H / CH > NSL * RLMAX || iters < 1) return -1;
  if (xin ? (K != H || so) : (K != HC * H)) return -2;
  if (so && (!post_in || !comb_in || !res_out)) return -3;
  if (T == 0) return 0;
  SeamArgs a{(const uint16_t*)res, (const uint16_t*)so, (const float*)post_in, (const float*)comb_in,
             (const uint16_t*)xin, (const float*)fn, (const float*)scale, (const float*)base, (const float*)pre_in,
             (uint16_t*)res_out, (float*)post_out, (float*)comb_out, (uint16_t*)li, (float*)pre_out, (float*)part,
             (int*)ctr, T, H, K, iters, rms_eps, pre_eps, sk_eps, post_mult};
  return xin ? launch_seam<1>(a, (hipStream_t)stream) : launch_seam<HC>(a, (hipStream_t)stream);
}

extern "C" int r9k_mhc_post(long res, long so, long post_in, long comb_in, long res_out, int T, int H, long stream) {
  if (T < 0 || H % (CH * NW)) return -1;
  if (T == 0) return 0;
  SeamArgs a{};
  a.res = (const uint16_t*)res; a.so = (const uint16_t*)so; a.post_in = (const float*)post_in;
  a.comb_in = (const float*)comb_in; a.res_out = (uint16_t*)res_out; a.T = T; a.H = H;
  dim3 grid(H / (CH * NW), T), block(NT);
  r9k_mhc_post_kernel<<<grid, block, 0, (hipStream_t)stream>>>(a);
  return (int)hipGetLastError();
}
