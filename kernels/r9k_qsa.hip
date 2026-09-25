// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas

// QSA sparse attention for gfx1201 (RDNA4, wave32): replaces vLLM's Triton qsa_sparse_paged_attention for
// Qwen4Exp (Qwen3.8-Flash-Next) full-attention layers.
//
// Independent implementation, written from the public online-softmax (FlashAttention) algorithm, the gfx12 ISA, our
// own r9k_attn.hip, and vLLM's Apache-2.0 Triton QSA kernels as the reference for the interface semantics. No
// third-party attention code consulted (libr4d in particular was not read or disassembled).
//
// What QSA computes. Every query row attends to its own selection of keys: the indexer's top-k complete 4-token
// groups (512 groups = 2048 tokens) plus the causal tail of the group the row sits in. vLLM hands the attention the
// expanded token list per row (width topk + 3, -1 padded) and gathers K/V row by row: ~2 MB of K+V reads per query
// row per layer, and adjacent rows re-read almost the same tokens.
//
// Exact reformulation used here. A selection is always a union of whole groups plus the open group's tokens <= the
// row's position, so it is fully described by a bitmap over groups plus the causal rule "key <= p". Kernel 1 turns
// each row's token list into that bitmap (bit g set iff some listed token lies in group g). Kernel 2 then processes
// a tile of 16 consecutive query rows of one request together: it walks the UNION of their groups in id order,
// stages 16 keys (4 groups) at a time into LDS once for the whole tile, and runs the r9k_attn WMMA pipeline over
// all 16 rows x GQA heads, masking each (row, key) by that row's bit and key <= p. The set of keys each row attends
// to is exactly the stock kernel's; only the summation order differs (fp32 online softmax either way).
//
// Mapping (same as r9k_attn): one workgroup = (tile of 16 rows, kv head, split); two waves per q head, each owning
// half of the 256-wide head dim; S^T = K . Q^T and O^T += V^T . P^T with WMMA 16x16x16 bf16, one query row per
// lane column. K staged coalesced, V staged with hardware-transposed loads from per-lane gathered addresses.
// Splits (decode: few rows, so few workgroups) partition the group-id range; kernel 3 merges them by LSE.
//
// Geometry: head_size 256, GQA <= 12, bf16 K/V in any paged layout given by element strides (block, token, head),
// page size a multiple of 4. No device printf/assert (the build enforces hostcall-free objects).
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

typedef __bf16 v8bf16 __attribute__((ext_vector_type(8)));
typedef short v8s __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef unsigned v4u __attribute__((ext_vector_type(4)));
typedef __bf16 v2bf16 __attribute__((ext_vector_type(2)));

constexpr int HD = 256;
constexpr int HH = HD / 2;
constexpr int DT = HH / 16;
constexpr int GS = 4;                 // tokens per QSA group (indexer_compress_ratio)
constexpr int TILE = 16;              // query rows per workgroup
constexpr float LOG2E = 1.4426950408889634f;
#ifdef R9K_QSA_NOLAZY
constexpr float RESCALE_THRESH = 0.f;     // debug: rescale on every increase of the row max
#else
constexpr float RESCALE_THRESH = 8.f;
#endif

__device__ __forceinline__ float lane_swap16(float x) {
  return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(x), __float_as_int(x), 0x76543210, 0xFEDCBA98,
                                                     false, false));
}
__device__ __forceinline__ void lds_barrier() {
#ifdef R9K_QSA_SYNC
  __syncthreads();                          // debug: full barrier instead of the LDS-only one
#else
  asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory");
#endif
}
__device__ __forceinline__ v8bf16 ld_frag(const uint16_t* p) { return *reinterpret_cast<const v8bf16*>(p); }
__device__ __forceinline__ v8bf16 ld_frag_tr(const uint16_t* p) {
  v8s r = __builtin_amdgcn_global_load_tr_b128_v8i16((v8s*)p);
  return __builtin_bit_cast(v8bf16, r);
}
__device__ __forceinline__ unsigned pack_hi16(float lo, float hi) {
  return __builtin_amdgcn_perm(__float_as_uint(hi), __float_as_uint(lo), 0x07060302u);
}
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
__device__ __forceinline__ v8bf16 f32x8_to_bf16_trunc(v8f f) {
  const v4u r = {pack_hi16(f[0], f[1]), pack_hi16(f[2], f[3]), pack_hi16(f[4], f[5]), pack_hi16(f[6], f[7])};
  return __builtin_bit_cast(v8bf16, r);
}
__device__ __forceinline__ float bf16x8_sum(v8bf16 v, float acc) {
  const v2bf16 one = {(__bf16)1.f, (__bf16)1.f};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const v2bf16 pr = {v[2 * i], v[2 * i + 1]};
    acc = __builtin_amdgcn_fdot2_f32_bf16(pr, one, acc, false);
  }
  return acc;
}

// Staged K / V^T layout, identical to r9k_attn.hip (bank-conflict-free fragment reads).
constexpr int LDS_K = 16 * 512;
constexpr int LDS_V = 256 * 32;
constexpr int LDS_BUF = LDS_K + LDS_V;
__device__ __forceinline__ int lds_k_off(int row, int chunk) { return row * 512 + (((chunk + row) & 31) << 4); }
__device__ __forceinline__ int lds_v_off(int d, int chunk) { return LDS_K + d * 32 + ((chunk ^ ((d >> 2) & 1)) << 4); }

// ---------------------------------------------------------------- kernel 1: token lists -> group bitmaps
// One workgroup per row. pos[row] = the row's context position (-1 for rows outside every request). The row's
// words [0, p/4/32] are zeroed first; words past that are never read for this row (kernel 2 bounds by pos).
__global__ void __launch_bounds__(256) r9k_qsa_bitmap_kernel(
    const int* __restrict__ idx, long idx_stride, int width, const int* __restrict__ t2r,
    const int* __restrict__ qsl, const int* __restrict__ seq_lens, int nreq, int rows,
    unsigned* __restrict__ bm, int bw, int* __restrict__ pos) {
  const int row = blockIdx.x;
  const int req = t2r[row];
  int p = -1;
  if (req >= 0 && req < nreq) {
    const int q0 = qsl[req], q1 = qsl[req + 1];
    if (row >= q0 && row < q1) p = seq_lens[req] - (q1 - q0) + (row - q0);
  }
  if (p >= 0 && (p / GS) >= bw * 32) p = -1;                 // outside the bitmap: treat as empty (never happens
  if (threadIdx.x == 0) pos[row] = p;                        // with bw sized from the block table)
  if (p < 0) return;
  unsigned* b = bm + (long)row * bw;
  const int nw = (p / GS) / 32 + 1;
  for (int w = threadIdx.x; w < nw; w += blockDim.x) b[w] = 0u;
  __syncthreads();
  const int* ir = idx + (long)row * idx_stride;
  for (int j = threadIdx.x; j < width; j += blockDim.x) {
    const int t = ir[j];
    if (t >= 0 && t <= p) {
      const int g = t / GS;
      atomicOr(&b[g >> 5], 1u << (g & 31));
    }
  }
}

// ---------------------------------------------------------------- kernel 2: tiled union-walk attention
template <int GQA>
struct Lds {
  static constexpr int NW = 2 * GQA;
  static constexpr int LISTN = GQA <= 8 ? 1024 : 256;       // groups per window (LDS budget at GQA 12)
  static constexpr int WINW = LISTN / 32;                    // bitmap words per window
  static constexpr int BYTES = 2 * LDS_BUF + NW * 256 * 4 + LISTN * 12 + (WINW + 1) * 4 + TILE * 8;
};

template <int GQA>
__global__ void __launch_bounds__(GQA * 64) r9k_qsa_attn_kernel(
    const uint16_t* __restrict__ q, long q_srow, long q_shead, const uint16_t* __restrict__ kc,
    const uint16_t* __restrict__ vc, long s_blk, long s_tok, long s_head, int page_size,
    const int* __restrict__ bt, long bt_stride, const int* __restrict__ t2r, const int* __restrict__ pos,
    const unsigned* __restrict__ bm, int bw, int rows, int hq, uint16_t* __restrict__ out, long o_srow,
    long o_shead, float* __restrict__ po, float* __restrict__ plse, int nsplit, float scale_log2) {
  typedef Lds<GQA> L;
  constexpr int NW = L::NW, NT = NW * 32;
  constexpr int NK = (512 + NT - 1) / NT;          // 16 B K chunks per thread per stage
  constexpr int NV = (16 + NW - 1) / NW;           // V d-tiles per wave per stage
  __shared__ __attribute__((aligned(16))) unsigned char lds[L::BYTES];
  unsigned char* xs = lds + 2 * LDS_BUF;                                   // S^T partial exchange
  long* l_off = reinterpret_cast<long*>(xs + NW * 256 * 4);               // per listed group: K/V element offset
  int* l_gid = reinterpret_cast<int*>(l_off + L::LISTN);                  // per listed group: group id
  int* l_cnt = l_gid + L::LISTN;                                          // per window word: prefix counts
  int* s_req = l_cnt + L::WINW + 1;                                       // per tile row: request (-2 = none)
  int* s_pos = s_req + TILE;                                              // per tile row: position

  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, t = threadIdx.x;
  const int r = lane & 15, kh = lane >> 4;
  const int kvh = blockIdx.y, split = blockIdx.z;
  const int head = kvh * GQA + (wave >> 1);
  const int d0 = (wave & 1) * HH;
  const int row0 = blockIdx.x * TILE;
  const long head_off = (long)kvh * s_head;

  if (t < TILE) {
    const int rr = row0 + t;
    const int p = rr < rows ? pos[rr] : -1;
    s_pos[t] = p;
    s_req[t] = p >= 0 ? t2r[rr] : -2;
  }
  __syncthreads();

  const int my_row = row0 + r;
  int k_frag_off[DT];
#pragma unroll
  for (int kk = 0; kk < DT; kk++) k_frag_off[kk] = lds_k_off(r, (d0 >> 3) + 2 * kk + kh);
  const int vfo = lds_v_off(d0 + r, kh);
  unsigned char* xw = xs + (wave * 256 + lane) * 4;
  const unsigned char* xr = xs + ((wave ^ 1) * 256 + lane) * 4;

  unsigned done = 0;                                          // tile rows already processed (workgroup-uniform)
  for (int it = 0; it < TILE; it++) {
    // next segment = the rows of the first unprocessed row's request
    int first = -1;
    for (int i = 0; i < TILE; i++)
      if (!((done >> i) & 1u)) { first = i; break; }
    if (first < 0) break;
    const int req = s_req[first];
    unsigned seg = 0;
    int gmax = 0;
    for (int i = 0; i < TILE; i++)
      if (s_req[i] == req) { seg |= 1u << i; gmax = max(gmax, s_pos[i] / GS + 1); }
    done |= seg;
    if (req < 0) continue;                                    // padding rows: nothing to do (caller zeroed out)
    const bool active = (seg >> r) & 1u;
    const int p = active ? s_pos[r] : -1;
    const unsigned* brow = bm + (long)(active ? row0 + r : row0 + first) * bw;   // never out of range
    const int* btr = bt + (long)req * bt_stride;

    // Q fragments for this lane's row (inactive lanes load the segment's first row; never stored)
    const int qrow = row0 + (active ? r : first);
    const uint16_t* qp = q + (long)qrow * q_srow + (long)head * q_shead + d0 + 8 * kh;
    v8bf16 qf[DT];
#pragma unroll
    for (int kk = 0; kk < DT; kk++) qf[kk] = ld_frag(qp + 16 * kk);
    v8f o[DT];
#pragma unroll
    for (int dt = 0; dt < DT; dt++) o[dt] = (v8f)(0.f);
    float m = -INFINITY, lsum = 0.f;

    // this split's group range
    const int g0 = (int)((long)split * gmax / nsplit), g1 = (int)((long)(split + 1) * gmax / nsplit);
    for (int wb = g0 & ~31; wb < g1; wb += L::LISTN) {
      // ---- union of the segment's bitmaps over [wb, wb + LISTN) & [g0, g1), compacted into the group list
      if (t < L::WINW) {
        const int w = (wb >> 5) + t;
        unsigned u = 0;
        if (w * 32 < g1) {
          for (int i = 0; i < TILE; i++)
            if (((seg >> i) & 1u) && w <= (s_pos[i] / GS) / 32) u |= bm[(long)(row0 + i) * bw + w];
          const int lo = g0 - w * 32, hi = g1 - w * 32;           // clip to [g0, g1)
          if (lo > 0) u &= lo >= 32 ? 0u : ~((1u << lo) - 1u);
          if (hi < 32) u &= hi <= 0 ? 0u : ((1u << hi) - 1u);
        }
        l_cnt[t + 1] = __builtin_popcount(u);
        l_gid[L::LISTN - 1 - t] = (int)u;                         // park the word at the list's tail for now
      }
      __syncthreads();
      if (t == 0) {
        l_cnt[0] = 0;
        for (int i = 0; i < L::WINW; i++) l_cnt[i + 1] += l_cnt[i];
      }
      __syncthreads();
      unsigned myword = 0;
      if (t < L::WINW) myword = (unsigned)l_gid[L::LISTN - 1 - t];
      __syncthreads();                                            // the parked words are read before the list fills
      if (t < L::WINW) {
        int at = l_cnt[t];
        const int wbase = wb + 32 * t;
        while (myword) {
          const int b = __builtin_ctz(myword);
          myword &= myword - 1;
          const int g = wbase + b, tok = g * GS;
          const int pg = tok / page_size;
          l_gid[at] = g;
          l_off[at] = (long)btr[pg] * s_blk + (long)(tok - pg * page_size) * s_tok;
          at++;
        }
      }
      __syncthreads();
      const int n = l_cnt[L::WINW];
      __syncthreads();                                            // everyone has read n before l_cnt is rewritten
      if (n == 0) continue;
      const int nst = (n + 3) / 4;
      if (t < 4 * nst - n) {                                      // pad the last stage with copies of group 0
        l_gid[n + t] = -1;                                        // (finite data for V; masked for every row)
        l_off[n + t] = l_off[0];
      }
      __syncthreads();

      // ---- staged K (coalesced) / V^T (transposed) loads for stage s, from the gathered groups
      v8bf16 sk[NK], sv[NV];
      auto load = [&](int s) {
#pragma unroll
        for (int i = 0; i < NK; i++) {
          const int c = t + i * NT;
          if (c < 512) {
            const int krow = c >> 5;
            sk[i] = ld_frag(kc + l_off[4 * s + (krow >> 2)] + (long)(krow & 3) * s_tok + head_off + (c & 31) * 8);
          }
        }
        const int kv_row = 8 * (lane >> 4) + (lane & 7);
        const uint16_t* vb = vc + l_off[4 * s + (kv_row >> 2)] + (long)(kv_row & 3) * s_tok + head_off +
                             8 * ((lane >> 3) & 1);
#pragma unroll
        for (int i = 0; i < NV; i++) {
          const int dt = wave + i * NW;
          if (dt < 16) sv[i] = ld_frag_tr(vb + 16 * dt);
        }
      };
      auto store = [&](unsigned char* buf) {
#pragma unroll
        for (int i = 0; i < NK; i++) {
          const int c = t + i * NT;
          if (c < 512) *reinterpret_cast<v8bf16*>(buf + lds_k_off(c >> 5, c & 31)) = sk[i];
        }
#pragma unroll
        for (int i = 0; i < NV; i++) {
          const int dt = wave + i * NW;
          if (dt < 16) *reinterpret_cast<v8bf16*>(buf + lds_v_off(16 * dt + (lane & 15), lane >> 4)) = sv[i];
        }
      };
      load(0);
      store(lds);
      lds_barrier();
      for (int s = 0; s < nst; s++) {
        const unsigned char* buf = lds + (s & 1) * LDS_BUF;
        load(min(s + 1, nst - 1));
        v8f sc = (v8f)(0.f);
#pragma unroll
        for (int kk = 0; kk < DT; kk++) {
          const v8bf16 kf = *reinterpret_cast<const v8bf16*>(buf + k_frag_off[kk]);
          sc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(kf, qf[kk], sc);
        }
#pragma unroll
        for (int e = 0; e < 8; e++) *reinterpret_cast<float*>(xw + e * 32 * 4) = sc[e];
        lds_barrier();
#pragma unroll
        for (int e = 0; e < 8; e++) sc[e] += *reinterpret_cast<const float*>(xr + e * 32 * 4);
        // mask: keys 8*kh + e of this stage = groups 2*kh (e < 4) and 2*kh + 1 (e >= 4)
        bool ok[2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int g = l_gid[4 * s + 2 * kh + h];
#ifdef R9K_QSA_NOMASK
          ok[h] = active && g >= 0 && g <= p / GS;                    // debug: every listed group, causal only
#else
          ok[h] = active && g >= 0 && g <= p / GS && ((brow[g >> 5] >> (g & 31)) & 1u);
#endif
        }
        const int gA = l_gid[4 * s + 2 * kh];
        float mloc = -INFINITY;
        bool valid[8];
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int g = e < 4 ? gA : l_gid[4 * s + 2 * kh + 1];
          valid[e] = ok[e >> 2] && (g * GS + (e & 3) <= p);
          sc[e] = valid[e] ? sc[e] : -INFINITY;
          mloc = fmaxf(mloc, sc[e]);
        }
        mloc = fmaxf(mloc, lane_swap16(mloc));
        if (__builtin_amdgcn_ballot_w32(mloc * scale_log2 > m + RESCALE_THRESH)) {
          const float m_new = mloc * scale_log2 > m + RESCALE_THRESH ? mloc * scale_log2 : m;
          // A row that has not met a valid key yet (m = -inf) rides along when another row rescales:
          // exp2(-inf - -inf) is NaN, so keep its (all-zero) state as is. Cannot happen in causal attention
          // (every row sees key 0 first) but a QSA row's first selected group can come late in the union.
          const float alpha = m_new == -INFINITY ? 1.f : __builtin_amdgcn_exp2f(m - m_new);
          m = m_new;
          lsum *= alpha;
#pragma unroll
          for (int dt = 0; dt < DT; dt++)
#pragma unroll
            for (int e = 0; e < 8; e++) o[dt][e] *= alpha;
        }
        v8f pf;
#pragma unroll
        for (int e = 0; e < 8; e++) pf[e] = valid[e] ? __builtin_amdgcn_exp2f(fmaf(sc[e], scale_log2, -m)) : 0.f;
        const v8bf16 pb = f32x8_to_bf16_trunc(pf);
        lsum = bf16x8_sum(pb, lsum);
#pragma unroll
        for (int dt = 0; dt < DT; dt++) {
          const v8bf16 vf = *reinterpret_cast<const v8bf16*>(buf + vfo + 512 * dt);
          o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(vf, pb, o[dt]);
        }
        store(lds + ((s + 1) & 1) * LDS_BUF);
        lds_barrier();
      }
      __syncthreads();                                            // the list is rebuilt by the next window
    }

    // ---- output: split 1 writes bf16; otherwise normalized fp32 partial + LSE (log2 domain) for the merge
    const float ltot = lsum + lane_swap16(lsum);
    if (active) {
      const int orow = row0 + r;
      const float inv = ltot > 0.f ? 1.f / ltot : 0.f;
      if (nsplit == 1) {
        uint16_t* op = out + (long)orow * o_srow + (long)head * o_shead + d0 + 8 * kh;
#pragma unroll
        for (int dt = 0; dt < DT; dt++) {
          v8f v = o[dt];
#pragma unroll
          for (int e = 0; e < 8; e++) v[e] *= inv;
          *reinterpret_cast<v8bf16*>(op + 16 * dt) = f32x8_to_bf16_rne(v);
        }
      } else {
        const long pr = ((long)split * rows + orow) * hq + head;
        float* opp = po + pr * HD + d0 + 8 * kh;
#pragma unroll
        for (int dt = 0; dt < DT; dt++)
#pragma unroll
          for (int e = 0; e < 8; e++) opp[16 * dt + e] = o[dt][e] * inv;
        if (kh == 0 && (wave & 1) == 0) plse[pr] = ltot > 0.f ? m + log2f(ltot) : -INFINITY;
      }
    }
  }
}

// ---------------------------------------------------------------- kernel 3: merge splits by LSE
__global__ void __launch_bounds__(256) r9k_qsa_merge_kernel(
    const float* __restrict__ po, const float* __restrict__ plse, uint16_t* __restrict__ out, long o_srow,
    long o_shead, const int* __restrict__ pos, int rows, int hq, int nsplit) {
  const int row = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
  if (pos[row] < 0) return;
  float mx = -INFINITY;
  for (int s = 0; s < nsplit; s++) mx = fmaxf(mx, plse[((long)s * rows + row) * hq + head]);
  float acc = 0.f, den = 0.f;
  if (mx > -INFINITY) {
    for (int s = 0; s < nsplit; s++) {
      const long pr = ((long)s * rows + row) * hq + head;
      const float l = plse[pr];
      if (l == -INFINITY) continue;
      const float w = exp2f(l - mx);
      den += w;
      acc += w * po[pr * HD + d];
    }
  }
  const float v = den > 0.f ? acc / den : 0.f;
  const unsigned x = __float_as_uint(v);
  out[(long)row * o_srow + (long)head * o_shead + d] = (uint16_t)((x + 0x7FFFu + ((x >> 16) & 1u)) >> 16);
}

template <int GQA>
int launch_attn(long q, long q_srow, long q_shead, long k, long v, long s_blk, long s_tok, long s_head, int ps,
                long bt, long bt_stride, long t2r, long pos, long bm, int bw, int rows, int hq, int hkv, long out,
                long o_srow, long o_shead, long po, long plse, int nsplit, hipStream_t st) {
  dim3 grid((rows + TILE - 1) / TILE, hkv, nsplit), block(GQA * 64);
  r9k_qsa_attn_kernel<GQA><<<grid, block, 0, st>>>(
      (const uint16_t*)q, q_srow, q_shead, (const uint16_t*)k, (const uint16_t*)v, s_blk, s_tok, s_head, ps,
      (const int*)bt, bt_stride, (const int*)t2r, (const int*)pos, (const unsigned*)bm, bw, rows, hq,
      (uint16_t*)out, o_srow, o_shead, (float*)po, (float*)plse, nsplit, (1.f / 16.f) * LOG2E);
  return (int)hipGetLastError();
}

}  // namespace

extern "C" int r9k_qsa_bitmap(long idx, long idx_stride, int width, long t2r, long qsl, long seq_lens, int nreq,
                              int rows, long bm, int bw, long pos, long stream) {
  if (rows <= 0) return 0;
  r9k_qsa_bitmap_kernel<<<rows, 256, 0, (hipStream_t)stream>>>(
      (const int*)idx, idx_stride, width, (const int*)t2r, (const int*)qsl, (const int*)seq_lens, nreq, rows,
      (unsigned*)bm, bw, (int*)pos);
  return (int)hipGetLastError();
}

// Element strides for q / out (row, head) and for the K/V caches (block, token, head); the head dim is contiguous.
// nsplit == 1 writes `out` directly; nsplit > 1 needs po [nsplit, rows, hq, 256] fp32 and plse [nsplit, rows, hq]
// fp32 and then r9k_qsa_merge. Returns 0, <0 for unsupported geometry, or the hip error code.
extern "C" int r9k_qsa_attn(long q, long q_srow, long q_shead, long k, long v, long s_blk, long s_tok, long s_head,
                            int page_size, int head_size, long bt, long bt_stride, long t2r, long pos, long bm, int bw,
                            int rows, int hq, int hkv, long out, long o_srow, long o_shead, long po, long plse,
                            int nsplit, long stream) {
  if (head_size != HD || page_size <= 0 || page_size % GS || hkv <= 0 || hq % hkv || nsplit <= 0) return -1;
  if (rows <= 0) return 0;
  hipStream_t st = (hipStream_t)stream;
#define QSA_CASE(G) \
  case G: return launch_attn<G>(q, q_srow, q_shead, k, v, s_blk, s_tok, s_head, page_size, bt, bt_stride, t2r, pos, bm, \
                                bw, rows, hq, hkv, out, o_srow, o_shead, po, plse, nsplit, st);
  switch (hq / hkv) {
    QSA_CASE(1) QSA_CASE(2) QSA_CASE(3) QSA_CASE(4) QSA_CASE(6) QSA_CASE(8) QSA_CASE(12)
    default: return -2;
  }
#undef QSA_CASE
}

extern "C" int r9k_qsa_merge(long po, long plse, long out, long o_srow, long o_shead, long pos, int rows, int hq,
                             int nsplit, long stream) {
  if (rows <= 0) return 0;
  r9k_qsa_merge_kernel<<<dim3(rows, hq), HD, 0, (hipStream_t)stream>>>(
      (const float*)po, (const float*)plse, (uint16_t*)out, o_srow, o_shead, (const int*)pos, rows, hq, nsplit);
  return (int)hipGetLastError();
}
