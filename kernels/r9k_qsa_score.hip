// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas

// QSA indexer scoring for gfx1201: replaces vLLM's Triton _qsa_mqa_paged_kernel for prefill widths.
//
// Independent implementation from the stock kernel's documented semantics (vLLM, Apache-2.0) and our own
// r9k_attn.hip WMMA pipeline. No third-party indexer code consulted.
//
// What it computes, per query row r (request req, context position p) and compressed key column c:
//   visible(r) = min((p + 1) / CR, seq_len(req) / CR)             (CR = indexer_compress_ratio)
//   score(r, c) = sum_h relu(q[r, h, :] . k[c, :]) / divisor       for c < visible(r)
//   logits[r, c] = score, and -inf for visible(r) <= c < visible_max(tile)
//   visible_blocks[r] = visible(r)
// Columns at or beyond the tile's visible_max are left untouched: the top-k that follows
// (top_k_per_row_decode) reads row r only up to visible(r). The stock kernel instead scores every column of the
// whole context window (8192 at 32k) and writes the full fp32 logits (128 MB per layer at 4096 rows); for a
// prefill chunk only p/4 columns are visible, so most of that work is discarded.
//
// Mapping: one workgroup = 16 consecutive query rows (a tile) x one column part (blockIdx.y of nsplit: the
// scores are independent per column, so few-row batches -- decode, MTP verify -- split the column range across
// workgroups instead of leaving most CUs idle). Rows of a tile can belong to several requests in a
// mixed batch; the workgroup loops over the distinct requests among its rows (uniform loop, as in r9k_qsa.hip).
// Per 64-column step: stage K[64 x 128] (bf16, gathered through the page table) into LDS with the r9k_attn K
// swizzle; 4 waves, one per 16-column sub-tile; per head, S^T[col][row] = K[16 x 128] . Q_h^T via 8 WMMA
// 16x16x16 bf16 (A = K fragment from LDS, B = Q_h fragment from LDS), relu, accumulate over the 4 heads, scale,
// store. WMMA D layout: lane l holds column (=query row) l % 16 and rows (=keys) 8 * (l / 16) .. +8.
// Geometry: head_dim 128, 1..8 indexer heads (Flash-Next: 4), compress ratio any, page size any.
// No device printf/assert (the build enforces hostcall-free objects).
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

typedef __bf16 v8bf16 __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));

constexpr int D = 128;                  // indexer head dim
constexpr int TILE = 16;                // query rows per workgroup
constexpr int CT = 64;                  // columns per step
constexpr int NW = CT / 16;             // 4 waves, one per 16-column sub-tile
constexpr int NT = NW * 32;
constexpr int MAXH = 8;
// LDS: K stage [64 cols][128 dims] bf16 = 16 KB, chunk (16 B) index rotated by the row so the 8 lanes of a
// ds_read_b128 fragment read (rows 0..7 at one chunk) hit distinct banks; Q [16 rows][MAXH*128] bf16 = 32 KB max.
constexpr int LDS_K = CT * D * 2;
__device__ __forceinline__ int k_off(int col, int chunk) { return col * 256 + (((chunk + col) & 15) << 4); }

__device__ __forceinline__ v8bf16 ld8(const uint16_t* p) { return *reinterpret_cast<const v8bf16*>(p); }

template <int H>
__global__ void __launch_bounds__(NT) r9k_qsa_score_kernel(
    const uint16_t* __restrict__ q, long q_srow, long q_shead,           // q [rows, H, 128], element strides
    const uint16_t* __restrict__ kc, long k_spage, long k_stok, int page_size,   // k cache [pages, page, 1, 128]
    const int* __restrict__ bt, long bt_stride, int bt_width,             // page table [nreq, width]
    const int* __restrict__ t2r, const int* __restrict__ pos, const int* __restrict__ seq_lens, int nreq,
    int rows, int cr, int num_columns, int num_pages, int nsplit, float inv_div,
    float* __restrict__ logits, long l_srow, int* __restrict__ visible_blocks) {
  __shared__ __attribute__((aligned(16))) unsigned char lds[LDS_K + TILE * H * D * 2];
  uint16_t* qs = reinterpret_cast<uint16_t*>(lds + LDS_K);              // [16][H*128]
  __shared__ int s_req[TILE], s_vis[TILE];

  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, t = threadIdx.x;
  const int r = lane & 15, kh = lane >> 4;
  const int row0 = blockIdx.x * TILE;

  // per-row request and visible count; -1 = padding row
  if (t < TILE) {
    const int rr = row0 + t;
    int req = -1, vis = 0;
    if (rr < rows) {
      req = t2r[rr];
      if (req >= 0 && req < nreq) {
        const int p = pos[rr];
        vis = min((p + 1) / cr, seq_lens[req] / cr);
        vis = min(vis, num_columns);
        if (vis < 0) vis = 0;
      } else {
        req = -1;
      }
      if (blockIdx.y == 0) visible_blocks[rr] = vis;
    }
    s_req[t] = req;
    s_vis[t] = vis;
  }
  // stage Q for the tile: 16 rows x H heads x 128 dims (padding rows use row 0's request... just zero them)
  for (int i = t; i < TILE * H * (D / 8); i += NT) {
    const int rr = i / (H * (D / 8)), rem = i % (H * (D / 8)), h = rem / (D / 8), c = rem % (D / 8);
    v8bf16 v = (v8bf16)((__bf16)0.f);
    if (row0 + rr < rows) v = ld8(q + (long)(row0 + rr) * q_srow + (long)h * q_shead + c * 8);
    *reinterpret_cast<v8bf16*>(qs + (rr * H + h) * D + c * 8) = v;
  }
  __syncthreads();

  unsigned done = 0;
  for (int it = 0; it < TILE; it++) {
    int first = -1;
    for (int i = 0; i < TILE; i++)
      if (!((done >> i) & 1u)) { first = i; break; }
    if (first < 0) break;
    const int req = s_req[first];
    unsigned seg = 0;
    int vmax = 0;
    for (int i = 0; i < TILE; i++)
      if (s_req[i] == req) { seg |= 1u << i; vmax = max(vmax, s_vis[i]); }
    done |= seg;
    if (req < 0 || vmax == 0) continue;
    const bool active = (seg >> r) & 1u;
    const int vis = active ? s_vis[r] : 0;
    const int* btr = bt + (long)req * bt_stride;

    // this lane's Q fragments live in LDS: row r, head h, dims 16*kk + 8*kh .. +8 (B operand: col = query row)
    const int steps = (vmax + CT - 1) / CT, per = (steps + nsplit - 1) / nsplit;
    const int cb = blockIdx.y * per * CT, ce = min(vmax, cb + per * CT);
    for (int c0 = cb; c0 < ce; c0 += CT) {
      // stage K[c0 .. c0+64) x 128: 64 cols x 16 chunks of 16 B = 1024 chunks over 128 threads = 8 each
      __syncthreads();                                                  // previous step's reads are done
      for (int i = t; i < CT * 16; i += NT) {
        const int col = i >> 4, ch = i & 15;
        const int c = c0 + col;
        v8bf16 v = (v8bf16)((__bf16)0.f);
        if (c < vmax) {
          const int lp = c / page_size;
          if (lp < bt_width) {
            const int pp = btr[lp];
            if (pp >= 0 && pp < num_pages) v = ld8(kc + (long)pp * k_spage + (long)(c % page_size) * k_stok + ch * 8);
          }
        }
        *reinterpret_cast<v8bf16*>(lds + k_off(col, ch)) = v;
      }
      __syncthreads();
      // this wave's 16 columns: cols c0 + 16*wave + (lane & 15) as the A operand rows
      const int colA = 16 * wave + r;
      v8f acc = (v8f)(0.f);
#pragma unroll
      for (int h = 0; h < H; h++) {
        v8f s = (v8f)(0.f);
#pragma unroll
        for (int kk = 0; kk < D / 16; kk++) {
          const v8bf16 kf = *reinterpret_cast<const v8bf16*>(lds + k_off(colA, 2 * kk + kh));
          const v8bf16 qf = *reinterpret_cast<const v8bf16*>(qs + (r * H + h) * D + 16 * kk + 8 * kh);
          s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(kf, qf, s);
        }
#pragma unroll
        for (int e = 0; e < 8; e++) acc[e] += fmaxf(s[e], 0.f);
      }
      // D layout: lane holds S^T[key = 16*wave + 8*kh + e][query row r]: column c = c0 + 16*wave + 8*kh + e
      if (active) {
        float* lrow = logits + (long)(row0 + r) * l_srow;
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int c = c0 + 16 * wave + 8 * kh + e;
          if (c < vmax) lrow[c] = c < vis ? acc[e] * inv_div : -INFINITY;
        }
      }
    }
  }
}

template <int H>
int launch(long q, long q_srow, long q_shead, long kc, long k_spage, long k_stok, int page_size, long bt,
           long bt_stride, int bt_width, long t2r, long pos, long seq_lens, int nreq, int rows, int cr,
           int num_columns, int num_pages, int nsplit, float inv_div, long logits, long l_srow, long visible,
           hipStream_t st) {
  dim3 grid((rows + TILE - 1) / TILE, nsplit), block(NT);
  r9k_qsa_score_kernel<H><<<grid, block, 0, st>>>(
      (const uint16_t*)q, q_srow, q_shead, (const uint16_t*)kc, k_spage, k_stok, page_size, (const int*)bt,
      bt_stride, bt_width, (const int*)t2r, (const int*)pos, (const int*)seq_lens, nreq, rows, cr, num_columns,
      num_pages, nsplit, inv_div, (float*)logits, l_srow, (int*)visible);
  return (int)hipGetLastError();
}

}  // namespace

// q [rows, heads, 128] bf16 (element strides row/head, dim contiguous); k cache [pages, page_size, 1, 128] bf16
// (element strides page/token); logits fp32 [rows, num_columns] with row stride l_srow (elements); visible [rows]
// int32. nsplit = column parts per row tile (grid.y). divisor = the stock score_divisor (sqrt(128) by default). Returns 0, <0 for unsupported geometry, or hip error.
extern "C" int r9k_qsa_score(long q, long q_srow, long q_shead, int heads, int head_dim, long kc, long k_spage,
                             long k_stok, int page_size, long bt, long bt_stride, int bt_width, long t2r, long pos,
                             long seq_lens, int nreq, int rows, int cr, int num_columns, int num_pages, int nsplit,
                             float divisor, long logits, long l_srow, long visible, long stream) {
  if (head_dim != D || heads < 1 || heads > MAXH || cr <= 0 || page_size <= 0 || divisor <= 0.f) return -1;
  if (rows <= 0) return 0;
  nsplit = max(1, min(nsplit, 64));
  const float inv = 1.0f / divisor;
  hipStream_t st = (hipStream_t)stream;
#define QS_CASE(HH) \
  case HH: return launch<HH>(q, q_srow, q_shead, kc, k_spage, k_stok, page_size, bt, bt_stride, bt_width, t2r, pos, \
                             seq_lens, nreq, rows, cr, num_columns, num_pages, nsplit, inv, logits, l_srow, visible, st);
  switch (heads) {
    QS_CASE(1) QS_CASE(2) QS_CASE(4) QS_CASE(8)
    default: return -2;
  }
#undef QS_CASE
}
