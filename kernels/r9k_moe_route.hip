// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas

// r9k_moe_route.hip -- MoE routing for gfx1201 at decode widths, in ONE launch:
//   softmax over the router logits, top-k (ties to the lower expert index), optional renormalisation of the
//   selected weights, and vLLM's moe_align_block_size tables (sorted_ids / expert_ids / num_tokens_post_padded)
//   for the grouped GEMMs that follow.
//
// Stock vLLM runs this as four launches per MoE layer (topk_softmax, moe_align_block_size, count_and_sort, and the
// fills behind them): 192 HIP-graph nodes per Flash-Next decode step, each a few microseconds of dispatch. Here it
// is one launch at any row count: a grid of blocks does the rows (one wave per row: softmax, top-k, the counts by
// global atomics), and the LAST block to arrive -- the classic threadfence-reduction pattern -- builds the tables
// from the counts and the ids the other blocks wrote, then zeroes the counters for the next call. The first
// single-workgroup version was level with stock's four launches at 8 rows and 5x slower at 64; the serial part
// (one prefix sum over 512 experts and the scatter) is now the only thing one block does alone. Above
// ROUTE_MAX_ROWS rows the caller keeps the stock path (prefill uses other block sizes anyway).
// Scratch (int32, zeroed by the caller ONCE at allocation; the kernel leaves it zero): counts[E], cursor[E], arrive.
//
// Numerics: fp32 softmax (max, expf(x - max), sum over the row, p = e / sum), top-k on the probabilities, and if
// renormalize, w_k = p_k / sum_k p_k. Same operations as vLLM's topk_softmax (TensorRT-LLM lineage); the only
// differences are the summation order of the row sums (wave tree here) and the rounding of expf, so selected ids
// agree exactly except on near-ties of the probabilities, and weights differ at the fp32 ulp level.
//
// Table contract (what r9k_moe_mxfp4a8_kernel reads; see align_block_size_ref in r9700_vllm/kernels/moe.py):
//   sorted_ids[i] for i < ntpp: flat routed row (token * topk + j), or `numel` for padding; expert_ids[b]: the
//   expert of block b (blk rows); ntpp[0]: rows incl. padding; experts appear in ascending order, each padded to a
//   multiple of blk. Row order WITHIN an expert is unspecified (atomics), which no consumer depends on: every row's
//   GEMM result is its own, and moe_sum reduces over (token, j) in its own order.
// No device printf/assert (P2P-safe behind switches that drop PCIe atomics).
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int WAVE = 32;
constexpr int WAVES = 8;
constexpr int NT = WAVE * WAVES;
constexpr int MAX_E = 1024;        // experts
constexpr int MAX_K = 16;          // top-k
constexpr int PER_LANE = MAX_E / WAVE;

__device__ __forceinline__ float bf2f(uint16_t u) { return __uint_as_float((unsigned)u << 16); }

__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, WAVE));
  return v;
}
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, WAVE);
  return v;
}
// argmax over the wave with ties to the LOWER index: compare (value, -index).
__device__ __forceinline__ void wave_argmax(float& v, int& i) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor(v, o, WAVE);
    const int oi = __shfl_xor(i, o, WAVE);
    if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
  }
}

// Phase 1 (one wave per row): softmax + top-k + renorm; writes weights/ids and bumps the expert counts.
template <int EPL>   // logits per lane = E / 32
__device__ __forceinline__ void route_row(const uint16_t* __restrict__ row, int E, int topk, bool renorm,
                                          float* __restrict__ w_out, int* __restrict__ id_out,
                                          int* __restrict__ counts, int lane) {
  float p[EPL];
  float mx = -INFINITY;
#pragma unroll
  for (int i = 0; i < EPL; ++i) {
    const int e = lane + WAVE * i;
    p[i] = e < E ? bf2f(row[e]) : -INFINITY;
    mx = fmaxf(mx, p[i]);
  }
  mx = wave_max(mx);
  // Robust to anything a dummy or warm-up step can feed us (NaN, +-inf, an all-equal row): a real entry that is
  // not a finite probability counts as 0, a padded lane (e >= E) is -1 = "taken", and the selection below always
  // lands on a real index because the compare starts from -2. Stock's kernel never indexes out of range on such
  // rows either; a NaN row here used to leave the index at its sentinel and write past the count table.
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < EPL; ++i) {
    if ((lane + WAVE * i) < E) {
      float v = expf(p[i] - mx);
      p[i] = (v == v && v <= 3.4e38f) ? v : 0.f;
      s += p[i];
    } else {
      p[i] = -1.f;
    }
  }
  s = wave_sum(s);
  const float inv = s > 0.f ? 1.f / s : 0.f;
#pragma unroll
  for (int i = 0; i < EPL; ++i) if (p[i] >= 0.f) p[i] *= inv;     // p = e / sum, as one multiply by the reciprocal
  float wsel = 0.f;   // lane k holds selection k
  int isel = 0;
  for (int k = 0; k < topk; ++k) {
    float bv = -2.f; int bi = 0x7fffffff;
#pragma unroll
    for (int i = 0; i < EPL; ++i) {
      const int e = lane + WAVE * i;
      if (p[i] > bv || (p[i] == bv && e < bi)) { bv = p[i]; bi = e; }
    }
    wave_argmax(bv, bi);
    if (lane == k) { wsel = bv; isel = bi; }
#pragma unroll
    for (int i = 0; i < EPL; ++i) if (lane + WAVE * i == bi) p[i] = -1.f;   // taken: below every real prob
  }
  if (renorm) {
    const float tot = wave_sum(lane < topk ? wsel : 0.f);
    wsel = tot > 0.f ? wsel / tot : 0.f;
  }
  if (isel < 0 || isel >= E) isel = 0;                              // cannot happen; keeps the table writes in range
  if (lane < topk) {
    w_out[lane] = wsel;
    id_out[lane] = isel;
    atomicAdd(&counts[isel], 1);
  }
}

template <int EPL>
__global__ void __launch_bounds__(NT) r9k_moe_route_kernel(
    const uint16_t* __restrict__ logits, int ldl, float* __restrict__ weights, int* __restrict__ ids,
    int* __restrict__ sorted_ids, int* __restrict__ expert_ids, int* __restrict__ ntpp, int* __restrict__ scratch,
    int M, int E, int topk, int renorm, int blk) {
  __shared__ int offs[MAX_E];       // first sorted position of each expert (padded layout)
  __shared__ int scan[WAVES];
  __shared__ int s_last;
  int* __restrict__ g_counts = scratch;           // [E] rows per expert
  int* __restrict__ g_cursor = scratch + E;       // [E] scatter cursor
  int* __restrict__ g_arrive = scratch + 2 * E;   // [1] blocks done with phase 1
  const int tid = threadIdx.x, wave = tid / WAVE, lane = tid % WAVE;

  // Phase 1: this block's rows, one per wave.
  const int r = blockIdx.x * WAVES + wave;
  if (r < M)
    route_row<EPL>(logits + (long)r * ldl, E, topk, renorm != 0, weights + (long)r * topk, ids + (long)r * topk,
                   g_counts, lane);
  // publish this block's ids and counts, then count the block in; the last one builds the tables
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(g_arrive, 1) == (int)gridDim.x - 1) ? 1 : 0;
  __syncthreads();
  if (!s_last) return;
  __threadfence();

  // Phase 2 (last block only): exclusive prefix sum of the block-padded counts over E experts, in NT-wide chunks.
  int carry = 0;
  for (int base = 0; base < E; base += NT) {
    const int e = base + tid;
    const int n = e < E ? ((__hip_atomic_load(g_counts + e, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + blk - 1)
                           / blk) * blk : 0;
    int x = n;
#pragma unroll
    for (int o = 1; o < WAVE; o <<= 1) {
      const int y = __shfl_up(x, o, WAVE);
      if (lane >= o) x += y;
    }
    if (lane == WAVE - 1) scan[wave] = x;
    __syncthreads();
    int wbase = 0;
#pragma unroll
    for (int wv = 0; wv < WAVES; ++wv) if (wv < wave) wbase += scan[wv];
    int total = 0;
#pragma unroll
    for (int wv = 0; wv < WAVES; ++wv) total += scan[wv];
    if (e < E) offs[e] = carry + wbase + x - n;
    carry += total;
    __syncthreads();
  }
  if (tid == 0) ntpp[0] = carry;
  const int numel = M * topk;
  for (int e = tid; e < E; e += NT) {
    const int c = __hip_atomic_load(g_counts + e, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    if (c == 0) continue;
    const int o = offs[e], n = ((c + blk - 1) / blk) * blk;
    for (int b = o / blk; b < (o + n) / blk; ++b) expert_ids[b] = e;
    for (int i = o + c; i < o + n; ++i) sorted_ids[i] = numel;
  }
  __syncthreads();

  // Phase 3: scatter every routed row (ids written by every block, visible after the fences above).
  for (int t = tid; t < numel; t += NT) {
    const int e = __hip_atomic_load(ids + t, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    const int pos = offs[e] + atomicAdd(g_cursor + e, 1);
    sorted_ids[pos] = t;
  }
  __syncthreads();
  // leave the scratch zero for the next call (stream order separates calls)
  for (int e = tid; e < E; e += NT) { g_counts[e] = 0; g_cursor[e] = 0; }
  if (tid == 0) g_arrive[0] = 0;
}

}  // namespace

// Returns 0 on success; nonzero for an unsupported shape (caller keeps the stock path). scratch: 2*E+1 int32,
// zero at the first call.
extern "C" int r9k_moe_route(long logits, int ldl, long weights, long ids, long sorted_ids, long expert_ids,
                             long ntpp, long scratch, int M, int E, int topk, int renorm, int blk, long stream) {
  if (M < 1 || E < 1 || E > MAX_E || topk < 1 || topk > MAX_K || topk > E || blk < 1 || (blk & (blk - 1)))
    return 1;
  const hipStream_t st = (hipStream_t)stream;
  const dim3 grid((M + WAVES - 1) / WAVES), block(NT);
#define LAUNCH(EPL) \
  hipLaunchKernelGGL(r9k_moe_route_kernel<EPL>, grid, block, 0, st, (const uint16_t*)logits, ldl, \
                     (float*)weights, (int*)ids, (int*)sorted_ids, (int*)expert_ids, (int*)ntpp, (int*)scratch, \
                     M, E, topk, renorm, blk)
  if (E <= 128) LAUNCH(4);
  else if (E <= 256) LAUNCH(8);
  else if (E <= 512) LAUNCH(16);
  else LAUNCH(32);
#undef LAUNCH
  return (int)hipGetLastError();
}

extern "C" int r9k_moe_route_max_experts(void) { return MAX_E; }
extern "C" int r9k_moe_route_max_topk(void) { return MAX_K; }
