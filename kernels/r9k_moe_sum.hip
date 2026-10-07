// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas

// r9k_moe_sum.hip -- the MoE block's output in one launch: the sum over a token's top-k expert rows (the router
// weights already folded into the down GEMM's epilogue) plus, optionally, the shared expert's output.
//
//   out[m, n] = bf16( sum_j down[m, j, n] + shared[m, n] )      j < topk, fp32 accumulation, ONE bf16 rounding
//
// Stock vLLM does this as moe_sum (bf16 round), a copy of the result into the layer's output buffer (the modular
// kernel does not alias its output on ROCm without aiter), and the runner's `shared_output + fused_output` add
// (second bf16 round): three launches per MoE layer, 144 HIP-graph nodes per Flash-Next decode step. Summation
// order: j ascending, then the shared term, as stock's two steps do; the only difference is the single rounding.
// No device printf/assert.
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int NT = 256;

__device__ __forceinline__ float bf2f(uint16_t u) { return __uint_as_float((unsigned)u << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {    // round to nearest even (torch's bf16 conversion)
  unsigned u = __float_as_uint(f);
  if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0xffffu) ? 0x40u : 0u));
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

typedef uint16_t v8u16 __attribute__((ext_vector_type(8)));

// One thread = 8 consecutive columns of one token; grid.x covers N/8 columns, grid.y the tokens.
__global__ void __launch_bounds__(NT) r9k_moe_sum_kernel(
    const uint16_t* __restrict__ down, const uint16_t* __restrict__ shared, uint16_t* __restrict__ out,
    int M, int topk, int N, long ld_shared, long ld_out) {
  const int m = blockIdx.y;
  const int c = (blockIdx.x * NT + threadIdx.x) * 8;
  if (m >= M || c >= N) return;
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.f;
  const uint16_t* row = down + (long)m * topk * N + c;
  for (int j = 0; j < topk; ++j) {
    const v8u16 v = *reinterpret_cast<const v8u16*>(row + (long)j * N);
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] += bf2f(v[i]);
  }
  if (shared != nullptr) {
    const v8u16 s = *reinterpret_cast<const v8u16*>(shared + (long)m * ld_shared + c);
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] += bf2f(s[i]);
  }
  v8u16 o;
#pragma unroll
  for (int i = 0; i < 8; ++i) o[i] = f2bf(acc[i]);
  *reinterpret_cast<v8u16*>(out + (long)m * ld_out + c) = o;
}

}  // namespace

// down [M, topk, N] bf16 contiguous; shared [M, N] bf16 with row stride ld_shared (0 = none); out [M, N] with
// row stride ld_out. N % 8 == 0. Returns 0 on success.
extern "C" int r9k_moe_sum(long down, long shared, long out, int M, int topk, int N, long ld_shared, long ld_out,
                           long stream) {
  if (M < 1 || topk < 1 || N < 8 || (N & 7)) return 1;
  const dim3 grid((N / 8 + NT - 1) / NT, M), block(NT);
  hipLaunchKernelGGL(r9k_moe_sum_kernel, grid, block, 0, (hipStream_t)stream, (const uint16_t*)down,
                     (const uint16_t*)shared, (uint16_t*)out, M, topk, N, ld_shared, ld_out);
  return (int)hipGetLastError();
}
