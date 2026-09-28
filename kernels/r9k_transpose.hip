// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas

// Batched 2-D transpose of 16-bit elements for gfx1201: out[b][c][r] = in[b][r][c], out contiguous.
//
// Replaces torch's generic permuted copy for the two [prefills, tokens, 4*hidden] <-> [prefills, 4*hidden, tokens]
// transposes around the PLE short conv, which ATen runs as a strided elementwise copy at ~14 GB/s (5.9 ms per
// 84 MB tensor in a 4096-token chunk). 64x64 tiles through LDS: coalesced 16-byte reads along the source rows,
// coalesced 16-byte writes along the destination rows. Any strides on the input (elements); padded LDS rows avoid
// bank conflicts on the transposed read. No device printf/assert (hostcall-free build).
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

constexpr int T = 64;                  // tile edge
constexpr int NT = 256;                // threads: 8 rows x 32 lanes... each thread moves 8 elements per phase
typedef uint16_t v8u16 __attribute__((ext_vector_type(8)));

__global__ void __launch_bounds__(NT) r9k_transpose16_kernel(const uint16_t* __restrict__ in, long s_batch, long s_row,
                                                             uint16_t* __restrict__ out, int rows, int cols) {
  __shared__ uint16_t tile[T][T + 8];          // +8 elements: the 16 B chunk pitch is odd -> conflict-free columns
  const int b = blockIdx.z, r0 = blockIdx.y * T, c0 = blockIdx.x * T;
  const uint16_t* src = in + (long)b * s_batch;
  uint16_t* dst = out + (long)b * (long)rows * cols;
  const int t = threadIdx.x;
  // load: 64 rows x 64 cols = 512 chunks of 8 elements; 2 per thread. chunk q -> row q / 8, col 8 * (q % 8)
  for (int q = t; q < T * (T / 8); q += NT) {
    const int r = q >> 3, c = (q & 7) << 3;
    const int gr = r0 + r, gc = c0 + c;
    if (gr < rows) {
      if (gc + 8 <= cols && (((uintptr_t)(src + (long)gr * s_row + gc)) & 15) == 0) {
        const v8u16 v = *reinterpret_cast<const v8u16*>(src + (long)gr * s_row + gc);
#pragma unroll
        for (int i = 0; i < 8; i++) tile[r][c + i] = v[i];
      } else {
#pragma unroll
        for (int i = 0; i < 8; i++) tile[r][c + i] = (gc + i < cols) ? src[(long)gr * s_row + gc + i] : 0;
      }
    }
  }
  __syncthreads();
  // store: destination row = source col; 64 dst rows x 64 dst cols (= source rows) as 8-element chunks
  for (int q = t; q < T * (T / 8); q += NT) {
    const int c = q >> 3, r = (q & 7) << 3;           // dst row index c (source col), dst col chunk at r (source row)
    const int gc = c0 + c, gr = r0 + r;
    if (gc < cols) {
      uint16_t* d = dst + (long)gc * rows + gr;
      if (gr + 8 <= rows && (((uintptr_t)d) & 15) == 0) {
        v8u16 v;
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = tile[r + i][c];
        *reinterpret_cast<v8u16*>(d) = v;
      } else {
#pragma unroll
        for (int i = 0; i < 8; i++) if (gr + i < rows) d[i] = tile[r + i][c];
      }
    }
  }
}

}  // namespace

// in: [batch, rows, cols] 16-bit elements with element strides s_batch / s_row (cols contiguous); out: contiguous
// [batch, cols, rows]. Returns 0, or a hip error.
extern "C" int r9k_transpose16(long in, long s_batch, long s_row, long out, int batch, int rows, int cols, long stream) {
  if (batch <= 0 || rows <= 0 || cols <= 0) return 0;
  dim3 grid((cols + T - 1) / T, (rows + T - 1) / T, batch), block(NT);
  r9k_transpose16_kernel<<<grid, block, 0, (hipStream_t)stream>>>((const uint16_t*)in, s_batch, s_row, (uint16_t*)out,
                                                                   rows, cols);
  return (int)hipGetLastError();
}
