// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas

// r9k_ple.hip -- PLE n-gram table row gather + int6 dequant (gfx1201).
//
// Table rows are fused int6 rows (tcclaviger Qwen3.8-Flash-Next GPTQ checkpoint format): head_dim codes in
// [-31, 31] stored +32, four per 24-bit little-endian word (code i = (w >> 6i) & 63), then one fp16 scale per
// 32 elements. head_dim 160 -> 120 packed bytes + 10 scale bytes = 130 bytes per row. The table may live in
// pinned host memory behind a UVA pointer (38.7 GiB does not fit in VRAM); each thread reads its own 3-byte
// word + 2-byte scale, so a row is 40 independent small reads -- latency-bound, which is fine at 16 rows per
// token.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>

__global__ void r9k_ple_gather_int6_kernel(const unsigned char* __restrict__ table, const int64_t* __restrict__ ids,
                                           __hip_bfloat16* __restrict__ out, long n, int row_bytes, int head_dim,
                                           long rows) {
  const int words = head_dim / 4;                    // 24-bit words per row
  const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * words) return;
  const long i = t / words;
  const int w = (int)(t % words);
  const int64_t id = ids[i];
  __hip_bfloat16* o = out + i * head_dim + 4 * w;
  if (id < 0 || id >= rows) {                        // out-of-range id: zeros instead of a host page fault
#pragma unroll
    for (int k = 0; k < 4; k++) o[k] = (__hip_bfloat16)0.f;
    return;
  }
  const unsigned char* row = table + (size_t)id * row_bytes;
  const unsigned char* p = row + 3 * w;
  const unsigned int word = (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16);
  const int g = (4 * w) / 32;                        // scale group of these four elements
  const unsigned char* sp = row + (head_dim * 6 / 8) + 2 * g;
  const __half hs = __ushort_as_half((unsigned short)(sp[0] | (sp[1] << 8)));
  const float s = __half2float(hs);
#pragma unroll
  for (int k = 0; k < 4; k++) o[k] = (__hip_bfloat16)((float)((int)((word >> (6 * k)) & 63u) - 32) * s);
}

extern "C" int r9k_ple_gather_int6(long table, long ids, long out, long n, int row_bytes, int head_dim, long rows,
                                   long stream) {
  if (n <= 0) return 0;
  if (head_dim % 32 || row_bytes != head_dim * 6 / 8 + head_dim / 32 * 2) return -1;
  const long total = n * (head_dim / 4);
  const int bs = 256;
  hipLaunchKernelGGL(r9k_ple_gather_int6_kernel, dim3((unsigned)((total + bs - 1) / bs)), dim3(bs), 0,
                     (hipStream_t)stream, (const unsigned char*)table, (const int64_t*)ids, (__hip_bfloat16*)out, n,
                     row_bytes, head_dim, rows);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}
