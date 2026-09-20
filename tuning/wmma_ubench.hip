// WMMA issue-rate microbenchmark for gfx1201 (not part of libr9k): fp8 / bf16 WMMA 16x16x16 throughput, and how
// much independent VALU FMAs between the WMMAs cost (the 4-bit kernels scale every 16x16 partial with 8 FMAs per
// lane per scale group). Measured on the R9700 (cold clocks, ~2.9 GHz): fp8 344 TFLOPS, bf16 174 (half rate),
// fp8 + 4 FMAs/WMMA 262, + 8 FMAs 208, + 16 FMAs 168 -- the FMAs do not hide behind the WMMA pipeline, which puts
// the NVFP4 prefill kernel's issue floor at ~200 TFLOPS (cold) / ~145 (2.2 GHz steady state) and MXFP4's ~25% above.
// hipcc -O3 --offload-arch=gfx1201 wmma_ubench.hip -o wmma_ubench && ./wmma_ubench
#include <hip/hip_runtime.h>
#include <cstdio>
typedef int v2i __attribute__((ext_vector_type(2)));
typedef short v8s __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));

template <int MODE, int NFMA>
__global__ void __launch_bounds__(256) k(float* out, int iters, int seed) {
  v8f acc[8];
  for (int i = 0; i < 8; i++) acc[i] = (v8f)(0.f);
  v2i a = {seed, seed + 1}, b = {seed + 2, seed + 3};
  v8s ah, bh;
  for (int i = 0; i < 8; i++) { ah[i] = (short)(seed + i); bh[i] = (short)(seed - i); }
  float f[16];
  for (int i = 0; i < 16; i++) f[i] = (float)(threadIdx.x + i);
  const float sc = 1.0001f;
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      if constexpr (MODE == 0) acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, acc[i]);
      if constexpr (MODE == 1) acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(ah, bh, acc[i]);
#pragma unroll
      for (int q = 0; q < NFMA; q++) f[q] = fmaf(f[q], sc, f[(q + 1) & 15]);
    }
    __builtin_amdgcn_sched_barrier(0);
  }
  float s = 0.f;
  for (int i = 0; i < 8; i++) for (int q = 0; q < 8; q++) s += acc[i][q];
  for (int q = 0; q < 16; q++) s += f[q];
  if (s == 12345.678f) out[threadIdx.x] = s;
}

template <int MODE, int NFMA>
void run(const char* name, int blocks) {
  float* out; (void)hipMalloc(&out, 4096);
  const int iters = 4000;
  hipLaunchKernelGGL((k<MODE, NFMA>), dim3(blocks), dim3(256), 0, 0, out, 10, 1);
  (void)hipDeviceSynchronize();
  hipEvent_t e0, e1; (void)hipEventCreate(&e0); (void)hipEventCreate(&e1);
  (void)hipEventRecord(e0);
  hipLaunchKernelGGL((k<MODE, NFMA>), dim3(blocks), dim3(256), 0, 0, out, iters, 1);
  (void)hipEventRecord(e1); (void)hipEventSynchronize(e1);
  float ms; (void)hipEventElapsedTime(&ms, e0, e1);
  const double wmma = (double)blocks * 8 * iters * 8;            // per wave: 8 per iter
  const double flop = wmma * 2.0 * 16 * 16 * 16;
  printf("%-28s blocks=%4d  %8.3f ms  %7.1f TFLOPS(wmma)  %6.2f ns/wmma/wave\n", name, blocks, ms, flop / ms / 1e9,
         ms * 1e6 / (8.0 * iters) / ((double)blocks * 8 / 256.0));   // per-wave time normalized to 256 waves resident
  (void)hipFree(out);
}

int main() {
  for (int blocks : {64, 128, 256, 512}) {
    run<0, 0>("fp8 wmma", blocks);
    run<1, 0>("bf16 wmma", blocks);
  }
  run<0, 4>("fp8 wmma + 4 fma", 256);
  run<0, 8>("fp8 wmma + 8 fma", 256);
  run<0, 16>("fp8 wmma + 16 fma", 256);
  run<1, 8>("bf16 wmma + 8 fma", 256);
  run<1, 16>("bf16 wmma + 16 fma", 256);
  return 0;
}
