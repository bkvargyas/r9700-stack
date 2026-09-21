// r9k_ar.hip -- one-shot 2-rank peer-to-peer all-reduce for gfx1201 (RDNA4), plus the IPC helpers it needs.
//
// Independent implementation: written from the standard one-shot all-reduce pattern (each rank pushes its whole
// input into the peer's IPC-shared scratch, then every rank reduces the pair locally), the HIP IPC API, and the
// requirements of our own caller in r9700_vllm/comm/r4d_ar.py. No libr4d source or binary was consulted; see
// notes/independence.md. The ABI deliberately mirrors what our Python side already passes so the two backends are
// interchangeable behind one wrapper.
//
// Protocol, per block b, per call:
//   seq[b] is a device-resident counter the kernel increments, so the buffer choice never depends on host state
//   and the launch is cudagraph-safe (nothing is re-registered between replays):
//     s   = seq[b]           (then seq[b] = s + 1)
//     buf = s & 1            (double buffering: a slow peer may still be reading buffer (s-1)&1)
//   1. write this block's slice of `in` into peer_scratch[buf * slot16 + i]
//   2. release fence, then publish: peer_flags[b] = s + 1
//   3. acquire: spin until flags[b] >= s + 1 (the peer has written OUR scratch slice for this call)
//   4. out[i] = in[i] + scratch[buf * slot16 + i], summed in fp32
//
// Both ranks run the identical reduction on the identical pair of values, so the two outputs are bit-identical to
// each other; for bf16/fp16 inputs the add is done in fp32 and rounded once (RNE), which matches a 2-rank RCCL
// sum. Element order never varies with block count, so results are deterministic across launches.
//
// Memory model on gfx1201: the scratch is fine-grained (device-uncached) memory when hipExtMallocWithFlags allows
// it, so peer writes land in the fabric without an explicit cache flush; `drain`/`acq` select how hard we fence:
//   drain: 0 none, 1 release fence (agent), 2 + buffer_wbl2, 3 system-scope release (default, correct everywhere)
//   acq:   0 plain volatile poll (fine-grained scratch), 1 poll + system-scope acquire fence (coarse-grained)
// The defaults the caller passes (3, 0) for fine-grained and (3, 1) for coarse match those two cases.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

#define R9K_AR_MAX_BLOCKS 64
#define AR_WAVE 32

typedef unsigned int  u32;
typedef unsigned long u64;

// ---------------------------------------------------------------- element conversion (fp32 accumulate)
__device__ __forceinline__ float ar_ld(const __hip_bfloat16 v) { return (float)v; }
__device__ __forceinline__ float ar_ld(const _Float16 v) { return (float)v; }
__device__ __forceinline__ float ar_ld(const float v) { return v; }
__device__ __forceinline__ void ar_st(__hip_bfloat16& d, float v) { d = (__hip_bfloat16)v; }
__device__ __forceinline__ void ar_st(_Float16& d, float v) { d = (_Float16)v; }
__device__ __forceinline__ void ar_st(float& d, float v) { d = v; }

// ---------------------------------------------------------------- fences
__device__ __forceinline__ void ar_release(int drain) {
  if (drain == 0) return;
  if (drain >= 3) { __threadfence_system(); return; }   // system scope: visible to the peer device
  __threadfence();
}

__device__ __forceinline__ void ar_acquire(int acq) {
  if (acq) __threadfence_system();
}

// ---------------------------------------------------------------- one-shot 2-rank all-reduce
// Each block owns a contiguous slice of the 16-byte words so that every lane's global access is a full uint4 and
// consecutive lanes touch consecutive words (coalesced both when pushing to the peer and when reducing locally).
template <typename T>
__global__ void __launch_bounds__(1024) r9k_ar_oneshot_2rank_kernel(
    uint4* __restrict__ peer_scratch, const uint4* __restrict__ scratch,
    u32* __restrict__ peer_flags, volatile u32* __restrict__ flags, int* __restrict__ seq,
    long slot16, const uint4* __restrict__ in, uint4* __restrict__ out, long n16,
    int drain, int acq) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;

  // Per-block sequence number: one thread advances it, the rest read it from LDS.
  __shared__ int s_seq;
  if (tid == 0) s_seq = seq[b]++;
  __syncthreads();
  const int s = s_seq;
  const long base = (long)(s & 1) * slot16;

  // This block's slice of the message, in 16-byte words.
  const long per = (n16 + nb - 1) / nb;
  const long lo = per * b, hi = min(lo + per, n16);

  // 1. push our slice into the peer's scratch
  for (long i = lo + tid; i < hi; i += nt) peer_scratch[base + i] = in[i];

  // 2. publish: every write above must be visible to the peer before the flag is
  __syncthreads();
  ar_release(drain);
  if (tid == 0) __hip_atomic_store(&peer_flags[b], (u32)(s + 1), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);

  // 3. wait for the peer's slice to land in OUR scratch
  if (tid == 0) {
    while ((int)__hip_atomic_load(&flags[b], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < s + 1) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  ar_acquire(acq);

  // 4. reduce the pair, fp32 accumulate
  constexpr int NE = 16 / sizeof(T);                       // elements per 16-byte word
  for (long i = lo + tid; i < hi; i += nt) {
    uint4 a = in[i], c = scratch[base + i];
    T* pa = (T*)&a;
    const T* pc = (const T*)&c;
#pragma unroll
    for (int e = 0; e < NE; e++) ar_st(pa[e], ar_ld(pa[e]) + ar_ld(pc[e]));
    out[i] = a;
  }
}

extern "C" int r9k_ar_max_blocks() { return R9K_AR_MAX_BLOCKS; }

// ---------------------------------------------------------------- push-only exchange (for a pre-packed payload)
// Same handshake as above, but it only delivers `src` into the peer's scratch slot and waits for the peer's
// delivery to land in ours; the caller reduces the two payloads itself. Used by the compressed path, where the
// thing on the wire is the packed byte stream (r9k_ar_wht.hip) rather than the raw tensor, so the reduction has
// to happen after the exchange rather than inside it. Byte count must be a multiple of 16.
__global__ void __launch_bounds__(1024) r9k_ar_push_2rank_kernel(
    uint4* __restrict__ peer_scratch, u32* __restrict__ peer_flags, volatile u32* __restrict__ flags,
    int* __restrict__ seq, long slot16, const uint4* __restrict__ src, long n16, int drain, int acq,
    int* __restrict__ slot_out) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ int s_seq;
  if (tid == 0) s_seq = seq[b]++;
  __syncthreads();
  const int s = s_seq;
  const long base = (long)(s & 1) * slot16;
  if (b == 0 && tid == 0 && slot_out) *slot_out = s & 1;

  const long per = (n16 + nb - 1) / nb;
  const long lo = per * b, hi = min(lo + per, n16);
  for (long i = lo + tid; i < hi; i += nt) peer_scratch[base + i] = src[i];

  __syncthreads();
  ar_release(drain);
  if (tid == 0) __hip_atomic_store(&peer_flags[b], (u32)(s + 1), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  if (tid == 0) {
    while ((int)__hip_atomic_load(&flags[b], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < s + 1) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  ar_acquire(acq);
}

// `slot_out` (device int, may be 0) receives which half of the scratch the peer's payload landed in, so the
// caller's reduce can address it without tracking the sequence counter on the host (which would not survive a
// graph replay).
extern "C" int r9k_ar_push_2rank(long peer_scratch, long peer_flags, long flags, long seq, long slot16,
                                 long src, long nbytes, long stream, int nblocks, int nthreads,
                                 int drain, int acq, long slot_out) {
  if (nbytes % 16) return -1;
  const long n16 = nbytes / 16;
  if (n16 <= 0) return 0;
  if (nblocks <= 0 || nblocks > R9K_AR_MAX_BLOCKS) return -2;
  if ((long)nblocks > n16) nblocks = (int)n16;
  if (nthreads <= 0) {
    const long per = (n16 + nblocks - 1) / nblocks;
    long t = ((per + AR_WAVE - 1) / AR_WAVE) * AR_WAVE;
    nthreads = (int)(t < AR_WAVE ? AR_WAVE : (t > 1024 ? 1024 : t));
  }
  if (nthreads > 1024 || nthreads % AR_WAVE) return -3;
  hipLaunchKernelGGL(r9k_ar_push_2rank_kernel, dim3(nblocks), dim3(nthreads), 0, (hipStream_t)stream,
                     (uint4*)peer_scratch, (u32*)peer_flags, (volatile u32*)flags, (int*)seq, slot16,
                     (const uint4*)src, n16, drain, acq, (int*)slot_out);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

// dtype: 0 bf16, 1 fp16, 2 fp32. `numel` is elements; the caller guarantees numel*sizeof % 16 == 0.
extern "C" int r9k_ar_oneshot_2rank(long peer_scratch, long scratch, long peer_flags, long flags, long seq,
                                    long slot16, long in, long out, long numel, int dtype, long stream,
                                    int nblocks, int nthreads, int drain, int acq) {
  const int esz = dtype == 2 ? 4 : 2;
  const long nbytes = numel * esz;
  if (nbytes % 16) return -1;
  const long n16 = nbytes / 16;
  if (nblocks <= 0 || nblocks > R9K_AR_MAX_BLOCKS) return -2;
  if (n16 <= 0) return 0;
  if (nthreads <= 0) {   // auto: enough waves to cover the block's slice once, so small messages stay cheap
    const long per = (n16 + nblocks - 1) / nblocks;
    long t = ((per + AR_WAVE - 1) / AR_WAVE) * AR_WAVE;
    nthreads = (int)(t < AR_WAVE ? AR_WAVE : (t > 1024 ? 1024 : t));
  }
  if (nthreads > 1024 || nthreads % AR_WAVE) return -3;
  if ((long)nblocks > n16) nblocks = (int)n16;            // never launch idle blocks: they would still bump seq
  hipStream_t st = (hipStream_t)stream;
#define AR_LAUNCH(T)                                                                                     \
  hipLaunchKernelGGL((r9k_ar_oneshot_2rank_kernel<T>), dim3(nblocks), dim3(nthreads), 0, st,             \
                     (uint4*)peer_scratch, (const uint4*)scratch, (u32*)peer_flags, (volatile u32*)flags, \
                     (int*)seq, slot16, (const uint4*)in, (uint4*)out, n16, drain, acq)
  switch (dtype) {
    case 0: AR_LAUNCH(__hip_bfloat16); break;
    case 1: AR_LAUNCH(_Float16); break;
    case 2: AR_LAUNCH(float); break;
    default: return -4;
  }
#undef AR_LAUNCH
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

// ---------------------------------------------------------------- IPC helpers
// `fine` asks for fine-grained (uncached) device memory, which lets a peer's writes reach us without an explicit
// L2 writeback; falls back to a plain hipMalloc when the flag is unsupported (the caller then passes acq=1).
extern "C" int r9k_ar_ipc_alloc(long nbytes, int fine, long* out_ptr, void* out_handle) {
  void* p = nullptr;
  // An IPC handle can only be taken for the base of a real allocation, and a small request may be suballocated
  // out of a pool whose base is not ours: round up so the flag array (a few hundred bytes) gets its own block.
  const long kMinIpc = 2 << 20;
  if (nbytes < kMinIpc) nbytes = kMinIpc;
  hipError_t e = fine ? hipExtMallocWithFlags(&p, (size_t)nbytes, hipDeviceMallocUncached)
                      : hipMalloc(&p, (size_t)nbytes);
  if (e != hipSuccess || !p) return -1;
  if (hipMemset(p, 0, (size_t)nbytes) != hipSuccess) return -2;
  if (hipIpcGetMemHandle((hipIpcMemHandle_t*)out_handle, p) != hipSuccess) return -3;
  *out_ptr = (long)p;
  return 0;
}

extern "C" int r9k_ar_ipc_open(const void* handle, long* out_ptr) {
  void* p = nullptr;
  hipIpcMemHandle_t h;
  __builtin_memcpy(&h, handle, sizeof(h));
  if (hipIpcOpenMemHandle(&p, h, hipIpcMemLazyEnablePeerAccess) != hipSuccess) return -1;
  *out_ptr = (long)p;
  return 0;
}

extern "C" int r9k_ar_ipc_handle_size() { return (int)sizeof(hipIpcMemHandle_t); }
