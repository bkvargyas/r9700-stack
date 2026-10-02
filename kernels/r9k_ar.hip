// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Brian Vargyas

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
//
//   EVERY CALL MUST ADVANCE EVERY COUNTER (the `grid` argument of the *_g entry points). Double buffering only
//   protects the previous call if all blocks agree on `buf`. Until 2026-10-02 the launch grid was the number of
//   data blocks, which follows the message size, so a block that sat out a small message kept its old parity and
//   on the next, larger message wrote into the SAME half as the message before -- at an offset that overlaps that
//   message's tail whenever the new slices are shorter (per' * nb_prev < n16_prev). A rank one call ahead then
//   overwrote the last rows of the previous message before its peer had summed them: the two ranks disagreed on
//   the last sequence of the batch, which decoded garbage from there on. It needed two back-to-back messages of
//   different sizes with the ranks slightly out of step -- a prompt joining a batch of running decodes under
//   graph replay -- and showed as 5-8% garbled answers once more requests were submitted than max_num_seqs.
//   Now the grid is fixed per (scratch, seq) pair and blocks beyond the data blocks only advance their counter.
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
//   drain: 0 none, 1/2 device-scope fence (NOT sufficient for a peer GPU -- see below), 3 system-scope fence
//          then a relaxed flag store, 4 (default) RELEASE store on the flag and no separate fence.
//   acq:   0 plain relaxed poll, 1 poll + system-scope acquire fence, 2 (default) ACQUIRE load on the poll.
//
// Why 4/2 rather than 3/0: the handshake needs the peer's *device* to observe our data writes before it observes
// our flag. A release store at system scope expresses exactly that, and the hardware emits the minimum barrier
// for it. A blanket __threadfence_system() plus a relaxed store expresses something stronger (drain everything)
// and measured ~54 us per call more at 2 MB. drain=1/2 measured fast AND correct in a lockstep two-rank test,
// but they are __threadfence(), which is agent scope and orders nothing for the peer device -- that is a race
// that happens to pass, not a valid setting. Kept only so the sweep can show the difference.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

#define R9K_AR_MAX_BLOCKS 256   // prefill-sized two-shot pushes want most CUs issuing stores
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
  if (acq == 1) __threadfence_system();        // acq == 2 does the acquire on the load itself
}

// Publish the flag so the peer sees our data writes first. drain >= 4: a single RELEASE store carries the
// ordering. Otherwise: an explicit fence of the requested strength, then a relaxed store.
__device__ __forceinline__ void ar_publish(u32* flag, u32 val, int drain, bool leader) {
  if (drain >= 4) {
    if (leader) __hip_atomic_store(flag, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    return;
  }
  ar_release(drain);
  if (leader) __hip_atomic_store(flag, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Spin until the peer has published. acq >= 2: ACQUIRE load, so the peer's data writes are visible to us
// afterwards without a separate fence.
__device__ __forceinline__ void ar_wait(volatile u32* flag, int target, int acq) {
  if (acq >= 2) {
    while ((int)__hip_atomic_load((u32*)flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < target)
      __builtin_amdgcn_s_sleep(1);
  } else {
    while ((int)__hip_atomic_load((u32*)flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < target)
      __builtin_amdgcn_s_sleep(1);
  }
}

// ---------------------------------------------------------------- one-shot 2-rank all-reduce
// Each block owns a contiguous slice of the 16-byte words so that every lane's global access is a full uint4 and
// consecutive lanes touch consecutive words (coalesced both when pushing to the peer and when reducing locally).
template <typename T>
__global__ void __launch_bounds__(1024) r9k_ar_oneshot_2rank_kernel(
    uint4* __restrict__ peer_scratch, const uint4* __restrict__ scratch,
    u32* __restrict__ peer_flags, volatile u32* __restrict__ flags, int* __restrict__ seq,
    long slot16, const uint4* __restrict__ in, uint4* __restrict__ out, long n16,
    int drain, int acq, int nd) {
  const int b = blockIdx.x, nb = nd, tid = threadIdx.x, nt = blockDim.x;

  // Per-block sequence number: one thread advances it, the rest read it from LDS.
  __shared__ int s_seq;
  if (tid == 0) s_seq = seq[b]++;
  __syncthreads();
  if (b >= nd) return;                  // no data for this block: it only keeps its counter in step
  const int s = s_seq;
  const long base = (long)(s & 1) * slot16;

  // This block's slice of the message, in 16-byte words.
  const long per = (n16 + nb - 1) / nb;
  const long lo = per * b, hi = min(lo + per, n16);

  // 1. push our slice into the peer's scratch
  for (long i = lo + tid; i < hi; i += nt) peer_scratch[base + i] = in[i];

  // 2. publish: every write above must be visible to the peer before the flag is
  __syncthreads();
  ar_publish(&peer_flags[b], (u32)(s + 1), drain, tid == 0);

  // 3. wait for the peer's slice to land in OUR scratch
  if (tid == 0) ar_wait(&flags[b], s + 1, acq);
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
    int* __restrict__ slot_out, int nd) {
  const int b = blockIdx.x, nb = nd, tid = threadIdx.x, nt = blockDim.x;
  __shared__ int s_seq;
  if (tid == 0) s_seq = seq[b]++;
  __syncthreads();
  const int s = s_seq;
  const long base = (long)(s & 1) * slot16;
  if (b == 0 && tid == 0 && slot_out) *slot_out = s & 1;
  if (b >= nd) return;

  const long per = (n16 + nb - 1) / nb;
  const long lo = per * b, hi = min(lo + per, n16);
  for (long i = lo + tid; i < hi; i += nt) peer_scratch[base + i] = src[i];

  __syncthreads();
  ar_publish(&peer_flags[b], (u32)(s + 1), drain, tid == 0);
  if (tid == 0) ar_wait(&flags[b], s + 1, acq);
  __syncthreads();
  ar_acquire(acq);
}

// `slot_out` (device int, may be 0) receives which half of the scratch the peer's payload landed in, so the
// caller's reduce can address it without tracking the sequence counter on the host (which would not survive a
// graph replay).
// grid: blocks launched (>= the data blocks; 0 = just the data blocks, which is only safe when every call on this
// scratch has the same block count). See the protocol note at the top.
extern "C" int r9k_ar_push_2rank_g(long peer_scratch, long peer_flags, long flags, long seq, long slot16,
                                   long src, long nbytes, long stream, int nblocks, int nthreads,
                                   int drain, int acq, long slot_out, int grid) {
  if (nbytes % 16) return -1;
  const long n16 = nbytes / 16;
  if (n16 <= 0) return 0;
  if (nblocks <= 0 || nblocks > R9K_AR_MAX_BLOCKS) return -2;
  if ((long)nblocks > n16) nblocks = (int)n16;
  if (grid < nblocks) grid = nblocks;
  if (grid > R9K_AR_MAX_BLOCKS) return -2;
  if (nthreads <= 0) {
    const long per = (n16 + nblocks - 1) / nblocks;
    long t = ((per + AR_WAVE - 1) / AR_WAVE) * AR_WAVE;
    nthreads = (int)(t < AR_WAVE ? AR_WAVE : (t > 1024 ? 1024 : t));
  }
  if (nthreads > 1024 || nthreads % AR_WAVE) return -3;
  hipLaunchKernelGGL(r9k_ar_push_2rank_kernel, dim3(grid), dim3(nthreads), 0, (hipStream_t)stream,
                     (uint4*)peer_scratch, (u32*)peer_flags, (volatile u32*)flags, (int*)seq, slot16,
                     (const uint4*)src, n16, drain, acq, (int*)slot_out, nblocks);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

extern "C" int r9k_ar_push_2rank(long peer_scratch, long peer_flags, long flags, long seq, long slot16,
                                 long src, long nbytes, long stream, int nblocks, int nthreads,
                                 int drain, int acq, long slot_out) {
  return r9k_ar_push_2rank_g(peer_scratch, peer_flags, flags, seq, slot16, src, nbytes, stream, nblocks, nthreads,
                             drain, acq, slot_out, 0);
}

// dtype: 0 bf16, 1 fp16, 2 fp32. `numel` is elements; the caller guarantees numel*sizeof % 16 == 0.
extern "C" int r9k_ar_oneshot_2rank_g(long peer_scratch, long scratch, long peer_flags, long flags, long seq,
                                      long slot16, long in, long out, long numel, int dtype, long stream,
                                      int nblocks, int nthreads, int drain, int acq, int grid) {
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
  if ((long)nblocks > n16) nblocks = (int)n16;            // data blocks; the grid may be larger (idle blocks)
  if (grid < nblocks) grid = nblocks;
  if (grid > R9K_AR_MAX_BLOCKS) return -2;
  hipStream_t st = (hipStream_t)stream;
#define AR_LAUNCH(T)                                                                                     \
  hipLaunchKernelGGL((r9k_ar_oneshot_2rank_kernel<T>), dim3(grid), dim3(nthreads), 0, st,                \
                     (uint4*)peer_scratch, (const uint4*)scratch, (u32*)peer_flags, (volatile u32*)flags, \
                     (int*)seq, slot16, (const uint4*)in, (uint4*)out, n16, drain, acq, nblocks)
  switch (dtype) {
    case 0: AR_LAUNCH(__hip_bfloat16); break;
    case 1: AR_LAUNCH(_Float16); break;
    case 2: AR_LAUNCH(float); break;
    default: return -4;
  }
#undef AR_LAUNCH
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

// grid = the data blocks: safe only when every call on this scratch has the same block count (tests, tuning).
extern "C" int r9k_ar_oneshot_2rank(long peer_scratch, long scratch, long peer_flags, long flags, long seq,
                                    long slot16, long in, long out, long numel, int dtype, long stream,
                                    int nblocks, int nthreads, int drain, int acq) {
  return r9k_ar_oneshot_2rank_g(peer_scratch, scratch, peer_flags, flags, seq, slot16, in, out, numel, dtype, stream,
                                nblocks, nthreads, drain, acq, 0);
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

// ---------------------------------------------------------------- one-shot N-rank all-reduce (TP=4)
// The same one-shot pattern for N <= R9K_AR_MAX_RANKS ranks: every rank pushes its whole input into slot `rank`
// of every peer's scratch, publishes one flag per peer, waits for the N-1 flags addressed to it, then reduces all
// N inputs locally. Latency-optimal for decode-sized messages (one handshake, no second pass); for large
// messages it moves (N-1)x the input per rank, so the caller keeps RCCL above a size threshold.
//
// Layout (per rank):  scratch[buf][src rank][slot16 words],  flags[src rank][R9K_AR_MAX_BLOCKS]
// Every rank sums the N inputs in rank order 0..N-1 in fp32 and rounds once, so all ranks produce bit-identical
// outputs (the order differs from RCCL's ring, so the result is not bit-equal to RCCL -- it is equal to an
// fp32-accumulated reference). Double buffering is safe for the same reason as the 2-rank kernel: a rank reaches
// call s+2 only after every peer has published call s+1, which each peer does only after its call-s kernel
// (and therefore its reads of buffer s&1) finished in stream order.
#define R9K_AR_MAX_RANKS 8

struct R9kArPeers {
  uint4* scratch[R9K_AR_MAX_RANKS];   // scratch[q] = rank q's scratch, mapped into this process (own for q == rank)
  u32* flags[R9K_AR_MAX_RANKS];       // flags[q]   = rank q's flag array, mapped likewise
};

template <typename T, int N>
__global__ void __launch_bounds__(1024) r9k_ar_oneshot_nrank_kernel(
    R9kArPeers P, int rank, int* __restrict__ seq, long slot16, const uint4* __restrict__ in,
    uint4* __restrict__ out, long n16, int drain, int acq, int nd) {
  const int b = blockIdx.x, nb = nd, tid = threadIdx.x, nt = blockDim.x;
  __shared__ int s_seq;
  if (tid == 0) s_seq = seq[b]++;
  __syncthreads();
  if (b >= nd) return;                  // idle block: keeps its counter in step with the data blocks
  const int s = s_seq;
  const long base = (long)(s & 1) * N * slot16;
  const long per = (n16 + nb - 1) / nb;
  const long lo = per * b, hi = min(lo + per, n16);

  // 1. push our slice into slot `rank` of every peer
  for (long i = lo + tid; i < hi; i += nt) {
    const uint4 v = in[i];
#pragma unroll
    for (int q = 0; q < N; q++)
      if (q != rank) P.scratch[q][base + (long)rank * slot16 + i] = v;
  }

  // 2. publish to each peer; 3. wait for every peer's publish to us
  __syncthreads();
  if (drain < 4) ar_release(drain);
  if (tid < N && tid != rank) {
    u32* f = &P.flags[tid][rank * R9K_AR_MAX_BLOCKS + b];
    if (drain >= 4) __hip_atomic_store(f, (u32)(s + 1), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    else __hip_atomic_store(f, (u32)(s + 1), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  if (tid < N && tid != rank) ar_wait(&P.flags[rank][tid * R9K_AR_MAX_BLOCKS + b], s + 1, acq);
  __syncthreads();
  ar_acquire(acq);

  // 4. reduce all N in rank order, fp32 accumulate, one rounding
  constexpr int NE = 16 / sizeof(T);
  const uint4* own = P.scratch[rank] + base;
  for (long i = lo + tid; i < hi; i += nt) {
    float acc[NE];
#pragma unroll
    for (int e = 0; e < NE; e++) acc[e] = 0.f;
#pragma unroll
    for (int q = 0; q < N; q++) {
      const uint4 c = q == rank ? in[i] : own[(long)q * slot16 + i];
      const T* pc = (const T*)&c;
#pragma unroll
      for (int e = 0; e < NE; e++) acc[e] += ar_ld(pc[e]);
    }
    uint4 r;
    T* pr = (T*)&r;
#pragma unroll
    for (int e = 0; e < NE; e++) ar_st(pr[e], acc[e]);
    out[i] = r;
  }
}

extern "C" int r9k_ar_max_ranks() { return R9K_AR_MAX_RANKS; }

// scratch_ptrs / flag_ptrs: host arrays of `nranks` device pointers (own + opened peer mappings), indexed by rank.
extern "C" int r9k_ar_oneshot_nrank_g(const long* scratch_ptrs, const long* flag_ptrs, int nranks, int rank,
                                    long seq, long slot16, long in, long out, long numel, int dtype,
                                    long stream, int nblocks, int nthreads, int drain, int acq, int grid) {
  const int esz = dtype == 2 ? 4 : 2;
  const long nbytes = numel * esz;
  if (nbytes % 16) return -1;
  const long n16 = nbytes / 16;
  if (nblocks <= 0 || nblocks > R9K_AR_MAX_BLOCKS) return -2;
  if (nranks < 2 || nranks > R9K_AR_MAX_RANKS || rank < 0 || rank >= nranks) return -5;
  if (n16 <= 0) return 0;
  if (n16 > slot16) return -7;
  if (nthreads <= 0) {
    const long per = (n16 + nblocks - 1) / nblocks;
    long t = ((per + AR_WAVE - 1) / AR_WAVE) * AR_WAVE;
    nthreads = (int)(t < AR_WAVE ? AR_WAVE : (t > 1024 ? 1024 : t));
  }
  if (nthreads < R9K_AR_MAX_RANKS) nthreads = AR_WAVE;    // the flag lanes are tid < nranks
  if (nthreads > 1024 || nthreads % AR_WAVE) return -3;
  if ((long)nblocks > n16) nblocks = (int)n16;
  if (grid < nblocks) grid = nblocks;
  if (grid > R9K_AR_MAX_BLOCKS) return -2;
  R9kArPeers P{};
  for (int q = 0; q < nranks; q++) { P.scratch[q] = (uint4*)scratch_ptrs[q]; P.flags[q] = (u32*)flag_ptrs[q]; }
  hipStream_t st = (hipStream_t)stream;
#define ARN_LAUNCH(T, N)                                                                                   \
  hipLaunchKernelGGL((r9k_ar_oneshot_nrank_kernel<T, N>), dim3(grid), dim3(nthreads), 0, st, P, rank,     \
                     (int*)seq, slot16, (const uint4*)in, (uint4*)out, n16, drain, acq, nblocks)
#define ARN_DT(N)                                                                                          \
  switch (dtype) {                                                                                         \
    case 0: ARN_LAUNCH(__hip_bfloat16, N); break;                                                          \
    case 1: ARN_LAUNCH(_Float16, N); break;                                                                \
    case 2: ARN_LAUNCH(float, N); break;                                                                   \
    default: return -4;                                                                                    \
  }
  switch (nranks) {
    case 2: ARN_DT(2); break;
    case 4: ARN_DT(4); break;
    case 8: ARN_DT(8); break;
    default: return -5;
  }
#undef ARN_DT
#undef ARN_LAUNCH
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

// grid = the data blocks (see r9k_ar_oneshot_2rank).
extern "C" int r9k_ar_oneshot_nrank(const long* scratch_ptrs, const long* flag_ptrs, int nranks, int rank,
                                    long seq, long slot16, long in, long out, long numel, int dtype,
                                    long stream, int nblocks, int nthreads, int drain, int acq) {
  return r9k_ar_oneshot_nrank_g(scratch_ptrs, flag_ptrs, nranks, rank, seq, slot16, in, out, numel, dtype, stream,
                                nblocks, nthreads, drain, acq, 0);
}

// ---------------------------------------------------------------- two-shot N-rank all-reduce
// Reduce-scatter then all-gather in one kernel, for messages where one-shot's (N-1)x push is bandwidth-bound:
// each rank moves 2(N-1)/N of the input instead of (N-1)x (1.5x vs 3x at N=4). The message is cut into N
// chunks of 16-byte words; rank c owns chunk c.
//   1. push chunk q of our input into slot `rank` of rank q's scratch; publish 2s+1 to every peer
//   2. wait for 2s+1 from every peer; reduce OUR chunk over all N in rank order (fp32, one rounding)
//   3. push the reduced chunk into the gather area of every peer (and our own out); publish 2s+2
//   4. wait for 2s+2 from every peer; copy their reduced chunks from our gather area into out
// One flag word per (source rank, block) carries both phases: 2s+2 >= 2s+1, so a peer that is already in
// phase 3 also satisfies a phase-1 wait. Only the chunk's owner reduces it, so all ranks hold identical bits.
// Double buffering (s & 1) on both areas is safe by the same stream-order argument as the one-shot kernel.
// Layout (per rank): scratch[buf][ rs: N slots of slot16 | ag: slot16 ] -- the caller sizes it 2*(N+1)*slot16.
template <typename T, int N>
__global__ void __launch_bounds__(1024) r9k_ar_twoshot_nrank_kernel(
    R9kArPeers P, int rank, int* __restrict__ seq, long slot16, const uint4* __restrict__ in,
    uint4* __restrict__ out, long n16, int drain, int acq, int nd) {
  const int b = blockIdx.x, nb = nd, tid = threadIdx.x, nt = blockDim.x;
  __shared__ int s_seq;
  if (tid == 0) s_seq = seq[b]++;
  __syncthreads();
  if (b >= nd) return;                  // idle block: keeps its counter in step with the data blocks
  const int s = s_seq;
  const long base = (long)(s & 1) * (N + 1) * slot16;
  const long ag = base + (long)N * slot16;
  const long csz = (n16 + N - 1) / N;                 // words per chunk (the last one may be short)
  const long per = (csz + nb - 1) / nb;               // this block's slice of every chunk
  const long lo = per * b, hi = min(lo + per, csz);

  // 1. scatter: chunk q -> rank q (4 words per lane per iteration so several stores are in flight per lane)
#pragma unroll
  for (int q = 0; q < N; q++) {
    if (q == rank) continue;
    const long c0 = (long)q * csz, cn = min(csz, n16 - c0), end = min(hi, cn);
    uint4* dst = P.scratch[q] + base + (long)rank * slot16;
    long i = lo + tid;
    for (; i + 3 * nt < end; i += 4 * nt) {
      const uint4 v0 = in[c0 + i], v1 = in[c0 + i + nt], v2 = in[c0 + i + 2 * nt], v3 = in[c0 + i + 3 * nt];
      dst[i] = v0; dst[i + nt] = v1; dst[i + 2 * nt] = v2; dst[i + 3 * nt] = v3;
    }
    for (; i < end; i += nt) dst[i] = in[c0 + i];
  }
  __syncthreads();
  if (drain < 4) ar_release(drain);
  auto publish = [&](u32 v) {
    if (tid < N && tid != rank) {
      u32* f = &P.flags[tid][rank * R9K_AR_MAX_BLOCKS + b];
      if (drain >= 4) __hip_atomic_store(f, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      else __hip_atomic_store(f, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  };
  publish((u32)(2 * s + 1));
  if (tid < N && tid != rank) ar_wait(&P.flags[rank][tid * R9K_AR_MAX_BLOCKS + b], 2 * s + 1, acq);
  __syncthreads();
  ar_acquire(acq);

  // 2+3. reduce our chunk, write it to out and to every peer's gather area
  constexpr int NE = 16 / sizeof(T);
  const uint4* own = P.scratch[rank] + base;
  {
    const long c0 = (long)rank * csz, cn = min(csz, n16 - c0);
    for (long i = lo + tid; i < min(hi, cn); i += nt) {
      float acc[NE];
#pragma unroll
      for (int e = 0; e < NE; e++) acc[e] = 0.f;
#pragma unroll
      for (int q = 0; q < N; q++) {
        const uint4 c = q == rank ? in[c0 + i] : own[(long)q * slot16 + i];
        const T* pc = (const T*)&c;
#pragma unroll
        for (int e = 0; e < NE; e++) acc[e] += ar_ld(pc[e]);
      }
      uint4 r;
      T* pr = (T*)&r;
#pragma unroll
      for (int e = 0; e < NE; e++) ar_st(pr[e], acc[e]);
      out[c0 + i] = r;
#pragma unroll
      for (int q = 0; q < N; q++)
        if (q != rank) P.scratch[q][ag + c0 + i] = r;   // gather area is indexed by global word
    }
  }
  __syncthreads();
  if (drain < 4) ar_release(drain);
  publish((u32)(2 * s + 2));
  if (tid < N && tid != rank) ar_wait(&P.flags[rank][tid * R9K_AR_MAX_BLOCKS + b], 2 * s + 2, acq);
  __syncthreads();
  ar_acquire(acq);

  // 4. collect the peers' reduced chunks
  const uint4* gat = P.scratch[rank] + ag;
#pragma unroll
  for (int q = 0; q < N; q++) {
    if (q == rank) continue;
    const long c0 = (long)q * csz, cn = min(csz, n16 - c0);
    for (long i = lo + tid; i < min(hi, cn); i += nt) out[c0 + i] = gat[c0 + i];
  }
}

// Same arguments as r9k_ar_oneshot_nrank; scratch must be 2*(nranks+1)*slot16 words and the flag values are
// NOT compatible with the one-shot kernel's, so the two need separate seq counters AND separate flag arrays.
extern "C" int r9k_ar_twoshot_nrank_g(const long* scratch_ptrs, const long* flag_ptrs, int nranks, int rank,
                                    long seq, long slot16, long in, long out, long numel, int dtype,
                                    long stream, int nblocks, int nthreads, int drain, int acq, int grid) {
  const int esz = dtype == 2 ? 4 : 2;
  const long nbytes = numel * esz;
  if (nbytes % 16) return -1;
  const long n16 = nbytes / 16;
  if (nblocks <= 0 || nblocks > R9K_AR_MAX_BLOCKS) return -2;
  if (nranks < 2 || nranks > R9K_AR_MAX_RANKS || rank < 0 || rank >= nranks) return -5;
  if (n16 <= 0) return 0;
  if (n16 > slot16) return -7;
  const long csz = (n16 + nranks - 1) / nranks;
  if (nthreads <= 0) {
    const long per = (csz + nblocks - 1) / nblocks;
    long t = ((per + AR_WAVE - 1) / AR_WAVE) * AR_WAVE;
    nthreads = (int)(t < AR_WAVE ? AR_WAVE : (t > 1024 ? 1024 : t));
  }
  if (nthreads > 1024 || nthreads % AR_WAVE) return -3;
  if ((long)nblocks > csz) nblocks = (int)csz;
  if (grid < nblocks) grid = nblocks;
  if (grid > R9K_AR_MAX_BLOCKS) return -2;
  R9kArPeers P{};
  for (int q = 0; q < nranks; q++) { P.scratch[q] = (uint4*)scratch_ptrs[q]; P.flags[q] = (u32*)flag_ptrs[q]; }
  hipStream_t st = (hipStream_t)stream;
#define ARN_LAUNCH(T, N)                                                                                   \
  hipLaunchKernelGGL((r9k_ar_twoshot_nrank_kernel<T, N>), dim3(grid), dim3(nthreads), 0, st, P, rank,     \
                     (int*)seq, slot16, (const uint4*)in, (uint4*)out, n16, drain, acq, nblocks)
#define ARN_DT(N)                                                                                          \
  switch (dtype) {                                                                                         \
    case 0: ARN_LAUNCH(__hip_bfloat16, N); break;                                                          \
    case 1: ARN_LAUNCH(_Float16, N); break;                                                                \
    case 2: ARN_LAUNCH(float, N); break;                                                                   \
    default: return -4;                                                                                    \
  }
  switch (nranks) {
    case 2: ARN_DT(2); break;
    case 4: ARN_DT(4); break;
    case 8: ARN_DT(8); break;
    default: return -5;
  }
#undef ARN_DT
#undef ARN_LAUNCH
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

// grid = the data blocks (see r9k_ar_oneshot_2rank).
extern "C" int r9k_ar_twoshot_nrank(const long* scratch_ptrs, const long* flag_ptrs, int nranks, int rank,
                                    long seq, long slot16, long in, long out, long numel, int dtype,
                                    long stream, int nblocks, int nthreads, int drain, int acq) {
  return r9k_ar_twoshot_nrank_g(scratch_ptrs, flag_ptrs, nranks, rank, seq, slot16, in, out, numel, dtype, stream,
                                nblocks, nthreads, drain, acq, 0);
}
