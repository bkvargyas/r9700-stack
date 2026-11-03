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

// ---------------------------------------------------------------- one-shot N-rank all-gather (TP=4)
// The push half of the one-shot all-reduce without the reduce: every rank pushes its input into slot `rank` of
// every peer's scratch, one handshake, then writes the gathered result in vLLM's concat layout for `dim`:
//   in  = [rows, cols] (cols = the trailing dims from `dim` on, contiguous), out = [rows, N * cols],
//   out[r, q * cols + c] = in_q[r, c]
// which is what GroupCoordinator.all_gather produces with all_gather_into_tensor + movedim + reshape, in one
// launch and without the movedim copy. Own pool (scratch / flags / seq) so the all-reduce protocol is untouched.
// cols * sizeof(T) must be a multiple of 16; rows * cols words of 16 bytes must fit the slot.
template <int N>
__global__ void __launch_bounds__(1024) r9k_ag_oneshot_nrank_kernel(
    R9kArPeers P, int rank, int* __restrict__ seq, long slot16, const uint4* __restrict__ in,
    uint4* __restrict__ out, long n16, long cols16, int drain, int acq, int nd) {
  const int b = blockIdx.x, nb = nd, tid = threadIdx.x, nt = blockDim.x;
  __shared__ int s_seq;
  if (tid == 0) s_seq = seq[b]++;
  __syncthreads();
  if (b >= nd) return;
  const int s = s_seq;
  const long base = (long)(s & 1) * N * slot16;
  const long per = (n16 + nb - 1) / nb;
  const long lo = per * b, hi = min(lo + per, n16);

  // 1. push our slice into slot `rank` of every peer, and place it in our own output
  for (long i = lo + tid; i < hi; i += nt) {
    const uint4 v = in[i];
#pragma unroll
    for (int q = 0; q < N; q++)
      if (q != rank) P.scratch[q][base + (long)rank * slot16 + i] = v;
    const long r = i / cols16, c = i - r * cols16;
    out[r * N * cols16 + (long)rank * cols16 + c] = v;
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

  // 4. the peers' slices from our scratch into their columns of the output
  const uint4* own = P.scratch[rank] + base;
  for (long i = lo + tid; i < hi; i += nt) {
    const long r = i / cols16, c = i - r * cols16;
#pragma unroll
    for (int q = 0; q < N; q++)
      if (q != rank) out[r * N * cols16 + (long)q * cols16 + c] = own[(long)q * slot16 + i];
  }
}

// in: rows x cols elements of `esz` bytes each, contiguous; out: rows x (nranks * cols). Grid rules as the
// all-reduce's (`grid` >= nblocks keeps the per-block sequence counters in step across calls of any size).
extern "C" int r9k_ag_oneshot_nrank_g(const long* scratch_ptrs, const long* flag_ptrs, int nranks, int rank,
                                    long seq, long slot16, long in, long out, long rows, long cols, int esz,
                                    long stream, int nblocks, int nthreads, int drain, int acq, int grid) {
  const long rowbytes = cols * esz;
  if (rowbytes % 16 || rows <= 0) return -1;
  const long cols16 = rowbytes / 16, n16 = rows * cols16;
  if (nblocks <= 0 || nblocks > R9K_AR_MAX_BLOCKS) return -2;
  if (nranks < 2 || nranks > R9K_AR_MAX_RANKS || rank < 0 || rank >= nranks) return -5;
  if (n16 > slot16) return -7;
  if (nthreads <= 0) {
    const long per = (n16 + nblocks - 1) / nblocks;
    long t = ((per + AR_WAVE - 1) / AR_WAVE) * AR_WAVE;
    nthreads = (int)(t < AR_WAVE ? AR_WAVE : (t > 1024 ? 1024 : t));
  }
  if (nthreads < R9K_AR_MAX_RANKS) nthreads = AR_WAVE;
  if (nthreads > 1024 || nthreads % AR_WAVE) return -3;
  if ((long)nblocks > n16) nblocks = (int)n16;
  if (grid < nblocks) grid = nblocks;
  if (grid > R9K_AR_MAX_BLOCKS) return -2;
  R9kArPeers P{};
  for (int q = 0; q < nranks; q++) { P.scratch[q] = (uint4*)scratch_ptrs[q]; P.flags[q] = (u32*)flag_ptrs[q]; }
  hipStream_t st = (hipStream_t)stream;
#define AGN_LAUNCH(N)                                                                                      \
  hipLaunchKernelGGL((r9k_ag_oneshot_nrank_kernel<N>), dim3(grid), dim3(nthreads), 0, st, P, rank,         \
                     (int*)seq, slot16, (const uint4*)in, (uint4*)out, n16, cols16, drain, acq, nblocks)
  switch (nranks) {
    case 2: AGN_LAUNCH(2); break;
    case 4: AGN_LAUNCH(4); break;
    case 8: AGN_LAUNCH(8); break;
    default: return -5;
  }
#undef AGN_LAUNCH
  return (int)hipGetLastError();
}


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
// mode (r9k_ar_twoshot_nrank_m): 0 = all-reduce; 1 = reduce-scatter only (steps 1-2: out holds OUR reduced chunk at
// its place, the rest untouched); 2 = all-gather only (steps 3-4: `in` holds our chunk at its place; out = every
// rank's chunk). The split 8-rank all-reduce of comm/r9k_ar8.py runs 1, then its cross stage on the chunk, then 2.
// Every mode waits only for flag values its own call publishes (2s+1 and/or 2s+2) and every call advances s, so the
// modes share the counters, flags and double buffering of mode 0: a rank reaches call s+2 only after waiting in call
// s+1 for a value each peer publishes after its call-s kernel finished.
template <typename T, int N>
__global__ void __launch_bounds__(1024) r9k_ar_twoshot_nrank_kernel(
    R9kArPeers P, int rank, int* __restrict__ seq, long slot16, const uint4* __restrict__ in,
    uint4* __restrict__ out, long n16, int drain, int acq, int nd, int mode) {
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

  auto publish = [&](u32 v) {
    if (tid < N && tid != rank) {
      u32* f = &P.flags[tid][rank * R9K_AR_MAX_BLOCKS + b];
      if (drain >= 4) __hip_atomic_store(f, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      else __hip_atomic_store(f, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  };
  constexpr int NE = 16 / sizeof(T);
  if (mode != 2) {
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
  publish((u32)(2 * s + 1));
  if (tid < N && tid != rank) ar_wait(&P.flags[rank][tid * R9K_AR_MAX_BLOCKS + b], 2 * s + 1, acq);
  __syncthreads();
  ar_acquire(acq);
  }

  // 2+3. reduce our chunk (mode 2: take it from `in`), write it to out and to every peer's gather area
  const uint4* own = P.scratch[rank] + base;
  if (mode == 2) {
    const long c0 = (long)rank * csz, cn = min(csz, n16 - c0);
    for (long i = lo + tid; i < min(hi, cn); i += nt) {
      const uint4 r = in[c0 + i];
      out[c0 + i] = r;
#pragma unroll
      for (int q = 0; q < N; q++)
        if (q != rank) P.scratch[q][ag + c0 + i] = r;
    }
  } else {
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
      if (mode == 1) continue;
#pragma unroll
      for (int q = 0; q < N; q++)
        if (q != rank) P.scratch[q][ag + c0 + i] = r;   // gather area is indexed by global word
    }
  }
  if (mode == 1) return;
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

// ---------------------------------------------------------------- two-shot N-rank all-reduce through host memory
// For pools in pinned host memory (comm/r9k_ar.py host=True: the cross stage of comm/r9k_ar8.py, 2026-11-02). The
// generic two-shot above pushes each chunk into its OWNER's pool and the reduced chunk into every peer's pool; with
// every pool in host memory both are host writes: 1.5 S up and 1.5 S down a rank. Here each rank writes only its
// own pool and the readers pull:
//   1. write chunks q != rank of our input into our pool's `in` area (0.75 S up); publish 2s+1 to every peer
//   2. wait for 2s+1 from every peer; reduce OUR chunk -- own input from `in`, peer q's from q's pool (0.75 S down)
//      -- in rank order, fp32, one rounding; write it to out and to our pool's `red` area (0.25 S up); publish 2s+2
//   3. wait for 2s+2 from every peer; read their reduced chunks from their `red` areas into out (0.75 S down)
// 1 S up + 1.5 S down, two barriers, the same bits as the generic kernel (the owner reduces its chunk in rank order).
// Host reads cost a round trip of ~1-2 us, so every lane keeps several loads in flight (2 words x N in the reduce,
// 4 words in the copies).
// Layout (per rank): scratch[buf][ in: slot16 | red: slot16 ] = 2*2*slot16 words, addressed by global word; flags and
// seq counters of its own, with the generic two-shot's flag values. Double buffering: rank r rewrites buffer s&1 in
// call s+2 only after every peer published 2(s+1)+1, which a peer does only after its call-s kernel -- and with it
// every read of r's buffer s&1 -- finished in stream order.
template <typename T, int N>
__global__ void __launch_bounds__(1024) r9k_ar_twoshot_host_nrank_kernel(
    R9kArPeers P, int rank, int* __restrict__ seq, long slot16, const uint4* __restrict__ in,
    uint4* __restrict__ out, long n16, int drain, int acq, int nd) {
  const int b = blockIdx.x, nb = nd, tid = threadIdx.x, nt = blockDim.x;
  __shared__ int s_seq;
  if (tid == 0) s_seq = seq[b]++;
  __syncthreads();
  if (b >= nd) return;                  // idle block: keeps its counter in step with the data blocks
  const int s = s_seq;
  const long base = (long)(s & 1) * 2 * slot16;
  const long red = base + slot16;
  const long csz = (n16 + N - 1) / N;
  const long per = (csz + nb - 1) / nb;
  const long lo = per * b, hi = min(lo + per, csz);
  uint4* mine = P.scratch[rank];

  // 1. our input, minus our own chunk, into our own pool
#pragma unroll
  for (int q = 0; q < N; q++) {
    if (q == rank) continue;
    const long c0 = (long)q * csz, end = min(hi, n16 - c0);
    uint4* dst = mine + base + c0;
    const uint4* src = in + c0;
    long i = lo + tid;
    for (; i + 3 * nt < end; i += 4 * nt) {
      const uint4 v0 = src[i], v1 = src[i + nt], v2 = src[i + 2 * nt], v3 = src[i + 3 * nt];
      dst[i] = v0; dst[i + nt] = v1; dst[i + 2 * nt] = v2; dst[i + 3 * nt] = v3;
    }
    for (; i < end; i += nt) dst[i] = src[i];
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

  // 2. reduce our chunk from the N pools, two words a lane per pass
  constexpr int NE = 16 / sizeof(T);
  {
    const long c0 = (long)rank * csz, end = min(hi, n16 - c0);
    for (long i = lo + tid; i < end; i += 2 * nt) {
      const bool two = i + nt < end;
      uint4 v[N][2];
#pragma unroll
      for (int q = 0; q < N; q++) {
        const uint4* src = q == rank ? in + c0 : P.scratch[q] + base + c0;
        v[q][0] = src[i];
        v[q][1] = two ? src[i + nt] : v[q][0];
      }
#pragma unroll
      for (int k = 0; k < 2; k++) {
        if (k == 1 && !two) break;
        float acc[NE];
#pragma unroll
        for (int e = 0; e < NE; e++) acc[e] = 0.f;
#pragma unroll
        for (int q = 0; q < N; q++) {
          const T* pc = (const T*)&v[q][k];
#pragma unroll
          for (int e = 0; e < NE; e++) acc[e] += ar_ld(pc[e]);
        }
        uint4 r;
        T* pr = (T*)&r;
#pragma unroll
        for (int e = 0; e < NE; e++) ar_st(pr[e], acc[e]);
        out[c0 + i + k * nt] = r;
        mine[red + c0 + i + k * nt] = r;
      }
    }
  }
  __syncthreads();
  if (drain < 4) ar_release(drain);
  publish((u32)(2 * s + 2));
  if (tid < N && tid != rank) ar_wait(&P.flags[rank][tid * R9K_AR_MAX_BLOCKS + b], 2 * s + 2, acq);
  __syncthreads();
  ar_acquire(acq);

  // 3. pull the peers' reduced chunks
#pragma unroll
  for (int q = 0; q < N; q++) {
    if (q == rank) continue;
    const long c0 = (long)q * csz, end = min(hi, n16 - c0);
    const uint4* src = P.scratch[q] + red + c0;
    uint4* dst = out + c0;
    long i = lo + tid;
    for (; i + 3 * nt < end; i += 4 * nt) {
      const uint4 v0 = src[i], v1 = src[i + nt], v2 = src[i + 2 * nt], v3 = src[i + 3 * nt];
      dst[i] = v0; dst[i + nt] = v1; dst[i + 2 * nt] = v2; dst[i + 3 * nt] = v3;
    }
    for (; i < end; i += nt) dst[i] = src[i];
  }
}

// host = 0: the generic two-shot (scratch 2*(nranks+1)*slot16 words); 1: the host-memory variant (2*2*slot16 words).
static int ar_twoshot_launch(int host, int mode, const long* scratch_ptrs, const long* flag_ptrs, int nranks,
                             int rank, long seq, long slot16, long in, long out, long numel, int dtype,
                             long stream, int nblocks, int nthreads, int drain, int acq, int grid) {
  if (mode < 0 || mode > 2 || (host && mode)) return -8;
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
  if (host) hipLaunchKernelGGL((r9k_ar_twoshot_host_nrank_kernel<T, N>), dim3(grid), dim3(nthreads), 0, st, P, \
                               rank, (int*)seq, slot16, (const uint4*)in, (uint4*)out, n16, drain, acq, nblocks); \
  else hipLaunchKernelGGL((r9k_ar_twoshot_nrank_kernel<T, N>), dim3(grid), dim3(nthreads), 0, st, P, rank,      \
                          (int*)seq, slot16, (const uint4*)in, (uint4*)out, n16, drain, acq, nblocks, mode)
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

// Same arguments as r9k_ar_oneshot_nrank; scratch must be 2*(nranks+1)*slot16 words and the flag values are
// NOT compatible with the one-shot kernel's, so the two need separate seq counters AND separate flag arrays.
extern "C" int r9k_ar_twoshot_nrank_g(const long* scratch_ptrs, const long* flag_ptrs, int nranks, int rank,
                                    long seq, long slot16, long in, long out, long numel, int dtype,
                                    long stream, int nblocks, int nthreads, int drain, int acq, int grid) {
  return ar_twoshot_launch(0, 0, scratch_ptrs, flag_ptrs, nranks, rank, seq, slot16, in, out, numel, dtype, stream,
                           nblocks, nthreads, drain, acq, grid);
}

// The same with a mode: 0 all-reduce, 1 reduce-scatter only, 2 all-gather only (see the kernel).
extern "C" int r9k_ar_twoshot_nrank_m(const long* scratch_ptrs, const long* flag_ptrs, int nranks, int rank,
                                    long seq, long slot16, long in, long out, long numel, int dtype,
                                    long stream, int nblocks, int nthreads, int drain, int acq, int grid, int mode) {
  return ar_twoshot_launch(0, mode, scratch_ptrs, flag_ptrs, nranks, rank, seq, slot16, in, out, numel, dtype,
                           stream, nblocks, nthreads, drain, acq, grid);
}

// ---------------------------------------------------------------- fused split all-reduce for 8 ranks on 4 switches
// comm/r9k_ar8.py's split path (pair reduce-scatter -> each quartet's host-memory two-shot on its half -> pair
// all-gather) as one kernel, so the switch-local pair traffic (GPU <-> switch links) overlaps the cross traffic
// (the switch's uplink to the host) instead of waiting for it. Run as three kernels, the stages' link time adds up
// (2 MB: 128 + 295 + 128 us, tests/test_ar8.py PARTS=3, 2026-11-02).
//
// Rank t: pair p = t / 2 (partner t ^ 1), pos = t % 2 = the half it owns (pair two-shot chunking, 2 chunks); its
// quartet = the four ranks with the same pos, in which it has index p and owns part p of the half (4-rank two-shot
// chunking of the half). Each part is cut into nl lanes x K sub-steps; five ROLE blocks per lane walk the sub-steps:
//   R0 push     sub-step j of our input's partner half (4 parts) into the partner's pair pool RS area; flag PP
//   R1 pair-sum wait PP from the partner; pair sum of our half (even + odd, fp32, one rounding): our part -> out,
//               the other three parts -> our host pool `in` area; flag HW to all four quartet members (us included)
//   R2 reduce   wait HW from the four; our part = sum of the four pair sums in quartet order (fp32, one rounding)
//               -> out, our host pool `red` area, the partner's pair pool AG area; flags HR (quartet), PA2 (partner)
//   R3 gather   wait HR from the three peers; their reduced parts -> out and the partner's AG area; flag PA3
//   R4 receive  wait PA2 + PA3 from the partner; its half from our AG area -> out
// Dependencies only point to lower roles, which have lower block ids and are dispatched first, and R0 waits for
// nothing: no reliance on all blocks being resident. Every element sees the split path's two roundings in the same
// order, so the bits equal the three-kernel path (and the whole-message path).
// Counters: seq[blockIdx] (5 * ARF_LM ints) counts calls; flag values c*64 + j + 1, compared by signed difference
// (they wrap after ~67M calls). Flag words (rows of 256 per source, R9K_AR_MAX_BLOCKS): pair rows PP 64+L, PA2 96+L,
// PA3 128+L; host rows HW 64+L, HR 96+L -- the two-shot kernels keep to [0, 64).
// Pools: pair = the pair two-shot pool with 4 extra slots: [6 s, 10 s) (s = pslot words), parity c & 1 -> base 6 s +
// 2 s * parity, RS area base + [0, s), AG area base + s + [0, s), indexed by global word; host = the quartet's host
// two-shot pool (2 x [in | red] of hslot words), indexed by the word offset in the half. Reuse is safe in every
// order (this kernel, the pair two-shot modes, the host two-shot): an area is rewritten only after a wait the
// readers can satisfy only once their previous call ended -- the pair areas are disjoint from the two-shot's
// slots because a partner may still be in R4 when we start the next call.
#define ARF_LM 32            // lanes (flag rows PP / PA2 / PA3 and HW / HR have 32 slots each)
#define ARF_TS 130           // trace words a block: start, then (after wait, after work) per sub-step

struct ArfGeo {     // one half's geometry
  long h0, hn, cq, per, st;   // half start (global word), length, part length, lane length, sub-step length
};

__device__ __forceinline__ ArfGeo arf_geo(long n16, int half, int nl, int K) {
  ArfGeo g;
  const long c2 = (n16 + 1) / 2;
  g.h0 = min((long)half * c2, n16);
  g.hn = min(g.h0 + c2, n16) - g.h0;
  g.cq = (g.hn + 3) / 4;
  g.per = (g.cq + nl - 1) / nl;
  g.st = (g.per + K - 1) / K;
  return g;
}

// [a, e): sub-step j of lane L of part k, as word offsets in the half
__device__ __forceinline__ void arf_range(const ArfGeo& g, int k, int L, int j, long& a, long& e) {
  const long p0 = (long)k * g.cq, pe = min(p0 + g.cq, g.hn);
  const long l0 = p0 + (long)L * g.per, le = min(l0 + g.per, pe);
  a = l0 + (long)j * g.st;
  e = min(a + g.st, le);
}

__device__ __forceinline__ void arf_put(u32* f, u32 v) {
  __hip_atomic_store(f, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void arf_wait(u32* f, u32 v) {
  while ((int)(__hip_atomic_load(f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) - v) < 0)
    __builtin_amdgcn_s_sleep(1);
}

__device__ __forceinline__ void arf_copy(uint4* __restrict__ dst, const uint4* __restrict__ src, long a, long e,
                                         int tid, int nt) {
  long i = a + tid;
  for (; i + 3 * nt < e; i += 4 * nt) {
    const uint4 v0 = src[i], v1 = src[i + nt], v2 = src[i + 2 * nt], v3 = src[i + 3 * nt];
    dst[i] = v0; dst[i + nt] = v1; dst[i + 2 * nt] = v2; dst[i + 3 * nt] = v3;
  }
  for (; i < e; i += nt) dst[i] = src[i];
}

template <typename T>
__device__ __forceinline__ uint4 arf_round(const float* acc) {
  constexpr int NE = 16 / sizeof(T);
  uint4 r;
  T* pr = (T*)&r;
#pragma unroll
  for (int x = 0; x < NE; x++) ar_st(pr[x], acc[x]);
  return r;
}

template <typename T>
__device__ __forceinline__ void arf_add(float* acc, const uint4& w) {
  constexpr int NE = 16 / sizeof(T);
  const T* pw = (const T*)&w;
#pragma unroll
  for (int x = 0; x < NE; x++) acc[x] += ar_ld(pw[x]);
}

template <typename T>
__global__ void __launch_bounds__(1024) r9k_ar8_fused_kernel(
    R9kArPeers PP_, R9kArPeers HQ, int rank, int* __restrict__ seq, long pslot, long hslot,
    const uint4* __restrict__ in, uint4* __restrict__ out, long n16, int nl, int K, unsigned long long* tr) {
  const int role = blockIdx.x / ARF_LM, L = blockIdx.x % ARF_LM, tid = threadIdx.x, nt = blockDim.x;
  __shared__ int s_seq;
  if (tid == 0) s_seq = seq[blockIdx.x]++;
  __syncthreads();
  if (L >= nl) return;                  // idle lane: keeps its counter in step
  // stage trace (r9k_ar8_fused_t, tests/test_ar8.py PARTS=5): wall clock at the start, after each sub-step's wait and
  // after its work, per role block
#define ARF_T(i) do { if (tr && tid == 0) tr[(long)blockIdx.x * ARF_TS + (i)] = wall_clock64(); } while (0)
  ARF_T(0);
  const u32 c64 = (u32)s_seq * 64u;
  const int pos = rank & 1, p = rank >> 1, mate = pos ^ 1;
  const ArfGeo gm = arf_geo(n16, pos, nl, K), go = arf_geo(n16, mate, nl, K);
  const long pb = 6 * pslot + (long)(s_seq & 1) * 2 * pslot;      // pair pool: RS area pb, AG area pb + pslot
  const long hb = (long)(s_seq & 1) * 2 * hslot;                   // host pool: in hb, red hb + hslot
  constexpr int NE = 16 / sizeof(T);
  constexpr int MB = R9K_AR_MAX_BLOCKS;

  for (int j = 0; j < K; j++) {
    const u32 v = c64 + (u32)j + 1u;
    long a, e;
    if (role == 0) {                    // R0: our input's partner half -> partner RS area
      ARF_T(1 + 2 * j);
      uint4* dst = PP_.scratch[mate] + pb + go.h0;
      for (int k = 0; k < 4; k++) {
        arf_range(go, k, L, j, a, e);
        arf_copy(dst, in + go.h0, a, e, tid, nt);
      }
      __syncthreads();
      ARF_T(2 + 2 * j);
      if (tid == 0) arf_put(&PP_.flags[mate][pos * MB + 64 + L], v);
    } else if (role == 1) {             // R1: pair sum of our half
      if (tid == 0) arf_wait(&PP_.flags[pos][mate * MB + 64 + L], v);
      __syncthreads();
      ARF_T(1 + 2 * j);
      const uint4* rcv = PP_.scratch[pos] + pb + gm.h0;
      const uint4* own = in + gm.h0;
      for (int k = 0; k < 4; k++) {
        arf_range(gm, k, L, j, a, e);
        uint4* dst = k == p ? out + gm.h0 : HQ.scratch[p] + hb;
        for (long i = a + tid; i < e; i += nt) {
          const uint4 x0 = own[i], x1 = rcv[i];
          float acc[NE];
#pragma unroll
          for (int x = 0; x < NE; x++) acc[x] = 0.f;
          arf_add<T>(acc, pos == 0 ? x0 : x1);           // rank order: even, then odd
          arf_add<T>(acc, pos == 0 ? x1 : x0);
          dst[i] = arf_round<T>(acc);
        }
      }
      __syncthreads();
      ARF_T(2 + 2 * j);
      if (tid < 4) arf_put(&HQ.flags[tid][p * MB + 64 + L], v);
    } else if (role == 2) {             // R2: our part = the four pair sums in quartet order
      if (tid < 4) arf_wait(&HQ.flags[p][tid * MB + 64 + L], v);
      __syncthreads();
      ARF_T(1 + 2 * j);
      arf_range(gm, p, L, j, a, e);
      uint4* red = HQ.scratch[p] + hb + hslot;
      uint4* ag = PP_.scratch[mate] + pb + pslot + gm.h0;
      for (long i = a + tid; i < e; i += 2 * nt) {
        const bool two = i + nt < e;
        uint4 w[4][2];
#pragma unroll
        for (int q = 0; q < 4; q++) {
          const uint4* src = q == p ? out + gm.h0 : HQ.scratch[q] + hb;
          w[q][0] = src[i];
          w[q][1] = two ? src[i + nt] : w[q][0];
        }
#pragma unroll
        for (int u = 0; u < 2; u++) {
          if (u == 1 && !two) break;
          float acc[NE];
#pragma unroll
          for (int x = 0; x < NE; x++) acc[x] = 0.f;
#pragma unroll
          for (int q = 0; q < 4; q++) arf_add<T>(acc, w[q][u]);
          const uint4 r = arf_round<T>(acc);
          const long ii = i + u * nt;
          out[gm.h0 + ii] = r;
          red[ii] = r;
          ag[ii] = r;
        }
      }
      __syncthreads();
      ARF_T(2 + 2 * j);
      if (tid < 4 && tid != p) arf_put(&HQ.flags[tid][p * MB + 96 + L], v);
      if (tid == 4) arf_put(&PP_.flags[mate][pos * MB + 96 + L], v);
    } else if (role == 3) {             // R3: the peers' reduced parts of our half
      if (tid < 4 && tid != p) arf_wait(&HQ.flags[p][tid * MB + 96 + L], v);
      __syncthreads();
      ARF_T(1 + 2 * j);
      uint4* ag = PP_.scratch[mate] + pb + pslot + gm.h0;
      for (int q = 0; q < 4; q++) {
        if (q == p) continue;
        arf_range(gm, q, L, j, a, e);
        const uint4* src = HQ.scratch[q] + hb + hslot;
        long i = a + tid;
        for (; i + nt < e; i += 2 * nt) {
          const uint4 v0 = src[i], v1 = src[i + nt];
          out[gm.h0 + i] = v0; ag[i] = v0;
          out[gm.h0 + i + nt] = v1; ag[i + nt] = v1;
        }
        for (; i < e; i += nt) { const uint4 v0 = src[i]; out[gm.h0 + i] = v0; ag[i] = v0; }
      }
      __syncthreads();
      ARF_T(2 + 2 * j);
      if (tid == 0) arf_put(&PP_.flags[mate][pos * MB + 128 + L], v);
    } else {                            // R4: the partner's half from our AG area
      if (tid == 0) arf_wait(&PP_.flags[pos][mate * MB + 96 + L], v);
      if (tid == 1) arf_wait(&PP_.flags[pos][mate * MB + 128 + L], v);
      __syncthreads();
      ARF_T(1 + 2 * j);
      const uint4* src = PP_.scratch[pos] + pb + pslot + go.h0;
      for (int k = 0; k < 4; k++) {
        arf_range(go, k, L, j, a, e);
        arf_copy(out + go.h0, src, a, e, tid, nt);
      }
      if (tr) { __syncthreads(); ARF_T(2 + 2 * j); }
    }
  }
}

#undef ARF_T

extern "C" int r9k_ar8_fused_lanes() { return ARF_LM; }
extern "C" int r9k_ar8_fused_trace_words() { return 5 * ARF_LM * ARF_TS; }
extern "C" int r9k_wallclock_khz() {
  int dev = 0, khz = 0;
  if (hipGetDevice(&dev) != hipSuccess || hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev) != hipSuccess)
    return 0;
  return khz;
}

// pair_scratch/pair_flags: the pair's 2 pointers (pair two-shot pool with >= 10 * pslot words); host_scratch /
// host_flags: the quartet's 4 (host two-shot pools, >= 4 * hslot words); rank = TP rank 0..7 (pair index * 2 + pos);
// seq: 5 * r9k_ar8_fused_lanes() ints; nl <= ARF_LM lanes, K <= 64 sub-steps, nthreads 0 = by size. Grid 5 * ARF_LM.
// r9k_ar8_fused_t: the same with a stage trace (trace: r9k_ar8_fused_trace_words() uint64, wall clock ticks; 0 = off).
extern "C" int r9k_ar8_fused_t(const long* pair_scratch, const long* pair_flags, const long* host_scratch,
                               const long* host_flags, int rank, long seq, long pslot, long hslot, long in, long out,
                               long numel, int dtype, long stream, int nl, int K, int nthreads, long trace) {
  const int esz = dtype == 2 ? 4 : 2;
  const long nbytes = numel * esz;
  if (nbytes % 16) return -1;
  const long n16 = nbytes / 16;
  if (rank < 0 || rank >= 8) return -5;
  if (n16 <= 0) return 0;
  const long c2 = (n16 + 1) / 2;
  if (n16 > pslot || c2 > hslot) return -7;
  if (nl <= 0 || nl > ARF_LM || K <= 0 || K > 64) return -2;
  const long cq = (c2 + 3) / 4;
  if ((long)nl > cq) nl = (int)cq;
  const long per = (cq + nl - 1) / nl;
  if ((long)K > per) K = (int)per;
  if (nthreads <= 0) {
    const long w = (per + K - 1) / K;
    long t = ((w + AR_WAVE - 1) / AR_WAVE) * AR_WAVE;
    nthreads = (int)(t < 64 ? 64 : (t > 1024 ? 1024 : t));
  }
  if (nthreads > 1024 || nthreads % AR_WAVE || nthreads < 8) return -3;
  R9kArPeers P{}, H{};
  for (int q = 0; q < 2; q++) { P.scratch[q] = (uint4*)pair_scratch[q]; P.flags[q] = (u32*)pair_flags[q]; }
  for (int q = 0; q < 4; q++) { H.scratch[q] = (uint4*)host_scratch[q]; H.flags[q] = (u32*)host_flags[q]; }
  hipStream_t st = (hipStream_t)stream;
#define ARF_LAUNCH(T)                                                                                      \
  hipLaunchKernelGGL((r9k_ar8_fused_kernel<T>), dim3(5 * ARF_LM), dim3(nthreads), 0, st, P, H, rank,      \
                     (int*)seq, pslot, hslot, (const uint4*)in, (uint4*)out, n16, nl, K, (unsigned long long*)trace)
  switch (dtype) {
    case 0: ARF_LAUNCH(__hip_bfloat16); break;
    case 1: ARF_LAUNCH(_Float16); break;
    case 2: ARF_LAUNCH(float); break;
    default: return -4;
  }
#undef ARF_LAUNCH
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

extern "C" int r9k_ar8_fused(const long* pair_scratch, const long* pair_flags, const long* host_scratch,
                             const long* host_flags, int rank, long seq, long pslot, long hslot, long in, long out,
                             long numel, int dtype, long stream, int nl, int K, int nthreads) {
  return r9k_ar8_fused_t(pair_scratch, pair_flags, host_scratch, host_flags, rank, seq, pslot, hslot, in, out, numel,
                         dtype, stream, nl, K, nthreads, 0);
}

// The host-memory two-shot: same arguments; scratch 2*2*slot16 words in pinned host memory; own seq and flag arrays.
extern "C" int r9k_ar_twoshot_host_nrank_g(const long* scratch_ptrs, const long* flag_ptrs, int nranks, int rank,
                                         long seq, long slot16, long in, long out, long numel, int dtype,
                                         long stream, int nblocks, int nthreads, int drain, int acq, int grid) {
  return ar_twoshot_launch(1, 0, scratch_ptrs, flag_ptrs, nranks, rank, seq, slot16, in, out, numel, dtype, stream,
                           nblocks, nthreads, drain, acq, grid);
}

// grid = the data blocks (see r9k_ar_oneshot_2rank).
extern "C" int r9k_ar_twoshot_nrank(const long* scratch_ptrs, const long* flag_ptrs, int nranks, int rank,
                                    long seq, long slot16, long in, long out, long numel, int dtype,
                                    long stream, int nblocks, int nthreads, int drain, int acq) {
  return r9k_ar_twoshot_nrank_g(scratch_ptrs, flag_ptrs, nranks, rank, seq, slot16, in, out, numel, dtype, stream,
                                nblocks, nthreads, drain, acq, 0);
}

// ---------------------------------------------------------------- one-kernel 8-rank all-reduce for small messages
// (Package C, 2026-11-03; comm/r9k_ar8.py R9kAllReduce8._oneshot / all_reduce_moe.) Below ~64 KB the split path and the
// whole-message path cost three kernels a call (8 KB 21 us, 32 KB 47, 60 KB 37 under HIP graphs): each stage pays a
// launch and a drain, while the data itself moves in a few microseconds. Here the same three stages are three PHASES of
// one kernel, connected by device-scope release / acquire flags between ranks exactly as the stages were:
//   A  pair reduce-scatter (P2P, switch-local): push the partner's half of our input into its pair pool; wait for its
//      push; pair sum of OUR half (even + odd in rank order, fp32, one rounding) -> out (staging) and our host pool
//   B  quartet exchange of the halves (evens: first half, odds: second): wait for the three quartet peers' pair sums;
//      our half's total = the four pair sums in quartet order (fp32, one rounding) -> out and the partner's AG area
//   C  pair all-gather: wait for the partner's total of its half -> out
// Every element sees the two roundings of the three-kernel composition in the same order, so the bits equal it (and
// the whole-message path). Transport of phase B (xp): 0 = pinned host pools (each rank writes its own pool, the peers
// pull: 0.5 S up, 1.5 S down a rank, nothing crosses a root complex), 1 = peer writes into the peers' VRAM pools
// (lower latency, but eight cross-chain senders: only for the smallest messages, see the fabric cliff in r9k_ar8.py).
//
// Blocks: nd data blocks of a fixed grid of AOS_MB; block b owns a slice of each half and talks only to block b of its
// partner / quartet peers (flag word per (source, block)); blocks of one rank never wait for each other, so no reliance
// on co-residency inside a rank -- across ranks every rank launches the kernel in the same graph order (the existing
// stage kernels assume the same). All AOS_MB counters advance every call (fixed grid), so all blocks agree on the
// buffer parity. Flag values 4s+1 (phase A push / host write), 4s+2 (phase B AG push), compared by signed difference.
// Reuse: a rank rewrites an area of parity s&1 in call s+2 only after its call s+1 waited (block 0 is always a data
// block) for a value every reader of that area publishes in ITS call s+1, i.e. after the reader's call-s kernel ended.
// Pools (words of 16 bytes, slot = the largest message): pair [2][RS slot | AG slot] (VRAM IPC, the pair's two ranks),
// host [2][slot] (pinned host, the quartet's four), xp [2][4][slot] (VRAM IPC, the quartet's four; only for xp = 1);
// flags: one u32 per (source, block) in AOS_FR-word rows, in the pool's own flag array.
//
// MOE = 1 (r9k_ar8_os_moe): the message is formed on the fly from the r9k expert tier's per-slot f32 outputs instead of
// being read from a bf16 buffer: x[t] = bf16( bf16(sum_k w[t,k] * os[t*topk+k]) + shared[t] ) over the local picks k in
// pick order (fp32, multiply then add, no contraction) -- exactly k_dn's per-token sum (kernels/r9k_moe_exl3.hip, RNE)
// followed by stock's `shared_output + fused_output` (bf16 add), so the all-reduced bits equal the unfused path's.
//
// Spin timeout (tmo ticks > 0, R9K_AR_SPIN_TIMEOUT_MS): a wait that exceeds it writes a record to the host-mapped diag
// words {magic, rank, block, phase, want, have, call} and traps after ~0.3 s (time for the Python watchdog to print it).
#define AOS_MB 16
#define AOS_FR 64
#define AOS_SMAX 512
#define AOS_MAGIC 0x52394b54u

struct AosPools {
  uint4* pair[2]; u32* pairf[2];    // by pair position (0 = even)
  uint4* host[4]; u32* hostf[4];    // by quartet index (rank / 2)
  uint4* xp[4];   u32* xpf[4];
};

struct AosMoe {
  const float* os;          // [S, H] per-slot f32 outputs (rows of non-local slots are not read)
  const float* w;           // [S] routing weights
  const int* ids;           // [S] global expert ids
  const long* g2l;          // [G] global -> local (outside [0, E) = not on this rank)
  const uint16_t* sh;       // [T, H] bf16 shared-expert output
  int G, E, topk, H, S;
};

__device__ __forceinline__ u32 aos_ld(u32* f) { return __hip_atomic_load(f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM); }
__device__ __forceinline__ void aos_put(u32* f, u32 v) {
  __hip_atomic_store(f, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __noinline__ void aos_timeout(u32* diag, int rank, int b, int phase, u32 want, u32 have, int s) {
  if (diag) {
    diag[1] = (u32)rank; diag[2] = (u32)b; diag[3] = (u32)phase; diag[4] = want; diag[5] = have; diag[6] = (u32)s;
    __hip_atomic_store(&diag[0], AOS_MAGIC, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  const u64 t0 = wall_clock64();
  while (wall_clock64() - t0 < 30000000ull) __builtin_amdgcn_s_sleep(127);    // ~0.3 s at 100 MHz
  __builtin_trap();
}

__device__ __forceinline__ void aos_wait(u32* f, u32 v, long tmo, u32* diag, int rank, int b, int phase, int s) {
  if (tmo <= 0) {
    while ((int)(aos_ld(f) - v) < 0) __builtin_amdgcn_s_sleep(1);
    return;
  }
  const u64 t0 = wall_clock64();
  u32 h;
  while ((int)((h = aos_ld(f)) - v) < 0) {
    __builtin_amdgcn_s_sleep(1);
    if ((long)(wall_clock64() - t0) > tmo) aos_timeout(diag, rank, b, phase, v, h, s);
  }
}

// RNE float -> bf16 bits, NaN quieted: the same function as kernels/r9k_moe_exl3.hip f_bf16
__device__ __forceinline__ u32 aos_bf16(float f) {
  u32 u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (u >> 16) | 0x40u;
  u += 0x7fffu + ((u >> 16) & 1u);
  return u >> 16;
}

// word i (8 bf16 elements) of the MoE message: k_dn's local weighted sum in pick order, rounded, plus the shared output
__device__ __forceinline__ uint4 aos_moe_word(const AosMoe& M, const float* lw, long i) {
  const int wpr = M.H >> 3;
  const int t = (int)(i / wpr), c = (int)(i - (long)t * wpr) * 8;
  float acc[8];
#pragma unroll
  for (int e = 0; e < 8; e++) acc[e] = 0.f;
  for (int k = 0; k < M.topk; k++) {
    const int s = t * M.topk + k;
    const float wk = lw[s];
    if (__float_as_uint(wk) == 0xffffffffu) continue;                // not a local slot (marker NaN)
    const float4* o = (const float4*)(M.os + (size_t)s * M.H + c);
    const float4 o0 = o[0], o1 = o[1];
    acc[0] += wk * o0.x; acc[1] += wk * o0.y; acc[2] += wk * o0.z; acc[3] += wk * o0.w;
    acc[4] += wk * o1.x; acc[5] += wk * o1.y; acc[6] += wk * o1.z; acc[7] += wk * o1.w;
  }
  const uint4 sw = *(const uint4*)(M.sh + (size_t)t * M.H + c);
  const u32 sv[4] = {sw.x, sw.y, sw.z, sw.w};
  u32 r[4];
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const float s0 = __uint_as_float(sv[e] << 16), s1 = __uint_as_float(sv[e] & 0xffff0000u);
    const float f0 = __uint_as_float(aos_bf16(acc[2 * e]) << 16), f1 = __uint_as_float(aos_bf16(acc[2 * e + 1]) << 16);
    r[e] = aos_bf16(s0 + f0) | (aos_bf16(s1 + f1) << 16);         // stock: shared_output + fused_output
  }
  return make_uint4(r[0], r[1], r[2], r[3]);
}

template <typename T, int MOE>
__global__ void __launch_bounds__(1024) r9k_ar8_os_kernel(
    AosPools P, int rank, int* __restrict__ seq, long slot, const uint4* __restrict__ in, uint4* __restrict__ out,
    long n16, int nd, int xp, long tmo, u32* diag, AosMoe M) {
  const int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ int s_seq;
  __shared__ float lw[MOE ? AOS_SMAX : 1];
  if (tid == 0) s_seq = seq[b]++;
  if (MOE && b < nd)
    for (int s = tid; s < M.S; s += nt) {
      const int id = M.ids[s];
      const long e = (id >= 0 && id < M.G) ? M.g2l[id] : -1;
      lw[s] = (e >= 0 && e < M.E) ? M.w[s] : __uint_as_float(0xffffffffu);
    }
  __syncthreads();
  if (b >= nd) return;                  // idle block: keeps its counter in step
  const int s = s_seq;
  const u32 v1 = 4u * (u32)s + 1u, v2 = 4u * (u32)s + 2u;
  const int pos = rank & 1, p = rank >> 1, mate = pos ^ 1;
  const long c2 = (n16 + 1) / 2;
  const long hm0 = pos ? c2 : 0, hmn = pos ? n16 - c2 : c2;           // our half
  const long ho0 = pos ? 0 : c2, hon = pos ? c2 : n16 - c2;           // the partner's half
  const long per = (c2 + nd - 1) / nd;
  const long lo = (long)b * per;
  const long hiM = min(lo + per, hmn), hiO = min(lo + per, hon);
  const long pb = (long)(s & 1) * 2 * slot;                            // pair pool: RS pb, AG pb + slot
  const long hb = (long)(s & 1) * slot;                                // host pool
  const long xb = (long)(s & 1) * 4 * slot;                            // xp pool: [4][slot]
  constexpr int NE = 16 / sizeof(T);
  auto load = [&](long i) -> uint4 {
    if constexpr (MOE) return aos_moe_word(M, lw, i);
    else return in[i];
  };

  // A. the partner's half of our input -> its RS area
  {
    uint4* dst = P.pair[mate] + pb + ho0;
    for (long i = lo + tid; i < hiO; i += nt) dst[i] = load(ho0 + i);
  }
  __syncthreads();
  if (tid == 0) aos_put(&P.pairf[mate][pos * AOS_FR + b], v1);
  if (tid == 0) aos_wait(&P.pairf[pos][mate * AOS_FR + b], v1, tmo, diag, rank, b, 1, s);
  __syncthreads();
  {
    const uint4* rcv = P.pair[pos] + pb + hm0;
    uint4* hdst = xp ? nullptr : P.host[p] + hb;
    for (long i = lo + tid; i < hiM; i += nt) {
      const uint4 x0 = load(hm0 + i), x1 = rcv[i];
      float acc[NE];
#pragma unroll
      for (int e = 0; e < NE; e++) acc[e] = 0.f;
      arf_add<T>(acc, pos == 0 ? x0 : x1);                             // rank order: even, then odd
      arf_add<T>(acc, pos == 0 ? x1 : x0);
      const uint4 r = arf_round<T>(acc);
      out[hm0 + i] = r;
      if (xp) {
#pragma unroll
        for (int q = 0; q < 4; q++)
          if (q != p) P.xp[q][xb + (long)p * slot + i] = r;
      } else {
        hdst[i] = r;
      }
    }
  }
  __syncthreads();
  if (tid < 4 && tid != p) aos_put(xp ? &P.xpf[tid][p * AOS_FR + b] : &P.hostf[tid][p * AOS_FR + b], v1);

  // B. our half's total from the four pair sums, in quartet order
  if (tid < 4 && tid != p)
    aos_wait(xp ? &P.xpf[p][tid * AOS_FR + b] : &P.hostf[p][tid * AOS_FR + b], v1, tmo, diag, rank, b, 2, s);
  __syncthreads();
  {
    uint4* ag = P.pair[mate] + pb + slot + hm0;
    const uint4* src[4];
#pragma unroll
    for (int q = 0; q < 4; q++)
      src[q] = q == p ? out + hm0 : (xp ? P.xp[p] + xb + (long)q * slot : P.host[q] + hb);
    for (long i = lo + tid; i < hiM; i += 2 * nt) {
      const bool two = i + nt < hiM;
      uint4 w[4][2];
#pragma unroll
      for (int q = 0; q < 4; q++) {
        w[q][0] = src[q][i];
        w[q][1] = two ? src[q][i + nt] : w[q][0];
      }
#pragma unroll
      for (int u = 0; u < 2; u++) {
        if (u == 1 && !two) break;
        float acc[NE];
#pragma unroll
        for (int e = 0; e < NE; e++) acc[e] = 0.f;
#pragma unroll
        for (int q = 0; q < 4; q++) arf_add<T>(acc, w[q][u]);
        const uint4 r = arf_round<T>(acc);
        out[hm0 + i + u * nt] = r;
        ag[i + u * nt] = r;
      }
    }
  }
  __syncthreads();
  if (tid == 0) aos_put(&P.pairf[mate][pos * AOS_FR + b], v2);

  // C. the partner's total of its half
  if (tid == 0) aos_wait(&P.pairf[pos][mate * AOS_FR + b], v2, tmo, diag, rank, b, 3, s);
  __syncthreads();
  {
    const uint4* src = P.pair[pos] + pb + slot + ho0;
    for (long i = lo + tid; i < hiO; i += nt) out[ho0 + i] = src[i];
  }
}

extern "C" int r9k_ar8_os_max_blocks() { return AOS_MB; }
extern "C" int r9k_ar8_os_flag_row() { return AOS_FR; }
extern "C" int r9k_ar8_os_max_slots() { return AOS_SMAX; }

static long aos_ticks(int tmo_ms) {
  if (tmo_ms <= 0) return 0;
  static int khz = 0;
  if (!khz) {
    int dev = 0, v = 0;
    if (hipGetDevice(&dev) == hipSuccess && hipDeviceGetAttribute(&v, hipDeviceAttributeWallClockRate, dev) == hipSuccess
        && v > 0) khz = v;
    else khz = 100000;
  }
  return (long)tmo_ms * khz;
}

static int aos_launch(int moe, const long* pair_s, const long* pair_f, const long* host_s, const long* host_f,
                      const long* xp_s, const long* xp_f, int rank, long seq, long slot, long in, long out, long n16,
                      int dtype, long stream, int nd, int nthreads, int xp, int tmo_ms, long diag, const AosMoe& M) {
  if (rank < 0 || rank >= 8) return -5;
  if (n16 <= 0) return 0;
  if (n16 > slot) return -7;
  if (xp && !xp_s) return -9;
  if (nd <= 0 || nd > AOS_MB) return -2;
  const long c2 = (n16 + 1) / 2;
  if ((long)nd > c2) nd = (int)c2;
  const long per = (c2 + nd - 1) / nd;
  if (nthreads <= 0) {
    long t = ((per + AR_WAVE - 1) / AR_WAVE) * AR_WAVE;
    nthreads = (int)(t < 64 ? 64 : (t > 1024 ? 1024 : t));
  }
  if (nthreads > 1024 || nthreads % AR_WAVE || nthreads < 64) return -3;
  AosPools P{};
  for (int q = 0; q < 2; q++) { P.pair[q] = (uint4*)pair_s[q]; P.pairf[q] = (u32*)pair_f[q]; }
  for (int q = 0; q < 4; q++) { P.host[q] = (uint4*)host_s[q]; P.hostf[q] = (u32*)host_f[q]; }
  if (xp_s)
    for (int q = 0; q < 4; q++) { P.xp[q] = (uint4*)xp_s[q]; P.xpf[q] = (u32*)xp_f[q]; }
  const long tmo = aos_ticks(tmo_ms);
  hipStream_t st = (hipStream_t)stream;
#define AOS_LAUNCH(T, MO)                                                                                   \
  hipLaunchKernelGGL((r9k_ar8_os_kernel<T, MO>), dim3(AOS_MB), dim3(nthreads), 0, st, P, rank, (int*)seq, slot, \
                     (const uint4*)in, (uint4*)out, n16, nd, xp, tmo, (u32*)diag, M)
  if (moe) {
    AOS_LAUNCH(__hip_bfloat16, 1);
  } else {
    switch (dtype) {
      case 0: AOS_LAUNCH(__hip_bfloat16, 0); break;
      case 1: AOS_LAUNCH(_Float16, 0); break;
      case 2: AOS_LAUNCH(float, 0); break;
      default: return -4;
    }
  }
#undef AOS_LAUNCH
  return hipGetLastError() == hipSuccess ? 0 : -6;
}

// pair_s/pair_f: the pair's 2 pool pointers by position (>= 4 * slot words); host_s/host_f: the quartet's 4 host pools
// (>= 2 * slot words); xp_s/xp_f: the quartet's 4 VRAM pools (>= 8 * slot words) or 0 when xp = 0. seq: AOS_MB ints,
// shared by the plain and the MoE entry (they are one protocol). rank: TP rank 0..7. nthreads 0 = by size.
extern "C" int r9k_ar8_os(const long* pair_s, const long* pair_f, const long* host_s, const long* host_f,
                          const long* xp_s, const long* xp_f, int rank, long seq, long slot, long in, long out,
                          long numel, int dtype, long stream, int nd, int nthreads, int xp, int tmo_ms, long diag) {
  const int esz = dtype == 2 ? 4 : 2;
  const long nbytes = numel * esz;
  if (nbytes % 16) return -1;
  AosMoe M{};
  return aos_launch(0, pair_s, pair_f, host_s, host_f, xp_s, xp_f, rank, seq, slot, in, out, nbytes / 16, dtype, stream,
                    nd, nthreads, xp, tmo_ms, diag, M);
}

// The fused MoE reduction: out [T, H] bf16 = all-reduce over the eight ranks of
//   bf16( bf16(sum over local picks k of w[t,k] * os[t*topk+k]) + shared[t] )
// os f32 [T*topk, H], w f32 [T*topk], ids int32 [T*topk], g2l int64 [G], shared bf16 [T, H] (rows contiguous).
extern "C" int r9k_ar8_os_moe(const long* pair_s, const long* pair_f, const long* host_s, const long* host_f,
                              const long* xp_s, const long* xp_f, int rank, long seq, long slot, long os, long w,
                              long ids, long g2l, int G, int E, int T, int topk, int H, long shared, long out,
                              long stream, int nd, int nthreads, int xp, int tmo_ms, long diag) {
  if (H % 8 || T <= 0 || topk <= 0 || T * topk > AOS_SMAX) return -1;
  AosMoe M{(const float*)os, (const float*)w, (const int*)ids, (const long*)g2l, (const uint16_t*)shared, G, E, topk, H,
           T * topk};
  return aos_launch(1, pair_s, pair_f, host_s, host_f, xp_s, xp_f, rank, seq, slot, 0, out, (long)T * H / 8, 0, stream,
                    nd, nthreads, xp, tmo_ms, diag, M);
}

// ---------------------------------------------------------------- one-kernel 8-rank all-gather (package C, 2026-11-03)
// comm/r9k_ar8.py R9kAllReduce8._ag_oneshot. The flat 8-rank one-shot all-gather (r9k_ag_oneshot_nrank_kernel<8>) pushes
// every rank's piece into all seven peers' VRAM: eight senders across the four root complexes, the fabric cliff --
// 0.65-1.07 ms for Engram's [T, 3200] and 3.2-3.6 ms for the logits [<= 4, 16160] (notes/exl3-ledger.md 1b). Here no
// store crosses a root complex: each rank
//   1  writes its piece to out, to its own pinned host pool and to its partner's pair pool (P2P, switch-local); flags
//      its quartet (host)
//   2  waits for its three quartet peers, pulls their pieces from host memory -> out and on to the partner's pair pool;
//      flags the partner
//   3  waits for the partner; copies the partner quartet's four pieces from its own pair pool -> out
// Up a rank 1 piece to the host, down 3, 4 to the partner. Pure copies: the result is the input bits.
// Layout: in [rows, cols] contiguous (cols16 words a row), out [rows, 8 * cols16] with rank q's piece at column block q
// (vLLM's concat along `dim`). Pools: host [2][slot], pair [2][4][slot] (slot = the largest piece, words); flags one u32 per
// (source, block), value 2s+1. Reuse as r9k_ar8_os: an area of parity s&1 is rewritten in call s+2 only after call s+1
// waited for every reader's flag of s+1 (block 0 always carries data).
__global__ void __launch_bounds__(1024) r9k_ag8_os_kernel(
    AosPools P, int rank, int* __restrict__ seq, long slot, const uint4* __restrict__ in, uint4* __restrict__ out,
    long n16, long cols16, int nd, long tmo, u32* diag) {
  const int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ int s_seq;
  if (tid == 0) s_seq = seq[b]++;
  __syncthreads();
  if (b >= nd) return;
  const int s = s_seq;
  const u32 v = 2u * (u32)s + 1u;
  const int pos = rank & 1, p = rank >> 1, mate = pos ^ 1;
  const long per = (n16 + nd - 1) / nd;
  const long lo = (long)b * per, hi = min(lo + per, n16);
  const long hb = (long)(s & 1) * slot, pb = (long)(s & 1) * 4 * slot;
  auto oidx = [&](int q, long i) -> long { const long r = i / cols16; return r * 8 * cols16 + (long)q * cols16 + (i - r * cols16); };

  // 1. own piece -> out, own host pool, partner's pair pool (slot p)
  {
    uint4* h = P.host[p] + hb;
    uint4* pp = P.pair[mate] + pb + (long)p * slot;
    for (long i = lo + tid; i < hi; i += nt) {
      const uint4 w = in[i];
      out[oidx(rank, i)] = w;
      h[i] = w;
      pp[i] = w;
    }
  }
  __syncthreads();
  if (tid < 4 && tid != p) aos_put(&P.hostf[tid][p * AOS_FR + b], v);
  if (tid < 4 && tid != p) aos_wait(&P.hostf[p][tid * AOS_FR + b], v, tmo, diag, rank, b, 4, s);
  __syncthreads();
  // 2. the quartet peers' pieces from host memory -> out and the partner's pair pool
  {
    uint4* pp = P.pair[mate] + pb;
    for (long i = lo + tid; i < hi; i += nt) {
      uint4 w[4];
#pragma unroll
      for (int q = 0; q < 4; q++)
        if (q != p) w[q] = P.host[q][hb + i];
#pragma unroll
      for (int q = 0; q < 4; q++) {
        if (q == p) continue;
        out[oidx(2 * q + pos, i)] = w[q];
        pp[(long)q * slot + i] = w[q];
      }
    }
  }
  __syncthreads();
  if (tid == 0) aos_put(&P.pairf[mate][pos * AOS_FR + b], v);
  if (tid == 0) aos_wait(&P.pairf[pos][mate * AOS_FR + b], v, tmo, diag, rank, b, 5, s);
  __syncthreads();
  // 3. the partner quartet's four pieces
  {
    const uint4* src = P.pair[pos] + pb;
    for (long i = lo + tid; i < hi; i += nt) {
      uint4 w[4];
#pragma unroll
      for (int q = 0; q < 4; q++) w[q] = src[(long)q * slot + i];
#pragma unroll
      for (int q = 0; q < 4; q++) out[oidx(2 * q + mate, i)] = w[q];
    }
  }
}

// pair_s/pair_f: the pair's 2 pools by position (>= 8 * slot words); host_s/host_f: the quartet's 4 host pools (>= 2 *
// slot words); seq: AOS_MB ints of its own. rows x cols elements of esz bytes (cols * esz % 16 == 0, rows * cols * esz
// <= slot * 16). nthreads 0 = by size.
extern "C" int r9k_ag8_os(const long* pair_s, const long* pair_f, const long* host_s, const long* host_f, int rank,
                          long seq, long slot, long in, long out, long rows, long cols, int esz, long stream, int nd,
                          int nthreads, int tmo_ms, long diag) {
  const long rowbytes = cols * esz;
  if (rowbytes % 16 || rows <= 0) return -1;
  if (rank < 0 || rank >= 8) return -5;
  const long cols16 = rowbytes / 16, n16 = rows * cols16;
  if (n16 > slot) return -7;
  if (nd <= 0 || nd > AOS_MB) return -2;
  if ((long)nd > n16) nd = (int)n16;
  const long per = (n16 + nd - 1) / nd;
  if (nthreads <= 0) {
    long t = ((per + AR_WAVE - 1) / AR_WAVE) * AR_WAVE;
    nthreads = (int)(t < 64 ? 64 : (t > 1024 ? 1024 : t));
  }
  if (nthreads > 1024 || nthreads % AR_WAVE || nthreads < 64) return -3;
  AosPools P{};
  for (int q = 0; q < 2; q++) { P.pair[q] = (uint4*)pair_s[q]; P.pairf[q] = (u32*)pair_f[q]; }
  for (int q = 0; q < 4; q++) { P.host[q] = (uint4*)host_s[q]; P.hostf[q] = (u32*)host_f[q]; }
  hipLaunchKernelGGL(r9k_ag8_os_kernel, dim3(AOS_MB), dim3(nthreads), 0, (hipStream_t)stream, P, rank, (int*)seq, slot,
                     (const uint4*)in, (uint4*)out, n16, cols16, nd, aos_ticks(tmo_ms), (u32*)diag);
  return hipGetLastError() == hipSuccess ? 0 : -6;
}
