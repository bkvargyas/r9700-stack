# ROCm 10.1 for the R9700: what it changes (research, 2026-10-13)

Brian: "can we build ROCm 10.1 in this?" / "set up an agent to look at 10.1 and see if any RDNA4 performance
increases are in it". A web-research pass over the release notes, component changelogs and the ROCm repositories
(not a measurement). Short answer: nothing in 10.1 should move decode or prefill on gfx1201; it is a compatibility
check at most.

## What 10.1 is

Released 2026-10-05 (TheRock `therock-10.1`), stated focus "AI inference, developer tooling and profiling across
AMD Instinct". Headline items: hipFILE async fast path, NUMA-aware host VMM in HIP, the ROCm CLI, rocprofiler
kernel replay (beta), hipThreads, ROCm SMI removed (AMD SMI only), R9600 support. Component versions 10.0 -> 10.1:
HIP 7.15 -> 7.16, Composable Kernel 1.2 -> 1.3, MIOpen 3.6.0 -> 3.6.1, rocBLAS 5.6 -> 5.7, rocprofiler-sdk 1.3.5
-> 1.4.1. **Unchanged: hipBLASLt 1.4.1, RCCL 2.30.7, rocWMMA 2.2.1, ROCr 1.21.0, LLVM 24.0.0.**
Sources: https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html ,
https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-10.1-blog/README.html ,
https://rocm.docs.amd.com/en/latest/release/changelog.html

## RDNA4 / gfx12 items found

- Composable Kernel 1.3.0: "improved FMHA forward performance for head dimension 128 on gfx11 and gfx12" (prefill
  attention through CK only; ours is not) and "fixed out-of-bounds asynchronous loads in the CK Tile compute
  async WMMA GEMM pipeline".
- HIP 7.16: no graph-launch, dispatch-latency, P2P or fine-grained-memory changes (entries: hipInitDevice,
  hipDeviceGetLuid, host-NUMA hipMemCreate, a DMA-buf host-alloc attribute, a stream/signal memory-leak fix).
  ROCr: inter-process sharing of host-pool and device-pool VMM handles.
- RCCL 2.30.7: P2P peer-count / channel fixes, suspend/resume APIs, the rest gfx1250. Not in a release yet:
  the gfx1201 all-reduce LL cutoff change (PR 8666, "+7-10% vLLM TP4 throughput on R9700", merged 2026-07-26;
  whether 10.0 or 10.1 carries it could not be confirmed) and a 2026-10-08 open PR defaulting gfx120x single-node
  to the Simple protocol because "LL can deadlock single-node gfx1200/gfx1201 collectives". Our decode
  all-reduce does not use RCCL below 512 KB.
  https://github.com/ROCm/rocm-systems/pull/8666 , https://github.com/ROCm/rocm-systems/pull/13151
- hipBLASLt unchanged; a gfx1201 tuning PR (+9.9% workload GEMM time) is open against `develop`, not in 10.1.
  https://github.com/ROCm/rocm-libraries/pull/13306
- Resolved on RX 9070 / R9700: GPU resets in PyTorch training with Llama-Factory / Unsloth. Known issues: nothing
  RDNA4-specific.
- Compiler: no documented AMDGPU / gfx12 codegen, WMMA-scheduling or fp8-builtin changes (not found).
- Driver / MES: the amdgpu 31.60.0 notes say nothing about MES or TLB invalidation on RDNA4. The kernel-side
  "rework GPU TLB invalidation" series (2026-09-11, GMC12 GART invalidation to SDMA, PASID to MES-or-SDMA because a
  stuck KIQ/MES "cannot be recovered by a queue reset") is not in mainline as of 2026-10-01; the open R9700
  host-freeze report with the `MES(1) failed to respond to msg=INVALIDATE_TLBS` precursor has no AMD response.
  https://github.com/ROCm/legacy-rocm-build/issues/6688
- Launch overhead: TheRock #8634 reports ~8 us extra per launch on the R9700 (ROCm 10.0) for grids of exactly
  2048 / 4096 / 6144 one-wave workgroups, reproducing in hipGraph replay; no 10.1 status.
  https://github.com/ROCm/TheRock/issues/8634
- PyTorch: `torch 2.14.0+rocm10.1.0` gfx1201 wheels exist; nothing gfx12-specific in the notes. vLLM: 10.1 lists
  vLLM 0.29.0 as supported (AMD's image is 0.29.1.dev0+rocm10.1.0); vLLM's own nightlies stay on ROCm 10.0.

## Risks for our HIP code on 10.1

- The compiler snapshot moved (CK hit "a build failure caused by the compiler change from vector to ext_vector
  types"): code touching HIP vector-type internals may break; `clang_major` gates must accept 24.
- An RCCL source build on 10.1 needs its tree (an AMD SMI ABI guard broke the 10.0-era build); gfx120x
  protocol defaults are in flux. rocm-smi-lib is gone.

## Verdict

ROCm 10.1 is unlikely to change R9700 decode or prefill: the runtime has no dispatch or graph changes, and
hipBLASLt, RCCL, rocWMMA, ROCr and the LLVM major are the 10.0 versions. The one gfx12 gain (CK attention) is not
on our path. Building vLLM from source on it would be a compatibility check with noise-level deltas expected. The
interesting target is what is landing on `develop` for 10.2: the gfx1201 hipBLASLt tuning and the RCCL gfx120x
protocol changes. Recommendation: do not build on 10.1; re-check at 10.2.
