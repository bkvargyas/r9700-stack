# EXL3 kernels on gfx1201 (package K, 2026-11-02)

exllamav3 v1.6.6 built for gfx1201 inside the serving pin, tested on one R9700 (VM100 card 7). Image
`r9700/vllm:exl3` = `r9700/vllm:dev` + `~/ds41/exl3/dist/exllamav3-1.6.6-cp312-cp312-linux_x86_64.whl`
(`pip install --no-deps`; Dockerfile `~/ds41/exl3/Dockerfile.exl3`, context `~/ds41/exl3/dist`). No exllamav3 source
patch, no kernel change. `import exllamav3_ext` and `import exllamav3` (the python package, for its Hadamard helper)
both work without extra deps. Scripts and raw outputs: VM100 `~/ds41/exl3/kerneltest/` (`*.py` -> `*.out`), run as
`~/ds41/exl3/gpu-run.sh 7 python3 /ds41/exl3/kerneltest/<script>.py`.

Short version for L and M:

- gfx1201 runs **native WMMA**, not the emulation; numerics are clean everywhere (tables below).
- `exl3_gemm(A, trellis, C, suh, A_had, svh, -1, mcg, mul1, 0)`: **force_num_sms = 0, not -1** (-1 disables the
  GEMV path and autotune and sets num_sms = -1).
- **Never use the autotuned WMMA trellis GEMM (9+ rows) on this GPU**: 1.6x slower than the best alternative at
  16 rows, 10-18x at 512-4096 rows.
  Rows <= 8: `exl3_gemm` (RDNA multi-row GEMV). 9-32 rows: `exl3_gemm` per 8-row chunk. Above ~24-48 rows:
  reconstruct + hgemm. exllamav3's AUTO_RECONSTRUCT_THRESHOLD = 144 is wrong for gfx1201; use **32**
  (24 for the <= 8 MB matrices, 48 for the 26-98 MB ones if you want the last few percent).
- hipBLAS f16 GEMM falls off a cliff on some prefill shapes (5120x2304 at 4096 rows: 24 TFLOPS); **split M into
  1024-row chunks** (84-92 TFLOPS).
- Decode MoE: `exl3_moe` costs 365 us per call even with one expert hit; the fastest graph-safe decode path is
  **`exl3_mgemm` with indices + expert range** (28 us + ~24 us per local pick at M = 1), which needs
  `C[:num_tokens].zero_()` first (it leaves C untouched when no pick is local).
- Everything captures in a HIP graph (gemm, recon, exl3_moe fused tier, coop, mgemm) **after one eager call per
  row count** (the gemm autotunes with host syncs on the first call at a new M and kills a capture).

## a. Kernel path on gfx1201

- Build (`~/ds41/exl3/build-full.log`): 152 compile units, every one `--offload-arch=gfx1201`, no wmma warning (the
  931 warnings are nodiscard / strict-aliasing / unused-variable). `util/arch_list.py` sets PYTORCH_ROCM_ARCH from the
  devices (wave32 required); `util/cuda_flags.py` adds `-O3 -fgpu-flush-denormals-to-zero --offload-compress`
  and drops the PTX-only units (`hgemm_f16acc.cu`, `exl3_gemv.cu` -> `rocm/quant/exl3_gemv_rdna.cu`, the M-tiled MoE
  instances).
- Selection is compile-time per device pass. `rocm/rdna_wmma_emu.cuh` defines `EXL3_WMMA_EMULATED` only when the pass
  is neither `__GFX11__` nor `__GFX12__` (RDNA1/2), so gfx1201 is native. The one wrapper the GEMM and MoE use,
  `rdna_wmma::mma_sync` (f16 x f16 -> f32), has an explicit `__GFX12__` arm:
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` on the per-lane 8-half operand split, converting the gfx11
  fragment layout in and out with four `__shfl_xor(.., 16)` swaps each way ("validated on gfx1201 against an fp64
  reference"; not optimal: a native gfx12 fragment layout would drop the swaps). The other wrappers (bf16, f16-acc,
  int8) keep the bare gfx11 builtins on purpose so a gfx12 instantiation fails to compile; the build succeeded, so
  none is instantiated for gfx12. `rocm/det_gemm_rocm.cuh` (int8 deterministic GEMM) has its own `__GFX12__` arms.
  `rocm/wmma_gemm.cuh` (vendored rocm_wmma_gemm backend for hgemm) has tables for gfx1151 / gfx1100 only: on gfx1201
  its kernels are empty stubs and **hgemm goes to hipBLAS** (`cublasGemmEx`, f32 compute).
- Host side: `DevCtx::get_cc` returns CC_OLD on every RDNA part (the host heuristics take the conservative class).
- ISA check (`isa_scan.sh` -> `isa_scan.out`): unbundling the 144 code objects of the installed `.so` (all
  `hipv4-amdgcn-amd-amdhsa--gfx1201`) and disassembling: 5991 `v_wmma_f32_16x16x16_f16`, 144 `v_wmma_i32_16x16x16_iu8`;
  kernels containing WMMA: `exl3_gemm_kernel` 216 of 216 instances, `exl3_mgemm_kernel` 216, `exl3_moe_kernel` 78,
  plus the int8 routing kernels. The GEMV paths (the ones decode uses) are `v_dot2_f32_f16` (153k instructions), no
  WMMA.
- Device facts: `g_get_num_sms` 32 (WGPs), `g_get_cc` 1 (CC_OLD), `g_get_smem_max` 65536, `exl3_moe_max_concurrency`
  8 (32 WGPs / group width 4; 32 with `EXL3_ROCM_MOE_GROUP=2`).

## Call conventions (what LinearEXL3 / block_sparse_mlp actually pass)

Dense (`modules/quant/exl3.py`, `libtorch/linear.cpp BC_LinearEXL3::run_gr`):

```
xh = torch.empty_like(x)                       # A_had: scratch, same shape as x (f16); bsz 1 reuses a cached (1, in)
ext.exl3_gemm(x_f16_2d, trellis, y, suh, xh, svh, -1, mcg, mul1, 0)   # y f16 or f32 (dtype picks c_fp32)
```
Returns a tag: 92 = RDNA multi-row GEMV (rows 1-8), 90/91 = GEMV, else the autotuned GEMM shape index (1-4 seen).
`mcg` / `mul1` are bools (the pack: `mul1` present on every tensor, `mcg` never). K comes from `trellis.shape[-1] / 16`.

Large M, `reconstruct_hgemm` (rows > 144 in exllamav3; see the threshold below):
```
# rows < 1024: rotated-basis weight + explicit Hadamards
ext.had_r_128(x, xh, suh, None, 1.0)
ext.reconstruct(w_f16[in, out], trellis, K, mcg, mul1)          # or reconstruct_slice(w, trellis, K, mcg, mul1, n0)
ext.hgemm_recon(xh, w, y)                                        # hipBLAS on gfx1201; torch.matmul is the same speed
ext.had_r_128(y, y, None, svh, 1.0)
# rows >= 1024 (exllamav3's rule): original-basis weight, no activation Hadamards
ext.reconstruct_had_slice(w, trellis, suh, svh[n0:], K, mcg, mul1, n0); ext.hgemm_recon(x, w, y[:, n0:n1])
```
Output slices of <= 32768 columns (MAX_RECONSTRUCT_SLICE_N); the head (129280 = 1010 x 128) takes four.
The exact f32 weight for references: `ext.reconstruct` (f16, exact decode) -> f32 -> 128-block Hadamard on the input
side, `* suh[:, None]`, 128-block Hadamard on the output side, `* svh[None, :]` (`kerneltest/common.py weight_f32`).
On the real tensors suh is not a pure sign vector: it carries an input scale (values around +-0.03), |svh| 0.6-1.8.

MoE fused tier (`modules/block_sparse_mlp.py` 1201-1376, as it runs on ROCm; `kerneltest/moe_common.py moe_exl3`):

- ROCm defaults (`util/backend.py`): MOE_FUSED_ROWS 512, batched-reconstruct tier off, m-tiles off, FUSED_DET on.
- Sentinel bucket: picks outside this rank's experts map to local id E (= 48); `expert_count` has E + 1 bins (long),
  `order = flat_local.argsort()`, `token_sorted = flat_token[order]` (long), `weight_sorted = flat_weight[order]` (f16),
  `inv_order` = inverse permutation.
- FusedBuffers: `temp_state_g/u` f16 `(C, R, H)`, `temp_intermediate_g/u` f16 `(C, R, I)` with C =
  `exl3_moe_max_concurrency(dev)` (8), R = fused rows (512): 2 x 42 MB + 2 x 19 MB a layer-shared set.
- `num_tokens <= 512`: **no host sync**. Tables on device: `expert_start = cumsum(count) - count`,
  `tables = stack([expert_start, expert_start, count > 0])`, `scratch` f32 `(num_tokens * 6, H)`,
  `exl3_moe(..., num_active = -1, scratch, tables[0], count_lo = 1, count_hi = 512, m_tile = 16)`.
- `num_tokens > 512`: **one host sync** (`expert_count.cpu()`), host-built tables (fused experts: kind 1, base =
  exclusive cumsum of their counts), `num_active` = number of experts with 1..512 rows; experts above 512 rows go
  through a per-expert path (exllamav3: `BC_BlockSparseMLP.run_single_expert_dq`; the test uses recon_hgemm for
  w1/w3, `ext.silu_mul`, recon_hgemm for w2, `* weight`, `index_add_`). At 4096 tokens with 6-of-384 routing the
  busiest local expert had 84 rows (all fused); all-local routing put 21 experts above 512 (max 553).
- Then always `exl3_moe_gather(final, scratch, flat_local, inv_order, tables[1, :E], tables[0, :E], tables[2, :E],
  weight_sorted)`. `final` is f32 `(num_tokens, H)` zeroed; `hidden` must be f16.
- exl3_moe argument list: `(hidden f16, out f32, expert_count, token_sorted, weight_sorted, ts_g, ts_u, ti_g, ti_u,
  act 0 (= silu), K_gate, K_up, K_down (float), g_tr, g_suh, g_svh, u_tr, u_suh, u_svh, d_tr, d_suh, d_svh,
  g_mcg, g_mul1, u_mcg, u_mul1, d_mcg, d_mul1, act_limit 10.0, num_active, scratch, tables[0], count_lo, count_hi,
  m_tile)`; pointer tables are int64 device tensors of `data_ptr()`s per local expert (w1 = gate, w3 = up, w2 = down),
  one table per (matrix, tensor), separate allocations per expert are fine, repeats are fine.
- act_limit semantics (`quant/hadamard_inner.cuh had_hf_r_128_guad_inner`, also the coop kernel and `ext.silu_mul`):
  `min(silu(g), L) * clamp(u, -L, L)`, in f16. vLLM's `SiluAndMulWithClamp` is `silu(min(g, L)) * clamp(u, -L, L)`;
  they differ only for g > L, by at most L - silu(L) = 4.5e-4 (negligible).
- `exl3_moe_coop` (exllamav3's own bsz <= 8 decode path, `BC_BlockSparseMLP.run_bszN`): takes the global
  `sel (bsz, 6)` long + `rw (bsz, 6)` f16 directly with `[min_expert, max_expert) = [0, 48)` masked in-kernel; scratch
  as in block_sparse_mlp.py 575-605 (`moe_common.Experts.c_*`); writes the routed sum to `out[:bsz]`.
- `exl3_mgemm` MoE (`moe_common.moe_mgemm`): gate / up as `exl3_mgemm(A, g_tr, g, g_suh, ah, g_svh, idx(1, M*6),
  None, K, -1, False, True, 0, 48, 0, M)`, `ext.silu_mul(g, u, a, 10.0)`, down with `weights (1, M*6)` f16 into
  `C` f32 `(M*6, 1, H)` (C must hold a row per slot: it is per-slot scratch; rows 0..M-1 get the reduced sums).
  Out-of-range picks are dropped (M = 1: compacted; M > 1: masked to -1 **in place in idx**, pass a copy).
  **No local pick -> C is not written** (`mgemm_zero.out`: a NaN prefill survives): zero `C[:M]` first. With >= 1 pick
  it overwrites.

## b. Dense numerics gate (`dense_gate.py` -> `dense_gate.out`; real tensors `real_gate.py` -> `real_gate.out`)

Synthetic tensors: random int16 trellis (every bit stream is a valid trellis encoding, so the whole decode path is
exercised), suh = random signs, svh = random signs x U(0.5, 1.5) / sqrt(in) so outputs are O(1); mul1; x ~ N(0, 1)
f16. 25 shapes: the 20 of the brief plus the K=6 / K=4 variants (layer 39 and DSpark: 5120->1280 K6, 5120->512 K6,
5120->2304 K6, 2304->5120 K4/K5/K6, 1280->32768 K6), M in {1, 2, 4, 8, 16, 64, 512, 4096}, 600 rows. Reference
x.float() @ W_f32. f16 floor = the same metric for torch's f16 matmul of x @ W.half() (what any f16 GEMM on the
dequantized weight scores): rms rel 2.9e-4 on every shape.

| path | C | max abs err (all shapes, M) | max abs / max abs ref | rms rel err | vs f16 floor |
|---|---|---|---|---|---|
| exl3_gemm (GEMV rows 1-8, autotuned GEMM above) | f32 | 1.98e-3 | 2.8e-4 | 2.3e-4 | 0.78x |
| exl3_gemm | f16 | 5.5e-3 | 8.7e-4 | 6.6e-4 | 2.3x (output rounding) |
| reconstruct + hgemm + had (rows 16-512) | f32 | 1.76e-3 | 2.6e-4 | 2.1e-4 | 0.72x |
| same | f16 | 5.0e-3 | 8.7e-4 | 4.2e-4 | 1.4x |
| reconstruct_had_slice + hgemm (rows 4096) | f32 | 7.8e-3 | 1.0e-3 | 9.1e-4 | 3.1x |
| same | f16 | 8.7e-3 | 1.19e-3 | 9.3e-4 | 3.2x |

Uniform across shapes, K (2-8) and M; no NaN/Inf; f16 C never near overflow at O(1) activations. The fused
reconstruct (`reconstruct_had_slice`) is 3x the floor: its in-kernel Hadamards run through f16; still 1e-3, below
bf16 activation rounding (3.9e-3). C in f32 vs f16 matters only by the output rounding (2-3x rms, both far below
bf16); take f32 where the next op wants f32, f16 otherwise (same speed).

Real tensors (layer 0 experts K=3 and layer 18 experts K=2, w1/w3/w2; copied out of the running download's partial
shard files by `extract_partial.py`, attention tensors were not on disk yet): exl3_gemm f32 rms rel 2.9-3.1e-4 vs
floor 2.9e-4, f16 4.6-5.3e-4, max abs / max ref <= 9.6e-4; recon same as gemm; fused reconstruct 9.3-9.6e-4.
max |y| at x ~ N(0,1): 3.7-9.4.

## c. MoE numerics gate (`moe_gate.py` -> `moe_gate.out`, `mgemm_moe.out`, real: `real_gate.out`)

48 local experts (rank 0 of EP8: global 0-47), H 5120, I 2304, K = 3 and 2, silu, act_limit 10. Routing 6-of-384
random (most tokens hit 0-2 local experts; M=1 drew 0 local picks) or all-local (6-of-48). x ~ N(0,1) (limit never
reached) and x ~ N(0, 64) (29% of gate/up values clipped). f32 reference per expert: `min(silu(x W1), 10) *
clamp(x W3, +-10) @ W2 * w`, summed.

| path | M | K | max abs / max abs ref | rms rel err |
|---|---|---|---|---|
| exl3_moe fused tier + gather (device-only, M <= 512) | 1-512 | 3, 2 | 6.0e-4 - 1.05e-3 | 7.1e-4 - 8.4e-4 |
| exl3_moe host-sync tiers (M = 4096; 21 experts > 512 rows on the per-expert path in the all-local case) | 4096 | 3, 2 | 7.0e-4 - 8.8e-4 | 6.9e-4 - 8.3e-4 |
| exl3_moe_coop | 1-8 | 3, 2 | 6.1e-4 - 1.0e-3 | 7.3e-4 - 8.3e-4 |
| exl3_mgemm (indices + range + weighted reduce) | 1-8 | 3, 2 | 6.8e-4 - 1.0e-3 | 7.7e-4 - 9.4e-4 |
| real experts, all three paths | 1-4096 | 3 (L0), 2 (L18) | 2.4e-4 - 9.4e-4 | 6.3e-4 - 8.8e-4 |

Zero local picks: exl3_moe and coop return exact zeros; mgemm needs the pre-zero (above). exl3_moe is bitwise
deterministic across repeats (FUSED_DET slot + gather). The ~8e-4 is the f16 intermediates (g, u, act in f16).

Host synchronization in exllamav3's tiering: none at num_tokens <= 512 (counts stay on device, num_active = -1 ->
launch at max concurrency, slot tables built on device). Above 512 tokens one `expert_count.cpu()` per layer plans
num_active / count_lo / count_hi (on ROCm a single fused launch `count_lo 1, count_hi 512, m_tile 16`; the CUDA build's
32/64-row tiles and the batched reconstruct tier + its `exl3_moe_gather` slots do not apply to ROCm) and the
over-512 experts run one at a time. `exl3_moe_gather` is used at every M (FUSED_DET).

## d. Timing (R9700, card 7; `timing.py`, `timing2.py`, `cross_graph.py`, `moe_probe.py`, `mgemm_moe.py`)

Method: torch.cuda.Event, warmup, median of 20-30. "graph" = 10-20 calls captured in one HIP graph, replay / count:
the GPU-only cost vLLM decode sees. Eager single-call numbers include ~8-20 us of host launch overhead per call at
decode sizes (timing.out). Caveat: repeated calls on one matrix < 64 MB hit the 64 MB Infinity Cache; GB/s above
~640 (the GDDR6 peak) are cache-assisted and real decode (weights cold) will be closer to 550-600.

Dense decode, graph, us per exl3_gemm (GB/s of trellis bytes), C f16 (f32 is the same within 1 us):

| in->out K | MB | M=1 | M=2 | M=4 | M=8 | M=16 (WMMA gemm) | M=32 |
|---|---|---|---|---|---|---|---|
| 5120->2304 K3 (expert w1/w3) | 4.4 | 16.2 (272) | 16.6 | 19.4 | 26.2 | 77.2 | 211 |
| 5120->2304 K2 | 2.9 | 14.8 (200) | 16.0 | 19.1 | 26.2 | 76.7 | 212 |
| 2304->5120 K3 (expert w2) | 4.4 | 15.0 (296) | 16.7 | 20.1 | 25.3 | 73.3 | 205 |
| 2304->5120 K2 | 2.9 | 14.7 (201) | 16.2 | 19.9 | 25.2 | 69.6 | 199 |
| 5120->1280 K5 (wq_a) | 4.1 | 13.3 (307) | 14.6 | 16.3 | 20.4 | 54.9 | 124 |
| 1280->32768 K5 (wq_b) | 26.2 | 35.2 (745) | 38.6 | 45.8 | 64.8 | 234 | 463 |
| 1280->4096 K5 (wq_b TP8, indexer wq_b) | 3.3 | 11.6 (283) | 12.5 | 14.0 | 17.8 | 40.5 | 76 |
| 5120->512 K5 (wkv, compressor) | 1.6 | 9.4 (175) | 10.0 | 11.3 | 14.0 | 37.0 | 69 |
| 4096->1024 K5 (wo_a slice) | 2.6 | 10.6 (248) | 11.4 | 12.8 | 16.3 | 40.0 | 76 |
| 8192->5120 K5 (wo_b) | 26.2 | 43.2 (606) | 44.8 | 50.8 | 67.6 | 234 | 463 |
| 1024->5120 K5 (wo_b TP8 rows) | 3.3 | 12.1 (272) | 12.5 | 14.0 | 18.0 | 39.8 | 94 |
| 512->128 K8 (indexer wk) | 0.07 | 7.9 | 8.3 | 9.2 | 10.9 | 13.9 | 22 |
| 6144->25600 K5 (engram wkv) | 98.3 | 165 (596) | 171 | 187 | 236 | 947 | 1877 |
| 6144->3200 K5 (engram wkv TP8 cols) | 12.3 | 25.5 (481) | 27.2 | 30.6 | 37.7 | 198 | 392 |
| 5120->2304 K5 / K4 (shared w1/w3) | 7.4 / 5.9 | 18.2 / 16.3 | 19.7 / 17.2 | 22.1 / 19.7 | 27.7 / 26.5 | 77 / 75 | 146 / 141 |
| 2304->5120 K5 / K4 (shared w2) | 7.4 / 5.9 | 17.4 / 23.2 | 18.8 / 15.9 | 22.1 / 18.5 | 26.9 / 25.4 | 75 / 67 | 139 / 127 |
| 5120->129280 K6 (head) | 496 | 806 (616) | 809 | 820 | 1014 | 4748 | 9478 |
| 15360->5120 K6 (mtp main_proj) | 59.0 | 75.4 (782) | 86.8 | 98.7 | 127 | 426 | 857 |

Floor ~8 us a call; small matrices (1.6-4.4 MB) reach 175-300 GB/s, 26 MB+ ones 600. Rows 1-8 cost 1.0-1.6x of
one row. At 9+ rows the autotuned WMMA GEMM takes over and is 1.4-2.6x slower than 8-row GEMV chunks.

AUTO_RECONSTRUCT_THRESHOLD crossover, graph, us (`cross_graph.out`; eager version with 768-4096 rows in
`timing.out`): gemm = autotuned WMMA trellis GEMM, gemv x8 = exl3_gemm per 8-row chunk, recon = had + reconstruct +
hgemm + had, fused = reconstruct_had_slice + hgemm.

| in->out K | M | gemm | gemv x8 | recon | fused |
|---|---|---|---|---|---|
| 5120->2304 K3 | 16 / 24 / 32 / 64 / 256 | 83 / 161 / 165 / 333 / 1124 | **53** / 79 / 104 / 207 / 822 | 63 / **63** / **63** / **64** / **112** | 98 / 98 / 98 / 98 / 152 |
| 5120->1280 K5 | 16 / 24 / 32 / 64 / 256 | 60 / 120 / 122 / 250 / 851 | **42** / 62 / 82 / 163 / 644 | 52 / **52** / **52** / **52** / **65** | 70 / 71 / 70 / 70 / 82 |
| 4096->1024 K5 | 16 / 24 / 32 / 64 / 256 | 41 / 75 / 76 / 146 / 567 | **33** / 49 / 64 / 129 / 503 | 42 / **42** / **42** / 69 / 78 | 50 / 51 / 51 / **54** / **60** |
| 5120->512 K5 | 16 / 24 / 32 / 64 / 256 | 37 / 69 / 71 / 131 / 496 | **28** / **41** / 55 / 109 / 430 | 44 / 44 / **44** / **44** / **48** | 47 / 47 / 47 / 47 / 48 |
| 8192->5120 K5 | 16 / 24 / 32 / 48 / 256 | 234 / 458 / 462 / 698 / 3660 | **129** / **196** / **275** / 402 / 2274 | 314 / 315 / 343 / **320** / **414** | 403 / 405 / 406 / 417 / 537 |
| 1280->32768 K5 | 16 / 24 / 32 / 48 / 256 | 235 / 459 / 466 / 702 / 3683 | **125** / **186** / **261** / 395 / 2159 | 307 / 307 / 309 / **326** / **466** | 402 / 402 / 404 / 434 / 525 |
| 6144->25600 K5 | 16 / 24 / 32 / 48 / 256 | 949 / 1875 / 1882 / 2814 / 14942 | **467** / **714** / **962** / 1432 / 7665 | 1236 / 1239 / 1243 / **1249** / **1607** | 1600 / 1604 / 1609 / 1618 / 1967 |

Rule for gfx1201: rows <= 8 exl3_gemm; 9-32 rows exl3_gemm in 8-row chunks (or recon for <= 8 MB matrices from 24
rows); above 32 rows recon (non-fused) up to ~1024-2048 rows, fused above that (and from ~384 rows for wide-N
1280->32768, where fused wins because `had_r_128` on the 32768-wide output is slow: 17.6 ms at 4096 rows). The
autotuned WMMA gemm never wins above 8 rows (at 512 rows: 2.3 ms vs 0.21 ms for 5120->2304).

Prefill, eager, us (timing.out; gemm f16 / recon f16):

| in->out K | 512 rows | 4096 rows |
|---|---|---|
| 5120->2304 K3 | 2331 / 213 | 18606 / 4061 |
| 2304->5120 K3 | 2135 / 235 | 16862 / 950 |
| 5120->1280 K5 | 1921 / 161 | 14719 / 1671 |
| 1280->32768 K5 | 7366 / 846 | 58872 / 4313 |
| 8192->5120 K5 | 7328 / 553 | 58807 / 4109 |
| 6144->25600 K5 | 29818 / 2316 | 238940 / 15193 |
| 5120->129280 K6 | 151474 / 10613 | 1212299 / 64692 |
| 15360->5120 K6 | 13614 / 1115 | 108701 / 7970 |

hipBLAS cliff (`cross_graph.out`, graph, us (TFLOPS)): hgemm at 4096 rows in one call vs 1024-row / 512-row chunks:
5120x2304 4027 (24) / **1152 (84)** / 1166; 5120x1280 1738 (31) / **612 (88)** / 692; 6144x3200 10876 (15) /
**1750 (92)** / 2157; 2304x5120 **1033 (94)** / 1252 / 1297; 8192x5120 3838 = 3836; 6144x25600 14196 = 14132.
torch.matmul shows the same cliff (timing2.out). 1024-row chunks never lose more than 20% and win up to 6x.
Components at 512 rows (timing2.out): reconstruct 25-800 us (k*n-proportional, 3-5 ps per weight),
reconstruct_had_slice 1.5-2x reconstruct, had_r_128 16-40 us.

MoE, 48 experts, graph, us per layer call (`moe_probe.out`, `mgemm_moe.out`; MB = trellis of the experts hit, 13.3 MB
an expert at K=3):

| M, routing, local experts hit | exl3_moe (-1) | exl3_moe (num_active = hit) | coop | mgemm | 3x exl3_gemm per expert (needs host-known picks) |
|---|---|---|---|---|---|
| 1, 6-of-384, 0 picks | 90 | - | 10 | 28 | - |
| 2, 6-of-384, 1 | 365 | 121 | 196 | 443 | 48 |
| 4, 6-of-384, 2 | 366 | 180 | 268 | 901 | 95 |
| 8, 6-of-384, 3 | 365 | 290 | 383 | 1344 | 152 |
| 1, all-local, 6 | 632 | 544 | 683 | **170** | 305 |
| 8, all-local, 35 | 2944 | 2944 | 3810 | 9566 | 1857 |
| 64, all-local, 48 | 4055 | 4057 | - | - | - |
| 512, 6-of-384, 48 | 4050 | 4048 | - | - | - |

The mgemm column of the M = 2 / 4 / 8 6-of-384 rows is from `mgemm_moe.out`, whose routing draw hit 2 / 3 / 3
local picks. K = 2 is within a few % of K = 3 for every MoE path (moe_probe / timing: the MoE kernels are not bandwidth-bound).
`EXL3_ROCM_MOE_PIPE=0` (non-pipelined mainloop) is 1.5-2x slower; `EXL3_ROCM_MOE_GROUP=2` makes num_active = -1 4x
slower, 8 changes nothing. Eager timings (timing.out) add ~330 us of host prologue to exl3_moe (argsort, scatter,
cumsum, stack: ~10 small launches).

Reading: exl3_moe's fused kernel streams ~155 GB/s at best (48 experts, 4.05 ms) and has a ~90 us floor with no
work; with num_active = -1 (the only graph-safe setting, since the hit count is on the device) it costs 365 us for 1-3
experts. coop is 10 us at zero picks but 70-120 GB/s per expert. **mgemm at M = 1 is the decode winner: ~28 us + ~24 us
per local pick (550 GB/s)**, graph-safe, no sort. At M = 2-8 the per-slot mgemm form is slow (masked slots still
cost); one M = 1 mgemm per token (M launches) costs ~M x 28 + 24 x picks, which beats coop and exl3_moe for M <= 8 at
realistic EP routing. Prefill (M >= 64 or so): exl3_moe at 155 GB/s is ~4 ms a layer a rank for all 48 experts;
the reconstruct alternative is ~39 us x 3 per expert (5.6 ms for 48) + hgemm, so exl3_moe stays the prefill path
until a per-expert reconstruct + grouped GEMM is measured (not done here).

HIP graph capture (`graph_test.py` -> `graph_test.out`): torch.cuda.graph of exl3_gemm at M = 1, 4, 8, 16, 64,
recon_hgemm at 512 and fused at 2048, exl3_moe fused path (whole block_sparse_mlp sequence incl. argsort and gather)
at M = 1, 4, 8, 64, 512, exl3_moe_coop and the mgemm MoE at M = 1, 4, 8: all capture; replay with new inputs / new
routing written into the static buffers is bitwise equal to eager on the same inputs and repeatable. **Failure mode:
the first exl3_gemm call at a row count (> 8 rows, the autotuned GEMM) inside a capture fails** ("operation failed
due to a previous error during capture"): CoopKernelAutotuner times candidates with cudaEventSynchronize /
cudaStreamSynchronize. Rows 1-8 (GEMV) never autotune. Fix in our code: warm every captured row count eagerly (vLLM's
pre-capture warmup runs do this for its capture sizes, verify) or avoid the autotuned GEMM entirely (the rule above).
The tune cache is per process plus a disk cache (`EXLLAMAV3_TUNE_CACHE`, default `$XDG_CACHE_HOME` or
`~/.cache/...`: inside the container that is ephemeral unless mounted).

## Unverified / not done

- Real attention / indexer / compressor / engram / head tensors (not on disk yet: the download writes all 43 shards in
  parallel and shard 2's partial file holds only routed experts so far). Synthetic covers their shapes; real experts
  behaved exactly like synthetic.
- Cold-cache decode bandwidth (all numbers re-read one matrix; matrices < 64 MB are cache-assisted).
- A grouped reconstruct + GEMM prefill MoE as an alternative to exl3_moe; the batched-reconstruct tier
  (`moe_batch_recon.py`) is disabled on ROCm by exllamav3 and was not tried.
- Half-integer bitrates (not in this pack).
