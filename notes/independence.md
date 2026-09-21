# Independence from libr4d

**Goal (Brian, 2026-09-21): remove the dependency on libr4d.** libr4d (codeberg.org/StillDeadcode/libr4d) ships
no license file, so anything we take from it -- binary or source -- is unlicensed. This note records what we
depend on, what has been replaced, and how each replacement was written, so the provenance of our own code is
documented rather than reconstructed later from memory.

## Method for every replacement

Clean implementation from public material only:

- the published algorithm (one-shot all-reduce; paged FlashAttention / online softmax; Hadamard rotation before
  low-bit quantisation),
- the HIP/ROCm API and the gfx1201 ISA,
- Apache-2.0 reference code where it helps (vLLM's `triton_unified_attention`, davetha's LRU kernels),
- **our own** Python host side, which already specifies each interface precisely because we wrote it.

**We do not disassemble, decompile or `objdump` `r4d.so`, and we do not read libr4d source.** This is deliberate.
Transcribing a disassembly would produce a stronger derivative-work claim than the dependency we are trying to
remove; the functionality is public, so the binary is not needed as a specification. Agents given this work are
told the same in their brief (see the attention task, 2026-09-21).

## What we depended on, and where each stands

| piece | kind | measured worth | status |
|---|---|---|---|
| 2-rank P2P all-reduce, exact | runtime, `r4d.so` | decode 118.6 vs 99.3 tok/s; 25.1 vs 30.3 ms/step | **replaced** -- `kernels/r9k_ar.hip`, `R9K_AR_IMPL=r9k` |
| 2-rank all-reduce, compressed (wht6) | runtime, `r4d.so` | prefill 4418 vs 3344 tok/s; conc-8 465.9 vs 390.8 | **open** -- ours is exact-only |
| paged attention, prefill + mixed batches | runtime, `r4d.so` | prefill 4418 vs 2027 tok/s (**-54%** without it) | **replaced** -- `kernels/r9k_attn.hip`, default |
| MXFP4xFP8 GEMM design | **source derivation** | folded unpack priced at ~5% | **open** -- see below |
| expert LRU cache kernels | vendored source | -- | not an issue: davetha, Apache-2.0 |

Ablation that produced those numbers (27B, 9k prefill, 2026-09-21; `~/nor4d.sh` on VM100):

| leg | prefill | single decode | conc-8 | ms/step |
|---|---|---|---|---|
| libr4d AR + attention | 4,418 | 118.6 | 462.8 | 25.0 |
| attention off | 2,027 | 121.6 | 494.4 | 25.0 |
| all-reduce off | 3,344 | 99.3 | 390.8 | 30.3 |
| neither | 1,782 | 97.3 | 413.4 | 30.4 |

## 1. All-reduce -- replaced (exact path)

`kernels/r9k_ar.hip` + `r9700_vllm/comm/r9k_ar.py`, selected with `R9K_AR_IMPL=r9k`.

Written from the standard one-shot pattern: each rank pushes its whole input into the peer's IPC-shared scratch,
fences, publishes a flag, waits for the peer's flag, then reduces the local pair in fp32. Double-buffered through
a device-resident per-block sequence counter so a replayed graph never depends on host state (cudagraph-safe with
only scratch and flags IPC-shared). The interface -- argument order, `drain`/`acq` fence selectors, `slot16`
buffer stride, per-block flags -- comes from our own `r4d_ar.py`, which we wrote as the caller.

Verified by `tests/test_ar_r9k.py` (`torchrun --nproc-per-node=2`): bit-exact against RCCL for bf16/fp16/fp32 from
8 to 5.2M elements, both ranks bit-identical, stable over repeated calls with a varying block count.

Results: **matches libr4d on decode-size messages** (25.4 vs 25.1 ms/step, 116.5 vs 118.9 tok/s single, against
RCCL's 30.3 / 99.3). On large messages it lands at RCCL's level (prefill 3,326 vs RCCL 3,344 vs libr4d 4,418),
because libr4d's advantage there is compression, not transport -- at 2 MB the microbenchmark is 233 us for ours
vs 256 for RCCL, both bandwidth-bound on this PCIe 3 link.

**Open:** a compressed large-message path. The scheme is a Hadamard rotation of each 64-element group shipped as
6 bits plus a bf16 scale (~2.6x fewer bytes). Rotating to flatten outliers before low-bit quantisation is a
well-established public technique (QuIP, QuaRot and others); implement from that, not from libr4d.

## 2. Paged attention -- replaced (2026-09-21)

`kernels/r9k_attn.hip` + `r9700_vllm/kernels/attn.py`, selected by `R9K_PAGED_ATTN` (**`r9k` is the default**;
`r4d` keeps libr4d for A/B, `0` falls back to stock `unified_attention`).

One workgroup = (seq, kv head, 16 query tokens), two waves per q head each owning half the head dim. Each KV
block is staged once per workgroup into LDS (V transposed in hardware by `global_load_tr_b128`, swizzled so a
fragment is one `ds_read_b128`), double-buffered a block ahead. S is computed **transposed** so each lane owns one
query row, which makes the online-softmax max/sum/rescale per-lane scalars and leaves P in the accumulator layout
with no shuffles; rescale is lazy (only when a row max grows past 2^8, decided wave-uniformly by ballot). fp8 KV
converts on stage with the K descale folded into the softmax scale and the V descale into the output normalizer.
Geometry: head_size 256, block 16, GQA 1-8 templated, causal, no window/alibi/softcap (those already fall back).

Correctness `tests/test_attn_r9k.py`: all shapes vs stock `unified_attention`, max rel err 2.3e-3 to 6.9e-3
(bar 2e-2); fp8 KV 2.8e-3 to 3.7e-3. Full 11-gate suite passes with it as the default.

Measured end to end on the 27B (independent A/B, three probes per leg):

| | ours | libr4d | stock |
|---|---|---|---|
| prefill 9k | 4,401 | 4,410 | 2,042 |
| prefill 20.7k | 4,165 | 4,190 | -- |
| ms/step | 25.1 | 25.2 | 25.2 |
| single decode | 120.2 | 120.4 | 111.7 |
| conc-8 | 477.5 | 472.9 | 487.3 |
| GSM8K-500 | 97.0% | 96.6% | -- |

**99.8% of libr4d at 9k, 99.4% at 20.7k**, step time and decode indistinguishable. Flash-Next unaffected.

**Not covered:** there is no decode-band kernel, so short-q groups inside a mixed prefill+decode batch run on the
prefill kernel (correct, but a 16-row tile with no KV split; [8,8,8,8] measures 0.07 ms vs libr4d's 0.05). This
costs nothing measurable end to end (25.1 vs 25.2 ms/step), so it is left undone; a flash-decoding split with a
combine pass is the piece to write if mixed-batch latency ever matters. The kernel is LDS-bandwidth-bound, not
compute or DRAM bound -- ~136 KB of LDS traffic per block-step against 1536 WMMA cycles -- so further speed would
come from shrinking the partner exchange (an fp16 exchange measured ~3%, left off to keep it exact).

Note the "3D" split-KV path for speculative verify is **stock vLLM's** Triton kernel, not ours and not libr4d's --
our contribution there is only the routing trick that forces `unified_attention` down its split-KV branch. So a
libr4d-free build already has working decode attention; what is missing is prefill and mixed batches.

## 3. The MXFP4xFP8 GEMM -- a source derivation, not a runtime dependency

`kernels/r9k_moe_mxfp4a8.hip` carries "Derived from libr4d's r4d_gemm_mxfp4a8_nt_m64". No runtime flag removes
this; it is about the source we adapted. An in-repo audit (2026-09-21, from git history and comments only) found:

- The file is **1,268 lines; 141 (11%) survive from the first commit**, and ~115 lines (~9%, strictly ~59) still
  plausibly trace to libr4d. The prefill kernel family (615 lines), the A-tiled family (247), the rewritten decode
  loop, all four launchers and all three quantizers were written here.
- Of the four elements the header calls inherited, three are **not really libr4d's**: fp8 WMMA and its fragment
  layout are fixed by the ISA; in-block split-K through LDS is textbook (and only the decode kernel uses it); the
  fragment-order weight layout is mostly forced by what the WMMA builtin requires, with libr4d's contribution
  being the specific blob ordering and the `Ws`/`Wref` sidecar shapes.
- The one genuinely libr4d-specific idea is the **folded-exponent MXFP4 scheme**: `kMag`, the 4x`v_perm_b32`
  unpack, the clamped exponent difference, and the `Wref` / `2^(Wref-127)` ABI. `notes/review-fable.md:270`
  prices replacing it with a shift/LUT unpack at **~5%**.

Two things worth knowing:

- **A nearly-independent build exists today.** The fp32 per-group scaling path is ours and is the code default
  (`R9K_FOLD` defaults to `0`); `serve/27b.sh` turns fold on. So the cost of GEMM independence is roughly what
  fold buys, not a rewrite.
- **But the fold's blast radius grew on 2026-09-20:** the A-tiled prefill kernel is folded-only, so the +12.7%
  prefill win currently rides on the inherited idea. Independence means either rewriting the unpack or extending
  the A-tiled kernel to the non-folded path.

To claim independence here: regenerate `kMag` from the OCP e2m1/e4m3 definitions with a checked-in generator;
re-derive the unpack (shift/LUT, ~5%); re-derive or replace the `Wref` fold ABI; re-derive the fragment-order
layout in `r9700_vllm/kernels/moe.py:100-109` from the WMMA fragment layout (the bytes will likely be identical --
the derivation path is what changes); then update the header, `CREDITS.md`, `PROGRESS.md` and
`notes/review-fable.md`. **`kernels/r9k_gemm_fp8.hip` derives from the same kernel and needs the same treatment.**

## Still the right first move

None of the above is a substitute for **asking StillDeadcode for a license** (and GGZ14 for the chat template).
One message resolves the derived GEMM -- the only piece no flag can turn off -- and covers `r4d.so` for anyone
else running this. The work here is what makes the answer not matter.

## Note on what we do and do not ship

`r4d.so` is **not** in this repo: `.gitignore` excludes `*.so`, it is untracked, and it lives only on the test
box. Publishing the repo therefore does not redistribute libr4d. Without it the stack still runs -- the
communicator falls through to RCCL and attention falls back to `unified_attention` -- at the cost in the table
above. What publishing *would* carry is the derived GEMM source, which is section 3.
