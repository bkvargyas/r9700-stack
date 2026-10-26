# The plugin on the vLLM 0.32-era nightly (e4a5b553, 0.32.1rc1.dev466, ROCm 10.0 SDK, torch 2.13)

Brian (2026-10-25): "Can you also pull vllm .32 and test with that on our repo?" The production pin is the
2026-10-13 nightly `fe4df3df` (0.31.1rc1.dev296, torch 2.12). Same method as
[vllm-upgrade-2026-10.md](vllm-upgrade-2026-10.md): build the plugin image on the new nightly, regenerate the
VM100 overlay, run the integration gate and the kernel gates, then the three serving configs.

## What "0.32" is on ROCm

- **`vllm/vllm-openai-rocm:v0.32.0`** is, like v0.31.0, a ROCm 7.2-era image (classic `/opt/rocm`, no pip SDK):
  not the platform this stack targets. Pulled, not pursued.
- **Where v0.32.0 sits on main:** the release branched at `fe341a53d` (2026-10-12 20:49 UTC), 62 commits after our
  pin `fe4df3df`, and took 16 cherry-picks (CI, quantization, non-ROCm fixes). The ROCm 10 nightly closest to the
  release is **`nightly-rocm100-5555a2d8f`** (2026-10-13: branch point + 18 commits, still torch 2.12, version string
  `0.31.1rc1.dev376`); an exact v0.32.0 on ROCm 10 would need a source build on the `base-nightly-rocm100` image.
  The daily ROCm 10 nightlies after it: 77154ca61 (10-14), 3f1bcfd82 (10-15), 49cb2bbf7 (10-16), 39983a5e7 (10-17),
  da9134bc8 + 1b2a66639 (10-18), e68efc685 (10-19), df3bd094a (10-20), df6603f05 (10-22, first with torch 2.13),
  83c078d95 (10-23), c50aca144 (10-24), e4a5b5537 (10-25). Plugin images were built for 5555a2d8f, da9134bc8,
  df6603f05 (`~/build-nightly.sh SHA`, which no longer writes into a serving tree) alongside e4a5b553.
- **`nightly-rocm100-e4a5b553`** is the ROCm 10 nightly 466 commits past the 0.32 branch point: vLLM `0.32.1rc1.dev466+ge4a5b5537`,
  **torch 2.13.0+rocm10.0.0** (from 2.12), triton 3.8.0 (same), aiter 0.1.26 (from 0.1.24.post1), the **same
  `rocm-sdk-core 10.0.0`** (build 6b0e43f3). This is the test target: `r9700/vllm:dev-e4a5b553`
  (`docker/Dockerfile`, `BASE=` that nightly; the plugin installs as `r9700-vllm 0.3.6`, libr9k builds unchanged).

## Overlay (VM100 only)

- The hostcall-free RCCL build still matches: same SDK, same HIP runtime. Mounted as before.
- `patch-hostcall.sh`: `_C_stable_libtorch.abi3.so` patches as before (15 hostcall kernels -> 0). **`_rocm_C.abi3.so`
  now ships compressed offload bundles (CCOB, 18 of them)**, so the byte patch reports `0 patched` and 782 kernels
  still request hostcall. All 782 are the gfx9-only families -- `paged_attention_ll4mi_QKV_mfma4` (768) and
  `wvSplitKrc` (14) -- which never launch on gfx1201 (stock's custom paged attention and skinny GEMM are
  `on_gfx9()`-gated). The unpatched copy serves; the script now says so instead of a bare count.

## Gates (card 4 of VM100, `~/gates032.sh`, repo tree = v0.3.6)

- `tests/test_integration.py`: **ALL OK** (24 checks; the int6 PLE scope section skips, as on fe4df3df).
- Kernel gates: **32 of 35 pass** on the first run. `test_attn_r4d.py` needs `kernels/r4d.so`, which the gate image
  does not build (pre-existing, not a 0.32 matter). `test_qsa_score.py` passed its checks but the runner's
  25-line tail cut the verdict behind the timing table (re-run below with `BENCH=0`).
- **`test_ple_conv.py` failed with `AttributeError: Qwen4ExpPLELayer has no attribute
  '_short_conv_dilated_prefill_batched'`** -- and the same is true on the production pin fe4df3df: upstream replaced
  the packed Python prefill short conv with a fused Triton kernel (`models/qwen4_exp/common/ops/ple.py: ple_conv`,
  token-major, no transposes, state write-back in a second kernel) in the October nightlies. The plugin's
  `ple/short_conv.py` hook has therefore been **dormant since v0.3.3** (no "r9k PLE short-conv prefill installed"
  line in any v0.3.3-v0.3.6 launch log) and nobody noticed because the test was not in the release gate list.
  Fixed here: the test compares our `prefill_batched` with stock's `ple_conv(mode="prefill")` (2 bf16 ulps on the
  output -- the fused kernel applies silu in fp32 on the fp32 conv sum -- and bit-exact conv state) and times both;
  `install()` is off by default now. Result: see "PLE conv" below.

## Serving (VM100, `~/try032.sh`, overlay on, REPO = the same tree)

**GLM-5.3-Flash TP8** (MTP=3, fp8 KV, 32k, the production launch): the first run died at the first prefill with
`HIP Function Failed (.../ck_tile/host/kernel_launch_hip.hpp,65) device kernel image is invalid` from
`flash_attn_gpu.varlen_fwd`. Upstream gave sparse MLA layers a dense-MHA prefill path (`sparse_mla_attention.py:
forward_mha`, taken when the longest prefill is within the top-k window) and `RocmPlatform.get_mla_prefill_backend_cls`
hands it the CK flash-attention backend: AITER FA is gfx942/gfx950-only and `FLASH_ATTN`'s validation never looks at the
arch, so it is "valid" on gfx1201 and its wheel has no gfx1201 code objects. On the fe4df3df pin the same selector raised
(flash-attn was not importable there), the layer logged "Sparse MLA layer has no dense-MHA prefill path; using the top-k
MQA path only" and prefill ran on our libr9k sparse path. **Fix (plugin):** `R9700Platform.get_mla_prefill_backend_cls`
raises ValueError on gfx1201 -- the API's "none valid" answer -- so the layer falls back exactly as before
(`R9K_MLA_PREFILL=stock` defers to stock's choice); `tests/test_integration.py` checks the override raises and that the
layer's fallback still exists. With it:

| GLM TP8, e4a5b553 | pinned fe4df3df (v0.3.5 record) |
|---|---|
| decode 76.5 tok/s single (512 tokens, 3 runs) | 79.6 |
| 12.7k needle: correct, ttft 5.0 s, prefill 2,562 tok/s, decode 104.5 tok/s, acceptance 88 % | correct |
| sanity: 0 bad of 200 at 5 streams; 0 bad of 80 long prompts; 0 server errors | 0 bad |
| KV cache at 0.94: 157,118 tokens | 203k (the dense-prefill workspace is gone, the profile run differs) |

**Flash-Next TP4** failed three times at startup: `Free memory on device cuda:N (31.17/31.86 GiB) on startup is less than
desired GPU memory utilization (0.98, 31.22 GiB)`. Not residue: the new torch 2.13 runtime's per-process baseline on an
idle card is ~0.7 GiB (was under 0.64), so a 0.98 start no longer fits. Re-run at `UTIL=0.97` below.

**27B TP2**: the first run sat in `[backbone] Starting torch.compile` for 13 minutes without a line (the overlay's RCCL
is fine: `tests/test_ar_nrank.py` on two ranks passes in this image). Retried with a 40-minute allowance and a py-spy dump
of a worker after 4 minutes in compile; result below.

**Flash-Next TP4, continued.** At `UTIL=0.97` loading reached the int6 PLE table and died on
`ImportError: cannot import name 'compute_ple_shard_overlap'` (moved into `PLEVocabParallelEmbedding._compute_shard_overlap`;
our loader now does the three lines of arithmetic itself), then on `'GatedResidual' object has no attribute 'lora_rank'`
inside our hyper-connection mix during the profiling forward (the AMD class renamed it `hc_lowrank`; `hc.py` reads
either). With both fixes, `UTIL=0.97`:

| Flash-Next TP4 (probe.py) | e4a5b553 | fe4df3df (v0.3.6 regression, same day) |
|---|---|---|
| decode mean / code / prose / json / math | 209.9 / 221.8 / 113.3 / 261.9 / 242.7 | 209.7 / 221.3 / 111.7 / 261.3 / 244.5 |
| tokens per step (MTP) | 3.09 | 3.09 |
| 8 / 16 streams | 473 / 913 | 465 / 933 |
| prefill 8k | 6,830 tok/s | 6,771 |
| KV cache (0.97 vs 0.98) | 477,002 tokens | 494k |
| sanity | 0 bad of 680 at 17 streams; 0 of 80 long prompts; 0 errors | 0 bad |

**27B TP2: root cause found by bisection.** Every first request died with two `Memory access fault by GPU node-N ...
Page not present` (workers killed by the HSA fault handler, so no Python frame). Eager runs on cards 6/7, one knob at a time:

| run | switched to stock | result |
|---|---|---|
| A | GDN short-prefill core (`R9K_GDN_PREFILL_MAX=0`) | faults |
| B | whole GDN forward (`R9K_GDN_DECODE=stock`) | faults |
| C | B + GDN state page (`R9K_GDN_STATE=stock`) | faults |
| D2 | no drafter at all (`DRAFT= MTP=`) | faults |
| **E4** | **stock `TRITON_ATTN` for the target, no drafter** | **0 faults, 0 bad of 440** |

So: our `CUSTOM` attention backend (`attn/triton3d.py`). The nightly's `TritonAttentionImpl.__init__` now sets
`supports_quant_query_input = not (gfx90a or gfx11)` on ROCm (it was `is_cuda()`), and `attention.py` quantizes the
query to fp8 before `impl.forward` whenever the impl says so and the KV cache is fp8 -- which the 27B's is. Our paged
prefill kernel takes bf16 queries and computes its row stride from `query.element_size()`: an fp8 q is half the bytes,
the kernel reads past the buffer, page fault on the first prefill. GLM (MLA, our sparse path) and Flash-Next (QSA,
ours end to end) never see this. **Fix:** `R9kTriton3DImpl.__init__` pins the flag to False, and `forward` falls back
to stock for any query/output that is not 16-bit; `tests/test_integration.py` checks both. (Two earlier bisections,
D and E2, were void: `27b.sh`'s own env assignments beat the inherited knobs -- overrides to that script go as
arguments, `~/try032.sh ... ARGS27=`.)

With the fix, eager (`EAGER=1`): 0 faults, 0 bad of 440, **4.49 tokens a step** (the pin: 4.48-4.51). Compiled, the
production launch: 0 faults, 0 bad of 440, but **1.0 tokens a step -- every draft rejected -- 43 tok/s instead of ~216**.

**Second root cause (2026-10-26), found by compiled A/B runs on cards 6/7**, all 0 bad of 440:

| run | what | tok/step | decode mean |
|---|---|---|---|
| C0b | production config (ours, HIP graphs + torch.compile) | 1.0 | 43 |
| KV8b | same, KV budget 8 GiB instead of 9 | 1.0 | 43 |
| X2b | stock `TRITON_ATTN` for target and drafter, NHD layout | 4.57 | 219 |
| X10 | stock `TRITON_ATTN`, our LBHNC layout | 4.55 | 220 |
| X7 | ours, HIP graphs, torch.compile off | 1.0 | 34 |
| X8 | ours, torch.compile, HIP graphs off | 4.43 | 55 |
| **FIX** | **ours, graphs + compile, draft metadata rebuilt every step** | **4.46** | **217** |

So: our `CUSTOM` backend, only under graph replay, not the layout, not memory, not Inductor. The nightly (36f397281,
2026-10-12) lets the speculator skip the per-step metadata build for backends whose builder claims
`supports_draft_decode_metadata_update`; stock's Triton builder inherits True and its update is a no-op because its
metadata points at the live block table. Ours inherits True too, but it declares kernel block 16 for the paged
kernels, so vLLM remaps the hybrid 27B's 1808-token KV blocks into a *kernel block table* at build time
(`worker/utils.py: map_to_kernel_block_table`) -- a copy nothing refreshes under graph replay. The drafter attended
to stale blocks and the target rejected everything; eager mode rebuilt the table every step and was fine. **Fix:**
`R9kTriton3DMetadataBuilder` sets `supports_draft_decode_metadata_update = False` (the behaviour before the skip
existed; `R9K_DRAFT_META_UPDATE=1` opts back in). Result: 217 tok/s mean, 4.46 tokens a step, prefill 3,838 tok/s
(our paged kernel; stock's 3,205), 8/16 streams 503/631. Follow-up for speed: declare `MultipleOf(16)` like stock and
do the 16-token remap inside our prefill path, so the in-graph update can stay on.

The Oct 13 (`5555a2d8f`, the v0.32.0 branch point + 18 commits, torch 2.12) and Oct 22 (`df6603f05`) nightlies showed
the same 1.0 before the fix, so the whole 0.32 line needs it. (Earlier runs X1/X2/X5/X6 in this session were void: the
long-wait launcher dropped the serve-script overrides; the table above is the valid set.)

## Verdict

- The plugin builds and installs on the 0.32-era nightly; kernels, integration gate and overlay are fine.
- **GLM-5.3-Flash TP8 and Flash-Next TP4 serve at the pinned numbers** after four plugin-side fixes (MLA prefill
  backend, int6 shard overlap, hyper-connection low-rank attribute, fp8-query pin) and one serving knob (`UTIL=0.97`).
- **27B TP2 at the pinned numbers** after a fifth plugin-side fix (draft metadata rebuilt every step; the in-graph
  update left our remapped kernel block table stale). The production pin stays `fe4df3df` for now; all five fixes are
  forward-compatible and go into the next release, which can then move the pin to the 0.32 line.
- Retired on the way: the PLE short-conv prefill hook (stock's fused Triton kernel is bit-identical and 8x faster).

## PLE conv: ours vs stock's fused Triton kernel (card 4, dev-e4a5b553)

| 1 x 4096 tokens x 10240 channels | stock `ple_conv(mode="prefill")` | ours (`prefill_batched`, tiled transposes) |
|---|---|---|
| time | **496 us** | 4,120 us |
| peak memory (skewed 8-prefill steps) | 77-80 MiB | 419-526 MiB |
| output / conv state | identical (max diff 0 on every case, state bit-exact) | |
| the transpose alone, torch vs `r9k_transpose16` | 3,075 us | 465 us |

Stock moved past us: the fused kernel reads the token-major layout directly, so the two 84 MB transposes our path
was built to speed up no longer exist. The hook is retired (off by default, `R9K_PLE_CONV=r9k` re-enables it where
the packed method exists); the transpose kernel stays in libr9k.

## Verdict

(pending)
