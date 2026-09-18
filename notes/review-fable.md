# r9700-stack code review (Fable, 2026-09-18)

Scope: `kernels/*.hip`, `kernels/third_party/davetha/r4d_lru.hip`, `r9700_vllm/**`, `tests/`, `bench/`,
`serve/serve-stock-fn.sh`, `docker/Dockerfile`, `pyproject.toml`, checked against the exact upstream tree in
`~/src/vllm-nightly-dee37d8/vllm`, `~/src/libr4d` and `~/src/r9700-lru-expert-cache`. Nothing was run on the GPUs.
Line numbers are from the current HEAD (`8c0279d`).

Headline: I found no definite silent-corruption bug in the three kernels or the LRU integration; the folded-exponent
unpack, WMMA fragment/accumulator layouts, split-K reduction, padding-row clamping and the "cold pass is provably
empty" shortcut all check out against the sources. What I did find is a handful of *latent* failure modes that would
corrupt output silently rather than fail loudly if an assumption moves (P0 = cheap guards worth adding now), a set
of robustness gaps against upstream drift (P1), several concrete perf items (P2), and packaging/cleanup work that
stands between the current state and a clean "stock image + `pip install`" story (P3).

---

## P0 - correctness guards (cheap, protect against silent garbage)

### P0.1 UVA views silently become *copies* if `is_pinned()` stops recognising `hipHostRegister`'d memory
`r9700_vllm/utils/hostmem.py:27-33`, `r9700_vllm/ple/int6.py:90-95`, `r9700_vllm/moe/cache.py:52-54`

vLLM's `get_cuda_view_from_cpu_tensor` (`csrc/libtorch_stable/cuda_view.cu`) branches on `aten::is_pinned`: pinned
-> zero-copy `from_blob` (keeps the CPU tensor alive via the deleter capture, so lifetime is fine); **not pinned ->
`cudaHostAlloc` + `cudaMemcpy` of the current contents**, i.e. a private copy. Every later CPU-side write is then
invisible to the GPU. The plugin writes through the CPU tensor after creating the view in `int6.py:108-114`
(`load_int6_shard` writes into `self._r9k_host`), so a torch/HIP change in how registered memory is classified would
yield an all-zero PLE table with no error (GSM8K would just collapse). Today it works because torch's
`isPinnedPtr` uses `hipPointerGetAttributes`, but it is an undocumented dependency. Guard it:

```python
# hostmem.py
def uva_empty(shape, dtype):
    host = pinned_empty(shape, dtype)
    if host.numel() and not host.is_pinned():
        raise RuntimeError("r9700: hipHostRegister'd buffer not seen as pinned by torch; UVA view would be a copy")
    v = get_accelerator_view_from_cpu_tensor(host)
    v._r9k_host = host
    return v
```
and the same check in `int6.py` before `get_accelerator_view_from_cpu_tensor(host)`. Also make `pinned_empty`'s
fallback (`hostmem.py:21-22`) log a warning: the power-of-two rounding it falls back to is exactly what OOM-killed
the first bring-up, so a silent fallback hides the regression.

### P0.2 `r9k_moe_mxfp4a8_kernel`: padding-row clamp trusts the align layout
`kernels/r9k_moe_mxfp4a8.hip:93-100`

`first = sorted_ids[by*BLK]` is assumed valid for any live block. That holds for `moe_align_block_size` with
`pad_sorted_ids=False` and for our identity tables, but a fully-padded block (e.g. upstream flipping
`pad_sorted_ids`, a block-size mismatch between `moe_align_block_size(blk)` and `MT`, or a stale `_align` buffer)
makes every lane read A row `numel/a_row_div` = one row past the end of A. One compare fixes it:

```c
  const int first = sorted_ids[by * BLK];
  if (first >= numel) return;          // fully padded / malformed block: nothing to store anyway
```
Same for `expert_ids[by] >= E`: pass `E` (you already have `num_experts` on the Python side) and `return` on
`e >= E`; the arena path hands the kernel slot ids < S but a corrupted table would index past `Wp_all`.

### P0.3 PLE gather has no bounds check on `ids`
`kernels/r9k_ple.hip:21`, `r9700_vllm/kernels/ple.py:32`

`row = table + ids[i] * row_bytes` with `int64` ids and a 38.7 GiB host table: a negative or >= rows id is a GPU
page fault on host memory (hang / `hipErrorIllegalAddress` that takes the whole TP group down). Today
`VocabParallelEmbedding.forward` masks out-of-partition ids to 0 for TP>1 and the hash is modulo vocab, so ids are
in range, but pass `rows` and zero-fill instead:

```c
  const long id = ids[i];
  if ((unsigned long)id >= (unsigned long)rows) { for (int k = 0; k < 4; k++) o[k] = 0; return; }
```
(`table.shape[0]` is already available in `ple.py:32`.)

### P0.4 Expert-cache GEMM rows that fall in neither pass are never written
`r9700_vllm/moe/experts.py:136-161`

`gate_up`/`down` are `torch.empty`; every routed row must be covered by exactly one of the hot/cold passes. The
argument in the comment (numel <= min(max_distinct, max_inserts) => every routed expert resident) is sound
against `r4d_lru.hip` (evictable = S - (distinct - misses) >= misses because distinct <= numel < S), but it is the
one invariant whose violation is silent: uninitialised bf16 rows go through silu -> NaN or plausible garbage into
`moe_sum`. Add an opt-in check (`R9K_CHECK=1`) that, after `update_fused`, verifies on device
`npad_hot + npad_cold == sum(ceil(cnt_e / blk) * blk)` or simply that `n_miss == total` (a 1-int flag written by the
kernel on `nact < nins` and read once per N steps), and in debug builds `down.zero_()` unconditionally (0.6 MB at
decode, negligible). Also drop the `if expert_map is not None` gate on `down.zero_()` in the cache path.

### P0.5 `_HeadMethod`/`compute_logits` override drops `spec_step_idx`
`r9700_vllm/spec/draft_head.py:105-107` vs `vllm/models/qwen4_exp/amd/mtp.py:454-457`

Stock `Qwen4ExpMTP.compute_logits(self, hidden_states, spec_step_idx=0)`; the wrapper is `(self, hidden_states)`.
`llm_base_proposer.py:446-488` currently calls it positionally with one arg, but `step3p5.py:272` already passes
`spec_step_idx=`; the day the MTP proposer does too, every draft step raises. Use `*a, **k`:

```python
    def compute_logits(self, hidden_states, *a, **k):
        head = getattr(self, "_r9k_head", None) or self.lm_head
        return self.logits_processor(head, hidden_states)
```

---

## P1 - robustness against stock vLLM, idempotency, error handling

### P1.1 Hooks that fail to install fall back *silently* to slow-but-working paths
- `comm/r4d_ar.py:112-125`: `r4d.so` is not built or copied by `docker/Dockerfile` or `kernels/build.sh`; it is
  only found at `r9700_vllm/kernels/r4d.so` or `R9K_R4D_SO`. A clean image build therefore runs RCCL (68 -> 62 tok/s)
  with only a warning line. Same for `linear/mxfp4.py:83-85` (falls to `EmulationMxfp4LinearKernel`, dequant per
  call) and `hostmem.py:21`.
- Recommend a strict mode, default on inside the Docker image: `R9K_STRICT=1` makes `register()` raise when a hook
  it was asked for (`R9K_R4D_AR=1`, `R9K_EXPERT_CACHE_SLOTS>0`, ...) cannot be installed, and `register()` should
  log the *effective* state (`r4d_allreduce=on/rccl`, `mxfp4_linear=libr9k/emulation`) rather than the list of
  patches applied (`__init__.py:69` prints "r4d_allreduce" even when the communicator later fails).

### P1.2 `compat/checkpoint.py` is not idempotent on partial failure
`compat/checkpoint.py:119-129`: `_fix_ct_formats()` and `_exact_uva_pinning()` wrap class methods *before* the
`try` that may `return False` without setting `_PATCHED`. `load_general_plugins` is called from four places
(`plugins/__init__.py:77-89` guards with `plugins_loaded`, but tests and `VLLM_PLUGINS` re-entry do not), so a
second `patch()` double-wraps `from_config` and `_maybe_offload_to_cpu`. Set `_PATCHED = True` first, or give each
sub-patch its own `_DONE` marker on the target function (`getattr(fn, "_r9k_wrapped", False)`).

### P1.3 Version gating is "import succeeded", not "signature matches"
Every hook replaces a *method* (`CompressedTensorsW4A4Mxfp4MoEMethod.process_weights_after_loading`,
`CudaCommunicator.__init__/all_reduce`, `UnquantizedLinearMethod.apply`, `CompressedTensorsW8A8Fp8.apply_weights`,
`SpecDecodeBaseProposer.__init__`, `Qwen4ExpMTP.load_weights/compute_logits`, `Qwen4ExpModel.load_weights`,
`Qwen4ExpNGramEmbedding.load_weights`) and re-implements part of its body (`ct_mxfp4.py:120-142` mirrors
`compressed_tensors_moe_w4a4_mxfp4.py:158-217`, including `_expert_routing_tables()` and `mxfp4_backend`). Upstream
moved 1506 files in 9 days. Add one `compat/versions.py` with the tested vLLM commit set (`vllm.__version__` +
`vllm.__commit__` if present); on mismatch, `logger.warning` once per hook and refuse the hooks that copy upstream
bodies unless `R9K_ALLOW_UNTESTED=1`. Cheap and it turns "garbage after a nightly bump" into a clear message.

### P1.4 `ops._mxfp4_linear` creates its routing tables lazily, possibly inside graph capture
`ops.py:36-54`: `_MX_TABLES` is filled on first call per `(device, mpad, blk)`. vLLM's dummy runs precede capture
for every graph size, so in practice the tables exist, but nothing enforces it; a table first created during
capture lives in the graph's private pool and its `arange`/`zeros` become graph nodes. Guard with
`assert not torch.cuda.is_current_stream_capturing()` in the miss path, or pre-build for `mpad in
cudagraph_capture_sizes` from `register()`/first `process_weights_after_loading`. The same lazy `_align_bufs`
pattern in `cache.py:197-210` is fine because `update_fused` is also exercised by the dummy runs, but the same
assert costs nothing.

### P1.5 `experts.py` launch configs are process-global env constants
`experts.py:37-39`: `CFG_DOWN=(4,2,1)` is only legal for `K2 % 64 == 0`; another TP size or model (e.g. the dense
27B) raises `r9k_moe_mxfp4a8 failed (-2)` at the first forward. Fall back to `K.pick_cfg(N, K)` when the env
config is illegal, and validate once in `process_weights_after_loading` so it fails at load, not at the first
request.

### P1.6 `draft_head._shadow` relies on `copy.copy(nn.Module)` semantics
`draft_head.py:85-91`: a shallow module copy shares `_parameters`/`_modules` dicts with the original; it works
because `quant_method` is a plain attribute, but any future `LogitsProcessor` access to `lm_head` state (bias,
`tie_word_embeddings` checks, `_vllm_...` markers) would see the copy's `__dict__`. Build a tiny explicit object
(`types.SimpleNamespace(weight=head.weight, bias=getattr(head, "bias", None), quant_method=...)`) or subclass
`ParallelLMHead` once; also route the fp8 path through `ops.fp8_linear` (`draft_head.py:72-73` calls ctypes
directly; harmless today because `compute_logits` is outside the compiled graph, but it is the one remaining
ctypes call not behind a custom op).

### P1.7 `LayerCache` thresholds are silent perf cliffs at wide batches
`cache.py:132-136`: `max_distinct = S/2 = 135`, `max_inserts = 64` at 270 slots. A B=16 MTP-3 decode step routes
640 rows -> distinct experts per layer routinely exceed 135 -> **read-through forever** (no inserts, the warm-start
set never adapts). That is by design (davetha's numbers), but the plugin should at least log the effective regime
once (`numel > no_cold_limit` and read-through counters); see P2.2 for tuning.

### P1.8 Padded cudagraph rows pollute the LRU
Under full-graph decode the padded rows (dummy hidden states) still route to real experts and count as "routed
this step" in `lru_fused_k` (they also get inserted). Mask them by writing `-1` into the flattened `topk_ids`
copy for rows >= real `num_tokens` (available from `get_forward_context().attn_metadata` / `num_actual_tokens`)
before `update_fused` - the kernel already skips `e < 0`.

### P1.9 `r4d` 2-rank all-reduce spin breaks silently after 4e9 iterations
`~/src/libr4d/r4d_ar_oneshot_2rank_exact.hip:108` (already noted in `notes/ggz14-libr4d-review.md:148`). Not a
practical race (4e9 spins is minutes), but a hung peer produces stale-data output instead of a trap. If you vendor
the kernel (P3.2), make it `__builtin_trap()` like `r4d_ar_wide.h:92`.

### P1.10 Minor
- `r9k_moe_mxfp4a8` host (`r9k_moe_mxfp4a8.hip:253`) puts `max_blocks` in `grid.y`; add `if (max_blocks > 65535)
  return -8;` (or swap the axes) so a hypothetical runtime limit shows up as an error, not a silent partial launch.
- `checkpoint._exact_uva_pinning` (`:99-113`) patches `torch.Tensor.pin_memory` process-wide during offload; fine
  single-threaded, but wrap in a `threading.Lock` or patch `uva` module's local name instead.
- `int6._checkpoint_ple_format()` re-reads the safetensors index + header per PLE layer; cache the result.
- `_Int6EmbeddingMethod.embedding` ignores `params_dtype`; fine for bf16 models, assert it.

---

## P2 - performance (with expected impact)

### P2.1 Prefill read-through streams each cold expert once per 16*MT-row block  (largest win)
`experts.py:119-140`, `r9k_moe_mxfp4a8.hip:87-140`. On a wide step the cold GEMM reads expert `e`'s 1.25 MB
(N1*K1/2 + N2*K2/2 + scales at TP2) over PCIe **once per routing block** of that expert (grid.y), i.e.
`ceil(rows_e / (16*MT))` times. At a 4096-token chunk, top-10, 512 experts: ~80 rows/expert, MT=4 -> ~2x traffic
= ~1.3 GB/layer, ~60 GB/chunk at 14 GB/s ~ 4.4 s -> matches the observed ~500 tok/s prefill ceiling. Two fixes:
1. **Stage cold experts in VRAM for wide steps**: when `numel > no_cold_limit`, gather the distinct cold experts
   (the `miss` list already exists; extend `lru_fused_k` to emit the full non-resident routed list when
   `!ins`) into a per-layer scratch `[<=E-S] x 1.25 MB` (max 300 MB at S=270; chunk if needed) with
   `r4d_lru_gather`, then run the "cold" pass over the staging buffer with `expert_ids` remapped. Traffic becomes
   exactly 1x per expert. Expected: prefill 500 -> ~900-1000 tok/s at 4k chunks (PCIe-bound at 1x = 30 GB/chunk).
2. Alternatively, `MT=8` instantiation (`acc[NPW][MT]` = 128 VGPRs at NPW=2, still fits wave32) so one block
   covers 128 rows -> ~1x traffic for this model without staging. Simpler, but only helps when rows/expert <= 128.

### P2.2 LRU regime tuning for concurrent decode
`cache.py:132-136`. At @8/@16 the cache is in permanent read-through (P1.7). Sweep `R9K_LRU_THRESH` (0.5 -> 0.8)
and `R9K_LRU_MAX_INSERTS` (64 -> 128) at @8/@16 with `bench/bench.py`; davetha's tests (`tests/lru/test_slots_prod.py`,
`bench_victim.py`) show the ranked victim path is flat in `nins`. Also add the telemetry counter davetha describes
(`notes/tcclaviger-fork-analysis.md:235`) so the regime is visible. Expected: modest at @16 (aggregate is
PCIe-bound anyway), possibly +5-10% at @4/@8.

### P2.3 Dense fp8 GEMM under-fills the GPU at N <= 4096
`kernels/fp8.py:60-70,85-98` defaults `WV=4, NPW=2` -> `grid.x = N/128` workgroups: 20 WGs for N=2560, 52 for
6656, 64 for 8192 on a 64-CU part, each WG only 16 waves. That is why `R9K_FP8_BLOCK=block` (exact math) is not a
clear win over the tuned Triton kernel (~4.7 ms/step of block GEMMs, ~130 launches). Pick `(WV, NPW, SK)` per
shape so `grid.x * WV * SK >= ~8 waves/CU * 64`: e.g. N=2560,K=3072 -> `WV=1, NPW=1, SK=8` (160 WGs of 8 waves,
`K % 1024 == 0`), N=8192,K=2560 -> `WV=2, NPW=1, SK=4`. Add a tiny shape->cfg table like `pick_cfg` for the MoE
kernel and re-run `tests/test_gemm_fp8.py`'s bench lines. Expected: 1.5-2.5x on those GEMMs at decode M, i.e.
1-2 ms of the ~37 ms step (3-5%), and it makes the exact `block` path the default instead of Triton.

### P2.4 PLE gather: coalesce the row read
`r9k_ple.hip:22-26`: each thread does 3 byte loads + 2 byte loads from uncached host memory; a 130 B row becomes
~10 PCIe read requests instead of 3. Read the row as 33 aligned `uint32` per wave (rows start at 2-byte
alignment; load `uint16` x 65 or shift-merge from the 4-byte-aligned `row & ~3` base), stage in LDS/registers, then
dequant. PLE is ~5% of single-stream in the ablation (`ablation/RESULTS.md` no_r4d_ple) so expect <= +2-3%
single-stream; cheap to try.

### P2.5 Launch count per MoE layer at decode
Current: lru_fused + gather + quant_rows + hot gate_up + silu_quant + hot down + moe_sum (+2 cold when
`numel > 64`) = 7-9 launches x 48 layers ~ 340-430 nodes/step at ~2.4 us each ~ 1 ms. Options ranked by
value/effort: (a) fold `quant_rows_fp8(hidden_states)` into the previous op (the HC/gate epilogue) - saves 48
launches; (b) merge `lru_gather` into `lru_fused` as a second grid (needs a cooperative launch or a persistent
loop - skip); (c) `moe_sum` into the down-GEMM epilogue via fp32 atomics - deterministic-order loss, skip.

### P2.6 Small things
- `experts.py:158` `topk_weights.reshape(-1).to(torch.float32)` allocates a copy per call when already fp32 and
  contiguous (it does not, `.to` is a no-op then) - fine; but `ids.to(torch.int32)` in `cache.py:216-217` copies
  every step: `topk_ids` from vLLM is already int32, assert instead.
- `_maybe_attach_cache` calls `torch.cuda.empty_cache()` per layer (48 syncs at load) - harmless.
- `-ffp-contract=off` in `build.sh:10` disables FMA formation in the silu/quant kernels; keep for reproducibility
  vs. the torch reference, but it is not needed for the WMMA paths. Consider `-ffp-contract=on` for the `.hip`
  files that do not need bit-exact matching (measure; likely noise).

---

## P3 - packaging, structure, naming, dead code

### P3.1 Target layout for "stock ROCm + stock vLLM + pip install r9700-vllm"
```
r9700-stack/
  pyproject.toml            scikit-build-core (or setuptools + custom build_ext) -> builds libr9k.so with the
                            image's hipcc; wheel tagged for gfx1201; `R9K_LIB` override kept for dev
  r9700_vllm/
    __init__.py             register(): env defaults (P3.4), version gate (P1.3), hooks, effective-state log
    platform.py             R9700Platform(RocmPlatform) via vllm.platform_plugins: check_and_update_config
                            (--language-model-only default for this arch, cudagraph sizes, spec defaults),
                            use_custom_allreduce=False, import_kernels() loads libr9k
    ops.py, kernels/{moe,fp8,ple}.py, moe/, ple/, linear/, spec/, comm/, compat/, utils/   (as now)
    _csrc/                  r9k_*.hip + third_party/davetha/ (LICENSE, NOTICE)  -- move kernels/ here so the
                            wheel sdist carries sources and license files
  tests/                    pytest: unit (GPU-marked) + cpu-only (layout/packing/regex/config rewriting)
  bench/  tuning/  notes/
  ops/                      everything host/topology-specific: docker/, rccl/ (hostcall-free rebuild),
                            p2p/ (scanhc.sh + the patched-.so overlay), serve/ launchers, ablation/, profiling/
  legacy/                   serve-flashnext*.sh, deploy.sh (tcclaviger / GGZ14 era scripts) -- or delete
```
- `pyproject.toml:17-18` ships `kernels/*.so` as package data but nothing builds it; the Dockerfile does it by
  hand. Either a `build_ext` that shells out to `hipcc` (`kernels/build.sh` logic, `GFX_ARCH` from
  `torch.cuda.get_device_properties().gcnArchName` at build) or runtime JIT into `~/.cache/r9700` on first
  `lib()` miss (like `torch.utils.cpp_extension.load`, same hipcc) so `pip install` in the stock image just works.
- Export and check an ABI tag: `extern "C" int r9k_abi_version(void)` in libr9k, compared in `moe.py:lib()` next
  to the existing `r9k_moe_block()` assert.
- Keep `vllm.general_plugins` (must run in every process) and add the platform plugin for the config-level
  defaults; do not move kernel hooks into the platform class.

### P3.2 libr4d dependency and licensing
- `r9k_moe_mxfp4a8.hip` and `r9k_gemm_fp8.hip` are derived from libr4d's `gemm_mxfp4a8_nt_m64` (fragment layout,
  unpack tables, split-K structure); libr4d has **no license** (`notes/ggz14-libr4d-review.md:12`). Until
  StillDeadcode grants one, the repo cannot be published. Options: (a) ask for Apache-2.0/MIT and record the grant
  in `NOTICE`; (b) clean-room the two kernels (the layout is standard WMMA fragment order; the folded-exponent
  `v_perm` unpack is the only clever part and could be replaced by a shift/LUT unpack at ~5% cost).
- The all-reduce (`comm/r4d_ar.py`) needs the *whole* `r4d.so` pybind module for one kernel plus
  `ar_ipc_alloc/open`. Once licensed, vendor `r4d_ar_oneshot_2rank_exact.hip` + the IPC helpers into libr9k as
  `r9k_ar_*` (C ABI, ctypes like the rest), drop the pybind dependency and the `select()` call; add the trap (P1.9).
  Until then treat `R9K_R4D_AR=1` as an explicit opt-in with a strict failure (P1.1).
- `kernels/third_party/davetha/` is correct (Apache-2.0, LICENSE+NOTICE, pinned commit, unmodified - verified
  identical to `~/src/r9700-lru-expert-cache/kernels/lru/r4d_lru.hip`). Add `tests/lru/test_victim_equiv.py`,
  `test_slots_prod.py`, `test_graph_lru.py` from that repo to `tests/` (same license).

### P3.3 Host-topology hacks stay out of the plugin
The hostcall-free RCCL rebuild (`rccl/build-nightly.sh`), the binary-patched `_rocm_C`/`_C_stable_libtorch`
(`p2p/scanhc.sh`, `PATCHED=` mounts in `serve-stock-fn.sh:17-20`) and `NCCL_PROTO=Simple` are only needed on the
emulated-PLX-switch VM. Keep them as an `ops/topology-plx/` overlay: a `detect.sh` (runs `scanhc.sh` + a P2P IPC
probe) and a documented `docker-compose`/`run` variant that mounts the rebuilt bits. The default Dockerfile should
produce a plugin image that runs on a normal Gen4 host with stock RCCL. `kernels/build.sh:11`'s hostcall grep is a
good invariant for libr9k itself; keep it in CI.

### P3.4 Environment requirements
`HSA_ENABLE_IPC_MODE_LEGACY=0`, `GPU_MAX_HW_QUEUES=1`, `HSA_ENABLE_MWAITX=1`, `VLLM_ROCM_USE_AITER=0` live only in
`serve-stock-fn.sh:54-55`. `register()` runs in the API-server/engine parent before workers are spawned, so
`os.environ.setdefault(...)` there propagates to the worker processes (HSA reads them at first HIP call in each
process). Do that with a one-line log per variable, and document them in README as "required on gfx1201, set by
the plugin unless overridden". Also record total pinned host RAM the plugin needs outside vLLM's
`--cpu-offload-gb` budget (PLE ~20.8 GB/rank + scales), since the profiler does not see it.

### P3.5 CI
- GPU runner (self-hosted gfx1201, one card): `kernels/build.sh` + hostcall grep; `tests/test_moe_mxfp4.py`,
  `test_gemm_fp8.py`, `test_ple_int6.py`, `test_cache_moe.py` (convert to pytest, keep the `ALL OK` exit codes);
  vendored davetha LRU tests; a `pytest -m cudagraph` case that captures `R9700Mxfp4Experts.apply` +
  `update_fused` into a `torch.cuda.CUDAGraph` and replays 3x, comparing to eager (the invariant the whole design
  rests on).
- CPU-only (any runner, no ROCm): `permute_fragments`/`permute_fp8` round-trips vs. a naive reference,
  `pack_scales`, `align_block_size_ref` vs. a brute-force pad, `quantize_mxfp4` round-trip error bound,
  `compat.checkpoint._rename_scale/_DROP/_MTP_MLP_FP8` regexes against a fixture of the checkpoint's tensor names,
  `_fix_ct_formats` on the checkpoint's `config.json`, `int6._checkpoint_ple_format` on a fixture header,
  `kernels/moe.py:lib()` ctypes `argtypes` counts vs. the `extern "C"` signatures (a regex over the `.hip` files).
- Nightly: build image from the pinned digest, serve, `bench/quality.py` gate (GSM8K >= 95/100, needle 3/3),
  `bench/bench.py` single/@16 with a +-5% band, and a job against the *latest* nightly that only reports (P1.3).

### P3.6 Cleanup / naming / dead code
- 29 `__pycache__/*.pyc` files are tracked (`git ls-files | grep pycache`); add `__pycache__/` and `*.pyc` to
  `.gitignore` and `git rm --cached`.
- `linear/mxfp4.py:26,53-61`: `self._ids` and `_tables_unused` are dead (ops.py owns the tables); `draft_head.py:
  59-61,75-77` constructs a `R9700Mxfp4LinearKernel.__new__` + fake layer only to reach `ops.mxfp4_linear` -> call
  `ops.mxfp4_linear(x2, self.W.wq, self.W.wsr, self.Np, self.K)` directly and delete `_lin`.
- `spec/draft_head.py` handles both target and draft heads -> rename `spec/lm_heads.py` (env names already say
  `R9K_TARGET_LMHEAD`/`R9K_DRAFT_LMHEAD`); `quantize_mxfp4` belongs in `kernels/moe.py` next to
  `prepare_mxfp4_weights`.
- `linear/fp8_block.py:27,40`: `out_rows` unused.
- `kernels/moe.py:151-169` `align_block_size_ref` is test-only -> `tests/helpers.py`.
- `comm/custom_ar.py`: documented as producing garbage on gfx1201; keep only if you intend to debug it, else
  delete (the docstring already records the finding for `notes/`).
- `serve/serve-flashnext*.sh`, `serve/deploy.sh`, `bench/*_prod.py`, `bench/fn-bb*.json`, `ablation/`,
  `profiling/night-chain.sh` are tcclaviger/GGZ14-era artefacts; move under `legacy/` or `ops/` (P3.1).
- `moe/cache.py:113,119,178-179`: `h_dummy`/`a_dummy` exist only to fill the 6-buffer `r4d_lru_gather` ABI;
  fine while the file is vendored unmodified, note it in the vendoring README so a future 4-buffer entry point
  can drop them.
- `serve-stock-fn.sh:3-4` header still says "Stage-1 bring-up ... no expert cache yet, no MTP"; update.
- `r9700_vllm/__init__.py` docstring says hooks "log instead of raising if the API moved" - once P1.1's strict
  mode exists, say which hooks are allowed to degrade.
- `PLAN.md` "Package layout" describes `platform.py`, `models/qwen4_exp.py`, `quant/ct_mxfp4_moe.py` that do not
  exist; the monkeypatch approach won instead - update PLAN.md to the actual layout (or the P3.1 target).

---

## Verified OK (so nobody re-audits these)
- MoE kernel: `v_perm` folded-exponent tables (`kMag`) map e2m1 codes to exact e4m3 for `dsh` 0..8 including the
  subnormal range; `dsh` clamp to 15 zeroes blocks >2^-12 below the row max (precision floor, not a bug);
  fragment order (`permute_fragments`) matches the lane addressing; accumulator row/col decode in the split-K
  epilogue (`(off>>7)<<3 | off&7`, `(off>>3)&15`) matches the gfx12 16x16 f32 WMMA layout; LDS `red` sizing and
  the per-M-tile `__syncthreads` pairing are race-free; `As`/`topk_w` indexed by the flat id; `Wref` scale
  `2^(e-127)` via `<<23`.
- FP8 GEMMs: row clamp `min(r, M-1)` + `m >= M` drop; block variant's per-128-group `tmp` fold reproduces
  vLLM's `w8a8_block_scaled_mm` math; `Bs[(n/128)][g]` indexing is right because 16 | 128; `As` is `[M][K/128]`
  row-major in both the quant kernel and the GEMM; `K % (128*SK)` enforced.
- `moe_gemm` `max_blocks` bound `ceil(numel/blk) + min(numel, E)` is a valid upper bound on what
  `moe_align_block_size` can fill; `_align_bufs` `L/NB` reproduce `moe_align_block_size.py:74-84` for
  `pad_sorted_ids=False`; both are host-known -> graph-safe.
- `lru_fused_k` single-exit design, victim ranking equivalence, `table`/`map_cold` complementarity, `mk <= 256`
  deterministic placement; `no_cold_limit` argument holds.
- UVA lifetime: `cuda_view.cu` captures the CPU tensor in the deleter; `_KEEP` + `_r9k_host` cover plugin
  buffers; `_permute_in_place` keeps the original storage.
- Dynamo: every in-graph ctypes call is behind a custom op (`torch.ops.r9700.*`, vLLM's `moe_forward`,
  `qwen4_exp_amd_ple_ngram_embedding`, `vllm.all_reduce`); only the LM-head path calls ctypes directly and it is
  outside the compiled region.
- TP determinism: r4d exact AR is commutative fp32 on 2 ranks; LRU state divergence between ranks would not
  affect outputs (each rank owns its shard); `moe_sum` is deterministic.
- ctypes `argtypes` in `moe.py:27-32`, `fp8.py:19-24`, `ple.py:17-18`, `cache.py:38-47` match the `extern "C"`
  signatures.
