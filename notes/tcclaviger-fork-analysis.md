# tcclaviger/vllm:dev fork: analysis for a plugin port to stock vLLM 0.29 + ROCm 10

Date: 2026-09-18.

**Trees compared**

| label | path | what it is |
|---|---|---|
| F | `~/src/tcclaviger-dev/vllm/` | the fork: 0.27.0.dev0+g55c98e370a, "Clav 28.06.1" |
| B | `~/src/vllm-55c98e3/vllm` | base commit |
| U | `~/src/vllm-0.29.0/vllm` | target |

**Related trees**
- Wrapper packages: `~/src/tcclaviger-dev/{clav_*,gdn_hip,ple_*.py,src/,tools/,recipes/}`.
- Public libr4d v0.5.0: `~/src/libr4d`.
- davetha's LRU cache repo (Apache-2.0): `github.com/davetha/r9700-lru-expert-cache` @3743f13. A scratch clone was used for this review.

**Method**
- Every file in F that differs from B was classified against U:
  - **510** files are byte-identical to U. These are upstream backports and need no work.
  - **214** are absent from U.
  - **517** differ from both B and U.
- "uniq" in the tables below = lines added vs B that appear nowhere in the U version of the same file. This is a measure of fork-original code.
- Things to ignore:
  - `F/vllm.py` and `F/parallel.py` (2469/922 uniq) are stray copies of `config/vllm.py` and `config/parallel.py`.
  - `F/vllm-rs` is a 52 MB ELF binary.
  - `third_party/triton_kernels` is vendored.
  - About 112 files are tuned JSON configs, mostly shapes for the 27B model.
- Much of each qwen4_exp diff is API drift where F is *older* than U: `tokens_per_state`, the `KVCacheLayout` enum, `wrap_modules(prefix=)`, `MultipleOf(16)`. Always keep the U side and port only the feature hunks.

**Serving recipe that is actually live**
- Source: `recipes/Qwen3.8-Flash-Next/tp2-expert-mem-45gb-ple-cache-8gb.md`.
- Checkpoint: Qwen3.8-Flash-Next-MXFP4-FP8 (compressed-tensors MXFP4 experts, FP8-block dense layers).
- Flags:
  - `TP2`, `--max-model-len 262144`, `--max-num-seqs 16`, `--max-num-batched-tokens 4096`, `--kv-cache-dtype fp8`, `--gpu-memory-utilization 0.95`
  - `--enable-expert-offload --expert-offload-mem 45`
  - `VLLM_PLE_CPU_OFFLOAD=1 --ple-nvme-offload --ple-nvme-dir /app/pleoffload --ple-cache-gb 8 --ple-cache-reuse true`
  - MTP-3, cudagraph sizes 4..32
- Result: about 84 tok/s single-stream and 238 tok/s aggregate at 8 streams.
- It runs on **Model Runner V2**. Qwen4Exp is not in `U/config/vllm.py:71` `ROCM_DEFAULT_MRV1_ARCHITECTURES`. So the fork's V1-runner edits are inert for this recipe:
  - `gpu_model_runner.py` spec staging
  - `v1/spec_decode/qwen4_exp.py`
  - DRY
- `models/qwen4_exp/config.py:207-244` (`_force_qwen4_exp_env`):
  - forces `R4D_QSA=1`, `R4D_AR=1`, `CLAV_TUNABLEOP_SWEEP=1`
  - defaults `CLAV_GDN=1`, `CLAV_PLECONV=1`

**Corrections to our `kernel-inventory.md`**
- "Every step runs one GEMM over slots and one over host tensors" is only true of *wide* steps (prefill, big batches). Decode steps are narrow: every routed expert is installed first and the GEMM runs over VRAM slots only (§2.5).
- MTP verify batches do **not** fall back to Triton for GDN. `gdn_hip.gdn_verify_r` has covered spec batches since 2026-08-09.
- `gdn_conv_{prep,update}_w4_h128` are public in libr4d. The fork's Python calls **no** r4d `gdn_*` symbol; GDN runs on the separate `gdn_hip` .so.
- Some items marked private have source in the repo:
  - `ple_hip`, `ple_rocr` and `ple_nvme`: `src/plehip/csrc`
  - `clav_hc`: `src/mischip/q4hc/csrc`
- The rdna4 scaled_mm and mxfp4_16 kernels are pure Triton, with source in F.
- Two sources were missing from the list:
  - **`libqnf_ple_cpu.so`** is private and CPU-only. It is used by the PLE offload process.
  - The LRU kernels have a public Apache-2.0 origin (davetha).

---

## 1. Feature groups: what the fork changes relative to 55c98e3

**Class key.**
- a = already in U
- b = pure Python, portable to a plugin (general_plugins entry point, `register_backend`, `PluggableLayer.register_oot`, `ModelRegistry` override, monkeypatch)
- c = needs a vLLM core patch (or a deep monkeypatch)
- d = depends on private kernels

"Live" means used by the Flash-Next recipe.

| # | group | main files (uniq lines) | class | private symbols | live? |
|---|---|---|---|---|---|
| 1 | **Qwen4Exp model port** | `models/qwen4_exp/**`, `transformers_utils/configs/qwen4_exp.py`, `v1/spec_decode/qwen4_exp.py` (158, V1 only) | **a** (U has `models/qwen4_exp/{common,amd,nvidia}`). Fork deltas on top: amd/model.py 169, mtp.py 230, qsa.py 198, ops/qsa.py 444, indexer_qsa.py 106, ops/hc.py 204, ple_layer.py 470, common/ple.py 133, qsa_cache.py 62, config.py 60 | b (model override via `ModelRegistry.register_model`) + d | see rows below | yes |
| 2 | **Expert offload** (plan, placement, UVA split, device LRU) | offloader/expert_plan.py 557, expert_placement.py 158, expert_activation_registry.py 115, expert.py 246, fused_moe/expert_topology.py 243, expert_lru.py 576, config/offload.py +140, arg_utils +~40, v1/worker/gpu/cudagraph_mem.py 73; hooks in routed_experts.py:1232-1247, offloader/base.py:141-145, gpu_worker.py:663-700,1303-1340, base_loader.py:86-101, engine/core.py:350-447 | b for the logic; c for 4 small hooks (§2.1) | `r4d.moe_lru_{manage,fused,gather}` + `MOE_LRU_*` (public analogue in davetha) | yes |
| 3 | **R4D MXFP4 MoE experts + route** | experts/r4d_mxfp4_moe.py 554, router/r4d_route_router.py 252, kernels/r4d_lib.py 78, compressed_tensors_moe_w4a4_mxfp4.py 25, router_factory.py 9 | b (swap `experts_cls`, factory patch) + d | `r4d_gemm_moe_mxfp4a8_nt_b16_m64{,_mt1}` (ctypes), `moe_route_softmax_bf16`, `clav_silu_quant.silu_mul_fp8` | yes (**U on ROCm otherwise picks CUDA-only Marlin for CT MXFP4 MoE**) |
| 4 | **Fused shared expert without aiter** (shared expert = routed id E_r, E=513) | fused_moe/utils.py:96-130, router/shared_routed_fused_topk_router.py 83, models/qwen3_next.py:139-160 (inherited by `Qwen4ExpSparseMoeBlock`, U amd/model.py:160) | b (monkeypatch the resolve function and the router factory; U runner already has `_fse_fuse_gate` at moe_runner.py:280,898) | – | yes (expert offload *requires* it: expert_lru.py:295) |
| 5 | **PLE offload** (CPU process, NVMe table) | v1/ple_offload/{worker 1102, connector 440, nvme_table 378, protocol 105}, layers/ple_offload_layer.py 188, layers/gpu_stream_ops.py 118, amd/ops/{ple_int6 91, ple_cpu_native 99}, config/offload.py PleNvmeConfig, hooks in gpu_worker.py, multiproc_executor.py:673-684, MRV2 model_runner.py, fix_functionalization.py:174-180, parallel.py | b + c (worker spawn, runner launch/release) | `ple_hip`/`ple_rocr`/`ple_nvme` (**source in `src/plehip`**), `libqnf_ple_cpu.so` (**no source**), `r4d.ple_dequant_i6g32_f16` (optional), `clav_pleconv.fwd` (optional) | yes |
| 6 | **QSA sparse-attention kernels** | amd/ops/qsa.py 444, indexer_qsa.py 106, amd/qsa.py 198 (fp8 KV owner, fused gate, `sel_len`, shared scratch), qsa_cache.py (~45 real: `update_draft_decode_metadata`) | b (module-attribute swap + model override) + d | `attn_sparse_score_h128_bf16`, `attn_sparse_topk_expand`, `attn_sparse_h256_{bf16kv,fp8kv}`, `attn_sparse_h256_scratch_bytes`, `qsa_index_prep_h128_bf16`, `clav_attn.qsa_{mqa_score,sparse_attn}_16` | yes (**fp8 KV has no Triton fallback**) |
| 7 | **Dense attention backends** | v1/attention/backends/r4d_attn.py 346 (`R4D_ATTN=1`, default off), clav_attn package (external plugin), triton_attn.py 204 + attn_autotune.py 439 + triton_unified_attention.py 30 (autotune, fp8-scale fold, fp8-Q on gfx12), platforms/rocm.py priority, registry.py `R4D` enum, selector.py (obsolete in U) | clav_attn b (already a plugin; must insert ahead of U's `ROCM_ATTN`); autotune c | `clav_attn.unified_attn_16`; r4d `attn_{prefill,decode}_h256_gqa6*` (public) | clav yes, r4d no |
| 8 | **RDNA4 dense GEMM dispatch** | scaled_mm/fp8hip.py 224, scaled_mm/rdna4.py 139, rdna4_matmul_fp8.py 323 (Triton), mxfp4/r4dhip.py 192, quantization/mxfp4_16{,_kernels}.py 536+567, experts/mxfp4_16_moe.py 244, compressed_tensors.py 129 (block-FP8 layers → Fp8LinearMethod, scale-name sniff), fp8.py 24 (block refine), ~112 tuned JSONs | b (`list.insert` into U `_POSSIBLE_*_KERNELS`, re-register quant config) + d | `libfp8hip_gemm.so` (`fp8hip_table_make`, `fp8hip_gemm_w8a8_launch_shape`), public `r4d_gemm_mxfp4a8_nt_m64` | fp8hip yes; mxfp4_16 no (other format) |
| 9 | **clav_\* glue** | ops/hc.py (`clav_hc.fused`), layers/mamba/ops/causal_conv1d.py 57, v1/worker/mamba_utils.py 77, triton_reshape_and_cache_flash.py 51, v1/worker/gpu/shutdown.py | hc b; others c (functions imported by name) + d | `clav_hc` (source present), `clav_conv1d`, `clav_state_copy`, `clav_memcpy`, `clav_reshape_cache` | yes (state_copy is inert on U's multi-Mamba-group layout) |
| 10 | **GDN** | external `gdn_hip` package (739 lines, already a `general_plugins` OOT `QwenGatedDeltaNetAttention`) | b (fix `attn_metadata[prefix]` → `.get`) + d | `gdn_hip.gdn_{decode_r,prefill_r2,verify_r}` | yes |
| 11 | **GDN side-cache** | only in `tools/image_entrypoint.sh:287-289,374` help text (`--gdn-side-cache-entries`, `VLLM_GDN_SIDECACHE_DEBUG`) | **no implementation in this checkout** (probably an image overlay) | ? | ? |
| 12 | **All-reduce / all-gather** | device_communicators/r4d_all_reduce.py 410, clav_all_gather.py 79, cuda_communicator.py 61 (dispatch at :139-170, :353-359, :411-429), logits_processor.py 14 | b (platform plugin `get_device_communicator_cls` → subclass, or monkeypatch) + d | public `ar_oneshot_2rank_{exact,wht6}`, `ar_ipc_*`, `select`; private `ag_oneshot_{4,8}rank_exact`, `clav_ag_ext` (opt-in) | yes (TP2 AR) |
| 13 | **Spec decode** | V2 autoregressive/speculator.py 338 (`FusedDraftStepUpdate` Triton, confidence early-exit), speculator.py 26, sample/confidence.py 97, config/speculative.py 35, spec metrics 54, mtp.py (replicated fc, shard-local argmax, draft W4 lm_head); DFlash/DFlash2/tap_export (~800, inert); radiance_plugin.py (already a plugin, off) | c (speculator internals) / b (model) / d (draft W4 uses public `gemm_w4a16`; `dflash2_prep_*` private) | `dflash2_prep_h128_*` (inert) | FusedDraftStepUpdate only |
| 14 | **DRY / degen-loop stop** | v1/sample/ops/dry.py 335, penalties, input_processor `_resolve_dry_params`, sched/utils.py `check_degeneration`, scheduler, chat serving notice | DRY is **V1-only, dead on V2**. A logitsproc plugin would force V1 (U config/vllm.py:2598). Degen: b (wrap `scheduler.check_stop`); U has per-request `repetition_detection` (partial a) | – | degen only |
| 15 | **KV / memory sizing** | v1/core/kv_cache_utils.py 346 (CSA+linear hybrid grouping for QSA layers), attn_utils.py 85 (page-aligned KV-first vs Mamba aliasing), utils/mem_utils.py 225 (no double-charged activation peak, inductor VRAM credit), cudagraph_mem.py (superseded in U) | c | – | yes (kv_cache_utils probably essential; check U's own QSA grouping first) |
| 16 | **TunableOp warmup** | model_executor/warmup/tunableop_warmup.py 347, base_loader.py hook | b (wrap loader) | – | forced on, but a no-op unless `PYTORCH_TUNABLEOP_ENABLED=1`; skipped under expert offload |
| 17 | **Engine core cold-compile respawn / quiet compile** | v1/engine/core.py 122 | c | – | no (only with `--expert-precise-memory`) |
| 18 | **Entrypoint tools** | tools/image_entrypoint.sh (router for `--tune/--quantize/--map-experts/--flags/--recipes`), tools/expert_map (3.9k lines, REAP-style activation mapping), tools/expert_offload/convert_hot_profile.py (sidecar writer), api_utils.py 52 (banner) | b / not needed | – | tools only |
| 19 | **Other models** (bailing_moe_v3, muse_glimmer, interns2_mobius, dots3_note, kimi_k3, …) | ADD vs B but in U with 1-58 lines of drift | a | – | no |

---
## 2. Expert offload — deep dive (the re-implementation spec)

Paths below are relative to `~/src/tcclaviger-dev/vllm/`. Fork-unique lines: expert_plan.py 557, expert_placement.py 158, expert_activation_registry.py 115, expert.py 246, expert_topology.py 243, expert_lru.py 576, r4d_mxfp4_moe.py 554, r4d_route_router.py 252, shared_routed_fused_topk_router.py 83, config/offload.py +140, cudagraph_mem.py 73, plus ~60 lines of hooks in core files. **None of it is in upstream 0.29.** All of it is Python except the kernels.

**Prior art you can build on (important):** `github.com/davetha/r9700-lru-expert-cache` (Apache-2.0, cloned for this review at 3743f13, 2026-09-10) is where the fork's cache comes from. The fork credits it at `r4d_mxfp4_moe.py:210` and `:446`. It ships **HIP source** for the LRU kernels (`kernels/lru/r4d_lru.hip`: `r4d_lru_manage`, `r4d_lru_fused`, `r4d_lru_gather`), a numpy reference LRU, and tests: `test_lru.py` (the kernel matches the numpy LRU after every step), `test_graph_lru.py` (graph replay), `test_fused.py` and `test_victim_equiv.py`. The fork's `moe_lru_*` in r4d.so is that kernel with three additions: `mode`, a per-expert `bonus[E]` stamp offset (PIN = 1<<30 marks VRAM-only experts), and a `host_row[E]` indirection so the host slabs can be compacted. The gather also takes a list of slabs instead of exactly six. **So the device-side LRU is not really a private-kernel blocker.** The MoE grouped GEMM is: davetha also uses tcclaviger's closed `r4d.so` for `gemm_moe_mxfp4a8*` (see davetha `prebuilt/README.md`).

### 2.1 Pipeline and hook points (fork file:line → what 0.29 needs)

| # | when / where | fork hook | 0.29 equivalent / plugin route |
|---|---|---|---|
| H1 | CLI | `engine/arg_utils.py:549-558` (fields), `:1334-1362` (argparse), `:2618-2627` (OffloadConfig ctor) | No plugin CLI hook. Use `--additional-config '{"expert_offload":{...}}'` (`config/vllm.py:413` in 0.29) or env vars. |
| H2 | config | `config/offload.py:79-151` `ExpertOffloadConfig` (+ `PleNvmeConfig` :154-201), `OffloadConfig.expert/.ple` :223-228, validator :270-290 (rejects combining with `--cpu-offload-gb` or prefetch), `compute_hash` :326-338 (hashes only the on/off switch) | Plugin keeps its own dataclass. If the plugin config is in `additional_config`, check that the compile-cache hash is not invalidated by placement lists. Keep those out of `additional_config`, or put a stable subset there. |
| H3 | API server, end of `create_engine_config` | `arg_utils.py:2667-2674` → `expert_plan.plan_expert_offload(config)` writes placement into `config.offload_config.expert` (`expert_plan.py:558-597`) | The plan is deterministic from (vllm_config, checkpoint headers, sidecar, /proc/meminfo). **A plugin can run it in each worker** inside the patched `create_offloader`, so the API-server hook is not needed. The only catch is that the host bound uses MemAvailable, which differs per rank over time. Fix: pass an explicit `expert_offload_mem`, or compute on rank 0 and broadcast. |
| H4 | worker, before model build | `offloader/base.py:141-145` `create_offloader()` returns `ExpertOffloader(expert)` | 0.29 imports `create_offloader` **by name** into `v1/worker/gpu_model_runner.py:102,994` and `v1/worker/gpu/model_runner.py:52,348`. Monkeypatch those module attributes (general plugin; plugins load in `worker_base.py:269-271` before the runner exists). Alternatively patch `set_offloader`. |
| H5 | model build | `get_offloader().wrap_modules(...)` from `make_layers` | 0.29 `models/utils.py:858`. **0.29's signature is `wrap_modules(modules_generator, prefix="")`**, and interfaces.py:388 also calls it for towers (`supports_tower_offload`). The plugin's offloader must accept `prefix` and set `supports_tower_offload=False`. |
| H6 | weight load | `ExpertOffloader._hook_loaders` wraps each expert Param's `weight_loader` (`expert.py:171-205`), then splits a layer the moment every (expert, shard) event arrived (`:196-197`) | Pure Python, no core patch. |
| H7 | after load | `offloader.post_init()` (`expert.py:222-244`) | 0.29 calls `get_offloader().post_init()` (`gpu_model_runner.py:5541`, `gpu/model_runner.py:491`). |
| H8 | after load | `base_loader.py:86-101` `drop_checkpoint_page_cache` (fadvise DONTNEED on shards) | Optional. Can be done in `post_init`. |
| H9 | MoE forward | `routed_experts.py:1232-1247` in `RoutedExperts.forward_modular`: if `self._expert_offload` is set, build `ExpertLRUCache` lazily and `return lru.forward(...)` | **Core patch** (0.29 `routed_experts.py:1224-1259` is identical apart from this). Monkeypatch `RoutedExperts.forward_modular` with a wrapper. That is safe because it is a plain Python method called from the runner, and torch.compile sees it through the custom ops. |
| H10 | experts class | `compressed_tensors_moe_w4a4_mxfp4.py:59-89` picks `R4dMxfp4MoEExperts` on ROCm | **0.29 on ROCm falls to `MarlinExperts`** (CUDA-only) in `CompressedTensorsW4A4Mxfp4MoEMethod.__init__` (0.29 same file :46-70), and `process_weights_after_loading` warns for anything else. So the plugin must patch this `__init__` (or register a new quant method) to select its own experts class and skip the Marlin repack. This is needed even without offload. |
| H11 | FSE (shared expert fused as expert id E_r) | `models/qwen3_next.py:139-160` + `fused_moe/utils.py:96-130` `resolve_layer_fused_shared_expert_no_aiter`. Router `SharedRoutedFusedTopKRouter` (new, 83 L), wired in `router_factory.py:236-260` | 0.29 fuses only on the AITER path (`utils.py:55-90`, needs `rocm_aiter_ops.is_fusion_moe_shared_experts_enabled`). **The 0.29 runner already has the `_fse_fuse_gate` gate-column layout** (`runner/moe_runner.py:280,898`). The plugin either patches `qwen3_next.resolve_layer_fused_shared_expert` plus `router_factory.create_fused_moe_router` to add the non-aiter FSE router, or has its cache accept `shared_experts is not None` and run the shared MLP itself. The fork's cache refuses unfused shared experts (`expert_lru.py:295-299`). |
| H12 | router | `router_factory.py:236-246` → `R4dRouteRouter` (one launch of `r4d.moe_route_softmax_bf16`) | Optional perf. Monkeypatch the factory. |
| H13 | profile | `gpu_worker.py:663-668` `verify_expert_plan(self, profile_result)` (log only) | Optional. |
| H14 | KV sizing | `gpu_worker.py:688-700`: cap `available_kv_cache_memory_bytes` at `planned_kv_reserve_bytes` | Set `cache_config.kv_cache_memory_bytes` (0.29 `config/cache.py:232`) from the plan. That needs no core patch **if** the plan runs before the worker computes KV (it does in H4). The `kv_cache_memory_bytes` path skips profiling-based sizing. |
| H15 | graph estimate | `v1/worker/gpu/cudagraph_mem.py` `estimate_cudagraph_memory` (V2 runner, `BASE 256 MiB + 52 MiB/graph`) used by the plan (`expert_plan.py:469`) | Vendor it into the plugin (pure Python). |
| H16 | profiler RPC | `gpu_worker.py:1303,1320-1340` telemetry reset/report | Optional. |
| H17 | TunableOp | `warmup/tunableop_warmup.py:180` skips the sweep under expert offload | n/a on 0.29 |
| H18 | engine | `v1/engine/core.py:350-365` `_cold_compile_respawn_wanted`: with `--expert-precise-memory`, respawn workers after a cold AOT compile so memory is measured clean | Optional. Skip it. |

### 2.2 Plan sizing (`offloader/expert_plan.py:406-555`, placement in `expert_placement.py:79-165`)

Per rank:
```
budget      = total_VRAM * gpu_memory_utilization                               :431
non_expert  = Σ checkpoint tensors (safetensors headers only) not under
              `.experts.` of a `layers.N` (mtp.* stays resident), 1-D tensors
              replicated, others / TP                                          :195-237
              + shared_expert tensors unless FSE (then they count as expert #E_r)
              - ple_embedding when VLLM_PLE_CPU_OFFLOAD=1 (excluded)
kv_reserve  = 1.08 * (kv_bytes_per_token * max_model_len + mamba_pages)      :458-467
              registry per-token figure: TP2 8040 B/tok, TP4 11041             registry:124-130
graphs      = estimate_cudagraph_memory(vllm_config)                            :469
overhead    = registry: 7.6 GiB + 0.22 GiB per 1024 NBT above 4096 (default)
              5.25 GiB + ... under --expert-precise-memory                      registry:114-121
room        = budget - (non_expert + kv_reserve + graphs + overhead)           :478-484
expert_bytes= max(layer_bytes) // E   (E = E_r + 1 with FSE; e.g. 513)         :448-456
total_slots = room // expert_bytes                                              :485
host bound  = --expert-offload-mem GiB (total across TP*DP ranks), else
              MemAvailable - PLE (NVMe cache*1.1 or table bytes)
              - 7 GiB*ranks - 5 GiB - 5% margin                                 :487-507
host_cap_slots = (bound // ranks) // expert_bytes
```
Placement (`place_experts`, `expert_placement.py:79-165`):
- `global_order`: (layer, expert) pairs sorted by **per-layer routing share** (count / layer total), descending; ties broken by flat index so every rank computes the same order (:67-76).
- `pinned_target = floor(0.33 * total_slots)` (`PINNED_FRAC`, expert_plan.py:69). With no sidecar it is 0 and only the shared experts are pinned. VRAM-only ("pinned") = the L shared experts plus the first `pinned_target - L` pairs in global order. A hot layer can therefore get many more pinned experts than a cold one.
- If `host_needed = L*E - pinned > host_cap_slots`, the next-hottest pairs are also pinned (`pinned_extra`) until the host set fits. It fails only if that would need more than every slot (:121-136).
- `swap_total = total_slots - pinned`, **split evenly over the layers** (`base + (l < rem)`), capped at E per layer. Anything over the cap is `unused_slots` (:138-154).
- Per layer the output is `layer_slots[l] = |pinned_l| + swap_l` and `layer_pinned[l]` = expert ids hottest first, with the shared expert id `E_r` last (:146-148).
- Written back into config (:578-593): `layer_ids`, `layer_slots`, `layer_pinned`, `planned_cache_bytes`, `planned_kv_reserve_bytes`, `model_dir`, `profile_path`, and `expert_offload_mem` overwritten with `(host_per_rank + 1 MiB) * ranks / GiB`.
- An unregistered architecture disables offload with a warning (:566-574). Only `Qwen4ExpForConditionalGeneration` is registered (`expert_activation_registry.py:133-142`).

**Why PP is rejected (`expert_plan.py:415-416`):** the plan is model-global per rank. The census treats every layer as being on every rank and only divides by TP (:215). The placement spreads one per-rank slot budget and one global hotness order over *all* L layers. `layer_ids`/`layer_slots` are indexed by model layer position, and `ExpertTopology.register` raises for any layer not in the plan (`expert_topology.py:116-119`). Under PP each rank would own a subset of layers, so the budget, the global order and the host split would all be wrong. (`_num_weight_ranks` in `expert.py:45-66` does multiply by PP, but that only splits the host budget.) EP is also rejected at runtime: `expert_map is not None` raises (`expert_lru.py:295-299`). **TP works with zero cross-rank coordination** because routing is replicated across TP ranks and the manager is deterministic: misses are enumerated in expert-id order, and victims are chosen by a total order on `(stamp<<SLOT_BITS)|slot` with no atomics (davetha `kernels/lru/README.md` "Determinism"). Every rank therefore makes identical slot choices over its own 1/TP shard of each expert.

### 2.3 Sidecar `model-expertprofile.safetensors`

Written by `~/src/tcclaviger-dev/tools/expert_offload/convert_hot_profile.py` from `tools/expert_map` output or from a davetha `hot_profile.json`.
- Tensor `expert_routing_counts`: float32 `[num_layers, num_routed_experts]`. The layer axis is the decoder-layer index; non-MoE rows are zero. The fused shared slot is **not** included.
- `__metadata__`: `format="expert-routing-profile/1"`, `source`, `column`, `prompts`, `decode_tokens`, `prefill_tokens`, `num_layers`, `num_experts`.
- Read by `load_profile_counts` (`expert_plan.py:243-256`) and sliced to the expert layer ids (:509-520). A shape mismatch means planning without a profile.
- For a repo id it is fetched separately with `hf_hub_download` (:164-178).
- Used for three things: the pinned set, the per-layer swap warm start (`ExpertLRUCache._warm_swap`, `expert_lru.py:533-567`), and `bonus` (currently all-zero apart from PIN, because `RANK_SCALE=0.0`, expert_plan.py:71). Eviction is pure recency.

### 2.4 Load-time data movement (`offloader/expert.py`, `layers/fused_moe/expert_topology.py`)

1. **Stage** (`expert.py:142-169`). For each module that has `forward_modular` and `quant_method`:
   - Every direct Parameter with `dim >= 2` whose dim 0 equals the largest Param's dim 0 counts as per-expert (`per_expert_parameters`, :75-91).
   - The experts class must publish `offload_recipe` (:94-110). The Params are moved to CPU (`p.data = p.data.to("cpu")`) so the loader fills them in pageable RAM, and the layer gets `sub._expert_offload = self`.
2. **Count loads** (`expert.py:171-205`). The expected event count is `E * (2 if name.startswith("w13") else 1)` per Param. The key is `(expert_id, shard_id)` bound from the loader signature.
3. **Split** (`ExpertTopology.split_layer`, `expert_topology.py:166-237`), per layer as soon as it has fully loaded:
   - Pinned rows are `index_select`ed and moved to the GPU, then `recipe.prepare(rows)` repacks them (`r4d_mxfp4_moe.py:547-562`: fragment permute, scale transpose, row-max `wref`). The results are copied into `arena[slab][base : base+P]`.
   - The arena is allocated once, at the first split: `arena[slab] = empty((T_total_slots, *slab.shape[1:]))` (:141-162).
   - Non-pinned rows are compacted into an exact-size CPU tensor, pinned with `cudaHostRegister(ptr, n, cudaHostRegisterPortable|Mapped=3)`, falling back to `.pin_memory()` (:36-48). `get_accelerator_view_from_cpu_tensor` then makes it a UVA device tensor, which becomes the Param's new `.data`, marked `_vllm_is_uva_offloaded`.
   - Tables (`_allocate`, :141-152, all `[L, E]` int32 on the device unless noted):
     - `table` (expert→local slot, or -1)
     - `host_row` (expert→compacted host row, -1 if pinned)
     - `map_cold` (starts equal to `host_row`)
     - `bonus`
     - `slot_expert[T]` int32
     - `slot_stamp[T]` int64
   - Pinned experts get `table[e] = 0..P-1`, `slot_expert[base+i] = e` and `bonus = PIN (1<<30)`, so they never age out (:228-233).
4. **First forward (profile run)**. `ExpertLRUCache._build` (`expert_lru.py:432-531`):
   - `recipe.ensure_ready` calls `R4dMxfp4MoEExperts._maybe_repack` on the layer's own (now host-UVA) `w13_weight`/`w2_weight`. This **permutes the host tensors in place through UVA** (a one-time GPU kernel over PCIe) and builds `w1_ws_t/w1_wref/w2_ws_t/w2_wref` for the host rows. Note: `.contiguous()` on a UVA view allocates in **VRAM**, so the cold-side scale slabs are device-resident, about 1/16 of the weight bytes.
   - Constraints checked: `len(slabs) <= r4d.MOE_LRU_MAX_SLABS`, `S <= r4d.MOE_LRU_MAX_SLOTS` (davetha: 1024), `PIN == r4d.MOE_LRU_PIN`, host slab rows == `seg.host_rows`, slab bytes per expert a multiple of 16.
   - Per-layer scratch: `routed[E]` u8, `step[1]` i64, `miss[max(1,S),2]` i32 filled with -1, `n_miss[1]` i32.
   - `max_inserts = D = S - P` (swap slots), `max_distinct = D - max(1, D//8)`.
   - The fused path is used iff `R4D_LRU_FUSED=1`, `r4d.moe_lru_fused` exists, and `E <= MOE_LRU_MAX_EXPERTS_FUSED` (1024 as built), with the same bound on `max_inserts`.
   - Warm start (`_warm_swap`, :533-567): the first D profile-ordered non-pinned experts get slots `P..P+n-1` with `slot_stamp = n..1`; `table`/`map_cold` are updated, and one `moe_lru_gather` runs with `n_miss = n`.

**Recipe interface** (what any experts class must provide, `r4d_mxfp4_moe.py:520-606`):
- `raw_params = (w13_weight_packed, w2_weight_packed, w13_weight_scale, w2_weight_scale)`
- `slabs = (w13, w2, w1_ws_t, w1_wref, w2_ws_t, w2_wref)`
- `hot_params = {w13: w13_weight, w2: w2_weight}`
- `block_size = 16`
- methods: `bind_hot` (a context manager that swaps the layer's Params and the experts' scale attrs to the arena slices), `prepare`, `ensure_ready`, `host_slabs`, `apply`

### 2.5 Per-step algorithm (`ExpertLRUCache._step`, `expert_lru.py:306-374`)

Inputs: `x [M,H] bf16`, `topk_ids [M, top_k(+1 shared)] int32`, `topk_weights` fp32. `ids32 = topk_ids.flatten().int32`, `mk = numel` (a host-known int, so it is graph-safe).

1. **Manage.** Choose the call:
   - fused: `torch.ops.vllm.moe_lru_fused(ids32, table, map_cold, slot_expert, slot_stamp, routed, step, miss, n_miss, bonus, host_row, sorted_hot[L], eids_hot[NB], npad_hot[1], sorted_cold[L], eids_cold[NB], npad_cold[1], max_distinct, max_inserts, MODE_RANKED=1, block_size=16)`
   - otherwise: `moe_lru_manage(...same minus align bufs...)`

   `L = max(align_capacity(mk, E, 16), 16)` where `align_capacity(mk,E,bs) = mk + E*(bs-1)`, or `min(mk*bs, that)` if `mk < E`, and `NB = ceil(L/16)`. The align buffers are persistent and grow only eagerly; growing during capture raises (`_fused_views`, :392-419).

   Kernel semantics (davetha `r4d_lru.hip` + fork deltas):
   - Mark distinct routed experts (padding ids < 0 are ignored), then `++step`.
   - If `distinct > max_distinct`: no inserts (read-through).
   - Otherwise:
     - Refresh `slot_stamp[table[e]] = step` for every resident routed e.
     - Enumerate misses (routed and `table[e] < 0`) in expert-id order.
     - For each miss (capped at `max_inserts`), take a victim slot: the minimum `(stamp + bonus[slot_expert]) << SLOT_BITS | slot` among slots whose expert is **not routed this step** (pinned slots carry PIN and are never chosen).
     - Update `table[e_new] = s`, `map_cold[e_new] = -1`, `table[e_old] = -1`, `map_cold[e_old] = host_row[e_old]`, `slot_expert[s] = e_new`.
     - Append `(e_new, s)` to `miss`; set `n_miss`.
   - The fused variant also writes both `moe_align_block_size` outputs for the **post-install** tables:
     - hot: tokens bucketed by `table[e]` (slot id)
     - cold: tokens bucketed by `map_cold[e]` (host row)
     - -1 entries are ignored, the same as `moe_align_block_size(..., expert_map=table|map_cold, ignore_invalid_experts=True)`.
2. **Gather.** `torch.ops.vllm.moe_lru_gather(dsts=[arena slab slices for this layer], srcs=[host slabs], miss, n_miss, host_row, chunks=8, lanes=16)`. Grid is `(chunks, lanes)`, 16 B per lane; blocks past `n_miss` exit. It copies `srcs[i][host_row[e]]` → `dsts[i][slot]` for each of the 6 slabs. Measured at 25-28 GB/s on Gen4 (davetha).
3. **Narrow step** (`mk <= max_distinct`, so every routed expert is resident after step 1 — typical decode):
   - `qx, xs` = the router's fp8 quant (from the route bundle) or `ops.scaled_fp8_quant(x, per_token)`.
   - Attach `R4dRouteBundle(sorted_ids=hot sorted[:L_hot], expert_ids=hot eids, npad, qx, xs, block_size=16, num_align=S, quant_mode=1, num_tokens=M, hidden_ptr=x.data_ptr())` to `topk_ids`, where `L_hot` = align_capacity with `num_align = S`.
   - Non-fused alternative: `slot_ids = table[topk_ids]`.
   - Then, under `recipe.bind_hot(...)` (layer Params → arena slices), call the normal `layer.quant_method.apply(...)`. That runs `R4dMxfp4MoEExperts.apply` over **slots only**: GEMM1 `[mtk, 2I/TP]`, then silu·mul + fp8 quant (`clav_silu_quant.silu_mul_fp8`), then GEMM2 with the topk-weight epilogue, then `ops.moe_sum`.
   - **No host-memory GEMM runs on a narrow step.** (Our kernel-inventory note said "every step runs a GEMM over host tensors". That holds only for wide steps.)
4. **Wide step** (`mk > max_distinct`: prefill chunks, large batches). `recipe.apply` → `apply_hot_cold` (`r4d_mxfp4_moe.py:455-517`):
   - `ic1 = empty[mtk, N1]` bf16.
   - `r4d_mxfp4_moe_gemm_into(ic1, qx, xs, w1_hot=arena, hot ws_t/wref, *align_hot, None, mtk, top_k, *cfg1)`, then the same for `w1_cold` (host UVA, compacted rows), `self.w1_ws_t/wref` and `*align_cold`. Each call writes only the rows named in its `sorted_ids` and skips -1 blocks, so together they fill `ic1`.
   - silu·mul + quant, then the same pair for w2 with `tw = topk_weights.float().flatten()`, then `moe_sum`.
   - **Cold experts are read zero-copy over PCIe by the GEMM itself**, so link bandwidth bounds prefill.
5. Telemetry (`R4D_LRU_TELEMETRY=1`): a Triton one-program kernel accumulates `[steps, distinct routed, installs, read-through]` (:202-242).

Per layer per step that is 2 launches (fused + gather) plus the GEMM chain.

### 2.6 Kernel signatures as called from Python (private r4d.so; the davetha public analogue is in brackets)

| call | args (order) | notes |
|---|---|---|
| `r4d.moe_lru_manage` (`expert_lru.py:87-106`) | `topk_ids_ptr (int32[mk])`, `mk`, `E=table.numel()`, `S=slot_expert.numel()`, `max_distinct`, `max_inserts`, `mode(1=ranked)`, `bonus_ptr int32[E]`, `host_row_ptr int32[E]`, `table_ptr int32[E]`, `map_cold_ptr int32[E]`, `slot_expert_ptr int32[S]`, `slot_stamp_ptr int64[S]`, `routed_ptr uint8[E]`, `step_ptr int64[1]`, `miss_ptr int32[S,2]`, `n_miss_ptr int32[1]`, `stream` | [`r4d_lru_manage(topk_ids, mk, E, S, max_distinct, max_inserts, table, map_cold, slot_expert, slot_stamp, routed, step, miss, n_miss, stream)`, r4d_lru.hip:579] |
| `r4d.moe_lru_fused` (`:151-179`) | as manage, then `block_size`, `L=sorted_hot.numel()`, `NB=eids_hot.numel()`, `sorted_hot`, `eids_hot`, `npad_hot`, `sorted_cold`, `eids_cold`, `npad_cold` (all int32), `stream` | [`r4d_lru_fused`, :596]; E ≤ 1024 |
| `r4d.moe_lru_gather` (`:254-264`) | `list[dst_ptr]`, `list[src_ptr]`, `list[bytes_per_expert]` (each a multiple of 16), `miss_ptr`, `n_miss_ptr`, `host_row_ptr`, `chunks`, `lanes`, `stream` | [fixed 6 slabs, no host_row: `r4d_lru_gather(d0,s0,b0,…,d5,s5,b5, miss, n_miss, chunks, lanes, stream)`, :612] |
| constants | `r4d.MOE_LRU_PIN` (=1<<30), `MOE_LRU_MAX_SLABS`, `MOE_LRU_MAX_SLOTS`, `MOE_LRU_MAX_EXPERTS_FUSED` (1024) | |
| `lib().r4d_gemm_moe_mxfp4a8_nt_b16_m64` (ctypes, `r4d_lib.py:69-74`; called `r4d_mxfp4_moe.py:140-160,216-236`) | `a (fp8e4m3 [mtk or M, K])`, `ascale (fp32 [M,1] per-token)`, `wq (uint8 [E_or_S, N, K/2]`, fragment-permuted)`, `ws (uint8 [E, K/32, N] = scale^T)`, `wref (uint8 [E, N] = row max of scale)`, `c (bf16 [mtk, N])`, `sorted_ids int32[EM]`, `expert_ids int32[ceil(EM/16)]`, `num_post_pad int32[1]`, `topk_weights fp32[mtk] or 0`, then ints `EM=sorted_ids.numel()` (capacity, not the live length), `Mtk`, `top_k` (A-row = sorted_id / top_k; 1 for GEMM2), `K`, `N`, `WV`, `SK`, `NPW`, then `stream` | contract: `N%16==0`, `K % (SK*32)==0`, `WV*SK*32 <= 256`, `NPW ∈ {1,2,4}`, `WV*NPW*(SK-1)*1024 <= 64K`. `pick_cfg`: `SK=2` if K≥1024, `NPW=2` if SK=2 else 4, `WV=2`. A throw means process abort. `_mt1` variant: same args, used when top_k==1 and mtk ≤ 176 (`R4D_MOE_MT1`). Weight permute: `w[e].view(N/16,16,K/32,2,4).permute(0,2,3,1,4)` (`r4d_mxfp4_moe.py:104-113`). **Private.** Base it on public `gemm_mxfp4a8_nt_m64`. |
| `r4d.moe_route_softmax_bf16` (`r4d_route_router.py:200-228`) | `logits_ptr` (bf16 [M, ≥E+n], row stride `ld`), `hidden_ptr` (bf16 [M,H] contiguous, H%256==0, ≤4096), `topk_w` fp32[M,k+n], `topk_ids` int32[M,k+n], `sorted_ids` int32[cap], `expert_ids` int32[ceil(cap/16)], `npad` int32[1], `qx` fp8[M,H], `xs` fp32[M,1], `ticket` int32[1] (persistent zero), `M`, `E`, `n_shared`, `top_k`, `H`, `ld`, `block=16`, `cap`, `num_align=E+n`, `renorm`, `1`, `0`, `shared_weight(float)`, `1e-10(eps)`, `1`, `1`, `stream` | E ∈ {128,256,512}, k+n ≤ 32, M ≤ `R4D_ROUTE_MAX_M` (256). Optional; the stock chain is `fused_topk` + `moe_align_block_size` + `scaled_fp8_quant`. |
| `torch.ops.clav_silu_quant.silu_mul_fp8(q2 fp8[mtk, N1/2], s2 fp32[mtk,1], ic1 bf16[mtk, N1])` | `r4d_mxfp4_moe.py:437` | Optional. Fallback: `silu_and_mul` + `scaled_fp8_quant`. |

### 2.7 What a plugin must re-create

1. **Required.** An MXFP4-weight MoE experts class for gfx1201 that reads packed weights directly and supports "write-into with an alignment over a subset" (the `*_into` semantics and -1 block skipping). Without it, 0.29 ROCm has only emulation (full per-forward bf16 dequant of 512 experts, which is unusable) and cannot pick Marlin.
2. The LRU kernels: port davetha's `r4d_lru.hip` and add `bonus` + `host_row` + an N-slab gather, or keep `map_cold` = expert id and do not compact host slabs (davetha's original form).
3. The Python offloader (plan, placement, topology, cache), which is all portable. Patch points are H4, H9, H10, H11 (and H14 through config).

---

## 3. PLE offload and QSA (lighter)

### 3.1 PLE offload

**Upstream status: U has no PLE offload.** U's `qwen4_exp/{amd,nvidia}/ple_layer.py` keeps the n-gram table GPU-resident as a TP-sharded bf16 `PLEVocabParallelEmbedding`. What U *does* have is MRV2 `amd/model_state.py:20-120`, which provides the GPU tensors `ngram_context [max_reqs, ngram-1]` and `ple_query_start_loc`. The fork's MRV2 connector binds to exactly those.

**The table.** One per `config.ple_layer_ids`.
- Hashing is multiplicative-XOR over the previous n-1 tokens, per n-gram head, modulo a per-head prime vocabulary, with cumulative row offsets (F `amd/ple_layer.py:200-212,415-460`).
- Row formats (F `common/ple.py`):
  - bf16
  - fp8 with one global scale
  - fused MXFP4
  - fused int6-g32: 6-bit codes packed four per 24 bits, then an fp16 scale per 32 codes; for hd=128 a row is 104 B
- The whole table is 47.7 GiB on Flash-Next. The docs say 160-byte rows.

**Processes** (F `v1/ple_offload/*`):
- Rank0/DP0 spawns one **CPU-only PLE process**.
  - Spawned at `multiproc_executor.py:673-684` and `gpu_worker.py:255-296`, before `load_model`; it waits for readiness afterwards.
  - The process never touches HIP: `_CudaToMeta` is active and `torch.cuda._lazy_init` is fenced.
  - It builds the model on meta, materializes only the `PleOffloadLayer` subtrees on the CPU (full vocab, TP1), and mlocks them, or builds the NVMe table.
- On the GPU side each PLE layer is a placeholder. `PleOffloadLayer.__init_subclass__` skips allocation.

**Per step:**
1. TP0 snapshots `input_ids`, `query_start_loc` and `ngram_context` into a /dev/shm staging ring (int32 rows padded to 64 B).
   - Ring depth: `max_num_seqs * max(1, num_spec) + 1`.
   - MRV2 uses non-blocking D2H plus an event.
2. TP0 sends a msgpack `PleOffloadRequest{dp_rank, num_tokens, num_reqs, slot}` over ZMQ PUSH.
3. The CPU process computes the rows into ROCr host slots (`ple_rocr.host_alloc`; 4 slots per layer).
4. For each TP rank it SDMA-copies the rows (`copy_h2d`) into that rank's IPC-attached output buffer (`ipc_export` of the whole allocation; **expandable_segments must be off**). A second SDMA then writes flag=1, depending on the rows' signal.
5. The GPU graph contains `ple_hip.wait_value32_eq(stream, flag, 1)`, wrapped as the custom op `vllm::ple_offload_wait(flag, buf, hidden)`, which needs the defunctionalize special case in `fix_functionalization.py:174-180`.
6. After the forward, `write_value32(flag=0)` runs on the compute stream.

**Properties:**
- The rows are full width and identical on every TP rank; there is no vocab sharding.
- Dummy or capture runs signal zeros locally.
- Spec-decode placeholder ids are negative and get clamped to 0.

**NVMe table** (F `nvme_table.py` + `src/plehip/csrc/ple_nvme.cpp`):
- On disk: one `.bin` file per layer, plus a manifest and a flock owner.
- In RAM: a clock-LRU arena of `--ple-cache-gb` rows, mlocked and MADV_HUGEPAGE.
- Misses are O_DIRECT aligned-span reads through a raw-syscall io_uring (queue depth `--ple-nvme-queue-depth`, default 512).
- `warm_start` fills the arena sequentially.
- CPU hashing and lookup go through **`libqnf_ple_cpu.so`** (no source): `qnf_ple_prepare_r2`, `qnf_ple_hash_r2`, `qnf_ple_lookup_r2`. The torch chain is a proven-equal fallback for the non-NVMe fused case.

**Plugin route on U.** This is all Python: about 2.2k lines of offload code plus 400 lines of layer code. The compiled pieces are rebuildable (`src/plehip`, 906 lines of C++/HIP).
- Monkeypatch `Worker.load_model` / `shutdown` for the spawn.
- Hook the MRV2 model state's `prepare_inputs` / `prepare_dummy_inputs`: launch the request before the forward, and signal on dummy runs.
- The release can be a captured custom op at the end of the forward.
- Avoid the `fix_functionalization` patch by making the wait op write a fresh `out`.
- Re-register the arch with a subclassed `Qwen4ExpNGramEmbedding`.
- Use environment variables instead of the `--ple-*` flags.
- Replace `libqnf_ple_cpu.so` with the torch chain or a small C/numba kernel.

### 3.2 QSA and sparse attention

**What QSA is in U (a).** Each full-attention layer has a weight-free indexer: `index_qk_proj` produces Hn query heads and one key head (head dim 128).
- Raw keys go into a per-request ring cache.
- When a group of `ratio` tokens closes, it is pooled, normalized, rotated with RoPE, and stored as one compressed key.
- Score (`U ops/qsa.py:591`): `logit = Σ_h relu(q_h·kc)/√d` over visible blocks, where `visible = min((pos+1)//ratio, seq//ratio)`.
- Select (`:729`): the top `budget/ratio` blocks are expanded to token indices, plus the open-group tail. Output is int32 `[rows, budget+ratio-1]`.
- Attend (`:822`): Triton split-K sparse GQA over the packed K|V paged cache (head dim 256). Bf16 only.
- MTP step 0 selects the indices and later draft steps reuse them.

**Fork deltas.**
- The dispatch order is **r4d → clav_attn → Triton** for score, select and attend, and **r4d → eager** for the index prep. Gates are `R4D_SCORE`, `R4D_SELECT`, `R4D_QSA`, `R4D_QSA_PREP`, `CLAV_ATTN`, `CLAV_GQN`, all probed once.
- FP8 KV for the QSA owner (`amd/qsa.py`):
  - `kv_cache_dtype` is masked for the FlashAttention base-class check.
  - There is a persistent `_v_descale_buf`.
  - Calibrated q/k/v scales are loaded.
  - **With fp8 and no HIP kernel, it raises** (`ops/qsa.py:1419`).
- The sigmoid gate is fused into the transaction op. That op's schema changes, so a plugin must use a new op name.
- `sel_len` (the live index width) is written by `topk_expand` and consumed by the attend kernel. It is compacted in MTP.
- `QSASharedScratch` gives one index buffer to all layers.
- `update_draft_decode_metadata`: U already has the protocol (`v1/attention/backend.py:599,724`), so a builder subclass is enough (b).

**Plugin route:**
- Replace the attributes of `vllm.models.qwen4_exp.amd.ops.qsa` (`qsa_mqa_paged`, `qsa_select_paged_tokens`, `qsa_sparse_paged_attention`). They are imported lazily, so the swap takes effect.
- Everything else needs an overridden model class: ship patched `amd/{qsa,indexer_qsa,model,mtp,hyperconnection,ops/hc}.py` and register it with `ModelRegistry.register_model`.
- The CSA+linear KV grouping (F `kv_cache_utils.py:1391-2302`) may be a core requirement. First check whether U's upstream QSA already groups correctly; it is U's own model.

---

## 4. Private kernel API to re-implement

Appendix A gives the argument orders not covered in this section.

**Conventions:**
- r4d is called through the pybind module (`import_r4d()`, RTLD_DEEPBIND, `R4D_LIB=/app/r4dhip/r4d.so`) or ctypes (`r4d_lib.lib()`).
- Every tensor is passed as `data_ptr()` int. The stream comes last.
- **A contract violation is `std::terminate`**, so the Python asserts before each call are part of the API.

### 4.1 Private, called, no source

These are the real work items.

| symbol | shapes / dtypes (from the calling code) | call site | fallback | priority |
|---|---|---|---|---|
| `r4d_gemm_moe_mxfp4a8_nt_b16_m64` (+`_mt1`) | see §2.6: fp8 A per-token; uint8 packed W `[E,N,K/2]` fragment-permuted; E8M0 `ws_t [E,K/32,N]`; `wref [E,N]`; bf16 C `[mtk,N]` rows only per sorted_ids; moe_align block 16; `-1` blocks skipped | r4d_mxfp4_moe.py:140,216 | OCP emulation (unusable at E=512) | **P0**. Base: public `gemm_mxfp4a8_nt_m64` in `~/src/libr4d` |
| `moe_lru_{manage,fused,gather}` + `MOE_LRU_*` | §2.6 | expert_lru.py:87,151,254 | none once offload is on | **P0**. Port davetha `r4d_lru.hip` and add bonus/host_row/N-slab |
| `attn_sparse_h256_{bf16kv,fp8kv}` + `_scratch_bytes(rows,heads,kv_heads,topk,0)` | q bf16 `[rows,H,256]`; packed K\|V cache (token stride 2·hd, V at +hd), bf16 or fp8 bytes; indices int32 `[rows,topk]` + `sel_len`; block_table int32; token_to_req; gate bf16 (fused sigmoid·out); v_descale fp32 `[reqs,hkv]`; scale = 256^-0.5·k_scale | ops/qsa.py:1320 | Triton, **bf16 only** | **P0 if fp8 KV**, else P2 (U Triton works for bf16 KV) |
| `attn_sparse_score_h128_bf16` | q bf16 `[rows,≤16,128]`; k bf16 `[pages,page,1,128]`; page_table/token_to_req/seq_lens int32; qpos int64; out logits fp32 `[rows,cols]` (tail unwritten) + visible int32; strides %4 | ops/qsa.py:887 | Triton (a) | P2 |
| `attn_sparse_topk_expand` | logits fp32; out int32 `[rows, topk+ratio-1]` + `sel_len`; `block_topk=topk//ratio`; rows chunked ≤128 MiB | ops/qsa.py:1115 | `top_k_per_row_decode` + Triton expand (a) | P2 |
| `qsa_index_prep_h128_bf16` | 41 args: qk bf16 `[T,2·Hn·128]`, int64 positions (mrope `[3,T]`), cos_sin, norms, q_out, comp/raw caches + slot maps + raw block table + t2r + qsl + logical positions, mrope sections, strides | indexer_qsa.py:329 | eager chain (a) | P3 |
| `moe_route_softmax_bf16` | §2.6 | r4d_route_router.py:200 | fused_topk + align + quant | P2 |
| `ple_dequant_i6g32_f16` | fused int6 rows → bf16 `[rows,hd]` | ple_int6.py:62 | torch, bit-exact | P3 |
| `ag_oneshot_{4,8}rank_exact`, `select("allgather")`, `ar_twoshot_8rank_ti8` | TP4/8 only | r4d_all_reduce.py:167,416 | RCCL | n/a at TP2 |
| `dflash2_prep_h128_*` | DFlash only | dflash/speculator.py:224 | Python | n/a |
| `libfp8hip_gemm.so`: `fp8hip_table_make(N,K,ws,dev,int* m_buckets,n)`, `fp8hip_gemm_w8a8_launch_shape(qx fp8[M,K], w_shuf fp8[N,K] (view(N/16,16,K/32,2,2,8).permute(0,2,4,1,3,5)), xs fp32[M,K/128], ws fp32[N/128,K/128], y bf16[M,N], M,N,K, stream)` | fp8hip.py:125,170 | rdna4 Triton (source present); davetha measured **no loss** vs the Triton path | P3 |
| `clav_attn_C`: `unified_attn_16(q, kv[blocks,hkv,bs,2hd], bt, cu_q, ctx, out, segm_out/max/sum, scale, causal, window, max_q, block_m, tile, nwarps, segments, partition, qslice, k/v/q_descale)` + `qsa_mqa_score_16`, `qsa_sparse_attn_16`, `qsa_sparse_occupancy` | clav_attn/backend.py:1492; ops/qsa.py:937,1399,206 | Triton | P2 |
| `gdn_hip_C`: `gdn_decode_r`, `gdn_prefill_r2`, `gdn_verify_r` (x = pre-conv mixed_qkv, conv_w fp32, conv_state DS, a, b, A_log, dt_bias, ssm_state, indices int64, counters int32[4096], scale, 1, 1) → `[T,Hv,128]` | gdn_hip/forward_core.py:104-192 | fla-Triton (a) | P2 |
| `clav_silu_quant.silu_mul_fp8(q fp8[mtk,N/2], s fp32[mtk,1], in bf16[mtk,N])` | r4d_mxfp4_moe.py:437 | silu_and_mul + scaled_fp8_quant | P3 |
| `clav_conv1d.{fwd,update}`, `clav_state_copy.{preprocess_align,precopy,postprocess}`, `clav_memcpy.batch_memcpy`, `clav_reshape_cache.rc_flash{,_go}`, `clav_pleconv.fwd` | argument lists mirror the Triton kernels they replace (causal_conv1d.py:765,1325; mamba_utils.py:525,1122,1199,1272,669; triton_reshape_and_cache_flash.py:430-436; ple_layer.py:1312) | same files | Triton / torch | P3 |
| `libqnf_ple_cpu.so`: `qnf_ple_{prepare,hash,lookup}_r2` | ple_cpu_native.py:47-98 | torch chain | P1 if PLE offload |

### 4.2 Public, or source available (build it; nothing to re-implement)

- **libr4d v0.5.0** (`~/src/libr4d`):
  - `gemm_mxfp4a8_nt_m64` (dense MXFP4 linear, M ≤ 64)
  - `gemm_w4a16_nt_m64` (draft lm_head)
  - `ar_oneshot_2rank_{exact,wht6}` + `ar_ipc_*` + `select`/`explain`
  - `attn_{prefill,decode}_h256_gqa6_*`
  - `dflash_conv_t2_g16_bf16`
  - `gdn_conv_*`
- **In this repo:** `clav_hc` (`src/mischip/q4hc`) and `ple_{hip,rocr,nvme}` (`src/plehip`).
- **Pure Triton in F:** rdna4 FP8 block matmul, mxfp4_16 kernels.
- **davetha** (`kernels/lru/r4d_lru.hip`, Apache-2.0): LRU manage/fused/gather.

---

## 5. Porting plan: plugin for stock vLLM 0.29 + ROCm 10

The whole port is one `vllm.general_plugins` package (plus an optional platform plugin for the communicator). Ordering is by dependency and value.

1. **Skeleton and loaders.**
   - Plugin package with a `register()` that no-ops off gfx12.
   - Port `r4d_lib.py`, pointed at a libr4d we build ourselves (public v0.5.0 plus our additions, the same ABI style).
   - Config through `--additional-config` and env vars, because U has no plugin CLI hook.
   - Port `_force_qwen4_exp_env`.
2. **MXFP4 MoE experts, the P0 kernel.**
   - Write the grouped MoE GEMM (`gemm_moe_mxfp4a8` equivalent) on top of public `gemm_mxfp4a8_nt_m64`. Requirements:
     - moe_align block 16
     - writes only the listed rows (the `*_into` semantics)
     - skips `-1` blocks
     - optional fused topk-weight epilogue
   - Port `R4dMxfp4MoEExperts` (`r4d_mxfp4_moe.py`, verbatim apart from the loader).
   - Patch `CompressedTensorsW4A4Mxfp4MoEMethod.__init__` and `process_weights_after_loading` (U same file :46-70, ~:219) to pick it on ROCm and skip the Marlin repack. Without this, U cannot serve the checkpoint on gfx1201 at usable speed.
   - Validate against OCP emulation.
3. **Fused shared expert on non-aiter.**
   - Monkeypatch `vllm.model_executor.models.qwen3_next.resolve_layer_fused_shared_expert`: fall back to F's `_no_aiter` logic.
   - Monkeypatch `router_factory.create_fused_moe_router` → `SharedRoutedFusedTopKRouter` (port 83 lines).
   - The U runner already builds the gate-column layout (`_fse_fuse_gate`).
   - Alternative: make the cache tolerate unfused shared experts and run the shared MLP outside it.
4. **Expert offload, Python side.**
   - Vendor `expert_placement.py`, `expert_plan.py`, `expert_activation_registry.py`, `expert_topology.py`, `expert.py`, `expert_lru.py` and `cudagraph_mem.py`.
   - Hook points:
     - (i) Patch `create_offloader` **in both** `vllm.v1.worker.gpu_model_runner` and `vllm.v1.worker.gpu.model_runner` (it is imported by name). Run `build_plan` there: every rank gets the same plan if `expert_offload_mem` is explicit.
     - (ii) The offloader's `wrap_modules(gen, prefix="")` and `supports_tower_offload=False` (U signature).
     - (iii) Monkeypatch `RoutedExperts.forward_modular` with the `_expert_offload` / `_expert_lru` wrapper (F `routed_experts.py:1232-1247`).
     - (iv) Set `cache_config.kv_cache_memory_bytes = plan.kv_reserve` in place of F's gpu_worker KV cap.
   - Drop the engine-core respawn and the TunableOp interplay.
   - Re-fit the `expert_activation_registry` overhead model on U (log `verify_expert_plan` numbers).
   - Tooling: port `convert_hot_profile.py` and generate `model-expertprofile.safetensors`. Get it from the image's checkpoint, or capture routing with davetha `tools/routecap`.
5. **Expert LRU kernels.**
   - Build davetha's `r4d_lru.hip` (Apache-2.0, tests included).
   - Extend it:
     - `bonus[E]` added to the stamp in the victim key, PIN excludes a slot
     - `host_row[E]` indirection in gather and in the cold alignment
     - an N-slab pointer list in gather
     - `mode`
   - Or keep davetha's semantics (`map_cold` = expert id, uncompacted host slabs) and adapt `expert_topology.split_layer`.
   - Re-run `test_lru.py`, `test_graph_lru.py`, `test_fused.py` and `test_victim_equiv.py`.
   - Measure narrow-step hit rate with `R4D_LRU_TELEMETRY`.
   - Measure H2D on the .100 PLX box: its measured 14 GB/s is well below the 25-28 GB/s gather rate measured on Gen4.
6. **QSA with fp8 KV.** Two options:
   - (a) Serve with `--kv-cache-dtype auto` (bf16) first. U's Triton QSA works, so there are no private kernels on this path.
   - (b) Then write `attn_sparse_h256_fp8kv` (sparse split-K GQA with a fused gate and v_descale) and port the fp8 owner/`sel_len`/fused-gate model deltas through a `ModelRegistry` override, with a new custom-op name.
   - Score, select and index-prep kernels are P2/P3 perf work.
   - Verify U's KV grouping for the QSA hybrid before porting F's `kv_cache_utils` CSA path.
7. **All-reduce.** A platform-plugin `CudaCommunicator` subclass with the r4d TP2 one-shot exact/wht6 AR. These kernels are all public libr4d; port `r4d_all_reduce.py`.
8. **Attention and GDN.**
   - Install `clav_attn` and `gdn_hip` as-is if the private wheels are acceptable, with two fixes:
     - clav: insert ahead of `ROCM_ATTN`, not `TRITON_ATTN`
     - gdn_hip: `attn_metadata.get(prefix)`
   - Otherwise stay on U's Triton and fla paths.
   - Do not port F's attention autotune (a core patch) until measured.
9. **PLE offload**, only if RAM or VRAM needs it. In U the table is on the GPU and TP-sharded, which is about 24 GiB per rank at TP2 for 47.7 GiB and competes with the expert cache.
   - Build `src/plehip`.
   - Port `v1/ple_offload` plus `ple_offload_layer.py` and `gpu_stream_ops.py`.
   - Hooks: `Worker.load_model`/`shutdown` spawn, MRV2 model-state `prepare_inputs` launch, a captured release op.
   - Replace `libqnf_ple_cpu.so` with the torch chain.
10. **Dense FP8.** The rdna4 Triton kernel via `_POSSIBLE_FP8_BLOCK_KERNELS[ROCM].insert(0, …)`, plus the CT block-FP8 bypass if TP2 shards are block-unaligned (check the checkpoint's `quantization_config`). Skip fp8hip; davetha measured it as a wash.
11. **Optional perf.** In rough order:
    - `moe_route_softmax_bf16`
    - `silu_mul_fp8`
    - `clav_hc`-style fused hyper-connection mix (source available)
    - V2 `FusedDraftStepUpdate` (core-deep)
    - degen-loop stop via a `scheduler.check_stop` wrap
    - TunableOp sweep via a loader wrap
    - Skip entirely: DRY, DFlash, radiance, and F's memory-profiler rewrite.

**Plugin constraints to watch on U:**
- The U MoE runner calls `forward_modular` inside torch.compile'd regions through custom ops. Keep every new GPU call a `direct_register_custom_op` with fake impls, as F does.
- Graph capture needs persistent buffers for the alignment, `v_descale` and `sel_len`, and the widest batch must run eagerly first (`expert_lru.py:398-402`).
- Expandable segments must stay off for PLE IPC.
- Pinned host memory needs `ulimit -l unlimited`.

---

## Appendix A. Argument orders not given above

**r4d all-reduce** (public; `r4d_all_reduce.py`)

- `ar_ipc_alloc(nbytes, fine) -> (ptr, handle)`, then `ar_ipc_open(handle)`.
- Scratch size: `nslots * max_bytes`. Flag buffer size: `maxb * ws * 4`.
- Calls:

| call | arguments in order | used when |
|---|---|---|
| `ar_oneshot_2rank_exact` (:350) | peer_scr, my_scr, peer_flg, my_flg, seq (int32 [AR_MAX_BLOCKS], device-incremented), slot16, inp, out, n, dtype (0 bf16, 1 f16, 2 f32), stream, nblocks, nthreads=1024, drain=3, acq=0 | default |
| `ar_oneshot_2rank_wht6` (:341) | peer_scr, my_scr, peer_flg, my_flg, seq, locpk (uint8 [max/2 + max/32 + 4096]), slot_bytes, scale_off=max/2, inp, out, n, dtype, stream, nb<=48, 1024, drain, acq | n*es >= 128 KiB, n % AR_WHT6_GROUP == 0, and R4D_AR_QUANT=1 |

- `should_custom_ar` requires all of: bf16/f16/f32, nbytes % 16 == 0, contiguous, at most 48 MiB at TP2.

**r4d dense attention** (public; `r4d_attn.py:372`; default off)

- Called once per run of equal q_len.
- Arguments in order:
  - pointers: q + tok*row, kv, bt + req*maxblk*4, seqlens + req*4, out
  - descales: k_descale or 0, v_descale or 0
  - scratch
  - ints: num_seqs, q_len, hq, hkv, 256, 16, max_blocks, kv.stride(0), kv.stride(1)
  - trailing: scale, splits=0, max_ctx, stream
- Layout: kv is [blocks, hkv, 16, 2*hd] (HND).
- Scratch size comes from `attn_decode_h256_gqa6_scratch_bytes(num_seqs, maxq, hq, hkv, hd, max_ctx, 0)`.

**gdn_hip** (private; `gdn_hip/forward_core.py`)

- `gdn_prefill_r2`: x, conv_w (fp32 [dim,width]), None, conv_state (kv_cache[0], DS layout), a, b, A_log, dt_bias, cu_seqlens, state_indices (int64), has_initial_state (uint8), ssm_state (kv_cache[1]), counters (int32 [4096]), scale = 128^-0.5, 1, 1.
- `gdn_decode_r`: the same without cu_seqlens and has_initial_state.
- `gdn_verify_r`: spec_query_start_loc, spec_state_indices[:n_spec] (2-D) and num_accepted_tokens[:n_spec] in place of those.
- Batch routing:

| batch | kernels |
|---|---|
| pure spec | verify |
| spec + prefill | verify + prefill_r2, merged with index_copy_ |
| decode + prefill | decode_r on the first num_decode_tokens, then prefill_r2 |

**clav_hc.fused** (source in `src/mischip/q4hc`; `ops/hc.py:465`)

- Arguments in order:
  - outputs: block_input [N, H], out [N, D] (optional), inj_out [N, hc] (optional)
  - workspaces: xn_ws bf16 [16*D], lora_ws bf16 [16*R], part_ws fp32 [(D/512)*J*16], barrier int32 [2]
  - inputs: residual [N, D], block_output (optional), injection (optional), norm_w, w_down [J, D], w_up [., R]
  - trailing: eps, hc, prof
- Only used when N <= CLAV_HC_MAX_M (8).

**fp8hip plan struct** (`fp8hip.py:51`)

- 9 ints: arm, geom, dbl, group_m, mrep, nrep, grid_x, grid_y, m_bucket.
- m_buckets are the cudagraph sizes plus max_num_batched_tokens.
