# Progress log

## 2026-09-18 (overnight session)

### Findings
- **P2P gives ~nothing at TP2 on Flash-Next** (clean A/B, same flags): single 69.7 vs 69.5 tok/s, agg within noise.
  Expert offload over PCIe dominates, not all-reduce. (bench/ab-p2p.log, bench/ab-nop2p.log)
- **Host->GPU link on the .100 box is PCIe Gen3** behind PLX PEX 8747 switches: 14 GB/s per GPU, 27.8 GB/s both.
  A Gen4 CPU-direct box roughly doubles it.
- **Kernel ablation of tcclaviger/vllm:dev** (ablation/run-ablation.sh; results ablation/summary.tsv):
  see table below. `clav_attn` is dead weight for this model (Triton attention equal); `fp8hip`, QSA r4d kernels,
  GDN HIP and the clav glue kernels each matter 5-13%.
- **Public prior art found**: davetha/r9700-lru-expert-cache (Apache-2.0: device-side LRU expert cache kernels +
  tests, 93-139 tok/s single-stream on the same 2x R9700 + model) and StillDeadcode/libr4d (gfx1201 GEMMs, AR,
  GDN; no license yet). The only truly private hard kernels are tcclaviger's MoE grouped GEMM and QSA sparse
  attention. Notes: notes/kernel-inventory.md, notes/ggz14-libr4d-review.md, notes/tcclaviger-fork-analysis.md,
  notes/upstream-0.29-rocm10.md.
- **Stock target exists**: `vllm/vllm-openai-rocm:nightly-rocm100-<sha>` = vLLM main + torch 2.12 rocm10 +
  triton 3.8 + hipcc, gfx1201 wheels. Pinned dee37d89.

### Built (all in this repo, compiled for gfx1201 with the image's hipcc, hostcall-free)
- `kernels/r9k_moe_mxfp4a8.hip` — grouped MXFP4 x FP8 MoE GEMM (derived from libr4d's dense kernel: fragment-order
  weights, folded-exponent unpack, fp8 WMMA, in-block split-K), MoE routing via vLLM's moe_align_block_size
  output, router weight folded in the epilogue, MT=1/2/4 M-tiles per block for prefill. + per-row fp8 quant.
- `kernels/r9k_ple.hip` — PLE int6 fused-row gather+dequant (reads the table from pinned host memory via UVA).
- `kernels/third_party/davetha/r4d_lru.hip` — vendored LRU expert cache kernels (Apache-2.0).
- `r9700_vllm/` plugin (vllm.general_plugins entry point), all hooks ROCm-gfx12-only and idempotent:
  - `moe/ct_mxfp4.py` — compressed-tensors MXFP4 MoE -> our experts (stock picks CUDA-only Marlin); in-place,
    UVA-aware weight re-layout.
  - `moe/experts.py` — `R9700Mxfp4Experts` (vLLM modular experts class).
  - `moe/cache.py` — per-layer VRAM slot arena + UVA host store + LRU (davetha kernels), warm start from the
    checkpoint's routing profile; two-pass hot/cold GEMM. `R9K_EXPERT_CACHE_GB` per rank.
  - `ple/int6.py` — int6 PLE table in pinned host memory (UVA) instead of stock's bf16-in-VRAM (51 GB/rank).
  - `linear/mxfp4.py` — dense MXFP4 linears (shared experts) on our kernel instead of per-call dequant emulation.
  - `spec/mtp_rocm.py` — MTP k>1 on ROCm (QSA metadata onto the spec-decode allowlist, cf. PR #55292).
- `docker/Dockerfile` (stock nightly + plugin), `serve/serve.sh` (was serve-stock-fn.sh), `bench/bringup-tests.sh`,
  `tests/` (MoE GEMM vs exact dequant, PLE vs reference, cache bit-identity + LRU invariants).

### Stock vLLM (ROCm 10 nightly) + plugin: bring-up and tuning log (2026-09-18 morning)
All unit tests pass on gfx1201 (MoE GEMM, PLE, cache bit-identity + LRU invariants, fp8 GEMM exactness).
Bring-up fixes needed (all in the plugin / launcher, no vLLM source patches):
- hostcall-free RCCL rebuilt IN the nightly image + hostcall-patched vLLM _rocm_C/_C_stable_libtorch (emulated switch)
- CT: fp8 groups inherit global `mxfp4-pack-quantized` format -> set float-quantized; `weight_scale_inv` -> `weight_scale`;
  drop fork q/k/v scales + fork MTP q4 head; MTP MLP fp8 block-128 can't shard 640/2 -> re-quantized to MXFP4 at load
- torch pinned memory rounds to 2^k (OOM) -> hipHostRegister exact-size pinning (PLE, cache, stock UVA offload)
- `--language-model-only` (ViT SDPA hipErrorInvalidValue on gfx1201/torch 2.12)
- stock CT W4A4 asks kMxfp4Dynamic -> our dense kernel must accept it (else per-call emulation)
- **HSA_ENABLE_IPC_MODE_LEGACY=0** (nightly sets 1 -> hipIpcGetMemHandle fails -> no P2P)
- **GPU_MAX_HW_QUEUES=1**: cross-stream waits inside HIP graphs were stalling ~50x/step (129 -> 25 ms ITL at bs1)
- libr4d 2-rank P2P all-reduce instead of RCCL (69 us -> ~3 us per call); vLLM's own custom AR = garbage on gfx1201

| config | single | @4 | @8 | @16 | pf 2k | GSM8K | needle |
|---|---|---|---|---|---|---|---|
| tcclaviger:dev (reference) | 82.5 | 81.1 | 146.4 | 145.6 | 2542 | 97/100 | 3/3 |
| stock+plugin eager, no cache, no MTP | ~5 | | | | | 39/40 | 3/3 |
| + graphs, cache 200, HWQ=1 | 38.3 | 114.6 | 120.0 | 115.4 | 333 | | |
| + MTP-3 (accept 2.7-2.8) | 57.3 | 98.8 | 135.4 | 139.5 | 503 | | |
| + fp8 LM heads + fp8 HC linears (cache 180) | 62.1 | 100.3 | 134.0 | 127.8 | 480 | 95/100 | 3/3 |
| + libr4d all-reduce | **68.0** | 105.8 | **139.5** | **139.6** | 487 | | |

| **VM100 256 GB RAM: all experts host-resident, LRU cache 270 slots on all 48 layers** | **75.3** | 120.5 | **193.8** | **194.7** | | **99/100** | **3/3** |

| same + Fable review P0 guards (commit 0f8af02) | 78.1 | | 203.4 | 187.9 | | 98/100 | 3/3 |

Prefill investigation (2026-09-18 pm): warm prefill ~1,550 tok/s @2k, ~2,050 @8k-31k. Profile of a 31k prompt:
43% of GPU time is expert gathers at ~14.5 GB/s = the Gen3 link. Every 4096-token chunk routes to nearly all 512
experts/layer, so all ~240 non-resident experts cross PCIe once per chunk. Staging cold experts into VRAM
(R9K_STAGE_COLD=1, bit-exact, tested) removes duplicate reads but gives no measurable prefill gain; bigger chunks do
(NBT=8192 + 240 slots: +25% @31k, +11% @8k, -10% @2k, KV only 43k tokens) -> trade-off, not default.
Real prefill fixes: Gen4 host link (~2x), more VRAM for experts (4 cards), or CPU-side compute of rarely-hit experts.
BENCH CAVEAT: bench.py runs are single-shot and confounded by compile-cache state (servers that loaded a cached
AOT graph (140 s startup) ran @4 at ~182 tok/s; fresh-compile starts (~470 s) ran ~118-121 with the same code).
Need a repeated-run harness before trusting <15% differences.

Launch (defaults now in serve/serve.sh, MTP=3 default): `OVERLAYS=emulated-switch serve/serve.sh` then `python3 ~/warmup.py`
(= OFFLOAD_GB=34, UTIL=0.94, R9K_EXPERT_CACHE_SLOTS=270, fp8 target+draft LM heads). 320 slots leaves no KV room.
Opt-ins measured and left off: R9K_FP8_BLOCK=rowwise|block (acceptance drop / ~1 ms), R9K_FP8_LINEARS=hyper_connection
(acceptance drop), R9K_DRAFT_LMHEAD=mxfp4 (throughput +, single -).

Remaining gaps: short-prompt prefill (host read-through of offloaded experts over Gen3 PCIe), LRU miss traffic
(~7.5 ms/step), untuned Triton fp8-block GEMMs (~4.7 ms/step); VM100 RAM upgrade would allow a full cache.
Correction: the "+ fp8 HC linears" row above ran on a stale torch.compile cache that still used the bf16 HC path
(Dynamo cannot trace our ctypes kernels; they are now torch custom ops, and the compile cache is keyed by R9K_* knobs):
that gain is the fp8 LM heads alone. With fp8 HC + row-wise fp8 block projections truly active: 58.5 single, MTP
acceptance 2.64 (vs ~3.0) -> worse; being A/B'd separately.
Correction: the earlier "P2P gives nothing" A/B was flawed (tcclaviger's r4d AR kept using P2P IPC in both arms).

### 2026-09-18 afternoon: extension points, overlay split, harness, NVFP4
- **Plugin on official extension points** (a529442): `R9kCompressedTensorsConfig` registered over
  "compressed-tensors"; `ModelRegistry` subclasses for Qwen4Exp CG/CausalLM/MTP; `R9700Platform` via
  `vllm.platform_plugins` -> `R9kCommunicator` (libr4d AR). One class patch left (MTP allowlist, version-gated).
  `tests/test_integration.py` covers every hook. Gate (MTP-3, TP2): 203.9 tok/s @8, 189.0 @16, prefill 8k 2727 tok/s,
  GSM8K 98/100, needle 3/3 -- parity with pre-refactor.
- **Overlay split** (cbffc52): `overlay/emulated-switch/` (hostcall-free RCCL, vLLM .so hostcall patcher, scanners),
  `serve/serve.sh` stock by default, `OVERLAYS=emulated-switch` on VM100. MTP=3 now the default.
- **Harness**: `bench/harness.py` (nonce prompts, streamed TTFT vs decode, repeats -> median/spread) + `bench/ab.sh`
  (interleaved A/B with restarts).
- **NVFP4**: (a) load-time NVFP4->MXFP4 conversion (dense + MoE; exact dequant, 2-candidate E8M0 search). Weight error
  sits at the double-rounding floor, ~1.55x NVFP4's own (synthetic: 0.095 -> 0.149). Qwen3.8-27B-NVFP4: GSM8K-200
  96.5% (GGZ14's conversion run on 09-16: 97.0%), decode 29.7 tok/s, 8-conc 210.7 tok/s.
  (b) native NVFP4 kernel variant (`r9k_moe_nvfp4a8`, template NV): WMMA on unscaled e2m1, per-(row, 16-K) e4m3 scale
  applied to the partial (8 FMAs/WMMA), per-row global in the epilogue -- exact weights; default for dense NVFP4
  (`R9K_NVFP4=native|mxfp4`). NVFP4 MoE layers still convert (expert cache carries MXFP4-layout scales).

### Next
1. Unit tests on GPU; VM100 RAM 128 -> 256 GB (host has 364 GB free) so experts (70 GB) + PLE (42 GB) fit pinned.
2. Stock + plugin bring-up (eager), correctness (GSM8K subset, needle), then cudagraphs + MTP.
3. Perf: rocprof of our stack vs tcclaviger; fused LRU+align, fused silu-quant, QSA fp8 KV + WMMA indexer,
   GDN HIP (libr4d), FP8 GEMM path, P2P AR communicator.

### Open questions for Brian
- Licenses: libr4d (StillDeadcode) and vllm-mxfp4 (GGZ14) have none — our MoE kernel is derived from libr4d.
- Canonical checkpoint: tcclaviger GPTQ int6-PLE (current) vs MXFP4-FP8 vs davetha heretic2.
