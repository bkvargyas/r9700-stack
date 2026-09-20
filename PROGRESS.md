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

### 2026-09-19/20: Qwen3.8-27B-NVFP4 speed work (target: GGZ14 radiance on the same box)

**Baseline discipline.** `~/bb-prod.log` (181.5 combined) is the PRODUCTION box (192.168.0.155, Quark MXFP4) --
not comparable. The same-box GGZ14 radiance NVFP4 baseline is `~/bb-full.log` (2026-09-16) and a 2026-09-19 rerun:
**combined decode 196.5 t/s, conc 177/294/428/549, update p50 23.5 ms, TTFT 65 ms, prefill ~4.8k t/s**, GSM8K 97.0,
HumanEval 98.2. Always check `results.json` `env.endpoint` before quoting a number.

**Our 27B config** (serve/serve.sh): `R9K_AR_QUANT=1 VLLM_KV_CACHE_LAYOUT=LBHNC KVMEM=9 R9K_BF16_TO_MXFP4=in_proj_ba
R9K_FP8_TO_MXFP4=1 MODEL=/models/Qwen3.8-27B-NVFP4 OFFLOAD_GB=0 MTP= DRAFT=/models/Qwen3.8-27B-DFlash2-FP8 SPEC=7
ATTN=CUSTOM DRAFT_ATTN=CUSTOM CHAT_TEMPLATE=qwen-fixed NSEQ=8 OVERLAYS=emulated-switch`.

| step | decode single | conc-8 | prefill 9k | ms/step |
|---|---|---|---|---|
| start (no spec, stock attn) | 29.7 | 211 | 1374 | - |
| + DFlash2 spec7 (drafter fp8 fix) | 69.7 | 295 | - | 43 |
| + TRITON_ATTN (target+drafter), fp8->MXFP4, tuned tiles | 103 | 314 | - | 27.4 |
| + split-KV verify attention (CUSTOM), quant kernel, drafter on libr9k | 115 | 405 | 1340 | 25.8 |
| + GGZ14 chat template (acceptance +14%) | 118 | 416 | - | 25.3 |
| + libr4d prefill attention (LBHNC + fp8 descales) | - | - | 2048 | - |
| + wht6 compressed all-reduce (>=128 KB, opt-in) | - | - | 2250 | - |
| + large-M prefill GEMM (Fable pass 1+2) | 114 | 437 | 3450 | 26.0 |
| + NVFP4->MXFP4 conversion at load (R9K_NVFP4=mxfp4) | 124 | 472 | 3740 | 25.5 |
| + folded-exponent MXFP4 (R9K_FOLD=1, bit-exact here) | 114 | 487 | **3932** | 25.3 |

Quality: GSM8K-200 96.5-97.0%, HumanEval 96.3%, 8/8 concurrent sanity answers.

**What moved the needle, in order:** speculative decoding (drafter fp8 dequant fix in models/dflash.py), the chat
template (acceptance: code 3.34 -> 4.49 tok/step), libr4d prefill attention (2413 -> 93 ms per 9k prompt), the
large-M GEMM (68-84 -> 117-142 TFLOPS), split-KV verify attention, fp8->MXFP4 requant, wht6 all-reduce.

**Measured dead ends:** unpadded drafter batch + sync scheduling (needs GGZ14's vLLM patch; 29.1 ms/step on stock);
drafter in W4 (-0.5 ms/step but -7% acceptance); LDS-staged A for the decode kernel (never wins once its gate is
correct); NT loads on MT>1 (Flash-Next prefill -24%); fp8->MXFP4 on Flash-Next (no gain, it is expert-traffic bound).

**Flash-Next on the same code:** decode 81 / conc-8 206 / prefill 2046, MTP acceptance 2.69 -- unchanged or better
at every step (the prefill GEMM gave conc-8 +13%).

**Gotchas found the hard way:** vLLM's memory profile underestimates once load-time requant/merges are on (use
KVMEM=); a stale `tuned.json` sync silently reverted the M=32/64 rows; LDS-A with WV=1,MT=4 staged half its tile
(NaN at batch 64) -- tests/test_tuned_cfgs.py now gates every tuned row.


**FINAL BetterBench 2026-09-20** (20 passes, same box, same settings as the GGZ14 baseline), ours vs GGZ14:
combined decode **189.3 vs 196.5 (96%)**, conc 165/274/404/522 vs 177/294/428/549 (93-95%), prefill sweep
3838/3902/3843/3649 vs 4776/4950/4906/4745 (~79%), step p50 24.4 vs 23.5 ms, TTFT 99 vs 65 ms, per-category
acceptance at parity (code 4.69 vs 4.64, file_edit 6.10 vs 6.03). Serve config: `R9K_FOLD=1 R9K_NVFP4=mxfp4
R9K_AR_QUANT=1 R9K_FP8_TO_MXFP4=1 R9K_BF16_TO_MXFP4=in_proj_ba VLLM_KV_CACHE_LAYOUT=LBHNC KVMEM=9 ATTN=CUSTOM
DRAFT_ATTN=CUSTOM CHAT_TEMPLATE=qwen-fixed DRAFT=.../Qwen3.8-27B-DFlash2-FP8 SPEC=7`.
Flash-Next on the same build: single 84.5, conc-8 206, prefill 2165, MTP acceptance 2.75 (all >= its pre-week numbers).

**Remaining prefill gap is GEMM only** (9k prompt, GPU-busy): ours 1361 ms / 768 calls vs GGZ14 930 ms / 484;
all-reduce 373 vs 360, attention 92 vs 58, glue comparable. Their edge: activations arrive WMMA-fragment-tiled
(256 B per 16x16 fp8 fragment) from their fused norm+quant, so their GEMM never stages A through LDS.

### 2026-09-20 evening: fragment-tiled activations (closes most of the prefill GEMM gap)

The prefill GEMM now takes its activation in WMMA-fragment layout (`r9k_quant_rows_fp8_tiled`: fragment (mt, ks)
= 256 contiguous bytes, lane l holding A[mt*16 + l%16][ks*16 + (l/16)*8 .. +8]) and loads each A fragment with one
SADDR `global_load_b64` straight into the register the WMMA consumes -- no LDS A slab, no per-fragment `ds_read`,
no staging stores, LDS down from 2x26 KB to 2x8 KB on the 256x128 tile. The quantizer writes that layout instead
of row-major at the same cost (+/-1%), so it is a re-layout and not an extra pass. W keeps the pass-3 folded
unpack through LDS. Output is bit-identical to the folded LDS-A kernel (same accumulation order), gated by
`tests/test_atiled_4bit.py`. Dense path only: `pick_cfg(..., fold=True) -> ("A", cfg)` at M >= 128 when
`R9K_ATILED=1` (default on), folded MXFP4, K % BK == 0 and K <= 20480; everything else keeps the LDS-A tiles.

Kernel-level (min of 5 interleaved, warmed): 0.74-0.92x of the tuned folded tile across the five served 27B
shapes at M 128-4096, i.e. ~150 -> 175-215 TF/s. Served, 27B, three probes per leg on two independent server
starts (cv < 0.1%):

| prompt | `R9K_ATILED=0` | `R9K_ATILED=1` | gain |
|---|---|---|---|
| 8.97k tok | 3936 tok/s (2281 ms) | **4434 tok/s** (2021 ms) | **+12.7%** |
| 20.7k tok | 3772 tok/s (5477 ms) | **4209 tok/s** (4919 ms) | **+11.6%** |

So 9k prefill is now **~93% of same-box GGZ14** (4776-4950), up from ~79%. Decode is untouched, as the M >= 128
gate implies: single 113.2/116.4 (on) vs 115.2/118.7 (off), conc-8 483.6 vs 483.8, GSM8K-500 96.40% both legs;
Flash-Next unchanged (84.6 / 205.1 / prefill 2151 / accept 2.73). A first conc-8 pair read 472.6 vs 503.8 with
non-overlapping ranges at n=2 -- re-measured on fresh servers it was 483.6 vs 483.8, i.e. run-to-run noise, which
is what the code says it must be (`pick_atiled` returns None below M=128, so the decode path is byte-identical).

Routed MoE deliberately not converted: those GEMMs are weight/L2-stream bound (gate_up moves the 420 MB expert set
at 397 GB/s), so a gather+tile pass would add ~25% of the weight traffic to a kernel whose A staging is not the
limit. Argued, not measured -- the kernel already accepts sorted-position tiled A, only the producer is missing.

**The 768-vs-484 launch-count gap was not apples to apples.** GGZ14 chunks at 8192 (2 chunks on a 9k prompt to our
3 at NBT=4096) and keeps MLP layers 56-63 on fp8, while `R9K_FP8_TO_MXFP4=1` converts them -- so their 930 ms
excluded 16 GEMMs per chunk that ours included. Untried from this: `NBT=8192` (check the libr4d AR max message
size at 8192x5120x2 = 80 MiB) and leaving layers 56-63 on the fp8 kernel.

**Measurement discipline (learned the hard way this week):**
- ALWAYS check `results.json` `env.endpoint` before quoting a baseline: ~/bb-prod.log is the PRODUCTION box.
- Never start a run while another is live (two servers on the same GPUs produced 30% CVs and nonsense numbers);
  `pgrep -f "[g]sm8k|[b]atch|[q]fn"` + `docker ps` before starting.
- Cold single-shot kernel timings overstate by ~15%: the card sits at 2.2-2.4 GHz / 222 W under sustained load
  (decode sees 2.82 GHz). Use tuning/prefill_ab.py (warmed, interleaved).
- GSM8K-500 CANNOT separate these configs: six runs scatter 95.6-97.4% with no consistent ordering. Quality claims
  need several thousand questions or a different eval.
- conc-8 (`agg8_tok_s`) at n=2 can separate identical builds by 6% with non-overlapping ranges: the prefill-probe
  words are ~6.9 tokens each (`w<random>`), so WORDS=1300 is the ~9k-token prompt and WORDS=6000 exceeds the 32k
  context and 400s. Before believing a decode delta, ask whether the code even has a path to it, then re-measure.
- Kernel gates that call the inner op miss served-path breakage: exercise r9700_vllm/ops.py's public wrappers too.
- `DRYRUN=1` only prints the assembled docker command: it does NOT source the overlays, so a launcher missing
  `OVERLAYS=` passes DRYRUN and then fails at startup. serve/27b.sh and serve/flashnext.sh shipped that way and
  every RCCL collective died with HIP "the operation cannot be performed in the present state" at enqueue.cc:2119
  (comm init completes, single-GPU compute is fine, all transports fail alike) -- the signature of the stock,
  hostcall-carrying RCCL on the .100 PLX box. Both launchers now default `OVERLAYS=${OVERLAYS-emulated-switch}`.
  A new launcher is not verified until it has actually served a request.
### Next
1. Unit tests on GPU; VM100 RAM 128 -> 256 GB (host has 364 GB free) so experts (70 GB) + PLE (42 GB) fit pinned.
2. Stock + plugin bring-up (eager), correctness (GSM8K subset, needle), then cudagraphs + MTP.
3. Perf: rocprof of our stack vs tcclaviger; fused LRU+align, fused silu-quant, QSA fp8 KV + WMMA indexer,
   GDN HIP (libr4d), FP8 GEMM path, P2P AR communicator.

### Open questions for Brian
- Licenses: libr4d (StillDeadcode) and vllm-mxfp4 (GGZ14) have none — our MoE kernel is derived from libr4d.
- Canonical checkpoint: tcclaviger GPTQ int6-PLE (current) vs MXFP4-FP8 vs davetha heretic2.
