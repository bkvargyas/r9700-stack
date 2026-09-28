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

### 2026-09-21: independence from libr4d (see notes/independence.md)

Both runtime dependencies replaced by our own kernels: paged attention (`kernels/r9k_attn.hip`, now the default)
and the 2-rank all-reduce (`kernels/r9k_ar.hip` + `r9k_ar_wht.hip`, `R9K_AR_IMPL=r9k`). A fully libr4d-free 27B
runs at **~11% below prefill and ~4% below decode** with identical GSM8K-500, against -55% prefill if the
dependency were simply dropped. Attention alone is 99.8% of libr4d; the whole remaining gap is the all-reduce.

**Prefill knobs measured and rejected** (both fell out of the finding that the "768 vs 484 GEMM launches" gap was
never apples to apples -- GGZ14 chunk at 8192 and keep MLP layers 56-63 on fp8):
- `NBT=8192`: no gain at 9k (4395 vs 4392) and the engine dies with HTTP 500 at a 20.7k prompt. Rejected.
- `R9K_FP8_TO_MXFP4=0` (keep layers 56-63 on fp8): prefill 3456 vs 4392 (-21%), decode 94.7 vs 115.1, step
  30.7 vs 25.1 ms. Much worse -- converting those layers is a clear win and the existing default is right.

### 2026-09-21: TTFT investigated and closed

101 ms vs the reference stack's 65 ms. Decomposed to **~80 ms fixed + 0-25 ms step-boundary wait**; the fixed part
is prefill (queue time 0.0 ms, prefill time ~78 ms for an *8-token* prompt), and prefill runs **eager**: 32.2 ms
GPU busy against ~78 ms wall, ~1,950 launches at ~24 us of gap each. `cudagraph_mode` already defaults to
FULL_AND_PIECEWISE so those launches survive capture; `FULL` is 55 ms *worse*. Ruled out by measurement:
speculative decoding (helps by 37 ms), chat-template rendering, HSA_ENABLE_MWAITX, GPU_MAX_HW_QUEUES, the API
server. Closed as upstream behaviour -- full detail and the one remaining lever in notes/picking-up.md.

Added `CGMODE=` and `MWAITX=` knobs to serve/serve.sh while investigating.

### 2026-09-22: power cap is a fixed condition, not a knob

Raising the 225 W cap is closed permanently. Every reference measurement (libr4d, GGZ14) was taken at 225 W, so
uncapping ours would invalidate the comparison in the same way the `bb-prod.log` baseline mix-up did; and Brian
runs capped in production, so an uncapped number is one he would never see. Prefill being clock-limited at
2.40 GHz is therefore a property of the target, not an opportunity.

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
### 2026-09-23: 45+48 on one PLX switch, switch-local P2P, and what routing costs in serving
Third card installed; VM100 now runs **45 + 48, both on the same PEX 8747** (was 45 + c6 on separate switches).
Host work is in `host/README.md`: a second card on one switch gets no BAR0 without `r9700_chainfix` (DKMS), and
peer traffic only stays in the switch with ACS redirect off **and** guest GPU addresses equal to host addresses.
- Synthetic, 45<->48 at 256 MB: both-ways peer copy 12.74 -> **25.26 GB/s**; RCCL all-reduce busbw 5.09 -> 10.08.
- **Full BetterBench 20-pass, `serve/27b.sh` (switch-local P2P) vs the 2026-09-22 baseline (45 + c6): parity.**
  Step 24.4 ms both; conc 166/280/379/510 vs 166/270/394/521; prefill 4,123/4,153/4,048/3,812 vs
  4,099/4,143/4,043/3,809 (2k/8k/16k/32k); per-category decode all within noise (`betterbench compare`).
- **Live A/B on one running server, ACS redirect off / on / off** (`host/acsab.sh`; prefill sweep + quick decode):

| | pf 2k | pf 8k | pf 16k | pf 32k | step p50 | step p99 |
|---|--:|--:|--:|--:|--:|--:|
| switch-local (off) | 4,167 | 4,187 | 4,070 | 3,821 | 24.47 ms | 26.62 ms |
| **via CPU (on)** | **3,741** | **3,770** | **3,683** | **3,491** | **25.29 ms** | **27.37 ms** |
| switch-local (off, repeat) | 4,121 | 4,149 | 4,047 | 3,810 | 24.49 ms | 26.50 ms |

  Routing peer traffic through the CPU costs **9-10% prefill and +3.3% decode step time**; the repeat arm
  returns to within ~1%, so it is not drift. Decode tok/s medians moved with speculative acceptance (199/200/203)
  and are not the signal here; step time is. No IOMMU/AER faults in any arm. Raw results: `~/bb0923/` on VM100.
- So two cards per switch cost nothing **only** with switch-local routing. For 4 cards on two switches the
  in-bank hops need it too, which means matching guest addresses for both banks in one VM (not yet solved:
  OVMF packs its 64-bit window contiguously and the two banks' host apertures are ~0.5 TB apart).
- Correction carried forward: the 2026-09-18 Flash-Next "P2P gives nothing" A/B is invalid (P2P IPC in both arms).

### 2026-09-24: Flash-Next at TP=4 on four cards (target: tcclaviger 29.04.4 on the same box)

Rob's published Flash-Next TP4/MTP4 run (163.8 BetterBench decode, 687 @16, 7.2k prefill) is on his hardware; his
image on OUR box (`~/serve-rob-tp4.sh`, hostcall-patched copies in `~/p2p-patched-2904`) is the real target. All
numbers below are the 1-minute probe (`~/probe.py`: 4 mixed prompts greedy / conc 8 and 16 / one 8.7k prefill);
BetterBench is reserved for a finalist. Step time is single-request MTP-3 (`~/tp4tune.sh` step.txt).

| config | step ms | dec 1 | c8 | c16 | prefill 8k |
|---|--:|--:|--:|--:|--:|
| ours, start of day (TP=4, RCCL all-reduce, stock QSA) | 28.1 | -- | 299* | 508* | 3,767* |
| + our 4-rank P2P all-reduce | 22.9 | 140 | 299 | 508 | 3,767 |
| + our QSA sparse attention | 22.9 | 139 | 479 | 770 | 4,689 |
| + hybrid block-fp8 + mxfp4 heads + fused qk-norm/rope | **21.6** | **148** | **501** | **770** | **4,688** |
| Rob's image (29.04.4, MTP-4, fp8 KV, expert offload) | 20.8 | 169 | 503 | 769 | 6,446 |

(*) measured after the all-reduce change; the RCCL baseline's probe was not run.

- **N-rank P2P all-reduce** (`kernels/r9k_ar.hip` `r9k_ar_oneshot_nrank` / `r9k_ar_twoshot_nrank`, `comm/r9k_ar.py`
  `R9kAllReduceN`, installed for TP>2): one-shot <= 16 KB, two-shot <= 512 KB, RCCL above. Graph-timed at 20 KB
  11 us vs RCCL 73; 320 KB 73 vs 102. Decode step 28.1 -> 22.9 ms. At prefill sizes (20 MB) the exact two-shot
  loses to RCCL's ring (4.05 vs 2.97 ms even at 256 blocks): the all-to-all push saturates the inter-switch uplink.
  Only compression or a topology-aware schedule would win there; parked.
- **Our QSA sparse attention** (`kernels/r9k_qsa.hip`, `attn/qsa.py`, bound per layer; `R9K_QSA=stock` reverts).
  Reformulation: a row's selection is a union of whole 4-token groups plus the causal tail, so each row becomes a
  bitmap over groups and a tile of 16 rows walks the UNION of its groups, staging K/V once per tile (the stock
  Triton kernel gathers ~2 MB per row per layer). Bit-for-bit the same attended set; fp32 online softmax either
  way. 3.6x on the 12k prefill chunk (14.0 -> 3.9 ms), 13x on a 4k prefill, 2.4x decode; `tests/test_qsa_r9k.py`.
  Bug found on the way and worth remembering: a row that has met no valid key yet (m = -inf) riding along another
  row's rescale computes exp2(-inf - -inf) = NaN; impossible in causal attention (every row sees key 0 first),
  routine in QSA. Guarded.
- **Decode profile, ours vs Rob (torch profiler, `~/step-audit.py` per-layer):** GPU time per layer is at parity
  (415 vs 462 us, his including ~62 us of host expert streaming); we launch 78 kernels per layer to his 37. Enabling
  stock's fused qk-norm/rope Triton kernel on ROCm (`R9K_FUSED_QKROPE`, off in stock only because of an
  `is_cuda()` check; `tests/test_fused_qk_rope.py`) removed ~30 launches per QSA layer for ~0.06 ms/step: launch
  count is worth far less than assumed. The remaining eager glue is the indexer's norm + rope; not worth fusing.
- **Knobs that paid:** `R9K_FP8_BLOCK=block` (exact block scales, our split-K GEMM) at decode widths with stock's
  Triton kernel above M=64 (`R9K_FP8_BLOCK_MAXM`); the dispatch lives INSIDE a custom op (`r9700.fp8_block_dispatch`,
  layer registry keyed by registration index) because a Python branch on M in compiled code is resolved once at the
  M=4096 profile pass and then serves decode with the wrong kernel, and because `id(layer)` baked into the AOT
  artifact differs per rank process. mxfp4 LM heads (`R9K_TARGET_LMHEAD` / `R9K_DRAFT_LMHEAD`) halve the head's bytes.
- **Knobs that did not:** MTP-4 (acceptance up, step +1.7 ms, net loss); fp8 or mxfp4 for the hyper-connection
  mixers (latency-bound at these sizes; the merged down+inject matrix is 324 rows and fails the %16 check anyway);
  MoE decode config sweep for TP=4 shapes (`tuning/decode_moe_sweep.py`: served configs within 2-11% of best;
  gate_up now keyed by N in `moe/experts.py`); fp8 KV is refused by stock QSA ("requires a BF16 main KV cache").
- **Quality gate** (`bench/eval.py`, 300 GSM8K, conc=1): reference 96.0% vs candidate 97.0%, McNemar p=0.51, sanity
  8/8 both. Long-chain paired eval still to run before the candidate becomes the default.
- **vLLM 0.30** (`nightly-rocm100-e975732`, torch 2.12): whole stack runs unchanged (`r9700/vllm:dev030`,
  `~/p2p-patched-030`); probe at parity with the 0.29-era nightly.
- **Full BetterBench (20 passes), final default vs Rob's image, same box:** decode 125.1 vs 134.1; step p50 21.22
  vs 20.44 ms; TTFT p50 137 vs 145 ms; prefill 2,853 / 4,637 / 4,701 / 4,614 vs 5,279 / 5,711 / 5,977 / 6,106
  (2k / 8k / 16k / 32k); concurrency 117 / 193 / 283 / 374 / 478 vs 126 / 197 / 303 / 427 / 542 (1 / 2 / 4 / 8 / 16).
  The greedy probe overstated concurrency parity (-12% under sampling); short prefill (2k) is the worst ratio.
- **2k prefill profile (2,142 tokens, rank 0): our all-reduce is 387 ms of 671 ms GPU time (222 RCCL calls at
  1.73 ms for 11 MB messages); Rob's 116 ms (0.58 ms per call, compressed two-shot).** Everything else is within
  ~50 ms of his. The short-prefill 2x is the all-reduce alone.
- Prefill gap that remains (12.7k tokens, rank 0): all-reduce 1,019 ms (RCCL) vs Rob's 607 (compressed two-shot);
  attention now ~250 vs his 159; MoE/dense/elementwise ~180 ms combined.

### 2026-09-25: compressed hierarchical all-reduce, vLLM 0.30 default

- **Base image switched to the vLLM 0.30 nightly** (`nightly-rocm100-e975732`, torch 2.12). Final-config probe on
  it: 21.60 ms/step, 148.0 / 484 / 752 / 4,685 -- identical to the dee37d89 image. On VM100 `r9700/vllm:dev` is now
  the 0.30 build (old image kept as `dev-0918`), `~/p2p-patched-nightly` links to the 0.30 patched extensions.
- **`kernels/r9k_ar4.hip` + `comm/r9k_ar4.py`: the all-reduce from notes/ar4-plan.md.** Hierarchical 2x2 over the
  switch pairs, Walsh-Hadamard 4/6-bit wire fused into the push kernels (two elements per lane; no LDS in the
  transform), rotated-domain sums, seven launches, four handshakes, bit-identical outputs on all ranks.
  `tests/test_ar4.py` (torchrun, 4 ranks): PASS at 4 and 6 bits, graph replay interleaved with the exact kernels.

| message | RCCL | ar4 4-bit | ar4 6-bit | exact two-shot |
|---|--:|--:|--:|--:|
| 1 MB | 204 us | 118 | 143 | 222 |
| 5 MB | 773 | 371 | 495 | 1,194 |
| 11 MB | 1,640 | **739** | 1,009 | 2,723 |
| 21 MB | 3,107 | **1,352** | 1,863 | -- |

  Rob's kernel: ~580 / ~1,200 us at 11 / 21 MB. Relative RMS error on a heavy-tailed input: 0.12 (4-bit), 0.027
  (6-bit); an already-quantised input round-trips to bf16 rounding.
- **Serving (Flash-Next TP=4 probe):** 8k prefill 4,685 -> **5,621 tok/s with 4-bit (+20%)**, 5,319 with 6-bit;
  decode 147-149, c8 497-506, c16 738-768 -- unchanged within noise. Prefill is now 87% of Rob's image (from 73%).
- **Quality (300 GSM8K, conc=1, same image):** default 96.67% vs 4-bit 96.00%, discordant 4/2, McNemar p=0.69: no
  detectable difference. The 800-question chain-of-thought paired eval decides whether R9K_AR4 becomes the default
  (opt-in `R9K_AR4=1` until then; `R9K_AR4_BITS=6` is the conservative wire).

### 2026-09-25 (cont.): the short-prefill floor, root-caused

BetterBench 2k prefill was 2,779 tok/s against Rob's 5,279 while 8k+ was within 3-8%. A TTFT-vs-length sweep
(`~/pfsweep.py`) showed TTFT quantized: ~285 ms for anything up to ~700 tokens, ~563 ms for 1.3k-3.1k tokens
regardless of length, then linear at ~5,650 tok/s. Two mechanisms, found with the torch profiler's CPU-side trace:

- **The prefill forward pass is CPU-bound at ~285 ms.** For a 672-token prefill the CPU issued ops for 407 of the
  426 ms span: 20,230 CPU ops per forward (compiled-graph pieces 267 ms, MoE Python 96, GDN 77, 351 unquantized
  GEMM dispatches 54, 101 all-reduces 32). Decode hides this behind cudagraphs; prefill runs the compiled graph
  eagerly. The rank skew this creates is what the all-reduce kernels spin on (ar4_pack_push at 498 us per call in
  the profile against ~40 us in the bench). Rob's stack has the same floor: his linear rate is 5,711 tok/s.
- **Prompts were split at the mamba block size.** With prefix caching on (default), the hybrid model runs the mamba
  cache in "align" mode and vLLM's scheduler aligns every chunk end to the mamba block size, so a 1,302-token prompt
  ran as two forward passes (202 all-reduces, 98 MoE calls, 26 QSA calls: exactly 2x the 672-token run), each paying
  the CPU floor. `--mamba-block-size` is ignored in align mode.

Fixes measured (TTFT, tokens -> ms): 1,325: 563 -> 313; 2,166: 564 -> 373; 3,086: 564 -> 512 (6,022 tok/s).

| | before | `--block-size 4096` | **`--no-enable-prefix-caching`** |
|---|--:|--:|--:|
| 1,325 tokens | 563 ms | 296 | 313 |
| 3,086 tokens | 564 ms | 510 | 512 |
| KV cache capacity | 296k tokens | 99k (mamba pages padded 9x) | 296k |

`serve/flashnext.sh` now defaults `PREFIX_CACHE=0` (serve.sh knob; `=1` restores prefix reuse across requests --
a serving-behaviour change, flagged to Brian). **Full BetterBench of this default:** decode 125.1, step p50 21.12 ms,
TTFT 144 ms, prefill 5,267 / 5,702 / 5,723 / 5,530 (2k / 8k / 16k / 32k), concurrency 116 / 190 / 277 / 370 / 463.
Prefill is at parity with Rob's image through 8k (5,279 / 5,711) and 4-9% behind at 16-32k; decode -7%, concurrency
-13..-15% under sampling remain. Removing the CPU floor itself (capturing prefill chunks in cudagraphs
at 1024/2048/4096) is being measured; it is what would pass Rob at short prompts.

### 2026-09-25: final default and where it stands

Default now: our QSA attention, N-rank + compressed (4-bit) all-reduce, hybrid block-fp8, mxfp4 LM heads, fused
qk-norm/rope, prefix caching off, prefill cudagraphs to 2048 tokens, vLLM 0.30. Full 20-pass BetterBench, same box
and checkpoint as Rob's image (tcclaviger 29.04.4, MTP-4, fp8 KV, expert offload):

| | ours (2026-09-24 morning) | **ours now** | Rob's image |
|---|--:|--:|--:|
| decode score | 125.1 | 125.0 | 134.1 |
| step p50 | 21.22 ms | 21.15 ms | 20.44 ms |
| TTFT p50 | 137 ms | **104 ms** | 145 ms |
| prefill 2k / 8k / 16k / 32k | 2,853 / 4,637 / 4,701 / 4,614 | **5,106 / 5,624 / 5,726 / 5,532** | 5,279 / 5,711 / 5,977 / 6,106 |
| concurrency 1 / 2 / 4 / 8 / 16 | 117 / 193 / 283 / 374 / 478 | **118 / 197 / 292 / 433 / 543** | 126 / 197 / 303 / 427 / 542 |

Concurrency at parity, TTFT 28% better, prefill within 1.5-9%, single-stream decode 7% behind (his MTP-4 vs our
MTP-3). The same full run with MTP-4: decode 130.5, step p50 22.48 ms, concurrency 122 / 197 / 294 / 402 / 466 --
+4.4% single-stream for -14% at 16 concurrent, so MTP-3 stays the default and `MTP=4` is the single-stream knob. Every lossy default passed two null paired evals at
conc=1 (300 short-answer, 800 chain-of-thought); the served config's sanity check passes.

### 2026-09-28: QSA indexer scoring kernel (r9k_qsa_score)

The 4k-chunk profile of 2026-09-25 (final stack vs Rob, `~/chain-pf4k.out`) put our GPU-busy at 731 ms vs his 625:
all-reduce 149 vs 200 (ours now faster), MoE 177 vs 141, elementwise 161 vs 97, QSA 96 vs 46. The QSA gap was
one kernel: vLLM's Triton `_qsa_mqa_paged_kernel` (indexer scoring) at 80 ms per chunk, 43 calls x 1.85 ms. It
launches a program per (row, 32 columns) over the whole context capacity, re-reads the compressed keys once per
row, and writes every column of an fp32 [rows, capacity] logits buffer (128 MB per layer at 4096 rows), although
`top_k_per_row_decode` reads row r only up to its visible count (position / 4).

`kernels/r9k_qsa_score.hip` (independent implementation from the stock kernel's documented semantics, same WMMA
pipeline as r9k_attn): one workgroup = 16 query rows x one column part; K[64 x 128] staged once per tile through
the page table, per head S^T = K . Q_h^T on 8 WMMA 16x16x16 bf16, relu, sum over heads, scale; the tile stops at
its last visible column and leaves the rest of the logits row untouched. Few-row batches (decode, MTP verify)
split the column range across workgroups (R9K_QSA_SCORE_SPLITS). `attn/qsa_score.py` mirrors the stock
`qsa_select_paged_tokens` (same 128 MB chunking, stock top-k and expansion) and is bound as `_select` on every
`QSAIndexer` instance from `qsa.install`; `R9K_QSA_SCORE=stock` keeps vLLM's kernel. The indexer runs inside
vLLM's `qwen4_exp_qsa_with_output` custom op, so the ctypes launch is outside torch.compile and cudagraph-safe.

`tests/test_qsa_score.py` (single GPU, 15 cases: prefill tiles, chunked prefill to 32k, mixed batches with
padding rows and forced splits, decode, MTP verify, tiny contexts, page sizes 4/16/64, 1/2/4/8 heads): logits on
every visible column vs an fp32 reference, ours 1.4-4.2e-7 relative vs stock 1.4e-7-1.1e-6; visible counts equal;
selected token lists identical except where the differing blocks tie with the k-th best reference score
(4041/4096 rows identical on the random 4k case, the rest ties). Timing, 4 heads, page 16, scorer alone / end to
end (score + top-k + expand):

| shape | stock | **ours** | stock e2e | **ours e2e** |
|---|--:|--:|--:|--:|
| 4k chunk, ctx 12288 | 2,430 us | **752** | 2,547 | **1,049** |
| 4k chunk, ctx 32768 | 6,561 | **2,029** | 7,253 | **2,511** |
| 2k prefill | 210 | **65** | 273 | **184** |
| decode 1 row, ctx 8000 | 79 | **38** | 214 | **180** |
| MTP 16 reqs x 4 rows | 82 | **57** | 219 | **179** |

Expected serving effect: ~-65 ms per 4k chunk at 12k context (~9% of GPU-busy), more at 32k. Serving A/B by probe,
same day, TP=4 (`~/tp4tune/{qsc-r9k,qsc-stock}`): decode 148.5 -> 149.6 tok/s, c8 556 -> 533 (probe spread on c8
across earlier identical configs is 517-562), c16 768 -> 770, **prefill 8k 5,549 -> 6,188 tok/s (+11.5%)**. Full
20-pass BetterBench of the new default (`~/tp4tune/bb-final5`; same box, checkpoint and settings as bb-final4):

| | bb-final4 (stock scorer) | **bb-final5 (r9k scorer)** | Rob's image |
|---|--:|--:|--:|
| decode score | 125.0 | 123.2 | 134.1 |
| step p50 | 21.15 ms | 21.27 ms | 20.44 ms |
| TTFT p50 | 104 ms | 103 ms | 145 ms |
| prefill 2k / 8k / 16k / 32k | 5,106 / 5,624 / 5,726 / 5,532 | **5,607 / 6,281 / 6,406 / 6,158** | 5,279 / 5,711 / 5,977 / 6,106 |
| concurrency 1 / 2 / 4 / 8 / 16 | 118 / 197 / 292 / 433 / 543 | 117 / 193 / 311 / 437 / 561 | 126 / 197 / 303 / 427 / 542 |

Prefill +10-12% at every length and now ahead of Rob's image at all four (+6 / +10 / +7 / +1%); concurrency at
parity or better (the probe's c8 dip was noise); decode -1.4%, inside the spread of identical earlier runs
(123.1-125.1). Remaining gap to Rob: single-stream decode (his MTP-4) and step p50. Numerics: scores differ from stock only in fp32 summation order, so
selections differ only at exact ties; no paired eval needed. Commit fd950d4 (local).

### 2026-09-28 (cont.): re-profiled, the TP=4 down GEMM, and the PLE transposes

4k-chunk profile with the r9k scorer (`~/tprof-ours-prefill4kb`): GPU-busy 731 -> 653 ms (Rob 625); QSA 96 -> 19 ms
(Rob 46). Remaining per family: elementwise 160 vs 97, MoE 176 vs 141, dense GEMM 125 vs 118, all-reduce 148 vs 200,
GDN 14.5 vs 20. Per kernel:

- **Down GEMM at TP=4 ran on the decode kernel.** Per rank K=160; `r9k_moe_4bit_prefill` and `pick_moe_prefill`
  required K % 64 although the BK=32 tiles (cfgs 8-11) walk K as whole 32-slabs. 49 calls x 1.5 ms = 73 ms per
  chunk. Guard is now per tile (`r9k_moe_prefill_bk`), plus a 64x128 BK=32 tile (cfg 17). Bench (E=512, top-10,
  4096 tokens, `tuning/prefill_moe_bench.py --tp 4`): down 1,946 us (old) -> 1,222 (cfg 11) / 1,250 (cfg 17);
  folded variants slower (1,328 / 1,429); cfg 11 stays the default. Still only 27 TFLOPs: the epilogue writes 2-byte
  scattered elements (a wave's store covers two 32 B row fragments) and the down output is 210 MB per chunk --
  ablation queued (`-DR9K_PF_ABL=8`, no epilogue stores) to size an LDS-staged epilogue.
- **`torch.zeros` of the down buffer** (168 MB, 260 us x 49 = 10.6 ms per chunk) dropped when there is no expert
  map: the hot and cold passes write every routed row (`R9K_MOE_ZERO_DOWN=1` restores it).
- **PLE short conv: 27.5 ms per chunk, of which 23.4 ms are six ATen strided copies** -- the two
  `transpose(1, 2).contiguous()` around `F.conv1d` on [prefills, 4096, 10240] bf16 at ~14 GB/s (5.9 ms each; the
  depthwise conv itself is 1.1 ms). `kernels/r9k_transpose.hip` (64x64 LDS tiles, 16 B coalesced both ways) +
  `ple/short_conv.py`: the stock prefill method with the two transposes on our kernel, bound per PLE layer
  (`R9K_PLE_CONV=stock` keeps vLLM's). Test vs the stock method (bit-identical output and conv state) pending.
- Also seen: `hc_combine_norm` 575 us x 97 (Rob ~420), `hc_gate_mix` 322 us x 100 (Rob ~270) -- stock Triton, one
  program per row, 512-wide blocks; a HIP rewrite is the next elementwise item. `aten::mul` in the compiled MoE
  graph 2.8 ms; `w8a8_triton_block_scaled_mm` 34 ms (Rob's fp8 GEMM 29).

Serving probe (TP=4) after the down tile + no zero fill: decode 149.6 -> 148.8, c8 533 -> 550, c16 770 -> 783,
**prefill 8k 6,188 -> 6,639 tok/s** (+7.3%; +20% over the stock-scorer 5,549 of the same morning). Commit 115ae91.

**Prefill epilogue root cause (ablations, cfg 11 down GEMM, TP=4, 4096 tokens):** 1,191 us as is; 976 with the
stores skipped; **447 with no epilogue at all**. The epilogue gathered `As[row]` and `topk_w[row]` per output
element, re-read for every column tile: ~45% of the K=160 kernel. Both prefill kernels now build a per-row scale
table in LDS next to `sRow` once per routing block. Single-buffered BK=32 tiles (cfgs 18/19, half the LDS) had
been tried first and changed nothing, which ruled out occupancy. With the fix, folded weights, 4096 tokens:

| routed GEMM | MT kernel | cfg 11 / 15 (old defaults) | **cfg 17 (64x128 BK=32)** |
|---|--:|--:|--:|
| TP=4 down 2560x160 | 1,982 us | 855 | **811** |
| TP=4 gate_up 320x2560 | 969 | 980 | **899** |
| TP=2 down 2560x320 | 2,719 | 1,270 | **1,169** |
| TP=2 gate_up 640x2560 | 2,085 | 1,638 | **1,531** |

cfg 17 is the default for both GEMMs (`R9K_MOE_PREFILL_CFG` / `_CFG1` still override). Remaining epilogue cost is
the 2-byte scattered C stores (~200-350 us of the 811); staging the C tile through LDS for 16 B row stores is the
next step there. `tests/test_prefill_4bit.py`, `test_atiled_4bit.py`, `test_cache_moe.py` pass.

**PLE short conv:** `r9k_transpose16` 3,121 -> 444 us on the [1, 4096, 10240] transpose; the prefill method
26.5 -> 4.1 ms per call, output and conv state bit-identical to stock on one long prefill, a mixed batch with a
decode prefix, tiny lengths, NULL-block and empty-state cases (`tests/test_ple_conv.py`).

**Hyper-connection kernels (`kernels/r9k_hc.hip`, `hc.py`):** one 128-thread workgroup per (row, stream) for
combine_norm (combine result held as packed bf16 between the two passes; the first cut, one wave per stream with
fp32 registers, was register-bound at 558 us and, worse, 12.5 us vs stock's 3.7 in graph replay at decode widths --
it cost 0.8 ms/step in the first serving probe), one wave per (row, 256 columns) for gate_mix. Graph-timed, 4096
rows: combine_norm 607 -> 469 us, gate_mix 318 -> 310; at 1-64 rows stock is at the launch floor (3.4 us) and ours
0.5-2 us behind, so rows below `R9K_HC_MIN_ROWS` (256; gate_mix 1024) stay on stock -- decode never sees ours.
Combine output bit-identical to stock, norm output at stock's own bf16 error (`tests/test_hc_r9k.py`). Installed by
binding over `hyperconnection.py`'s imported names as torch.ops.r9700 custom ops before tracing (they sit inside the
compiled graph; the op signatures need full type annotations or the engine refuses to start).

**Round-2 serving result (2026-09-28, TP=4).** Probe after all of the above: decode 150.1, c8 521, c16 794,
prefill 8k 7,287 tok/s (5,549 with the stock scorer this morning: +31%). Full 20-pass BetterBench, same box,
checkpoint and settings (`~/tp4tune/bb-final6`):

| | bb-final4 (stock QSA scorer) | bb-final5 (r9k scorer) | **bb-final6 (round 2)** | Rob's image |
|---|--:|--:|--:|--:|
| decode score | 125.0 | 123.2 | **125.4** | 134.1 |
| step p50 | 21.15 ms | 21.27 ms | **21.06 ms** | 20.44 ms |
| TTFT p50 | 104 ms | 103 ms | **100 ms** | 145 ms |
| prefill 2k / 8k / 16k / 32k | 5,106 / 5,624 / 5,726 / 5,532 | 5,607 / 6,281 / 6,406 / 6,158 | **6,308 / 7,415 / 7,537 / 7,254** | 5,279 / 5,711 / 5,977 / 6,106 |
| concurrency 1 / 2 / 4 / 8 / 16 | 118 / 197 / 292 / 433 / 543 | 117 / 193 / 311 / 437 / 561 | 118 / 193 / 308 / 442 / 550 | 126 / 197 / 303 / 427 / 542 |

Prefill +19 / +30 / +26 / +19% over Rob's image at 2k / 8k / 16k / 32k (+24-31% over this morning's default);
concurrency at parity; single-stream decode unchanged (-6.5% vs his MTP-4). Sanity check below.

**Staged C stores in the prefill epilogues.** The WMMA D layout leaves each lane 8 rows of one column, so the
direct store was 2 bytes per lane. Each wave now writes its TM*16 x 16-column strip into LDS (the slab buffers,
free after the K loop) and stores whole 16 B row chunks. Whole-wave-tile staging (JS = TN) needs more LDS than the
slabs on cfg 17 and measured worse (down 745 us) than one strip at a time (690), so `PF_STAGE_MAX` = 16 KB keeps
cfg 17 at one strip. TP=4 folded 4096 tokens: down 811 -> 690 us (1,932 on the decode kernel this morning), gate_up
899 -> 883; TP=2: down 1,169 -> 1,077, gate_up 1,531 -> 1,510. A first cut put the stage past the slabs but left
the row tables at the slab end, so cfg 17's stage overwrote them (memory fault): the tables now sit past
max(slabs, stage). Probe: decode 150.6, c8 522, c16 794, prefill 8k 7,346 tok/s.

### Next
1. Serving numbers for the r9k QSA scorer (`~/chain-qsc.out`); if the probe and BetterBench confirm, it stays the
   default (already on) and the 4k-chunk profile is re-taken.
2. Remaining per-family gaps at 4096-token chunks vs Rob (2026-09-25 profile): MoE 177 vs 141 ms (our
   `r9k_moe_mxfp4a8<4,1>` x147 + `4bit_prefill` x49 vs his m64 kernels), elementwise 161 vs 97 (`hc_combine_norm`
   57 vs 41, `hc_gate_mix` 32 vs 27, ~450 eager ATen glue launches in the QSA region = 30 ms).
3. Compressed all-reduce: pipelining message halves across the local link and the uplink (~5% long prefill).
4. Single-request decode: 0.7 ms/step of per-step overhead (MTP drafts, sampling) vs Rob; MTP-4 acceptance
   (his 3.35 vs our 3.13 tok/step).

### Open questions for Brian
- MTP default: MTP-3 recommended (MTP-4 is +4.4% single-stream, -14% at 16 concurrent; `MTP=4` stays a knob).
- Prefix caching off by default for Flash-Next (`PREFIX_CACHE=1` restores cross-request prefix reuse).
- Push: commits ecdf7b8..fd950d4 are local only.
- mxfp4 for the TARGET LM head (Rob's w4a16 does the same): ~0.5 ms/step, changes logits; two null paired evals.
