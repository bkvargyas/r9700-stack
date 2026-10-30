# GLM-5.3-Flash on eight R9700s (glm53 branch, 2026-10-13/14)

Checkpoint: `coolbho3k/GLM-5.3-Flash-NVFP4-Optimized` (compressed-tensors; NVFP4 routed experts, FP8 weight-only
dense / shared-expert / sparse-attention projections, BF16 KDA projections, no MTP layer). Stock vLLM nightly
`fe4df3df`, stock ROCm 10, unmodified weights. TP=8 on four PLX switches, `serve/serve.sh` through VM100's
`~/glm-launch.sh`.

## What the plugin adds for this model

| piece | where | why |
|---|---|---|
| W8A16 fp8 linears -> W8A8 dynamic | `quant/ct.py get_scheme` | stock serves fp8 weight-only through CUDA Marlin only |
| indexer `is_enabled()` gate | `models/glm5next.py` (proxy) | the ROCm indexer refuses to run unless AITER (gfx9) is on; everything it calls on this model is Triton / in-tree |
| 8-rank all-reduce `ar8` | `comm/r9k_ar8.py` | the flat one-shot collapses at eight senders on four switches (126-288 us); pair / evens-cross / pair = 22 us |
| 8-rank all-gather `ag8` | `comm/r9k_ar8.py` | same for the logits gather (1.0-1.3 ms -> ~0.3) |
| indexer head weights | `attn/sparse_mla.py SkinnyWeight` + `r9k_mm_f32_skinny` | hipBLASLt/rocBLAS take ~200 us on the 1 x 4096 x 32 fp32 mm, 11 per step |
| KDA projections -> per-row fp8 | `models/glm5next.py` post-load swap (`R9K_FP8_LINEARS`) | the model builds them with `quant_config=None`; 4.6B params, 3.4 ms/step in bf16 |
| native NVFP4 routed experts | `quant/nvfp4.py prep_native_moe`, `moe/experts.py` (`nv`), `r9k_moe_nvfp4a8` | the load-time NVFP4->MXFP4 conversion is lossy (~1.55x the checkpoint's own weight error); `R9K_NVFP4_MOE=native` keeps the bits for VRAM-resident experts (opt-in: same speed single stream, 127 vs 129 at four streams, GSM8K 96 vs 97 of 100, MTP acceptance 78.3 vs 78.0%, but ~1.1 GB/card of KV: 322k vs 436k tokens at 8k) |
| sparse MLA on an fp8 (or bf16) pool | `kernels/r9k_sparse_mla.hip`, `attn/sparse_mla.py`, three backend hooks | stock's fp8 path is CDNA assembly, its Triton path bf16-only, aiter's kernel gfx950-only |

The three sparse-MLA hooks: `forward_mqa` -> our kernel; `supports_quant_query_input = False` (the MLA layer would
otherwise hand over an fp8-pre-quantized query); `_use_rocm_sparse_triton` -> True (the metadata builder would
otherwise call `aiter.get_mla_metadata_v1`, a CDNA kernel, and the pending error surfaces at the next launch).

## Numbers (single stream, 512-token decode, temp 0)

| step | tok/s |
|---|---|
| first light, eager | 6.6 |
| HIP graphs (FULL_DECODE_ONLY; the model class has no torch.compile support) | 24-28 |
| two-shot all-reduce | 40-51 |
| ar8 | 42 (stable probe) |
| + skinny indexer mm + KDA fp8 + ag8 | 52.5 |
| fp8 KV cache (our sparse kernel) | 52.9, KV capacity 291k -> 436k tokens at 8k max-len |
| MTP, 1 draft token (Red Hat's unmodified checkpoint, which ships the MTP layer) | 72.3, 75% draft acceptance |
| MTP 1 draft + fp8 KV | 75.2-75.9; four streams 108 aggregate (MTP off: 127-129), so MTP for one or two streams only |
| draft acceptance (1 draft, temp 0) | 72% on a story, 99.5% on a verbatim copy: the proposer path is right, the head is as good as the text is predictable |
| native NVFP4 experts (exact checkpoint bits, opt-in) | 52.8 MTP off, 74.1 MTP 1 draft; GSM8K 96/100 |
| MTP 3 drafts: in-graph draft metadata + 8 KB cross-stage slices + RCCL above 40 KB + local-argmax drafts | **79.6** (78.4 the run before), 2.3 tokens a step on a story; four streams 158 aggregate (second run), 124-129 first; GSM8K 97/100 in 1.6 min; verbatim copy 97.5% accepted |

Concurrency (256-token streams): 1 -> 44, 2 -> 76 aggregate, 4 -> 129 aggregate (native experts: 127).
Kernel level (one MoE layer at the TP8 shapes, both GEMMs + activation): native NVFP4 equals MXFP4 at one row,
is 6-9% slower at 4-256 rows (MT=1) and 37% slower at the MT=2 shape (8,192 routed rows); the exact e4m3 scales
cost ~1.1 GB a card of KV, and draft acceptance with MTP is the same (78.3% vs 78.0%): the default stays MXFP4,
`R9K_NVFP4_MOE=native` is the exact option. Reference (djdeniro, forked
vLLM 0.24 + overlay, rfi quant): 34-39 single stream MTP off, 82-88 with MTP-3, 155 at four streams with MTP.
Ours after the MTP round: 79.6 single stream with 3 drafts, 158 at four streams.

## MTP on this nightly (2026-10-14)

The engine runs the new GPU worker (`vllm/v1/worker/gpu/`) and its `TargetDependentARSpeculator`; the legacy
`EagleProposer` is not on the path. Draft forwards are full-graph captured (decode mode FULL gives the drafter
FULL_DECODE_ONLY). With two or more draft tokens every draft step runs inside ONE graph only when each drafter
attention builder declares `supports_draft_decode_metadata_update`; the ROCm sparse builder does not, so the server
rebuilt metadata eagerly between draft steps. `_install_draft_metadata_update` (glm5next.py) adds the refresh for
the Triton-shaped metadata our kernel serves: the token -> request map and `paged_kv_indptr = cumsum(min(seq_len,
topk))`, both over persistent buffers; `build()` is wrapped to keep the runner's `seq_lens` view. MTP=3: 71.3 -> 73.5
tok/s, acceptance unchanged (1.52 of 3). The drafter is the sparse-attention block only (no KDA), and the indexer
/ kpool builders already supported the update.

Where an MTP=3 step goes (profile, ~34 ms, GPU-bound): the 8-rank all-reduce at four rows is 32 KB a message, above
the 16 KB one-shot limit, and the cross-chain two-shot stage costs 80 us x ~99 = 8 ms; the pair stages 2.3 ms; the
logits all-gather (one per draft step) 2.5 ms; bf16 skinny GEMMs (the Red Hat checkpoint keeps shared experts,
dense MLPs and the MLA projections in bf16) 3.2 ms; MoE 3.5 ms. Draft acceptance itself is fine: 99.5% on a
verbatim copy, 72% on a story at temperature 0 (83% at 0.7 -- stochastic acceptance is looser than argmax equality).

Levers, measured (`tests/test_ar8.py`, us): at 32 KB ar8 96 (RCCL 117), at 64 KB 250 (RCCL 102); a raised one-shot
limit is worse (133 / 311); slicing the cross stage into 8 KB pieces gives 69 / 157 (`R9K_AR8_CROSS_KB`, default 8);
above 40 KB the 8-rank object now declines and RCCL runs the message (`R9K_AR8_MAX_KB`), which alone took the
four-stream MTP=3 aggregate from 87 to 116. `use_local_argmax_reduction` (speculative config; `SPEC_EXTRA` on the
serve.sh MTP branch, the GLM launch helper's default) replaces the per-draft-step logits gather with a per-shard
argmax. Together: 78.4 single stream, 158 at four streams. Not taken: per-row fp8 for the Red Hat checkpoint's
remaining bf16 projections (33 more linears: -6% single stream, lower acceptance, GSM8K 97) and a DMA-engine
all-reduce for 32 KB (the ar4 SDMA mode's flag kernels alone cost 20-28 us each).

## Long context (2026-10-19 to 10-22)

A 262,144-token launch fits (415k KV tokens with a fixed 2.7 GiB KV budget, `KVMEM=2.7`) and serves 236k-token
requests without faults. The fixed budget matters: at 0.94 utilisation a 131k request died in the stock ROCm
indexer's prefill, which materialises a `[step tokens, context]` logits buffer per step and found 0 bytes free.
Prefill: 2,100-2,300 tok/s at 2k-token steps, 3,100 at 8k steps (`NBT=8192`, worth keeping); 32k-token steps run
the cards out of memory in that same buffer. Decode at depth with MTP: 97 tok/s at 73k, 82 at 151k.

Retrieval was unreliable above roughly 12k tokens on this nightly: a unique needle sentence found at 3k and 6k,
missed from 12.7k on, identically with every piece of this stack switched off. The offline reproducer
(`tests/test_indexer_logits_r9k.py`) found the cause on the decode side. The indexer's K cache is written by vLLM's
kpool compress kernel in pages of 64 pooled entries laid out in 16-entry x 16-dim tiles. On gfx942/gfx950 stock reads
it back with aiter's pre-shuffle Gluon kernel; on every other GPU its wrapper falls through to aiter's stage-1
kernel, which reads a linear page. The scores it returns are scrambled: against the torch reference on the same
cache bytes, top-512 overlap 70% at 760 pools, 18% at 3.2k, 8% at 8k, 3% at 32k, and a planted needle is never the
argmax. The prefill side (gather, aiter prefill logits, top-k op) is exact. `attn/indexer_logits.py` is a Triton
kernel that reads the tiled layout and matches the reference to 0.1 on logits of 100-180 at every size up to 65k
pools, for one row and for MTP verify rows, in 92-100 us against the stock kernel's 153-158; `_install_paged_logits`
(glm5next.py) swaps it in where the wrapper would take the fall-through (`R9K_INDEXER_LOGITS=stock` restores).
With it, on a 262k launch, the unique needle is found at 67k, 150k and 216k (prefill 2.7-2.8k tok/s, decode at
depth 61-107 tok/s, acceptance 76-91%); the near-duplicate numbered needle holds to 35k and is hit-or-miss
beyond (the model reasons without concluding), which is the task's difficulty rather than a kernel fault.
GSM8K 95/100, decode probe 74.7 tok/s (the usual 70-80 band).

Stability at 32k: 16 streams, sanity 96/96 after 24k-token prefills (`THINK=1`, the strict check needs the
reasoning in its own field for this model), 15-minute mixed soak 0 errors, VRAM flat.

## Four cards (2026-10-25)

Brian: "Is there any way to run GLM-5.3-Flash with just 4 or 6 cards?" Six is out (64 attention heads: TP must divide
64, and PP would still need the experts offloaded). Four runs, with the routed experts in pinned host memory (stock
`--cpu-offload-params experts`) and the plugin's device LRU expert cache, `~/glm-launch4.sh` on VM100:

- Per rank after the MXFP4 conversion: 40.5 GB of routed experts (288 x 42 x 3.35 MB) + ~8 GB of everything else.
  The guest has 251 GB; each worker ends up with ~43 GB of pinned memory plus ~4 GB anon regardless of
  `OFFLOAD_GB` (24: 43.4 GB; 30 and 44 were OOM-killed while loading), so 4 ranks = ~188 GB and nothing else may
  run on the box while a 4-rank server loads (a gate run on card 4 OOM-killed attempt 3 in graph capture).
- The config that fits: `OFFLOAD_GB=24 R9K_EXPERT_CACHE_GB=2 R9K_EXPERT_CACHE_SLOTS=0 MAXLEN=16384 NSEQ=4 NBT=4096
  KVMEM=1.0 MTP=3 KVDTYPE=fp8` (24.4 GiB of weights per card leaves room for a 2 GiB cache and a 55k-token KV).
  Ready in 659 s.

| four cards (TP4, experts offloaded) | eight cards (TP8, v0.3.5 record) |
|---|---|
| decode 8.1 tok/s single; 10.8-11.5 aggregate at four streams | 79.6 single; 158 at four |
| prefill 636 tok/s (12.7k prompt, ttft 20 s) | 5,000+ |
| draft acceptance 43-57 % (79 % on the needle probe) | 70-80 % |
| 85 / 12.7k needle correct | correct to 216k |

- Host memory per rank is about **2 x OFFLOAD_GB** (shmem per worker: 43 GB at 24, 76 GB at 40): stock's UVA parameter
  copy and the plugin's fragment-order pinned copy (`moe/cache.py`) coexist. A 40 GB/rank, 12 GiB-cache run needs
  ~320 GB of pinned memory for four ranks -- it was OOM-killed even in the 384 GiB guest while a 2-card eager server
  shared the box, and **was OOM-killed again alone in the 384 GiB guest** (22:09: 4 x 76 GB shmem + 26 GB anon + the
  loader's page cache exceed 377 GiB). Freeing stock's copy after the re-layout would halve this (open item); until
  then the only four-card config that loads is the 24 GB / 2 GiB-cache one above.

Ten times slower: the 2 GiB cache (~0.6 % of the experts) misses on most steps and every missed expert streams over
PCIe 3 at 11-13 GB/s. A useful four-card server would need all experts host-resident (`OFFLOAD_GB=40`, freeing ~12 GiB
of VRAM per card for the cache), which this guest cannot hold: VM100's RAM ceiling is 256 GiB because guest RAM
(GPA 1 TiB upward) meets chain B's GPA==HPA placement at 0x140 (the 400 GiB attempt broke every card's probe;
384 GiB is possible only by moving chain B to 0x160 -- host notes). Even then the PCIe stream, not the cache,
bounds decode at perhaps 15-20 tok/s. Verdict: feasible, correct, not worth running; eight cards stay the config.

## Qwen gates on this tree (2026-10-22, `~/validate-glm53-qwen.sh`)

The production configurations must not move; they did not. 27B TP2: single-stream 219.4 tok/s (v0.3.3 ~221),
4.5 tokens a step, 515 / 585 at 8 / 16 streams, 8k prefill 3,830; 26-minute soak 286 requests 0 errors, VRAM flat;
strict sanity 0 bad of 2,900. Flash-Next TP4: 223.1 tok/s single-stream (v0.3.3 225.0), 3.11 tokens a step, 610 /
930 at 8 / 16 streams, 8k prefill 6,847; BetterBench combined 185.3 (v0.3.3 188.0), update p99 15.2 ms, prompt
processing 7.1-7.4k tok/s; 25-minute soak 806 requests 0 errors, VRAM flat; strict sanity 0 bad of 5,380. Comm tests
(ag_nrank, ar_nrank 4 ranks, ar_race) and every single-GPU kernel gate pass. No server errors, no MES timeouts.

## The fabric cliff (measured, `tests/test_ar_nrank.py`, `tests/test_ar8.py`)

Four ranks on any two switches: one-shot 6.8-10 us. Eight ranks on four: 126-288 us. Four ranks on four switches
with the other four idle: 7.8-10 us. Eight concurrent cross-chain senders at >= 6 KB each: 140-250 us; at 4 KB:
8 us. Any switch-local phase adjacent to a both-quartets cross phase: 65-75 us. Two independent TP=4 groups, or
one TP=4 plus two TP=2 groups, do not interfere at all. So only the single 8-rank group needs the hierarchical
collectives, and only the even ranks ever send cross-chain.

## Knobs

`KVDTYPE=fp8`, `R9K_SPARSE_MLA=auto|r9k|stock`, `R9K_SPARSE_MLA_SPLITS` (8), `R9K_INDEXER_MM=0`, `R9K_AR8=0`,
`R9K_AR8_FP32=1` (exact, 45 us), `R9K_AG8=0`, `R9K_AG_MAX_KB` (2048 for the logits at 4 tokens), `R9K_AR8_CROSS_HOST=0` (peer-write
cross stage; default 1 = pinned host memory, below), `R9K_AR8_HOST_MIN_KB` (12) / `R9K_AR8_HOST_MAX_KB` (96) / `R9K_AR8_HOST_1S_KB` (128),
`R9K_AR8_PAIR_1S_KB` (64), `R9K_AR8_HOST_AG=0` (logits all-gather on the peer path), `R9K_ARN_HOST=1` (every N-rank pool in host memory),
`R9K_FP8_LINEARS=<regex>`, `R9K_NVFP4_MOE=mxfp4|native` (default converts; native = exact bits for VRAM-resident experts, offloaded experts
always convert, the expert cache is MXFP4-only), `R9K_DISABLE=glm5next`. Everything keys on world size 8, the glm5next module path or
checkpoint content the Qwen configurations do not have.

## Open

GSM8K (100 questions, four streams, medium reasoning) on the fp8-KV configuration: 97/100, no truncations.
MTP works on this nightly with the stock proposer and the same hooks (no proposer patch, unlike the vLLM 0.24
overlay). Open: more draft tokens, fp8 KV together with MTP, longer contexts, the next profile, HumanEval, and the
QFN / 27B gates on the glm53 tree before any merge.

## Production server through the full checklist (2026-10-29, v0.3.7, 32k recipe, in place)

Brian: "Can we do some benchmarks of GLM while we wait?" (the DeepSeek-V4.1-Flash queue, notes/radiance-test-run.md). The production
server as it ran: deployed v0.3.7 tree on `r9700/vllm:dev` (e4a5b553), the 32k recipe (RedHat NVFP4, TP8, MTP-3, fp8 KV, NSEQ 4,
NBT 8192), KV cache 177,283 tokens, up 17 h, SCLK -500 MHz on all eight cards (the 120 V rule), 210 W / -42 mV. `~/glm-bench-1029.sh`
-> `~/glm-bench-1029.log`, BetterBench `~/glm-bench-1029.bb.log`, evals `~/.r9keval/glm53-prod-1029.json`.

| | production GLM 2026-10-29 | v0.3.7 gate (10-26) | v0.3.5 record |
|---|--:|--:|--:|
| probe: decode mean (code / prose / json / math) | **89.0** (90.5 / 75.8 / 94.4 / 95.4), 2.86 tokens a step | 75.0 | 79.6, 2.3 a step on a story |
| probe: 8 / 16 streams (4 run at once), 8k prefill | 138 / 163, 3,148 tok/s | 157 at four, 2,466 on the needle | 158 |
| strict sanity, thinking on: long prompts / 5 streams / 8 streams / after the soak | 0 bad of 160 / 300 / 320 / 200 | 0 of 780 | |
| GSM8K, all 1,319, no thinking, conc 4 | **95.98%** (1,266) -- the first full run on GLM | 100-question samples 95-97 | |
| HumanEval 164 (thinking) | 151 (92.1%) at the client's 3,072-token budget; **156 (95.1%) at 8,192** (`~/humaneval8k.py`, 22 min): five of the thirteen were cut off, eight remain (32, 38, 47, 50, 95, 97, 132, 145) against 2-4 for the Qwen models | | |
| BetterBench combined / update p99 / TTFT p50 | 91.4 / 32.1 ms / 230 ms | | |
| BetterBench 1 / 2 / 4 / 8 streams (aggregate) | 86.5 / 116.5 / 157.9 / 157.5 (8 queue behind NSEQ 4: TTFT p50 9.5 s) | | |
| prompt processing 2k / 8k / 16k / 32k | 3,343 / 3,284 / 3,137 / 3,103 tok/s | | |
| soak, 16 clients, 200-30k tokens, 977 s | 156 ok, 0 errors, 2,445 prompt tok/s, median prompt 15.6k, latency 98 s | 155 in 964 s | |
| server errors; VRAM before / after | 0; 31.1-31.4 -> 32.2 GiB a card (the KV pool filled by the soak) | | |

Reading it: the step is unchanged (~31 ms, BetterBench update p99 32.1 ms) and the gain over the gate is acceptance -- 2.86 tokens a
step on the probe's code / json / math prompts against 2.3 on prose and stories; prose alone is 75.8, inside the old 70-80 band. Four
streams give 1.8x one, eight give nothing more because NSEQ is 4 (and sized by the 32k context against the 177k-token pool); the
server is bound by the per-layer all-reduce across the PLX switches, not by the cards (the -500 MHz offset is invisible). Quality:
GSM8K 95.98 on the full set, the same number radiance's Flash-Next posts; HumanEval 151 against 160-162 for the Qwen models under the
same client -- GLM reasons at length: at 8,192 tokens five of the thirteen pass and it lands at 156, four to six behind the Qwen models. Thermals under the
run (ROCm telemetry inside the container; the guest's sysfs reads a flat 18 C through the passthrough): edge 43-52 C, junction 49-62
C (chains C/D ~10 C warmer than A/B), memory 52-60 C, 134-135 W a card on decode, on-card fans 1,040-1,070 rpm; host BMC: inlet 41 C,
CPU 68-70 C, chassis GPU fans 8,250-8,400 rpm, wall 1,500-1,560 W under the bench, 415 W idle; the BMC's "maximum during sampling
period" of 2,960 W is the power-on inrush after the 10-28 reboot (not reached again).

## The eight-rank collective through host memory (2026-10-30, master after v0.3.7)

Brian: "Look into the eight-rank collective on our stack" -- radiance decodes this model at 152 tok/s on the same eight cards and
the same PLX topology where ours did 86-89, with a 24 ms step against our 32. Method: a torch-profiler trace of a 256-token decode
on the production recipe (vLLM's `--profiler-config {"profiler":"torch","torch_profiler_dir":"/prof"}` through `EXTRA=`, the
old `VLLM_TORCH_PROFILER_DIR` env no longer exists on this build; `~/prof/analyze.py` sums kernel time by class, `~/prof/gaps.py`
the GPU gaps), then `tests/test_ar8.py` sweeps on the idle box, then an A/B and the full gate from `~/r9700-build/repo-ar8`.

**Where the step went (rank 2 of the peer-write build, 89 steps at 2.76 tokens a step):** GPU kernel time 25.7 ms a step of a
29-32 ms step. Collectives 8.6 ms (33%): the sliced cross stage 361 one-shots x 10.8 us = 3.9 ms, the switch-local pair stage
running as a 14 us two-shot twice an all-reduce (32 KB is over the 16 KB one-shot cap) 2.5 ms, the per-step logits all-gather
(4 rows x 154,880 bf16, 1.24 MB a rank through the cross stage) 1.41 ms, RCCL for the 40 KB+ messages 0.5 ms. GEMMs 7.4 ms
(29%: the Red Hat checkpoint's bf16 projections on `wvSplitK` at 25 us x 114 a step, our fp8 linears 0.9 ms, hipBLASLt 1 ms).
MoE 4.9 ms (19%: `r9k_moe_mxfp4a8` 3.4, top-k 0.8). Glue 3.9 ms (the mHC mix 1.9, casts, copies). Attention 0.3 ms. The
remaining 3-6 ms a step is GPU idle: launch gaps and the inter-step scheduler (the profiler itself stretches these; unprofiled the
step is 29.4 ms on the code prompt, 94 tok/s).

**The cross stage pays per kilobyte.** `tests/test_ar8.py` on the idle box, us an all-reduce under HIP graphs:

| message | RCCL | flat two-shot | ar8, peer writes (production to v0.3.7) | ar8, cross stage through host memory |
|---|--:|--:|--:|--:|
| 8 KB (1 row) | 82 | 40 | **21.6** | 31.6 |
| 16 KB (2 rows) | 87 | 133 | 41.6 | **35.9** |
| 32 KB (4 rows, the MTP-3 verify step) | 120 | 244 | 71.7 | **47.4** |
| 64 KB (8 rows) | 104 | 446 | 163 (RCCL in production) | **60.6** |
| 128 KB (16 rows, four streams) | 109 | 872 | RCCL | 103 (one-shot) / 126 (two-shot) |
| 256 KB | **128** | 1,982 | RCCL | 153 |

The peer-write stage grows linearly with bytes because the four evens' stores cross four root complexes and stall past ~100 KB in
flight (the cliff above); slicing into 8 KB one-shots only serialises the stall. The host path is the same one-shot / two-shot
kernels with the exchange pools in pinned host memory: `R9kAllReduceN(host=True)` creates a `/dev/shm` file (scratch | flags),
maps and pins it with `hipHostRegister(Portable | Mapped)` in every rank of the sub-group, and hands the device pointers to the
kernels in place of the IPC-opened VRAM ones. A push is then one link to the host, a read one link back; nothing crosses a root
complex; the bits are those of the peer path. It costs a ~10 us round trip, so 1-row messages keep the peer path; from 12 KB to
96 KB the host stage carries the whole message in one shot; 128 KB and up stay on RCCL (103-148 there against RCCL's 109, and the
four-stream probe lost 5-10% when the host stage took them). Two more changes rode along: the pair stage keeps its one-shot to
64 KB (`R9K_AR8_PAIR_1S_KB`; switch-local, no cliff), and the logits all-gather goes through the host stage as well.

**First try deadlocked.** The evens accepted messages to 128 KB on the new path while the odds still capped at 40 KB and ran RCCL:
each pair sat in two different collectives waiting for the other, the first forward hung, and `py-spy dump` showed every worker
spinning in `forward_mqa`'s stream sync. Rule, now in the code: every size decision reads group-wide state (`host_on`,
`host_min`, `host_max`) agreed by `all_gather_object` at startup, never the evens' objects.

**Gate (repo-ar8 = v0.3.7 + the two comm files, production recipe, `~/ar8-gate.sh` -> `~/ar8-gate.log`):**

| GLM-5.3-Flash TP8 | peer writes (production bench 10-29 / A/B 10-30) | host-memory cross stage (gate 10-30) |
|---|--:|--:|
| probe: decode mean (code / prose / json / math), tokens a step | 89.0 (90 / 76 / 94 / 95), 2.86 / 86.5, 2.76 | **100.0** (105 / 84 / 108 / 103), 2.86; 100.4 and 101.8 on the two A/B launches |
| probe: 8 / 16 streams; 8k prefill | 138 / 163; 150 / 156 | 129 / 156; 3,148 |
| BetterBench combined / update p99 / TTFT p50 | 91.4 / 32.1 ms / 230 ms | **103.3** / **28.5 ms** / 258 ms |
| BetterBench 1 / 2 / 4 / 8 streams | 86.5 / 116.5 / 157.9 / 157.5 | **94.6 / 127.3 / 161.1 / 160.4** |
| prompt processing 2k / 8k / 16k / 32k | 3,343 / 3,284 / 3,137 / 3,103 | 3,361 / 3,289 / 3,140 / 3,102 |
| strict sanity, thinking on | 0 bad of 980 | 0 bad of 980 |
| GSM8K 1,319, no thinking, conc 4 | 95.98% | **95.98%**; paired 13 / 13 discordant, McNemar p = 1.0, 21% identical outputs |
| HumanEval 164 (3k budget) | 151 | 152 |
| soak, 16 clients, 200-30k tokens | 156 ok in 977 s, 0 errors, 2,445 prompt tok/s | 158 ok in 997 s, 0 errors, 2,453 |
| server errors; VRAM after | 0; 32.2 GiB a card | 0; 32.1 GiB a card |

So the step drops from 32 to 28.5 ms and single-stream decode gains 13-16% with identical acceptance and equal quality (the 21%
identical outputs are the 40-96 KB messages that moved from RCCL's seven roundings to ar8's two; accuracy is the same to the
question). The host stage is the default from this commit (`R9K_AR8_CROSS_HOST=0` restores peer writes); the production GLM recipe
needs no new flag. What remains of the gap to radiance's 24 ms is the compute side: the bf16 skinny GEMMs (7.4 ms), MoE at four
rows (4.9), the mHC glue (3.9), and the 3-6 ms the GPU waits on the CPU between launches.
