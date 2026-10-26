# r9700-stack

Tuned GPU kernels and a vLLM plugin that make **Qwen3.8** and **GLM-5.3-Flash** run fast on **AMD Radeon AI PRO R9700**
cards (gfx1201 / RDNA4) on **stock vLLM and stock ROCm**. No fork, no patched source, no vendored binaries, no
re-quantised weights: everything loads at runtime through vLLM's own extension points, so vLLM and ROCm update
underneath it.

Current release: **v0.3.7** (2026-10-26, the vLLM 0.32 line), [release notes](notes/release-v0.3.7.md),
[CHANGELOG.md](CHANGELOG.md).

![r9700-stack v0.3.7: what runs on it, the weight formats, the speed](docs/r9700-stack-v0.3.7.svg)

## What v0.3.7 gives you

- **Stock vLLM 0.32 on stock ROCm 10.** Pinned to the 2026-10-25 ROCm 10 nightly (vLLM 0.32.1rc1, torch 2.13); the
  plugin registers through vLLM's platform, quantization, model and attention-backend entry points and carries its
  own kernels (`libr9k`, HIP and Triton). Five compatibility fixes for the 0.32 line are in this release.
- **Three models, unmodified checkpoints:**
  - **Qwen3.8-27B-NVFP4** (dense) with the **DFlash2-FP8** block drafter: two cards, TP2.
  - **Qwen3.8-Flash-Next** (MoE, 288 routed experts, MXFP4 experts + fp8 dense): four cards TP4, or two cards with the
    experts streamed from pinned host RAM through a device LRU cache.
  - **GLM-5.3-Flash** (sparse MLA with the DeepSeek-V3.2-style indexer, RedHat NVFP4 checkpoint with its MTP layer):
    eight cards TP8, 32k production recipe; 262k launches retrieve correctly to ~75k tokens on this nightly (216k on
    the previous pin, see the release notes).
- **Weight formats, handled at load:** NVFP4 checkpoints converted to MXFP4 with MSE-searched block exponents (folded
  exponent, bit-exact on the 27B); fp8 dense, drafter and lm-head GEMMs with per-token fp8 activations; fp8 KV cache;
  the Flash-Next per-layer-embedding table as int6 in pinned host memory; MXFP4 x fp8 grouped MoE GEMMs with routing,
  activation and top-k sum fused around them.
- **Speed at a 210 W cap per card (single stream / aggregate / prompt processing):**
  - Qwen3.8-27B TP2: **216 tok/s** (4.5 tokens a step with the drafter), 527 / 640 tok/s at 8 / 16 streams, 3.9k tok/s
    on an 8k prompt, 367k-token KV cache.
  - Qwen3.8-Flash-Next TP4: **210-225 tok/s** (MTP-3), 913-944 tok/s at 16 streams, 6.8-7.5k tok/s on an 8k prompt,
    70-80 ms to first token, ~480k-token KV cache.
  - GLM-5.3-Flash TP8: **75-80 tok/s** (MTP-3), 157 tok/s at four streams, 2.4-2.8k tok/s prompt processing at depth,
    needles found at 12.7k / 30k / 67k tokens on this nightly (216k on the previous pin).
- **Speculative decoding on both families:** MTP-3 for Flash-Next and GLM, the DFlash2 drafter for the 27B, all on the
  same kernels, with the draft metadata, indexer logits and all-reduces inside the HIP graphs.
- **Multi-card communication over plain PCIe:** P2P one-shot / two-shot all-reduce and all-gather for 2 and 4 cards,
  a hierarchical 8-rank all-reduce across four PCIe switches, a compressed 4/6-bit all-reduce for prefill-sized
  messages, RCCL above that.
- **Gated like a release, every time:** kernel unit gates, the integration gate against the installed vLLM, strict
  sanity above `max_num_seqs`, mixed-length soaks and long-context needles on every configuration before a tag;
  paired GSM8K (1,319 questions, McNemar) and HumanEval on the numerics (27B: GSM8K 95.75%, HumanEval 162/164;
  GLM: GSM8K 97/100).

## What the kernels do

- **MoE experts**: grouped MXFP4 x fp8 GEMMs with the routing, activation, per-token quant and top-k sum fused
  around them; a device LRU expert cache with the experts streamed from pinned host memory for the two-card
  Flash-Next configuration.
- **Dense layers**: fp8 and NVFP4/MXFP4 weight-only GEMMs on WMMA, the activation quant fused into the GEMM at
  decode widths; NVFP4 checkpoints converted to MXFP4 at load.
- **Hyper-connections** (Flash-Next): the down + hc_silu and up + gated-mean GEMMs on fp8 copies of the weights at
  every row count, the bf16 copies freed after quantisation; combine-norm and gate-mix kernels for prefill.
- **Attention and recurrence**: paged attention for prefill, the query-sparse attention indexer and attention
  kernels, one fused Gated DeltaNet decode core per layer (conv update, gating, recurrence, norm), one GDN state
  page per request under speculative decoding.
- **Communication**: P2P all-reduce and all-gather over PCIe for 2 and 4 cards (one-shot and two-shot, fine-grained
  memory, HIP-graph safe), a compressed 4/6-bit all-reduce for prefill-sized messages, RCCL above that.
- **Speculative decoding**: MTP (Flash-Next) and the DFlash2 block drafter (27B) on the same kernels, with an int6
  per-layer-embedding table in pinned host memory.
- **Memory**: the KV cache sized from what the model actually uses, not torch.compile's transients; cudagraphs
  only where they pay.

Every default ships gated: the full 1,319-question GSM8K set paired per question against the previous numerics
(McNemar), HumanEval, strict sanity under overload, a 25-minute mixed-length soak on every configuration, the
multi-rank communication tests and the kernel unit gates. v0.3.2 kernels (unchanged in v0.3.3): GSM8K 95.75% (paired
p = 1.00 against v0.3.1), HumanEval 162 / 164; v0.3.3 validation: 0 bad answers in 5,060 strict-sanity checks,
0 errors in 1,224 soak requests.

## Speed (210 W with a -42 mV offset, 32k context; the v0.3.7 gate numbers are in the release notes)

**Qwen3.8-Flash-Next (MXFP4 experts, fp8 dense), four cards, TP4, MTP-3**

| | |
|---|---:|
| single-stream decode | 225 tok/s, 14.3 ms a step |
| aggregate decode at 1 / 2 / 4 / 8 streams (BetterBench) | 175 / 277 / 410 / 530 tok/s |
| aggregate decode at 8 / 16 streams (probe) | 620 / 944 tok/s |
| time to first token | 70-80 ms |
| prompt processing at 2k / 8k / 16k / 32k | 5,175 / 7,141 / 7,489 / 7,233 tok/s |
| KV cache | 518,481 tokens |

**Qwen3.8-Flash-Next, two cards, TP2, experts in host RAM**

| | PCIe Gen3 host (test box) | PCIe Gen5 host (production) |
|---|---:|---:|
| single-stream decode | 122 tok/s | 143 tok/s |
| aggregate at 8 / 16 streams | 177 / 163 | 257 / 245 |
| prompt processing at 8k | 2,594 | 3,724 |
| KV cache | 221,184 tokens | |

**Qwen3.8-27B-NVFP4, two cards, TP2, DFlash2 speculative decoding**

| | |
|---|---:|
| single-stream decode | 222 tok/s (4.5 tokens a step), 22 ms a step |
| aggregate at 8 / 16 streams | 527 / 640-658 tok/s (probe; equal-prompt bursts match the previous release) |
| prompt processing at 8k | 3,901-4,143 tok/s |
| KV cache | 367,494 tokens |

How each number was produced, and what was tried and rejected: [PROGRESS.md](PROGRESS.md) and `notes/`.

## Deploying it

Requirements: 2x, 4x or 8x Radeon AI PRO R9700, ROCm 10 on the host, docker, the model weights under `~/models`
(`Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ`, `Qwen3.8-27B-NVFP4` with `Qwen3.8-27B-DFlash2-FP8`,
`GLM-5.3-Flash-NVFP4-RedHat` for eight cards).

1. **Build the image** (stock vLLM ROCm 10 nightly plus the plugin; the kernels compile inside):
   ```
   docker build -t r9700/vllm:dev docker/
   ```
   The pinned nightly (2026-10-25, vLLM 0.32.1rc1, torch 2.13) is in `docker/Dockerfile` (`--build-arg BASE=` for another).
2. **Serve**:
   ```
   bash serve/27b.sh                                                  # 27B-NVFP4, two cards
   GPUS=0,1,2,3 TP=4 OFFLOAD_GB=0 NSEQ=16 bash serve/flashnext.sh     # Flash-Next, four cards
   bash serve/flashnext.sh                                            # Flash-Next, two cards, experts in host RAM
   MODEL=/models/GLM-5.3-Flash-NVFP4-RedHat GPUS=0,1,2,3,4,5,6,7 TP=8 OFFLOAD_GB=0 EAGER=0 MTP=3 KVDTYPE=fp8 \
     MAXLEN=32768 NSEQ=4 NBT=8192 PREFIX_CACHE=0 VLLM_ROCM_USE_AITER=1 R9K_AG_MAX_KB=2048 \
     'SPEC_EXTRA="use_local_argmax_reduction": true' 'R9K_FP8_LINEARS=self_attn\.(in_proj_qkvbfg_a|o_proj|f_b_proj|g_b_proj)$' \
     bash serve/serve.sh                                               # GLM-5.3-Flash, eight cards (notes/glm53.md)
   ```
   OpenAI-compatible API on port 8080 (`PORT=`), model name `Qwen3.8`. Every serve script carries the measured
   defaults with the reason next to each knob; `R9K_*=stock` or `=0` switches any kernel back to vLLM's.
3. **Check it**: `python3 bench/warmup.py`, then `bench/sanity_stress.py` (strict 8-way sanity under load),
   `bench/soak.py` (mixed-length soak), `bench/eval.py` (paired GSM8K), `bench/humaneval.py`. Kernel unit tests
   are in `tests/`; `bench/bringup-tests.sh` runs the single-GPU set in the image.

Notes for the host: cards on the same PCIe switch for tensor parallelism, one card per switch for offloaded
experts; a KVM guest with an emulated PCIe switch needs the overlay in `overlay/emulated-switch`
(`OVERLAYS=emulated-switch`, the serve scripts' default) and the guest kernel flags in
[notes/mes-timeouts.md](notes/mes-timeouts.md); the two-card Flash-Next configuration needs about 120 GB of host RAM.
The release checklist and the upgrade recipe for a newer vLLM nightly are in [notes/picking-up.md](notes/picking-up.md).

## Licence and credits

Apache-2.0 ([LICENSE](LICENSE)); provenance of the kernel structures and the libraries this grew out of in
[CREDITS.md](CREDITS.md) and [notes/independence.md](notes/independence.md).
