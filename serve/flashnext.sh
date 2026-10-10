#!/bin/bash
# Qwen3.8-Flash-Next (tcclaviger GPTQ). Two ways to run it:
#   TP2 with experts in host RAM (the defaults here; needs the 256 GB VM): the plugin's LRU expert cache keeps
#     270 slots per layer in VRAM. Full BetterBench 2026-09-29 (v0.2.1): single decode 97.4 tok/s (probe 117),
#     step 25.6 ms, TTFT 484 ms, prefill 2.2k / 3.6k / 3.8k / 3.8k tok/s at 2k..32k, conc 87 / 109 / 116 / 118 / 113.
#     Prefill and concurrency are bound by the expert stream over PCIe. On a 4-card box use ONE CARD PER PLX
#     SWITCH (GPUS=0,2): the stream comes down each switch's single Gen3 uplink, and the same-switch pair
#     measured 81 / 54 / 1,379 (single / conc-8 / prefill 8k) against 114 / 132 / 2,464 on the split pair.
#     CGSIZES= (vLLM's default graph sizes) at TP2: the 2048-token prefill graphs leave no KV room there.
#     RESTART ONCE after the first launch of a new configuration: the launch that compiles keeps ~0.45 GiB
#     that the next one gives to the KV pool (45k vs 72k tokens at NSEQ=16, 94k vs 121k at NSEQ=8).
#     Memory utilization is 0.96 here (serve.sh sets it for TP2 with offload; UTIL=0.94 for the old value):
#     KV cache 129k tokens on the first launch, 156k restarted, against 95k / 121k at 0.94.
#     What bounds it (PROGRESS.md 2026-09-29): the link. Every routed expert that is not resident is 1.245 MiB
#     per rank over PCIe at the link rate, so real traffic levels off near 115 tok/s from 4 requests up; eight
#     copies of ONE prompt share their experts and run at 434-577. R9K_EXPERT_CACHE_STATS=1 logs the misses.
#   TP4 with everything in VRAM: GPUS=0,1,2,3 TP=4 OFFLOAD_GB=0 NSEQ=16 (the headline numbers in README.md).
#     2026-10-07 defaults (notes/prefill-fp8-pipe.md): the fp8 hyper-connection up GEMM + gate mix (+4% prefill),
#     memory utilization 0.98 and graphs for prefill chunks <= 256 tokens: KV cache 279k -> 442-475k tokens
#     (vLLM's compile-time transients were being counted as activation; the model's real peak is 0.6 GiB),
#     25-minute soak 684 ok at 31.8 of 32.6 GB peak. KVMEM=7.0 pins the budget (475k) when the launch-to-launch
#     estimate matters; 7.5 was the measured edge (440 MiB free), not a setting.
# Attention is the model's own QSA (ATTN=CUSTOM is for standard-attention models only).
exec env \
  OVERLAYS=${OVERLAYS-emulated-switch} `# host overlay, not the product: VM100 on the .100 PLX box
                                        # needs the hostcall-free RCCL or every collective fails at
                                        # launch ("operation cannot be performed in the present state").
                                        # OVERLAYS= (empty) on a host that does not need it.` \
  R9K_FOLD=1 `        # folded-exponent MXFP4 experts (bit-exact on this checkpoint): +5% prefill, +4% conc-8` \
  R9K_AR_QUANT=1 `    # wht6 all-reduce >= 128 KB (decode single-stream messages stay exact)` \
  MTP=3 `             # the checkpoint's own MTP head` \
  NSEQ=${NSEQ-8} `    # requests running at once. Each holds 18 KV blocks whatever its length (vLLM: 4 state
                      # groups x (1 + 3 MTP blocks) + 2 attention blocks), and the graphs for 16 sequences take
                      # ~0.9 GiB from the KV pool: at TP2 with offloaded experts NSEQ=16 ran 4-6 requests at
                      # once, NSEQ=8 runs 8 (conc-8, one prompt type: 283 -> 401 tok/s). NSEQ=16 for TP4.` \
  PREFIX_CACHE=${PREFIX_CACHE-0} `  # off: avoids the mamba-aligned prefill chunking (2k prefill 2.8k -> ~4.7k tok/s,
                                    # see serve.sh); =1 restores prefix reuse across requests` \
  CGSIZES=${CGSIZES-1,2,4,8,16,24,32,48,64,96,128,192,256} `  # graphs for prefill chunks <= 256 tokens
                                    # (short-prompt TTFT 336 -> 91 ms). The sizes up to 2048 cost 1.4 GiB of graph
                                    # memory a card and 1.4 GiB of vLLM's activation estimate for no soak-measured
                                    # throughput (6677 vs 6726 prompt tok/s): +94k KV tokens at TP4 without them.
                                    # CGSIZES= for vLLM's default (max 512)` \
  UTIL=${UTIL-$([ "${TP:-2}" = 4 ] && echo 0.98)} `  # TP4: 0.98 (soaked: 31.8 of 32.6 GB at the peak); TP2 keeps
                                    # serve.sh's 0.96 / 0.94` \
  R9K_HC_FP8=${R9K_HC_FP8-r9k} `    # the hyper-connection up GEMM + sigmoid-gated mean fused in fp8 from 256 rows
                                    # (268 vs 674 us a layer at 4096): +4% prefill, decode untouched; GSM8K paired
                                    # p=0.86, HumanEval 160/164. =stock for the bf16 pair` \
  R9K_HC_FP8_DECODE=${R9K_HC_FP8_DECODE-$([ "${TP:-2}" = 4 ] && echo 1 || echo 0)} `  # TP4: the hyper-connection
                                              # down + up GEMMs at decode widths (<= 16 tokens) on the same fp8 copies
                                              # (bf16 WMMA, W8A16): 16.42 -> 15.33 ms/step, 2026-10-09; GSM8K paired
                                              # with the fused quant below, HumanEval 161/164 (notes/decode-step.md).
                                              # Off at TP2: the fp8 down copy there costs 30k KV tokens (133k -> 103k)
                                              # for +10% single-stream and nothing at 8 streams. =0 / =1 overrides` \
  R9K_FP8_BLOCK=${R9K_FP8_BLOCK-block} `   # block-fp8 projections on our split-K GEMM at decode widths, stock's
                                           # Triton kernel above M=64 (R9K_FP8_BLOCK_MAXM); 2026-09-24: +6% decode` \
  R9K_FP8_QA=${R9K_FP8_QA-1} `             # ... with the per-token fp8 quant fused into that GEMM (same operands bit
                                           # for bit, 96 launches fewer a step): 15.33 -> 15.01 ms/step. =0 splits it` \
  R9K_DRAFT_LMHEAD=${R9K_DRAFT_LMHEAD-mxfp4} `   # MTP draft head at 4 bits: draft-only, cannot change outputs` \
  R9K_TARGET_LMHEAD=${R9K_TARGET_LMHEAD-mxfp4} ` # target head at 4 bits (as Rob's w4a16): ~0.5 ms/step. Paired evals
                                                 # 2026-09-24, conc=1: 300 short 96.0 -> 97.0% p=0.51; 800 CoT
                                                 # 97.62 -> 97.50% p=1.0. =fp8 restores the exact-er head.` \
  "${@}" \
  bash "$(dirname "$(realpath "$0")")/serve.sh"
