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
#     What bounds it (PROGRESS.md 2026-09-29): the link. Every routed expert that is not resident is 1.245 MiB
#     per rank over PCIe at the link rate, so real traffic levels off near 115 tok/s from 4 requests up; eight
#     copies of ONE prompt share their experts and run at 434-577. R9K_EXPERT_CACHE_STATS=1 logs the misses.
#   TP4 with everything in VRAM: GPUS=0,1,2,3 TP=4 OFFLOAD_GB=0 (the headline numbers in README.md).
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
  CGSIZES=${CGSIZES-1,2,4,8,16,24,32,48,64,96,128,192,256,384,512,768,1024,1280,1536,1792,2048} `  # graphs for
                                    # prefill chunks <= 2048 tokens (short-prompt TTFT 336 -> 91 ms); CGSIZES= for
                                    # vLLM's default (max 512)` \
  R9K_FP8_BLOCK=${R9K_FP8_BLOCK-block} `   # block-fp8 projections on our split-K GEMM at decode widths, stock's
                                           # Triton kernel above M=64 (R9K_FP8_BLOCK_MAXM); 2026-09-24: +6% decode` \
  R9K_DRAFT_LMHEAD=${R9K_DRAFT_LMHEAD-mxfp4} `   # MTP draft head at 4 bits: draft-only, cannot change outputs` \
  R9K_TARGET_LMHEAD=${R9K_TARGET_LMHEAD-mxfp4} ` # target head at 4 bits (as Rob's w4a16): ~0.5 ms/step. Paired evals
                                                 # 2026-09-24, conc=1: 300 short 96.0 -> 97.0% p=0.51; 800 CoT
                                                 # 97.62 -> 97.50% p=1.0. =fp8 restores the exact-er head.` \
  "${@}" \
  bash "$(dirname "$(realpath "$0")")/serve.sh"
