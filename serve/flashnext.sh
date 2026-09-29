#!/bin/bash
# Qwen3.8-Flash-Next (tcclaviger GPTQ). Two ways to run it:
#   TP2 with experts in host RAM (the defaults here; needs the 256 GB VM): the plugin's LRU expert cache keeps
#     270 slots per layer in VRAM. Full BetterBench 2026-09-29 (v0.2.0): single decode 93.8 tok/s (probe 114),
#     step 25.7 ms, TTFT 555 ms, prefill 2.1k / 3.2k / 3.4k / 3.3k tok/s at 2k..32k, conc 84 / 104 / 108 / 111 / 95.
#     Prefill and concurrency are bound by the expert stream over PCIe. On a 4-card box use ONE CARD PER PLX
#     SWITCH (GPUS=0,2): the stream comes down each switch's single Gen3 uplink, and the same-switch pair
#     measured 81 / 54 / 1,379 (single / conc-8 / prefill 8k) against 114 / 132 / 2,464 on the split pair.
#     CGSIZES= (vLLM's default graph sizes) at TP2: the 2048-token prefill graphs leave no KV room there.
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
