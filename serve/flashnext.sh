#!/bin/bash
# Qwen3.8-Flash-Next (tcclaviger GPTQ) on 2x R9700 (TP2), measured 2026-09-20:
#   single decode 84.5 tok/s, conc-8 206, prefill ~2165 tok/s, MTP-3 acceptance 2.75, GSM8K-500 97.6-97.8%.
# Experts stream from pinned host memory (needs the 256 GB VM); the plugin's LRU expert cache keeps 270 slots
# per layer in VRAM. Attention is the model's own QSA (ATTN=CUSTOM is for standard-attention models only).
exec env \
  OVERLAYS=${OVERLAYS-emulated-switch} `# host overlay, not the product: VM100 on the .100 PLX box
                                        # needs the hostcall-free RCCL or every collective fails at
                                        # launch ("operation cannot be performed in the present state").
                                        # OVERLAYS= (empty) on a host that does not need it.` \
  R9K_FOLD=1 `        # folded-exponent MXFP4 experts (bit-exact on this checkpoint): +5% prefill, +4% conc-8` \
  R9K_AR_QUANT=1 `    # wht6 all-reduce >= 128 KB (decode single-stream messages stay exact)` \
  MTP=3 `             # the checkpoint's own MTP head` \
  R9K_FP8_BLOCK=${R9K_FP8_BLOCK-block} `   # block-fp8 projections on our split-K GEMM at decode widths, stock's
                                           # Triton kernel above M=64 (R9K_FP8_BLOCK_MAXM); 2026-09-24: +6% decode` \
  R9K_DRAFT_LMHEAD=${R9K_DRAFT_LMHEAD-mxfp4} `   # MTP draft head at 4 bits: draft-only, cannot change outputs` \
  R9K_TARGET_LMHEAD=${R9K_TARGET_LMHEAD-mxfp4} ` # target head at 4 bits (as Rob's w4a16): ~0.5 ms/step. Paired evals
                                                 # 2026-09-24, conc=1: 300 short 96.0 -> 97.0% p=0.51; 800 CoT
                                                 # 97.62 -> 97.50% p=1.0. =fp8 restores the exact-er head.` \
  "${@}" \
  bash "$(dirname "$(realpath "$0")")/serve.sh"
