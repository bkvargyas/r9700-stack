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
  "${@}" \
  bash "$(dirname "$(realpath "$0")")/serve.sh"
