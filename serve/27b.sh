#!/bin/bash
# Qwen3.8-27B-NVFP4 on 2x R9700 (TP2), the configuration measured on 2026-09-20:
#   BetterBench combined decode 189.3 tok/s, conc 165/274/404/522, prefill 3838-3902 tok/s (2k-32k),
#   GSM8K-500 ~96.6-97.4%, HumanEval ~96-98%. Same-box GGZ14 radiance: 196.5 / 177-549 / 4776-4950.
# Every knob below is measured; see PROGRESS.md for what each one bought and what was rejected.
exec env \
  OVERLAYS=${OVERLAYS-emulated-switch} `# host overlay, not the product: VM100 on the .100 PLX box
                                        # needs the hostcall-free RCCL or every collective fails at
                                        # launch ("operation cannot be performed in the present state").
                                        # OVERLAYS= (empty) on a host that does not need it.` \
  R9K_FOLD=1 `                   # folded-exponent MXFP4 (bit-exact on this checkpoint): +5% prefill` \
  R9K_NVFP4=mxfp4 `              # NVFP4 -> MXFP4 at load: +6% everywhere (R9K_NVFP4=native keeps the checkpoint format)` \
  R9K_FP8_TO_MXFP4=1 `           # fp8 attention/GDN/last-8-MLP layers -> MXFP4` \
  R9K_BF16_TO_MXFP4=in_proj_ba ` # so the GDN in_proj pair merges into one GEMM` \
  R9K_AR_QUANT=1 `               # compressed all-reduce >= 128 KB: prefill all-reduce 910 -> 378 ms, no measurable quality cost` \
  `# Attention is OUR kernel by default (R9K_PAGED_ATTN=r9k): 99.8% of libr4d, so the dependency costs nothing.` \
  `# The all-reduce still defaults to libr4d (R9K_AR_IMPL=r4d). Add R9K_AR_IMPL=r9k for a FULLY libr4d-free` \
  `# build: ~11% prefill and ~4% decode, identical GSM8K. See notes/independence.md. Brian's call which to ship.` \
  VLLM_KV_CACHE_LAYOUT=LBHNC `   # libr4d paged attention needs contiguous per-head slots (serve.sh sets this for ATTN=CUSTOM anyway)` \
  KVMEM=9 `                      # fixed KV budget: vLLM's estimate OOMs once load-time requant is on` \
  MODEL=/models/Qwen3.8-27B-NVFP4 OFFLOAD_GB=0 MTP= \
  DRAFT=/models/Qwen3.8-27B-DFlash2-FP8 SPEC=7 `# DFlash2 speculative decoding: ~3x` \
  ATTN=CUSTOM DRAFT_ATTN=CUSTOM `# libr4d prefill attention + split-KV verify` \
  CHAT_TEMPLATE=qwen-fixed `     # +14% acceptance vs the checkpoint template (CREDITS.md)` \
  NSEQ=8 "${@}" \
  bash "$(dirname "$(realpath "$0")")/serve.sh"
