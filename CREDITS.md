# Credits

This project stands on other people's work. What we use, where it came from, and its license status as of
2026-09-19. Items marked **license pending** have no license file upstream; ask the authors before publishing
anything derived from them.

| What | Source | Used in | License |
|---|---|---|---|
| MXFP4 x FP8 GEMM design (fragment-order weights, folded e2m1->e4m3 unpack, fp8 WMMA, in-block split-K) | libr4d, StillDeadcode -- https://codeberg.org/StillDeadcode/libr4d (5dc6302) | `kernels/r9k_moe_mxfp4a8.hip` (derived) | **license pending** |
| libr4d 2-rank / N-rank P2P all-reduce (`r4d.so`, loaded at runtime) | libr4d + tcclaviger fork -- https://codeberg.org/tcclaviger/libr4d | `r9700_vllm/comm/r4d_ar.py` | **license pending** |
| Device-side expert LRU cache kernels | davetha (vendored) | `kernels/third_party/davetha/` | Apache-2.0 |
| Chat template `qwen-fixed-v22.3.jinja` (copied verbatim) | GGZ14/vllm-mxfp4 -- https://github.com/GGZ14/vllm-mxfp4 (92eed82) | `serve/templates/` | **license pending** |
| Ideas: NVFP4->MXFP4 requant, GDN in_proj merge, fp8->MXFP4 for dense layers, decode-band tiling, NT loads, DFlash drafter handling | GGZ14/vllm-mxfp4 (radiance) | `r9700_vllm/quant/`, `r9700_vllm/models/gdn.py`, kernels | ideas only, no code copied |
| Qwen3.8-Flash-Next GPTQ checkpoint, int6 PLE format, DFlash2 drafter, `tcclaviger/vllm` reference image | tcclaviger | serving, format compatibility | per model cards |
| Kernel review and tuning (loop rewrite, fp32 MXFP4 scales, NT loads) | Fable (Anthropic) review, 2026-09-19 | `kernels/r9k_moe_mxfp4a8.hip`, `tuning/fable_bench.py` | this repo |
| vLLM, ROCm, PyTorch, Triton | upstream projects | runtime (stock, unmodified) | Apache-2.0 / MIT / BSD |
