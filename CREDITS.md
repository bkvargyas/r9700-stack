# Credits

This project stands on other people's work. What we use, where it came from, and its license status as of
2026-09-21.

**Permission granted, not yet a license.** Brian reports that StillDeadcode (libr4d) and GGZ14 (vllm-mxfp4) both
confirmed by DM on 2026-09-21 that our use is fine. That is permission from the rights holders and it is recorded
here while it is fresh -- but neither upstream repository ships a LICENSE file yet, so the grant is personal to
this project: **it does not pass to anyone who forks or vendors this code.** Asking each author to add a LICENSE
file, or to repeat the sentence in a public issue, would make it durable and transferable. Drafts for that ask
are in `notes/license-requests.md`.

Work is under way to remove the libr4d dependency entirely -- see `notes/independence.md` for what has been
replaced, what each piece is worth in measured tok/s, and the rule we follow when writing replacements (public
algorithms only; we never disassemble `r4d.so` or read libr4d source).

| What | Source | Used in | License |
|---|---|---|---|
| MXFP4 x FP8 GEMM design (fragment-order weights, folded e2m1->e4m3 unpack, fp8 WMMA, in-block split-K) | libr4d, StillDeadcode -- https://codeberg.org/StillDeadcode/libr4d (5dc6302) | `kernels/r9k_moe_mxfp4a8.hip`, `kernels/r9k_gemm_fp8.hip` (derived) | permission by DM 2026-09-21; no upstream LICENSE |
| libr4d 2-rank / N-rank P2P all-reduce (`r4d.so`, loaded at runtime) | libr4d + tcclaviger fork -- https://codeberg.org/tcclaviger/libr4d | `r9700_vllm/comm/r4d_ar.py` (default; `R9K_AR_IMPL=r9k` uses ours) | permission by DM 2026-09-21; no upstream LICENSE. Binary is NOT redistributed here (`.gitignore`) |
| libr4d paged attention (`r4d.so`, loaded at runtime) | libr4d -- https://codeberg.org/StillDeadcode/libr4d | `r9700_vllm/attn/triton3d.py` (`R9K_PAGED_ATTN=r4d`; ours is the default) | permission by DM 2026-09-21; no upstream LICENSE. Binary is NOT redistributed here |
| One-shot 2-rank P2P all-reduce, exact path -- independent implementation, no libr4d consulted | this repo (`notes/independence.md`) | `kernels/r9k_ar.hip`, `r9700_vllm/comm/r9k_ar.py` (`R9K_AR_IMPL=r9k`) | this repo |
| Device-side expert LRU cache kernels | davetha (vendored) | `kernels/third_party/davetha/` | Apache-2.0 |
| Chat template `qwen-fixed-v22.3.jinja` (copied verbatim) | GGZ14/vllm-mxfp4 -- https://github.com/GGZ14/vllm-mxfp4 (92eed82) | `serve/templates/` | permission by DM 2026-09-21; no upstream LICENSE |
| Ideas: NVFP4->MXFP4 requant, GDN in_proj merge, fp8->MXFP4 for dense layers, decode-band tiling, NT loads, DFlash drafter handling | GGZ14/vllm-mxfp4 (radiance) | `r9700_vllm/quant/`, `r9700_vllm/models/gdn.py`, kernels | ideas only, no code copied |
| Qwen3.8-Flash-Next GPTQ checkpoint, int6 PLE format, DFlash2 drafter, `tcclaviger/vllm` reference image | tcclaviger | serving, format compatibility | per model cards |
| Kernel review and tuning (loop rewrite, fp32 MXFP4 scales, NT loads) | Fable (Anthropic) review, 2026-09-19 | `kernels/r9k_moe_mxfp4a8.hip`, `tuning/fable_bench.py` | this repo |
| vLLM, ROCm, PyTorch, Triton | upstream projects | runtime (stock, unmodified) | Apache-2.0 / MIT / BSD |
