# r9700-stack v0.3.6

**Flash-Next's current Hugging Face revision (6fe3e746, 2026-10-01) loads and drafts again.** That revision added fp8
KV scales to the MTP attention layer (the loader crashed) and moved the MTP MLP to bf16 (the MXFP4 conversion
silently dropped the shared expert, so every draft token was rejected: ~65 tok/s instead of ~200 on four cards).
Contributed by Altay Sansal (@tasansal) in [#1](https://github.com/bkvargyas/r9700-stack/pull/1).

## What changed for someone running it

- `r9700_vllm/models/qwen4_exp.py`: the MTP attention's KV scales are dropped like the main model's (the KV cache is
  bf16); a bf16 MTP MLP (fused experts and shared expert) is quantized to MXFP4 at load, with the MSE-searched block
  exponents from `quant/nvfp4.py` (2.7-4.5% lower error than round-to-nearest on bf16 MLP weights, same format).
  `r9700_vllm/quant/ct.py`: the checkpoint's `re:^mtp\.` ignore rule is narrowed so vLLM builds MXFP4 layers for
  that MLP. Older fp8 MTP checkpoints take the same path as before.
- `tests/test_integration.py` covers the new loader paths and skips its PLE scope section where the nightly has no
  `PLEVocabParallelEmbedding` (renamed in the October nightlies; unrelated to the loader).

## Validation

The contributor, on 4x R9700 TP4 with revision 6fe3e746: 19.44 GiB of weights a GPU, draft acceptance 62.2% against
63.9% with a bf16 MTP (`R9K_MTP_MLP=bf16`), the same 184 tok/s single stream, 527 / 607 tok/s at 8 / 16 streams,
GSM8K on all 1,319 questions 96.06% against 95.83% (paired McNemar p = 0.58).
This tree on the 2026-09-18 revision (the one validated for v0.3.3-v0.3.5), Flash-Next TP4: 3.09 tokens a step
(3.11 before), BetterBench combined 187.0 (v0.3.5 tree: 185.3; v0.3.3: 188.0), 1 / 2 / 4 / 8 streams 173 / 274 / 400 /
541 (177 / 277 / 409 / 531), prompt processing 5.1k-7.4k tok/s unchanged, strict sanity 0 bad of 2,460, no server
errors. GLM-5.3-Flash and the 27B are untouched by this change.
