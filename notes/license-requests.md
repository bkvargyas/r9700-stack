# License requests (drafts for Brian to send)

Neither upstream ships a LICENSE file, so by default both are all rights reserved. Credit alone does not grant
permission to copy, modify or distribute, which is why these asks exist. Both are best sent as a public issue on
the project's own tracker (that is the normal channel and leaves a citable record); no email address needed.

Status 2026-09-21: **Both authors confirmed by DM that our use is fine** (per Brian), and the repo is now
Apache-2.0 with those two items carved out in `NOTICE`. Brian's read is that neither author is likely to add a
licence file, and he is content to ship the chat template on credit alone.

So these drafts are no longer blocking anything. They are kept for one reason: **the libr4d carve-out is the one
that limits what Apache-2.0 actually means here**, because the derived GEMM is the core of the project. Anyone who
takes this repo under Apache-2.0 finds the central kernel is not theirs to reuse. If that ever matters -- someone
wants to build on it, or it goes somewhere that needs clean provenance -- there are two ways out, and the ask
below is the cheap one. The rewrite priced at the bottom (~5%) is the other.

If the ask is made, the useful wording is specific: not "would you add a licence?" but **"would you license libr4d
under Apache-2.0 or MIT?"** -- only that unblocks sublicensing.

Context that makes both asks small: our own kernels now replace libr4d's paged attention and 2-rank all-reduce
(`notes/independence.md`), so the only thing still at stake for libr4d is the **derived GEMM**, and for
vllm-mxfp4 the **chat template**.

---

## 1. StillDeadcode / libr4d — https://codeberg.org/StillDeadcode/libr4d

**Title:** Would you consider adding a LICENSE file?

> Hi — thanks for libr4d; the gfx1201 work in it is genuinely useful and there is very little else out there for
> RDNA4.
>
> I have been building an open kernel library and vLLM plugin for the Radeon AI PRO R9700 (Qwen3.8-27B and
> Flash-Next on stock vLLM + ROCm 10). Two things from libr4d ended up involved:
>
> 1. **A derived GEMM.** My MXFP4 x FP8 MoE kernel started from `r4d_gemm_mxfp4a8_nt_m64` (commit 5dc6302) and
>    still carries its fragment-order weight layout, the folded e2m1 -> e4m3 unpack with the `kMag` permute table,
>    and the in-block split-K structure. It has been rewritten heavily since — most of the file is new work — but
>    the lineage is real and the header says so.
> 2. **`r4d.so` at runtime**, for paged attention and the 2-rank P2P all-reduce. I have since written my own
>    replacements for both, so this one is no longer a dependency; I mention it for completeness.
>
> The repository has no LICENSE file, which by default means all rights reserved, so I do not have permission to
> publish the derived code even with attribution. Would you be willing to add a license? Something permissive
> such as Apache-2.0 or MIT would let me credit you properly and publish; if you would rather I did not use it at
> all, that is a fair answer too and I will rewrite the remaining pieces.
>
> Either way, thank you for putting the work out there.

---

## 2. GGZ14 / vllm-mxfp4 — https://github.com/GGZ14/vllm-mxfp4

**Title:** Would you consider adding a LICENSE file?

> Hi — vllm-mxfp4 has been a really useful reference while getting Qwen3.8 running well on 2x Radeon AI PRO
> R9700, and it is the bar I have been benchmarking against.
>
> One concrete thing I have copied: **`qwen-fixed-v22.3.jinja`** (from 92eed82), verbatim, into my serving
> repo — it is worth about +14% speculative acceptance over the checkpoint's own template, which is a large
> effect and not something I wanted to reinvent. It is credited in my CREDITS.md.
>
> Separately I followed the integration *approach* of `radiance_r4d_attn.py` and `radiance_allreduce.py`, and
> took several ideas (NVFP4 -> MXFP4 requantisation at load, merging the GDN `in_proj` pair into one GEMM,
> converting fp8 dense layers to MXFP4) — but wrote my own implementations rather than copying code.
>
> The repository has no LICENSE file, so by default I do not have permission to redistribute the template even
> with credit. Would you be willing to add one? Apache-2.0 or MIT would be ideal. If you would prefer I not ship
> the template, say so and I will drop it and generate my own.
>
> Thanks for the project either way.

---

## If neither answers

Both are replaceable, and the cost is known:

- **libr4d's GEMM lineage** — regenerate `kMag` from the OCP e2m1/e4m3 definitions with a checked-in generator,
  replace the 4x`v_perm_b32` unpack with a shift/LUT version (**~5%**, priced in `notes/review-fable.md:270`),
  re-derive the `Wref` fold ABI, and re-derive the fragment-order layout in `r9700_vllm/kernels/moe.py:100-109`
  from the WMMA fragment layout. `kernels/r9k_gemm_fp8.hip` needs the same treatment. Note the non-folded fp32
  per-group path is already ours and is the code default, so a non-folded build is close to independent today —
  but the A-tiled prefill kernel is folded-only, so the +12.7% prefill win currently rides on the fold.
- **The chat template** — generate our own from the checkpoint's template plus the specific fixes, and re-measure
  acceptance. Riskier than it sounds: the +14% came from details we did not derive ourselves.
