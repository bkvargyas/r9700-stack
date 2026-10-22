# r9700-stack v0.3.5

**GLM-5.3-Flash long-context retrieval fixed: the decode-side indexer scores were scrambled on this GPU family by a
cache-layout fall-through in stock vLLM; a tiled-layout kernel in the plugin replaces it.** Needle lookups now
succeed at 12.7k, 23.7k and 34.7k tokens (unique and near-duplicate needles) where v0.3.4 failed from 12.7k on, and
decode at depth rises to 92-110 tok/s (from 82-97) because the kernel is also faster and the drafts land more often.

## What changed for someone running it

- `r9700_vllm/attn/indexer_logits.py`: `paged_mqa_logits`, a Triton kernel that reads the indexer's paged K cache
  in the 16x16 tiled layout vLLM's kpool compress kernel writes (pages of 64 pooled entries). Stock reads that layout
  back only on gfx942/gfx950 (aiter's pre-shuffle Gluon kernel); elsewhere its wrapper falls through to a kernel
  that reads a linear page. `_install_paged_logits` (models/glm5next.py) swaps ours in exactly where stock would
  take that fall-through; `R9K_INDEXER_LOGITS=stock` restores stock. Installed with the other GLM hooks only.
- `tests/test_indexer_logits_r9k.py`: the cache written by the real kpool writer, stock's reader and ours against
  the torch reference at 760 to 65k pools, one row and MTP verify rows. Add it to the single-GPU gate list.
- Recommended GLM launch now uses 8k-token prefill steps (`NBT=8192`, +30% prefill) and, above 32k context, a fixed
  KV budget (`KVMEM`) so the stock indexer's prefill buffer keeps headroom; see `notes/glm53.md`.

## Validation

Offline: ours vs the torch reference max |diff| 0.03-0.17 on logits of 100-180, top-512 overlap 99.6-100%, planted
needle the argmax in every row at 760 / 3.2k / 8k / 32k / 65k pools, 92-100 us a call (stock: 153-158 us, overlap
70% -> 2.5%, needle lost). Server (65k launch, MTP=3, fp8 KV): needles correct at 3k / 12.7k / 23.7k / 34.7k, answers
in 50-70 tokens; prefill 2.5-3.1k tok/s; acceptance 80%. Qwen configurations untouched (the hook lives in the GLM
module); the v0.3.4 gates stand for them.

## Known

- Eight cards at the 210 W cap draw ~2,375 W at the wall in prefill; on a 120 V feed with the G292-Z22's 2+0
  supplies that is the limit and the platform parks the CPU at ~400 MHz until a power cycle. The GLM launch helper
  applies a -500 MHz clock offset (peak ~2,175 W); a 208/240 V feed is the real fix.
