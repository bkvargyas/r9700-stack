# r9700-stack v0.3.7

**The pin moves to the vLLM 0.32 line** (`nightly-rocm100-e4a5b553`, 2026-10-25: vLLM 0.32.1rc1.dev466,
torch 2.13.0+rocm10.0.0, the same ROCm 10.0 SDK). Five plugin-side fixes make that nightly serve all three
configurations at the v0.3.6 numbers; the method and the bisections are in [vllm-0.32.md](vllm-0.32.md). Brian:
"Cut v0.3.7 and move the pin to .32".

## What changed for someone running it

- `docker/Dockerfile` builds on the 2026-10-25 nightly. `v0.32.0` itself branched from main on 2026-10-12 and the
  ROCm 10 nightly closest to it is `5555a2d8f` (10-13); the pin is the later nightly every fix was validated on.
- **GLM-5.3-Flash**: `R9700Platform` answers "none" to the MLA prefill backend question on gfx1201 (stock now picks the
  CK flash-attention backend, whose wheel has no gfx1201 kernels); sparse MLA keeps its top-k prefill path.
- **Flash-Next**: the int6 PLE loader computes the checkpoint-shard overlap itself (stock's helper moved); the
  hyper-connection mix reads `hc_lowrank` (was `lora_rank`); `serve/flashnext.sh` TP4 utilization 0.98 -> 0.97
  (torch 2.13 holds ~0.7 GiB a process at start).
- **27B**: our Triton-attention impl pins `supports_quant_query_input = False` (stock advertises fp8 query input on
  gfx12; our paged prefill kernel takes bf16) and falls back to stock for non-16-bit queries; our metadata builder
  declines in-place draft metadata updates (its kernel block 16 under 1808-token KV blocks left the remapped block
  table stale under graph replay: every draft rejected). `R9K_DRAFT_META_UPDATE=1` opts back in.
- **Retired**: the PLE short-conv prefill hook (`ple/short_conv.py`, `R9K_PLE_CONV=r9k` re-enables): stock's fused
  Triton kernel is bit-identical and 8x faster on gfx1201.
- `serve/serve.sh`: `CCMODE=` (torch.compile mode) next to `CGMODE=`. `overlay/emulated-switch/patch-hostcall.sh`
  reports compressed offload bundles (the nightly's `_rocm_C` ships them; its remaining hostcall kernels are gfx9-only).
- Compat gate: `mtp_allowlist` and `glm5next_rdna4` tested on `e4a5b5537`; the integration gate checks the MLA
  prefill fallback, the two renamed Flash-Next attributes, the fp8-query pin and the draft-metadata claim.

## Validation (VM100, 8x R9700 at 210 W in a 384 GiB guest, `~/validate-037.sh`)

Same tree and image as the tag, overlay on, nothing else on the box. Numbers in parentheses are the previous records
(v0.3.6 regression day for the Qwen configurations, v0.3.5 for GLM).

**Qwen3.8-27B-NVFP4 TP2 (cards 0,1, DFlash2 drafter):** probe 217.3 tok/s mean (code 231 / prose 110 / json 273 /
math 255), 4.56 tokens a step, 530 / 627 tok/s at 8 / 16 streams, prefill 3,826 tok/s on 8k (216 / 4.48 / 527-548 /
631-658 / 3,906). BetterBench combined **194.7** (189.3), 1 / 2 / 4 / 8 streams 177.6 / 302.8 / 441.6 / 593.4
(165 / 274 / 404 / 522), prompt processing 3.6-3.9k tok/s at 2k-32k (3.8-3.9k). Strict sanity: 0 bad of 160 after
long prompts, 0 of 540 at 9 streams, 0 of 640 at 16. Soak (16 clients, 200-30k-token prompts, 943 s): 218 requests, 0 errors, median prompt 13.1k tokens, 3,031 prompt tok/s. 0 server errors, 0 untested-patch warnings.

**Qwen3.8-Flash-Next TP4 (cards 0-3, MTP-3, `UTIL` 0.97):** KV cache 477,002 tokens (494k at 0.98 on torch 2.12). Probe
221.8 tok/s mean (code 222 / prose 160 / json 262 / math 244), 3.1 tokens a step, 468 / 917 tok/s at 8 / 16 streams,
prefill 6,711 tok/s on 8k (209.7 / 3.09 / 465 / 933 / 6,771). BetterBench 1 / 2 / 4 / 8 streams 173.0 / 275.9 / 400.7 /
542.6 (173 / 274 / 400 / 541), prompt processing 5.0 / 7.0 / 7.4 / 7.1k tok/s at 2k / 8k / 16k / 32k (5.1-7.4k); the
weighted combined 182.6 against 187.0, a single-shot composite moving while every stream level matches. Strict sanity:
0 bad of 160 after long prompts, 0 of 1,020 at 17 streams, 0 of 1,280 at 32. Soak (16 clients, 904 s): 364 requests,
0 errors (20 prompts over the 32k limit rejected, by design), median prompt 17.1k tokens, 6,689 prompt tok/s.
0 server errors, 0 untested-patch warnings.

**GLM-5.3-Flash TP8 (32k production recipe: RedHat checkpoint, MTP-3, fp8 KV, NSEQ 4, NBT 8192):** KV cache 190,726
tokens. Decode-only 75.0 tok/s median over three 512-token runs (79.6 record; the usual 70-80 band, 76.5 on this nightly
the day before), four streams 131 then 157 tok/s aggregate (158), draft acceptance 44-59 % on the probes. 12.7k needle
correct: ttft 5.2 s, prefill 2,466 tok/s, 99 tok/s at depth, acceptance 80 %. Strict sanity (thinking on): 0 bad of 160
after long prompts, 0 of 300 at 5 streams, 0 of 320 at 8. Soak (16 clients, 964 s): 155 requests, 0 errors (7 over
the 32k limit rejected), median prompt 13.5k tokens. 0 server errors.

**GLM-5.3-Flash TP8, 262k launch -- a regression on this nightly.** The v0.3.5 recipe (`MAXLEN=262144 KVMEM=2.7
NBT=8192`) comes up with the same 414,860-token KV cache as on the pin, then the first long prompt runs out of VRAM in
stock's prefill-side indexer (`sparse_attn_indexer_kpool -> rocm_fp8_mqa_logits`, logits of chunk tokens x context
pools): a 30k prompt at 8192-token chunks; with `NBT=4096` the 30k and 67k (75k-token) needles are correct (2.4-2.8k
tok/s prefill, 73-117 tok/s at depth, 80-84 % acceptance) and 150k fails; `KVMEM=2.0 NBT=4096` and `KVMEM=2.0
NBT=2048` fail at 150k too. PyTorch holds 26-27.5 GiB at the failure with 0 bytes free: about 5 GiB a card sits
outside PyTorch's allocator on this nightly, ~1.5 GiB more than the pin left room for. **On the 0.32 line the validated
GLM context is the 32k production recipe; a 262k launch retrieves correctly to ~75k tokens (216k on v0.3.5's pin).**
First open item after this release: measure the idle footprint on both images and recover the headroom.

## Known limits

- **GLM-5.3-Flash above ~75k tokens of prompt on this nightly** (see Validation): the 262k launch loses ~1.5 GiB of VRAM
  headroom a card to allocations outside PyTorch and stock's prefill indexer runs out of memory. The 32k production
  recipe is unaffected. Under investigation; v0.3.6 on the previous pin keeps the 216k retrieval if you need it today.
- Four-card GLM-5.3-Flash loads only with 24 GB of experts a rank and a 2 GiB device cache (8 tok/s): each offload
  rank pins about twice its offloaded gigabytes (stock's UVA copy and the plugin's re-layout). Freeing stock's copy
  after the re-layout is the open item ([glm53.md](glm53.md)).
- The draft metadata rebuild costs a per-step CPU build on the 27B; declaring `MultipleOf(16)` like stock and
  remapping inside our prefill path would let the in-graph update stay on.
