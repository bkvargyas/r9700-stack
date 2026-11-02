# DeepSeek-V4.1-Flash EXL3: layer map and quant-config touch points (package N, 2026-11-02)

What the pinned vLLM's ROCm DeepSeek-V4.1 model (`vllm/models/deepseek_v41/amd/*`, nightly e4a5b553) builds, which of its
modules ask the quant config, where the code assumes `DeepseekV4FP8Config`, and how the EXL3 pack's tensors reach each
parameter. Brief: notes/exl3-plan.md. Loader: r9700_vllm/models/deepseek_v41_exl3.py. Line numbers are in the pin's source
copy (`$TMP/ds41/vllm-src/`, paths relative to `vllm/`).

Method of record: the target (`DeepseekV41ForCausalLM` -> amd/vl_model.py wrapping amd/model.py `DeepseekV41LLMForCausalLM`)
and the DSpark drafter (`DSparkV41DraftModel` -> amd/dspark.py) were **built on the meta device inside the serving image at
TP1** (`--language-model-only`, dspark speculative config) through our subclasses, with a recording quant config that logs
every `get_quant_method(layer, prefix)` call. Parameter names do not depend on TP; shapes below are TP8. Scratch:
VM100 `~/ds41/exl3/n-work/` (meta_params.py, meta.json, expected.json, dryrun-detail.json).

## Table A -- weight-bearing modules (per decoder layer N = 0..39 unless noted)

Prefix root: `language_model.model.layers.N.` (target), `model.layers.i.` (drafter, i = 0..2; its quant-config prefixes say
`model.layers.{40+i}` -- **K must come from the tensors (`k_of(trellis)`), never from a prefix-derived layer index**).

| module (vLLM) | class | in -> out (full) | TP8 (+EP) | asks quant config? -> under exl3 | pack tensors |
|---|---|---|---|---|---|
| `attn.fused_wqa_wkv` | MergedColumnParallelLinear, disable_tp | 5120 -> [1280, 512] | replicated | yes (`...attn.fused_wqa_wkv`) -> Exl3LinearMethod, shards 0/1 | `attn.wq_a.*` (shard 0), `attn.wkv.*` (shard 1); K=5 (layer 39 K=6) |
| `attn.wq_b` | ColumnParallelLinear | 1280 -> 32768 | 4096 a rank (8 heads x 512) | yes -> Exl3 | `attn.wq_b.*` K=5 |
| `attn.wo_a` | ColumnParallelLinear (`is_bmm` reset False on ROCm, rocm.py:569) | 4096 -> 8192 (8 o-groups x 1024) | 1024 a rank = o-group r | **routed to UnquantizedLinearMethod** by our construction scope (`bf16_linears`) -> bf16 [1024, 4096] | `attn.wo_a.slice.{0..7}.*` [4096 -> 1024] K=5: dequantized, assembled in slice order to [8192, 4096], stock column loader slices by rank |
| `attn.wo_b` | RowParallelLinear | 8192 -> 5120 | 1024 inputs a rank | yes -> Exl3 (row split, partial + all-reduce) | `attn.wo_b.*` K=5 |
| `attn.q_norm`, `attn.kv_norm` | RMSNorm | 1280 / 512 | replicated | no | `.weight` bf16 |
| `attn.attn_sink` | Parameter f32 [64] (padded_heads) | -- | stock loader | no | `attn.attn_sink` f32 |
| `attn.indexer.wq_b` (N in 2,8,14,20,24,28,32,36) | ReplicatedLinear | 1280 -> 4096 | replicated | yes -> Exl3 | `attn.indexer.wq_b.*` K=5 |
| `attn.indexer.weights_proj` | ReplicatedLinear, quant_config=None (attention.py:1396) | 5120 -> 32 | replicated | no | `.weight` **f16** -> bf16 param (stock copy_) |
| `attn.indexer.wk` (N in 2,8,14,20) | ReplicatedLinear, quant_config=None (attention.py:1422) | 512 -> 128 | replicated | no -> bf16 | `attn.indexer.wk.*` K=8: **dequantized at load** |
| `attn.indexer.k_norm` | RMSNorm | 128 | replicated | no | bf16 |
| `attn.compressor.fused_wkv_wgate` (N in 2,8,14,20) | MergedColumnParallelLinear, quant_config=None (compressor.py:300), disable_tp | 5120 -> [512, 512] ([512] on 20: ratio 1 has no gate) | replicated | no -> bf16; **read directly** as `.weight.T` (rocm.py:768, attention.py:1006) | `attn.compressor.wkv.*` (shard 0), `.wgate.*` (shard 1, layers 2/8/14) K=5: **dequantized at load** |
| `attn.compressor.norm` | RMSNorm | 512 | replicated | no | bf16 |
| `ffn.gate` | GateLinear(ReplicatedLinear), no quant_config | 5120 -> 384 (draft 128) | replicated | no | `.weight` **f16** -> bf16 param |
| `ffn.gate.e_score_correction_bias`, `ffn.gate.bias_vl` | Parameter f32 [384] | -- | replicated | no | `ffn.gate.bias` **f16** -> f32 (mapper suffix rule), `bias_vl` f32 |
| `ffn.shared_experts.gate_up_proj` | MergedColumnParallelLinear (DeepseekV4MLP) | 5120 -> [2304, 2304] | 288 a rank a shard | yes -> Exl3LinearMethod in **dense mode** (288 is not a 128-multiple: L dequantizes to bf16 at load, stock TP slicing) | `shared_experts.w1.*` (0), `.w3.*` (1) K=5/4 |
| `ffn.shared_experts.down_proj` | RowParallelLinear, reduce_results=False | 2304 -> 5120 | 288 inputs a rank | yes -> Exl3 dense | `shared_experts.w2.*` (mapper: `.shared_experts.w2` -> `.down_proj`) |
| `ffn.experts.routed_experts` (`w13_*`, `w2_*`) | RoutedExperts via FusedMoEFactory (prefix `...ffn.experts`) | 384 x (5120 -> 2x2304, 2304 -> 5120); draft 128 | EP: 48 whole experts a rank (draft 16) | yes (`...ffn.experts`, RoutedExperts) -> Exl3MoEMethod | `ffn.experts.{e}.w1|w3|w2.*` -> `w13_<t>` (shard w1/w3) / `w2_<t>`, expert_id e; K=3 (2 on 18-22), draft K=4 |
| `attn_norm`, `ffn_norm` | RMSNorm | 5120 | replicated | no | bf16 |
| `hc_attn_fn`, `hc_ffn_fn` [24, 20480]; `hc_*_base` [24]; `hc_*_scale` [3] | Parameter f32 | -- | replicated | no | f32, stock |
| `engram.wkv` (N = 1, 14) | ColumnParallelLinear, gather_output (engram.py:1562) | 6144 -> 25600 | 3200 a rank (25 x 128) | yes -> Exl3 | `engram.wkv.*` K=5 / 4 |
| `engram.q_weight`, `engram.k_weight` | Parameter **bf16** [4, 5120] | replicated | no | pack f32 -> bf16 (stock copy_); base's bf16 duplicates dropped |
| `engram.embed_tokens.weight` / `.weight_scale_inv` | ParallelEngramEmbedding (nn.Module, never asks) fp8 [rows, 256] + u8 [rows, 8] | 384.0M rows each | heads TP-split: ~48M rows a rank, pinned host under cpu_offload (11.8 GiB a rank a table, probe) | no | base shards 47/48 `engram.embed.weight|scale` (mapper: `embed.weight` suffix + `(engram\.embed)\.scale` regex); e8m0 viewed as u8 by its loader |
| `model.embed_tokens` | VocabParallelEmbedding (quant_config passed) | 129280 x 5120 | 16160 rows a rank | yes -> UnquantizedEmbeddingMethod | `embed.weight` bf16 |
| `model.norm` | RMSNorm | 5120 | -- | no | bf16 |
| `lm_head` | ParallelLMHead, **no quant_config** (amd/model.py:1349) | 5120 -> 129280 | 16160 rows a rank | no -> bf16 | `head.*` K=6 [320, 8080, 96]: **dequantized at load** (column chunks of 16384), full [129280, 5120] bf16 to the stock loader (host-assembled, pinned) |
| `image_start/end/newline` | Parameter f32 [5120] on the VL wrapper | -- | -- | no | bf16, stock |
| vision tower / aligner | DeepseekV4ViT / Aligner | -- | StageMissingLayer under `--language-model-only` | no | `vision.*`, `aligner.*`: stock skips them (stubbed) |
| drafter `model.main_proj` | ReplicatedLinear (dspark.py:92) | 15360 -> 5120 | replicated | yes -> Exl3 | `mtp.0.main_proj.*` K=6 |
| drafter `model.main_norm`, `model.norm`, `markov_head.markov_w1/w2`, `confidence_head.proj` | RMSNorm / Parameters | -- | -- | no | bf16 / **f16** (markov head.weight, confidence) -> bf16 |
| drafter layers 0..2 | as a backbone layer, compress ratio 0 (no compressor / indexer / engram) | | | fused_wqa_wkv, wq_b, wo_b, shared experts, experts -> Exl3; wo_a -> bf16 | `mtp.{i}.*`; embed / lm_head alias the target's |

Ask-the-config census (meta build): target 251 calls = embed 1 + per layer fused_wqa_wkv / wq_b / wo_b / shared gate_up /
shared down / experts (6 x 40) + indexer.wq_b 8 + engram.wkv 2 (wo_a's 40 are intercepted by bf16_linears); drafter 19 =
main_proj + 6 x 3.

## Table B -- DeepseekV4FP8Config-specific touch points

| where | what it reads | exl3 status / what we did |
|---|---|---|
| config/model.py `_verify_quantization`; quant_config.py:162 `override_quantization_method` | `quant_method in ("fp8", "deepseek_v4_fp8")` claims V4 model types | `exl3` is never claimed; `register_quantization_config` appends to RocmPlatform.supported_quantization (quantization/__init__.py:97) -- no platform change needed |
| amd/model.py:1043 `pad_shared_expert`; dspark.py:314 | `getattr(quant_config, "weight_block_size", None) is not None` | Exl3Config.weight_block_size = None (L): no padding |
| amd/model.py:1195 `_linear_scale_param_name`; vl_model.py:208; dspark.py:311 | `weight_block_size == [32, 32]` | None -> "weight_scale_inv"; only the base's `engram.embed.scale` has a `.scale`, and the engram regex catches it first |
| amd/model.py:1340 / vl_model.py:207 `expert_dtype` | `hf_config.expert_dtype` (default "fp4") | pack config has none -> fp4 mapper; harmless (no `.scale` tensors in the pack) |
| amd/model.py:1084 E8M0 expert scales `.view(uint8)` | `"weight_scale" in name` | never matches EXL3 names |
| amd/model.py:1104 expert loader call | `weight_loader(param, w, name_mapped, shard_id=, expert_id=, return_success=True)`; breaks on success only | **M: accept `return_success` and return True/False** (False for non-local experts, without touching the data -- tensors are mmap-lazy, see below) |
| attention.py:401 `_o_proj_block_size` | `getattr(wo_a, "weight_block_size")` | irrelevant (bf16 path) |
| rocm.py:569 + 945 `rocm_inv_rope_einsum` -> rocm_aiter_mla_sparse.py:1707 `_get_cached_wo_a_bf16` | `wo_a.weight` viewed [groups, 1024, 4096] (+ `weight_scale_inv` when 1-byte) | **wo_a must have a bf16 `weight`**: built unquantized (bf16_linears), loaded by dequant. (EXL3 wo_a later needs an `_o_proj` override; the loader already routes slices to an EXL3 wo_a with shard_id g if one is built.) |
| rocm.py:768, attention.py:1006 compressor `torch.mm(h, fused_wkv_wgate.weight.T)` | plain weight | stock builds it unquantized; we dequantize |
| rocm.py:606 `prepare_attn_preshuffle`, deepseek_v4/amd/model.py:146 `prepare_gateup_preshuffle` | `.weight` + `weight_scale_inv` | V4.1 AMD never calls them (only deepseek_v4/amd/model.py:1599); and EXL3 layers have no `weight` |
| rocm.py:845 `_wq_b_uses_aiter_block_scaled` / 873 `_split_qkv_and_norm` | `linear.quant_method.fp8_linear` isinstance | False for exl3 -> shared bf16 norm path |
| rocm.py:663 / 958 `_wqa_wkv_scale`, `_wo_b_scale` | set only by prepare_attn_preshuffle | None -> `self.fused_wqa_wkv(x)` / `self.wo_b(z)` through the method |
| attention.py:790 `_can_fuse_query_quant` | MXFP8 FlashInfer kernels | CUDA-only -> False |
| rocm.py:913 `_alloc_attn_out` MXFP8 / `rocm_mxfp8_wo_a_bmm` | gfx950 only | off on gfx1201 |
| deepseek_v4/amd/model.py:225 heterogeneous shared expert; :527 `_fuse_shared_experts_enabled` | `get_name() == "deepseek_v4_fp8"`, gfx950, no EP | off (EP on, gfx1201): **shared experts stay a separate DeepseekV4MLP, not fused into the routed experts** |
| deepseek_v4/amd/model.py:690 MegaMoE | `moe_backend == aiter_mega_moe` | off |
| amd/mono_decode.py:349 `MonoDecodeLayer.create` | `--monokernel`, CDNA, TP 2/4, fp8/fp4 weights | **off** (never pass `--monokernel`; CDNA-only anyway) |
| engram.py:1447 `EngramBatch.create` | dp_size > 1, MXFP8 wkv input | off at TP8 / DP1 |
| default_loader.py:512 weights tracking | `quantization is None` | off for quantized models; if enabled, empty placeholders (`numel()==0`) count as loaded (L's `trellis/suh/svh/mul1` placeholders) |

## Decisions

- **Shared experts**: not fused on this path (EP; heterogeneous fusion is gfx950/no-EP/fp8-only). They reach L's
  Exl3LinearMethod, which goes dense (dequant to bf16, stock TP slicing) because 288 is not a 128-multiple. Nothing special
  in the loader.
- **wo_a**: bf16 for v1 (stock ROCm runs it in bf16 anyway; 320 MB a rank). Built unquantized by `bf16_linears` (the exl3
  config's `get_quant_method` is shadowed on the instance during `__init__` for `\.attn\.wo_a$`), the eight dequantized slices
  are concatenated in slice order and handed to the stock column loader.
- **lm_head / compressor / indexer.wk**: stock gives them no quant config; dequantized at load (generic fallback below).
  Exl3LMHeadMethod is not engaged on this model (would need a quant_config on the head constructor).
- **embed_tokens**: bf16 through UnquantizedEmbeddingMethod.
- **mono decode**: off. **Vision**: `--language-model-only`; stock stubs the tower and skips its weights.
- **attn_sink, hc_*, gate.bias_vl, norms**: stock paths. dtype conversions are stock `copy_`: pack f16 `gate.weight`,
  `weights_proj`, `markov_head.head`, `confidence_head` -> bf16 (the base stores bf16; f16 holds those values exactly, so the
  round trip is lossless); f16 `gate.bias` -> f32 (the base has f32: the pack's own rounding, as exllamav3 runs it); f32
  `engram.k/q_weight` -> bf16 param.
- **Generic dequant fallback** (owner's design): for every non-expert EXL3 matrix the loader resolves the vLLM module
  (stock's stacked/mapper renames: wq_a/wkv -> fused_wqa_wkv 0/1, compressor wkv/wgate -> fused_wkv_wgate 0/1,
  shared w1/w3 -> gate_up_proj 0/1, w2 -> down_proj, wo_a.slice.g -> wo_a, head -> lm_head). Quant method Exl3LinearMethod /
  Exl3LMHeadMethod -> the tensors pass through to the stock loader untouched; anything else -> collected, `dequant_f32` in
  16384-column chunks (L's routine; f32 [in, out] -> bf16 [out, in]), then `param.weight_loader(param, W[, shard_id])`.
  `.mul1`/`.mcg` are consumed as flags, never fed to a stock loader.
- **Stream safety**: `serialize_attention()` sets `aux_stream_list = None` on every V4.1 attention (target and drafter) after
  construction. rocm.py `forward` then takes its sequential fallback (`enable_multi_stream_overlap(None, ...)` is False,
  rocm.py:1254 platforms), and `_run_parallel_input_projections` / `_prepare_and_attn` hand `execute_in_parallel` no streams
  (sequential, utils/multi_stream_utils.py:66). The shared experts may still run on vLLM's shared-experts aux stream next to
  the routed MoE: they are bf16 (L dense), not EXL3; if M's kernels share the GEMV parameter block with anything on another
  stream, `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` is the switch.
- **Base shards 47/48**: they hold the whole Engram module (embed weight/scale, bf16 k/q_weight, fp8 wkv weight/scale).
  The loader drops `engram.wkv.weight|scale` (fp8) and bf16 `engram.k|q_weight`; a generic duplicate guard skips any repeated
  plain tensor name (pack shards sort first: natural sort, 2..43 < 47).
- **Loading cost**: safetensors 0.8.0 `get_tensor` is mmap-lazy (measured on VM100: a 150 MB tensor +3 MB RSS), so the
  98 GB Engram tables are only touched in each rank's 1/8 head range, and non-local experts are never read if M returns
  False without touching them. vLLM's EP weight filter (`--enable-ep-weight-filter`, off by default) only skips names ending
  `.weight`, so it does nothing for EXL3 names -- not needed.

## Dry run (gate 6c)

`python3 -m r9700_vllm.models.deepseek_v41_exl3 pack-tensors.json base-tensors.json expected.json` in the image: every pack
tensor (192,495, 43 shards) + the 12 base shard 47/48 tensors through `classify()` (the runtime function), the stock VL
mapper, stacked and expert mappings, against the meta-built parameter set translated to exl3 names (EXL3 modules ->
trellis/suh/svh/mul1, experts -> w13_/w2_ x 4).

| side | tensors | params reached | unmapped | reach a non-param | params never loaded |
|---|---|---|---|---|---|
| target | 192,507 | 1,798 / 1,798 | 0 | 0 | 0 |
| drafter | 192,507 | 138 / 138 (+ embed / lm_head aliased) | 0 | 0 | 0 |

Target: 184,320 expert tensors (40 x 384 x 3 x 4), 640 stacked, 1,109 direct, 1,328 dequantized (head 4, wo_a slices
1,280, compressor 28, indexer.wk 16), 5,110 skipped (4,839 `mtp.*`, 263 vision/aligner, 8 base engram duplicates).
Drafter: 4,608 expert tensors, 48 stacked, 87 direct, 96 dequantized (wo_a slices), the rest skipped as target weights.
Synthetic on card 4 with L's real `dequant_f32`: chunked dequant equals the one-shot result up to f32 summation order
(max |diff| 2.4e-4 on bf16 values ~0.1, 0.007% of elements one bf16 ulp apart); stock-loader hand-off shapes / shard ids,
wo_a assembly, EXL3 pass-through, skips and the duplicate guard as intended.

## Composed model dir (VM100 `~/ds41/exl3/compose-model-dir.sh`, idempotent)

`~/models/DeepSeek-V4.1-Flash-1017-EXL3/` = relative symlinks (`../../ds41/...`; resolve on the host, under gpu-run.sh and
under serve/ds41-exl3.sh, which mounts ~/ds41 at /ds41): the pack's config.json, chat_template.jinja, exl3_k_map.json,
README.md and whichever of its 43 shards have landed; base shards 47/48 and base tokenizer.json / tokenizer_config.json.
No index: vLLM globs `*.safetensors` in a local dir and filters by `model.safetensors.index.json` only when one exists
(default_loader.py:229-255) -- the pack ships an index (and quantization_config.json, its own tokenizer files) that would
drop the base shards, so the script never links one and removes any.

## Open

- First eight-card load: KVMEM (2 GiB a rank is an estimate), eager vs graphs, attention numerics on gfx1201 (the probe
  only reached weight loading).
- Head dequant transient: 1.3 GB pinned host a rank while the stock loader slices it; fine at 377 GB.
- DSpark: `--enable-ep-weight-filter` would be wrong for the drafter (its 128 experts against the 384-expert id split) --
  keep it off.
