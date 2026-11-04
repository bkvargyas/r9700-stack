# DeepSeek-V4.1-Flash at EXL3 2.9 bpw on r9700-stack -- architecture brief (2026-11-02)

Brian: "Lets do a 3-bit / 2.9-bit EXL3 version" of DeepSeek-V4.1-Flash on OUR stack, resident on the eight R9700s. This is the
brief the implementing agents work from; the owner (Claude Fable, this session) reviews every piece against it and does the
eight-card integration. Agents: read this whole file first, then the files named in your package. Do not commit; the owner
commits. Own only the files listed under your package; anything else you need changed, write it up in your report.

## Why this shape

- The official checkpoint (deepseek-ai/DeepSeek-V4.1-Flash-1017: MXFP4 experts, FP8 attention) is 292 GiB of device weights
  and does not fit 8 x 32 GiB (probe OOMed at load, VM100 2026-11-01). Streaming experts over the PLX uplinks is the slow path we
  measured on GLM four-card (8-43 tok/s).
- **LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw** (196.3 GiB, 43 shards, MIT) fits resident with ~55 GiB to spare. Two DGX
  Sparks serve this exact pack at 73 tok/s single / 136 at four users (TensorFold TP2, 2026-10-18), tool eval 97.1.
- EXL3 is ExLlamaV3's trellis format. **exllamav3 v1.6.6 (MIT) builds on ROCm and ships RDNA kernels**; the gfx1201 wheel is built:
  VM100 `~/ds41/exl3/dist/exllamav3-1.6.6-cp312-cp312-linux_x86_64.whl` (built inside r9700/vllm:dev, PYTORCH_ROCM_ARCH=gfx1201).
  exllamav3 also has its own DeepSeek-V4.1 architecture (exllamav3/architecture/deepseek_v41*.py, modules/dsv4.py, mla_attn.py,
  qsa_indexer.py, ngram_embedding.py, block_sparse_mlp.py) -- a second reference for how the pack's tensors are consumed.
- Rules: r9700-stack is a plugin on STOCK vLLM (pin r9700/vllm:dev = nightly-rocm100-e4a5b553, vLLM 0.32.1rc1.dev466, torch
  2.13.0+rocm10.0.0, hip 7.15) serving unmodified published checkpoints. Load-time transforms (dequantize a tensor to bf16 while
  loading, re-layout) are fine; patching vLLM/ROCm source or binaries or rewriting the checkpoint on disk is not. exllamav3 is a
  third-party library we link against and read freely. libr4d (radiance) source is never read.

## The checkpoint (read from every shard header: `pack-tensors.json`, summarized in `pack-summary.txt`; base in `base-*`)

config.json (pack): architectures ["DeepseekV41ForCausalLM"], model_type deepseek_v41, text_config: 40 layers, hidden 5120,
**every layer is MoE** (384 routed experts + 1 shared, 6 a token, moe_intermediate_size 2304, scoring sqrtsoftplus, noaux_tc,
routed_scaling_factor 1.5, swiglu_limit 10.0), 64 heads x head_dim 512, one KV head, qk_rope 64, q_lora_rank 1280, o_lora_rank 1024,
o_groups 8, sliding_window 128, compress_ratios (2 on layers 2-19, 1 on 20-39), kv_source_layer_ids [2,8,14,20], indexer 32 heads x
128, index_topk 512, hc_mult 4 (hyper-connections: the residual stream is 4 x 5120 wide), engram_layer_ids [1,14] (tables of 384.0M
and 384.0M rows x 256, fp8 + e8m0 scales -- in the BASE checkpoint's shards 47/48 only, ~197 GB), num_nextn_predict_layers 3 with
the DSpark drafter (block 5, 128 experts, 3 a token, markov rank 256), yarn rope factor 16 to 1,048,576 positions, vocab 129280.
quantization_config: {"quant_method": "exl3", "version": "1.6.2", "avg_bits": 2.9, "head_bits": 6, "codebook": "mul1",
"out_scales": "always", "mtp_bits": 4}. `exl3_k_map.json`: routed-expert bits per layer (3, except layers 18-22 at 2).
The pack has config.json, chat_template.jinja, README.md, exl3_k_map.json, encoding/ (tokenizer code); tokenizer.json +
tokenizer_config.json come from the base checkpoint (~/ds41/hf/DeepSeek-V4.1-Flash-1017).

**Names are the Hugging Face names** (no `model.` prefix; identical in pack and base) with each quantized linear's `.weight`
replaced by four tensors, exllamav3 style:
- `<name>.trellis` int16 [in/16, out/16, 16*K]  K = bits (tile width 48 -> 3, 32 -> 2, 64 -> 4, 80 -> 5, 96 -> 6, 128 -> 8)
- `<name>.suh` f16 [in]   input-side sign vector (applied with the blockwise 128-point Hadamard rotation of the input)
- `<name>.svh` f16 [out]  output-side sign/scale vector (applied with the 128-point Hadamard on the output side)
- `<name>.mul1` int32 []  codebook marker (present = codebook mul1; an `mcg` tensor would mean the other codebook)
Reconstruction: W = de-Hadamard(trellis-decode) with suh/svh; `ext.reconstruct(unpacked_f16[in,out], trellis, K, mcg, mul1)` is
the exact dequant (exllamav3/modules/quant/exl3.py `_reconstruct_inner` / `reconstruct_had_slice`). Tiles (16 in x 16 out) decode
independently; the Hadamard blocks are 128 inputs / 128 outputs, so **any split of a matrix must fall on 128-multiples of the
split dimension** (and 16 for the tile index).

Per layer N (0-39), pack shapes as [in -> out, K]; shard = model-000NN-of-00043:
- attention (K=5; layer 39's wq_a/wkv at K=6): `attn.wq_a` 5120->1280, `attn.wq_b` 1280->32768 (64 heads x 512), `attn.wkv`
  5120->512, `attn.wo_a.slice.{0..7}` eight slices 4096->1024 (one per o-group: 8 heads x 512 -> o_lora 1024; the base stores
  `attn.wo_a.weight` [8192, 4096] = the eight [1024, 4096] slices stacked), `attn.wo_b` 8192->5120; `attn.q_norm.weight` bf16 [1280],
  `attn.kv_norm.weight` bf16 [512], `attn.attn_sink` f32 [64]; `attn_norm.weight`, `ffn_norm.weight` bf16 [5120]
- indexer (layers in index_source_layer_ids [2,8,14,20,24,28,32,36]): `attn.indexer.wq_b` 1280->4096 (K=5), `attn.indexer.wk`
  512->128 (K=8; layers 2,8,14,20 only), `attn.indexer.k_norm.weight` bf16 [128], `attn.indexer.weights_proj.weight` f16 [32, 5120]
- compressor (kv_source layers 2,8,14,20): `attn.compressor.wkv` 5120->512 (K=5), `attn.compressor.wgate` 5120->512 (K=5; 3 layers),
  `attn.compressor.norm.weight` bf16 [512]
- routed experts: `ffn.experts.{0..383}.w1|w3` 5120->2304, `.w2` 2304->5120, K=3 (K=2 on layers 18-22); w1 = gate, w3 = up, w2 = down
- shared expert: `ffn.shared_experts.w1|w3` 5120->2304, `.w2` 2304->5120, K=5 (21 layers) or 4 (19 layers)
- router: `ffn.gate.weight` f16 [384, 5120], `ffn.gate.bias` f16 [384], `ffn.gate.bias_vl` f32 [384]
- hyper-connections f32: `hc_attn_fn` / `hc_ffn_fn` [24, 20480], `hc_attn_base` / `hc_ffn_base` [24], `hc_attn_scale` / `hc_ffn_scale` [3]
- engram (layers 1, 14): `engram.wkv` 6144->25600 (K=5 / 4), `engram.k_weight`, `engram.q_weight` f32 [4, 5120]; the tables
  `engram.embed.weight` f8_e4m3 [384M, 256] + `engram.embed.scale` f8_e8m0 [384M, 8] exist only in base shards 47 and 48
- top level: `embed.weight` bf16 [129280, 5120], `head` 5120->129280 at K=6 (trellis [320, 8080, 96]), `norm.weight` bf16
- DSpark drafter `mtp.{0,1,2}.*`: attention like a main layer at K=6, `ffn.experts.{0..127}` at K=4, shared expert K=6, `ffn.gate.*`
  [128, ...], `hc_*`; `mtp.0.main_proj` 15360->5120 (K=6), `mtp.0.main_norm.weight`, `mtp.0.markov_head.embed.weight` bf16
  [129280, 256], `mtp.0.markov_head.head.weight` f16 [129280, 256], `mtp.0.confidence_head.proj.weight` f16 [1, 5376], `mtp.0.norm.weight`
- vision tower + aligner (`vision.*`, `aligner.*`, `image_*`) bf16 -- loaded as-is or dropped; text serving does not run them.

## How the pinned vLLM serves this model (source copy: `$TMP/ds41/vllm-src/`, see Paths)

`DeepseekV41ForCausalLM` -> `vllm/models/deepseek_v41/amd/vl_model.py` on ROCm, wrapping `amd/model.py:DeepseekV41LLMForCausalLM`
(DeepseekV4Model, DeepseekV4DecoderLayer, DeepseekV4MoE(DeepseekV4MoEBase), attention from `attention.py` + `amd/rocm.py`
(DeepseekV41ROCMAiterMLAAttention, indexer classes), compressor.py, common/engram.py (ParallelEngramEmbedding, cpu_offload through
`--engram-config '{"cpu_offload": true}'`), the DSpark drafter in `amd/dspark.py` through `--speculative-config
'{"method": "dspark", ...}'`). The quant config today is `quant_config.py:DeepseekV4FP8Config` (fp8 + fp4 experts) -- every place
that assumes it is a touch point for us. Loader: `amd/model.py` load_weights (~line 1013: stacked_params_mapping fuses `wq_a`+`wkv`
-> `fused_wqa_wkv` and `compressor.wkv`+`wgate` -> `compressor.fused_wkv_wgate`; `pad_shared_expert` ~1053; experts through
`fused_moe_make_expert_params_mapping(..., ckpt_down_proj_name="w2", ...)` ~1168 so `experts.{e}.w1.<t>` -> `experts.w13_<t>` with
expert_id e and shard "w1"), `_make_deepseek_v4_weights_mapper` ~1211 (regex renames incl. `engram.embed.scale` ->
`engram.embed_tokens.weight_scale_inv`, `.shared_experts.w2` -> `.down_proj`), `packed_modules_mapping` ~1328, `lm_head =
ParallelLMHead(..., quant_config)` ~1349. The stock probe of the fp8 checkpoint on VM100 (`~/ds41/probe-stock.sh`, container log
excerpt in `~/ds41/probe-stock.log` if present) came up through model init on 8 workers with our plugin's glm5next/sparse-MLA
patches engaged and died loading weights (OOM) -- so the ROCm attention path at least constructs on gfx1201.

## Design decisions (owner)

1. **Quant method `exl3`** registered with `register_quantization_config("exl3")` (a new name; vLLM picks it from the pack's
   config.json `quant_method`). `Exl3Config.get_quant_method(layer, prefix)`: LinearBase -> `Exl3LinearMethod`; the MoE experts
   layer -> `Exl3MoEMethod`; `embed_tokens` (VocabParallelEmbedding, bf16) -> stock UnquantizedEmbeddingMethod; `lm_head`
   (ParallelLMHead) -> **dequantize at load**: `ext.reconstruct` the 6-bit head to f16 [5120, 129280], transpose, take the rank's
   vocab slice, store as a bf16 ParallelLMHead weight (165 MB a rank; 129280/8 = 16160 is not a 128-multiple so the EXL3 head cannot
   be vocab-sharded directly, and replicating it costs 0.5 GB a rank plus a redundant 0.5 GB read a step).
2. **Tensor parallel 8 + expert parallel** (`--enable-expert-parallel`): each rank holds 48 whole routed experts a layer (384/8),
   no expert matrix is ever split. Attention: `wq_b` column-split by heads (4096 outputs a rank = 32 x 128), `wo_a` rank r = slice r
   whole (the o_groups=8 design is TP8), `wo_b` row-split (1024 inputs a rank = 8 x 128; output transform per partial then the
   all-reduce -- linear, so exact), `wq_a` / `wkv` / indexer / compressor replicated (as vLLM already does for MLA). Where vLLM
   fuses two matrices into one layer (`fused_wqa_wkv`, `fused_wkv_wgate`) the EXL3 method keeps **one sub-matrix per shard** and
   runs one gemm per shard into the output slice: tiles concatenate along the output dimension, but `suh` is per matrix.
   **Shared experts**: 2304/8 = 288 is not a 128-multiple -> dequantize at load to bf16 and let stock shard them (8.8 MB a layer a
   rank, 350 MB a rank total; cheaper in memory and bandwidth than replicating EXL3) -- unless the AMD path fuses the shared expert
   into the routed experts (`pad_shared_expert`, `DeepseekV4HeterogeneousSharedRoutedExperts`), in which case agent N decides.
   `engram.wkv`: follow how vLLM shards it (25600/8 = 3200 = 25 x 128 is split-safe).
3. **Linear forward** (`Exl3LinearMethod.apply`): x bf16 -> f16 (kernel takes kHalf A), `ext.exl3_gemm(A, trellis, C, suh, A_had,
   svh, force_shape_idx=-1, mcg, mul1, force_num_sms=-1)` with C f16 or f32 (`exllamav3_ext/quant/exl3_gemm.cuh/.cu`; A_had is the
   caller-provided buffer for the Hadamard-rotated input when suh is given -- read how `linear_bc.h` / `LinearEXL3.bc` fills it),
   -> bf16. Above exllamav3's AUTO_RECONSTRUCT_THRESHOLD rows (prefill) it dequantizes and runs an f16 GEMM
   (`reconstruct_hgemm`): mirror that (reconstruct_had_slice / had_r_128 + torch.matmul), threshold from K's microbench. Keep
   every buffer static and avoid host syncs: vLLM captures decode under HIP graphs; the gemm autotunes shapes on first call, so
   warm the decode shapes in `process_weights_after_loading` or rely on vLLM's dummy runs (verify).
4. **MoE forward** (`Exl3MoEMethod.apply`): `ext.exl3_moe(hidden f16, out, expert_count, token_sorted, weight_sorted, temp buffers,
   act_fn, K_gate, K_up, K_down, nine pointer tables (gate=w1, up=w3, down=w2; int64 device pointers per local expert), six
   codebook flags, act_limit (swiglu_limit 10.0), num_active, scratch, tables, count_lo, count_hi, m_tile)` exactly as
   `exllamav3/modules/block_sparse_mlp.py` ~1240-1350 calls it, including its tiering (fused kernel for experts with <= fused_rows
   tokens, batched reconstruct groups + `exl3_moe_gather` above) and `FusedBuffers` sizing (~740-760); pointer tables built like
   `modules/multilinear.py`. Routing in: vLLM's topk_ids / topk_weights (global expert ids, 6 a token) -> this rank's local experts
   (expert_map) -> exllamav3's sorted layout. Output: this rank's partial sums; the FusedMoE layer all-reduces (naive EP).
5. **Loader** (`r9700_vllm/models/deepseek_v41_exl3.py`): a thin subclass of the stock ROCm model registered for
   "DeepseekV41ForCausalLM" (ModelRegistry.register_model, the mechanism in `r9700_vllm/models/qwen4_exp.py`), engaged only when the
   quant config is exl3. Its load_weights renames `attn.wo_a.slice.{g}.<t>` to what the vLLM `wo_a` parameter expects (shard g),
   passes `<name>.trellis|suh|svh|mul1` through the stock name mapping to parameters named `trellis|suh|svh|mul1` on the target
   layer (so the stacked mapping delivers `fused_wqa_wkv.trellis` with shard_id 0/1 and the expert mapping delivers
   `experts.w13_trellis` with expert_id / shard_id), accepts from the base shards only `engram.embed.*`, and drops or forwards
   `vision.*` / `aligner.*` / `image_*`. Weight loaders stash the per-shard / per-expert tensors (shapes differ per shard, so the
   named Parameter is a placeholder) and `process_weights_after_loading` builds the contiguous device tensors and pointer tables.
   The model directory is composed, not copied: `~/models/DeepSeek-V4.1-Flash-1017-EXL3/` = symlinks to the pack's 43 shards,
   config.json, chat_template.jinja, exl3_k_map.json + base shards 47/48 + base tokenizer.json / tokenizer_config.json
   (`~/ds41/exl3/compose-model-dir.sh`, idempotent; vLLM globs *.safetensors, so no index file is needed -- verify).
6. **Numerics gates** before any eight-card run: (a) dense: `x @ reconstruct(W)` vs `exl3_gemm` within f16 tolerance for every
   distinct (in, out, K) in the model and M in {1, 2, 4, 8, 16, 64, 512, 4096}; (b) MoE: 48 local experts, random routing, vs a
   per-expert dequantized SiLU-gated MLP reference in f32; (c) the loader's parameter mapping covers every pack tensor name (a dry
   run over `pack-tensors.json` names: each maps to exactly one parameter or an explicit skip list).
7. **Attention / Engram / DSpark are stock features to verify, not to write**: Engram host offload, the ROCm sparse MLA + indexer
   on gfx1201 (our glm5next patch already engages), DSpark with `enable_adaptive_verification: false` (AMD). Fixes, if any, go in
   `r9700_vllm/platform.py` / `models/glm5next.py` the way the GLM MLA prefill fix did.

## Work packages (parallel; each agent owns only its files)

- **K -- kernels** (VM100 card 7; owns `~/ds41/exl3/kerneltest/`, the derived image `r9700/vllm:exl3`, `notes/exl3-kernels.md`):
  derived image = r9700/vllm:dev + the wheel (pip install --no-deps; add exllamav3's python deps only if the quantizer is needed
  for synthetic test tensors); confirm `import exllamav3_ext` and that gfx1201 took the RDNA WMMA path and not an emulation
  (`rocm/rdna_wmma.cuh` vs `rdna_wmma_emu.cuh`: how does exllamav3 pick for gfx12? RDNA4 WMMA operand layouts differ from RDNA3);
  numerics gate 6(a) for every (in, out, K) above; `exl3_moe` gate 6(b) at M = 1..64 tokens with 48 experts; timing table (us a
  call, effective GB/s against the trellis bytes) for decode shapes M = 1, 2, 4, 8 and prefill 512 / 4096; `g_get_num_sms`,
  `exl3_moe_max_concurrency`, smem; `had_r_128` / `reconstruct` behavior; the AUTO_RECONSTRUCT_THRESHOLD crossover measured.
- **L -- linear** (card 6; owns `r9700_vllm/quant/exl3.py`, `tests/test_exl3_linear.py`): design 1-3 and 5's placeholder/stash
  parameter scheme for LinearBase layers (Replicated / ColumnParallel / RowParallel / Merged-stacked), the lm_head dequant path,
  TP slicing rules with 128-alignment assertions, the gemm + reconstruct_hgemm forward; test on one card with synthetic or real
  pack tensors (shard 2 holds layers 0-1) against `reconstruct`, including a simulated TP8 split (sum of row-partials == whole).
- **M -- moe** (card 5; owns `r9700_vllm/quant/exl3_moe.py`, `tests/test_exl3_moe.py`): design 4 and 5's expert stash on this
  pin's FusedMoE / FusedMoEMethodBase (`vllm-src/model_executor/layers/fused_moe/`), EP local-expert mapping, routing conversion,
  buffers, prefill tiering; test against the dequantized reference with 48 experts, K=3 and K=2, M = 1..4096.
- **N -- model** (no GPU until integration; owns `r9700_vllm/models/deepseek_v41_exl3.py`, `~/ds41/exl3/compose-model-dir.sh`,
  `serve/ds41-exl3.sh`, `notes/exl3-layer-map.md`): the layer map (every linear / MoE / embedding the ROCm model builds, with
  prefix pattern, in/out, how it is sharded at TP8, and which quant method it asks for), every DeepseekV4FP8Config-specific touch
  point in `vllm-src/models/deepseek_v41/**` and `deepseek_v4/**` that an exl3 config must satisfy or route around (attributes
  read off quant_config, isinstance checks, `.weight_scale_inv` reads in indexer/compressor/engram/mono paths, the shared-expert
  fusion), the loader subclass (design 5) with the name dry run (gate 6c) over `pack-tensors.json`, the composed model dir, the
  serve recipe (TP=8, EP, engram cpu_offload, DSpark off first, `--max-model-len 32768`, OVERLAYS=emulated-switch, GPUS=0..7,
  clock offset -500) and how the DSpark drafter would load its `mtp.*` EXL3 tensors.

Interfaces between packages: L exposes `Exl3Config` (registered name "exl3", `from_config`, `get_quant_method`, attributes
`avg_bits`, `head_bits`, `codebook`, `k_map`), `Exl3LinearMethod`, and helpers `reconstruct_f16(trellis, suh, svh, K, mul1, mcg)
-> f16 [in, out]`, `k_of(trellis) -> int`, `exl3_gemm_f16(A_f16, trellis, suh, svh, K, mul1, mcg, out_f32: bool) -> C`; M exposes
`Exl3MoEMethod(moe_config, quant_config)`; N exposes `ARCHS` for `__init__.py` and may ask L/M for hooks through the owner.
Placeholder stubs for all three live in the repo (branch `exl3`) so imports resolve from the start.

## Paths

- Mgmt VM (this machine, user devops): repo `/home/devops/r9700-stack` on branch `exl3`; `$TMP` = `/home/devops/.claude/jobs/5e4e947f/tmp`:
  `$TMP/ds41/vllm-src/` (the pin's `vllm` package, .py only), `$TMP/ds41/pack-tensors.json` + `pack-summary.txt` (pack headers),
  `base-tensors.json` + `base-summary.txt`, `$TMP/ds41/exl3/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/` (pack config/README/k_map),
  `$TMP/ds41/hf/DeepSeek-V4.1-Flash-1017/` (base config, index, tokenizer), `$TMP/exl3/exllamav3/` (exllamav3 v1.6.6 source).
  `~/ds41-tools/exl3-sync.sh` copies the working tree to VM100 `~/r9700-exl3/` (no rsync here; it tars over ssh).
- VM100 = `devops@192.168.0.123` (ssh BatchMode works; sudo without password; docker). The eight-card box. `~/r9700-build/repo` is
  the PRODUCTION deploy copy -- never touch it; use `~/r9700-exl3` (synced) with `REPO=~/r9700-exl3`. `~/ds41/exl3/`: pack dir
  (downloading, ~7 MB/s, marker EXL3_DOWNLOAD_DONE in download.log; shards land as `model-000NN-of-00043.safetensors`), `dist/`
  (the wheel), `src/exllamav3` (source, hipified in place by the build), `build-full.log`, `tensor-meta.py`. Base checkpoint
  `~/ds41/hf/DeepSeek-V4.1-Flash-1017` (complete). Images: `r9700/vllm:dev` (serving pin), `r9700/vllm:exl3` once K builds it.
  `~/ds41/exl3/gpu-run.sh <card> <cmd...>` runs a command in the image on one card with the repo and ~/ds41 mounted (owner-provided).
  GPU etiquette: K card 7, L card 6, M card 5; cards 0-4 free for the owner; never run on a card someone else owns; an
  eight-card run needs `bash ~/gpu-offset.sh -500` first (120 V feed). pkill rules: never `pkill -f` a pattern that also appears in
  your own ssh command line (it kills your shell); use `pkill -f "hf downloa[d]"`-style patterns.
- Serving: `serve/serve.sh` (knobs in its header; `MODEL=/models/<dir>` with `~/models` mounted; `EXTRA=` appends engine args;
  `DOCKER_ARGS=` extra docker args; `IMG=`, `REPO=`). `~/probe.py` on VM100 is the chat probe; `bench/` in the repo.

## Milestones

- M1 wheel + kernel gates (K). M2 quant method + loader mapping, dense and MoE numerics (L, M, N). M3 eight-card load, first
  tokens, Engram in host RAM, then DSpark (owner). M4 attention path work if needed; sanity, GSM8K, HumanEval, soak; release.

## Bring-up log

**2026-11-02 01:10-01:35 UTC, dummy weights (`--load-format dummy`, R9K_EXL3_DUMMY=1), eight cards, TP8 + EP.** Four launches:
1. `dequant_f32` built its Hadamard constant with the default dtype, which vLLM sets to bf16 -> f32/bf16 matmul error. Fixed.
2. Stock `SparseAttnIndexer.forward_hip` raises on gfx12 ("requires AITER or gfx950/gfx11"); its ROCm op is Triton/torch
   underneath. `models/deepseek_v41_rdna4.py` opens the gate on RDNA4 (+ `patch:dsv41_rdna4` in register()).
3. Up, eager: load 25.5 GiB a rank in 19 s, KV 2 GiB = 234,740 tokens (fp8_ds_mla, block 128, BLHNC), backends
   ROCM_FLASHMLA_SPARSE_DSV41 / DEEPSEEK_SPARSE_SWA / DEEPSEEK_V41_INDEXER / CompressorBackend, 30.2-31.0 GiB used a card.
   A 2k-token prefill OOMed in `fp8_mqa_logits_torch` (stock's prefill indexer logits: [32 heads, M, N] bf16 + f32 scores,
   512 MiB at 2k x 2k, 16 GiB for a 4k chunk against 32k). The image's aiter carries the Triton kernel with gfx1201
   configs; the same module now routes `rocm_fp8_mqa_logits` to it on RDNA4.
4. Up with HIP graphs (EAGER=0, KVMEM=1, NBT=2048): graph capture 7 s (piecewise 4 + full 3 + decoder replay; torch warns
   "CUDA Graph is empty" for some segments), 28.2-28.7 GiB used a card. Timing probe (`~/ds41/exl3/tprobe.py`, random
   weights so the routing and top-k are random too): **single stream 47.3 tok/s (21 ms a step), 4 streams 114 tok/s
   aggregate (30 a stream), prefill 3,030 tokens 0.8-4.3 s (first prefill JIT-compiles four Triton kernels: compact ragged,
   cp_gather indexer cache, fp8_mqa_logits, block_scores)**. No DSpark yet. Memory: weights ~25 GiB + KV 1 GiB + graphs leaves
   ~3.3 GiB a card; the real pack's footprint should match (the dummy run synthesized K=5 attention, K=3 experts).
5. DSPARK=1 on dummy weights (graphs, KVMEM=1, NBT=2048): drafter builds and captures (18 s, 1.55 GiB of graphs), 29.7-30.2
   GiB used a card. Random weights accept nothing (2,410 drafted, 0 accepted), so the numbers are the speculative loop's
   pure overhead: single stream 24.6 tok/s, 4 streams 54.8 aggregate; at a mean acceptance length of 3-4 the same loop
   would give ~75-100 tok/s. Warm prefill 3,030 tokens in 0.82 s = 3,685 tok/s (the first prefill of a process pays ~11 s
   of Triton JIT: compact-ragged, cp_gather indexer cache, fp8_mqa_logits, block_scores -- a warm-up request belongs in
   the launcher).
6. Decode profile (graphs, single stream, 64 tokens, PROF=1 PROFACT=CUDA): 20.3 ms of GPU time a step over 2,550 launches;
   the sparse-attention decode kernel 1.9 ms, our ar8 pair+cross stages 3.2 ms (~79 all-reduces), expert GEMVs 2.1 ms,
   attention GEMVs 1.3 ms, hyper-connection tilelang kernels 1.8 ms, bf16 skinny GEMMs 0.9 ms, hipBLASLt 1.2 ms, casts
   ~1.2 ms. The RCCL "Generic" kernel (188 ms over the run) is the PREFILL's all-reduces: RCCL's collective log during
   capture shows every [tokens, 5120] bf16 all-reduce above the ar8 host-stage ceiling (~96 KB, i.e. more than ~9 tokens
   a step) going to RCCL at ~2.3 ms a call for 380 KiB. Decode at <= 8 tokens a step stays on ar8; multi-token steps
   (DSpark x several streams, 10+ concurrent streams) will pay 80 x 1-2 ms a step until the ar8 cross stage takes larger
   messages. Open item for after correctness.

**2026-11-02 08:27-09:15 UTC, real weights.** Pack complete (43 shards, 196 GiB), composed dir, launch: weights 139 s, 25.5 GiB a
rank, Engram 2 x 11.8 GiB pinned host a rank, KV (KVMEM 1) 193,534 tokens, graphs 7 s, 28.6-29.2 GiB a card; probe 47.1 tok/s
single / 100 at four streams / 3.4k tok/s warm prefill -- the dummy numbers, as expected.
Correctness: short answers right (Paris, the first primes, three sentences), retrieval from 1.7k and 3.3k-token prompts
right; long generations degrade within ~20-60 tokens with doubled words ("of the the sea", "Recently Recently") and loops.
Bisection by relaunch (each ~6 min): EAGER=1 + R9K_DISABLE=glm5next -> same; EXL3 decode tiers replaced by the dequant
paths (R9K_EXL3_RECON_M=0 R9K_EXL3_MOE_MGEMM_M=0) -> same, and `divergence.py` (greedy decode vs fresh prefill of each
prefix, prefix cache off) shows top-1 agreement 14-20/24 with |dlogprob| 0.6-1.3; **R9K_DSV41_ALL_PREFILL=1 (every step
through the prefill path) -> correct**: 511-token essay with a proper ending, clean LRU code, 17*23=391, agreement 22-24/24
at |dlogprob| 0.02-0.18. So the defect is in the decode-only attention path (SWA ragged decode kernel over the fp8 record,
decode compressor, decode indexer), not in EXL3 / MoE / graphs / Engram slot cache / our GLM hooks. Next split: bf16 KV.

**ROOT CAUSE (09:40 UTC): stock vLLM's ROCm sparse-MLA decode on non-gfx942/950 GPUs.** `rocm_sparse_attn_decode` takes
`inv_rope_positions` so the split-K reduce kernel can fold the attention output's inverse RoPE into its epilogue and
reports the rows as rotated; the caller (`forward_mqa`) un-rotates only the remaining rows. The "fallback path for un-tuned
architectures" (everything but gfx942/gfx950) runs the single-pass ragged kernel, which has no such epilogue, yet the
wrapper still reports every row as rotated -> decode attention outputs reach wo_a with the RoPE rotation still on their
64 rope dims. Prefill rows take the standalone pass and are right. Diagnosed with `divergence.py` (greedy decode vs fresh
prefill of each prefix) and bisected by relaunch: EXL3 decode tiers / graphs / GLM hooks / bf16 KV / Engram all
exonerated; ALL-PREFILL correct; dumping the decode kernel's slot lists showed them correct; dropping
`inv_rope_positions` (standalone inverse RoPE for decode rows) fixed it: agreement 23-24/24, |dlogprob| 0.02-0.13.
Fix: `models/deepseek_v41_rdna4.py` `decode:inverse-rope-unfused`, default on RDNA4 (R9K_DSV41_FUSED_INVROPE=1 opts
out). Worth an upstream report (any ROCm GPU outside gfx942/gfx950 serving DeepSeek V4/V4.1 decodes wrong).
With the fix (graphs, no drafter): single stream 45.7 tok/s, 4 streams 104 tok/s aggregate, warm prefill 3.4k tok/s;
thinking-mode answers correct (Paris; 17*23=391 with the 340+51 explanation; clean Fibonacci; train problem 8pm).
DSPARK=1 with the fix (graphs): quality intact (557-token essay to a clean stop, LRU code); mean acceptance length 3.6 on
code / 2.5 on prose, per-position 0.77/0.61/0.52/0.39/0.33 (code); usage-based speed 96 tok/s on code, ~42 tok/s on prose
(vs 43-46 without the drafter): the verify step at 6 tokens costs ~2x a single-token step (MoE mgemv per slot, attention
over 6 queries), so prose breaks even and code doubles. Multi-token decode MoE is the lever (also task 49 for 4+ streams).
**Validation with the fix (10:00-10:40 UTC, DSPARK=1, graphs, KVMEM 1, NSEQ 4):** GSM8K 200 problems greedy with thinking:
**196/200 = 98.0 %**, 0 truncations, 5.2 min at ~140 tok/s aggregate (4 streams; drafter acceptance 3.2-3.6 on math).
Needles 3/3 at 8k, 16k and 28k tokens (prefill 2.5 / 6 / 8.6-11.5 s). Strict 8-way sanity after 24k-token prefills:
0 bad of 80 (10 rounds). Soak: 8 clients x 10 min, 10k-28k prompts -- see below.
Soak (8 clients, 641 s, prompts 10.3k-30.6k tokens, median 21k): 85 requests, 0 errors, 0 rejections, 2,744 prompt tok/s,
median latency 59 s; VRAM settled at 30.7-31.0 GiB a card (allocator growth under long prefills; no OOM). Probe after the soak
(usage-based, DSpark on): single stream 50.7 tok/s, 4 streams 117 tok/s aggregate, warm prefill 3.4k tok/s.
Server left running on VM100 (container ds41, port 8080) for Brian's check-in.

## Phase 2 -- performance (started 2026-11-02 ~14:00 UTC; Brian: "begin the payoff process")

Target: the two DGX Sparks on TensorFold serve this pack at 80.7 tok/s prose / 118 code single stream (thinking off), 154-166
at four streams, 2.1k tok/s prefill. We are at ~42-51 prose / ~96 code / 117 at four streams / 3.4k prefill. Where the step
goes (dummy-weight profile, graphs, single stream: 20.3 ms GPU, 2,550 launches): our ar8 pair+cross stages 3.2 ms (~79
all-reduces), expert GEMVs 2.1 ms, sparse-attention decode 1.9 ms, hyper-connection tilelang kernels 1.8 ms, attention GEMVs
1.3 ms, hipBLASLt 1.2 ms, casts ~1.2 ms, bf16 skinny GEMMs 0.9 ms. The DSpark verify step (6 tokens) costs ~2x a single
token: the MoE decode tier (exl3_mgemm per slot) runs ~70 us a local pick plus ~45 us fixed (T=1 77 us a layer, T=8 376 us),
and stock's decode attention kernel on RDNA4 is the "fallback for un-tuned architectures": one workgroup per (query,
16-head block) walking every key serially (48 us a layer a query, 1 of 32 CUs busy). Above ~9 tokens a step the
[tokens, 5120] bf16 all-reduces leave ar8's host stage (96 KB ceiling) for RCCL at 1-2 ms each.

Packages (parallel Opus agents; the owner integrates on eight cards and re-measures):
- **P1 -- batched EXL3 expert kernel for the decode tier** (owns `kernels/r9k_moe_exl3.hip`, `r9700_vllm/moe/exl3_decode.py`,
  the `R9K_EXL3_MOE_DECODE=r9k` tier in `r9700_vllm/quant/exl3_moe.py`, `tests/test_exl3_moe_r9k.py`; card 5, 4 if needed).
  Physics: at T=6 x 6 picks = 36 slots, ~4.5 are local a rank a layer = 60 MB of trellis = 0.1 ms at 600 GB/s; we spend ~300 us.
  Step 1: measure exllamav3's exl3_mgemm grid/occupancy at 1-6 local slots (rocprof or torch profiler) and say why it is slow.
  Step 2: our kernel: grid over (slot, 128-output tile), gate and up for the same slot/tile in one workgroup (the rotated input
  read once), per-slot input Hadamard with that expert's suh (in-kernel from LDS or a small pre-pass), EXL3 trellis decode
  reusing exllamav3's device functions (MIT; copy with attribution in CREDITS.md; the 16x16 tile layout and the mul1 codebook
  must match bit for bit -- validate against reconstruct), SiLU x up with act_limit, down projection the same way, 128-point
  output Hadamard + svh in LDS, per-slot f32 outputs gathered per token in pick order (deterministic, as the current tier).
  K = 2 and 3 (main), 4-6 (drafter: 16 local experts). No host syncs; graph-capturable; non-local slots skipped on device.
  Targets: T=1 <= 50 us a layer, T=6 <= 120 us, T=24 <= 400 us; >= 400 GB/s on trellis bytes. Then integrate as the decode tier
  and keep exllamav3's tier as `R9K_EXL3_MOE_DECODE=mgemv`.
- **P2 -- ar8 beyond 96 KB** (owns `r9700_vllm/comm/r9k_ar8.py`, `r9700_vllm/comm/r9k_ar.py`, `kernels/r9k_ar*.hip` if needed,
  `tests/test_ar8.py`; all eight cards for short collective tests, timing runs when the other cards are quiet). Extend the
  pinned-host cross stage among the four evens to messages up to `R9K_AR8_HOST_MAX_KB` (default 2048): keep the one-shot host
  exchange to ~96 KB, add a two-shot (reduce-scatter over host memory, then all-gather) above it -- each even writes its
  buffer, reduces its quarter from the three peers, writes the reduced quarter back, reads the other quarters (2.75 S of
  traffic vs 4 S, two barriers). Pair stage stays P2P (two-shot <= 512 KB today; raise if the sizes need it). Graph-safe
  (device-mapped flags, no host syncs). Targets: 240 KB (24 tokens) <= 100 us, 1 MB <= 250 us, bit-exact across ranks, within
  1 bf16 ulp of the rank-ordered fp32 reference, graph replay of chained calls (tests/test_ar8.py, extend TOKS to 24, 64, 200).
- **P3 -- RDNA4 decode attention** (owns `r9700_vllm/models/deepseek_v41_rdna4.py` (the decode launcher part),
  `r9700_vllm/attn/dsv4_decode.py` if the launcher grows, `tests/test_dsv4_decode.py`; card 7, 6 if needed). On RDNA4 route the
  decode attention launch to stock's split-K Triton kernels (`_sparse_attn_decode_partial_kernel` + `_sparse_attn_decode_reduce_kernel`,
  whose FUSE_INV_ROPE epilogue is correct) instead of the serial ragged fallback: our wrapper replaces the body of
  `rocm_sparse_attn_decode` for the fallback case, choosing `num_splits` / `block_k` / `num_warps` for 32 CUs and wave32 (the
  gfx950 tuning assumes 256 CUs), and then reports the rotated rows truthfully (dropping the current no-fused-invrope
  workaround for that path). Both the fp8 and bf16 launchers. Validate with bench/ds41/divergence.py (>= 22/24 at |dlogprob|
  < 0.2 on the real server -- ask the owner to run it) and a unit test against the ragged kernel's output on synthetic caches.
  Targets: T=1 <= 15 us a layer, T=6 <= 40 us (now 48 us x queries). Stretch: our own HIP kernel (attn/r9k_sparse_mla.hip
  does the GLM geometry; DSv4 is 448 nope + 64 rope, fp8 records with e8m0 scales per 64, SWA + top-k segments, attn_sink).
- **L (resumed) -- casts** (owns `r9700_vllm/quant/exl3.py`; card 3): remove the standalone cast/copy kernels around the EXL3
  GEMV (bf16 -> f16 input, f32 -> bf16 output, .contiguous() copies): had_r_128 or the kernel taking bf16 input if exllamav3
  allows, f16 C with one fused cast, preallocated buffers; count launches per apply before/after and the per-step saving.

Integration: the owner relaunches `serve/ds41-exl3.sh` with the new tiers, runs divergence.py, needles, GSM8K, tprobe at 1 and
4 streams with and without the drafter, and compares with the Phase 1 numbers above.

### Phase 2 results so far (2026-11-02 18:15 UTC)
- **P3 (decode attention) + L (casts) integrated** (a807cd4, d67dd59). On the eight-card server with graphs: divergence 23-24/24
  at |dlogprob| 0.02-0.14 (the split-K reduce's fused inverse RoPE is right; the ragged workaround stays as
  R9K_DSV41_DECODE=ragged). Short-context probe: no drafter 51.7 tok/s single (was 45.7), 125.9 at 4 streams (was 104);
  drafter 55.6 single (was 50.7), 122.6 at 4 streams; essay 50 tok/s (was 42), code 97; prefill 3.4k tok/s. Decode after a
  long prefill is flat: 51.6 / 51.5 / 51.5 / 51.4 tok/s at 2k / 8k / 16k / 28k tokens of context (no drafter) -- the stock
  fallback kernel cost 180 us a layer at SWA 128 + top-k 512 (P3's table), the split-K path 19 us.
- Remaining step time is in the MoE decode tier (P1), the collectives (P2, also the >9-token steps), the hyper-connection
  kernels and the EXL3 GEMV launch count.

### Phase 2 results, all four packages integrated (2026-11-02 19:00 UTC, commits d67dd59 a807cd4 6ad1c94 + P2)
Eight cards, graphs, KVMEM 1, drafter on (default). Divergence 22-24/24 at |dlogprob| 0.04-0.10. **Single stream 70.4 tok/s
(morning: 50.7), 4 streams 161.9 tok/s aggregate (117), code 137 tok/s (97; the Sparks: 118), prose essay 61 tok/s (42; Sparks
80.7), decode after long prefills 81.8 / 80.0 / 74.5 tok/s at 2k / 16k / 28k context, prefill 3.4k tok/s (Sparks 2.1k).**
Drafter acceptance 2.4-2.5 on the probe prompt. VRAM 30.7-30.9 GiB a card (the r9k tier reserves ~1.2 GiB of scratch a rank;
headroom ~1 GiB). Lesson: the tree sync preserves source mtimes, so serve.sh's "kernel newer than libr9k.so" rebuild check
missed P1's kernel once (the first "P1" integration measured the old tier); exl3-sync.sh now touches changed kernel sources.
Launches a step with the drafter: ~810 (51,932 over 64 steps) vs ~2,550 before.
Remaining gap to the Sparks is prose single-stream (61-82 vs 80.7); code and four-stream aggregate are at or above parity.
GSM8K with all Phase 2 tiers (drafter on, graphs): **194/200 = 97.0 %**, 0 truncations, **2.9 min** (morning: 196/200 in 5.2 min
-- the two-problem difference is within the noise of 200 problems; P1's kernel is the more accurate of the two expert tiers).
P2's clean ar8 timing run (~/ds41/exl3/ar8-final.sh -> ar8-final.log) runs on the idle box after this; its SPLIT_KB / HOST_1S_KB
sweep decides whether the 96 KB defaults move down.

### ar8 clean timing (P2's quiet-box run, 2026-11-02 19:15-19:21 UTC, us per all-reduce, HIP graphs, best of 3)
| tokens x 5120 bf16 | KiB | RCCL | old ar8 (to 96 KB, then RCCL) | new ar8 (defaults 96/2048) | new, SPLIT_KB=32 |
|---|---|---|---|---|---|
| 1 (4096-wide) | 8 | 112 | 21 | 21 | 21 |
| 4 (4096-wide) | 32 | 117 | 48 | 47 | 46 |
| 6 | 60 | 105 | -- | -- | 37 |
| 8 | 80 | 99 | 123 | 78 | 47 |
| 10 | 100 | 102 | RCCL | 49 | 49 |
| 16 | 160 | 111 | RCCL | 65 | 65 |
| 24 | 240 | 125 | RCCL | 83 | -- |
| 64 | 640 | 182 | RCCL | 157 | -- |
| 100 | 1000 | 240 | RCCL | 230 | -- |
| 128 | 1280 | 281 | RCCL | 281 | -- |
| 200 | 2000 | 402 | RCCL | 419 | -- |
SPLIT_KB 32 and 48 tie (6 tokens 37 us, 8 tokens 47.5); 64 loses at 60 KB (58.5); HOST_1S_KB=48 loses at 80-90 KB (61-66).
Defaults moved to SPLIT_KB 32; HOST_MAX_KB stays 2048 (the split path ties RCCL at 1.28 MB and loses 4 % at 2 MB -- not worth a rule; a 1.5 MB ceiling also sent the test burst's 2 MB message into the flat collective, which has no 4-rank fallback object -- that stray-oversize case now goes to RCCL); the
three-kernel variant (R9K_AR8_FUSED=0) is 10-45 % slower than the fused kernel at every size.

## Phase 3 -- KV cache headroom, hyper-connection kernels, draft tuning; then release v0.4.0 and deploy (Brian, 2026-11-02 21:xx UTC)

Memory today (a 32 GiB card, graphs, drafter on): weights 25.5 GiB (EXL3 experts 22.8, attention 0.6, bf16 pieces: wo_a 0.32,
shared experts 0.35, lm_head 0.16, embed 0.17), the r9k expert tier's scratch ~1.2 GiB (reserve() for 32 rows), the drafter
~0.3 GiB + graphs (capture "took 1.55 GiB" with DSpark vs ~0 without), KV 1 GiB (= 193,534 tokens; ~5.5 KB a token across
the SWA / compressed / indexer caches), prefill transients ~1-1.5 GiB at NBT 2048 -> 30.8 GiB used, ~1 GiB free. Target:
3+ GiB of KV (>= 500k tokens) so 4 streams x 128k or one 500k-token session fit.
Step time today (drafter, single stream ~14 ms): hyper-connections 1.8 ms (stock TileLang `mhc_pre_big_fuse`, `mhc_post`,
`hc_prenorm_gemm`: 2 launches a layer each, 9-12 us each, latency-bound at M=1), the rest measured in Phase 2.

Packages (parallel; the owner integrates, then runs the release gate):
- **S -- serving tuning** (owns `serve/ds41-exl3.sh` defaults, `notes/exl3-serving.md`, `bench/ds41/` additions; the
  eight-card server is S's -- H and P1 keep their single-card tests under 1 GiB on cards 6 / 5 meanwhile):
  (1) memory audit per rank at steady state (what torch holds vs what the driver reports: `torch.cuda.memory_stats` in a
  worker, the "Model loading took" / "Graph capturing took" lines, sysfs VRAM) -> a table by component; (2) reclaim:
  cudagraph capture sizes (only the sizes the drafter produces: multiples of 6 up to 24 plus the drafter's own; vLLM's
  `--cuda-graph-sizes` / compilation config), the expert tier's scratch (P1 shrinks it; take P1's new size), unused MoE
  tier buffers, prefill transients (NBT 2048 vs 1024), the dequantized wo_a (320 MB bf16: keep for now); (3) raise KVMEM
  to the maximum that survives the soak at 32k and a 128k-context needle (MAXLEN 131072; prefill 128k tokens ~40 s at
  3.4k tok/s) with >= 0.7 GiB left; (4) draft length: num_speculative_tokens k in {3, 4, 5, 6} x {prose, code, GSM8K-style}
  at 1 and 4 streams (tprobe + ask3 + longdecode; the verify step costs grow with T), and `enable_adaptive_verification`
  true/false (the brief said AMD must disable it -- verify whether this pin runs it); (5) defaults into the launcher:
  KVMEM, MAXLEN, NBT, DSPARK k; validation: divergence.py, needle at 8k / 28k / 100k, sanity_stress, soak 10 min, GSM8K 200.
- **H -- hyper-connection kernels** (owns `kernels/r9k_mhc.hip`, `r9700_vllm/attn/mhc.py` or `r9700_vllm/hc_dsv4.py`,
  the hook in `r9700_vllm/models/deepseek_v41_rdna4.py` (mhc part only), `tests/test_mhc_r9k.py`; card 6, < 1 GiB):
  stock's ops live in `vllm/model_executor/layers/mhc.py` (MHCPreDelayedOp / MHCPostOp, `hc_collapse_triton`) with backends
  in `vllm/model_executor/kernels/mhc/` (tilelang.py / tilelang_kernels.py = what RDNA4 runs today, triton.py, torch.py =
  the reference, aiter.py, warmup.py) and the model's use in `vllm/models/deepseek_v41/amd/model.py` ~228-480 (hc_mult 4,
  sinkhorn 20 iterations, eps, post_alpha 2.0; `hc_attn_fn` [24, 20480], `hc_attn_base` [24], `hc_attn_scale` [3], same for
  ffn). First measure the Triton backend on RDNA4 against TileLang at M = 1, 6, 24 (a backend switch may already win);
  then one HIP kernel per op for decode widths (M <= 32): the [M, 20480] x [20480, 24] mix GEMV, sinkhorn, the stream mix,
  RMSNorm, and the post update, fp32 math with the torch reference's rounding points, bit-exact against torch.py (or within
  bf16 rounding where stock itself differs), graph-safe; prefill widths stay on stock. Targets: <= 3 us a launch at M=1,
  <= 6 us at M=24 (from 9-12 us), i.e. ~1.2 ms a step. Our Qwen/GLM hc kernels (kernels/r9k_hc.hip, r9700_vllm/hc.py) show
  the house style for these elementwise-plus-skinny-GEMV kernels.
- **P1 (resumed) -- expert tier scratch** (owns `r9700_vllm/moe/exl3_decode.py`, `kernels/r9k_moe_exl3.hip`; card 5, < 1 GiB):
  reserve() for the real decode shapes (default R9K_EXL3_MOE_R9K_M = 24: four streams x a 6-token verify), bf16 or chunk
  count cuts for the f32 partial buffers, no per-layer duplication: target <= 256 MiB a rank with no change to numerics or
  timing at T <= 24; report the exact new footprint for S.
Then (owner): integrate, release gate on the same tree -- GLM-5.3-Flash TP8 (ar8's split path now starts at 32 KB, so GLM's
multi-token steps take it), Flash-Next TP4 and 27B TP2 (R9kAllReduceN changed) at the v0.3.8 numbers -- then v0.4.0:
release note, CHANGELOG, README section, annotated tag, GitHub Release, deploy to VM100's production copy.

### Phase 3 results (2026-11-03 03:00 UTC; commits 88f6cb5 P1, 5a943a7 H, 79d25b5 RoPE trim, S's notes/exl3-serving.md)
- **P1 (scratch):** the r9k expert tier sizes its scratch once for the largest decode call (R9K_EXL3_MOE_R9K_M = 24: four streams
  x a 6-token verify) from one allocation shared by every layer and the drafter: 23.7 MiB a rank (31.6 at M = 32) instead of
  ~1.2 GiB of allocator-pool growth from incremental reserve() calls (the buffers themselves were ~30 MiB; every outgrown set
  stayed in the pool and each mid-size buffer rounded up to a 20 MB block). Numerics and timing unchanged (bit-exact trellis
  decode, graph replay = eager). Steps above 24 tokens go to the fused tier.
- **H (hyper-connections):** one HIP launch a seam (`kernels/r9k_mhc.hip`, `r9700_vllm/hc_dsv4.py`): the post update of layer l
  folded into the delayed pre of layer l+1, the [M, 20480] x [20480, 24] mix as a deterministic last-workgroup reduction, fp32
  with the torch reference's rounding points; bit-exact residual / collapse / post against torch at M = 1..32. 25.7 -> 8.0 us a
  seam at M = 1 (~1.4 ms a step back); prefill widths stay on stock's TileLang kernels. `R9K_DSV41_MHC=stock` restores them.
- **S (serving):** memory audit against the driver's sysfs counter (hipMemGetInfo under-reports free VRAM by 0.9-1.3 GiB on this
  pin): weights + fixed buffers 27.3 GiB a rank (experts 23.5, drafter 1.1, attention 1.0, RoPE caches 0.5, shared experts 0.34,
  embed / head 0.16 each, router 0.15, workspace 0.17), torch cache 0.8, C++-held 0.7, non-torch 1.2. Reclaimed: expandable
  segments -470 MiB, the drafter-shaped graph list -35 MiB (capture 27 -> 19 s), RoPE trim -496 MiB at 32k (owner, 79d25b5),
  P1's scratch -1.2 GiB. Draft length (table in notes/exl3-serving.md): k = 5 wins; `enable_adaptive_verification` runs on
  RDNA4 (+22-28 % per stream on four-stream prose, -5-6 % single-stream prose, code unchanged, +78 MiB of varlen graphs, capture
  71 s). With H: prose 74.6 / 70.1 (adaptive) tok/s single, 38.3 / 46.8 per stream x 4; code 158-159 / 84; GSM8K-style 121 / 114,
  70 per stream x 4. Final KV budget, the 128k launch and the validation: notes/exl3-serving.md sections 3 and 5 (release note).

## Phase 4 -- 2x the Sparks (the v0.4.1 program; Brian 2026-11-03: "We just need to beat the DGX numbers ... We should beat it by 2x")

Targets (two DGX Sparks on TensorFold, the same pack, times two) against v0.4.0: single-stream prose **161** tok/s (we are at
70-75), code **236** (158), four-stream aggregate prose **308** (187), code **332** (336, done), prompt processing **4.2k**
(3.4k; not in items 1-4). Speculation gives ~2.0 accepted tokens a step on prose and ~4.4 on code with the checkpoint's drafter,
so prose is step latency: 27 ms today for a 6-token verify step plus the draft (74.6 tok/s x 2.06 tokens), 12.5 ms for 2x. At
four streams the 24-token step is 38 ms (46.8 x 1.78).

Physics of one verify step on one rank (estimates from the layer map; the ledger below replaces them with measurements):
experts 4.3 local a layer at T=6 (34 distinct of 384 for 36 picks) x 12.2 MiB x 40 = 2.2 GB; attention and dense linears
(fused_wqa_wkv 5.7 MB, wq_b 3.3, wo_a **8.4 bf16**, wo_b 3.3, indexer, shared experts **8.8 bf16**, router 3.9 bf16, hc 3.9 f32)
~1.5 GB; the drafter ~0.2 GB: **~4 GB = 6.3 ms at 640 GB/s**, plus 80 all-reduces at an ideal ~12 us = 1 ms. The floor is
~7.5 ms a step = ~270 tok/s on prose. We spend 27 because the step is ~810 launches (51,932 over 64 steps) at a ~4 us
graph-node floor each (~3.2 ms of nothing), the collectives are three kernels a call (8 KB 21 us, 32 KB 47, 60 KB 37; x 80 = 1.7-3.8
ms), the attention GEMVs run at 150-270 GB/s through exllamav3's per-call path with casts around them, the drafter repeats the
whole structure for three layers, and the expert kernel sits on the -500 MHz clock. At 24 tokens the all-reduce is 240 KB
(83 us x 80 = 6.6 ms of the 38) and the expert kernel reads ~15 local experts a layer (7.3 GB, 11.5 ms).

**Z -- the step ledger first (owner, eight cards, ~1 h after the v0.4.0 deploy).** On the real server with graphs: (1) per-kernel
GPU time of one 6-token verify step and of the draft, by category (expert tier, EXL3 GEMVs, hipBLAS bf16, attention decode,
indexer, collectives by size, hc seams, norms / casts / glue, sampler), from rocprof kernel traces of an eager step and the
captured graph's node list (hipGraphGetNodes through a worker extension like memprobe_ext.py; one line a node: kernel, grid,
bytes); (2) the wall-clock split: graph replay time vs the CPU gaps between replays (scheduler, rejection sampler, async
scheduling) from the torch profiler on the model runner; (3) the same at 24 tokens (four streams) and 48 (eight, NSEQ 8 launch);
(4) with ADAPTIVE true and false. Output: notes/exl3-ledger.md with a table "us a step by category, T = 6 / 24 / 48" and the
launch list; every package below quotes its row of that table before and after.

Packages (parallel Opus agents; the owner integrates on the eight cards and re-measures; v0.4.1 ships when the four are in
and the gate holds):
- **A -- eight-stream verify steps** (owns `r9700_vllm/moe/exl3_decode.py`, `kernels/r9k_moe_exl3.hip`, `tests/test_exl3_moe_r9k.py`,
  the NSEQ-8 recipe in `serve/ds41-exl3.sh`; card 5, then the eight cards for the collective part when quiet). The r9k expert
  tier to T = 48 (R9K_EXL3_MOE_R9K_M 48, scratch ~48 MiB; a 48-token step has ~36 local picks over ~25 distinct experts a rank
  a layer -- confirm the device-side grouping reads each expert's trellis once per group, not once per pick, and that the
  per-tile workgroups keep >= 400 GB/s at 20-30 groups), graph sizes 1,2,4 + 6 x 1..8, KV for 8 x 32k (262k; v0.4.0 has
  329k+). Then the collective at 480 KB (48 x 5120 bf16; today ~130 us on the split path, RCCL ~160): measure the split path's
  stage breakdown (pair reduce-scatter, host exchange, pair all-gather, the two barriers) and stream the host stage with more
  workgroups and 16-byte loads; target <= 90 us at 480 KB, <= 50 us at 240 KB (the four-stream message, 83 today). Acceptance:
  spectune at 8 streams (new column) prose aggregate >= 310 tok/s and code >= 340; sanity at 9 and 16 streams; a 10-minute
  8-client soak; divergence unchanged.
- **B -- launch fusion** (owns `r9700_vllm/quant/exl3.py` (apply path), `r9700_vllm/models/deepseek_v41_rdna4.py` (new hooks),
  new `kernels/r9k_dsv4_glue.hip` + `r9700_vllm/dsv4_glue.py`, tests; card 6; the launch list from Z). From the ledger's per-layer
  launch list, fold the glue: RMSNorm (attn_norm / ffn_norm / q_norm / kv_norm) + the EXL3 input Hadamard (`had_r_128` with
  `suh`) + the bf16 -> f16 cast into one pre-pass kernel a GEMV group (3 launches -> 1); the router (gate GEMV [5120 -> 384] f16
  + bias + sigmoid + group top-k + weight normalization) into one kernel in the style of r9k's Qwen / GLM routing kernels (4-6
  -> 1); the MoE combine (pick weights x per-slot partials + the shared expert + residual) into the expert kernel's epilogue or
  the collective's load stage (with C); the attention projections of a layer as one multi-matrix launch where exllamav3's
  `exl3_mgemm` does not already stack them (fused_wqa_wkv + indexer wq_b share K = 5120 / 1280 -- check). Target: ~810 -> <= 350
  launches a step, ~2 ms. Everything bit-exact or within bf16 rounding of today's path (divergence.py, GSM8K 200).
- **C -- collectives** (owns `r9700_vllm/comm/r9k_ar8.py`, `r9700_vllm/comm/r9k_ar.py`, `kernels/r9k_ar.hip`, `tests/test_ar8.py`;
  all eight cards for short tests, timing when the box is quiet). (1) A single-kernel all-reduce for messages <= 64 KB: the pair
  stage, the quartet host exchange and the pair all-gather as three phases of one kernel with device-scope acquire / release
  flags (the mechanism r9k_ar.hip already uses between kernels), one or two workgroups a rank, a spin timeout knob for debugging
  (every rank launches it in graph order, so all eight are resident). Targets: 8 KB 21 -> <= 10 us, 32 KB 47 -> <= 24, 60 KB
  37 -> <= 26; ~1.5-2 ms a step over 80 calls. (2) The fused MoE reduction with B: the collective's first phase reads the
  expert tier's per-slot f32 partials with the pick weights (+ the shared expert's output and the residual) instead of a
  pre-reduced bf16 buffer -- deterministic (fixed pick order), one fewer write + read of [T, 5120] and 2-3 launches a layer.
  Bit-exact across ranks, within 1 bf16 ulp of the rank-ordered fp32 reference, graph replay of chained calls, the GLM /
  Flash-Next / 27B gates untouched (`R9K_AR8_ONESHOT=0` keeps the three-kernel path).
- **D -- attention GEMV bandwidth** (owns a new `kernels/r9k_exl3_gemv.hip` + `r9700_vllm/quant/exl3_gemv.py`, the M <= 8 path
  in `r9700_vllm/quant/exl3.py`, the `wo_a` override in `r9700_vllm/models/deepseek_v41_exl3.py`, tests; card 7). Our own
  multi-row trellis GEMV from P1's expert kernel (the same device functions, MIT with attribution): a list of (matrix, input,
  output) per launch, Hadamard / suh / svh in-kernel, bf16 in and out (no casts), one workgroup a 128-output tile streaming K,
  >= 450 GB/s on trellis bytes at M = 1..8 (exllamav3's path: 150-270). Then `wo_a` back to EXL3: the pack's eight 128-aligned
  slices are exactly the o-groups (= ranks), so rank r serves slice r as a [4096 -> 1024] EXL3 GEMV through an `_o_proj`
  override of stock's bf16 `_get_cached_wo_a_bf16` path (the loader already routes slices to an EXL3 wo_a with shard_id g):
  -320 MiB a rank and 8.4 -> 2.6 MB a layer read. Target: the attention + dense GEMV row of the ledger halved (~2 ms a step),
  bit-exact against `reconstruct` + torch matmul in fp32.

Expected after A-D: the 6-token step 27 -> ~15-17 ms (prose single ~105-120 tok/s, code ~220, four-stream prose ~300, eight-stream
prose > 310 aggregate). The rest of the way to 161 on prose is item 5 (full SCLK: the expert kernel is ALU-bound at -500 MHz;
needs the 208 / 240 V feed) and a fused per-layer decode kernel (research). Item 6 (decode context parallel: the KV striped
across ranks instead of replicated) is the route past the Sparks' 524k validated context and is scheduled after v0.4.1.
Rules unchanged: stock vLLM / ROCm / exllamav3, unmodified checkpoints, plugin only; every package bit-exact or within bf16
rounding; the Qwen and GLM gates on the same tree before the tag.

### Phase 4, the ledger's verdict (2026-11-03 12:30 UTC; notes/exl3-ledger.md, package Z)
The default single-stream step is **25.4 ms** (74 tok/s at 1.89 accepted tokens) and **GPU-bound**: the kernels sum to 23.5 ms,
launch + CPU gaps are <= 1.9 ms, so the "~810 launches at a 4 us floor" picture was wrong (it counted kernels a token; a step is
1,754-1,866 launches, glue 1.2-1.4 ms of GPU time). Where the 23.5 ms goes: **collectives 9.6 ms (41 %)** -- of which **4.1 ms is one
kernel, the flat eight-rank all-gather `r9k_ag_oneshot_nrank_kernel<8>`** (the target logits gather at <= 4 verify rows lands on it
because the message fits R9K_AG_MAX_KB 128 and leaves RCCL, which does the 6-row gather in 0.20 ms; Engram's two gathers 0.65-1.07
ms each), 81 all-reduces at 71-81 us (40-60 KB) + 106 us (90-160 KB); bf16 GEMMs 4.3 ms (wo_a 1.1, shared experts 1.2, router 0.5,
heads 1.6 incl. the drafter's markov head 0.54-0.67 = five sequential [129280 x 256] f16 GEMVs + argmax); expert tier 2.6 ms;
EXL3 GEMVs 2.0 ms at 288-292 GB/s; glue 1.6 ms; decode attention 0.9; hc seams 0.75; drafter 2.6-3.2 ms in all. At four streams
(40.0 ms): collectives 13.3, expert tier 7.1, **EXL3 GEMVs 5.0 because every EXL3 linear dequantizes its whole matrix above 16 rows**
(7.6 ms at a true 24-row verify), bf16 5.5. Adaptive verification costs 8-12 % single-stream only because of the slow gather
(74 vs 81 tok/s), and wins at four streams (202 vs 155). NSEQ 8 (T = 48) does not launch on v0.4.0 (HSA memory aperture
violation in the sparse-MLA warm-up; v0.4.0's r9k tier also caps at 42 tokens).
**Re-prioritised:** C first fixes the all-gather (route <= 128 KB gathers back to RCCL as the stop-gap, then a proper one-kernel
all-gather; ~3.5-4 ms a step = ~15 %), then the one-kernel all-reduce; D's multi-row GEMV must cover M <= 48 so the dequant path
never runs in decode (~5 ms at four streams), then wo_a; A fixes the NSEQ 8 launch fault before its T = 48 tier work; B's pool is
the 1.6 ms of glue plus the drafter's head path (markov head, lm_head gather), not launch gaps. New floor estimate after A-D:
single-stream step ~15-16 ms (~115-125 tok/s prose), four streams ~26-28 ms (~290-310 aggregate).

### Phase 4 results, package C (collectives; quiet box, HIP graphs, best of 3; 2026-11-03 12:49-12:51 UTC, tables in VM100 ~/ds41/exl3/C4/)
- **The all-gather.** v0.4.0 routes gathers whose eight-way result exceeds R9K_AG_MAX_KB (128) but whose piece does not onto the flat
  eight-rank kernel, where every rank stores its whole piece into seven peers across the root complexes -- the same fabric cliff as
  the flat all-reduce: **403 / 709 / 919 / 1040 us at [3 / 4 / 5 / 6, 3200]** (Engram) and **890 / 2585 / 3896 us at [1 / 2 / 4,
  16160]** (the target logits at the adaptive-trimmed verify widths), against RCCL's 76-85 and 80-144. C's one-kernel all-gather
  (own piece -> own host pool + partner's pair pool over P2P; pull the quartet's pieces from host memory and forward to the
  partner; copy the partner quartet's four from the pair pool; nothing crosses a root complex; pure copies, exact bits):
  **15.7-47.5 us on the Engram shapes, 39.7 / 71.9 / 129.1 / 181.7 / 233.9 us at 1 / 2 / 4 / 6 / 8 rows x 16160** (RCCL 80 / 103 /
  144 / 187 / 229). Defaults: R9K_AG8_OS_KB 192 (RCCL above), R9K_AG8_FLAT_KB 0 (the flat kernel never). Worth ~3.8 ms of the
  25.4 ms single-stream step.
- **One-kernel all-reduce** (three phases with system-scope flags, host pools, 16 blocks a rank): **14.6 / 17.3 / 23.5 / 27.2 us at
  8 / 16 / 32 / 40 KiB** (stage kernels 21 / 34.5 / 46 / 34.7; RCCL 107-114), 39.8 at 60 KiB and 41.8 at 64 against the fused
  split path's 37.6 / 38.5 -> default R9K_AR8_OS_KB 48. Peer writes instead of host pools (R9K_AR8_OS_P2P_KB) are 83-153 us at
  every size: off. Bit-identical to the stage kernels (same rounding order), so GLM TP8's results cannot change; its speed goes
  through the GLM gate (its 1-row logits gather and its <= 48 KiB all-reduces take the new kernels).
- **Fused MoE reduction** (the expert tier's per-slot f32 partials x pick weights + the shared expert formed inside the collective's
  first phase; no residual add before the all-reduce on this pin): T = 1 **18.5 us vs 40.9** for add + stage kernels; no gain at
  60 KiB, hence the 48 KiB cap (fused for T <= 4). One launch a layer saved. Hook r9700_vllm/moe/exl3_combine.py, installed from
  the EXL3 model classes (integration step), R9K_EXL3_MOE_AR=0 to disable.
- **Real server, v0.4.0 + C on a clean tree** (git archive of v041 a970cd5 -> VM100 ~/r9700-exl3-C2, combine hook wired; 14:37-14:55
  UTC, `~/ds41/exl3/C4/clean-C2.log`; v0.4.0 in parentheses): adaptive off -- prose x1 **80.1-80.3** (74.6), prose x4 39.0-39.3 per
  stream (38.3), code x1 **173.5-175.3** (158.1), code x4 85.5-85.9 (84.3); adaptive on -- prose x1 **75.9-83.0** (72.5-73.0), prose x4
  **43.5-51.5** (44.3-45.6; 159-188 aggregate), code x1 **176.6-177.7** (152.2), code x4 **86.6** (81; 316-320 aggregate). Divergence
  22 / 24 / 24. Single stream +8-16 %, four streams +0-13 %, no regression: the all-gather fix is worth less on the server than the
  per-call arithmetic said (the logits gather only hits the slow kernel at <= 4 verify rows, i.e. with adaptive trimming) and the
  one-kernel all-reduce saves ~20-30 us x 81 calls. Committed as b9c07e8 + a970cd5 (build.sh) on v041.
  LESSON: an earlier "HEAD + C" server run was really the union of every package's uncommitted edits (the agents' private trees were
  copies of the shared working tree), and it showed a four-stream regression (prose x4 26-29, code x4 63) that the clean tree does
  not have -- so one of A / B / D's in-progress changes is slower at 24 rows; each package is integrated on a clean tree from now on.

### Phase 4 results, package B (decode glue; card 6, graph replay, us a call, warm weights; 2026-11-03; notes/exl3-glue.md)
- **FFN front in one launch** (`r9k_dsv4_sk`: ffn_norm + router gate GEMV + sqrt-softplus top-k with bias / bias_vl / padding /
  renormalize / routed scaling + the shared expert's gate_up GEMV + silu-and-mul with clamp; 5 launches -> 1): 39.7 -> 21.0 us at
  1 row, 38.6 -> 22.2 at 2, 39.3 -> 25.2 at 4, 35.8 -> 27.1 at 6, 40.2 -> 32.3 at 8; stock above 8 rows (the first version lost
  217-560 us a layer at 24 rows -- the union tree's four-stream regression -- hence R9K_DSV41_ROUTE_MAX_M 8).
- **DSpark markov head** (one launch a draft step + one final; greedy): five steps 720 -> 245 us at 1 request, 747 -> 313 at 4
  (an fp8 head copy, R9K_DSV41_MARKOV_FP8=1, is 158 / 394: off by default). Token-identical drafts.
- **Attention input on the 8 compressor / indexer layers** (attn_norm + compressor kv_score + indexer weights_proj): 29.1 -> 13.4 us
  at 1 row, 28.3 -> 18.9 at 4, stock from 8. **wq_b + indexer wq_b stacked** on D's GEMV (untimed until D lands). A norm + cast +
  rotation pre-pass (`dsv4_glue.prepass`) for D's rotated input.
- Projected a single-stream step: ~214 fewer launches of 1,754 and 1.0-1.3 ms of GPU time (FFN front 0.4-0.7, markov 0.45,
  attention input 0.1). 116 checks pass: norms within 1 bf16 ulp of stock's rms_norm, top-k bit-exact, graphs = eager.
  Committed as 828115a on v041 (B's clean-tree versions of build.sh and deepseek_v41_rdna4.py, without D's hunks).
- **Real server, v0.4.0 + C + B on a clean tree** (git archive 828115a -> VM100 ~/r9700-exl3-CB; 16:02-16:24 UTC, `~/ds41/exl3/B4/clean-CB.log`;
  v0.4.0 + C / v0.4.0 in parentheses): adaptive off -- prose x1 **83.2-84.2** (80.2 / 74.6), prose x4 35.2-38.9 per stream (39.1 /
  38.3), code x1 **178.3-178.9** (174.4 / 158.1), code x4 **87.5-88.7** (85.7 / 84.3; 327-331 aggregate); adaptive on -- prose x1
  **89.9-92.4** (75.9-83.0 / 72.5-73.0), prose x4 42.8-50.2 (43.5-51.5 / 44.3-45.6), code x1 **179.3** (177 / 152.2), code x4 84.0-85.3
  (86.6 / 81). Divergence 23 / 24 / 23 (|dlogprob| 0.084 / 0.031 / 0.048), GSM8K 98 / 100. B adds +4-11 % single stream on top of C;
  four streams unchanged (its fusions fall back to stock above 8 rows). Running total vs v0.4.0: single-stream prose **+24-27 %**
  (73 -> 90-92), code **+18 %** (152 -> 179); the Sparks' 80.7 / 118 are now beaten by 1.1x / 1.5x single stream.

### Phase 4 results, package A (expert tier to 48 tokens, split-path collective; card 5 / eight cards; 2026-11-03 17:35 UTC)
- **r9k expert tier** (card 5, graph replay, us a layer, K = 3, 48 local experts, cold weights, SCLK offset 0; T / local picks /
  distinct experts): 6/4/4 **123 -> 118-124**, 6/5/5 146 -> 141-152, 24/18/15 **518 -> 414-435**, 36/27/21 795 -> 616-620, 48/36/25
  **2324 (exllamav3's fused tier) -> 674-741** (449-492 GB/s on trellis bytes; gate+up 586 us at 377 GB/s, down 211 at 525),
  48/36/12 676, 48/36/36 990. K = 2: 24 tok 548 -> 369, 48 tok 2309 -> 717. Drafter (K = 4, 16 local, top-3): T = 20 223 -> 213,
  T = 40 407 -> 342. Slot cap 512, grouping compacts local slots first (O(nl^2)), batched input rotation, 2-row groups above 8
  tokens, `r9k_moe_exl3_groups` validation export; R9K_EXL3_MOE_R9K_M default 48, scratch 47.5 MiB a rank. Bit-exact decode,
  graphs = eager, rms rel err <= 4.2e-4 to T = 48. Target T = 48 <= 700 narrowly missed (steady state 735-741).
- **NSEQ 8 launches** with the tier at 48: the v0.4.0 fault (HSA aperture violation in the mixed warm-up) is in the > 24-token path
  through exllamav3's fused tier in-server (R9K_EXL3_MOE_R9K_M=24 on A's tree faults the same way; the fused tier is clean on one
  card at 25-64 tokens; AMD_SERIALIZE_KERNEL names no kernel). Known limit: steps above 48 tokens (NSEQ > 8 at k = 5) stay off.
- **Split-path collective** (eight cards, quiet box, us per all-reduce; before -> after, RCCL): 100 KiB 49.8 -> 45.4 (101), 160 65.1
  -> 60.3 (112), **240 81.7 -> 77.2** (124), 320 99.8 -> 90.9 (135), **480 140.2 -> 121.8** (161), 640 158 -> 156 (182). The stage
  trace: the two host-memory read roles (reduce, gather) take 31 + 23 us of the 85 us span at 240 KiB (~180 KB of host reads a
  rank at ~3 GB/s, request-limited); more lanes or sub-steps lose (15 lanes 134 us; 30 lanes at 480 KiB 252). Rule: one lane with
  1024-word sub-steps (R9K_AR8_FUSED_W1). Targets (<= 50 / <= 90) missed; the remaining gain needs a data path with fewer host
  reads (v0.4.2). Rank skew inside a pair < 2 us; the ledger's 161 vs 80 us at 240 KB is arrival skew from the preceding compute.
- **Launcher finding (unverified by the owner yet):** exporting ANY extra `-e` variable through DOCKER_ARGS into the container left
  the DSpark drafter at MAL 1.01-1.04 (vs 1.7-1.8) in 5 of 5 launches, including a dummy `R9K_A4_DUMMY=48`; the bad runs JIT a
  different rejection-sampler path. A's launcher export of R9K_EXL3_MOE_R9K_M was the union tree's four-stream regression. Being
  reproduced on the clean C+B+A tree (clean-CBA.sh: `-e FOO_DUMMY=1` and `-e R9K_A4_DUMMY=48`).
- Committed as 3f2167a on v041 (A's files + the build.sh export; the trace / lane-cap hunks were in b9c07e8).

### Phase 4 results, package D (EXL3 decode GEMV M = 1..48, wo_a on EXL3; card 7 at -500 MHz, graph replay, cold ring; 2026-11-03)
- **`kernels/r9k_exl3_gemv.hip`** (P1's trellis device functions, MIT): one launch takes up to 16 (matrix, rows, columns) groups, grid =
  128-column tiles x input chunks; each block rotates its input chunk into LDS (x * suh, Hadamard; a pre-rotated f16 input is only
  copied, for B's pre-pass), streams the trellis to f32 partials, and the last block of a tile sums them in a fixed order, applies the
  output Hadamard + svh and stores bf16 / f16 / f32 at the group's column offset. M = 1..8: the fdot row tile; M = 9..48: WMMA
  16x16x16 f16 (gfx12 layout) with a per-wave LDS transpose, one decode feeding three 16-row tiles, 2 x 128 columns a block.
  Deterministic, graph replay = eager bitwise. Bit-exact decode vs reconstruct; rms rel 0.6-2.1e-6 vs fp32 on the same rotated
  input at every M; 2.1-2.6e-4 with the in-kernel rotation (exllamav3's own path: 3.0-3.8e-4).
- **us a call, exllamav3 path (with its casts; the dequant path above 16 rows) -> ours:** fused_wqa_wkv 26.3 -> 18.9 at M=1 (304 GB/s),
  35.3 -> 23.8 at 6, **119.5 -> 44.3 at 24**, 121 -> 66 at 48; wq_b / indexer wq_b 19.9 -> 11.4 / 22.9 -> 14.7 / 51.9 -> 26.2 /
  58.1 -> 36.9; wo_b the same; drafter K6 similar; engram wkv (25 tiles, odd) 31.8 -> 24.0 at M <= 16, dequant kept above;
  mtp main_proj (59 MB) 115.8 -> 108.7 at 1, exllamav3 kept at 3-8, 660 -> 289 at 24. **wo_a bf16 einsum -> EXL3: 18.0 -> 12.1 at 1,
  17.7 -> 14.9 at 6, but 18.1 -> 27.8 at 24 and 18.4 -> 37.3 at 48** (-230 MiB a rank; R9K_DSV41_WOA=bf16 restores v0.4.0).
  Synthetic layer (wqa_wkv + wq_b + wo_a + wo_b): 86 -> 54 us at M=1, 104 -> 66 at 6, **243 -> 122 at 24**, 254 -> 178 at 48;
  launches 13-24 -> 4. Projection a step: -1.5 ms at M=1, -1.7 at 6, **-5.8 at 24**, -3.7 at 48; ~390 fewer launches at M=6.
  The 450 GB/s target at M <= 8 is not met (285-340; a ~9 us fixed latency chain a launch: prefetching the first tiles before the
  rotation is the next step). Open: NB2=4 crashes on main_proj (default 2).
- Committed as the package D commit on v041 (D's files; build.sh now carries A, B, C and D's exports; C's combine hook wired in the
  EXL3 model classes in the same commit).
- **Package A on the clean tree (git archive 3f2167a = C + B + A; 17:33-18:35 UTC, `~/ds41/exl3/A4/clean-CBA.log`, `bisect-A.log`):
  the DSpark drafter accepts nothing -- MAL 1.01, acceptance 0.002 at every width, NSEQ 4 and 8, with and without extra container
  variables** (prose x1 50-54 tok/s, code 49-55; the target is still right: divergence 22 / 24 / 24, GSM8K 99 / 100). The "any
  extra -e breaks the drafter" finding was this. Bisect by knob: `R9K_EXL3_MOE_DECODE=mgemv` (A's kernel bypassed) MAL 1.80 / 4.43,
  prose 87.3 / code 196; `R9K_EXL3_MOE_R9K_M=24` (A's kernel, old cap) MAL 1.71 / 4.49 but prose **62.8** / code 182.6. So A's
  rewritten kernel (a) corrupts the drafter when the tier is sized for 48 rows (the warm-up at 36 / 48 rows or the 2-row
  group path leaves the drafter's small calls wrong -- the single-card tests at R9K_TEST_DRAFT_EP=32 did not see it), and (b) costs
  ~+6 ms a step at the 1-4-row widths the single-stream step actually runs (not in A's table, which starts at T = 6). Held back:
  commit "Hold back package A's 48-token expert tier" returns the kernel, tier wiring and tests to v0.4.0; the split-path lane
  rule (+4-13 % at 240-480 KiB), spectune STREAMS and the NSEQ-8 caveat stay. v0.4.2 item: redo the 48-token tier with the
  drafter geometry (16 local, top-3, K = 4) and T = 1-4 in the acceptance, on a clean tree, server-validated.
- **Plain decode (no drafter), release-candidate tree** (by accident: the hold-back edit had cut the launcher's speculative-config
  line; `~/ds41/exl3/D4/clean-REL.log`, 18:39-19:10 UTC): **83.2-83.5 tok/s single stream** (v0.4.0's no-drafter probe: 51.7),
  52.9-60.4 per stream at four (v0.4.0: 31.5), prose = code as expected without speculation; divergence 22 / 24 / 22, GSM8K
  99 / 100. wo_a on EXL3 vs bf16 at these widths: 83.3 vs 80.5 single, 59.8-60.4 vs 58.0-58.3 at four -> EXL3 stays the default.
  The launcher is restored (commit after a9ec6a9) and the drafter runs are repeated as REL2.

### Where the time goes on v0.4.1 (2026-11-04 00:03-00:37 UTC; notes/exl3-ledger.md section Z2; rank 0, GPU time a step)
Single-stream default step **19.11 ms** (v0.4.0 25.42) = 105.3 tok/s on the ledger workload at 2.01 accepted tokens; four streams
**32.26 ms** (40.0) = 247.7 aggregate; true 6-row verify 21.74 ms (97 tok/s), true 24-row 43.48 (186 aggregate). Still GPU-bound:
kernels 17.75 / 31.12 ms, gaps <= 1.4 / 1.1 ms; launches 1,865 -> 1,132 (T = 6), 2,082 -> 1,318 (T = 24). What moved: the all-gather
4.05 -> 0.19 ms (C), which is also why adaptive verification now wins at one stream (105 vs 97; v0.4.0 74 vs 81); GEMVs + wo_a + casts
3.3 -> 2.27 ms at T = 6 and 8.8 -> 4.63 at 24 rows (D: the launches removed; the kernels run at 248-267 GB/s, wo_a 166); B's FFN front
48 launches / 1.43 ms, a plain layer 32 -> 18 launches. What did not: **the 81 all-reduces cost the same at rank 0 (5.24 ms at T = 6,
64.6 us a call) -- the one-kernel all-reduce is 23.2 us (minimum over ranks) and ~3.4 ms a step is rank 0 waiting for slower ranks,
mostly on the post-MoE call (68 us vs 25 after attention): uneven expert load across the expert-parallel ranks. At 24 rows 152.8 us
a call vs a 73 us minimum = 6.5 ms a step: the largest row, unowned (rank-skew upper bound: ~15.7 ms step, ~128 tok/s).** Two
things that should have run and did not: C's fused MoE combine ("fusion skipped ... runner shape changed" on all 43 runners: the
`cfg.tp_size == 8` check -- under --enable-expert-parallel the MoE config has tp_size 1, ep_size 8; ~0.9 ms at 1-4 rows) and B's
markov head under the default adaptive verification (stock's five GEMVs + argmax 0.67 ms + adaptive's f32 GEMM 0.62 ms run instead;
B's kernel runs only with ADAPTIVE=false; the decline is in dsv4_glue._markov_why_not). Expert tier unchanged (2.38 / 3.80 / 7.18 /
12.52 ms at 2-4 / 6 / 9-16 / 24 rows). v0.4.2 package ceilings: A2 ~-2.3 ms at 24 rows (+ the 8-stream shape); F ~-1.2 ms at T = 6,
-3.2 at 24 (14.9 MB of attention GEMVs a layer at 450 GB/s = 33 us vs 63); G -1.1 to -1.9 ms at 24 rows (kernel part only);
C2 (combine gate) ~-0.9; B2 (markov under adaptive + the f32 GEMM) ~-0.5; **new H3: expert-parallel load skew** (replicas of hot
experts / a routing-histogram placement; vLLM's EPLB as the reference) is the biggest remaining item.

### v0.4.2 results, package CB2 (2026-11-04 04:04-04:16 UTC; clean tree v0.4.1 + CB2; `~/ds41/exl3/CB2/a1/run.log`)
- **Root causes.** The fused MoE combine was skipped on all 43 runners because `exl3_combine.install` required `tp_size == 8`; under
  `--enable-expert-parallel` the pinned vLLM's MoE parallel config is tp 1 / ep 8 and still all-reduces over the eight-rank group, so
  the tail C's kernel replaces is the same. B's markov kernel declined under adaptive verification because this pack carries
  `mtp.2.confidence_head.proj.weight`: with adaptive on, stock enables the confidence head and disables the acceptance estimator;
  the "adaptive f32 GEMM" of the ledger (0.62 ms) is stock's confidence projection (cat([hidden, w1[prev]]) -> f32 [5376 -> 1] ->
  sigmoid), not the estimator.
- **Fixes.** `_eight_rank_reduce` accepts (tp 8, ep 1) or (tp 1, ep 8); skips now log the failing check. The markov hook serves the
  adaptive path with a one-launch confidence tail (`r9k_dsv4_markov_conf`: f64 dot, one f32 rounding, f32 sigmoid; 300 -> 7.8 us;
  probabilities within 1e-6 of stock's, i.e. inside stock's own summation error; `R9K_DSV41_MARKOV_CONF=stock` is bit-exact).
- **Server (defaults, adaptive on):** both hooks engaged on 40 + 3 runners, "fused into the ar8 one-kernel all-reduce (first call
  T=2)", 0 skips. Ledger workload step **18.07 ms** (v0.4.1 19.11) at one stream, **31.25** (32.26) at four. spectune prose x1
  **98.1** tok/s (MAL 1.74), x4 54-59 per stream (188-219 aggregate); code x1 **208** (4.45), x4 106-107 (393-395 aggregate).
  divergence 23 / 24 / 24, GSM8K 98 / 100. Committed on v042.

### v0.4.2 results, package H3 (expert-parallel rank skew; 2026-11-04 02:50-06:34 UTC; notes/exl3-ep-skew.md; commit bc34a56)
- **What the wait is.** Z2's per-rank traces, matched call by call: the segment before the post-MoE all-reduce skews across the
  eight ranks exactly as much as the expert tier does (to 0.01-0.02 ms); the slow-tier rank is the late rank in 93-98 % of calls.
  Tier skew 2.20 / 2.97 / 4.25 / 6.22 ms a step at 2-4 / 6 / 9-16 / 24 rows. The slow rank moves from call to call (every rank
  is hottest in 9-16 % of calls; the drafter's rank 6 is the exception at 52-60 %); the part a static placement can reach is
  0.5 ms at one stream and 1.3 at four, the rest is per-step routing noise. The tier's cost on a rank fits as distinct experts +
  rows (R2 0.99) from a routing recorder (`bench/ds41/epskew/epskew_ext.py`, a device ring buffer the HIP graphs replay).
- **Remedies costed** (`skew_place.py` on 12 prose + 12 code recorded prompts): (a) a load-time placement map from the routing
  histogram, 0 memory: model -0.62 ms held-out at one stream / -1.60 at four; a map trained on one domain does nothing (or a
  little harm) on the other. (b) replicas with a per-step least-loaded copy: -0.24 / -0.43 more at 8 copies a layer for 0.49 GiB
  a rank (the worst card has ~0.86 GiB free at KVMEM 1.7; 16 / 32 copies do not fit). (c) stock EPLB: unsupported for
  Exl3MoEMethod, load-blind replica choice, simulated worse than the static map. (d) overlapping the post-MoE all-reduce with
  the next layer: impossible, the hc seam consumes the all-reduced output.
- **Shipped: `R9K_EXL3_EP_MAP`** (`r9700_vllm/moe/exl3_placement.py`, a wrapper on `Exl3MoEMethod.create_weights` that maps
  the per-rank local experts through the JSON map at load; same experts, same weights, nothing in the forward path).
  Measured A/B on the clean v0.4.1 + H3 tree: ledger step **19.11 -> 18.52 ms** at one stream, **32.43 -> 31.41** at four;
  post-MoE all-reduce 86 -> 70 us; spectune prose x1 94.4 -> 100.5 tok/s, prose x4 214.8 -> 222, code x4 372.5 -> 391.9;
  divergence 24 / 23 / 23, GSM8K 98. Not bit-identical to the linear layout (each rank's bf16 partial sum groups the picks
  differently). On by default in `serve/ds41-exl3.sh` (`EP_MAP=`, empty restores the linear placement); the map
  `bench/ds41/epskew/map-ds41-v041-prose-code.json` was trained on prose + code at one stream and should be regenerated from
  the served mix (`run_skew.sh TAG hist:...` + `skew_place.py`, ~15 min of card time).
- **Left after the map:** ~1.9 ms a step at one stream / ~2.9 at four of per-step routing noise; shrinks only with fewer bytes a
  distinct expert (A2 / F) or replicas chosen for free in the tier prologue (A2's prologue + 0.5 GiB a rank).

### v0.4.2 results, package A2 (48-token r9k expert tier; 2026-11-03 20:xx - 11-04 12:57 UTC; notes/exl3-drafter-heisenbug.md)
- **Package A's kernel was never the cause of the drafter collapse.** On a clean v0.4.1 tree plus 3f2167a's four tier files the
  server is fine at cap 48 (MAL 1.69-1.89 / 4.2-4.4). The collapse reproduces only on the 3f2167a base (C + B + A, before D),
  and there v0.4.1's OWN tier with nothing changed but the scratch reserved for 48 rows breaks the same way; a 64 MiB pad after
  the scratch hides it; eager is fine. ~16 launches (one hypothesis each) excluded the tier, every plugin scratch, counters,
  collectives, weights, the shared-experts stream, markov / mhc / route hooks: it is computed at run time inside the drafter's
  layers.0 MoE at T = 5 from a bit-identical input, in particular graph-pool layouts only. Parked as task 71; mitigation
  `bench/ds41/drafter_selftest.py` (SELFTEST=1 in the launcher; MAL from /metrics; exit 1 below 1.3 prose / 3.0 code).
- **The tier (kernels/r9k_moe_exl3.hip, exl3_decode.py):** A's kernel (slot cap 512, compacted grouping, batched rotation,
  2-row groups above 8 tokens) plus a `k_rot` input-rotation pre-pass above XR_T = 8 tokens (each local slot's gate / up
  input rotated once into an [S, 2, H] f16 scratch instead of in every k_gu block; entry point `r9k_moe_exl3_decode_xr`, the
  old one kept). Same ops in the same order. Cap R9K_EXL3_MOE_R9K_M 48 (scratch sized once). Card 5, graph replay, us a layer,
  v0.4.1 -> A2: T=1 39.9 -> 37.6; T=2 66.7 -> 64.0; T=4 101.2 -> 98.3; T=6 134.8 -> 128.5 (the 120 target not met: the
  per-group trellis stream at 4-5 groups dominates, P / KS launch geometry does not move it); T=24 532 -> 391; T=48 2,333
  (fused tier) -> 700; drafter T=5 82.4 -> 81.2, T=20 272 -> 239, T=40 1,016 -> 343.
- **Tests:** tests/test_exl3_moe_r9k.py (lazy f32 reference so EP 8 runs under 1 GiB; groups at every T 1..48; a "warm"
  routing; `check_server_sequence()` = target warm-up then the exact drafter geometry 128 / 16 local / top-3 / K=4, both
  rotation paths, graphs at 5-48): ALL OK on card 5 (`~/ds41/exl3/A2/testF.log`).
- **Server acceptance** (tree tF = `git archive v042` + the four files, launcher defaults, `~/ds41/exl3/A2/acc.out`):
  NSEQ 4: self-test prose 1.71 / code 3.46; spectune prose x1 **104-105 tok/s** (MAL 1.79-1.82), x4 234 aggregate (one rep
  137); code x1 **208-210** (4.24), x4 **411-412** aggregate; divergence 22 / 24 / 24; GSM8K 98. NSEQ 8: self-test 1.64 / 4.08;
  prose x1 102.8, x4 191.5; strict sanity 9 streams 0 bad of 45, 16 streams 0 of 80; 10-min 8-client 200-28k soak 128 ok,
  0 errors, 3,149 prompt tok/s; self-test after the soak unchanged. The 8-stream spectune raised an IndexError (more streams
  than prompts; fixed in spectune.py, to be measured at the gate). Launcher: NSEQ 8 supported, default stays 4.

### v0.4.2 results, package G (split-path eight-rank all-reduce 160 KiB - 2 MiB; 2026-11-04; `~/ds41/exl3/G/{mb,final1.log,accG.out}`)
- **Targets not met** (240 KiB 76.7 -> 67.1 us, target 50; 480 KiB 122.5 -> 116.3, target 90; 1 MiB 230 -> 202, target 150).
- **What limits the fabric** (microbenchmarks): each switch's uplink moves ~14 GB/s each way (reads and writes do not share it);
  one GPU reading host memory reaches 13.4 GB/s with >= 2 blocks of 1024 threads (the old trace's 3 GB/s was not a per-lane
  request limit); pair traffic stays inside the switch (~12 GB/s each way); ~2.7 us a flag hop GPU -> GPU through host memory.
  Copy engines lose (15-20 us latency, ~12 GB/s); push vs local root complex no difference; wider loads no gain; each rank
  owning 1/8 is already the current kernel. The old path is not link-bound (floor ~27 us at 240 KiB): the time is two dependent
  host round trips plus per-sub-step release / poll latency.
- **Built: the "hub" path** (`r9k_ar8_hub` kernel + one CPU reducer thread a rank in kernels/r9k_ar.hip; `_setup_hub` / `_hub` in
  r9700_vllm/comm/r9k_ar8.py; routing (128 KiB, 1536 KiB], agreed over the group). Pair reduce-scatter and pair all-gather stay
  peer-to-peer; the quartet sum runs on CPU threads over the pinned host pools, chunk by chunk behind per-chunk flags; 1 S up +
  1 S down a switch with one host hop instead of two; same two roundings in the same order, **bit-identical to the fused split
  kernel** (18 bf16 sizes + fp16 / fp32, 2 ulp of the reference, identical across ranks; mixed bursts and graph replay pass).
  Quiet box, graphs, best of 3, us: 160 KiB 59.9 -> 51.5, 240 KiB 76.7 -> 67.1, 480 KiB 122.5 -> 116.3, 1000 KiB 230 -> 202,
  1280 KiB 283 -> 255; GLM 16 / 48 / 160 x 6144: 65.8 / 146 / 399 -> 57.1 / 131 / 408. With the CPU sum skipped 240 KiB is still
  61.8, so the CPU costs ~6 us; the rest is the pair-sum role publishing chunks at ~6 GB/s a rank plus quartet skew.
- **Server:** DeepSeek NSEQ 4 on v042 + G: prose x1 104.8-105.1 (v042 104-105), x4 231.8 (234); code x1 218.6 at MAL 4.46
  (208-210 at 4.24: the acceptance length, not the collective), x4 422-428 (411); self-test ok; divergence 20 / 24 / 24; GSM8K 97
  (1 truncated). GLM TP8 A/B same tree, hub on / off: decode mean 107.7 / 107.1, 8 streams 174 / 172, **16 streams 220 / 228**,
  prefill 3,149 / 3,148, strict sanity 0 / 0 bad of 150, GSM8K 250 97.6 / 98.0 %.
- **Risk:** each rank runs a CPU reducer thread that spins while calls arrive (then polls every 20 us); with TPR=4 (32 spinning
  threads on 32 vCPUs) calls took ~4 ms. **Shipped opt-in (`R9K_AR8_HUB=1`), default off:** v0.4.1's fused split path is
  untouched and remains the default; tests/test_ar8.py exercises the hub by default (HUB=0 tests the fused path). A segv in
  the test (pools freed while reducer threads still read them) is fixed by holding the pools at module scope and joining the
  threads at exit. Next (v0.4.3): batch the pair-sum publishes or publish from a separate flag block; no design measured here
  reaches 50 us at 240 KiB on this fabric.

### v0.4.2 results, package F (EXL3 decode GEMV latency chain, NB2=4; card 7, graph replay, cold ring; `~/ds41/exl3/F/{full-base,full-g,num-g}.log`)
- **Trace of one launch** (wall_clock64 a block, `tests/test_exl3_gemv.py ONLY=trace`, -DR9K_GEMV_TRACE): graph-node floor 3.54 us
  for a tiny kernel; wq_b (3.28 MB) v0.4.1 -> F: block start skew 0.76 -> 0.44, x / suh load + rotation 1.24 -> 1.48 (one trellis
  k-tile in flight), stream 5.08 -> 4.04, partials store + fence 0.40 -> 0.36, arrival atomic 0.24, epilogue 2.0-2.2 -> 1.72 (fence
  0.28 + partial loads 0.76 + LDS slice-sum / Hadamard / store 0.68), kernel end ~10.2 -> ~9.5. v0.4.1's per-matrix chunking put
  fused_wqa_wkv at 360 blocks x 136 VGPRs (not all resident; blocks starting 14-19 us late). 450 GB/s does not fit: wq_b <= 7.5 us
  minus the node leaves 4 us for 3.28 MB (above peak); fused <= 13 minus the node leaves ~9.5 and the stream alone is 8.2 at ~640 GB/s.
- **Changes:** SHARED_KS planner (one chunk count a launch; fused 360 -> 280 blocks; most of the M = 1 gain); early trellis loads
  (R9K_GEMV_PI=1) behind an LDS-only barrier (deeper prefetch worse: PI=4 +1-2 us, PI=8 +4-5); split epilogue for row tiles M <= 4
  (R9K_GEMV_FSPLIT_M=4; slower at 6-8 rows, so D's code above 4 and the whole WMMA path are verbatim); scalar load bases 136 -> 85
  VGPRs. Dropped: a rolling 4-8 tile register pipeline, BLOCKS 96 / 288, QB=8.
- **NB2=4 crash:** the WMMA kernel's static transpose buffers (6 KiB a 128-column tile, 24 KiB at NB2=4) were not in the LDS budget;
  `ks_for` allowed a 49 KiB chunk -> HSA_STATUS_ERROR_INVALID_ALLOCATION (exit 139). Fix: `_lds_mma(nb2) = min(50 KiB, 63 KiB - nb2 x
  6 KiB)` and the launcher refuses with -1 above 64 KiB. main_proj us at 16 / 24 / 48 rows: NB2=1 434 / 508 / 786, NB2=2 245 / 285 /
  404, **NB2=4 180 / 221 / 298**; attention shapes NB2=2 best (NB2=4 0-7 % slower). Default 4 when k >= 8192, else 2.
- **Timing (us, v0.4.1 -> F) at M = 1 / 2 / 4 / 6 / 8 / 24 / 48:** fused_wqa_wkv 18.9 -> 15.1 / 19.2 -> 16.0 / 19.9 -> 18.1 / 24.0 ->
  22.0 / 26.5 -> 24.6 / 45.1 -> 45.0 / 65.2 -> 65.4; drafter fused K6 20.2 -> 17.0 ... 66.6 -> 67.1; wq_b 11.3 -> 10.9 ... 36.8 ->
  36.9; wo_a 12.1 -> 10.1 / 12.4 -> 10.9 / 12.7 -> 12.5 / 14.7 -> 15.4 / 16.1 -> 16.4 / 27.8 -> 28.0 / 38.0; wo_b 11.1 -> 11.1 ...;
  main_proj 106.7 -> 108.2 at 1 row, 286 -> 221 at 24, 404 -> 299 at 48; synthetic layer of 4 linears 53.5 -> 47.3 at 1 row.
  Projection a step: -0.28 ms at M = 1-2, -0.04 at 4-8, -0.07 / -0.09 at 24 / 48 (main_proj).
- **Tests:** full test_exl3_gemv.py ALL OK (decode == reconstruct bit for bit; rms gates; stacked parts; apply r9k vs exllamav3;
  wo_a hook; graph replay x2 == eager). New: ONLY=trace section, a graph-floor probe, a read-only cache flush. Knobs:
  R9K_EXL3_GEMV_SHARED_KS (1); compile-time R9K_GEMV_PI (1), _QB (4), _FSPLIT (1), _FSPLIT_M (4). Server acceptance: the v0.4.2 gate.
