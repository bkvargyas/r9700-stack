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
