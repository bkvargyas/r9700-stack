# DeepSeek-V4.1-Flash at EXL3 2.9 bpw on r9700-stack -- architecture brief (2026-11-02)

Brian: "Lets do a 3-bit / 2.9-bit EXL3 version" of DeepSeek-V4.1-Flash on OUR stack, resident on the eight R9700s. This file is the
brief the implementing agents work from; the owner (Claude Fable) reviews every piece against it.

## Why this shape

- The official release (deepseek-ai/DeepSeek-V4.1-Flash-1017) is MXFP4 experts + FP8 attention + FP8 Engram tables: 292 GiB of
  device weights with the Engram tables in host RAM -- does not fit 8 x 32 GiB. Streaming a fifth of the experts over the PLX
  uplinks is the slow path we measured (GLM four-card: 8-43 tok/s).
- The published pack **LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw** (196.3 GiB, 43 safetensors shards, MIT) fits resident with
  ~55 GiB to spare. Same pack two DGX Sparks serve at 73 tok/s single / 136 at four (TensorFold, 2026-10-18), eval 97.1.
- EXL3 is ExLlamaV3's trellis format (QTIP-style). **exllamav3 (MIT, github.com/turboderp-org/exllamav3, v1.6.6) builds on ROCm**:
  setup.py detects torch.version.hip, uses the pip ROCm SDK devel (our image has hipcc at
  /usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/bin/hipcc), links hipBLAS, ptx_portable.cuh replaces inline PTX, and
  `exllamav3/exllamav3_ext/rocm/quant/` holds RDNA (WMMA) kernels: exl3_gemm_inner_rdna, exl3_gemv_*_rdna, exl3_mgemv_rdna,
  exl3_moe_inner_rdna. It also has its own DeepSeek-V4.1 architecture (exllamav3/architecture/deepseek_v41*.py, modules/dsv4.py,
  mla_attn.py, qsa_indexer.py, ngram_embedding.py) and a tensor-parallel mode (-tp). So the kernels exist; our job is glue in the
  plugin plus whatever the attention path needs on gfx1201.
- Rules that still hold: r9700-stack runs as a plugin on STOCK vLLM (our pin: r9700/vllm:dev = nightly-rocm100-e4a5b553, vLLM
  0.32.1rc1.dev466, torch 2.13.0+rocm10.0.0) with unmodified published checkpoints. exllamav3 is a third-party library we may link
  against and read. libr4d (radiance) source is never read.

## The checkpoint, as read from the shard headers

quantization_config: {"quant_method": "exl3", "version": "1.6.2", "avg_bits": 2.9, "head_bits": 6, "codebook": "mul1",
"out_scales": "always", "mtp_bits": 4, "original_quantization_config": {fp8 / fp4 experts}}. Config: 40 layers, hidden 5120, 384
routed experts + 1 shared a layer (first_k_dense_replace 3), 6 experts a token, intermediate 2048 (the pack's expert matrices are
5120 x 2304: 2304 = 18 x 128, so the quantizer padded 2048 -> 2304 -- confirm against exllamav3's padding rule), MLA with one KV
head (head_dim 512, qk_rope 64, q_lora 1280, o_lora 1024), compress_ratios 2 on layers 2-19, kv_source_layer_ids [2,8,14,20],
index_source_layer_ids [2,8,14,20,24,28,32,36], index_topk 512, sliding_window 128, engram layers [1,14] (~197B params, the
tables come from the BASE checkpoint's shards 47/48, ~203 GB, not in the pack), num_nextn_predict_layers 3 (DSpark: block 5,
128 experts, 3 a token, target layers 37-39; the pack's `mtp.0.*` tensors at 4-6 bpw).

Per quantized linear the pack holds four tensors (exllamav3 naming, NOT Hugging Face naming):
- `<name>.trellis`  int16  [in/16, out/16, 16*K]   (K = bits: 48 -> 3, 32 -> 2 on layers 18-22, 80 -> 5, 64 -> 4, 96 -> 6)
- `<name>.suh`      f16    [in]    input-side sign/scale vector (applied with the Hadamard rotation of the input)
- `<name>.svh`      f16    [out]   output-side vector
- `<name>.mul1`     int32  scalar  codebook marker (mul1 = codebook 2; `mcg` would be codebook 1)
Examples: `layers.N.ffn.experts.E.w1|w2|w3.*` (w1 = gate, w3 = up: [320,144,K16] i.e. in 5120, out 2304; w2 = down: [144,320,K16]),
`layers.N.ffn.shared_experts.w1|w2|w3.*`, `layers.N.ffn.gate.weight` f16 [384,5120] + `gate.bias` f16 + `gate.bias_vl` f32,
`layers.N.ffn_norm.weight` bf16, `embed.weight` bf16 [129280,5120], `mtp.0.ffn.experts.E.*` (128 experts). The attention tensors
(`layers.N.attn.*`, 5 bpw; indexer `wk` 8 bpw) and the head (6 bpw) are in shards 1 and 43: read their headers before coding.

## Design

1. **Quant method `exl3`** in `r9700_vllm/quant/exl3.py`, registered like our `R9kCompressedTensorsConfig` (see
   r9700_vllm/__init__.py and r9700_vllm/quant/ for the registration mechanism vLLM uses in this pin). `Exl3Config.from_config`
   reads the fields above. `get_quant_method`: LinearBase -> `Exl3LinearMethod`; FusedMoE -> `Exl3MoEMethod`; embeddings stay bf16;
   the LM head gets the linear method with K = head_bits.
2. **Linear forward** = exllamav3's `ext.exl3_gemm(x_f16, trellis, suh, svh, out, K, mcg, mul1, ...)` -- take the exact call from
   exllamav3/modules/linear.py (the forward around lines 580-600 and 1010-1020) and its temp-buffer handling; vLLM hands bf16
   activations: cast to f16 at the boundary (or check whether the kernel has a bf16 path) and cast the output back.
   Tensor parallel: the output dimension splits by 16-column tiles (trellis dim 1, svh); the INPUT dimension carries the Hadamard
   rotation in blocks (find the block size in exllamav3 -- 128 is the usual) so a row-parallel split must be a multiple of that
   block (5120/8 = 640 ok; check every row-parallel input dim of this model at TP8). Where a dim does not divide, replicate that
   layer across ranks instead of sharding it.
3. **MoE** = `ext.exl3_moe(...)` exactly as exllamav3/modules/block_sparse_mlp.py calls it (lines ~1300-1345): per-expert pointer
   tables (trellis/suh/svh device-pointer tensors for gate/up/down), K per matrix, mcg/mul1 flags, the routing in exllamav3's sorted
   layout (expert_count, token_sorted, weight_sorted) built from vLLM's topk_ids / topk_weights, the temp buffers it sizes. Run the
   experts **expert-parallel** (vLLM `--enable-expert-parallel`): each rank holds whole experts (384 / 8 = 48 a layer), so no expert
   matrix is ever split and the Hadamard blocks stay whole; FusedMoE's EP path gives each rank its local experts and reduces the
   outputs. The shared expert (one a layer, 4-5 bpw) is replicated or sharded by output tiles -- replicate first. Per-rank memory at
   TP8+EP: experts 48 x 40 x 25.2M params x 3 bits ~ 18 GiB, shared ~0.6 GiB, attention shard ~0.6 GiB, embed/head ~1.8 GiB ->
   ~21 GiB + KV + activations, inside 32 GiB.
4. **Loader**: translate exllamav3 names to the vLLM `DeepseekV41ForCausalLM` parameter names (read vLLM's model file in the image:
   `docker run --rm --entrypoint python3 r9700/vllm:dev -c "import inspect, vllm.model_executor.models.deepseek_v41 as m; print(inspect.getsource(m))"`
   or wherever this pin keeps it; also its `load_weights` / stacked-params mapping), deliver the four tensors of each linear to the
   right parameter, handle `mtp.*` for the DSpark drafter and the Engram tables from the base checkpoint (the vLLM engram loader
   expects the release's names: point `--model` at a directory that holds the pack AND symlinks to the base shards 47/48 with a
   merged safetensors index, or load them in a second pass). `embed.weight` bf16 maps to the embedding; the head is EXL3 at 6 bits.
5. **Numerics**: exllamav3 ships `reconstruct` (dequant to f16). Tests: for a dense layer, `x @ reconstruct(W).T` vs
   `exl3_gemm(x, W)` within f16 tolerance; for the MoE, a one-token routing to a known expert vs the dequantized SiLU-MLP. Then a
   layer-0 forward of the model on one rank against a CPU/dequantized reference.
6. **Attention on gfx1201**: whatever the stock DeepseekV41 model does on ROCm; the probe (`~/ds41/probe-stock.sh` on VM100) shows
   the backend it picks and where it fails; fixes go in our platform plugin the way the GLM MLA prefill fix did
   (r9700_vllm/platform.py `get_mla_prefill_backend_cls`). Engram offload (`--engram-config {"cpu_offload":true}`) and DSpark
   (`--speculative-config {"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":false}` -- AMD must disable
   adaptive verification) are stock features to verify, not to write.

## Milestones

- M1: exllamav3 wheel for gfx1201 built in the image; its kernel tests pass on an R9700; exl3_gemm / exl3_moe timed at decode shapes.
- M2: `exl3` quant method loads the pack (TP8 + EP) with the four-tensor mapping; dense and MoE numerics match reconstruct.
- M3: end-to-end forward (whatever attention stock gives), Engram in host RAM, first tokens; then DSpark; probe numbers.
- M4: gfx1201 attention path if stock's is slow or absent; sanity, GSM8K, HumanEval, soak; release.

## Where things are (VM100 = devops@192.168.0.123, the eight-card box; the mgmt VM holds the repo at ~/r9700-stack)

- Pack: ~/ds41/exl3/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw (downloading; marker EXL3_DOWNLOAD_DONE in ~/ds41/exl3/download.log).
- Base checkpoint (Engram shards 47/48): ~/ds41/hf/DeepSeek-V4.1-Flash-1017 (complete, 476 GiB).
- exllamav3 source: ~/ds41/exl3/src/exllamav3 (v1.6.6); build script ~/ds41/exl3/build-exl3.sh -> ~/ds41/exl3/build.log, wheel dir
  ~/ds41/exl3/dist. On the mgmt VM a read-only clone sits in the job tmp (exl3/exllamav3).
- Serving image r9700/vllm:dev (torch 2.13.0+rocm10.0.0, hip 7.15, vLLM 0.32.1rc1.dev466); serve script serve/serve.sh (knobs in
  its header; EXTRA= appends engine args; DOCKER_ARGS= extra docker args; MODEL=/models/<dir> with ~/models mounted -- bind-mount
  anything outside ~/models through DOCKER_ARGS). Our test tools: ~/probe.py, bench/ in the repo.
- GPU etiquette on the shared box: the probe owns cards 0-3 (TP4 launches); kernel tests use card 7 (HIP_VISIBLE_DEVICES=7);
  never two servers on the same cards; the eight-card clock offset is `bash ~/gpu-offset.sh -500` (0 for four-card Qwen work).
