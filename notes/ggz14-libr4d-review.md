# Review: libr4d + GGZ14/vllm-mxfp4 ("radiance") for a stock-vLLM-0.29 gfx1201 kernel plugin

Date 2026-09-18. Read-only review of `~/src/libr4d` (HEAD `5dc6302`, 2026-08-26) and `~/src/vllm-mxfp4`
(HEAD `a2fdd4b`, VERSION file `0.13.0`, README says 0.12.0), checked against `~/src/vllm-0.29.0`
(tag v0.29.0, `98dff2a`). Paths below are relative to those roots unless absolute.
Target: Qwen3.8-Flash-Next (GDN + QSA full-attn, 512x top-10 MoE, moe_inter 640, hidden 2560, PLE, MTP;
CT-MXFP4/32 experts + FP8/GPTQ rest) and Qwen3.8-27B on 2x R9700, stock vLLM 0.29 + ROCm 10.

---------------------------------------------------------------------------------------------------
## 0. Headline findings

* **Licensing: neither repo has a LICENSE file** (checked: no `LICENSE*`/`COPYING*`, and no licence text in
  README/sources). By default that means all rights reserved. libr4d authors: StillDeadcode (+ PRs from
  ggz14 = GGZ14/bkuyper and tcclaviger). vllm-mxfp4 authors: GGZ14 (commits as "Brian"/ggz14), Daniel
  Cherubini, StillDeadcode. vllm-mxfp4 also *contains* third-party code under other licences:
  `escha/EXLLAMAV3-LICENSE.txt` (MIT, Turboderp; escha kernels derive from exllamav3), `dflash2/*.py`
  (backport of vLLM PR #52816, Apache-2.0), and the patch_* anchors are Apache-2.0 vLLM text.
  The user OK'd borrowing from GGZ14; **libr4d is StillDeadcode's repo**, so ask for a licence
  (Apache-2.0/MIT) before redistributing libr4d-derived code outside our own boxes.
* libr4d HEAD reports `R4D_VERSION "0.5.0"` (`r4d.h:40`) but tag `v0.5.0` = `e8de4bc` does **not** contain
  the N-rank all-reduce (`9e0728e`) or `gemm_mxfp4a8_nt_m64` (`8021e32`). Pin by commit, not by version.
  HEAD **does** contain the GDN exp-overflow NaN fix (PR #1, `2739650`/`b9e42ab`; `r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip:489-490`).
* radiance's production image `stilldeadcode/vllm-radiance:0.9.3` = **vLLM 0.27.1, torch 2.11.0, triton 3.6.0,
  aiter 0.1.17, transformers 5.14.1, ROCm 7.14, py3.12** (`Dockerfile:10-11`, `Dockerfile.ggz14.top:45-47`,
  `DOCKERHUB.md` stack table). Their `Dockerfile:43-47` already pins `VLLM_VERSION=0.29.0` (upgrade in
  progress; torch kept at 2.11 because torch 2.12 fails to build against ROCm 7.14 RCCL symmem, `Dockerfile:22-42`).
  Patches are written to "carry both shapes" (`_patchlib.apply_any`, `_patchlib.py:31-56`).
* **I ran the whole radiance patch chain against v0.29.0** (tolerant `_patchlib` shim, image order
  `Dockerfile:278-283` then `serve-mxfp4.sh:989-1011`): every vLLM-targeting patch applies except
  `patch_gdn_lazy` (reverted feature, 8/16 hunks miss). `patch_dflash2`, `patch_dflash_base`,
  `patch_radiance_fusion` are NOOP (upstream 0.29 already carries them). Patches targeting
  torch/aiter/triton/transformers could not be checked here (trees absent). All 26 monkeypatch
  watchlist symbols (`audit_upgrade.py:44-71`) exist in 0.29.0. => radiance is a near-drop-in reference for 0.29.
* **Nothing in libr4d or radiance_*.hip uses device printf/assert (no hostcall)**. Only device-side abort
  is `__builtin_trap()` in the wide AR handshake (`r4d_ar_wide.h:92`, s_trap, not hostcall). Verify
  after build with `~/r9700-stack/p2p/scanhc.sh`.
* Flash-Next specifics: GDN uses `QwenGatedDeltaNetAttention` with `gqa_interleaved_layout=False`
  (`vllm/models/qwen4_exp/amd/model.py:211-216`), so the ROCm path falls through to `_forward_core`
  (`qwen_gdn_linear_attn.py:1195-1257`) — exactly where radiance's R4D GDN hook sits. Full-attention
  layers are **all QSA** when `indexer_n_heads` is set (`amd/model.py:218-233`); stock QSA needs an
  FA varlen func (aiter triton on ROCm) and a **bf16** main KV cache (`amd/qsa.py:105-116,151-152`).
  R4D's dense paged attention therefore does not drop in for Flash-Next; see §1.5.

---------------------------------------------------------------------------------------------------
## 1. libr4d

### 1.1 Architecture
* **C ABI, raw pointers + stream, no torch coupling** (`r4d.h:1-5`). Rationale: "wheel-ROCm ships no
  rocThrust, which torch/extension.h needs" (`r4d_ar_oneshot_2rank_exact.hip:6-7`) — relevant to TheRock wheels too.
  Attention/GDN entry points return negative codes on shape mismatch; GEMM/AR throw `std::runtime_error`
  (`r4d.h:30-32`). No launch path allocates or syncs => HIP-graph capturable (`r4d.h:3-5`).
* Naming = geometry; entry point rejects mismatches (`README.md:7-25`).
* **Registry** `r4d_registry.hip`: table of `R4DKernelInfo{name,family,op,computes,shape,dtypes,constraints[]}`
  (`r4d.h:305-333`), constraints are predicates EQ/LE/GE/DIV/IN on named params (`r4d_registry.hip:23-27`).
  Rows in preference order per op; `select(op, **geometry)` returns the first match or None, `explain()`
  gives the failing constraint, `selections()` logs every question (`r4d_module.hip:503,545,694-699`;
  README `:44-95`). Predicates are *necessary only*; entry points still validate strides/sizes.
* **Python binding** `r4d_module.hip` (pybind11, `PYBIND11_MODULE(r4d)` at `:581`): `m.def` per entry point
  (names minus `r4d_`), module constants `ATTN_*`, `GDN_*`, `AR_*`, `GEMM*_MAX_M/GROUP` (`:587-683`),
  `kernels/ops/select/explain/constraints/selections`. Only file that knows Python.
* **Build** `build.sh`: needs only `hipcc` + `python -m pybind11 --includes`; must use the same ROCm as the
  importing process (`build.sh:4-6`). Flags `-O3 -std=c++17 -fPIC --offload-arch=${GFX_ARCH:-gfx1201}
  -Wno-unused-result -ffp-contract=off` (`:35`); per-TU extras: `-mcumode` for the GDN chunk scan (60 KB LDS,
  tuned 1 WG/CU, `:21-24,41`), `-DR4D_GEMM_W4A8_GROUP=128` (`:54`). TUs compiled separately and linked
  with `hipcc -shared` (no `-fgpu-rdc`, `:75-82`). `-ffp-contract=off` is **load-bearing** for the lossy
  AR (ranks would diverge 1 ULP), otherwise reproducibility (`:12-19`). No `/opt/rocm` or ROCM_PATH
  references anywhere => **should build under ROCm 10 / TheRock** as long as `hipcc` (TheRock: under
  `rocm-sdk path --root`/bin) and HIP headers are reachable; untested here (no hipcc on this VM).
  Risk surface = clang builtins used: `wmma_f32_16x16x16_{f16,bf16,fp8_fp8}_w32_gfx12`,
  `wmma_i32_16x16x16_iu8_w32_gfx12`, `global_load_tr_b128_v8i16`, `permlanex16`, `sched_group_barrier`(21x),
  `cvt_pk_f32_fp8`, `cvt_pkrtz`, `fdot2(_f32_bf16)`, `perm`, `fence`, `s_sleep`, `s_setprio`, plus inline
  `s_wait_storecnt 0x0`. Compile-test first under ROCm 10's clang.
* HIP runtime APIs used: `hipExtMallocWithFlags(hipDeviceMallocFinegrained)`, `hipIpcGetMemHandle`,
  `hipIpcOpenMemHandle(..., hipIpcMemLazyEnablePeerAccess)`, `hipMemGetAddressRange`,
  `hipDeviceEnablePeerAccess` (test only).

### 1.2 Kernel catalogue (shape envelope / formats)
| entry point | envelope (rejects otherwise) | formats |
|---|---|---|
| `attn_prefill_h256_gqa6_{fp8,bf16}kv` | head 256, q_heads/kv_heads==6, paged block 16, causal, varlen (`r4d_attn_paged_h256_gqa6.hip:36-40,52-62`) | bf16 Q; KV `(blocks, kv_heads, 16, 2*256)` K|V packed per slot (`r4d.h:42-58`); fp8 KV with per-(seq,kv_head) descales; bf16 out |
| `attn_decode_h256_gqa6_{fp8,bf16}kv` | same + `q_len*gqa <= 64` (q_len<=10), split-KV + combine, caller scratch `..._scratch_bytes` (`:94-150`) | f16 partials |
| `attn_vit_h72_bf16` | head 72, MHA, non-causal, varlen cu_seqlens | bf16 |
| `gdn_conv_prep_w4_h128_bf16` | conv width 4, K=V=128, chunk 64 | bf16 x/conv-state, fp32 g/beta out (`r4d.h:128-134`) |
| `gdn_kkt_solve_k128_c64_bf16` | K=128, chunk 64, `H % Hg == 0` | bf16 k, fp32 beta/g, bf16 A |
| `gdn_chunk_scan_k128_v128_c64_bf16` | K=V=128, chunk 64; state in WMMA accumulators; built `-mcumode` | bf16 q/k/v/A, fp32 g/beta/h0/ht (`r4d.h:92-106`) |
| `gdn_conv_update_w4_h128_bf16` | decode conv, rolling spec window (state width-1+num_spec) | bf16 |
| `gdn_recurrent_update_k128_v128_bf16_fp32state` | K=V=128, `H%Hg==0`, paged fp32 state, one state per candidate token | optional fused gated-rmsnorm (`r4d.h:158-164`) |
| `gdn_gated_rmsnorm_h128_bf16` | 128 channels/row | bf16 x/z/out, fp32 w |
| `ar_oneshot_2rank_{exact,wht6}` | ws==2; wht6: numel%64, bf16/fp16 | exact fp32 accumulate; wht6 = WHT-rotated 6-bit + bf16 scale per 64 |
| `ar_{oneshot,twoshot}_{4,8}rank_exact`, `ar_twoshot_4rank_ti8` | ws 4/8; twoshot numel%64 and <= 20,971,520 elems (`r4d.h:222`); ti8 numel%128 | — |
| `gemm_bf16_nt_m16` | M<=16, K%(SK*256) | bf16 |
| `gemm_bf16_nt_m64` | M<=64, K%(SK*16), 16x16x16 WMMA split-K in LDS | bf16 |
| `gemm_w4a16_nt_m64` | M<=64, N%16, K%(SK*128) | f16 A; 4-bit asym, per-128 `Wsz` dword = f16 scale | f16(-(1024+zero)); weight pre-permuted to fragment order (`r4d.h:270-279`) |
| `gemm_w4a8_nt_m64` (+`quant_act_i8`) | same | int8 A (per-row f32 scale, byte-shuffled by `quant_act_i8`), signed int4 + f16 scale/128, iu8 WMMA |
| `gemm_mxfp4a8_nt_m64` | M<=64, N%16, K%(SK*32), WV*SK*32<=1024, LDS WV*NPW*SK*1KB<=64KB, MB 1..4, NPW∈{1,2,4,8} (`r4d_gemm_mxfp4a8_nt_m64.hip:235-250`) | see 1.3 |
| `dflash_conv_t2_g16_bf16` | DFlash2 drafter only | — |

### 1.3 `gemm_mxfp4a8_nt_m64` in detail (base for our MoE GEMM)
* Computes `C[M,N](bf16) = A_fp8[M,K]·As[M] @ dequant(W_mxfp4)[N,K]^T`. fp8 WMMA chosen because on gfx1201
  `v_wmma_f32_16x16x16_fp8_fp8` = 412 TF/s vs f16 207 vs iu8 407 (`:12-14`).
* **Scale fold** (`:16-31,105-128`): offline per-row `Wref[n] = max_k E8M0[n][blk]`; in-loop
  `d = Wref[n] - E8M0[blk][n]` clamped to [0,15] (`:190-191`) indexes a 16x8-byte table `kMag[d]` of the
  8 e2m1 magnitudes pre-scaled by 2^-d as **e4m3 bytes, including e4m3 subnormals** (exact for d<=8,
  rounding for 9..12, zero for >=13). Sign is bit3 -> bit7. Unpack = 4x `v_perm_b32` per 8 weights, no
  multiply. Epilogue applies `2^(Wref-127) * As[m]` (`:223-224`). Sound only because **gfx12 fp8 WMMA
  honours e4m3 subnormals** (verified on HW, `:27-30`). On the 27B checkpoint, stopping at normals zeroed
  6.7% of weights on 36 channels (18.7% channel error) — for our experts, histogram `d` per tensor before trusting it.
* **Weight layout (fragment order)** `mxfp4_layout.py:11-24`: `Wp[((n/16)*(K/16) + k/16)*32 + lane]`,
  one uint32 (8 e2m1) per lane; lane l = row `16*nt + (l&15)`, k offset `16*ks + 8*(l>>4)`; i.e. checkpoint
  `[N/16][16][K/16][8 bytes]` -> `[N/16][K/16][half][row][4 bytes]`. Scales stay **`[K/32][N]` uint8**
  (transpose of CT/Quark `[N, K/32]`), `Wref[N]` uint8. Measured 1.44x vs reading checkpoint order,
  1.15x (M=8)/1.3x (M=64) vs the LDS-staged radiance decode kernel it was ported from (`:62-79`).
* Tiling: wave owns NPW 16-col tiles x MT(=MB) 16-row tiles; A read straight from global (8 contiguous
  bytes/lane; rows past M **clamped**, not masked, `:86-89,177-181`); W column tiles clamped to last tile
  (tail blocks would otherwise fault, `:151-165`); split-K across waves of one WG reduced in LDS
  (`:199-226`) => no HBM partials, no counters, graph-safe with no caller scratch.
* **Extending to a grouped MoE GEMM over expert slots** (tcclaviger's private `gemm_moe_mxfp4a8_nt_b16_m64{,_block,_group,_mt1}`
  names imply exactly this):
  1. Inputs: `A_fp8[T,K]`,`As[T]` quantized **once per token** (per-row scale is token-level, so sharing
     across the token's 10 experts is exact); `sorted_token_ids`/`expert_ids`/`num_tokens_post_padded`
     from vLLM's `moe_align_block_size` with block 16; `W[E_local]` fragment-order blobs (stride N*K/2),
     `Ws[E_local][K/32][N]`, `Wref[E_local][N]`.
  2. grid.y = 16-row expert blocks (MT=1; decode blocks hold ~1 row), grid.x = N-tile groups; per block:
     `e = expert_ids[by]`, uniform early-exit on `e<0` (EP-remote) or `by*16 >= num_tokens_post_padded`
     (grid sized for the max so it is capturable); A row = `sorted_token_ids[by*16+i]` (clamp pad
     sentinel to a valid row, drop on store); W/Ws/Wref base += e * stride.
  3. Epilogues: gate_up -> write `[T*topk, 2*I]` then a fused silu·mul + per-row e4m3 quant (row spans
     all N tiles, so a second tiny kernel or a two-pass amax; radiance's `launch_silu_mul_quant`,
     `radiance_mxfp4_fp8.hip:1435-1439`, is the reference numerics); down -> scale by routing weight and
     reduce over top-k (fp32 atomics into a scratch or `[T*topk,H]` + reduce kernel).
  4. Flash-Next shapes: gate_up N=1280 K=2560, down N=2560 K=640 per expert. With **TP-sharded experts at
     TP2**, down K=320: fine for this kernel (K%16 steps; SK∈{1,2,5,10} satisfy K%(SK*32)) but
     **K%128 != 0 breaks radiance's BK=128 decode kernel** (see §4). With EP2, K stays 640 (640%128=0).
  5. Split-K matters less than in dense decode: 10*M active experts x N/16 tiles already gives
     hundreds of waves (M=1: 10 x 80 = 800 waves for gate_up at N=1280).

### 1.4 All-reduce kernels
* 2-rank one-shot **push** (`r4d_ar_oneshot_2rank_exact.hip:9-24`): each rank writes its input into the
  peer's IPC scratch, sets peer flag per block, spins on own flag, reduces local+peer. Only scratch+flags are
  IPC-shared (input/output local => no graph buffer registration). Multi-block, per-block flags, no grid
  barrier. **Device-resident per-block seq counters** (graph replay safe) + scratch double-buffered by
  `seq&1`. Requires identical SPMD call sequences on both ranks (including grid size).
* **P2P requirements**: scratch/flags from `hipExtMallocWithFlags(..., hipDeviceMallocFinegrained)` +
  `hipIpcGetMemHandle`, peers open with `hipIpcOpenMemHandle(hipIpcMemLazyEnablePeerAccess)` (`:129-147`),
  i.e. working PCIe P2P + large BAR between the two GPUs (our .100 VM already serves TP2 with P2P).
  drain modes (`:92-102`): 1 = all-thread `__threadfence_system`, 3 = `s_wait_storecnt 0` (correct only for
  fine-grained memory; radiance uses drain=3/acq=0, `radiance_allreduce.py:84-85`).
* Hazard: the 2-rank spin **silently breaks** after 4e9 iterations and reduces stale data
  (`:108`, same in `r4d_ar_oneshot_2rank_wht6.h:180`); the wide family traps instead (`r4d_ar_wide.h:92`).
  Wide family validates scratch extents with `hipMemGetAddressRange` (`r4d_ar_wide.h:100+`).
* wht6: WHT-64 rotation + 6-bit symbols + bf16 scale/group; cost ~1/3 wire, 1/3 local DRAM, 1/3 skew
  (`r4d_radiance_extras.patch` comment at wht6 hunk). 80 MiB prefill AR 1.32-1.47 ms vs RCCL 3.145 ms.
* No hostcall, no device asserts. 3-rank kernel exists only in radiance extras (§2.4).
* For Flash-Next (hidden 2560) messages are half the 27B's; decode AR sizes are latency-bound, so the
  exact 2-rank kernel is the one that matters; wht6 for prefill.

### 1.5 Attention
* Compiled geometry `A_HEAD_DIM 256, A_GQA 6, A_BLOCK_SIZE 16, A_MAX_DECODE_ROWS 64`
  (`r4d_attn_paged_h256_gqa6.hip:36-40`), checked at `:54-56,113-117`. **GQA is only a template parameter**
  (`r4d_attn_prefill_h256_gqa6.hip:83,101-102,130-134`; decode `:49,100-103`); LDS depends on TILE/HEAD_DIM,
  not GQA (prefill TILE 48 because bf16 KV at 64 needs 68,096 B > 64 KiB, `paged:45-48`).
  => a `h256_gqa12` instantiation is: new wrappers + `A_GQA 12` + registry rows + `m.def`. Prefill
  BLOCK_Q becomes 384/12 = 32 positions/CTA; decode holds `q_len <= 5` at 64 rows (MTP k<=4 fits).
* Transposed score `S^T = K·Q^T` so one lane owns one query row, softmax lane-private
  (`radiance_r4d_attn.py` doc). Measured +14.6% prefill @64K, +37.8% @260K vs AITER unified;
  decode unchanged (attention ~7% of a spec-decode step) (`README.md:889`, MXFP4-NOTES `:70-71`).
* KV layout is vLLM Triton-backend layout `(blocks, kv_heads, block, 2*head)` (backend subclasses
  TritonAttentionBackend, `radiance_r4d_attn.py:212-291`); kernel block 16 via `get_supported_kernel_block_sizes`.
* **Flash-Next**: every full-attention layer is QSA (per-token top-k blocks from an MQA indexer,
  `indexer_kv_heads == 1`, `config.py:146-152`). Dense R4D kernels do not apply as-is. Path: a sparse
  variant of the decode kernel (per-token visible-block list instead of the block table; at q_len=1 all 12
  heads of a KV head share the token's selection, which fits the lane-per-row design; MTP verify rows have
  different selections per position -> per-position launch or union). QSA requires bf16 KV, which R4D has.
  Prefill tiling (BLOCK_Q positions share KV tiles) conflicts with per-token sparsity.

### 1.6 GDN kernels
* All compiled for head_k = head_v = 128, chunk 64, conv width 4; H/Hg are runtime args (`H % Hg == 0`).
  Chunked prefill = conv_prep -> kkt_solve -> chunk_scan (conv output never hits HBM; ~1/3 the bytes of FLA);
  decode = conv_update + recurrent_update (fp32 state, one state per candidate token).
* Numerics: exp-split overflow bug fixed by clamping at e^80 (`chunk_scan:489-490`); upstream note: the
  clamp bounds damage but spans >~160 still attenuate — real fix is fp32 V' staging (PERFORMANCE.md `:220-230`).
* radiance extras add fused conv+recurrent decode (one launch, grid barrier, <= 32 WGs) and fp16/bf16
  "narrow state" variants (rx9) — see §2.4.

---------------------------------------------------------------------------------------------------
## 2. vllm-mxfp4 (radiance)

### 2.0 Mechanics
* Two tiers: *baked* source patches (`Dockerfile:278-283`) and *launcher* patches run at container start
  (`serve-mxfp4.sh:989-1012`). Each `patch_*.py` = idempotent unique-anchor string replace + `ast.parse`
  guard (`_patchlib.py:13-28`). Runtime hooks: `install_radiance_hooks.py` edits `vllm/plugins/__init__.py`
  to call `radiance_kernels.install_all()` after general plugins load (`:1-5,11-21`) — i.e. they
  reimplemented a general plugin by source patch. `install_all` (`radiance_kernels.py:406-446`) arms
  w4 load hook, preshuffle load hook, attention-config hook, custom AR, dynamic draft, draft head, ViT attn,
  R4D report. Quant methods registered via stdlib `sitecustomize.py` append (`Dockerfile.ggz14.top`) or
  quant `__init__` patch (`patch_autoround.py`), because site-packages sitecustomize is shadowed on Ubuntu.
* **Stock-0.29 plugin feasibility**: 0.29 loads general plugins in `EngineArgs.__post_init__` *before*
  ModelConfig (`vllm/engine/arg_utils.py:853-856`) and in `add_cli_args` (`:2977`), so
  `register_quantization_config` from a `vllm.general_plugins` entry point is early enough (the reason for
  `patch_autoround`/`patch_escha` does not hold on 0.29 plugins). Attention backends: 0.29 has
  `register_backend(AttentionBackendEnum.CUSTOM, ...)` and mamba-backend override (`registry.py:130,243-273`)
  => no Enum source edit needed. GDN core runs inside splitting op `vllm::qwen_gdn_attention_core`
  (`config/compilation.py:775-776`) => eager Python, monkeypatching `_forward_core` works.
  MXFP4 linear kernels: plugin list `_POSSIBLE_MXFP4_KERNELS[ROCM]` (`kernels/linear/__init__.py:559-572`)
  can be prepended at plugin load. MoE MXFP4 backends are an Enum oracle (`fused_moe/oracle/mxfp4.py:102-279`) —
  no registration API; a plugin must monkeypatch the oracle selection / `backend_to_kernel_cls`.

Hook legend: **REG** = official registry/plugin API; **MP** = monkeypatch from a general plugin
(class method/module attr, before compile or outside the compiled region); **MP-copy** = replace a whole
function with a patched copy (works, fragile across versions); **SRC** = needs source edit (literal inside
a compiled/Triton function, dataclass field threading, non-vLLM tree); **UPSTREAM** = 0.29 already has it.
Relevance: FN = Flash-Next, 27B, both, none. "0.29 audit" = my isolated/sequential apply result.

### 2.1 patch_*.py catalogue
| patch | purpose | measured (doc) | 0.29 audit | rel. | hook for plugin |
|---|---|---|---|---|---|
| patch_gfx1201 | `_get_gcn_arch` from env (amdsmi arch empty on gfx1201), AITER enable on gfx12, triton `HIPDriver.is_active` via `torch.version.hip`, AITER sampler gate (C++ fails to build on RDNA4) | correctness | 3/4 apply (triton hunk: tree absent) | both | MP of `vllm/platforms/rocm.py` / `_aiter_ops.py` fns; triton hunk SRC |
| patch_radiance_dispatch | aiter `compute_splitk_params` K-split 128-alignment (K=8704 page-faulted) + route `TritonFp8BlockScaledMMKernel.apply_block_scaled_mm` to `radiance_kernels.block_scaled_mm` (M<=8 aiter split-K 1.5-1.9x) | 1.5-1.9x small-M FP8 block GEMM | vLLM hunk OK; aiter hunk n/a | FN (FP8 block linears) | MP (method) + aiter SRC |
| patch_skinny_gemm | ROCm unquantized-linear chokepoint -> R4D `gemm_bf16_nt_m64` for M 6..64 (table-gated shapes) | 2.4-5.8x on tiny projections; in_proj_ba 28.5->3.6 us | OK | **FN** (512x2560 router gate, GDN ba) | MP of `layers/utils.py` dispatch fn (traced; r4d call must be a custom op) |
| patch_unified_attention_lds | aiter unified_attention: shrink tile/stages to fit 64 KiB LDS (h256 bf16 KV = 65,792 B -> OutOfResources at capture); bf16 3D decode tune (+14% decode) | correctness + 14% | n/a (aiter) | FN if Triton/aiter attention with bf16 KV (QSA forces bf16) | SRC (aiter) |
| patch_gdn_wmma | FLA `solve_tril` 64x64 inverse dots cast to fp16 -> WMMA | not quantified | OK | FN (FLA fallback) | SRC (Triton kernel body); moot if r4d kkt_solve used |
| patch_preshuffle | FP8 blockscale preshuffled weight `[N/16,K*16]` output shape fix | correctness | OK | FN if preshuffle used | MP |
| patch_radiance_fusion | register native-quant variant of rms+group-fp8 fusion | exact | **UPSTREAM** (NOOP) | — | drop |
| install_radiance_hooks | call `radiance_kernels.install_all()` from plugin loader | — | OK | — | replaced by our entry point |
| patch_unpad | `CommonAttentionMetadata.unpadded()` keeps `seq_lens_cpu_upper_bound` (MTP + disable_padded_drafter_batch EngineDeadError) | enables ~+50% single-stream MTP unpad (MXFP4-NOTES/serve) | OK (still needed) | **FN (MTP)** | MP (method) |
| patch_mtp_mm_mask | re-index multimodal mask with drafter `token_indices` | correctness (MM+MTP) | OK (still needed) | FN if multimodal | MP-copy |
| patch_mtp_loopbreak | honour `_radiance_stop` in MTP draft loop | enables dynamic draft | OK | FN (MTP) | MP-copy of `propose` |
| patch_qwen3_toolparse | stream/non-stream tool-call divergence (#47137) | correctness | OK | both | MP |
| patch_qwen3_thinkoff | qwen3 reasoning parser mirrors template's think-off conditions | correctness | OK | both | MP |
| patch_from_json_filter | `from_json` jinja filter in transformers | correctness (templates) | n/a | both (template) | MP of transformers fn |
| patch_dynamo_metrics | torch 2.11 compile-metrics JSON crash noise | cosmetic | n/a | — | skip unless torch 2.11 |
| patch_conv1d_blockn | FLA/Triton `causal_conv1d_fn` BLOCK_N 256->1024 (2^14-byte pitch aliasing) | 2.22x kernel (304->138 us), bit-identical | OK | FN (if FLA conv path) | MP-copy; moot with r4d conv_prep |
| patch_r4d | (1) R4D attention Enum member, (2) GDN `_forward_core` prologue -> 5 R4D kernels, (3) FLA chunk fwd -> r4d chunk_scan for declined steps | attn +14.6%@64K..+37.8%@260K prefill; GDN part of TP1 +20-24% prefill | OK | **FN (GDN)**, 27B | (1) REG `register_backend(CUSTOM)`; (2) MP `QwenGatedDeltaNetAttention._forward_core`; (3) MP module attrs |
| patch_dflash_base / patch_dflash2 | DFlash2 drafter backport + speculator fixes | — | **UPSTREAM** (NOOP) | none | drop |
| patch_dflash_fused_kv_fp8 / _mxfp4_kv / _w4 / _calib / _selector_topk | DFlash2 drafter with fp8/mxfp4/int4 weights, calibration, top-k knob | draft pass -9.1% (w4) | OK (mxfp4_kv needs fused_kv_fp8 first) | none (FN uses MTP) | — |
| patch_gdn_metadata | numpy bookkeeping, cached arange, slice-not-gather in `GDNAttentionMetadataBuilder.build` | ~1.5 ms of 35 ms/step host time | OK | **FN** | REG (override mamba backend w/ subclassed builder) or MP-copy |
| patch_quark_mxfp4 | put `RadianceMxfp4W4A8LinearKernel` first in `_POSSIBLE_MXFP4_KERNELS[ROCM]`, relax aiter `is_fp4_avail`, aiter module path | native 6.1x@M16 vs emulation; W4A8 1.47-2.26x vs aiter, 4.2x more accurate | OK | **27B**; FN only for dense MXFP4 linears | REG (list insert) + MP aiter |
| patch_ar_maxbytes | `RADIANCE_AR_MAX_KB` knob (upstream 48 MB gate; CHUNK*hidden*2 must fit) | 924 prefill ARs 3.145->1.317 ms (2.18x); +0.9-12.8% prefill | OK (targets radiance module) | both | own module |
| patch_topk_triton_rows | Triton top-k/top-p from 1 row instead of 8 (sort slower at every row count, vocab 248k) | -2.8% step, bit-identical | OK | both | MP-copy (literal in fn) |
| patch_nvfp4_mxfp4 | CT NVFP4 (and FP8 per-channel) -> requantize to MXFP4 at load | conc-8 597 vs 512 t/s | OK | none (pattern for CT scheme override) | MP of `_get_scheme_from_parts` |
| patch_tp3_pad | TP=3 dummy-head padding hooks (config, loader, vocab pad multiple) | GSM8K parity | OK | 27B TP3 only | MP (3 method wraps) |
| patch_rmsquant_fusion | swap aiter FUSED_OP for radiance rms+quant (aiter RMSNorm broken on RDNA4) | default off | OK | low | MP (class attr) |
| patch_verify_head | per-step gate for int2 target verify head | +2.9% combined decode | OK | 27B (dflash); FN possible | MP |
| patch_kv_group_size | hybrid KV group size by capacity not min bucket | +20.7% KV tokens (739,544->892,799) | OK | **FN** (hybrid GDN/QSA/MTP buckets — recheck) | MP-copy of kv_cache_utils fn |
| patch_topk_composite | small-k top-k/top-p via torch.topk candidate window; threads `max_top_k` | 907 us/step kernel removed (~51 us) | OK | both | SRC (dataclass field across 6 files) |
| patch_gdn_shared_build | share GDN metadata build across KV groups | 573 us/step python | OK | **FN** | SRC-ish (runner + builder) |
| patch_gdn_merge_inproj | call site for `radiance_gdnmerge` after model load (both runners) | -2.9% decode | OK | FN only if in_proj layouts allow concat (FP8/GPTQ in FN) | MP of `load_model` |
| patch_dynwidth / patch_async_dynwidth | per-request verify width from acceptance EMA (sync + async scheduler) | conc-8 +11-13% | OK | both (spec decode) | MP (scheduler methods) |
| patch_step_trace | per-step CPU timing trace | diagnostic | OK | tool | MP |
| patch_ar_geometry / patch_ar_qbits | env knobs for wht6 geometry / 5,4-bit wire (needs libr4d "rx8", **not shipped**) | — | OK | low | own module |
| patch_ar_3rank | make `radiance_allreduce.py` ws-generic, 3-rank kernel | untested on 3 cards | OK | 27B TP3 | own module |
| patch_gdn_glue | strided gates / empty core_attn_out | **neutral**, dark | OK in sequence | none | — |
| patch_gdn_lazy | lazy GDN snapshots (TP1) | **REVERTED: corrupts multi-turn chat** (PERFORMANCE.md `:41-66`) | 8/16 hunks miss | none | do not reuse |
| patch_draft_attn_blockm | fold drafter query rows (BLOCK_M 32) | **REJECTED**: 4-34% slower | OK | none | — |
| patch_autoround / patch_escha | register quant configs early | — | OK | none | REG on 0.29 |

### 2.2 radiance_*.py catalogue
| module | purpose | measured | rel. | plugin hook |
|---|---|---|---|---|
| radiance_mxfp4.py (862) | `MxFp4LinearKernel` plugin -> `radiance_mxfp4_fp8.so`; W4A8 fp8 act quant (traced), WPERM, decode band <= `DECODE_MAX_M`, optional r4d decode (`RADIANCE_MXFP4_R4D_DECODE_MAX_M`, default 0, needs WPERM, `:59-104,448-451`), K%64 gate (`:644-653`), scratch owned by torch | 0.7.4 vs 0.5.8: prefill +13..+50%; decode band -8.3% step, +28.5% conc-4 | 27B (FN dense MXFP4 only) | REG |
| radiance_kernels.py | dispatcher: FP8 block GEMM route, preshuffle load hook, tuned unified-attn configs, `install_all`, R4D report | — | FN (FP8) | MP |
| radiance_allreduce.py | wraps `CudaCommunicator.__init__/all_reduce` (not replacing ca_comm: aiter fusion asserts its type), 2-rank exact + wht6, TP group ws==2 only (`:240-281`) | prefill AR 2.18x | **both** | MP |
| radiance_arnq.py | fp8 residual-stream contract: fused AR + residual + Gemma RMSNorm + fp8 quant per RowParallel site (needs extras `_nq`) | with traced quant: 25.4->22.66 ms/step | 27B; FN needs rework (hyper-connections change the residual) | MP of model forward (invasive) |
| radiance_aroverlap.py | slice GEMM+AR overlap at prefill | **worse** (-2..-5.6%) | none | — |
| radiance_gdn.py (720) | R4D GDN dispatch, select() by state dtype, bails to FLA per call (`:203-323`) | TP1 prefill +20-24% | **FN** | MP |
| radiance_gdn_lazy.py | lazy snapshot materialize | reverted | none | — |
| radiance_gdnmerge.py | concat in_proj_qkvz + in_proj_ba MXFP4 tensors along N | -2.9% | FN if layouts concat | post-load MP |
| radiance_gemm.py | skinny bf16 GEMM table (MoE gate W[256,2048], GDN ba W[96,5120]) | 2.4-5.8x | **FN** (router 512x2560) | MP |
| radiance_r4d_attn.py | R4D attention backend (subclass of Triton backend), per-(seq,head) descales, equal-q_len runs | +14.6..37.8% prefill | 27B; FN only for non-QSA / as sparse template | REG CUSTOM |
| radiance_vit_attn.py | ViT h72 SDPA drop-in | — | FN if vision tower is h72 | MP |
| radiance_rmsquant.py | plain-torch rms+quant replacement | off | low | — |
| radiance_topk.py | composite top-k/top-p | 907 us -> ~51 us | both | MP (with patch) |
| radiance_drafthead.py | int2 draft lm_head + exact rerank | 2002->473 us/call; +6.5% decode | both (MTP shares lm_head) | MP |
| radiance_verifyhead.py | int2 target verify head, gated to stay exact | +2.9% | both (risk class higher) | MP |
| radiance_draft.py / _draft_gpu.py | confidence-gated dynamic MTP depth + GPU n-gram | lossless; throughput | FN (MTP) | MP |
| radiance_w4.py | DFlash2 drafter -> int4 on r4d w4a16/w4a8 | draft -9.1% | none | — |
| radiance_tp3pad.py | TP=3 zero-head padding | GSM8K parity | 27B TP3 | MP |
| radiance_nvfp4.py | NVFP4->MXFP4 requant | — | none | — |
| radiance_autoround.py / radiance_escha.py / paroquant/* | other quant formats | see §3 | none | — |
| radiance_dflash_capture.py | drafter training capture | tool | none | — |
| radiance_preamble.py / radiance_amdsmi.py | banner/prechecks; **amdsmi_init before HIP** (else enumeration empty) | correctness | both | .pth / plugin import-time |
| fp8_mtp.py / tp3pad_selftest.py | checkpoint rewrite (MTP head fp8), TP3 self-test | — | 27B | — |

### 2.3 TP=3 padding and 3-rank AR (TP3_PADDING_PLAN.md)
* Megatron-style zero heads at runtime: 27B widened to 36 q / 6 kv (keeps per-rank GQA 6 for R4D),
  18/54 GDN k/v heads, MLP 17472, vocab 248448 (pad multiple 64->192) (`:83-92`, `radiance_tp3pad.py`).
  Fills: MXFP4 weight 0x00, **e8m0 scale 0x7F** (never 0xFF=NaN), fp8 scales 1.0, A_log 0 (`:110-114`).
  Streaming padder wraps the weights iterator before sharded loaders (`:138-145`). Upstream rejected
  head padding (vllm#11797).
* Lessons (`:29-41`): (1) MXFP4 decode GEMM needs per-rank **K%128** (BK=128 slabs have no k bound; padded
  K read the next row -> token-0 garbage; fix: K%128 -> BK=64); (2) drafter KV bytes/token must equal the
  target's or the prefix-cache lookup asserts; (3) GQA ratio is part of the trained weights (pad keeping GQA).
* 3-rank AR (`:165-203`): symmetric one-shot ties a hub under full-duplex PCIe; 2 receive regions x 2 slots;
  canonical rank-ascending fp32 sum `((x0+x1)+x2)` to keep cross-rank bit identity; decode-only cutoff
  2 MiB at ws=3; gates Phase0 (3 procs on 2 GPUs PASS), Gate M (link microbench). Untested on 3 cards.
* For us (TP2): not needed. Only relevant if a 3rd R9700 is added.

### 2.4 `r4d_radiance_extras{,_rx9,_rx10}.patch` (against libr4d `b9e42ab`)
* rx6 (`r4d_radiance_extras.patch`, 874 added lines): `r4d_ar_oneshot_2rank_exact_nq` (AR + residual +
  Gemma RMSNorm + per-token e4m3 quant, one block per row, `:81-240`), wht6 measurement notes, new
  `r4d_ar_oneshot_3rank_exact.hip`, attention `R4D_ATTN_FP8` opt-in bits O_QK8/O_PV8 (fp8 QK / PV legs,
  +10.8% prefill @106K, ppl unchanged, `:466-700`), new `r4d_gdn_fused_update_w4k128v128.hip` (decode conv +
  recurrent in one launch, grid barrier, <=32 WGs, bit-identical, `:704-998`), registry/module/build rows.
* rx9 (1459 lines) = rx6 + narrow-state GDN: `r4d_gdn_state.h`, recurrent/fused update for bf16- and
  **f16-state** caches (fp16 preferred: 10-bit mantissa) — lets `--mamba-ssm-cache-dtype float16` halve the
  GDN page (TP1: 3 -> 6 concurrent). rx10 = rx9 + lazy snapshots (reverted feature).
* **Applicability to libr4d HEAD**: `git apply --check` fails; `git apply -3` of rx9 applies every kernel
  file cleanly and conflicts only in the 4 list files (`build.sh`, `r4d.h`, `r4d_module.hip`,
  `r4d_registry.hip`) — mechanical rebase (tested in a scratch clone).

---------------------------------------------------------------------------------------------------
## 3. radiance's own HIP kernels
* **radiance_mxfp4_fp8.hip (1720 lines)** — W4A8 MXFP4 x e4m3 GEMM family + fused producers (pybind `:1700-1718`):
  - folded prefill: BM=256 (TM=4), BK=64, TN 2/4 (TN=4 from M>=2048: +10% @8192, -8.8% @512), kMag fold in the
    LDS upconvert, LDS rows padded 8 B (8-way conflicts otherwise), clamp-never-predicate staging (`:110-120,150-174`);
  - A-tiled prefill (activation arrives WMMA-fragment tiled from the producer; no A in LDS; 215-220 TF/s vs 190;
    SGPR-based A loads, kMag in LDS, LDS-scoped barriers; +1.8..2.6% served prefill) (`:474-523`);
  - decode (M<=128: DTM=ceil(M/16)<=8, DBK=128 or 64 by K%128/DTM, **fused split-K**: partials fp32
    `[DKS][M][N]` in torch-owned scratch, last-arrival reduce via atomic counter, graph-safe), nontemporal
    weight loads (`:760-792,1011-1088`). Split-K rule: smallest DKS that fills the machine, falling with M
    (table `:1047-1053`, rule on nblk*ks);
  - `launch_p/at_p`: merged linear with up to 3 differently-rotated activation copies (ParoQuant);
  - `launch_add_rms_quant` (residual add + Gemma (1+w) RMSNorm + per-token e4m3, bit-identical to traced),
    `launch_silu_mul_quant`, `launch_gdn_norm_quant` (`:1325-1544`). Scale rule everywhere:
    `scale = max(amax/448, 1/(448*512))` over the **bf16-rounded** value.
  - Perf: 1.47-2.26x vs tuned aiter, fp8 WMMA measured 325 TF/s vs f16 160 vs Triton fp8 tl.dot 43.
* **radiance_autoround.hip** (+`radiance_autoround_kernels.h`) — int4 g128 symmetric (GPTQ zero 8 folded into
  the e4m3 LUT; codes c-8 exact in e4m3) x fp8 W4A8; group 128 == decode slab so rescale sits on an existing
  barrier, one scale per lane per slab; decode M<=128, split-K rule mirrors MXFP4. 55-89% of streaming
  roofline at M<=16 (`autoround-tests/RESULTS.md`). Not needed for our checkpoints (but a GPTQ-sym W4A8 path
  if Flash-Next's "GPTQ others" are symmetric g128 — check).
* **radiance_escha.hip** (+`escha/*.h`, MIT exllamav3-derived) — whole EXL3-trellis W2/W3 layer:
  `Had128((x*s_in)*rin) @ trellis_decode(code) -> Had128 -> *rout -> *s_out`, M-branch in C++ (a Python
  shape branch cost ~30% decode). Tuned decode 1.07-1.37x slower than MXFP4 per shape; prefill 162-204 TF/s
  (`escha/RESULTS_tuning.md:15-33`). Split-K rule: `ks = largest pow2 <= 8 with ktiles/ks >= 36 and
  nblk*ks <= 576`. Not relevant beyond tuning lessons.

---------------------------------------------------------------------------------------------------
## 4. gfx1201 lessons / gotchas (collected)
Numerics
* fp8 WMMA honours e4m3 **subnormals** (fold tables rely on it). e2m1 x 2^-d, d<=8, exact in e4m3.
* No `v_cvt_pk_bf16_f32`, no direct-to-LDS, no `ds_read_b64_tr_b16` (`libr4d README.md:134-141`, `r4d_common.h:27-29`).
* `v_perm_b32` selectors: 0-3 = bytes of S1, 4-7 = S0, 8-12 zero but **11 data-dependent**, 13-15 = 0xFF
  (= e4m3 NaN). Use only 0..7 (`autoround-tests/RESULTS.md`). Uniform-value tests cannot detect permutation
  bugs — vary codes in both n and k.
* W4A8 (fp8 activations) is more precise than checkpoint W4A4 but not bit-identical to emulation.
* GDN exp split overflow -> NaN (fixed); a single NaN activation -> NaN per-token fp8 scale -> poisoned row;
  judge state-cache changes with **multi-turn prefix-cache chat**, not needles (lazy GDN lesson).
* Per-token quant scale uses the bf16-rounded value; `-ffp-contract=off` for anything summed on 2 ranks.
* e8m0 0xFF is NaN; pad scales with 0x7F.
WMMA / LDS / codegen
* Fragment layout (wave32 16x16x16): `idx = lane%16`, `k = 8*(e>>2) + 4*(lane>>4) + (e&3)`; A and B both
  K-contiguous, swapping operands transposes C for free (`libr4d README.md:134-135`). radiance's form:
  A lane l holds `A[l%16][(l/16)*8+j]`, C `C[(l/16)*8+j][l%16]` (`radiance_mxfp4_fp8.hip:20-22`).
  Only 16x16x16 (aiter/Triton `matrix_instr_nonkdim=32` configs must be pinned to 16).
* 64 KiB LDS/WG: attention TILE capped at 48; BK=128 decode tile loses at DTM=4 (2 vs 4 resident blocks);
  prefill BK=128 -34% purely from LDS occupancy.
* `__syncthreads()` emits `global_inv` + full loadcnt drain on gfx12 -> use LDS-scoped fence+`s_barrier`
  (`r4d_common.h:70-74`). LLVM single-buffers LDS -> `sched_group_barrier` (coarse groups) needed.
* Clamp, never predicate, staging loads (predication forces `s_wait_loadcnt 0` per load; 7-26%).
* Constant-memory table loads after A loads serialize on in-order loadcnt; put small tables in LDS.
* `-mcumode` per TU for 60 KB-LDS kernels; radiance offers `MXFP4_CUMODE` A/B.
* Triton will not emit fp8 WMMA (43 TF/s); `tl.dot_scaled` lowers via bf16 WMMA.
Build / runtime
* hipcc flags used: `-O3 -std=c++17 -fPIC -shared --offload-arch=gfx1201 [-w] [-ffp-contract=off] [-mcumode]`.
  Build with the exact ROCm the process loads; build needs no GPU.
* Any C++ exception from a pybind .so is relabelled "libamdhip64.so not found ... TileLang" by quark's
  translator — don't trust that message. Never hipMalloc lazily from launch (lands in graph capture):
  let torch own scratch and pass pointers (`radiance_mxfp4_fp8.hip:1011-1020`).
* amdsmi must init + enumerate before HIP (`radiance_amdsmi.py`). AITER RMSNorm and AITER sampler do not
  work on RDNA4. aiter module paths move between versions.
* Keep M-dependent dispatch inside the custom op (C++), not in `apply_weights` (graph break cost 30-33% decode).
* torch 2.13/triton 3.7.1 hung GPU under sustained TP load; `torch._scaled_mm` (hipBLASLt fp8) wedged the GPU
  twice under 8-way load (README NVFP4 section). Move torch/triton/torchvision together.
P2P / PCIe / AR
* Their box: Gen5 x8/x8 (27.7 GB/s push); ours (.100): Gen3 PLX, 14 GB/s per GPU H2D. The 80 MiB wht6 AR is
  ~1/3 wire, 1/3 local DRAM, 1/3 peer skew. AR size gate must cover `CHUNK*hidden*2` or prefill silently
  falls to RCCL (hidden 2560 => 8192-chunk = 40 MiB, under the 48 MB default).
* 2-rank spin timeout silently continues with stale data; add a counter in soak tests.
TunableOp / cudagraphs / memory
* **TunableOp: not used or mentioned anywhere in either repo.**
* Cudagraph sizes: stock list kept; finer sizes (3,5,6,7,...) neutral (decode GEMMs ~M-invariant below 16);
  cap capture sizes at `MAXSEQS*(SPEC+1)` (saved 1.05 GiB at TP1, `serve-mxfp4.sh:654-665`); decode-kernel band
  must cover `MAXSEQS*(SPEC+1)` (`:600-605`). Inductor static compile sizes changed numerics and cost MTP
  acceptance (1.837 vs 2.069 acc/draft) — keep off. Async scheduling: no gain (worker CPU fits in step).
* Explicit `--kv-cache-memory` pin beats profiling by ~0.93 GiB/rank (+5.7% KV); derive from the card under
  load, re-derive after any change to weights/capture sizes/CHUNK.
* Benchmarks: 64 MB Infinity Cache flatters any buffer <64 MB — rotate >=4 weight copies (NCOPY=6);
  210 W cap makes cards differ ~12% on prefill — compare same-card; "fill the GPU before judging a tiling"
  (M=2048 microbench win was an M=8192 regression).

---------------------------------------------------------------------------------------------------
## 5. Components to reuse for our plugin (ranked by value/effort)
1. **libr4d `gemm_mxfp4a8_nt_m64` -> grouped MoE MXFP4xFP8 GEMM** (value: highest — experts dominate
   Flash-Next decode bytes; effort: medium). Reuse kMag fold, `permute_w`, Wref, clamped A/W loads, LDS
   split-K; add sorted-token gather + expert-indexed W/Ws/Wref + early-exit + top-k combine (§1.3). Pair with
   an fp8 per-token quant kernel and radiance's `silu_mul_quant` numerics. Hook: monkeypatch the MXFP4 MoE
   oracle / `backend_to_kernel_cls` (`fused_moe/oracle/mxfp4.py:154-279`). Measure the E8M0 `d` histogram of
   the CT expert scales first; use EP or BK=64 for TP-sharded down (K=320).
2. **2-rank P2P all-reduce** (libr4d exact + wht6, `radiance_allreduce.py` CudaCommunicator wrap) —
   value high, effort low; pure MP plugin, works on 0.29 unchanged. Add spin-timeout counter.
3. **libr4d GDN set + `radiance_gdn.py` + patch_r4d's `_forward_core` prologue as a monkeypatch**, plus
   rx9 `fused_update` / f16-state kernels (rebase: 4 list-file conflicts) — value high for FN (head 128
   assumed; confirm `linear_*_head_dim=128`, conv width 4), effort low-medium.
4. **Skinny bf16 GEMM (`gemm_bf16_nt_m64`) for the 512x2560 router gate and GDN ba projections** via
   `radiance_gemm.py` table — value medium, effort low (measure the new shapes; table is shape-keyed).
5. **Host-side/scheduler fixes as monkeypatches**: `patch_gdn_metadata` (~1.5 ms/step), `patch_topk_triton_rows`
   (-2.8%), `patch_unpad` (MTP unpad), `patch_kv_group_size` (+20% KV if buckets match), `patch_gfx1201` + amdsmi
   early init, `patch_ar_maxbytes` logic. Value medium, effort low; all apply to 0.29.
6. **radiance fused producers** (`add_rms_quant`, `silu_mul_quant`, `gdn_norm_quant` in `radiance_mxfp4_fp8.hip`)
   — value medium (launch count), effort medium for FN because hyper-connections change the residual contract.
7. **R4D attention**: `h256_gqa12` instantiation (template-only change) for any dense full-attn/MTP layers,
   and as the template for a QSA sparse decode kernel (bf16 KV). Value medium-high if attention share grows
   with context; effort medium (sparse: high). Register via `register_backend(CUSTOM)`.
8. **27B MXFP4 dense path** (`radiance_mxfp4.py` + `radiance_mxfp4_fp8.hip`, `_POSSIBLE_MXFP4_KERNELS` insert)
   — value high for 27B only, effort medium (large module; W4A8 numerics gate). libr4d's r4d decode kernel is
   an opt-in alternative there (1.15x/1.3x kernel-level, never flipped to default in serve).
9. Methodology/tools: `audit_upgrade.py` (patch/anchor audit vs new vLLM), KV-pin calibration, NCOPY
   benchmarking, GSM8K-paired + multi-turn gates. Effort nil.
Skip: TP3 padding/3-rank AR (unless a 3rd card), DFlash2 stack, escha, autoround, ParoQuant, lazy GDN,
AR overlap, draft-attn BLOCK_M fold, gdn_glue.
