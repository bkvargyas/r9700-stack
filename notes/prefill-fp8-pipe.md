# Prefill on the Gen3 switch box: fp8 tiled GEMMs, the fused hyper-connection mix, and all-reduce pipelining (2026-10-07)

Brian: "let's keep going on the kernel items from last answer, as we will have 8 cards on this Gen3 PLX system at
some point. I cannot change the hardware, so let's do what we can do." The items were the prefill side of
Flash-Next TP4: a 4096-token chunk is 602 ms wall / 548 ms GPU-busy, of which (profile `~/tprof-ours-prefill4kc`,
two chunks, per chunk): all-reduce ~147 ms (two 21 MB messages a layer over the single inter-switch uplink),
hyper-connection combine-norm 20.6, the hc down GEMM 18 (hipBLASLt bf16, 336 x 10240), the hc up GEMM 16.2 +
its gate mix 16.4, the Triton block-fp8 projections 17, MoE ~120, launch gaps 54.

## 1. fp8 tiled prefill GEMM (F8) -- `r9k_moe_4bit_prefill` nv=3

The LDS-tiled double-buffered WMMA kernel takes e4m3 weights in `permute_fp8`'s fragment order straight into its
slab layout (no unpack, no group scales) with one fp32 scale per column in the epilogue. Card 4, 4096 rows, 210 W,
`tests/test_fp8_prefill_r9k.py` (exact against the dequantised reference, 1e-5 against the split-K decode kernel):

| shape (N x K) | tiled fp8 (best cfg) | hipBLASLt bf16 | note |
|---|---:|---:|---|
| 336 x 10240 (hc down) | 172-177 us, cfg 10, 159-163 TFLOPS | 245-334 us | + a 212 us per-token quant of xn |
| 10240 x 320 (hc up) | 197-210 us, cfg 9/17 | 279-313 us | output-write bound (84 MB) |
| 2560 x 2560 | 296-303 us, cfg 9, 177-182 TFLOPS | 394-418 us | |
| 4096 x 2560 | 473 us, cfg 9, 181 TFLOPS | 611-672 us | |
| 1536 x 2560 | 182 us, cfg 9, 175 TFLOPS | 231-250 us | |

cfg 9 (256 x 128, 8 waves, BK=32 double-buffered) is the default from 512 rows; 64 x 128 (cfg 17) below 256 rows,
128 x 128 (cfg 10) between. `ops.fp8_linear` (per-channel fp8 linears, rowwise block-fp8) takes it from 64 rows.

**The hc down GEMM alone does not pay**: its per-token quant of xn (84 MB read, 42 written) costs 212 us against
the ~160 us the GEMM saves. It stays behind `R9K_HC_FP8_DOWN=1` until the combine-norm kernel (ours, one
workgroup per row and stream) emits the fp8 copy and a per-(row, stream) scale itself -- that needs the block
variant below with a 2560-wide group.

## 2. The fused hc up GEMM + gate mix (MIX) -- `r9k_fp8_prefill_mix`

The up GEMM's output is the 4-stream gate (10240 wide) that `hc_gate_mix` immediately reduces to 2560: write
84 MB, read 168 MB. With the weight rows interleaved 16 columns x 4 streams per 64 (`hc4_interleave`), a wave's
64-column tile holds all four streams of 16 output columns, and the staged epilogue applies sigmoid x xn and the
mean itself, writing [M, 2560]. Card 4, 4096 rows: **268-298 us** fused vs hipBLASLt 351-359 + gate mix 315 =
665-674 us (+ 36 us for the lora row quant). Exact against the stock formula on the same fp8 gate (bf16 output
rounding only). ~370 us x 49 layers = ~18 ms per 4k chunk. `R9K_HC_FP8=r9k` (default stock until the GSM8K /
HumanEval gate passes: the gate's operands are fp8 now, per-token activations, per-row weights; the bf16 path's
result differs by 0.8% relative on random data).

## 3. Exact block-scaled fp8 (F8B) -- `r9k_fp8_prefill_block`: correct, slower than Triton

Per-(row, 128-group) activation scales and per-(128-block, group) weight scales promoted into the accumulator at
each group's end (two accumulator sets, so 128 x 128 tiles at most). Reproduces stock's block-fp8 math (2e-5
against the decode block kernel), but on the served shapes it loses to vLLM's Triton block GEMM, which reaches
167 TFLOPS here: 3584 x 2560 528 vs 451 us, 4096 x 2560 611 vs 529, 2560 x 1536 258 vs 200, 6656 x 2560 1083
vs 860. The promotion (64 FMAs + 32 LDS reads a lane per 64 WMMAs) and the smaller tile cost ~20%. Kept opt-in
(`R9K_FP8_BLOCK_PREFILL=r9k`), the hybrid dispatch stays on Triton above the decode cap. A per-token-activation
variant (weight block scales only: no LDS reads at the promotion, 256-row tiles possible) would be the next try,
but it changes numerics; not worth ~8 ms a chunk against the pipeline below.

## 4. All-reduce pipelining (comm/pipe.py, `R9K_AR_PIPE=r9k`)

The all-reduce is the chunk's largest item and it is bandwidth, not kernels: 21 MB over one Gen3 uplink per
message, ~1.5 ms each, two a layer. Everything after the attention core is row-local (hc combine + mix, the MoE,
their all-reduces), so the tail runs in row parts: all attention-output parts are all-reduced on a comm stream
(a second ar4 instance, own IPC scratch, 24 MiB messages) while the main stream combines / mixes / runs the MoE
on the parts already reduced, and each part's MoE output is all-reduced while the next part computes. With two
2048-row parts the exposed all-reduce per layer is about one message instead of two (the routed experts' prefill
tiles stay as full as at 4096 rows: ~40 rows an expert = one 64-row block each, the same work); more parts only
at longer chunks. The layer's forward is rebound (compat gate `layer_tail_pipe`): the attention RowParallelLinear
no longer reduces, the MoE runner skips its final all-reduce, and the tail is one opaque op that runs the stock
order (same kernels, captured as before) at decode widths. Expected: ~1.5 ms x 49 = ~70 ms of 602 per chunk;
with 8 cards the all-reduce share grows and the same structure hides half of it.

### What the hardware said (16:00-17:30 UTC)

`tests/test_ar_pipe.py` on four ranks: the part-wise all-reduce is correct (same 4-bit wire error as the whole
message, ranks agree), but the stand-in timing is sobering: one 21 MB ar4 1.23 ms, the compute stand-in 2.45 ms,
the stock tail 5.15 ms, pipelined 4.93 (P=2) / 5.19 (P=4) against an ideal 3.68. The fused-push all-reduce
kernels (128 workgroups spinning on handshakes) time-share the CUs with the GEMM instead of running beside it.

Serving (Flash-Next TP4, MTP-3, 210 W, `~/tp4tune.sh`, ms/step | decode tok/s | conc 8 | conc 16 | 8k prefill):

| config | ms/step | dec | c8 | c16 | prefill 8k |
|---|---:|---:|---:|---:|---:|
| base (v0.2.6 defaults), two runs | 16.41 / 16.43 | 199.6 / 199.4 | 592.6 / 591.2 | 898.5 / 894.7 | 6769 / 6761 |
| hc fp8 mix (`R9K_HC_FP8=r9k`), two probes | 16.44 | 198.6 | 575.8 / 612.9 | 905.2 / 900.8 | **7044 / 7009** |
| all-reduce pipeline (`R9K_AR_PIPE=r9k`), two probes | 16.58 | 197.0 | 578.1 / 609.5 | 896.0 / 891.4 | 6403 / 6355 |

The hc fp8 mix: +3.9% at 8k prefill, decode untouched, strict sanity clean at conc 8 / 17 and on long prompts.
Three serving bugs on the way: (1) the fp8 weight copies were quantised at `install_mix` time, i.e. at model
construction before the checkpoint is loaded -- single-stream decode (bf16 below 256 rows) looked fine while every
batched or long prefill was garbage (sanity 140 bad of 160 at conc 8; the probe's concurrency halved because the
drafter stopped accepting). Fixed: `hc.quantize_fp8(model)` from the three `load_weights`. (2) the pipe op
returned the mix's injection as a column slice of its [M, 336] buffer; the compiled graph asserts the fake impl's
(contiguous) strides. (3) vLLM sizes and LOCKS the MoE modular kernel's workspace on the profiling run; a split
profiling run sized it for one part and the first unsplit chunk between part and chunk width crashed the engine
(`Workspace is locked but allocation ... requires 13.55 MB, current size is 10.00 MB`): the first full-width
call now runs unsplit.

The pipeline itself is correct in serving but a net loss (-5.5% at 8k, +1% on the decode step from the moved
all-reduces and the injection copy): no overlap on this topology with the fused-push kernels, plus the split's
own overhead. Next: the comm instance with fewer push workgroups or DMA-engine pushes (`R9K_AR_PIPE_BLOCKS`,
`R9K_AR_PIPE_SDMA`; the sweep is in the test). If DMA pushes overlap, the structure is worth keeping for 8 cards;
if not, the pipeline is shelved and the all-reduce stays what it is: bandwidth.

## Where things are

Commits (local master): b0eb1d2 F8 + MIX, 246d35f F8B, 10b22e6 pipe. Test box tree `~/r9700-build/repo-f8`
(libr9k.so built), card-4 runs `~/f8gates.log` / `~/f8sweep.log` / `~/f8mix.log` / `~/f8blk2.log`, tools
`~/f8run.sh` (one test, full output, no build) and `~/f8wait.sh`. Production copy `~/r9700-build/repo` untouched.

### Gate, sweep, and where the KV cache went (18:00-18:35 UTC)

**Gate (hc fp8 + pipe, `f8-both-fn4`)**: GSM8K 95.45% (1,259 / 1,319) vs the v0.2.6-era record `r9knodes-fn4`
95.60%; discordant 14 vs 16, McNemar p = 0.86, identical outputs 14.9% (the fp8 gate changes bits in every
layer); HumanEval 160 / 164 = the record. No detectable difference: the hc fp8 mix is clean to ship as a knob.

**Comm-instance sweep** (4-rank stand-in, stock tail 5.15 ms, ideal 3.68): fused pushes 128 blocks 4.93, 16 blocks
6.06 (AR alone 1.96 ms: too few pushers for the uplink); DMA pushes 128 blocks **4.67** (AR alone 1.21), 32 blocks
5.61. DMA engines overlap a third of the all-reduce; the rest still serialises. One serving run with DMA pushes
is queued; if it does not beat the hc fp8 config alone the pipeline is shelved.

**The KV cache.** Brian: "So TP4 QFN only has 259k of KV cache?" Per card at utilisation 0.94 (31.86 GiB, budget
29.95): weights + non-torch 22.59 GiB, "peak activation" 3.83, CUDA graphs 3.2 (captured after the KV cache is
sized, so outside vLLM's arithmetic), KV 3.53 GiB = 279k tokens; the fn4 soak peaked at 30.0 GB of 32.6 a card.
`compat/memsnap.py` (R9K_MEMSNAP=1, EAGER=1) on the same 4096-token profiling forward: **the model's peak is
608 MiB above entry** (largest part the layer-1 PLE gather at 840 MiB), and eager vLLM sized the KV cache at
7.57 GiB = **513,088 tokens**. The 3.83 GiB is torch.compile / inductor autotuning during the profiling run,
counted as activation. Levers measured or queued: utilisation 0.98 (`kv-hc-u98`): 348,834 tokens, probe
unchanged (dec 199.7, c16 910, prefill 7069), 25-min soak with VRAM sampling + conc 17/32 sanity running;
capture sizes <= 256 + explicit `--kv-cache-memory` (KVMEM 6.5 / 7.5 GiB a rank, ~440k / ~500k tokens) queued
with soaks; a 2048-token chunk for comparison. fp8 KV cache (2x) assessed as its own item (QSA / decode /
drafter fp8 read paths, ~a day, then the long-context gate).

### KV cache runs (Flash-Next TP4, hc fp8 on, `~/kv-exp.sh`, 18:18-19:23 UTC)

| config | KV tokens | vLLM's "peak activation" / graphs (GiB) | soak | VRAM peak / card (MiB of 32,624) | probe (dec / c16 / 8k prefill) |
|---|---:|---|---|---:|---|
| 0.94 (base) | 279,564 | 3.83 / 3.2 | validation soak 627 ok | 30,001 | 199.6 / 898 / 6769 |
| 0.98 | 348,834 | 3.83 / 3.19 | 25 min, 650 ok, 0 err | 30,500 | 199.7 / 910 / 7069 |
| 0.98 + capture sizes <= 256 | **442,575** | 2.45 / 1.82 | 10 min, 270 ok, 0 err, 6677 prompt tok/s | 31,301 | 198.8 / 887 / 6810 |
| 0.94 + 2048-token chunk | 311,503 | 3.63 / 3.2 | 5 min, 151 ok | 28,552 | 197.8 / 898 / 6480 |

Sanity 0 bad at conc 17 (680) and conc 32 (960) on every row. The VRAM peak is weights + graphs + KV to within
a few hundred MB: the serving activations are small, and the 2048-token chunk changed vLLM's estimate by 0.2 GiB
(it is not the chunk's activations) while costing 8% of prefill. Trimming the graphs to 256 tokens is free in the
soak's prompt throughput (6677 vs 6726 tok/s) and worth +94k tokens on top of 0.98. Next: an explicit budget
(`KVMEM`, --kv-cache-memory) of 7.0 and 7.5 GiB a rank with the trim (~460k / ~495k tokens), to find the edge.

Explicit budgets with the trim (`~/kv-exp2.sh`, 19:20-19:53 UTC):

| config | KV tokens | soak | VRAM peak / card (MiB of 32,624) | free at peak | probe (dec / c16 / 8k prefill) |
|---|---:|---|---:|---:|---|
| trim + `KVMEM=7.0` | 474,928 | 15 min, 395 ok, 0 err, 6680 prompt tok/s | 31,672 | 952 MiB | 199.2 / 887 / 6847 |
| trim + `KVMEM=7.5` | 508,940 | 10 min, 308 ok, 0 err, 6558 prompt tok/s | 32,184 | 440 MiB | 198.9 / 883 / 6814 |

Sanity 0 bad at conc 17 / 32 on both. Recommendation for the fn4 defaults: utilisation 0.98 + capture sizes
<= 256 (442k tokens, 1.3 GB free at the soak peak; a 25-minute soak is still owed on exactly that pair, the
0.98 and the trim were soaked separately for 25 and 10 minutes); `KVMEM=7.0` (475k, 0.95 GiB free) as the
documented opt-in for the longest contexts; 7.5 is the measured edge, not a setting.

### The pipeline's verdict (20:01 UTC)

Serving with DMA pushes (`R9K_AR_PIPE=r9k R9K_AR_PIPE_SDMA=1`, hc fp8 on): 8k prefill 6637 against 7044-7069 for
the hc fp8 config alone, decode 198.0. The pipeline is shelved: correct (sanity clean, gate passed with it on),
in the tree behind `R9K_AR_PIPE` (default stock), and not worth its own overhead on this topology because the
all-reduce kernels, fused or DMA, do not run beside the GEMMs. What would change the verdict: an all-reduce that
holds no CUs while the bytes move (a DMA-only push with a host-side or SDMA-side completion), or a topology where
the message is short enough that the split's overhead is the smaller term. For eight cards the message grows and
the split's overhead does not, so the structure is worth keeping in the tree.
