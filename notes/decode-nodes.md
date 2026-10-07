# Decode node count: three fusions, what they bought, and what the profiler got wrong (2026-10-07)

Brian: "Is there any more tuning we can do on our kernel to speed things up?" -> "Go ahead and work on it." The
list's item 2 was the decode node count: a Flash-Next TP4 decode step was 1,990 kernel launches, and the 2026-09-28
profile had blamed the gap between our GPU time and the step time on them. This note is the result: three fusions
built, validated and measured, a bug found in the first serving run of each, and a smaller number than the profile
promised. Everything is in the tree behind knobs; the measurements are Flash-Next TP4, MTP-3, 210 W / -42 mV, the
test box, `~/tp4tune.sh` (512-token essay for the step time, `probe.py` for decode / 8 conc / 16 conc / 8k prefill).

## The baseline profile (v0.2.5)

Torch profiler, graph mode, 90 steady decode steps (`~/tprof-v025-decode`, `~/dec-inv.py`): 1,763 kernels inside
the step annotation (1,990 launches per step with the sampler), 13.3 ms GPU busy, 24.9 ms wall under the profiler
(16.7 unprofiled). The glue that was not ours, per step:

| launches | what | GPU ms |
|---:|---|---:|
| 48 | `topkGating` (vLLM's softmax top-k) | 0.36 |
| 96 | `moe_align_block_size` + `count_and_sort_expert_tokens` | 0.19 |
| 48 | `moe_sum` | 0.13 |
| 48 | finalize copy (the modular kernel does not alias its output on ROCm without aiter) | 0.10 |
| 48 | the runner's `shared_output + fused_output` add | 0.11 |
| 10 | RCCL all-gathers (MTP head `gather_output`, logits) | 0.71 |

The eager trace with Python stacks (`~/dec-attr.py`) attributed the rest: QSA gate `sigmoid * mul` (24), the PLE
layer's int64 index arithmetic (~50), GDN metadata (~21, outside the graph), `causal_conv1d_update` (36).

## What was built

1. **One-launch routing** (`kernels/r9k_moe_route.hip`, `moe/route.py`): softmax, top-k (ties to the lower index),
   renormalisation and the `moe_align_block_size` tables in one kernel. A grid of blocks does the rows (one wave
   each, counts by global atomics); the last block to arrive builds the tables and zeroes the scratch. Installed as
   a `FusedTopKRouter` replacement on every MoE runner; the tables reach the experts' `apply` through the
   `RoutedExperts` object. Ids identical to stock on every row of the unit test (M 1..300), weights within fp32
   rounding, tables validated, NaN / inf / all-equal rows and padded lanes covered. `R9K_MOE_ROUTE=stock`.
2. **Fused top-k sum with the shared expert folded in, and the output alias** (`kernels/r9k_moe_sum.hip`,
   `moe/fold.py`): one kernel sums a token's expert rows and adds the shared expert's output with one bf16
   rounding (bit-equal to `moe_sum` without the shared term); the modular kernel's output alias is taken on ROCm
   (no finalize copy); the runner's forward is rebound to never add the shared output. Both runner patches are in
   `compat/gate.py`. `R9K_MOE_FOLD=stock`, `R9K_FUSED_SUM=0`.
3. **One-shot P2P all-gather** (`r9k_ag_oneshot_nrank` in `kernels/r9k_ar.hip`, `R9kAllReduceN.all_gather`,
   `R9kCommunicator.all_gather`): the push half of the one-shot all-reduce on its own IPC pool, writing vLLM's
   concat layout for any dim directly. Four ranks, bit-exact incl. graph replay: 7.8 us vs RCCL 57 at [4, 640],
   40 vs 299 at [64, 640]; RCCL wins at the logits size (496 KB: 213 vs 190), so the cap stays at
   `R9K_AG_MAX_KB=128` and only the MTP head's gathers move. `R9K_AG=0`.

## What each cost to get right in serving

Every one of the three passed its unit tests and broke in its first serving launch, each for a reason the unit
test could not see:

- **Routing: a GPU page fault at the 24-token cudagraph capture, on one rank, twice.** vLLM feeds garbage
  (NaN rows) during dummy and warm-up steps; every compare in the selection came out false, the index stayed at its
  sentinel and the count update wrote past the table. The first theory (buffers allocated inside vLLM's memory-
  profiling pass) was wrong and cost a cycle. Rule: a kernel that derives an index from input values must be total
  over NaN and inf.
- **Routing: level with stock at 8 rows, 5x slower at 64.** The first version was one workgroup; in serving it
  showed as -5% at 16 concurrent. The grid version is flat at 17-20 us against stock's 17 for its four launches.
- **Fold: three bugs.** Storing the runner on its own child module made a cycle that vLLM's tied-weight scan
  recursed on at load; `SharedExperts.output` consumes its slot and the shared expert runs on an aux stream at
  decode (peek the slot, wait on its event); and a per-call flag read in the rebound forward was baked in at
  torch.compile trace time -- the graph kept the add while the kernel added too, and the drafter's acceptance fell
  to 1.3 tokens a step. Rule: anything read in a model forward is a trace-time constant; per-call decisions belong
  inside the opaque custom ops.
- **A/B hygiene:** the launcher refuses to start while any container exists, so a unit test on card 4 silently
  skipped a whole chain of runs.

## What it bought

| Flash-Next TP4, MTP-3, 512-token essay (`~/tp4tune.sh`) | ms/step | probe: decode / c8 / c16 / 8k prefill |
|---|---:|---|
| stock, three runs | 16.70, 16.70, 16.67 | 196 / 566-586 / 889-899 / 6,717-6,771 |
| fused routing, single-block kernel (two runs) | 16.56, 16.58 | 193 / 585 / 843-845 / 6,801-6,811 |
| fused routing, grid kernel | 16.71 | 193 / 588 / 893 / 6,785 |
| fused routing, hybrid kernel (LDS path at one block) | 16.67 | 193 / 598 / 893 / 6,797 |
| + fold (fused sum + shared add + output alias), three runs | 16.56, 16.56, 16.56 | 196 / 547-558 / 883-889 / 6,835-6,866 |
| routing on, fold off (control on the same tree) | 16.68 | 192 / 555 / 898 / 6,767 |
| + all-gather (everything on), three runs | 16.28, 16.27, 16.25 | 198 / 549-561 / 892-930 / 6,822-6,854 |
| everything on but the all-gather (control on the same tree) | 16.58 | 195 / 546 / 889 / 6,847 |
| everything on, hybrid routing kernel (two runs) | 16.25, 16.25 | 198 / 560 / 892 / 6,793-6,874 |
| **all-gather alone**, routing and fold on stock (two runs) | 16.40, 16.41 | 200 / 570-578 / 900-904 / 6,798-6,819 |

Strict sanity (conc 1 x 20, conc 17 x 40) clean on every configuration. The quality gate on the full
configuration (everything on, `r9knodes-fn4`): GSM8K 1,319 no-think conc 1 **95.60%** (1,261) against the v0.2.4
four-card record's 95.68% -- paired 14 vs 13 discordant, McNemar p = 1.00, no detectable difference; outputs
identical on 12.8% of questions, which is the fused sum's single rounding changing bf16 bits in 48 layers (the
all-gather is exact and the routing ids are identical). HumanEval **160/164** (v0.2.4 record: 158). Strict sanity
after the evals 0 bad of 1,020 at conc 17. Against radiance 1.1.1 at four ranks (95.83%): p = 0.73.

**The honest number: the all-gather is worth 0.3 ms a step (1.8%) and 2% of decode throughput; the two launch
fusions together 0.15 ms (0.9%) in step time and nothing in decode throughput.** The step-time differences between
routing variants are of the size of the essay's acceptance swings (2.33 vs 2.52 tokens a step changes the work per
step), so they are the noise floor; the all-gather's gain stands at a higher acceptance than its control. The
2026-09-28 analysis measured ~5.6 us of gap per node under the profiler and ~1.6 us unprofiled, and reasoned that
fewer nodes would close the step toward the GPU time. In graph replay on this ROCm the dispatch of a tiny node
overlaps the previous node's execution almost entirely: removing 240 launches a step moved it 0.15 ms. The
profiler's gaps are the profiler's. What remains between 13.3 ms of GPU time and the 16.7 ms step is not launch
overhead to be fused away; it is the serial dependency chain of ~1,700 small kernels, each finishing before the
next starts, with the GPU idle at every tail. The lever for that is fewer, fatter kernels -- the GEMM families fused
with their neighbours (gate_up + silu + down in one MoE kernel, the hyper-connection down + up + mix in one) -- which
is a different project from this one.

**Defaults:** the all-gather is on (`R9K_AG=1`, cap 128 KB); the routing and the fold are in the tree but off
(`R9K_MOE_ROUTE=r9k`, `R9K_MOE_FOLD=r9k` turn them on): 0.15 ms does not buy a replaced router and two patches of
vLLM's runner on the production path. Both passed the gate with everything on, and a future fatter-kernel MoE step
would start from the routing kernel.

## Where things are

Commits on master (local): c03cb72 routing + fold, 87dfc33 all-gather, e108bfb / 4b189c5 / 1ed57d6 routing fixes,
b5019a3 / a0ebaac / 6ec8854 fold fixes. Test box: dev trees `~/r9700-build/repo-{dev,fold,ag}` (route; +fold; +ag),
runs in `~/tp4tune/<label>/`, logs `~/ab-*.log`, profiles `~/tprof-v025-decode`, tools `~/dec-inv.py`,
`~/dec-attr.py`, `~/agtest.sh`. Production copy `~/r9700-build/repo` untouched (v0.2.5).
