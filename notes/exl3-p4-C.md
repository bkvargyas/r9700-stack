# Phase 4 package C -- collectives (2026-11-03)

Eight R9700s on four PCIe switches, TP8, v041 tree. Every number below is us per call under HIP graphs, best of 3, max over the
eight ranks, quiet box, -500 MHz (tests/test_ar8.py on VM100, logs `~/ds41/exl3/C4/{ag1,t1,t2,t4,t5}.raw`).

## What and why

1. **One-kernel eight-rank all-gather** (`kernels/r9k_ar.hip` `r9k_ag8_os`, `R9kAllReduce8._ag_oneshot`). The ledger's largest single
   item (notes/exl3-ledger.md 1b: 4.05 ms a default step) was v0.4.0's fallback to the FLAT eight-rank one-shot all-gather whenever a
   gather's 8x size exceeded the pair path's R9K_AG_MAX_KB (128) but the piece did not: every rank stores its piece into all seven
   peers' VRAM, eight senders across four root complexes -- the fabric cliff. The new kernel crosses no root complex: own piece ->
   own pinned host pool + the partner (P2P); the three quartet peers' pieces from host memory -> out + the partner; the partner
   quartet's four pieces from our pair pool. Pure copies (bits = inputs). And the stop-gap: the flat 8-rank kernel is no longer used
   for gathers at all (`R9K_AG8_FLAT_KB` 0 -> RCCL), with or without the new kernel.
2. **One-kernel eight-rank all-reduce for messages <= 64 KiB** (`r9k_ar8_os`, `R9kAllReduce8._oneshot`): the pair reduce-scatter
   (P2P), both quartets' exchange of the halves (pinned host pools) and the pair all-gather (P2P) as three phases of one kernel with
   system-scope release / acquire flags. Same two roundings in the same order as the stage / split kernels: bit-identical to them.
3. **Fused MoE reduction** (`r9k_ar8_os_moe`, `R9kAllReduce8.all_reduce_moe`, hook `r9700_vllm/moe/exl3_combine.py`): the collective's
   first phase forms `bf16(bf16(sum over local picks in pick order of w * os) + shared)` from the r9k expert tier's per-slot f32
   outputs -- k_dn's per-token sum and stock's `shared_output + fused_output`, same operations, same order -- so the all-reduced bits
   equal the unfused path. Stock's order on this pin: router-scaled pick weights, runner scale 1.0, **no residual add before the
   all-reduce** (the hc seam mixes the residual after it). Saving: the bf16 add launch and its [T, 5120] write + read a layer
   (k_dn still writes y; dropping it is a flag in package A's kernel). Taken for T <= 6 (<= 64 KiB); larger steps keep the stock
   add + all-reduce inside the same op.

Shared machinery: `R9kPool` (r9k_ar.py; the IPC / pinned-host pools of R9kAllReduceN without its collective), dedicated pools and
counters per kernel, a fixed 16-block grid with one flag word per (source, block), the double-buffer reuse argument in the kernel
comment, all knobs agreed over the group at startup (`all_gather_object`; any disagreement turns the path off on every rank), and
`R9K_AR_SPIN_TIMEOUT_MS` (0 = off): a flag wait longer than that writes {rank, block, phase, wanted, seen, call} to host-mapped
memory, a watchdog thread prints it, the kernel traps ~0.3 s later.

## Tables

### All-gather (bf16, vLLM concat along dim -1)
| shape a rank | KiB a rank | RCCL | v0.4.0 path | one kernel |
|---|---:|---:|---:|---:|
| [1, 3200] | 6.2 | 76.2 | 34.3 (pair/cross) | **15.5** |
| [2, 3200] | 12.5 | 70.5 | 50.7 (pair/cross) | **20.5** |
| [3, 3200] | 18.8 | 74.4 | 405.1 (flat) | **25.9** |
| [4, 3200] | 25.0 | 78.0 | 728.0 (flat) | **32.1** |
| [5, 3200] | 31.2 | 82.2 | 881.7 (flat) | **39.4** |
| [6, 3200] (Engram, T = 6) | 37.5 | 85.6 | 1,033.2 (flat) | **47.5** |
| [1, 16160] (logits) | 31.6 | 81.0 | 880.5 (flat) | **40.6** |
| [2, 16160] | 63.1 | 102.8 | 2,527.8 (flat) | **71.2** |
| [4, 16160] | 126.2 | 143.8 | 3,919.4 (flat) | **129.3** |
| [6, 16160] | 189.4 | 186.7 | RCCL | 182.4 |
| [8, 16160] | 252.5 | 229.2 | RCCL | (233.9 at a 256 KiB cap: RCCL) |
| [24, 16160] | 757.5 | 576.9 | RCCL | RCCL |

Default cap `R9K_AG8_OS_KB` 192 (to 6 rows of logits); above it RCCL. A default step's gathers (Engram 2 x [T, 3200] + the trimmed
logits [<= 4, 16160]): ~4.05 ms -> ~0.2 ms.

### All-reduce (bf16)
| message | KiB | RCCL | v0.4.0 (stage kernels <= 32 KiB, fused split above) | one kernel (wpb 256) |
|---|---:|---:|---:|---:|
| 1 x 4096 | 8 | 107-112 | 21.0-22.3 | **14.6-16.1** |
| 1 x 5120 | 10 | 111 | 38.3-40.5 | **15.0-15.5** |
| 2 x 4096 | 16 | 114 | 33.9-34.5 | **16.5** |
| 2 x 5120 | 20 | 114 | 36.5-36.9 | **17.8-18.9** |
| 4 x 4096 | 32 | 107-118 | 45.8-46.0 | **22.1** |
| 4 x 5120 | 40 | 108-110 | 34.6-34.7 | **25.2** |
| 6 x 5120 | 60 | 106 | 37.6-38.1 | **33.8-34.2** |
| 8 x 4096 | 64 | 104-107 | 38.3-38.5 | **36.7-37.2** |

Targets were 8 KB <= 10, 32 KB <= 24, 60 KB <= 26: 32 KB met, 8 KB at 15-16 (one host round trip + two P2P hops is the floor of
this shape), 60 KB at 34. Words a block: wpb 256 beats 128 (32 KiB 22.3 vs 23.6, 60 KiB 34.5 vs 39.2; at wpb 128 the 60-64 KiB sizes
lost to the split kernel, which is why an earlier cut proposed a 48 KiB cap). Peer writes for phase B (`R9K_AR8_OS_P2P_KB`): 83-153
us at 8-16 KiB (eight cross-chain senders: the cliff) -- stays 0.

### Fused MoE combine (T x 5120, topk 6, random routing over 384 experts)
| T | add + v0.4.0 all-reduce | add + one-kernel all-reduce | fused |
|---|---:|---:|---:|
| 1 (10 KiB) | 40.2-47.9 | 17.9-18.3 | 17.7-18.5 |
| 4 (40 KiB) | 37.4 | 27.9 | 27.3 |
| 6 (60 KiB) | 40.1-40.2 | 38.1-38.4 | 37.7-37.8 |

The win is the one-kernel all-reduce; the fusion itself saves ~0.5 us and one launch a layer (as the ledger predicted: the add is
1.4 us). On the real server the post-MoE all-reduce also absorbs the expert-load imbalance between ranks (99 vs 46 us, ledger 1b).

## Correctness (tests/test_ar8.py, eight cards; check1 / check2 / t3 / t5: RESULT PASS)
- One kernel vs the stage kernels: bit-identical on 22 eager messages (16 B .. 64 KiB, bf16 / fp16 / fp32), within 2 ulp of the
  rank-ordered fp32 reference, a 17-call burst of mixed one-kernel / split / stage sizes with no host sync, graph replay of 8 chained
  mixed calls x 7; every size routes identically on every rank.
- Fused MoE vs unfused: bit-identical at every T in 1..48 for topk 6 / 384 experts and topk 3 / 128 (the drafter), non-local rows
  filled with NaN (never read), g2l sentinels -1 and >= E, graph replay chained with plain calls; T > 6 declined -> stock.
- All-gather: exact (= torch.cat) at [1..6, 3200], [1/2/4/6/8/24, 16160], [1/4, 19360], dim 0, 3-D, tiny; routing agreed;
  graph replay of gathers chained with all-reduces x 7.

## Knobs
`R9K_AR8_ONESHOT` (1), `R9K_AR8_OS_KB` (64), `R9K_AR8_OS_WPB` (256), `R9K_AR8_OS_NB` (16), `R9K_AR8_OS_NT` (0 = by size),
`R9K_AR8_OS_P2P_KB` (0; keep), `R9K_AG8_ONESHOT` (1), `R9K_AG8_OS_KB` (192), `R9K_AG8_OS_WPB` (256), `R9K_AG8_OS_NB` (16),
`R9K_AG8_FLAT_KB` (0 = never the flat 8-rank gather; 128 restores v0.4.0), `R9K_EXL3_MOE_AR` (1; the hook), `R9K_AR_SPIN_TIMEOUT_MS`
(0). All `R9K_AR8_OS_*` / `R9K_AG8_*` must be equal on every rank (enforced: a mismatch disables the path everywhere).

## What the gates must re-check
- **GLM-5.3-Flash TP8** uses the same object: its 1-row logits gather ([1, 19360] bf16, 38 KiB a rank) now takes the one-kernel
  all-gather (pair/cross/pair before), and every all-reduce <= 64 KiB (1-5 rows x 6144) the one-kernel all-reduce. Bits identical
  (gathers are copies; the all-reduce rounds as the stage kernels did), speed changes: run the GLM gate. Its larger logits gathers
  (R9K_AG_MAX_KB 2048 in its recipe) keep the pair/cross path above 192 KiB.
- Flash-Next TP4 and 27B TP2: no code path changed (R9kAllReduceN / R9kAllReduce untouched; R9kPool is new).
- DeepSeek: divergence.py / GSM8K as usual; check the log lines "ar8 one-kernel all-reduce for messages <= 64 KiB", "ag8 one-kernel
  all-gather for pieces <= 192 KiB" and (after install) "exl3 MoE combine -> ar8 one-kernel all-reduce on N runners" + "fused into
  the ar8 one-kernel all-reduce (first call ...)".

## Integration
1. `exl3_combine.install(self)` after `serialize_attention(self)` in both `__init__`s of `r9700_vllm/models/deepseek_v41_exl3.py`
   (`from ..moe import exl3_combine`); not done here (not this package's file).
2. Merge points against package A in the shared files: `kernels/r9k_ar.hip` -- C's two sections are appended at the end (from
   "one-kernel 8-rank all-reduce for small messages"); A's edits are inside `r9k_ar8_fused` / `ARF_*` above. `r9k_ar8.py` -- C: one
   call in `__init__` (`self._setup_oneshot(...)` after `_zbuf`), the first lines of `should_ag` / `all_gather` / `all_reduce`, and
   the methods from `os_on = False` to `all_reduce_moe`; A: `fused_lanes`, `_split`, the `r9k_ar8_fused_t` argtypes. `tests/test_ar8.py`
   -- C: the functions before `main()` (`_rccl_gather` .. `oneshot_bench`) and the OS / AG / OSBENCH / AGBENCH calls; A: PARTS=5.
   `kernels/build.sh` -- the six exports after `r9k_ar8_fused_lanes`.
