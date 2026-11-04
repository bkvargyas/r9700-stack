# The DSpark drafter Heisenbug (package A2 hunt, 2026-11-04; owner task 71)

## Symptom
In some launches the DeepSeek-V4.1 EXL3 server serves correct target text, but the DSpark drafter accepts nothing:
mean acceptance length (MAL) 1.00-1.12 instead of 1.7-2.0 (prose) / 4.1-4.5 (code). Throughput drops to ~45-55 tok/s.
Divergence and GSM8K stay normal, because speculation is lossless and only the drafter is wrong. This is the
"package A broke the drafter" result of 2026-11-03 (notes/exl3-plan.md, Phase 4 A): package A's kernel is not the cause.

## Reproduction (VM100, eight cards, launcher defaults NSEQ 4, graphs, adaptive verification)
- `git archive 3f2167a` (v0.4.0 + C + B + A; C's combine hook wired after `serialize_attention` as in clean-CBA.sh):
  MAL 1.01 in every launch (r1, e3c, p2, q1, q2, c2, g1-g4, t1c, e1 below).
- The same tree with v0.4.1's own expert tier (P1's kernel, cap 24) and only the r9k scratch reserved for 48 rows
  (`R9K_A2_RESERVE=48`, 47.5 MiB instead of ~24): MAL 1.05-1.12 (e2). The scratch size alone triggers it.
- Not reproduced: v0.4.1 + 3f2167a's four tier files (s1, s2, r2, e3b: MAL 1.69-1.89 / 4.2-4.4). D's allocations
  shift the layout there. Also not reproduced: tC + a 64 MiB pad allocated after the scratch (e1-pad, p1, c1, t3, d1:
  MAL 1.74-1.84), tC with EAGER=1 ADAPTIVE=false (f1: MAL 1.92), and tC with extra captured copies of the drafter FFN
  children (d2: MAL 1.75; the copies moved the graph pool).

## Where it is
Captured copies of every drafter module at depth <= 4, padded vs broken, same prompt, greedy (c1 / c2). The input to
`drafter.model.layers.0.ffn` is bit-identical (ffn_norm, mhc, attn_norm, ffn#in: rel 0). Its output is |y| 108.8 padded vs
2.87e6 broken (max |y| 6.3e4), both in the eager T = 5 call and in the T = 5 graph, and everything downstream inherits it.
Inside that FFN, the routed experts (the r9k tier) are correct in the broken server: captured inputs / outputs of all 12
drafter MoE calls re-run eagerly through r9k and through exllamav3's fused tier give rel 1.8-2.2e-3, i.e. bf16 rounding
(h1-gdbg). That leaves the shared expert, the router / top-k, or the combine inside stock DeepseekV4MoE / MoERunner at
the drafter's shapes (T = 5, 128 experts, top-3, 16 local, dense bf16 shared expert of 288 rows a rank).

## Excluded (launch -> result)
| hypothesis | launch | result |
|---|---|---|
| r9k expert-tier math | single card, 48 local main / 16 local drafter geometry, every T 1..48, server warm-up order (mainA, draftA) | f32 reference rms rel 1.7e-3 (bf16 out), all OK |
| r9k tier in the server | h1-gdbg: captured copies in graphs vs eager r9k vs fused | equal (rel 2e-3) |
| A's kernel | e2: P1's kernel + 48-row reserve | still broken |
| OOB write past the scratch | p1: 64 MiB 0xA5 sentinel 256 B after the scratch | 0 bytes changed at READY and after serving |
| plugin scratch reallocated / moved | p2 pointer audit (4,151 tensors), q1 allocation log | each scratch allocated once, before capture, never moved; no plugin tensor in a graph pool |
| arrival counters / seq arrays | e3c / e3b, q1 sentinel | decode / mhc / glue counters zero; ar8 _os_seq / _fseq / _ag_seq identical on 8 ranks |
| freed regular-pool holes reused | q2: every free block filled right after READY | still broken |
| corrupted weights | e3c vs e3b checksums; t1c vs t3 dequant timeline (91 dense tensors, dequant -> 347 capture points -> serving) | stable in both layouts; differ only in the last bits (layout-dependent dequant rounding) |
| adaptive verification | g4 ADAPTIVE=false | still broken |
| B's markov head | g1 R9K_DSV41_MARKOV=stock | still broken |
| B's FFN front / attention input | g2 ROUTE=stock ATTN_IN=stock | still broken |
| H's mhc seams | c4 R9K_DSV41_MHC=stock | still broken |
| C's one-kernel ar8 / ag8 | c5 R9K_AR8_ONESHOT=0 R9K_AG8_ONESHOT=0 | still broken |
| exllamav3 GEMV on main_proj (3-8 rows) | c3 R9K_EXL3_RECON_M=2 | still broken |
| shared-experts aux stream join | e1 VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 | still broken |
| combine hook | inactive in tC ("runner shape changed") | n/a |
| expandable segments | g3 ALLOC_CONF= | OOM at KVMEM 1.7 (untested; needs KVMEM <= 1.2) |

## Related: the mixed-16 warm-up fault
Eager + MEMPROBE launches of tC (t2, d3, b1) die with HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION right after "Warming up
DeepSeek V4 sparse MLA attention for mixed tokens=16". That is the same place as the v0.4.0 NSEQ-8 launch fault
(notes/exl3-ledger.md item 9). AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 gives only a queue error (no kernel name, no
traceback). A bounds check on the split-K decode wrapper's ragged indices (attn/dsv4_decode.py, b1) did not fire in the 16
calls before the fault. An aperture violation is a wild address: a garbage pointer or index, most likely from
uninitialised memory, the same class that fits the drafter symptom.

## Graph-pool-layout sensitivity
Any change to what the HIP-graph pools or the regular pool hold flips the symptom: scratch size, a pad, D's GEMV plans,
captured debug copies, MEMPROBE (it does not flip it), the plugin source (torch.compile cache key). Diagnostics must
be cheap in allocations, and every result must be checked against the launch's own MAL.

## Next steps
1. NaN-prefill: fill the drafter graphs' pool (and every torch.empty the drafter FFN path allocates) with NaN at capture;
   a NaN in drafter layers.0.ffn's output names a read-before-write.
2. AMD_LOG_LEVEL=3 in the faulting eager layout: the last dispatched kernel before the aperture fault.
3. The reviewer's index-buffer candidate: fill topk_indices_buffer with -1 at allocation, so unwritten rows cannot count
   as valid under (ex >= 0).sum(-1).
Guard meanwhile: bench/ds41/drafter_selftest.py after every launch (exit 1 below MAL 1.3 prose / 3.0 code).
Logs, scripts and diffs: VM100 ~/ds41/exl3/A2/ (srv1..srv16, *.server.log, *-tl.json, a2chain-*.pt in ~/exl3-tune).
