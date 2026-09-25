# Plan: compressed, topology-aware 4-rank all-reduce (prefill)

Written 2026-09-24 night. Status: measured and designed, NOT started. Brian's call whether to spend the 1-2 days.

## Why this is the next kernel

Flash-Next TP=4 prefill profiles (torch profiler, rank 0, `~/tprof-*`):

| prompt | ours: all-reduce / total GPU | Rob's image: all-reduce / total |
|---|---|---|
| 2,142 tokens | **387 ms / 671 ms** (222 RCCL calls, 1.73 ms each, 11 MB) | 116 / 351 (200 calls, 0.58 ms) |
| 12,722 tokens | 1,019 / 3,309 (RCCL, 1.84 ms avg) | 607 / 1,936 (1.2 ms) |

Everything else is within ~50 ms of Rob's at 2k. Full BetterBench prefill: 2,853 vs 5,279 tok/s at 2k, 4,637 vs
5,711 at 8k. Matching his per-call time would put 2k prefill near 4,500 and 8k near 5,500.

## What the fabric gives (`tuning/ar_push_bench.py`, 8 MB per destination, hipMemcpyAsync into IPC scratch)

- Any single push, same-switch or cross-switch: **13.5 GB/s** (x16 Gen3 link rate).
- Ring (each rank one peer, two cross-switch hops, one per direction): 13.1-13.4 GB/s per rank.
- Same-switch pairs simultaneously (0<->1, 2<->3): 13.5 each.
- **Cross-switch pairs simultaneously (0<->2, 1<->3): 6.7 GB/s each.** The two PEX 8747 switches meet through ONE
  x16 Gen3 path per direction (~13.5 GB/s total), shared by all cross traffic.
- All-to-all (3 destinations each): 7.5 GB/s per rank; staged one-per-round: 8.1. Both are uplink-bound.

Consequences:
- The exact two-shot (`r9k_ar_twoshot_nrank`) pushes 2 x 3/4 N per rank and 4 x 2 x N/4 across the uplink per
  direction = N per direction -> 20 MB: 1.5 ms of uplink time alone; measured 4.05 ms vs RCCL 2.97. Block count is
  irrelevant (16..256 blocks: 4.40..4.05 ms).
- RCCL's ring sends 1.5 N per rank and crosses the uplink once per direction: lower bound 1.5N/13.5 = 1.22 ms at
  11 MB; measured 1.73. Close to optimal for an EXACT reduction. Only compression wins.
- Rob's 0.58 ms at 11 MB is below the exact bound, so his wire is compressed (kernel name `twoshot_wide_ti8`;
  clav_ar docs: "layered Q8 wire, 0.625x fabric bytes").

## Design

Hierarchical 2x2, compressed on the wire, compression fused into the push kernels (no separate codec passes: the
2-rank compressed path spent ~0.4 ms per 15 MB in pack/reduce kernels, which is what made it lose to libr4d).

1. **Intra-pair reduce-scatter** (0<->1 and 2<->3 over local links, in parallel): each rank packs half of its
   input (the half its partner owns) into the WHT wire and pushes it; the partner unpacks, adds its own half.
   Now each rank holds the pair-sum of N/2.
2. **Cross reduce-scatter** (0<->2, 1<->3): each rank packs half of its pair-sum (N/4) and pushes across; the
   partner adds. Each rank now holds the full 4-way sum of N/4. Cross bytes per direction: 2 ranks x N/4 x c.
3. **Cross all-gather**: push the packed N/4 result back across (2 x N/4 x c per direction).
4. **Intra all-gather**: push the packed N/2 to the pair partner.

Bytes per rank on the wire with compression factor c (bf16 = 1): local link N x c, uplink per direction N x c / 2.
- 4-bit WHT (c = 0.27): ~0.45 ms of link time at 11 MB, ~0.85 at 21 MB.
- 6-bit WHT (c = 0.39): ~0.65 / ~1.25 ms.
Plus four handshakes and the codec math (in-kernel). Target: <= 0.7 ms at 11 MB (RCCL 1.73, Rob 0.58).

Every rank's output must be bit-identical: the all-gather phases carry PACKED data and every rank decodes the same
bytes for chunks it did not own; for its own chunk it must decode its own packed copy too (not use the fp32 sum), as
the 2-rank path does.

Quantization count: the input is quantized twice on the way to the sum (phases 1 and 2 each quantize a partial) and
the result once for the gather. The 2-rank path quantizes once. Rel error per 4-bit quantization ~0.1; expect
~1.7x the 2-rank path's perturbation. **Needs the paired eval at conc=1 with EVAL_THINK=1** before it can default;
6-bit is the fallback (still ~2x faster than RCCL).

## Codec kernel notes

- Reuse the wire format of `kernels/r9k_ar_wht.hip` (64-element groups, H64 rotation, B-bit codes + bf16 scale),
  so `r9k_wht_pack` / `r9k_wht_reduce` stay as reference implementations for the tests.
- Fused push+pack: one 64-thread pair of waves per group is what the existing codec does (LDS for the stride-32
  butterfly). For a fused kernel, hold 2 elements per lane in a wave32 so all six butterfly strides are wave
  shuffles and no LDS or block barrier is needed; write the 34/50-byte group straight to the peer's scratch.
- Receiver: unpack peer group + own packed group -> fp32 sum -> (phase 1/2) keep as bf16 partial for the next
  pack, or (phase 3/4) decode to bf16 output.
- Handshake: the release/acquire flags of `r9k_ar.hip` (drain=4/acq=2), per block per phase, sequence counter
  device-resident (cudagraph-safe), double-buffered scratch.
- Message sizes to test: prefill NBT=4096 -> 21 MB; 2k prompts -> 11 MB; chunk tails smaller. Messages below
  R9K_ARN_MAX_KB (512 KB) keep the exact one/two-shot; RCCL for anything the kernel does not accept.

## Tests / gates

- `tests/test_ar_nrank.py` style 4-rank test: bit-identical across ranks; error vs fp32 reference bounded like
  the 2-rank compressed test (`tests/test_ar_wht.py`); graph-replay safety with the exact kernels interleaved.
- Graph-timed latency at 1, 5, 11, 21 MB vs RCCL and vs the exact two-shot.
- Serving: `~/tp4tune.sh` probe (prefill 8k) then `bench/eval.py` paired run (q-ref vs new, conc=1, EVAL_THINK=1,
  800 questions) before any default change.
