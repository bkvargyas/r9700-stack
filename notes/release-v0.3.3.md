# r9700-stack v0.3.3

**The image pin moves to the 2026-10-12 vLLM ROCm 10 nightly: Flash-Next on four cards decodes a step in 14.3 ms
(v0.3.2: 15.0), 225 tok/s single-stream, with the same kernels, the same quality gate and the same KV cache.**
The code of v0.3.2 plus the pin, the October compatibility fixes it rides on (`notes/vllm-upgrade-2026-10.md`), and
round 4 of `notes/decode-step.md`.

## What changed for someone running it

- `docker/Dockerfile` builds on `vllm/vllm-openai-rocm:nightly-rocm100-fe4df3df...` (vLLM 0.31.1rc1.dev296,
  torch 2.12+rocm10.0) instead of the September nightly. Rebuild the image; on a host that needs the
  emulated-switch overlay, regenerate its patched pair for this image (`overlay/emulated-switch/patch-hostcall.sh`)
  and point `~/p2p-patched-nightly` at it.
- The upstream DFlash draft-model exclusion bug ([vllm#61003](https://github.com/vllm-project/vllm/issues/61003))
  is still present in this nightly; the plugin's alias workaround covers it (the 27B's drafter runs at full
  acceptance).
- `R9K_MOE_ROUTE=r9k` / `R9K_MOE_FOLD=r9k` (off since v0.3.0) do not start on this vLLM: the runner's forward
  changed shape. They stay off and were not re-ported (worth 0.13 ms a step on the old one).
- New in the tree, off: the fused MoE decode path (`R9K_MOE_FUSE=1`: silu(gate) * up in the gate_up GEMM's
  epilogue, the fp8 row quant inside the down GEMM's A-load; bit-identical outputs, `tests/test_moe_fused.py`).
  It measured the same step time as the three-launch path, so it ships switched off.
- A cold torch.compile cache on a new image makes the first launch's 8-stream numbers read 20% low (compile
  transients during the first requests of that batch shape); the second launch is representative. The validation
  below warms the cache first.

## Numbers (test box, 4x R9700, 210 W and -42 mV)

| Flash-Next TP4, warm cache | v0.3.2 (Sept nightly) | v0.3.3 (Oct 12 nightly) |
|---|---:|---:|
| decode step, 512-token essay | 15.0 ms | **14.3** |
| single-stream decode (probe) | 219 tok/s | 225 |
| BetterBench combined decode / 1 / 2 / 4 / 8 streams | 176.8 / 172 / 261 / 403 / 514 | **188.0 / 175 / 277 / 410 / 530** |
| 8 / 16 streams (concurrent bursts, draft counters) | 617-623 / 841 | 608-622 / 868 |
| 8k prefill | 6,729-6,753 | 6,723-6,780 |
| KV cache | 515,577 tokens | 513,088 |

## Validation record (`~/validate-033.sh` on the release tree, 2026-10-13, `~/validate-033.log`)

The chain warmed the image's compile cache with a throwaway four-card launch first. A quoting slip then skipped the
four-card phase, which ran afterwards on its own (`~/validate-033-fn4.sh`, `~/validate-033-fn4.log`); the 27B and
two-card phases, the comm tests and the gates ran in the chain.

**27B-NVFP4 TP2** (first launch of this configuration on the image: cold cache). Probe: 221.5 tok/s single-stream
(4.51 tokens a step: the DFlash2 drafter at full acceptance through the plugin's exclusion aliases), 526 / 641 at
8 / 16 streams, 3,899 prompt tok/s at 8k, KV 367,494 tokens. Soak 16 clients, 1,553 s: 290 ok, 0 errors, 4 refused
for exceeding the 32,768-token context (prompts to 32,753 tokens), 3,120 prompt tok/s. Strict sanity: 0 bad of
320 after long prompts, 0 of 900 / 720 / 960 at 9 / 12 / 16 clients. Server errors 0.

**MES timeouts** (`MES(1) failed to respond to msg=INVALIDATE_TLBS`, the KVM-passthrough issue of
`notes/mes-timeouts.md`): dmesg had two from 2026-10-09 and gained five on 2026-10-13 between 02:06 and 02:43 UTC,
all during the A/B chains on the *October 9* nightly image (container launches and teardowns on HIP 0/1), none
during this validation on the pinned October 12 image (the count stayed at seven through all of it). The runs they fell in completed with normal
numbers; the guest's `gpu_recovery=0` / `mes_log_enable=1` flags are what keep them from hanging the card.

**Flash-Next TP2 (experts in host RAM)** (first launch of this configuration on the image: cold cache). Probe:
124.7 tok/s single-stream (3.16 tokens a step), 165 / 163 at 8 / 16 streams, 2,511 prompt tok/s at 8k, KV 155,648
tokens (cold; v0.3.2 warm: 221,184). Soak 16 clients, 1,559 s: 244 ok, 0 errors, 4 refused for exceeding the
context, 2,430 prompt tok/s. Strict sanity: 0 bad of 320 after long prompts, 0 of 900 / 720 / 960 at 9 / 12 / 16
clients. Server errors 0.

**Comm tests.** 4-rank all-gather and all-reduce (one-shot + two-shot, graph replay interleaved) bit-exact, 2-rank
race test ALL OK.

**Kernel gates (single GPU, in the pinned image):** atiled_4bit, fold_mxfp4, prefill_4bit, tuned_cfgs, moe_mxfp4,
nvfp4, cache_moe, gemm_fp8, gdn_merge ALL OK; moe_route_r9k, moe_sum_r9k, fp8_prefill_r9k PASS (chain, 04:57 UTC);
the full set plus hc_f8, fp8_qa and moe_fused passed in the same image on 2026-10-13 02:30 (`~/gates-fe4d.log`).

**Flash-Next TP4** (warm cache, `~/validate-033-fn4.log`). Probe on launch: 225.0 tok/s single-stream (3.11 tokens
a step), 623 / 944 at 8 / 16 streams, 6,796 prompt tok/s at 8k, KV cache 518,481 tokens. Full 20-pass BetterBench
(`r9700-fn4-v033-210w`): **combined decode 188.0 tok/s** (v0.3.2: 176.8; radiance 1.4.0 on this box: 213.7),
update p99 15.0 ms; concurrency 1 / 2 / 4 / 8: **174.9 / 276.5 / 410.0 / 530.0** aggregate tok/s (v0.3.2:
172 / 261 / 403 / 514; radiance 1.4.0: 194 / 292 / 422 / 536); prefill 5,175 / 7,141 / 7,489 / 7,233 tok/s at
2k / 8k / 16k / 32k (v0.3.2: 5,189 / 7,076 / 7,388 / 7,137). Mixed-length soak, 16 clients, 1,508 s: 690 ok, 0
errors, 1 refused for exceeding the context, 6,688 prompt tok/s, VRAM flat at 30,788-30,824 of 32,624 MiB a card
throughout. Strict sanity: 0 bad of 320 after long prompts, 0 of 1,700 / 1,440 / 1,920 at 17 / 24 / 32 clients.
Server errors 0.

**27B TP2 again, warm cache** (`~/val033-27b-warm.log`): probe 221.6 tok/s single-stream (4.51 tokens a step),
527 / 658 at 8 / 16 streams, 3,901 prompt tok/s at 8k, KV 367,494; concurrent 384-token essay bursts 445-454 tok/s
at 8 streams (2.6 tokens a step on that prompt), 456 at 16. The 8-stream probe reads about 10% below the v0.3.2
record on the September pin (582-604) in both the cold and the warm launch, while single-stream, 16 streams and
prefill are level. Measured head to head on the same tree with warm caches (`~/val033-27b-cmp.log`, old / new /
old / new): the probe's 8-stream phase repeats it (583 / 526 / 583 / 526), but equal-prompt concurrent bursts of
384-token essays with the draft counters read level or better on the new pin: 8 streams 432-463 (old) vs 452-453
(new), 16 streams 442-448 vs 453, 2.55-2.67 tokens a step on both. The probe's 8-stream phase mixes prompt types
whose acceptance differs; its gap is not a throughput regression of the image. Single-stream the new pin is ahead
(221.6 vs 213.0 tok/s, 4.51 vs 4.32 tokens a step: the drafter's acceptance is higher on this vLLM).
