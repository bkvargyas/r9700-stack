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
| 8 / 16 streams (concurrent bursts, draft counters) | 617-623 / 841 | 608-622 / 868 |
| 8k prefill | 6,729-6,753 | 6,723-6,780 |
| KV cache | 515,577 tokens | 513,088 |

## Validation record (`~/validate-033.sh` on the release tree, 2026-10-13, `~/validate-033.log`)

TBD.
