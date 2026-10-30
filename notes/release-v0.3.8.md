# v0.3.8 -- the eight-rank collective through host memory (DRAFT until tagged)

Same pin as v0.3.7 (`nightly-rocm100-e4a5b553`, vLLM 0.32 line, torch 2.13); one change, measured on 8x R9700 in VM100 (384 GiB
guest, 210 W / -42 mV, the -500 MHz offset on eight-card runs).

## Changed

- **Eight-rank all-reduce: the cross-switch stage exchanges through pinned host memory** (`comm/r9k_ar.py` `host=True`,
  `comm/r9k_ar8.py`; notes/glm53.md "The eight-rank collective through host memory"). The four even ranks pushed peer writes
  across four root complexes, where the stores stall past ~100 KB in flight: 72 us for the 32 KB message of a four-row MTP-3
  verify step. The same kernels over pools mapped from `/dev/shm` and pinned with `hipHostRegister` take 47 us (64 KB: 61 against
  RCCL's 107). Messages of 12-96 KB take the host stage whole, 1-row messages keep the peer path, 128 KB and up stay on RCCL; the
  switch-local pair stage keeps its one-shot to 64 KB; the logits all-gather goes through the host stage too. Default on;
  `R9K_AR8_CROSS_HOST=0` restores the peer-write stage. TP2 / TP4 are unchanged (world-size-8 code).
- **Fixed on the way:** every size rule of the eight-rank object reads group-wide state agreed at startup (an even rank on ar8
  while its odd partner ran RCCL deadlocked the first forward).

## Validation (VM100, `~/ar8-gate.sh` for GLM, `~/gate-038.sh` for the Qwen configurations, 2026-10-30)

**GLM-5.3-Flash TP8 (32k production recipe):** probe **100.0** tok/s mean (105 / 84 / 108 / 103 by type), 2.86 tokens a step,
129 / 156 at 8 / 16 streams, prefill 3,148; BetterBench combined **103.3** (v0.3.7: 91.4), update p99 28.5 ms (32.1), TTFT 258 ms,
1 / 2 / 4 / 8 streams **94.6 / 127.3 / 161.1 / 160.4** (86.5 / 116.5 / 157.9 / 157.5), prompt processing 3,361 / 3,289 / 3,140 /
3,102 at 2k / 8k / 16k / 32k (unchanged). Strict sanity (thinking on) 0 bad of 160 / 300 / 320 / 200. GSM8K 1,319 **95.98%**
(1,266), paired against the v0.3.7 production run: 13 / 13 discordant, McNemar p = 1.0. HumanEval 152 (151). Soak (16 clients,
200-30k tokens, 997 s): 158 requests, 0 errors, 2,453 prompt tok/s. 0 server errors; VRAM 32.1 GiB a card after.

**Qwen3.8-27B-NVFP4 TP2 (cards 0,1):** probe 217.6 (code 241 / prose 105 / json 273 / math 252), 4.48 tokens a step, 516 / 643 at
8 / 16 streams, prefill 3,877. BetterBench combined **198.2** (194.7), 1 / 2 / 4 / 8 streams 176.9 / 296.4 / 437.2 / 595.5 (177.6
/ 302.8 / 441.6 / 593.4), prompt processing 3.7-3.95k. Strict sanity 0 bad of 160 / 1,020 / 1,280. Soak 183 requests in 976 s,
0 errors (1 over the 32k limit rejected, by design), 3,072 prompt tok/s. 0 server errors.

**Qwen3.8-Flash-Next TP4 (cards 0-3):** KV cache 477,002 tokens. Probe 221.4 (222 / 160 / 262 / 242), 3.09 tokens a step, 470 /
924 at 8 / 16 streams, prefill 6,743. BetterBench combined **184.6** (182.6), 1 / 2 / 4 / 8 streams 174.0 / 276.3 / 402.4 / 535.6
(173.0 / 275.9 / 400.7 / 542.6), prompt processing 4.96 / 6.95 / 7.31 / 7.07k. Strict sanity 0 bad of 160 / 1,020 / 1,280. Soak
462 requests in 906 s, 0 errors, 6,457 prompt tok/s. 0 server errors.

## Known limits

Unchanged from v0.3.7: the GLM 262k launch retrieves correctly to ~75k prompt tokens on this nightly (notes/release-v0.3.7.md).
The remaining gap to radiance's 24 ms GLM step is compute: the bf16 skinny projections (7.4 ms a step), MoE at four rows (4.9),
the hyper-connection glue (3.9) and 3-6 ms of CPU-side launch gaps (notes/glm53.md).
