# Root-port boxes: every card on its own CPU root port, no PCIe switch (2026-11-06/07)

Brian: "So maybe we fix it for a non PLX switch setup". Everything before this note was measured on PLX boxes (two
R9700s behind each PEX 8747, switch-local P2P with ACS redirect off, `host/README.md`). This is the first box without
a switch, and the collectives behave differently enough there that the fixed cutoffs cost 2x at 16 streams.

## The box

`gpusrv1`: ASRock Rack ROMED8-2T, EPYC 7532 (32 cores, Zen 2, one socket), 251 GB, Proxmox 9.2 (kernel 7.0.14-pve).
Four R9700s, each alone on a CPU root port (`00:01.1`, `80:01.1`, `80:03.1`, `c0:01.1`), Gen4 x16, 32 GB BAR from the
firmware (no barfix needed), each in its own IOMMU group. VM 101: q35 + OVMF, `cpu: host`, 32 cores, 184 GB, the four
cards passed as function `.0` only (`hostpci0: 0000:03:00.0,pcie=1,rombar=0`, ...), args
`-global q35-pcihost.pci-hole64-size=512G -fw_cfg name=opt/ovmf/X-PciMmio64Mb,string=262144`. Guest: Debian 13 with the
backports 7.2.11 kernel rebuilt with `CONFIG_HSA_AMD_P2P=y` (Debian ships it off: no KFD peer links, and the plugin's
IPC all-reduce fails with `invalid device pointer`), `amdgpu.mes_log_enable=1 amdgpu.gpu_recovery=0
amdgpu.ppfeaturemask=0xfff7ffff`, 210 W / -42 mV / a fan curve from a boot service.

Two passthrough details that matter on this layout and not on the PLX box:

- **Pass the GPU function alone.** Proxmox passes `03:00` as a multifunction device (GPU + HDMI audio); QEMU >= 8.1
  then does not advertise AtomicOps on the emulated root port (it only does for single-function devices), and amdgpu
  logs `PCIE atomic ops is not supported`. With `.0` alone the root port shows `AtomicOpsCap: 32bit+ 64bit+` and the
  `emulated-switch` overlay (hostcall-free RCCL, patched extensions) is not needed: run with `OVERLAYS=`.
- **No guestplace, no barfix.** The cards are not behind switches, so there is nothing to keep switch-local; every
  peer transfer goes through the root complex and the host IOMMU.

## What the fabric does

Bulk P2P is fast: `host/p2pbidir.py` 24 GB/s each way, 44.6 GB/s both ways (PLX box: 13 / 25.2); 4-rank RCCL
all-reduce busbw 19.4 GB/s at 256 MB (PLX box: 10.0). Small P2P writes from inside kernels are slow:
`tests/test_ar_nrank.py`, graph-timed, us:

| message | RCCL | one-shot | two-shot | PLX box: RCCL / one-shot / two-shot |
|---|---:|---:|---:|---|
| 5 KB (1 token) | 40.1 | **6.4** | 19.4 | 59.2 / 6.8 / 9.7 |
| 20 KB (4 tokens, MTP-3 decode) | 36.2 | **18.2** | 24.0 | 72.6 / 13.4 / 11.3 |
| 80 KB | **35.7** | 69.8 | 92.8 | 71.2 / 37.2 / 24.2 |
| 160 KB | **40.4** | - | 216.4 | |
| 320 KB | **53.4** | - | 466.1 | 102.0 / 130.4 / 73.2 |

The handshake is as cheap as on the PLX box (5 KB one-shot 6.4 vs 6.8 us), but data moved by kernel stores crosses the
root complex and the IOMMU at a fraction of the link rate, so two-shot (which reads peers) is the slowest path at every
size and RCCL wins from ~40 KB. The fixed cutoffs (one-shot <= 16 KB, two-shot <= 512 KB) therefore put single-stream
decode on a 24 us path instead of 18, and 8/16-stream decode (160-320 KB a call) on 216-466 us instead of 40-53.

The r9k all-gather has the same problem: 9 calls a step at ~100 us (0.9 ms of a 16.5 ms step).

## The step, compared with the PLX ledger

Torch profiler, CUDA activity only (`PROF=1 PROFACT=CUDA`), rank 0, Flash-Next TP4 MTP-3, single-stream 512-token
essay, 208 steps, with one-shot <= 40 KB and RCCL above (hand-set):

| ms a step | PLX box (decode-step.md, v0.4.x) | root-port box |
|---|---:|---:|
| MoE expert GEMMs | 2.78 | 2.76 |
| hc fp8 down/up + router | 2.32 | 2.53 |
| dense fp8 block GEMMs | 1.36 | 1.34 |
| GDN verify / topk | 0.53 / 0.37 | 0.53 / 0.37 |
| **all-reduce** | **1.27** (98 x 12.8 us two-shot) | **3.10** (107 x 28.8 us one-shot) |
| **all-gather** | - | **0.91** (9 x 100 us) |
| kernel time | ~12.0 | 13.86 |

Compute matches to the noise; the whole gap is the collectives. Kernel preemption (lazy vs full, at runtime) and the
voltage offset changed the step by <= 0.1 ms.

## The change: calibrate the cutoffs at startup (`comm/calibrate.py`)

`R9kCommunicator` now times one-shot / two-shot / RCCL (all-reduce, 5-480 KB) and r9k / RCCL (all-gather, 2-128 KB) in
HIP graphs on every rank at TP4, takes the slowest rank's time, and keeps the fastest exact path per size: one-shot up
to the end of its leading run of wins, two-shot over the run right after it, RCCL above. The compressed all-reduce
(`ar4`) keeps its validated range (above the old 512 KB two-shot ceiling, `R9kAllReduceN.ar4_min`), so a smaller exact
range sends the sizes in between to RCCL, not to the lossy path. Knobs: `R9K_ARN_CALIBRATE=0` restores the fixed
cutoffs; an explicit `R9K_ARN_1S_KB` / `R9K_ARN_MAX_KB` or `R9K_AG` / `R9K_AG_MAX_KB` is respected as given. TP8
(`comm/r9k_ar8.py`) and TP2 are untouched.

### What it picked on gpusrv1, and the gate (2026-11-07, `~/gate-cal.sh`, no hand-set cutoffs)

Calibration, 2.6 s at launch (us per call, slowest rank):

    all-reduce   rccl/one/two:  5K 40/9/16  10K 46/10/18  20K 40/23/27  40K 38/69/46  60K 41/107/71  80K 39/162/95
                                120K 42/264/165  160K 48/-/220  240K 51/-/322  320K 60/-/432  480K 71/-/666
                 -> one-shot <= 20 KiB, two-shot off, RCCL above, compressed from 512 KiB
    all-gather   rccl/r9k:      2K 32/15  8K 27/10  32K 34/50  64K 33/121  128K 49/277
                 -> r9k all-gather <= 8 KiB, RCCL above

Better than the hand-set trial (one-shot <= 40 KiB, r9k all-gather off): 40 KB is already an RCCL size here, and the
r9k all-gather is 2-3x faster than RCCL below 8 KB.

| Flash-Next TP4, MTP-3, 210 W / -42 mV, prefix caching on | fixed cutoffs | hand-set | **calibrated** | PLX box (v0.4.2 gate) |
|---|---:|---:|---:|---:|
| step (single stream) | 17.3 ms | 15.9 | **15.9** | ~14.4 |
| probe mean (code / prose / json / math) | 191 | 205 | **207.0** (217 / 140 / 245 / 227) | 222 (218 / 160 / 262 / 249) |
| 4 / 8 / 16 streams (`bench/bench.py`, 3 runs) | 355 / 413 / 466 | 405 / 443 / 910 | **461-465 / 703-742 / 995-1,015** | - / 466 / 909 |
| prefill 8k / 31k / 124k (`bench/prefill.py`) | 7.9k / - / - | 8.4k / 7.8k / 6.7k | **8.2k / 7.9k / 6.9k** | 6.6k (8k) |
| session TTFT, ~61-66k tokens, turns 1-3 | - | 0.56-0.58 s | **0.57-0.58 s** | - |
| KV cache | | | 592,236 tokens | 479,076 |

`tests/test_ar_nrank.py` correctness PASS (4 ranks) with the 128 KiB one-shot pool; strict sanity 0 bad of 80 at 16
streams; 10-minute soak, 8 clients, 19-34k-token prompts: 165 ok, 0 errors, 7,209 prompt tok/s; 0 server errors.
(The first `bench.py` pass of the launch read 298 / 742 / 903; the three repeats above did not reproduce the 298.)

## Also on this box

- **Prefix caching for agent workloads.** `serve/flashnext.sh` defaults to `PREFIX_CACHE=0` (single-prompt TTFT,
  2025-09-25). For Claude Code, which resends the whole conversation every turn, turning it on is the bigger lever by
  far: a ~61-70k-token session (`~/turns.py`, ~1.8k new tokens a turn) went from 8.2-9.3 s TTFT a turn to 0.55-0.88 s
  (78-84% of prompt tokens from cache). Cost: short cold prompts 0.30 -> 0.45 s (2k tokens), KV 577k -> 547k tokens;
  decode, multi-stream and 8k+ prefill unchanged. Extending `CGSIZES` to 416 (the attention block in align mode) did
  not remove the short-prompt floor and cost 12% of the KV cache.
- **Prefill is faster than on the PLX box**: 8.3-8.4k tok/s at 8k (unique random prompts, `bench/prefill.py`), 7.8k
  at 31k, 6.7-7.0k at 124k, against 6.6-7.5k at 8k there: twice the all-reduce bandwidth.
- **MTP head conversion OOM at TP2 with offloaded experts** (`models/qwen4_exp.py`): the bf16 -> MXFP4 search ran
  2^18 rows a chunk (2.5-5 GiB of scratch) with 1.5 GiB free; 2^15 rows, same bits (branch `fix/mtp-convert-chunk`).
- **MES firmware 0x93** (firmware-amd-graphics 20260916): 7 soft `INVALIDATE_TLBS` timeouts over ~10 launches with
  `mes_log_enable=1`, all during load / graph capture, none escalated. mes-timeouts.md only validated the flag on 0x91
  and 0x8b.

## Open

- Why kernel-issued P2P stores are slow here while DMA runs at link rate: the host IOMMU translating every peer write
  (4 KB IOVA pages over 32 GB BARs?) or the EPYC root complex itself. The test that tells them apart: the same
  `test_ar_nrank.py` on the host with amdgpu instead of vfio-pci. If it is the IOMMU, the remaining ~1.5 ms a step is
  recoverable in the VFIO mapping; if it is the root complex, only a switch gets it back.
