# Draft bug report: MES stops answering INVALIDATE_TLBS on Navi 48 (gfx1201) under vfio passthrough; one escalation to a GPU reset took the device off the bus and reset the host

**Where this goes (checked 2026-10-03):** the amdgpu kernel/firmware tracker is https://gitlab.freedesktop.org/drm/amd/-/issues
(not GitHub; ROCm's GitHub is user space). It already has our issue:

- **#5759** (2026-09-05, open) "amdgpu/mes12: MES(0) intermittently fails to respond to REMOVE_QUEUE on Navi 48
  (gfx1201), forcing MODE1 reset and VRAM loss" -- two R9700, vLLM TP2 with MXFP4 weights and K=7 speculative
  decoding (the same stack as ours), the same soft variant `MES(1) failed to respond to msg=INVALIDATE_TLBS` on both
  cards. One comment so far. **File ours as a comment there**, not as a new issue: it adds the passthrough angle
  (the MODE1 reset that the driver "correctly" falls back to is what loses the device and resets the host), the
  per-card spread over five cards, the clustering at model load / graph capture, and the firmware matrix.
- **#4815** (2025-12, open, 8 comments): the soft `INVALIDATE_TLBS` message on an RX 9070 XT since kernel 6.18,
  "does not seem to affect anything".
- **#5909** (2026-09-29): R9700 over Thunderbolt, MES stops after large host<->VRAM transfers, reproduced on MES
  **0x91 and 0x93** -- which tells us a newer GC 12.0.1 firmware exists: linux-firmware commit caf919cc
  (2026-09-11, "amdgpu: update GC 12.0.1 firmware": `gc_12_0_1_me/mec/pfp.bin` and a new unified
  `gc_12_0_1_uni_mes.bin`), shipped in the 20260916 release. Our guest has the 20260810 set (MES 0x91, no
  uni_mes file). Staged on VM100 in `~/fw-20260916/` for a test after the validation run (Brian's call: whole-guest
  reboot, reversible by restoring the 20260810 files).
- **#5125** (2026-03, open): pipe reset disabled on GFX11/12, so any MES hang forces a full MODE1 reset.

Posting needs a gitlab.freedesktop.org account; none is configured on the management VM.

---

**Comment for #5759 (paste as is; the long form below it is the standalone version if a new issue is preferred).**

Same symptom family on five Radeon AI PRO R9700 (Navi 48, `1002:7551`, `1849:5413`, VBIOS 113-APM107573-101) --
with one important difference: ours sit behind vfio-pci in a KVM guest (Debian 13, kernel 7.2.6, QEMU 11.0.3,
EPYC 7H12 host, PLX PEX 8747 switches, an emulated PCIe switch in the guest for p2p), and there the MODE1 reset
does not recover anything. Data points:

1. The soft variant `MES(1) failed to respond to msg=INVALIDATE_TLBS`: 131 occurrences 2026-09-17 to 2026-10-03,
   on every card (45 / 20 / 20 / 31 on the four long-serving cards, 0 so far on a fifth added on 10-02 that has had
   little load), in bursts during model weight loading and CUDA-graph capture under vLLM (ROCm 10 nightly). Not
   power or heat: two episodes caught at 32-75 W and 32-38 C; the rest at 210-225 W and 55-78 C. Also seen on an
   idle-ish card during another card's load.
2. The escalation, once (2026-10-03 02:21 UTC), on the card that is only ever used in four-card runs: three
   `INVALIDATE_TLBS` timeouts two seconds apart, then `MES(0) failed to respond to msg=REMOVE_QUEUE`, `SUSPEND`,
   "failed to suspend all gangs", "MES might be in unrecoverable state, issue a GPU reset", `GPU reset begin!.
   Source: 3`, "Dumping IP State" -- and eight seconds later `device lost from bus!` with SMU bus errors
   (`response:0xFFFFFFFF`). Under passthrough the device never came back; about two minutes later the host
   platform reset itself with "an uncorrected error caused a data fabric sync flood event" and a fatal MCE on one
   core (SMCA EX bank, code 0: watchdog timeout on a stalled transaction). The host BMC log shows no power event;
   AER afterwards shows only advisory correctable errors; the card re-enumerated normally after the power cycle.
   So on this platform the reset that #5759 calls a 1-2 s recovery costs the whole host.
3. Firmware: linux-firmware 20260810 set -- MES 0x91 (`gc_12_0_1_mes.bin`/`mes1.bin`, no `uni_mes`), SMC
   104.79.0, PSP SOS 0x003a1214, MEC 0x0d66, PFP 0x0c6c, ME 0x0c08, RLC 0x00be7da0, IMU 0x0c302b00, SDMA
   0x00798e96. The guest kernel has the Linux 7.0 TLB-fence rework. We are about to try the 2026-09-11 GC 12.0.1
   drop (`gc_12_0_1_uni_mes.bin`, MES 0x93 per #5909) and will report back.
4. A deliberate reproduction of the configuration that crashed (three vLLM servers launched at once across five
   cards, then concurrent stress) ran clean, and so did ~8 hours of serving since; the timeouts recur, the
   escalation has not.

Questions for the amdgpu side: (a) is the INVALIDATE_TLBS timeout a known gfx12 MES issue with a firmware fix in
0x93 or planned? (b) for passthrough guests, is `amdgpu.gpu_recovery=0` (keep the hung queues, let the VM be
rebooted, which does a working FLR) the recommended containment, or is there a knob to retry / lengthen the MES
timeout before declaring the MES unrecoverable? (c) could the emulated switch's lack of PCIe atomics (we run with
`amdgpu.pcie_gen_cap=0x001F001F amdgpu.pcie_lane_cap=0x003F003F`) add MES message latency, or is the trigger the
mapping churn alone? Full dmesg of the escalation boot, `amdgpu_firmware_info`, `lspci -vv` (host and guest), the VM
topology and the per-card history are available.

---

---

**Title:** gfx1201 (Radeon AI PRO R9700): recurring `MES(1) failed to respond to msg=INVALIDATE_TLBS` under
allocation-heavy compute; one escalation to REMOVE_QUEUE/SUSPEND failure and GPU reset left the device unreachable
("device lost from bus") and the host platform reset

## Brief summary

Five Radeon AI PRO R9700 (Navi 48, gfx1201, `1002:7551`, subsystem `1849:5413`, VBIOS 113-APM107573-101) in one
Linux KVM guest via vfio-pci log `amdgpu: MES(1) failed to respond to msg=INVALIDATE_TLBS` recurrently: 131 times
between 2026-09-17 and 2026-10-03, on every card, in bursts that coincide with heavy memory mapping activity
(model weight loading and CUDA-graph capture under vLLM / ROCm 10). Normally a single timeout and the driver carries
on. Once (2026-10-03 02:21 UTC) three consecutive timeouts two seconds apart were followed by `MES(0) failed to
respond to msg=REMOVE_QUEUE`, `SUSPEND`, "failed to suspend all gangs", and amdgpu issued a GPU reset. Eight
seconds later the device read back as all ones (`device lost from bus!`, SMU bus errors), and about two minutes
later the host (EPYC 7H12, passthrough host) hard-reset with the firmware reason "an uncorrected error caused a data
fabric sync flood event" and a fatal machine check (SMCA EX bank, code 0: watchdog timeout) -- i.e. a host core
stalled on a transaction to the dead device.

## Hardware

- GPUs: 5x ASRock Radeon AI PRO R9700 32 GB (Navi 48 rev c0, gfx1201). Power cap 210 W, GPU voltage offset
  -42 mV at the time of the last two episodes; 225 W, no offset, for the earlier ones. Junction temperatures
  26-78 C across all episodes (two were at idle power during a model load: 32-75 W, 32-38 C).
- Host: Gigabyte MZ22-G20, AMD EPYC 7H12, 440 GB RAM. GPUs behind three PLX PEX 8747 switches (two cards per
  switch on two of them). Host kernel 7.0.14-20-pve (Proxmox VE 9.2), QEMU 11.0.3-4, vfio-pci, `iommu=pt
  pcie_acs_override=downstream,multifunction pcie_aspm=off`. Resizable BAR 32 GB set on the host before VM start.
- Guest: Debian 13, kernel 7.2.6 (XanMod build of mainline), q35 + OVMF, the four original GPUs on an emulated
  PCIe switch (two x3130-upstream + xio3130-downstream pairs, for p2p DMA), the fifth on a plain root port.
  `amdgpu.pcie_gen_cap=0x001F001F amdgpu.pcie_lane_cap=0x003F003F` (the emulated switch does not carry PCIe
  atomics; without the overrides SMU init fails), `amdgpu.ppfeaturemask=0xffffffff`. `GPU_MAX_HW_QUEUES=1` in the
  compute containers.

## Software

- Guest firmware: linux-firmware 20260810 (Debian firmware-amd-graphics 20260810-1~bpo13+1): MES 0x91 (both
  pipes), SMC 104.79.0, PSP SOS 0x003a1214, MEC 0x0d66, PFP 0x0c6c, ME 0x0c08, RLC 0x00be7da0, IMU 0x0c302b00,
  SDMA 0x00798e96, VCN 0x0910d001, DMCUB 0x0a000c00.
- User space: ROCm 10.0 nightly (HIP 7.15), PyTorch 2.12+rocm10, vLLM (stock) with a plugin of HIP kernels.
  Workloads: LLM serving with speculative decoding; the timeouts cluster at model load and CUDA-graph capture.
- Upstream linux-firmware commits after 20260810 (through 2026-09-24) update MES for GC 11.7.0 only; no newer GC
  12.0.1 firmware exists to try. The guest kernel includes the TLB-fence rework from Linux 7.0 that fixed MES
  deadlocks on Strix Point.

## Logs

The escalation (guest, 2026-10-03, UTC; the device is HIP 3, host c5:00.0, chain B second card):

```
02:21:15 amdgpu 0000:08:00.0: MES(1) failed to respond to msg=INVALIDATE_TLBS
02:21:17 amdgpu 0000:08:00.0: MES(1) failed to respond to msg=INVALIDATE_TLBS
02:21:19 amdgpu 0000:08:00.0: MES(1) failed to respond to msg=INVALIDATE_TLBS
02:21:21 amdgpu 0000:08:00.0: MES(0) failed to respond to msg=REMOVE_QUEUE
02:21:23 amdgpu 0000:08:00.0: MES(0) failed to respond to msg=SUSPEND
02:21:23 amdgpu 0000:08:00.0: failed to suspend all gangs
02:21:23 amdgpu 0000:08:00.0: failed to suspend gangs from MES
02:21:23 amdgpu 0000:08:00.0: MES might be in unrecoverable state, issue a GPU reset
02:21:23 amdgpu 0000:08:00.0: failed to remove hardware queue from MES, doorbell=0x1202
02:21:23 amdgpu 0000:08:00.0: GPU reset begin!. Source:  3
02:21:23 amdgpu 0000:08:00.0: MES might be in unrecoverable state, issue a GPU reset
02:21:23 amdgpu 0000:08:00.0: Failed to evict queue 2
02:21:23 amdgpu 0000:08:00.0: Failed to evict process queues
02:21:23 amdgpu 0000:08:00.0: remove_all_kfd_queues_mes: Failed to remove queue 1 for dev 27745
02:21:23 amdgpu 0000:08:00.0: Dumping IP State
02:21:31 amdgpu 0000:08:00.0: device lost from bus!
02:21:31 amdgpu 0000:08:00.0: SMU: bus error for message: EnableSmuFeaturesLow(8) response:0xFFFFFFFF
02:21:31 amdgpu 0000:08:00.0: Failed to enable GFXCLK DS!
02:21:32 amdgpu 0000:08:00.0: device lost from bus!
02:21:32 amdgpu 0000:08:00.0: SMU: bus error for message: SetWorkloadMask(36) response:0xFFFFFFFF
02:21:32 amdgpu 0000:08:00.0: Failed to set workload mask 0x00000001
```

All user processes holding any GPU received `HW Exception by GPU node-4 ... reason :GPU Hang` and aborted. The
guest kept running; the host reset at about 02:24 with no kernel log of its own. Host firmware at the next boot:

```
x86/amd: Previous system reset reason [0x08000800]: an uncorrected error caused a data fabric sync flood event
BERT: Error records from previous boot: event severity: fatal; IA32/X64 processor error, Local APIC_ID 0x5,
  cache error, Processor Context Corrupt: true, Uncorrected: true
mce: CPU 5: Machine Check: 0 Bank 5: bea0000000000108  (IPID 0x500b000000000, SMCA EX, ext code 0)
```

The host BMC log has no power-supply or thermal event. Host-side AER on the chain afterwards shows only advisory
correctable errors. The card re-enumerated normally after the power cycle and has run since.

The recurring, non-escalating form (same day, during a Flash-Next model's CUDA-graph capture on four cards, cards
at 32-75 W and 32-38 C):

```
04:06:40 amdgpu 0000:08:00.0: MES(1) failed to respond to msg=INVALIDATE_TLBS
04:07:46 amdgpu 0000:03:00.0: MES(1) failed to respond to msg=INVALIDATE_TLBS
15:30:52 amdgpu 0000:03:00.0: MES(1) failed to respond to msg=INVALIDATE_TLBS
15:30:52 amdgpu 0000:03:00.0: MES(1) failed to respond to msg=INVALIDATE_TLBS
15:30:57 amdgpu 0000:03:00.0: MES(1) failed to respond to msg=INVALIDATE_TLBS
```

Counts per card since the four-card layout (2026-09-23 to 2026-10-03): 03:00.0 (HIP 0, in every workload) 45;
04:00.0 20; 07:00.0 20; 08:00.0 (only in four-card runs) 31; the fifth card (added 2026-10-02, light use) 0.
Per day: 1 to 28, tracking how much loading and capture happened. A deliberate reproduction of the configuration
that crashed (three servers launched at once across five cards, then concurrent stress) ran clean.

## Steps to reproduce

Not deterministic. Any session that loads large models and captures CUDA graphs repeatedly on these cards produces
a few timeouts per day; a vLLM launch of a ~250B MoE across four cards (weights ~120 GB streamed to VRAM, then
graph capture for ~20 batch sizes) is the most reliable trigger we have. The escalation was seen once in ~130
timeouts.

## Questions

1. Is the INVALIDATE_TLBS timeout on gfx12 a known MES firmware issue, and is a GC 12.0.1 MES update planned?
2. Under virtualization the GPU reset path is what turned a recoverable hang into a lost device and a host reset.
   Is there a recommended setting for passthrough guests -- `amdgpu.gpu_recovery=0`, a longer MES timeout, or a
   retry before declaring the MES unrecoverable?
3. Does the emulated PCIe switch (no atomics; `pcie_gen_cap`/`pcie_lane_cap` overrides) plausibly contribute to MES
   message latency, or is the trigger purely the mapping churn?

Attachments available on request: full guest dmesg of the escalation boot, `amdgpu_firmware_info`, `lspci -vv`
for the cards and the switches on host and guest, the VM device topology, and the per-card timeout history.
