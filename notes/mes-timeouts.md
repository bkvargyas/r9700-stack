# R9700 (RDNA4) under KVM passthrough: MES timeouts, one host crash, and the two kernel parameters that stop it

*2026-10-04. Status: workaround in use on both of our boxes; reported to AMD on
[drm/amd issue 5759](https://gitlab.freedesktop.org/drm/amd/-/issues/5759#note_3693873).*

**If you run Radeon AI PRO R9700 (Navi 48, gfx1201) cards inside a VM and your guest kernel log shows
`amdgpu ...: MES(1) failed to respond to msg=INVALIDATE_TLBS`, add these to the guest's kernel command line:**

```
amdgpu.mes_log_enable=1 amdgpu.gpu_recovery=0
```

The first one made the timeouts disappear in our testing (0 in 60 heavy launches, 36 of them followed by sanity
checks and a mixed-length soak under 16 clients, against about 1 per launch without it). The second one stops the driver from resetting a hung card, because under passthrough that reset takes
the card off the bus and, on our EPYC host, resets the whole host. Neither costs measurable performance.

## What it looks like

The soft form is a single kernel line, and the driver carries on:

```
amdgpu 0000:08:00.0: MES(1) failed to respond to msg=INVALIDATE_TLBS
```

We logged about 150 of them between 2026-09-17 and 2026-10-03 on five cards, on every card, always during model
weight loading or CUDA-graph capture -- the phases where the driver is writing GPU page tables as fast as it can and
asking the MES firmware to invalidate TLBs for each change -- and never during serving. The MES answers within its
2,100 ms deadline almost always; these are the misses.

Once it did not stop at one. Three misses two seconds apart, then the firmware also failed to answer `REMOVE_QUEUE`
and `SUSPEND`, and amdgpu did what it is designed to do:

```
amdgpu 0000:08:00.0: MES might be in unrecoverable state, issue a GPU reset
amdgpu 0000:08:00.0: GPU reset begin!. Source:  3
amdgpu 0000:08:00.0: device lost from bus!
amdgpu 0000:08:00.0: SMU: bus error for message: EnableSmuFeaturesLow(8) response:0xFFFFFFFF
```

On bare metal that MODE1 reset recovers the card in a second or two (that is how issue 5759's reporter sees it). Behind
vfio-pci the card never came back, every process holding a GPU aborted, and two minutes later the host hard-reset
with the firmware reason "an uncorrected error caused a data fabric sync flood event" and a machine check on the core
that was stalled waiting on the dead device. The BMC log showed no power or thermal event. The card re-enumerated fine
after the power cycle.

## What we ruled out

Six launches of our heaviest configuration (a ~250B MoE across four cards, which loads ~30 GB per card and captures
graphs for twenty batch sizes) per condition, same box, same day, kernel log watched:

| condition | launches | MES timeouts |
|---|--:|--:|
| MES firmware 0x91 (linux-firmware 20260810, the Debian package), defaults | 4 | 5 |
| MES firmware 0x93 (20260916), defaults | 6 | 16 |
| MES firmware 0x8b (20260622, our production box's version), defaults | 6 | 6 |
| 0x8b, `amdgpu.vm_update_mode=3` (CPU writes page tables) + `mes_log_enable=1` | 6 | 0 |
| 0x8b, `amdgpu.ras_enable=0` + `mes_log_enable=1` | 6 | 0 |
| 0x8b, `mes_log_enable=1` alone | 6 | 0 |
| 0x8b, defaults again (negative control) | 6 | 6 |
| 0x91, `mes_log_enable=1` + `gpu_recovery=0` (what we run now) | 6 | 0 |

Not the firmware version (three generations behave the same, so a firmware update is not the fix and a downgrade
is not either). Not power or heat (two episodes were caught with the cards at 32-75 W and 32-38 C). Not the
page-table update path or the RAS layer on their own: both runs that included them were clean, but so was the run
with only `mes_log_enable=1`, and the negative control right after went back to six. Not a bad card: all four
long-serving cards logged them, in rough proportion to how often each is loaded.

## Why `mes_log_enable=1` works, as far as we can tell

In `mes_v12_0.c` the flag changes exactly one thing that reaches the firmware: the `SET_HW_RESOURCES` packet sent at
MES startup gets `enable_mes_event_int_logging = 1` and the address of a small event-log buffer. The message timeout
and the driver's polling are identical either way. So the firmware services messages on pipe 1 differently when it is
logging, which looks like a completion-signalling race inside the MES that the logging path happens to cover. That
is AMD's to confirm; the bug report asks.

## How to apply it

Debian/Ubuntu guest with GRUB (this is what both of our VMs run):

```bash
sudo sed -i -E 's/^(GRUB_CMDLINE_LINUX="[^"]*)"/\1 amdgpu.mes_log_enable=1 amdgpu.gpu_recovery=0"/' /etc/default/grub
sudo update-grub && sudo reboot
# verify
grep -oE 'amdgpu\.(mes_log_enable|gpu_recovery)=[^ ]+' /proc/cmdline
cat /sys/module/amdgpu/parameters/mes_log_enable /sys/module/amdgpu/parameters/gpu_recovery   # 1 and 0
```

Keep watching the kernel log afterwards; a timeout is `journalctl -k | grep 'MES.*failed to respond'`. If one ever
escalates with `gpu_recovery=0` in place, the affected card's queues stay hung and the processes using it die, but
the host and the other cards keep going; a reboot of the guest recovers the card through the function-level reset
that every VM start already performs.

## What `gpu_recovery=0` trades

With recovery on, a hung MES leads to a MODE1 reset that would save the card on bare metal and that, in our one
observation, lost it and the host under passthrough. With recovery off you lose the chance of a clean in-place
recovery in exchange for never taking the host down. For a serving box behind a hypervisor that is the right trade;
on bare metal it is not obviously so.

## Related reports

- [drm/amd #5759](https://gitlab.freedesktop.org/drm/amd/-/issues/5759): the same stack (two R9700, vLLM,
  speculative decoding), MODE1 resets on bare metal, the soft variant on both cards. Our data is
  [this comment](https://gitlab.freedesktop.org/drm/amd/-/issues/5759#note_3693873).
- [drm/amd #4815](https://gitlab.freedesktop.org/drm/amd/-/issues/4815): the soft variant on an RX 9070 XT since
  kernel 6.18.
- [drm/amd #5909](https://gitlab.freedesktop.org/drm/amd/-/issues/5909): an R9700 whose MES stops after large
  host-to-VRAM transfers, on MES 0x91 and 0x93.
- [drm/amd #5125](https://gitlab.freedesktop.org/drm/amd/-/issues/5125): pipe reset is disabled on RDNA 3 and 4,
  so any MES hang goes straight to a full reset.

The investigation in full, including the day-by-day counts and the experiments that led here, is in
[PROGRESS.md](../PROGRESS.md) under 2026-10-03 and 2026-10-04; the report as filed is
[bug-mes-invalidate-tlbs.md](bug-mes-invalidate-tlbs.md).
