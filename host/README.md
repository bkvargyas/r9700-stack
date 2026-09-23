# Host setup: 32 GB BARs for R9700s behind PLX switches

The Proxmox host (`192.168.0.100`) has its R9700s behind PLX PEX 8747 switches, with up to two cards per switch.
Every card has to be passed to the VM with its full 32 GB BAR0. That's what lets amdgpu map all of VRAM, and it's
also what KFD/RCCL require before they allow peer-to-peer. This board's BIOS has no Resizable-BAR support, so all
of the setup below happens in Linux.

| File | Installed at | Role |
|---|---|---|
| `r9700-barfix.sh` | `/usr/local/sbin/r9700-barfix.sh` | Sets every card to 32 GB and re-enumerates each chain |
| `r9700-chainfix/` | `/usr/src/r9700-chainfix-1.2` (DKMS) | Kernel module that gives the second card on a switch its window |
| `gpu-reset.sh` | `/var/lib/vz/snippets/gpu-reset.sh` | Proxmox hookscript that runs barfix at VM pre-start / post-stop |
| `p2pbidir.py`, `p2pbw.py`, `p2ptest.py` | guest | Peer-copy (one-way and both ways, with a data check) and RCCL tests |
| `p2ptest4.py` | guest | N-GPU peer-copy matrix (with a data check) + N-rank RCCL all-reduce |
| `acsab.sh` | mgmt VM | Live ACS off/on/off A/B with BetterBench prefill + decode on a running server |

## Why a single card is easy and two cards on one switch are not

The firmware sizes each PLX chain's prefetchable window for a 256 MB BAR. The single-card fix (in barfix) goes
like this:

1. Set the card's ReBAR control to 32 GB.
2. Pre-program a large window on the root port and the PLX upstream port.
3. Remove the root port and rescan so the kernel re-places everything.

That works when a chain has one card. **When a chain has two, the rescan only places the first.** Each card needs
a child window of 32 GB + 2 MB (BAR0 plus the 2 MB BAR2), and that window has to start on a 32 GB boundary. Linux
sizes the parent window as the plain *sum* of the children, 64 GB + 4 MB. Card one takes 0 to 32 GB + 2 MB, and
the next 32 GB boundary is at 64 GB, which leaves no room for card two:

```
pci 0000:42:10.0: bridge window [mem size 0x800200000 64bit pref]: failed to assign
pci 0000:48:00.0: BAR 0 [mem size 0x800000000 64bit pref]: can't assign; no space
```

The starved card still reports `BAR 0: current size: 32GB` in its ReBAR capability, but it has no `Region 0`. Only
the Region line proves the BAR was actually placed. barfix checks both.

Things that do **not** work, so nobody tries them again:

- **Rescanning in a different way:** pre-programming the windows and/or the card BARs, clearing them, or using
  `resource0_resize`. The kernel re-sizes the chain on every rescan and always lands on the same sum.
- **`pci=resource_alignment=36@…`:**
  - GRUB treats the `;` between devices as a command separator, which silently truncated the boot line
    (ACS override, `pcie_aspm=off` and `efifb:off` were lost).
  - The alignment applies to *every* BAR on the device, so BAR5 and the ROM can't be placed either.

## The fix: `r9700_chainfix`

Once the rescan has placed the first card, barfix sees the second card has no Region 0 and does two things:

1. **Loads `r9700_chainfix`** with that chain's root port, upstream port, and the starved card's PLX downstream
   port. The module:
   - grows the root port's and the upstream's prefetchable windows in place (`adjust_resource()` plus the bridge
     registers), giving 64 GB per card on the chain;
   - leaves the first card untouched.

   The window only grows inside that chain's own root-complex aperture: roughly 1 TB each on buses 00/40 and
   512 GB each on 80/c0. Chain A (bus 40) and chain B (bus c0) are on separate root complexes. A chain never
   takes space from another chain's window. With 8 cards on 4 switches at 128 GB per switch, even two switches
   sharing a root complex would use at most 256 GB of its aperture.
2. **Removes and rescans only the starved card's PLX downstream port.** On its own, the kernel sizes that port's
   window correctly (32 GB + 2 MB) and places it in the free space.

Result with four cards (2026-09-23), both chains forced by `place_at` (see below):

```
40:01.1 / 41:00.0  prefetchable window 0x26000000000-0x27fffffffff   128G
42:08.0 -> 45:00.0 BAR0 32G @ 0x26000000000   BAR2 2M @ 0x26800000000
42:10.0 -> 48:00.0 BAR0 32G @ 0x27000000000   BAR2 2M @ 0x27800000000
c0:01.1 / c1:00.0  prefetchable window 0x14000000000-0x15fffffffff   128G
c2:08.0 -> c5:00.0 BAR0 32G @ 0x14000000000   BAR2 2M @ 0x14800000000
c2:10.0 -> c8:00.0 BAR0 32G @ 0x15000000000   BAR2 2M @ 0x15800000000
```

The module is DKMS-managed (`r9700-chainfix/1.2`). With `proxmox-default-headers` installed, apt rebuilds it for
every new kernel. If it's missing for the running kernel anyway, barfix runs `dkms autoinstall` once.

**Licence:** `r9700-chainfix/` is GPL-2.0, like any kernel module that links against the PCI core. It is a
standalone host tool and doesn't link with the rest of the repo, which stays Apache-2.0.

## Switch-local P2P on the PEX 8747

Out of the box, peer traffic between two cards on one switch doesn't stay in the switch. The PLX downstream ports
have ACS `ReqRedir+ CmpltRedir+`, so every peer transfer goes up the switch's single Gen3 x16 uplink to the root
complex, through the IOMMU, and back down the same link. Both cards share that uplink, so same-switch P2P ran at
half the cross-chain bandwidth. Two things together keep the traffic in the switch:

1. **Turn ACS redirect off** on the two PLX downstream ports (clear ReqRedir and CmpltRedir).
2. **Give the guest the host's addresses.** With redirect off, the switch routes by the address in the request. The
   guest's GPU driver uses *guest*-physical peer addresses, so those must equal the host BAR addresses.
   Otherwise the switch doesn't recognize them and forwards the request upstream anyway.

The GPUs have no ATS capability, which rules out ACS Direct-Translated P2P, the standard way to do this under an
IOMMU. That's why the addresses have to match instead.

### How the addresses are matched

- **Guest side (config only):** the guest firmware (`OVMF_CODE_4M.secboot.fd`) puts its 64-bit PCI window at
  `ALIGN_UP(reserved-memory-end, <window size>)` (window size = `X-PciMmio64Mb`) and fills the emulated switch
  bottom-up in *reverse* port order, 64G per card: with four ports, `p2pdn4` at the base, then `p2pdn3`, `p2pdn2`,
  `p2pdn1`.
  - `-m <ram>,slots=1,maxmem=…` moves reserved-memory-end. It reserves memory-hotplug address space and uses no
    RAM.
  - On AMD hosts, once that space crosses the HyperTransport hole below 1 TB, QEMU moves above-4G RAM to 1 TB.
  - With 256G of RAM and `maxmem=1100G`: a 128G window (two cards) starts at **0x22000000000**, a 256G window
    (four cards, needed since 4 x 64G no longer fits in 128G) at **0x24000000000**.
  - Test placement on a disk-less scratch VM with the same efidisk type (`qm monitor` → `info pci`). The legacy
    `OVMF_CODE.fd` uses a completely different "dynamic" window near the top of the address space.
- **Host side (`place_at`):** the kernel ignores pre-programmed bridge bases and first-fits the chain at the
  bottom of the bus aperture (0x20800000000, which isn't 128G-aligned). barfix therefore:
  1. removes the root port;
  2. loads `r9700_chainfix place_at=0x26000000000 root_bdf=40:00.0`, which reserves every free range of the
     aperture below the target (1.2+: it steps around windows already there, e.g. the NVMe's on bus c0);
  3. rescans, so the first fit lands the chain at the target;
  4. unloads the module, releasing the reservation;
  5. places the second card at +64G, as before.

### Four cards: only one chain can be switch-local

The two chains live in different root-complex apertures (chain A on bus 40: 2-3 TB; chain B on bus c0: 1-1.5 TB),
while the guest's 64-bit window is one contiguous range. So the guest can reproduce the host addresses of **one**
chain only. Chain A keeps switch-local P2P (`CHAIN_A_P2P=1`); chain B keeps ACS redirect on (`CHAIN_B_P2P=0`), so
its traffic, same-switch included, goes through the IOMMU, which translates guest to host addresses.

Two placement rules keep that safe:

- **Chain A's host window must not cover a guest address of another card.** With redirect off, a request that falls
  inside the PLX's window but hits no downstream port is not forwarded upstream. So chain A goes at the *top* of the
  guest window (`p2pdn2`/`p2pdn1` = 0x260/0x270, host window 0x260-0x280), and chain B's guest addresses
  (0x240/0x250) sit below it.
- **Chain B's host window must not overlap guest RAM.** Above-4G guest RAM sits at GPA 1 TB-1.25 TB, and chain B's
  first fit lands at 0x10800000000, inside it. `CHAIN_B_AT=0x14000000000` puts it above guest RAM.

VM100 (`/etc/pve/qemu-server/100.conf` `args:`):

```
-m 262144,slots=1,maxmem=1100G -fw_cfg name=opt/ovmf/X-PciMmio64Mb,string=262144
-device pcie-root-port,id=p2prp,bus=pcie.0,chassis=90,slot=90,x-speed=32,x-width=16
-device x3130-upstream,id=p2pup,bus=p2prp
-device xio3130-downstream,id=p2pdn{1..4},bus=p2pup,chassis=9{1..4},slot={1..4}   # four ports
-device vfio-pci,host=0000:48:00.0,bus=p2pdn1,addr=0x0,rombar=0     # guest 0x27000000000 == host
-device vfio-pci,host=0000:45:00.0,bus=p2pdn2,addr=0x0,rombar=0     # guest 0x26000000000 == host
-device vfio-pci,host=0000:c8:00.0,bus=p2pdn3,addr=0x0,rombar=0     # guest 0x25000000000 (host 0x15000000000)
-device vfio-pci,host=0000:c5:00.0,bus=p2pdn4,addr=0x0,rombar=0     # guest 0x24000000000 (host 0x14000000000)
```

Result (2026-09-23, `p2ptest4.py`): all four GPUs init (32 GB VRAM and BAR, SMU OK), each has 3 KFD P2P links.
Every peer pair copies at 12.5-13.0 GB/s one-way with the data check passing. A 4-rank RCCL all-reduce is correct,
at 3.7 GB/s busbw @256MB. No IOMMU or AER faults on the host or in the guest.

### Result (2026-09-23, 45 <-> 48, 256 MB)

| test | ACS redirect on | redirect off + matched addresses |
|---|---|---|
| peer copy 0->1 alone | 12.71 GB/s | 13.01 GB/s |
| peer copy 1->0 alone | 12.71 GB/s | 13.02 GB/s |
| **both at once, total** | 12.74 GB/s | **25.26 GB/s** |
| **RCCL all-reduce busbw** | 5.09 GB/s | **10.08 GB/s** (24.8 ms vs 49.1 ms) |

- **Data and logs:** data checks pass both ways, and neither host nor guest logged IOMMU or AER errors.
- **Per-direction limit:** the ~13 GB/s per direction is now the Gen3 x16 link between the switch and each card.
- **Checked paths:** a forced re-enumeration, a VM start and a `qm reboot 100` all came back at the same numbers.
- **Idle link speed:** the switch-to-card links read 2.5 GT/s when idle. That's amdgpu's idle power state, and they
  train to 8 GT/s under load.

### Security trade-off

With redirect off, DMA from 45 to 48's address ranges (and back) goes card to card with no IOMMU check. Only
traffic aimed at the other card's windows is routed locally; everything else still goes through the IOMMU. That's
fine while both cards belong to the same VM. **If the cards on a switch are ever split between VMs, set
`CHAIN_A_P2P=0`**; otherwise one VM's GPU could write into the other VM's GPU.

### Effect on serving (2026-09-23, `serve/27b.sh`, 27B NVFP4 TP2)

- **Parity with separate switches:** a full BetterBench 20-pass on 45 + 48 with switch-local P2P matches the
  2026-09-22 baseline on 45 + c6 (separate switches) across decode, concurrency and prefill.
- **Live A/B:** on one running server, ACS redirect off / on / off (`acsab.sh`):

| | prefill 2k | 8k | 16k | 32k | decode step p50 |
|---|--:|--:|--:|--:|--:|
| switch-local | 4,167 | 4,187 | 4,070 | 3,821 | 24.47 ms |
| **via the CPU (ACS on)** | **3,741** | **3,770** | **3,683** | **3,491** | **25.29 ms** |
| switch-local, repeat | 4,121 | 4,149 | 4,047 | 3,810 | 24.49 ms |

- **What routing through the CPU costs:**
  - **prefill, −9 to −10%:** its all-reduces are large, so it's bandwidth-bound even with 4-bit compression;
  - **decode step time, +3.3%:** every small all-reduce makes a round trip through the root complex.
- **Why the second card per switch is free:** only because of the switch-local routing above.

## Install

```
apt install dkms proxmox-default-headers
cp -r r9700-chainfix /usr/src/r9700-chainfix-1.2
dkms add r9700-chainfix/1.2 && dkms install r9700-chainfix/1.2
install -m755 r9700-barfix.sh /usr/local/sbin/
install -m755 gpu-reset.sh /var/lib/vz/snippets/
qm set 100 --hookscript local:snippets/gpu-reset.sh
```

## Operating it

- **Automatic:** the hookscript runs barfix at every VM100 pre-start and post-stop.
- **Idempotent:** a chain is left alone when every card has an assigned 32 GB Region 0 and, with `CHAIN_*_AT` set,
  the first card sits at that address. Repeated resets leave the RDNA PSP unable to reload without a full power
  cycle, so healthy cards aren't touched.
- **ACS:** switch-local P2P is re-applied on every run. The kernel re-enables ACS whenever it rescans a port.
- **Forced:** `FORCE=1 /usr/local/sbin/r9700-barfix.sh` re-enumerates everything. The VM must be stopped.
- **Knobs:**
  - `CHAIN_A` / `CHAIN_B`: the cards on each chain. An empty value skips that chain.
  - `CHAIN_A_AT` / `CHAIN_B_AT`: forced chain addresses, defaults `0x26000000000` / `0x14000000000`. Set one
    empty for the kernel's own first fit.
  - `CHAIN_A_P2P`: default 1. `CHAIN_B_P2P`: default 0.
  - `CHAIN_*_WIN`: the pre-programmed windows.
- **Check:**
  - `lspci -vvs 48:00.0 | grep -E "Region 0|BAR 0: current"`: both lines must show 32G.
  - `setpci -s 42:08.0 f2a.w`: `0011` means P2P mode, `001d` means redirect is on.
- **Guest VM:** its 64-bit window must hold every passed card's BAR. Adding cards changes the placement: recheck
  it with the scratch-VM probe and adjust `maxmem` / `CHAIN_*_AT` together.

## Decode vs prefill

Decode all-reduces are small and limited by latency: the P2P all-reduce kernel cut per-call time from 69 µs
(RCCL) to ~3 µs and gave Flash-Next +9.5% single-stream decode (PROGRESS.md). Prefill all-reduces are large and
limited by bandwidth, which is what switch-local P2P doubles. The 2026-09-18 "P2P gives nothing" A/B is **not**
valid: both arms used P2P IPC.
