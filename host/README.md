# Host setup: 32 GB BARs for R9700s behind PLX switches

The Proxmox host (`192.168.0.100`) has its R9700s behind PLX PEX 8747 switches, with up to two cards per switch.
Every card has to be passed to the VM with its full 32 GB BAR0. That's what lets amdgpu map all of VRAM, and it's
also what KFD/RCCL require before they allow peer-to-peer. This board's BIOS has no Resizable-BAR support, so all
of the setup below happens in Linux.

| File | Installed at | Role |
|---|---|---|
| `r9700-barfix.sh` | `/usr/local/sbin/r9700-barfix.sh` | Sets every card to 32 GB and re-enumerates each chain |
| `r9700-chainfix/` | `/usr/src/r9700-chainfix-1.0` (DKMS) | Kernel module that gives the second card on a switch its window |
| `gpu-reset.sh` | `/var/lib/vz/snippets/gpu-reset.sh` | Proxmox hookscript that runs barfix at VM pre-start / post-stop |

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

Result on chain A (root `40:01.1`, upstream `41:00.0`):

```
40:01.1 / 41:00.0  prefetchable window 0x20800000000-0x227ffffffff   128G
42:08.0 -> 45:00.0 BAR0 32G @ 0x20800000000   BAR2 2M @ 0x21000000000
42:10.0 -> 48:00.0 BAR0 32G @ 0x21800000000   BAR2 2M @ 0x22000000000
```

The module is DKMS-managed (`r9700-chainfix/1.0`). With `proxmox-default-headers` installed, apt rebuilds it for
every new kernel. If it's missing for the running kernel anyway, barfix runs `dkms autoinstall` once.

**Licence:** `r9700-chainfix/` is GPL-2.0, like any kernel module that links against the PCI core. It is a
standalone host tool and doesn't link with the rest of the repo, which stays Apache-2.0.

## Install

```
apt install dkms proxmox-default-headers
cp -r r9700-chainfix /usr/src/r9700-chainfix-1.0
dkms add r9700-chainfix/1.0 && dkms install r9700-chainfix/1.0
install -m755 r9700-barfix.sh /usr/local/sbin/
install -m755 gpu-reset.sh /var/lib/vz/snippets/
qm set 100 --hookscript local:snippets/gpu-reset.sh
```

## Operating it

- **Automatic:** the hookscript runs barfix at every VM100 pre-start and post-stop.
- **Idempotent:** a chain whose cards all have an assigned 32 GB Region 0 is left alone. Repeated resets leave the
  RDNA PSP unable to reload without a full power cycle, so healthy cards aren't touched.
- **Forced:** `FORCE=1 /usr/local/sbin/r9700-barfix.sh` re-enumerates everything. The VM must be stopped.
- **Choosing chains:** `CHAIN_A="…" CHAIN_B="…"` selects which cards are on each chain. An empty value skips that
  chain.
- **Check:** run `lspci -vvs 48:00.0 | grep -E "Region 0|BAR 0: current"`. Both lines must show 32G.
- **Guest VM:** its 64-bit MMIO (`-fw_cfg name=opt/ovmf/X-PciMmio64Mb`) must hold every passed card's BAR.
  Today it's 131072 (128 GB); 8 cards need at least 512 GB.

## Verified (2026-09-23, VM100 = 45 + 48)

- **Forced barfix run:**
  - the rescan placed card 45;
  - barfix detected that card 48 was starved;
  - the module grew the chain window to 128G;
  - the port rescan placed card 48;
  - both cards were bound to vfio-pci at 32G.
- **VM start and `qm reboot 100`:** the hooks ran at post-stop and pre-start and correctly skipped the healthy
  chain. The BARs held.
- **Guest after the reboot:** both cards show `VRAM RAM=32624M, BAR=32768M` and `p2p_links_count 1`.
- **P2P test** (`p2ptest.py`, RCCL 2.30.4 rebuild): all-reduce is correct and peer access is True. Bus bandwidth is
  ~5.0 GB/s at 256 MB.

### P2P path

Two things aren't native here:

- **Guest topology:** the guest sees the GPUs under an *emulated* QEMU switch (xio3130). That only exists so
  amdgpu/RCCL will enable peer access. The transfers themselves are real GPU-to-GPU DMA into the peer's BAR, with
  no staging in host RAM.
- **Routing:** transfers don't stay inside the PLX switch. Its downstream ports have ACS `ReqRedir+ CmpltRedir+`,
  so peer requests go up to the root complex, get translated by the IOMMU, and come back down. The GPUs expose no
  ATS capability, so ACS Direct-Translated P2P (switch-local under an IOMMU) isn't possible. Switching redirect off
  would make the switch route by *guest*-physical addresses, which are wrong on the host.

The ~5 GB/s is about half of the ~9.9 GB/s measured on 2026-09-18 with the cards on separate chains (45 + c6).
That drop is the next thing to investigate. The 2026-09-18 A/B test also found P2P doesn't matter for Flash-Next
at TP=2, where expert offload dominates.
