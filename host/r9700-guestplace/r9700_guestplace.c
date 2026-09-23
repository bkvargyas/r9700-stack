// SPDX-License-Identifier: GPL-2.0
/*
 * r9700_guestplace: in the VM, move passed-through R9700s to their HOST BAR addresses.
 *
 * Switch-local P2P (ACS redirect off on the PLX) routes a peer write by the address the GPU emits. The
 * GPU has no ATS, so it emits the guest-physical address, which only reaches the peer when guest BAR ==
 * host BAR. OVMF packs every 64-bit BAR bottom-up in its own order, so it cannot be told where to put
 * them. This module moves every listed card to a given address, after OVMF and before amdgpu.
 *
 * Guest topology mirrors the host: one emulated switch (root port -> upstream -> downstream ports) per
 * PLX chain. Each switch is re-windowed around its own cards only, so a switch never has to span the gap
 * between two host chains (other guest devices sit in it). Every card behind a switch must be listed.
 *
 * Load BEFORE amdgpu binds (the cards must have no driver). The VM's 64-bit PCI hole must cover the
 * targets (q35-pcihost.pci-hole64-size). All work happens at load; the module can be removed after.
 *
 *   modprobe r9700_guestplace place=0000:04:00.0@0x26000000000,0000:03:00.0@0x27000000000,...
 *
 * Each card gets BAR0 at <addr> and BAR2 right after BAR0; its downstream port window covers both.
 */
#include <linux/module.h>
#include <linux/pci.h>

#define MAX_CARDS 8
static char *place[MAX_CARDS];
static int nr_place;
module_param_array(place, charp, &nr_place, 0444);
static bool dry_run;
module_param(dry_run, bool, 0444);

#define PREF_WIN PCI_BRIDGE_PREF_MEM_WINDOW

struct card {
	struct pci_dev *dev, *port;
	u64 at, end;
	u16 cmd;
};

/* one emulated switch: root port -> upstream port, and the span of its cards' new windows */
struct sw {
	struct pci_dev *root, *up;
	struct resource *parent;	/* root port window's parent (host bridge window) */
	resource_size_t lo, hi;
};

static void write_pref_window(struct pci_dev *br, struct resource *r)
{
	u16 base, limit;

	pci_read_config_word(br, PCI_PREF_MEMORY_BASE, &base);
	pci_read_config_word(br, PCI_PREF_MEMORY_LIMIT, &limit);
	base  = (base  & 0xf) | (((u32)r->start >> 16) & 0xfff0);
	limit = (limit & 0xf) | (((u32)r->end   >> 16) & 0xfff0);
	pci_write_config_dword(br, PCI_PREF_BASE_UPPER32,  upper_32_bits(r->start));
	pci_write_config_dword(br, PCI_PREF_LIMIT_UPPER32, upper_32_bits(r->end));
	pci_write_config_word(br, PCI_PREF_MEMORY_BASE, base);
	pci_write_config_word(br, PCI_PREF_MEMORY_LIMIT, limit);
}

static void write_bar64(struct pci_dev *dev, int resno, u64 addr)
{
	int reg = PCI_BASE_ADDRESS_0 + 4 * resno;
	u32 lo;

	pci_read_config_dword(dev, reg, &lo);
	pci_write_config_dword(dev, reg, (lower_32_bits(addr) & ~0xfU) | (lo & 0xf));
	pci_write_config_dword(dev, reg + 4, upper_32_bits(addr));
}

/* (re)claim a bridge's prefetchable window at [start, end] inside <parent> and program it */
static int claim_window(struct pci_dev *br, struct resource *parent, resource_size_t start,
			resource_size_t end)
{
	struct resource *r = &br->resource[PREF_WIN];
	int ret;

	r->start = start;
	r->end = end;
	r->flags &= ~IORESOURCE_UNSET;
	ret = request_resource(parent, r);
	if (ret) {
		pci_err(br, "guestplace: cannot place window %pR in %pR: %d\n", r, parent, ret);
		return ret;
	}
	write_pref_window(br, r);
	pci_info(br, "guestplace: window %pR\n", r);
	return 0;
}

/* stop decoding and drop the card's BARs + port window */
static void release_card(struct card *c)
{
	struct pci_dev *dev = c->dev;
	int i;

	pci_read_config_word(dev, PCI_COMMAND, &c->cmd);
	pci_write_config_word(dev, PCI_COMMAND, c->cmd & ~PCI_COMMAND_MEMORY);
	for (i = 0; i <= 2; i += 2)
		if (dev->resource[i].parent)
			release_resource(&dev->resource[i]);
	if (c->port->resource[PREF_WIN].parent)
		release_resource(&c->port->resource[PREF_WIN]);
}

/* port window at [at, end] inside the upstream window, BAR0 at <at>, BAR2 right after it */
static int place_card(struct card *c)
{
	struct pci_dev *dev = c->dev;
	struct resource *win = &c->port->resource[PREF_WIN];
	struct resource *b0 = &dev->resource[0], *b2 = &dev->resource[2];
	resource_size_t s0 = resource_size(b0), s2 = resource_size(b2);
	int ret;

	ret = claim_window(c->port, &c->port->bus->self->resource[PREF_WIN], c->at, c->end);
	if (ret)
		goto out;
	b0->start = c->at;
	b0->end = c->at + s0 - 1;
	b2->start = c->at + s0;
	b2->end = c->at + s0 + s2 - 1;
	b0->flags &= ~IORESOURCE_UNSET;
	b2->flags &= ~IORESOURCE_UNSET;
	ret = request_resource(win, b0);
	if (!ret)
		ret = request_resource(win, b2);
	if (ret) {
		pci_err(dev, "guestplace: cannot claim BARs in %pR: %d\n", win, ret);
		goto out;
	}
	write_bar64(dev, 0, b0->start);
	write_bar64(dev, 2, b2->start);
	pci_info(dev, "guestplace: BAR0 %pR BAR2 %pR\n", b0, b2);
out:
	pci_write_config_word(dev, PCI_COMMAND, c->cmd);
	return ret;
}

static int parse_card(const char *spec, struct card *c)
{
	unsigned int dom, bus, slot, fn;
	char *at = strchr(spec, '@');
	resource_size_t s0;

	if (!at || sscanf(spec, "%x:%x:%x.%x@", &dom, &bus, &slot, &fn) != 4 ||
	    kstrtoull(at + 1, 0, &c->at)) {
		pr_err("r9700_guestplace: bad entry '%s' (want dddd:bb:ss.f@addr)\n", spec);
		return -EINVAL;
	}
	c->dev = pci_get_domain_bus_and_slot(dom, bus, PCI_DEVFN(slot, fn));
	if (!c->dev) {
		pr_err("r9700_guestplace: %s not found\n", spec);
		return -ENODEV;
	}
	if (c->dev->vendor != PCI_VENDOR_ID_ATI || (c->dev->class >> 16) != PCI_BASE_CLASS_DISPLAY) {
		pci_err(c->dev, "guestplace: %s is not an AMD GPU\n", spec);
		return -ENODEV;
	}
	if (c->dev->driver) {
		pci_err(c->dev, "guestplace: bound to %s; load before amdgpu\n", c->dev->driver->name);
		return -EBUSY;
	}
	c->port = c->dev->bus->self;
	if (!c->port || !c->port->bus->self || !c->port->bus->self->bus->self) {
		pci_err(c->dev, "guestplace: expected root port -> upstream -> downstream -> card\n");
		return -ENODEV;
	}
	s0 = resource_size(&c->dev->resource[0]);
	if (!s0 || !IS_ALIGNED(c->at, s0)) {
		pci_err(c->dev, "guestplace: %#llx is not aligned to BAR0 %pR\n", c->at, &c->dev->resource[0]);
		return -EINVAL;
	}
	c->end = ALIGN(c->at + s0 + resource_size(&c->dev->resource[2]), SZ_1M) - 1;
	return 0;
}

static int __init guestplace_init(void)
{
	struct card cards[MAX_CARDS] = {};
	struct sw sws[MAX_CARDS] = {};
	struct pci_dev *child;
	int i, j, nr_sw = 0, ret = 0;

	if (!nr_place)
		return -EINVAL;
	for (i = 0; i < nr_place; i++) {
		struct pci_dev *up;

		ret = parse_card(place[i], &cards[i]);
		if (ret)
			goto put;
		up = cards[i].port->bus->self;
		for (j = 0; j < nr_sw && sws[j].up != up; j++)
			;
		if (j == nr_sw) {
			sws[j].up = up;
			sws[j].root = up->bus->self;
			sws[j].lo = ~(resource_size_t)0;
			nr_sw++;
		}
		sws[j].lo = min(sws[j].lo, (resource_size_t)cards[i].at);
		sws[j].hi = max(sws[j].hi, (resource_size_t)cards[i].end);
	}

	/* every port behind a switch that owns a window must be moving, or the re-window would orphan it */
	for (j = 0; j < nr_sw; j++) {
		list_for_each_entry(child, &sws[j].up->subordinate->devices, bus_list) {
			bool listed = false;

			for (i = 0; i < nr_place; i++)
				listed |= cards[i].port == child;
			if (!listed && child->resource[PREF_WIN].parent) {
				pci_err(child, "guestplace: has a window but its card is not listed\n");
				ret = -EINVAL;
				goto put;
			}
		}
		sws[j].parent = sws[j].root->resource[PREF_WIN].parent;
		if (!sws[j].parent || !sws[j].up->resource[PREF_WIN].parent) {
			pci_err(sws[j].root, "guestplace: switch has no prefetchable window\n");
			ret = -ENODEV;
			goto put;
		}
		pci_info(sws[j].root, "guestplace: %pR -> [%#llx-%#llx]%s\n", &sws[j].root->resource[PREF_WIN],
			 (u64)sws[j].lo, (u64)sws[j].hi, dry_run ? " [dry run]" : "");
	}
	if (dry_run)
		goto put;

	pci_lock_rescan_remove();
	/* 1: release every card, then each switch's windows (upstream before its root port) */
	for (i = 0; i < nr_place; i++)
		release_card(&cards[i]);
	for (j = 0; j < nr_sw; j++) {
		release_resource(&sws[j].up->resource[PREF_WIN]);
		release_resource(&sws[j].root->resource[PREF_WIN]);
	}
	/* 2: re-window each switch around its own cards */
	for (j = 0; !ret && j < nr_sw; j++) {
		ret = claim_window(sws[j].root, sws[j].parent, sws[j].lo, sws[j].hi);
		if (!ret)
			ret = claim_window(sws[j].up, &sws[j].root->resource[PREF_WIN], sws[j].lo, sws[j].hi);
	}
	/* 3: cards */
	for (i = 0; !ret && i < nr_place; i++)
		ret = place_card(&cards[i]);
	pci_unlock_rescan_remove();
	if (ret)
		pr_err("r9700_guestplace: FAILED part way (%d); cards may be unusable until reboot\n", ret);
put:
	for (i = 0; i < nr_place; i++)
		pci_dev_put(cards[i].dev);
	return ret;
}

static void __exit guestplace_exit(void)
{
}

module_init(guestplace_init);
module_exit(guestplace_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Place passed-through R9700 BARs at their host addresses (switch-local P2P in a VM)");
