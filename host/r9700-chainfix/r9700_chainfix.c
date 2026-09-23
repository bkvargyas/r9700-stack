// SPDX-License-Identifier: GPL-2.0
/*
 * r9700_chainfix: give a second R9700 on the same PLX chain a 32GB BAR.
 *
 * Linux sizes a bridge window as the SUM of its child windows, but each child window here is
 * 32GB+2MB (BAR0 + BAR2) and must be 32GB-aligned, so two of them need ~96GB while the parent
 * gets 64GB+4MB -> the second card is left with no BAR0 (no firmware ReBAR support on this box).
 *
 * After r9700-barfix.sh has placed the first card, this module grows the root-port and PLX-upstream
 * prefetchable windows in place (adjust_resource + program the bridge registers), then asks the
 * kernel to assign only the starved PLX downstream port subtree, which it sizes correctly alone.
 * All work happens at load time; the module does nothing afterwards and can be removed.
 *
 * place_at=<addr> mode (load BEFORE the chain's root-port remove+rescan, unload after): reserves the
 * part of the root bus aperture below <addr>, so the rescan's first-fit puts the chain window exactly at
 * <addr>. Used to give the cards host addresses a guest can reproduce (OVMF places the guest's 64-bit
 * window on a 128GB boundary), which is what lets peer traffic stay inside the PLX switch.
 */
#include <linux/module.h>
#include <linux/pci.h>

static char *root_bdf = "40:01.1";   /* root port */
static char *up_bdf   = "41:00.0";   /* PLX upstream */
static char *port_bdf = "42:10.0";   /* PLX downstream port of the starved card */
static unsigned int window_gb = 128; /* new size of the root/upstream prefetchable windows */
static bool dry_run;
static char *place_at = "";          /* reserve mode: chain window start to force, e.g. 0x22000000000 */
module_param(root_bdf, charp, 0444);
module_param(up_bdf, charp, 0444);
module_param(port_bdf, charp, 0444);
module_param(window_gb, uint, 0444);
module_param(dry_run, bool, 0444);
module_param(place_at, charp, 0444);

static struct resource placeholder = { .name = "r9700_chainfix placeholder" };

static struct pci_dev *get_dev(const char *bdf)
{
	unsigned int bus, slot, fn;

	if (sscanf(bdf, "%x:%x.%x", &bus, &slot, &fn) != 3)
		return NULL;
	return pci_get_domain_bus_and_slot(0, bus, PCI_DEVFN(slot, fn));
}

/* Write a bridge's 64-bit prefetchable window registers from its resource. */
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

static int grow(struct pci_dev *br, resource_size_t size)
{
	struct resource *r = &br->resource[PCI_BRIDGE_PREF_MEM_WINDOW];
	int ret;

	if (!r->parent || !(r->flags & IORESOURCE_PREFETCH)) {
		pci_err(br, "chainfix: prefetchable window not assigned %pR\n", r);
		return -ENODEV;
	}
	if (resource_size(r) >= size) {
		pci_info(br, "chainfix: window already %pR\n", r);
		return 0;
	}
	if (dry_run) {
		pci_info(br, "chainfix: [dry] would grow %pR to %llu GB\n", r, (u64)size >> 30);
		return 0;
	}
	ret = adjust_resource(r, r->start, size);
	if (ret) {
		pci_err(br, "chainfix: adjust_resource %pR -> %llu GB failed: %d\n",
			r, (u64)size >> 30, ret);
		return ret;
	}
	write_pref_window(br, r);
	pci_info(br, "chainfix: grew window to %pR\n", r);
	return 0;
}

/* Reserve [start of the root bus aperture holding <at>, at) so nothing below <at> can be allocated. */
static int reserve_below(u64 at)
{
	struct pci_dev *root = get_dev(root_bdf);
	struct resource *r;
	int ret = -ENODEV;

	if (!root) {
		pr_err("r9700_chainfix: root port %s not found\n", root_bdf);
		return ret;
	}
	pci_bus_for_each_resource(root->bus, r) {
		if (!r || !(r->flags & IORESOURCE_MEM) || at <= r->start || at > r->end)
			continue;
		placeholder.start = r->start;
		placeholder.end = at - 1;
		placeholder.flags = IORESOURCE_MEM | IORESOURCE_BUSY;
		ret = request_resource(r, &placeholder);
		if (ret) {
			pr_err("r9700_chainfix: cannot reserve %pR in %pR (in use)\n", &placeholder, r);
		} else {
			pci_info(root, "chainfix: reserved %pR so the chain window lands at %#llx\n", &placeholder, at);
		}
		break;
	}
	if (ret == -ENODEV)
		pci_err(root, "chainfix: no bus aperture contains %#llx\n", at);
	pci_dev_put(root);
	return ret;
}

static int grow_chain(void)
{
	struct pci_dev *root = get_dev(root_bdf), *up = get_dev(up_bdf), *port = get_dev(port_bdf);
	resource_size_t size = (resource_size_t)window_gb << 30;
	int ret = -ENODEV;

	if (!root || !up || !port) {
		pr_err("r9700_chainfix: device not found (%s %s %s)\n", root_bdf, up_bdf, port_bdf);
		goto out;
	}
	if (port->resource[PCI_BRIDGE_PREF_MEM_WINDOW].parent) {
		pci_info(port, "chainfix: window already assigned %pR, nothing to do\n",
			 &port->resource[PCI_BRIDGE_PREF_MEM_WINDOW]);
		ret = 0;
		goto out;
	}

	pci_lock_rescan_remove();
	ret = grow(root, size);   /* outer window first so the inner one has room */
	if (!ret)
		ret = grow(up, size);
	if (!ret && !dry_run) {
		pci_assign_unassigned_bridge_resources(port);
		pci_info(port, "chainfix: port window now %pR\n",
			 &port->resource[PCI_BRIDGE_PREF_MEM_WINDOW]);
	}
	pci_unlock_rescan_remove();
out:
	pci_dev_put(port);
	pci_dev_put(up);
	pci_dev_put(root);
	return ret;
}

static int __init chainfix_init(void)
{
	u64 at;

	if (*place_at) {
		if (kstrtoull(place_at, 0, &at))
			return -EINVAL;
		return dry_run ? 0 : reserve_below(at);
	}
	return grow_chain();
}

static void __exit chainfix_exit(void)
{
	if (placeholder.parent)
		release_resource(&placeholder);
}

module_init(chainfix_init);
module_exit(chainfix_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Grow a PLX chain prefetch window so a second R9700 gets its 32GB BAR");
