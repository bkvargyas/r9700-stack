#!/bin/bash
# 32GB ReBAR for R9700s behind the PLX PEX 8747 chips on this box.
# Firmware sizes the PLX prefetch windows for 256MB, so each chain's root port + PLX upstream
# get a large 64-bit prefetchable window programmed by hand, then the chain is re-enumerated
# (remove root port + rescan) so the kernel can hand each card a 32GB BAR inside it.
#
# chain A: root 40:01.1  up 41:00.0  -> 45:00.0 (PLX port 42:08.0), 48:00.0 (PLX port 42:10.0)
# chain B: root c0:01.1  up c1:00.0  -> c6:00.0 (PLX port c2:10.0)
#
# A chain may carry MORE THAN ONE card (45 and 48 are on the same PEX 8747): every card on the
# chain gets its ReBAR control set before the single remove+rescan, otherwise the cards that were
# skipped come back at their firmware 256MB.
#
# Two or more cards on one chain: the rescan only places the FIRST card. Linux sizes the chain
# window as the sum of the per-card windows (32GB+2MB each) but each must be 32GB-aligned, so the
# others get no BAR0 at all. For those, r9700_chainfix.ko grows the root-port + PLX-upstream windows
# in place (64GB per card, inside this chain's own root-complex aperture), then only the starved
# card's PLX downstream port is removed+rescanned, which the kernel sizes and places correctly alone.
#
# Idempotent: a chain whose cards all have an ASSIGNED 32GB BAR0 is left alone, so repeated VM
# stop/start does not re-hit the GPUs (chained resets leave the RDNA PSP unable to reload without a
# power cycle). FORCE=1 re-enumerates anyway.
set -u

# Override to change which cards are attached to which chain, e.g. CHAIN_A="45:00.0" CHAIN_B="c6:00.0"
CHAIN_A="${CHAIN_A-45:00.0 48:00.0}"
CHAIN_B="${CHAIN_B-c6:00.0}"
FORCE="${FORCE:-0}"

CHAINFIX=r9700_chainfix   # DKMS package r9700-chainfix/1.0 (src /usr/src/r9700-chainfix-1.0), rebuilt per kernel

chain_is_32g() {
  local c
  for c in "$@"; do
    card_is_32g "$c" || return 1
  done
  return 0
}

# ReBAR control alone is not enough: the BAR must also have been ASSIGNED (Region 0 present at 32G);
# a card whose chain window could not fit it reports "current size: 32GB" with no Region 0.
card_is_32g() {
  local v
  v=$(lspci -vvs "$1" 2>/dev/null)
  grep -q "BAR 0: current size: 32GB" <<<"$v" && grep -q "Region 0: Memory at .*\[size=32G\]" <<<"$v"
}

unbind_card() {
  local card="$1" aud="${1%.0}.1" d
  for d in "0000:$card" "0000:$aud"; do
    [ -e /sys/bus/pci/devices/$d/driver ] && echo "$d" > /sys/bus/pci/devices/$d/driver/unbind 2>/dev/null
  done
  return 0
}

rebind_card() {
  local card="$1" aud="${1%.0}.1" d
  for d in "0000:$card" "0000:$aud"; do
    [ -e /sys/bus/pci/devices/$d ] || continue
    [ -e /sys/bus/pci/devices/$d/driver ] || {
      echo vfio-pci > /sys/bus/pci/devices/$d/driver_override 2>/dev/null
      echo "$d" > /sys/bus/pci/drivers_probe 2>/dev/null
    }
  done
  return 0
}

# ReBAR control 208.L bits[8:13] = 15 -> 32GB. Card must be unbound from its driver first.
set_rebar_32g() {
  local card="$1" o
  o=$(setpci -s "$card" 208.L 2>/dev/null) || { echo "  $card: no ReBAR cap at 208"; return 1; }
  setpci -s "$card" 208.L=$(printf "%08x" $(( (0x$o & ~0x3F00) | (15<<8) )))
}

# The PLX downstream port a card hangs off = the path component right after the PLX upstream.
plx_port_of() {
  local up="$1" card="$2"
  readlink -f /sys/bus/pci/devices/0000:$card | sed -nE "s#.*/0000:$up/0000:([0-9a-f:.]+)/.*#\1#p"
}

# The chainfix module is DKMS-managed, so apt rebuilds it for every new kernel (headers come with
# proxmox-default-headers). If it is still missing for the running kernel, try one DKMS build here.
ensure_chainfix() {
  modinfo "$CHAINFIX" >/dev/null 2>&1 && return 0
  echo "  $CHAINFIX missing for $(uname -r), running dkms autoinstall"
  dkms autoinstall -k "$(uname -r)" >/dev/null 2>&1
  modinfo "$CHAINFIX" >/dev/null 2>&1 \
    || { echo "  ERROR: no $CHAINFIX for $(uname -r) (see: dkms status; headers installed?)"; return 1; }
}

# fix_starved <root> <upstream> <card>...: second+ cards on a chain that the rescan left without BAR0.
fix_starved() {
  local root="$1" up="$2"; shift 2
  local cards=("$@") card port starved=()
  for card in "${cards[@]}"; do card_is_32g "$card" || starved+=("$card"); done
  [ ${#starved[@]} -gt 0 ] || return 0
  ensure_chainfix || return 1

  for card in "${starved[@]}"; do
    port=$(plx_port_of "$up" "$card")
    [ -n "$port" ] || { echo "  $card: cannot find its PLX downstream port under $up"; continue; }
    echo "  $card starved -> grow $root/$up windows to $(( 64 * ${#cards[@]} ))GB, re-place via port $port"
    modprobe -r "$CHAINFIX" 2>/dev/null
    modprobe "$CHAINFIX" root_bdf="$root" up_bdf="$up" port_bdf="$port" window_gb=$(( 64 * ${#cards[@]} )) \
      || { echo "  ERROR: r9700_chainfix failed, see dmesg"; continue; }
    modprobe -r "$CHAINFIX"
    unbind_card "$card"
    echo 1 > /sys/bus/pci/devices/0000:$port/remove; sleep 2
    echo 1 > /sys/bus/pci/rescan; sleep 3
  done
}

# resize_chain <root> <upstream> <base> <limit_root> <limit_up> <card>...
resize_chain() {
  local root="$1" up="$2" base="$3" limr="$4" limu="$5"; shift 5
  local cards=("$@") card
  [ ${#cards[@]} -gt 0 ] || { echo "  (no cards)"; return; }
  if [ "$FORCE" != 1 ] && chain_is_32g "${cards[@]}"; then echo "  ${cards[*]} already 32GB"; return; fi

  for card in "${cards[@]}"; do unbind_card "$card"; done
  sleep 1
  for card in "${cards[@]}"; do set_rebar_32g "$card"; done

  setpci -s "$root" 28.L=$base 2c.L=$limr 24.W=0001 26.W=fff1
  setpci -s "$up"   28.L=$base 2c.L=$limu 24.W=0001 26.W=fff1

  echo 1 > /sys/bus/pci/devices/0000:$root/remove; sleep 2
  echo 1 > /sys/bus/pci/rescan; sleep 3

  fix_starved "$root" "$up" "${cards[@]}"

  for card in "${cards[@]}"; do rebind_card "$card"; done
  sleep 1
  for card in "${cards[@]}"; do
    echo "  $card rebar=$(lspci -vvs "$card" | grep -oE 'current size: [0-9]+[MG]B' | head -1)" \
         "region0=$(r=$(lspci -vvs "$card" | grep -m1 -oE 'Region 0: Memory at [0-9a-f]+ .*size=[0-9]+[MG]\]' | sed -E 's/.*at ([0-9a-f]+).*size=([0-9]+[MG]).*/\2@\1/'); echo "${r:-UNASSIGNED}")" \
         "driver=$(basename "$(readlink -f /sys/bus/pci/devices/0000:$card/driver 2>/dev/null)" 2>/dev/null)"
  done
  chain_is_32g "${cards[@]}" || echo "  WARNING: not every card on this chain has a 32GB BAR0"
}

echo "chain A (${CHAIN_A:-none}):"
resize_chain 40:01.1 41:00.0 00000208 000002a0 0000029f $CHAIN_A
echo "chain B (${CHAIN_B:-none}):"
resize_chain c0:01.1 c1:00.0 00000148 00000168 00000167 $CHAIN_B
