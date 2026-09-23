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
# Chain windows pre-programmed before the rescan: "<upper32 base> <upper32 limit root> <upper32 limit upstream>"
CHAIN_A_WIN="${CHAIN_A_WIN-00000220 000002a0 0000029f}"
CHAIN_B_WIN="${CHAIN_B_WIN-00000148 00000168 00000167}"
# Force a chain's window (= its first card's BAR0) to start at a given host address, e.g. 0x22000000000.
# Used so the guest can put its GPUs at the SAME addresses (switch-local P2P); empty = kernel's first fit.
CHAIN_A_AT="${CHAIN_A_AT-0x22000000000}"
CHAIN_B_AT="${CHAIN_B_AT-}"
# Switch-local P2P: clear ACS ReqRedir/CmpltRedir on the chain's PLX downstream ports so peer traffic between
# its cards stays inside the switch. Only valid when the VM sees the cards at their HOST addresses (VM100:
# maxmem=1100G + 48 on p2pdn1, 45 on p2pdn2), and only while every card on the chain belongs to ONE VM
# (it removes IOMMU checks on card-to-card DMA). The kernel re-enables ACS on every rescan, so it is re-applied
# on each run. Set CHAIN_A_P2P=0 to leave ACS alone.
CHAIN_A_P2P="${CHAIN_A_P2P-1}"
CHAIN_B_P2P="${CHAIN_B_P2P-0}"

CHAINFIX=r9700_chainfix   # DKMS package r9700-chainfix/1.0 (src /usr/src/r9700-chainfix-1.0), rebuilt per kernel

chain_is_32g() {
  local c
  for c in "$@"; do
    card_is_32g "$c" || return 1
  done
  return 0
}

# chain_ok <at|-> <card>...: every card has an assigned 32GB BAR0 and, if <at> is set, the first card's
# BAR0 starts at <at> (so the guest's matching addresses stay valid).
chain_ok() {
  local at="$1"; shift
  chain_is_32g "$@" || return 1
  [ "$at" = - ] && return 0
  lspci -vvs "$1" 2>/dev/null | grep -qiE "Region 0: Memory at 0*${at#0x} "
}

# acs_p2p <upstream> <card>...: clear ACS ReqRedir (bit 2) + CmpltRedir (bit 3) on each card's PLX port.
acs_p2p() {
  local up="$1"; shift
  local card port off ctl new
  for card in "$@"; do
    port=$(plx_port_of "$up" "$card"); [ -n "$port" ] || continue
    off=$(lspci -vvv -s "$port" | grep -oE "Capabilities: \[[0-9a-f]+ v1\] Access Control" | grep -oE "\[[0-9a-f]+" | tr -d "[")
    [ -n "$off" ] || { echo "  $port: no ACS capability"; continue; }
    ctl=$(setpci -s "$port" "$(printf %x $((0x$off + 6))).w")
    new=$(printf %04x $(( 0x$ctl & ~0xc )))
    [ "$ctl" = "$new" ] || setpci -s "$port" "$(printf %x $((0x$off + 6))).w=$new"
    echo "  $port ACS ctl $ctl -> $new (switch-local P2P)"
  done
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

# resize_chain <root> <upstream> <base> <limit_root> <limit_up> <at|-> <p2p 0|1> <card>...
resize_chain() {
  local root="$1" up="$2" base="$3" limr="$4" limu="$5" at="$6" p2p="$7"; shift 7
  local cards=("$@") card
  [ ${#cards[@]} -gt 0 ] || { echo "  (no cards)"; return; }
  if [ "$FORCE" != 1 ] && chain_ok "$at" "${cards[@]}"; then
    echo "  ${cards[*]} already 32GB$([ "$at" != - ] && echo " at $at")"
    if [ "$p2p" = 1 ]; then acs_p2p "$up" "${cards[@]}"; fi
    return 0
  fi

  for card in "${cards[@]}"; do unbind_card "$card"; done
  sleep 1
  for card in "${cards[@]}"; do set_rebar_32g "$card"; done

  setpci -s "$root" 28.L=$base 2c.L=$limr 24.W=0001 26.W=fff1
  setpci -s "$up"   28.L=$base 2c.L=$limu 24.W=0001 26.W=fff1

  echo 1 > /sys/bus/pci/devices/0000:$root/remove; sleep 2
  # place_at: with the root port gone, reserve the bus aperture below <at> (anchored on the host bridge
  # <bus>:00.0, which is never removed) so the rescan's first fit lands the chain window at <at>.
  if [ "$at" != - ]; then
    ensure_chainfix && modprobe -r "$CHAINFIX" 2>/dev/null
    modprobe "$CHAINFIX" root_bdf="${root%%:*}:00.0" place_at="$at" || echo "  WARNING: could not reserve below $at"
  fi
  echo 1 > /sys/bus/pci/rescan; sleep 3
  [ "$at" != - ] && modprobe -r "$CHAINFIX" 2>/dev/null

  fix_starved "$root" "$up" "${cards[@]}"

  for card in "${cards[@]}"; do rebind_card "$card"; done
  sleep 1
  for card in "${cards[@]}"; do
    echo "  $card rebar=$(lspci -vvs "$card" | grep -oE 'current size: [0-9]+[MG]B' | head -1)" \
         "region0=$(r=$(lspci -vvs "$card" | grep -m1 -oE 'Region 0: Memory at [0-9a-f]+ .*size=[0-9]+[MG]\]' | sed -E 's/.*at ([0-9a-f]+).*size=([0-9]+[MG]).*/\2@\1/'); echo "${r:-UNASSIGNED}")" \
         "driver=$(basename "$(readlink -f /sys/bus/pci/devices/0000:$card/driver 2>/dev/null)" 2>/dev/null)"
  done
  chain_ok "$at" "${cards[@]}" || echo "  WARNING: not every card on this chain has a 32GB BAR0 at the expected address"
  if [ "$p2p" = 1 ]; then acs_p2p "$up" "${cards[@]}"; fi
  return 0
}

echo "chain A (${CHAIN_A:-none}):"
resize_chain 40:01.1 41:00.0 $CHAIN_A_WIN "${CHAIN_A_AT:--}" "$CHAIN_A_P2P" $CHAIN_A
echo "chain B (${CHAIN_B:-none}):"
resize_chain c0:01.1 c1:00.0 $CHAIN_B_WIN "${CHAIN_B_AT:--}" "$CHAIN_B_P2P" $CHAIN_B
