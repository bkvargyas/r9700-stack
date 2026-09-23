#!/bin/bash
# Proxmox hookscript VM100: clean-enumerate the PLX R9700s and set BAR sizes before QEMU opens them.
VMID="$1"; PHASE="$2"
[ "$VMID" = "100" ] || exit 0
case "$PHASE" in
  pre-start|post-stop) echo "[gpu-reset] $PHASE: clean PLX enumerate + BAR fix"; /usr/local/sbin/r9700-barfix.sh ;;
esac
exit 0
