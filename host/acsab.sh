#!/bin/bash
# Live A/B on one running server: ACS redirect off (switch-local) / on (hairpin) / off. Runs from mgmt VM.
set -u
arm() { # label acsvalue
  ssh root@192.168.0.100 "setpci -s 42:08.0 f2a.w=$2; setpci -s 42:10.0 f2a.w=$2; echo \"[$1] ACS 42:08.0=\$(setpci -s 42:08.0 f2a.w) 42:10.0=\$(setpci -s 42:10.0 f2a.w)\""
  ssh devops@192.168.0.123 "PYTHONUNBUFFERED=1 ~/bb-venv/bin/betterbench run --no-update-check --endpoint http://localhost:8080/v1 --model Qwen3.8 --prefill --no-html --out ~/bb0923/ab-$1-pf.json > ~/bb0923/ab-$1-pf.log 2>&1; PYTHONUNBUFFERED=1 ~/bb-venv/bin/betterbench run --no-update-check --endpoint http://localhost:8080/v1 --model Qwen3.8 --decode --quick --no-html --out ~/bb0923/ab-$1-dec.json > ~/bb0923/ab-$1-dec.log 2>&1; echo \"[$1] done\""
}
arm off1 0011
arm on  001d
arm off2 0011
ssh root@192.168.0.100 'echo "final ACS 42:08.0=$(setpci -s 42:08.0 f2a.w) 42:10.0=$(setpci -s 42:10.0 f2a.w)"; dmesg | tail -300 | grep -ciE "IO_PAGE_FAULT|AER"'
