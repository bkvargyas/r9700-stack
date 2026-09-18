# Host-topology overlays

The product is `r9700_vllm` (plugin) + `kernels/` (libr9k) on the **stock** vLLM ROCm image. Nothing in this
directory is part of it. An overlay is a set of binary replacements / env for one specific broken host, mounted
over the stock image at container start by `serve/serve.sh` (`OVERLAYS=name[,name]`). Each overlay says what host
symptom it works around and how to check whether a host still needs it.

## emulated-switch

**Host:** VM100 on the .100 box (both R9700 behind PLX PEX 8747 chips, passed through to a q35 VM).
**Symptom:** any GPU kernel whose code object requests `hidden_hostcall_buffer` fails with
`hipErrorIllegalState` -- the hostcall path needs PCIe atomics, which this passthrough topology drops. Stock RCCL
(`ncclDevKernel_Generic_*`) and several kernels in vLLM's `_rocm_C` / `_C_stable_libtorch` request it.
**Overlay:**
- `rccl/build-nightly.sh` -- RCCL 2.30.4 rebuilt inside the pinned nightly image with NDEBUG, fault-injection/
  trace off and `ENABLE_DEVICE_LINKER=OFF` (no hostcall kernels). Mounted over `librccl.so.1`.
- `patch-hostcall.sh` -- copies of vLLM's two extensions with the `hidden_hostcall_buffer` argument-metadata name
  rewritten to the equal-length `hidden_global_offset_x` (the kernels never use hostcall). Mounted over the stock files.
- `overlay.sh` -- the mounts. `scanhc.sh` / `hcnames.sh` list kernels in a .so that still request hostcall;
  `p2ptest.py` checks P2P IPC; `ab-p2p.*` is the P2P vs no-P2P A/B that motivated it.
**Rebuild on a new image pin:** re-run `rccl/build-nightly.sh` and `patch-hostcall.sh` against the new image.
**Retire when:** the host has working PCIe atomics (bare metal, or a Gen4/Gen5 box without the PLX switch):
run without `OVERLAYS` -- if RCCL init and the first forward pass succeed, the overlay is not needed.

The libr4d P2P all-reduce the plugin uses (`comm/r4d_ar.py`) does not need this overlay; it is product.
