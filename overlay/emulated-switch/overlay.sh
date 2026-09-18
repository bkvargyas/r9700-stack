# Sourced by serve/serve.sh when OVERLAYS contains "emulated-switch". Appends docker args to MNT.
# Mounts a hostcall-free RCCL (rccl/build-nightly.sh) and hostcall-free vLLM extension copies (patch-hostcall.sh)
# over the stock files in the image. See README.md for why this exists and why it is not part of the product.
SDKLIB=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_libraries/lib
VLIB=/usr/local/lib/python3.12/dist-packages/vllm
RCCL=${RCCL:-$HOME/rccl10/rocm-systems/projects/rccl/build-nightly/librccl.so.1.0}
PATCHED=${PATCHED:-$HOME/p2p-patched-nightly}
[ -f "$RCCL" ] || { echo "overlay emulated-switch: $RCCL missing (run rccl/build-nightly.sh)" >&2; exit 1; }
[ -f "$PATCHED/_rocm_C.abi3.so" ] || { echo "overlay emulated-switch: run patch-hostcall.sh" >&2; exit 1; }
MNT+=(-v "$RCCL:$SDKLIB/librccl.so.1:ro"
      -v "$PATCHED/_rocm_C.abi3.so:$VLIB/_rocm_C.abi3.so:ro"
      -v "$PATCHED/_C_stable_libtorch.abi3.so:$VLIB/_C_stable_libtorch.abi3.so:ro")
echo "overlay emulated-switch: hostcall-free RCCL + vLLM extensions mounted"
