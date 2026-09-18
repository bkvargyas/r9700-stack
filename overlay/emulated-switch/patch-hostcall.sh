#!/bin/bash
# Produce hostcall-free copies of vLLM's _rocm_C / _C_stable_libtorch for the pinned image (OUT, default
# ~/p2p-patched-nightly). Rewrites the kernel-argument metadata name "hidden_hostcall_buffer" -> the equal-length
# "hidden_global_offset_x" so the runtime never sets up hostcall for those kernels (none of them calls printf/
# malloc on device; the buffer is only requested because of -O0/assert paths). BINARY PATCH: overlay-only.
set -e
IMG=${IMG:-vllm/vllm-openai-rocm:nightly-rocm100-dee37d89115db4c94a820a79a78a7828e141c910}
OUT=${OUT:-$HOME/p2p-patched-nightly}
mkdir -p "$OUT"
docker run --rm --entrypoint bash -v "$OUT":/out -v "$(dirname "$(realpath "$0")")/hcnames.sh":/h.sh "$IMG" -c '
V=/usr/local/lib/python3.12/dist-packages/vllm
cp $V/_rocm_C.abi3.so $V/_C_stable_libtorch.abi3.so /out/
python3 - <<PY
import glob
a, b = b"hidden_hostcall_buffer", b"hidden_global_offset_x"
assert len(a) == len(b)
for f in glob.glob("/out/*.so"):
    d = open(f, "rb").read(); n = d.count(a); open(f, "wb").write(d.replace(a, b)); print(n, "patched", f)
PY
echo "kernels still requesting hostcall:"; /h.sh /out/_rocm_C.abi3.so /out/_C_stable_libtorch.abi3.so | grep -c "^  " || true'
