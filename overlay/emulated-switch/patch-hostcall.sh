#!/bin/bash
# Produce hostcall-free copies of vLLM's _rocm_C / _C_stable_libtorch for the pinned image (OUT, default
# ~/p2p-patched-nightly). Rewrites the kernel-argument metadata name "hidden_hostcall_buffer" -> the equal-length
# "hidden_global_offset_x" so the runtime never sets up hostcall for those kernels (none of them calls printf/
# malloc on device; the buffer is only requested because of -O0/assert paths). BINARY PATCH: overlay-only.
set -e
IMG=${IMG:-vllm/vllm-openai-rocm:nightly-rocm100-e9757321527ca1ecd514c07c1418dd2c53da3d19}
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
for f in /out/*.so; do n=$(grep -c -a CCOB $f); [ "$n" != 0 ] && echo "$f: $n compressed offload bundles (CCOB): the byte patch cannot reach their metadata (0 patched is expected); check the families below are gfx9-only"; done
for f in /out/_rocm_C.abi3.so /out/_C_stable_libtorch.abi3.so; do
  echo "$f: $(/h.sh $f | grep -c "^  ") kernels still requesting hostcall, by family:"
  /h.sh $f | grep "^  " | sed -E "s/^  _Z[0-9]+//; s/^  //" | grep -oE "^[A-Za-z_]+[A-Za-z0-9_]{0,24}" | sort | uniq -c | sort -rn | head -6
done'
