#!/bin/bash
set -e
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git >/dev/null)
# runtime image lacks some static libs its cmake configs reference; stub them (throwaway container)
for f in $(grep -ho '_IMPORT_PREFIX}/lib/[^" ]*\.a' /opt/rocm/lib/cmake/*/*Targets-release.cmake | sed 's|_IMPORT_PREFIX}|/opt/rocm|'); do
  [ -e "$f" ] || { ar rc "$f"; echo "stubbed $f"; }
done
export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:/opt/vllm/bin:$PATH ROCM_PATH=/opt/rocm
cd /work/rocm-systems/projects/rccl
mkdir -p build && cd build
cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
  -DGPU_TARGETS=gfx1201 -DAMDGPU_TARGETS=gfx1201 -DBUILD_TESTS=OFF \
  -DFAULT_INJECTION=OFF -DTRACE=OFF -DROCTX=ON -DENABLE_COMPRESS=OFF \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DROCM_PATH=/opt/rocm
time ninja -j8
echo BUILD_DONE
