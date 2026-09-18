#!/bin/bash
# Rebuild RCCL 2.30.4 (rocm-systems 6b0e43f, TheRock ROCm 10 pin) hostcall-free INSIDE the stock vLLM ROCm 10
# nightly image (pip ROCm SDK layout, Ubuntu 22.04 libstdc++). Same recipe as build-nodl.sh: NDEBUG +
# FAULT_INJECTION/TRACE off + ENABLE_DEVICE_LINKER=OFF (otherwise ncclDevKernel_Generic_* keep hostcall).
set -e
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git >/dev/null)
SDK=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel
for f in $(grep -ho '_IMPORT_PREFIX}/lib/[^" ]*\.a' $SDK/lib/cmake/*/*Targets-release.cmake 2>/dev/null | sed "s|_IMPORT_PREFIX}|$SDK|"); do
  [ -e "$f" ] || { ar rc "$f"; echo "stubbed $f"; }
done
export PATH=$SDK/bin:$SDK/lib/llvm/bin:$PATH ROCM_PATH=$SDK HIP_PATH=$SDK
cd /work/rocm-systems/projects/rccl
grep -q "add_compile_definitions(NDEBUG)" CMakeLists.txt || sed -i '0,/^project(rccl CXX)/s//project(rccl CXX)\nadd_compile_definitions(NDEBUG)/' CMakeLists.txt
mkdir -p build-nightly && cd build-nightly
cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$SDK/bin/hipcc \
  -DGPU_TARGETS=gfx1201 -DAMDGPU_TARGETS=gfx1201 -DBUILD_TESTS=OFF \
  -DFAULT_INJECTION=OFF -DTRACE=OFF -DROCTX=ON -DENABLE_DEVICE_LINKER=OFF -DENABLE_COMPRESS=OFF \
  -DCMAKE_PREFIX_PATH=$SDK -DROCM_PATH=$SDK > /work/cmake-nightly.log 2>&1 || { tail -30 /work/cmake-nightly.log; exit 1; }
time ninja -j${JOBS:-24} > /work/ninja-nightly.log 2>&1 || { tail -30 /work/ninja-nightly.log; exit 1; }
ls -la librccl.so*; echo "hostcall refs: $(grep -c -a hidden_hostcall_buffer librccl.so.1.0 2>/dev/null || grep -c -a hidden_hostcall_buffer librccl.so)"
echo BUILD_DONE
