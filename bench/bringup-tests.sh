#!/bin/bash
# Unit tests for libr9k + plugin pieces inside the stock ROCm 10 dev image (needs free GPUs).
REPO=${REPO:-$HOME/r9700-build/repo}
IMG=${IMG:-r9700/vllm:dev}
sudo docker run --rm --ipc=host --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
  -e HIP_VISIBLE_DEVICES=0 -e TESTS="${TESTS:-test_moe_mxfp4.py test_ple_int6.py test_cache_moe.py test_gemm_fp8.py}" -e R9K_LIB=/opt/r9700/r9700_vllm/kernels/libr9k.so -e PYTHONPATH=/opt/r9700:/opt/r9700/tests \
  -v $REPO:/opt/r9700 --entrypoint bash $IMG -c '
set -o pipefail
cd /opt/r9700/kernels && ./build.sh >/dev/null && cp libr9k.so /opt/r9700/r9700_vllm/kernels/ || exit 1
cd /opt/r9700/tests
for t in $TESTS; do
  echo "=== $t"; timeout 600 python3 $t 2>&1 | grep -vE "^(INFO|WARNING|DEBUG)|Warning" | tail -40; echo "rc=$?"
done'
