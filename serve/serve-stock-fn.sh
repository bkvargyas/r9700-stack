#!/bin/bash
# Qwen3.8-Flash-Next GPTQ on STOCK vLLM (ROCm 10 nightly) + r9700_vllm plugin, 2x R9700 TP2.
# Stage-1 bring-up config: experts partly UVA-offloaded by stock --cpu-offload-params (no expert cache yet),
# PLE int6 table in pinned host via the plugin, bf16 KV (stock QSA), no MTP (stock ROCm k>1 blocked).
# Knobs: OFFLOAD_GB (per rank), MAXLEN, EAGER=1, MTP=n, UTIL, P2P=1 (mount our hostcall-free RCCL), EXTRA="...".
IMG=${IMG:-r9700/vllm:dev}
REPO=${REPO:-$HOME/r9700-build/repo}
SDKLIB=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_libraries/lib
RCCL=$HOME/rccl10/librccl-p2p.so.1.0
MNT=(-v $REPO:/opt/r9700)
[ "${P2P:-1}" = 1 ] && MNT+=(-v $RCCL:$SDKLIB/librccl.so.1:ro -e NCCL_PROTO=Simple) || MNT+=(-e NCCL_P2P_DISABLE=1)
ARGS=()
[ "${EAGER:-0}" = 1 ] && ARGS+=(--enforce-eager)
[ -n "$MTP" ] && ARGS+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $MTP}")
sudo docker rm -f vllmstock 2>/dev/null
sudo docker run -d --name vllmstock --ipc=host --network=host --shm-size 32g \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 --ulimit memlock=-1 \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0,1 -e VLLM_ROCM_USE_AITER=0 -e R9K_LIB=/opt/r9700/r9700_vllm/kernels/libr9k.so \
  "${MNT[@]}" -v $HOME/models:/models -v $HOME/vllmstock-cache:/root/.cache/vllm \
  $IMG /models/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ \
  --served-model-name Qwen3.8 --host 0.0.0.0 --port 8080 \
  --tensor-parallel-size 2 --max-model-len ${MAXLEN:-32768} --max-num-seqs ${NSEQ:-4} \
  --max-num-batched-tokens 4096 --gpu-memory-utilization ${UTIL:-0.92} \
  --cpu-offload-gb ${OFFLOAD_GB:-24} --cpu-offload-params experts \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  "${ARGS[@]}" $EXTRA
echo "started stock vLLM + r9700 plugin (offload ${OFFLOAD_GB:-24} GB/rank, maxlen ${MAXLEN:-32768})"
