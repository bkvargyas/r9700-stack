#!/bin/bash
# Optimized Flash-Next: compiled path (persistent cache fixes cubin bug) + HIP tuning + persistent expert LRU
sudo docker rm -f vllmflashnext 2>/dev/null
sudo docker run -d --name vllmflashnext --restart no --privileged --ipc=host --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 --ulimit memlock=-1 \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -e VLLM_PLE_CPU_OFFLOAD=1 -e HIP_VISIBLE_DEVICES=0,1 \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0 \
  -e GPU_MAX_HW_QUEUES=1 -e HSA_ENABLE_INTERRUPT=1 -e HSA_ENABLE_MWAITX=1 \
  -e OMP_NUM_THREADS=8 -e VLLM_ROCM_USE_AITER=0 \
  -v $HOME/models:/models -v $HOME/flashnext-ple:/app/pleoffload \
  -v $HOME/flashnext-cache:/cache/vllm -v $HOME/flashnext-lru:/lru_store \
  tcclaviger/vllm:latest \
  /models/Qwen3.8-Flash-Next-MXFP4-FP8 \
  --served-model-name Qwen3.8-Flash-Next Qwen3.8 --port 8080 \
  --tensor-parallel-size 2 --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.97 \
  --enable-expert-offload --expert-offload-mem 60 --expert-cache-dir /lru_store \
  --ple-nvme-offload --ple-nvme-dir /app/pleoffload --ple-cache-gb 16 --ple-cache-reuse true \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
echo "started (compiled path + opts)"
