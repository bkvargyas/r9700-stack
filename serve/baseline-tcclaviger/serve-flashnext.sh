#!/bin/bash
# Qwen3.8-Flash-Next-MXFP4-FP8 TP2 on 2x R9700, expert-offload-to-RAM + PLE-offload, recipe tp2-expert-mem-40gb-ple-cache-16gb
sudo docker rm -f vllmflashnext 2>/dev/null
sudo docker run -d --name vllmflashnext --restart unless-stopped --privileged --ipc=host --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
  --ulimit memlock=-1 \
  -e VLLM_PLE_CPU_OFFLOAD=1 -e HIP_VISIBLE_DEVICES=0,1 \
  -v $HOME/models:/models -v $HOME/flashnext-ple:/app/pleoffload \
  tcclaviger/vllm:latest \
  /models/Qwen3.8-Flash-Next-MXFP4-FP8 \
  --served-model-name Qwen3.8-Flash-Next Qwen3.8 \
  --port 8080 \
  --tensor-parallel-size 2 --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.95 \
  --enable-expert-offload --expert-offload-mem 40 \
  --ple-nvme-offload --ple-nvme-dir /app/pleoffload --ple-cache-gb 16 --ple-cache-reuse true \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --compilation-config '{"cudagraph_capture_sizes": [4,8,12,16,20,24,28,32], "max_cudagraph_capture_size": 32, "inductor_compile_config": {"combo_kernels": false, "benchmark_combo_kernel": false}}'
echo "started vllmflashnext"
