#!/bin/bash
# Deploy GGZ14/vllm-mxfp4 with unsloth/Qwen3.8-27B-NVFP4 on 2x R9700 (TP=2 auto-detected).
set -eu
cd ~/vllm-mxfp4
export RUNTIME=docker
IMAGE=stilldeadcode/vllm-radiance:0.9.3
MODELS=$HOME/models; mkdir -p "$MODELS"
HF_CACHE=$HOME/.cache/huggingface; mkdir -p "$HF_CACHE"
echo "===== STEP 1: pull image $IMAGE ====="
sudo docker pull "$IMAGE"
hf_get() { # repo dest
  sudo docker run --rm --network=host -e HF_HOME=/root/.cache/huggingface \
    -v "$HF_CACHE":/root/.cache/huggingface -v "$MODELS":/models \
    --entrypoint python3 "$IMAGE" -c 'import sys;from huggingface_hub import snapshot_download;print(snapshot_download(repo_id=sys.argv[1],local_dir=sys.argv[2]))' "$1" "$2"
}
echo "===== STEP 2: download NVFP4 model (~23GB) ====="
[ -f "$MODELS/Qwen3.8-27B-NVFP4/config.json" ] || hf_get unsloth/Qwen3.8-27B-NVFP4 /models/Qwen3.8-27B-NVFP4
echo "===== STEP 3: download DFlash2 drafter (~2GB) ====="
[ -f "$MODELS/Qwen3.8-27B-DFlash2-FP8/config.json" ] || hf_get tcclaviger/Qwen3.8-27B-DFlash2-FP8 /models/Qwen3.8-27B-DFlash2-FP8
echo "===== STEP 4: build libr4d kernels (PREPARE_ONLY) ====="
MODELS="$MODELS" IMAGE="$IMAGE" RUNTIME=docker PREPARE_ONLY=1 ./serve-mxfp4.sh || true
echo "===== STEP 5: serve NVFP4 at TP=2 (detached) ====="
RADIANCE_NVFP4_MXFP4=1 SNAP="$MODELS/Qwen3.8-27B-NVFP4" DRAFTER="$MODELS/Qwen3.8-27B-DFlash2-FP8" \
  NAME=vllmnvfp4 KV_MEM=0 GPU_UTIL=0.95 CACHE="$HOME/.radiance-cache-nvfp4-093" \
  SERVED_NAMES="Qwen3.8-NVFP4 Qwen3.8 Qwen3.6" RADIANCE_USE_R4D_AR=0 DETACH=1 RUNTIME=docker \
  ./serve-mxfp4.sh
echo "===== SERVE LAUNCHED — polling /health ====="
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health 2>/dev/null || echo 000)
  echo "health[$i]=$code"; [ "$code" = "200" ] && { echo "SERVER UP"; break; }; sleep 15
done
echo "===== DEPLOY_DONE ====="
