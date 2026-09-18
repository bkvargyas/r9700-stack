#!/bin/bash
# Flash-Next GPTQ (int6 PLE) on 2x R9700, TP=2, :dev image.
# Expert offload with image-default memory sizing (no --expert-offload-mem), 8 GiB PLE row cache.
# P2P: rebuilt RCCL 2.30.4 (rocm-systems 6b0e43f, ROCm10, gfx1201, NDEBUG, no device linker => no hostcall)
#      bind-mounted over the image's librccl; requires the VM's emulated PCIe switch topology + XanMod guest.
RCCL=$HOME/rccl10/librccl-p2p.so.1.0
P2P_MOUNT=()
# The emulated switch drops PCIe atomics, so ANY kernel whose metadata requests hidden_hostcall_buffer fails
# (hipErrorIllegalState). The image's gdn_hip / clav_ar / vllm _rocm_C extensions ship device asserts; ~/p2p-patched
# holds copies with that metadata kind byte-swapped to hidden_global_offset_x (same length; slot gets 0). Only
# effect: a device assert that fires would fault instead of printing.
SP=/opt/vllm/lib/python3.14/site-packages
PP=$HOME/p2p-patched
[ "${P2P:-1}" = 1 ] && P2P_MOUNT=(-v $RCCL:/opt/rocm/core-10.0/lib/librccl.so.1.0:ro -e NCCL_PROTO=Simple
  -v $PP/gdn_hip_C.cpython-314-x86_64-linux-gnu.so:$SP/gdn_hip/gdn_hip_C.cpython-314-x86_64-linux-gnu.so:ro
  -v $PP/clav_ar_ext.cpython-314-x86_64-linux-gnu.so:$SP/clav_ar_ext.cpython-314-x86_64-linux-gnu.so:ro
  -v $PP/_rocm_C.abi3.so:$SP/vllm/_rocm_C.abi3.so:ro)
# MEM=<GiB> sets --expert-offload-mem (unset = image auto plan); CG=default drops the [4]-only cudagraph list.
# NOP2P=1: keep patched libs but force non-P2P transport (fair A/B on the emulated-switch topology).
EXTRA=(); ENVX=(); [ "$NOP2P" = 1 ] && { EXTRA+=(--disable-custom-all-reduce); ENVX+=(-e NCCL_P2P_DISABLE=1); }
[ -n "$MEM" ] && EXTRA+=(--expert-offload-mem $MEM)
CGCFG=(--compilation-config '{"cudagraph_capture_sizes": [4], "max_cudagraph_capture_size": 4}'); [ "$CG" = default ] && CGCFG=()
# XENV="K=V K2=V2": extra container env (kernel ablation toggles). PROF=1: enable torch profiler (/start_profile, /stop_profile) -> ~/fn-prof.
for kv in $XENV; do ENVX+=(-e "$kv"); done
[ "$PROF" = 1 ] && { mkdir -p ~/fn-prof; ENVX+=(-v $HOME/fn-prof:/prof); EXTRA+=(--profiler-config '{"profiler": "torch", "torch_profiler_dir": "/prof", "torch_profiler_with_stack": false, "torch_profiler_use_gzip": false}'); }
# WRAP=rocprof: run the server under rocprofv3 kernel tracing (output ~/fn-prof/rp, finalized on docker stop).
ENTRY=(); PRE=()
[ "$WRAP" = rocprof ] && { mkdir -p ~/fn-prof; ENVX+=(-v $HOME/fn-prof:/prof); ENTRY=(--entrypoint /opt/rocm/bin/rocprofv3)
  PRE=(--kernel-trace --memory-copy-trace --stats -f csv -d /prof/rp -o %nid%_%pid% -- /app/tools/image_entrypoint.sh); }
mkdir -p ~/flashnext-cache-gptq ~/flashnext-ple-gptq ~/flashnext-tunableop
sudo docker rm -f vllmflashnext 2>/dev/null
sudo docker run -d --name vllmflashnext --restart unless-stopped --privileged --ipc=host --network=host \
  --shm-size 32g --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 --ulimit memlock=-1 \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -e VLLM_PLE_CPU_OFFLOAD=1 -e HIP_VISIBLE_DEVICES=0,1 \
  -e VLLM_CACHE_ROOT=/cache/vllm -e TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0 \
  -e GPU_MAX_HW_QUEUES=1 -e HSA_ENABLE_INTERRUPT=1 -e HSA_ENABLE_MWAITX=1 \
  -e OMP_NUM_THREADS=8 -e VLLM_ROCM_USE_AITER=0 \
  -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_RECORD_UNTUNED=0 \
  "${P2P_MOUNT[@]}" "${ENVX[@]}" \
  -v $HOME/models:/models -v $HOME/flashnext-ple-gptq:/app/pleoffload \
  -v $HOME/flashnext-cache-gptq:/cache/vllm -v $HOME/flashnext-tunableop:/tunableop \
  "${ENTRY[@]}" tcclaviger/vllm:dev "${PRE[@]}" \
  /models/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ \
  --served-model-name Qwen3.8-Flash-Next Qwen3.8 --host 0.0.0.0 --port 8080 \
  --tensor-parallel-size 2 --max-model-len 262144 --max-num-seqs 16 \
  --enable-chunked-prefill --max-num-batched-tokens 4096 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.95 \
  --enable-expert-offload "${EXTRA[@]}" \
  --ple-nvme-offload --ple-nvme-dir /app/pleoffload --ple-cache-gb 8 --ple-cache-reuse true \
  --tool-call-parser qwen3_coder --enable-auto-tool-choice --reasoning-parser qwen3 \
  --override-generation-config '{"max_tokens": 65536, "temperature": 0.8, "top_p": 0.95, "top_k": 40, "presence_penalty": 1}' \
  "${CGCFG[@]}" \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --hf-overrides '{"text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 1.0, "original_max_position_embeddings": 262144, "mrope_section": [11, 11, 10], "mrope_interleaved": true, "partial_rotary_factor": 0.25, "rope_theta": 10000000}}}'
echo "started Flash-Next GPTQ (P2P=${P2P:-1} MEM=${MEM:-auto} CG=${CG:-4} NOP2P=${NOP2P:-0} PROF=${PROF:-0} XENV=${XENV:-none} WRAP=${WRAP:-none})"
