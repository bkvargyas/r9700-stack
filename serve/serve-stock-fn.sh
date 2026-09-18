#!/bin/bash
# Qwen3.8-Flash-Next GPTQ on STOCK vLLM (ROCm 10 nightly) + r9700_vllm plugin, 2x R9700 TP2.
# Stage-1 bring-up config: experts partly UVA-offloaded by stock --cpu-offload-params (no expert cache yet),
# PLE int6 table in pinned host via the plugin, bf16 KV (stock QSA), no MTP (stock ROCm k>1 blocked).
# Knobs: OFFLOAD_GB (per rank), MAXLEN, EAGER=1, MTP=n, UTIL, P2P=1 (mount our hostcall-free RCCL), EXTRA="...".
IMG=${IMG:-r9700/vllm:dev}
REPO=${REPO:-$HOME/r9700-build/repo}
SDKLIB=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_libraries/lib
VLIB=/usr/local/lib/python3.12/dist-packages/vllm
# HSA_ENABLE_IPC_MODE_LEGACY=0: the nightly image sets =1, which makes hipIpcGetMemHandle fail ("invalid argument")
# on this box -> no RCCL P2P -> SHM transport, whose proxy round-trips cost ~1-2 ms per all-reduce inside HIP graphs
# (130 ms/step with 99 all-reduces). tcclaviger's image runs =0.
# Hostcall-free RCCL built IN this image (rccl/build-nightly.sh) + hostcall-metadata-patched copies of vLLM's
# _rocm_C / _C_stable_libtorch (p2p/scanhc.sh finds them). Required on the emulated-switch VM topology, where any
# kernel requesting hidden_hostcall_buffer fails with hipErrorIllegalState (PCIe atomics are dropped).
RCCL=${RCCL:-$HOME/rccl10/rocm-systems/projects/rccl/build-nightly/librccl.so.1.0}
PATCHED=${PATCHED:-$HOME/p2p-patched-nightly}
MNT=(-v $REPO:/opt/r9700 -v $RCCL:$SDKLIB/librccl.so.1:ro)
[ -d "$PATCHED" ] && MNT+=(-v $PATCHED/_rocm_C.abi3.so:$VLIB/_rocm_C.abi3.so:ro
                           -v $PATCHED/_C_stable_libtorch.abi3.so:$VLIB/_C_stable_libtorch.abi3.so:ro)
[ "${P2P:-1}" = 1 ] && MNT+=(-e NCCL_PROTO=Simple) || MNT+=(-e NCCL_P2P_DISABLE=1)
# (re)build libr9k.so into the mounted repo when missing or older than any kernel source
SO=$REPO/r9700_vllm/kernels/libr9k.so
if [ ! -f $SO ] || [ -n "$(find $REPO/kernels -name '*.hip' -newer $SO)" ]; then
  sudo docker run --rm --entrypoint bash -v $REPO:/opt/r9700 $IMG -c \
    "cd /opt/r9700/kernels && ./build.sh && cp libr9k.so /opt/r9700/r9700_vllm/kernels/" || exit 1
fi
# torch.compile/AOT cache per plugin configuration: vLLM's cache key does not see R9K_* knobs, and a graph traced
# with different weight layouts fails at runtime ("wrong number of dimensions").
CKEY=$( (env | grep '^R9K_' | sort; echo "$MTP") | md5sum | cut -c1-10)
# recommended defaults (VM with >=256 GB RAM): all experts in host memory, LRU cache on every layer, fp8 LM heads
: ${R9K_EXPERT_CACHE_SLOTS:=270}; : ${R9K_TARGET_LMHEAD:=fp8}; : ${R9K_DRAFT_LMHEAD:=fp8}
export R9K_EXPERT_CACHE_SLOTS R9K_TARGET_LMHEAD R9K_DRAFT_LMHEAD
# forward every R9K_* plugin knob from the caller's environment into the container
for v in $(env | grep -o '^R9K_[A-Z0-9_]*'); do MNT+=(-e "$v=${!v}"); done
ARGS=()
[ "${EAGER:-0}" = 1 ] && ARGS+=(--enforce-eager)
# WRAP=rocprof: rocprofv3 kernel trace, collection window ROCPROF_WINDOW="delay_s:dur_s" after process start
ENTRY=(); PRE=()
if [ "$WRAP" = rocprof ]; then
  mkdir -p $HOME/stock-prof; MNT+=(-v $HOME/stock-prof:/prof)
  ENTRY=(--entrypoint /usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/bin/rocprofv3)
  PRE=(--kernel-trace --memory-copy-trace --stats -f csv -d /prof/rp -o %nid%_%pid%
       --collection-period "${ROCPROF_WINDOW:-600:20}:1" --collection-period-unit sec -- vllm serve)
fi
# PROF=1: torch profiler (POST /start_profile, /stop_profile) -> ~/stock-prof (use with EAGER=1 to see kernels)
[ "${PROF:-0}" = 1 ] && { mkdir -p $HOME/stock-prof; MNT+=(-v $HOME/stock-prof:/prof)
  ARGS+=(--profiler-config '{"profiler": "torch", "torch_profiler_dir": "/prof", "torch_profiler_with_stack": false, "torch_profiler_use_gzip": false}'); }
[ -n "$MTP" ] && ARGS+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $MTP}")
sudo docker rm -f vllmstock 2>/dev/null
sudo docker run -d --name vllmstock --ipc=host --network=host --shm-size 32g \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 --ulimit memlock=-1 \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0,1 -e VLLM_ROCM_USE_AITER=0 -e HSA_ENABLE_IPC_MODE_LEGACY=0 \
  -e GPU_MAX_HW_QUEUES=${HWQ:-1} -e HSA_ENABLE_MWAITX=1 -e OMP_NUM_THREADS=8 -e R9K_LIB=/opt/r9700/r9700_vllm/kernels/libr9k.so \
  "${MNT[@]}" -v $HOME/models:/models -v $HOME/vllmstock-cache-$CKEY:/root/.cache/vllm \
  -v $HOME/vllmstock-triton:/root/.triton \
  "${ENTRY[@]}" $IMG "${PRE[@]}" /models/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ \
  --served-model-name Qwen3.8 --host 0.0.0.0 --port 8080 \
  --tensor-parallel-size 2 --max-model-len ${MAXLEN:-32768} --max-num-seqs ${NSEQ:-4} \
  --max-num-batched-tokens 4096 --gpu-memory-utilization ${UTIL:-0.94} \
  --cpu-offload-gb ${OFFLOAD_GB:-34} --cpu-offload-params experts \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice ${LMONLY---language-model-only} \
  "${ARGS[@]}" $EXTRA
echo "started stock vLLM + r9700 plugin (offload ${OFFLOAD_GB:-34} GB/rank, maxlen ${MAXLEN:-32768})"
