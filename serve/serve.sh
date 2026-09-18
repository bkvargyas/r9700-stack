#!/bin/bash
# Qwen3.8-Flash-Next GPTQ on STOCK vLLM (ROCm 10 nightly image) + the r9700_vllm plugin, 2x R9700 TP2.
# Experts in pinned host memory (stock --cpu-offload-params) + the plugin's device LRU expert cache, PLE int6
# table in pinned host, bf16 KV (stock QSA), MTP via the plugin's allowlist patch.
# Knobs: MODEL (path inside the container, /models/...), OFFLOAD_GB (per rank, 0 = none), MAXLEN, EAGER=1, MTP=n, DRAFT=/models/x SPEC=n, UTIL, NBT, NSEQ, P2P=1, OVERLAYS=..., EXTRA="...",
# plus every R9K_* plugin knob (forwarded). WRAP=rocprof / PROF=1 for profiling.
IMG=${IMG:-r9700/vllm:dev}
REPO=${REPO:-$HOME/r9700-build/repo}
# HSA_ENABLE_IPC_MODE_LEGACY=0: the nightly image sets =1, which makes hipIpcGetMemHandle fail ("invalid argument")
# on this box -> no RCCL P2P -> SHM transport, whose proxy round-trips cost ~1-2 ms per all-reduce inside HIP graphs
# (130 ms/step with 99 all-reduces). tcclaviger's image runs =0.
MNT=(-v $REPO:/opt/r9700)
# Host-topology overlays (NOT the product; binary replacements for specific broken hosts). OVERLAYS=a,b sources
# overlay/<name>/overlay.sh, which appends to MNT. VM100 on the .100 PLX box needs OVERLAYS=emulated-switch.
for o in ${OVERLAYS//,/ }; do
  f=$REPO/overlay/$o/overlay.sh; [ -f "$f" ] || f=$(dirname "$(realpath "$0")")/../overlay/$o/overlay.sh
  [ -f "$f" ] || { echo "unknown overlay $o" >&2; exit 1; }
  source "$f" || exit 1
done
[ "${P2P:-1}" = 1 ] && MNT+=(-e NCCL_PROTO=Simple) || MNT+=(-e NCCL_P2P_DISABLE=1)
# (re)build libr9k.so into the mounted repo when missing or older than any kernel source
SO=$REPO/r9700_vllm/kernels/libr9k.so
if [ ! -f $SO ] || [ -n "$(find $REPO/kernels -name '*.hip' -newer $SO)" ]; then
  sudo docker run --rm --entrypoint bash -v $REPO:/opt/r9700 $IMG -c \
    "cd /opt/r9700/kernels && ./build.sh && cp libr9k.so /opt/r9700/r9700_vllm/kernels/" || exit 1
fi
# torch.compile/AOT cache per plugin configuration: vLLM's cache key does not see R9K_* knobs, and a graph traced
# with different weight layouts fails at runtime ("wrong number of dimensions").
# The plugin's own source is part of the key too: a code change can change weight layouts under the same knobs.
PSRC=$(find $REPO/r9700_vllm -name '*.py' -print0 | sort -z | xargs -0 cat | md5sum | cut -c1-8)
CKEY=$( (env | grep '^R9K_' | sort; echo "${MTP-3}${MODEL:+ $MODEL}${DRAFT:+ $DRAFT $SPEC $DRAFT_ATTN}${ATTN:+ $ATTN} $PSRC") | md5sum | cut -c1-10)
# recommended defaults (VM with >=256 GB RAM): all experts in host memory, LRU cache on every layer, fp8 LM heads
: ${R9K_EXPERT_CACHE_SLOTS:=270}; : ${R9K_TARGET_LMHEAD:=fp8}; : ${R9K_DRAFT_LMHEAD:=fp8}
export R9K_EXPERT_CACHE_SLOTS R9K_TARGET_LMHEAD R9K_DRAFT_LMHEAD
# forward every R9K_* plugin knob from the caller's environment into the container
for v in $(env | grep -o '^R9K_[A-Z0-9_]*'); do MNT+=(-e "$v=${!v}"); done
ARGS=()
# ATTN=<backend> (e.g. TRITON_ATTN) for the target; DRAFT_ATTN=<backend> for a separate drafter (must support
# full cudagraphs or vLLM runs the draft eagerly)
[ -n "$ATTN" ] && ARGS+=(--attention-backend "$ATTN")
# OFFLOAD_GB=0: no expert offload (models that fit in VRAM, e.g. the dense 27B checkpoints)
OFFL=(); [ "${OFFLOAD_GB:-34}" != 0 ] && OFFL=(--cpu-offload-gb ${OFFLOAD_GB:-34} --cpu-offload-params experts)
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
MTP=${MTP-3}
# DRAFT=/models/<drafter> (e.g. a DFlash2 checkpoint) + SPEC=n: separate-drafter speculation instead of MTP
if [ -n "$DRAFT" ]; then
  ARGS+=(--speculative-config "{\"model\": \"$DRAFT\", \"num_speculative_tokens\": ${SPEC:-7}${SPEC_METHOD:+, \"method\": \"$SPEC_METHOD\"}${DRAFT_ATTN:+, \"attention_backend\": \"$DRAFT_ATTN\"}}")
elif [ -n "$MTP" ]; then
  ARGS+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $MTP}")
fi
sudo docker rm -f vllmstock 2>/dev/null
sudo docker run -d --name vllmstock --ipc=host --network=host --shm-size 32g \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 --ulimit memlock=-1 \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0,1 -e VLLM_ROCM_USE_AITER=0 -e HSA_ENABLE_IPC_MODE_LEGACY=0 \
  -e GPU_MAX_HW_QUEUES=${HWQ:-1} -e HSA_ENABLE_MWAITX=1 -e OMP_NUM_THREADS=8 -e R9K_LIB=/opt/r9700/r9700_vllm/kernels/libr9k.so \
  "${MNT[@]}" -v $HOME/models:/models -v $HOME/vllmstock-cache-$CKEY:/root/.cache/vllm \
  -v $HOME/vllmstock-triton:/root/.triton \
  "${ENTRY[@]}" $IMG "${PRE[@]}" ${MODEL:-/models/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ} \
  --served-model-name Qwen3.8 --host 0.0.0.0 --port 8080 \
  --tensor-parallel-size 2 --max-model-len ${MAXLEN:-32768} --max-num-seqs ${NSEQ:-4} \
  --max-num-batched-tokens ${NBT:-4096} --gpu-memory-utilization ${UTIL:-0.94} \
  "${OFFL[@]}" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice ${LMONLY---language-model-only} \
  "${ARGS[@]}" $EXTRA
echo "started stock vLLM + r9700 plugin (offload ${OFFLOAD_GB:-34} GB/rank, maxlen ${MAXLEN:-32768})"
