#!/bin/bash
# DeepSeek-V4.1-Flash-1017 at EXL3 2.9 bpw (LibertAIDAI pack, 196.3 GiB) resident on eight R9700s: TP8 + expert parallel
# (48 whole routed experts a rank a layer), Engram tables (2 x 98 GB fp8, from the base checkpoint's shards 47/48) in pinned
# host memory, text only. Weights through quant/exl3.py + quant/exl3_moe.py, names through models/deepseek_v41_exl3.py
# (notes/exl3-plan.md, notes/exl3-layer-map.md; serving defaults justified in notes/exl3-serving.md).
# The model dir is composed of symlinks: bash ~/ds41/exl3/compose-model-dir.sh. Its links are relative (../../ds41/...), so
# on VM100 they land in $HOME/ds41, which is mounted at /ds41 when that directory exists.
# 120 V feed: run `bash ~/gpu-offset.sh -500` before any eight-card launch (2x2200 W PSUs, 8-card prefill trips the SCMP).
# Defaults (notes/exl3-serving.md): HIP graphs, DSpark drafter k=5 with adaptive verification, MAXLEN 32768, NSEQ 4,
# NBT 2048, KVMEM 1.7 GiB (= 329,008 KV tokens), PYTORCH_ALLOC_CONF=expandable_segments:True, capture sizes 1,2,4 + 6 x 1..4.
# Long context: MAXLEN=131072 KVMEM=0.8 (313,831 KV tokens, needles retrieved at 126.7k; notes/exl3-serving.md section 3).
# DSPARK=1 (default) adds the checkpoint's DSpark drafter (mtp.{0,1,2}, block 5); DSPARK=0 for a plain run.
# WARM=1 waits for the server and sends two short decodes and one prefill so the Triton JITs (~7-13 s) and the
# drafter's first-step warm-up land before the first user request. (serve.sh starts the container detached and returns,
# so the warm-up runs after it; the launcher used to exec serve.sh, which made WARM a no-op.)
# DRYRUN=1 prints the docker command without launching. Every knob is overridable from the caller's environment.
# (serve.sh's MTP / DRAFT paths cannot express method "dspark" without a draft model path: it rides in EXTRA instead)
ROOT=$(cd "$(dirname "$(realpath "$0")")/.." && pwd)
mkdir -p "$HOME/exl3-tune"
# DSPARK_K = num_speculative_tokens (default 5; the verify step is DSPARK_K + 1 tokens a stream; 3/4/6 measured slower);
# ADAPTIVE = DSpark's enable_adaptive_verification (default true: +22-28 % prose at 4 streams; false buys back ~6 %
# single-stream prose); SPEC_EXTRA='"k":v,...' appends speculative-config fields (no spaces).
SPEC=; [ "${DSPARK:-1}" = 1 ] && SPEC="--speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":${DSPARK_K:-5},\"enable_adaptive_verification\":${ADAPTIVE:-true}${SPEC_EXTRA:+,$SPEC_EXTRA}}"
# CGSIZES (serve.sh): HIP-graph capture sizes. With the drafter every decode step is a multiple of DSPARK_K + 1 tokens,
# so the default is 1,2,4 + (DSPARK_K + 1) x 1..NSEQ (k=5, NSEQ 4: 1,2,4,6,12,18,24) instead of vLLM's
# 1,2,4,6,8,12,16,24,32,40,48; CGSIZES= (empty) restores vLLM's list (notes/exl3-serving.md: same speed, -0.4 GiB
# of hipMemGetInfo accounting, ~35 MiB of real VRAM).
if [ "${DSPARK:-1}" = 1 ] && [ -z "${CGSIZES+x}" ]; then
  q=$(( ${DSPARK_K:-5} + 1 )); CGSIZES=1,2,4; for i in $(seq 1 ${NSEQ-4}); do CGSIZES=$CGSIZES,$(( q * i )); done
  CGSIZES=$(echo $CGSIZES | tr , '\n' | sort -nu | paste -sd,); export CGSIZES
fi
# ALLOC_CONF: PYTORCH_ALLOC_CONF for the workers. Default expandable_segments:True: -470 MiB of VRAM a card (no
# fragmentation of the load-time temporaries), same speed. ALLOC_CONF= (empty) drops it.
ALLOC_CONF=${ALLOC_CONF-expandable_segments:True}
[ -n "${ALLOC_CONF}" ] && DOCKER_ARGS="${DOCKER_ARGS-} -e PYTORCH_ALLOC_CONF=$ALLOC_CONF"
# MEMPROBE=1: memory audit hooks (bench/ds41/memprobe_ext.py as a stock --worker-extension-cls, dev endpoint
# POST /collective_rpc); client bench/ds41/memprobe.py. Diagnostic only.
MP=; if [ "${MEMPROBE:-0}" = 1 ]; then
  MP="--worker-extension-cls r9k_memprobe.MemProbe"
  DOCKER_ARGS="${DOCKER_ARGS-} -e VLLM_SERVER_DEV_MODE=1 -v ${REPO-$ROOT}/bench/ds41/memprobe_ext.py:/usr/local/lib/python3.12/dist-packages/r9k_memprobe.py:ro"
fi
DS41=; [ -d "$HOME/ds41" ] && DS41="-v $HOME/ds41:/ds41:ro"
env \
  OVERLAYS=${OVERLAYS-emulated-switch} `# host overlay, not the product: VM100 on the .100 PLX box needs the hostcall-free
                                        # RCCL or every collective fails at launch. PATCHED (~/p2p-patched-nightly) must
                                        # match the image's vLLM build.` \
  IMG=${IMG-r9700/vllm:dev} `            # the serving image; since 0.4.0 it carries exllamav3 1.6.6's gfx1201 extension
                                        # (docker/Dockerfile)` \
  REPO=${REPO-$ROOT} `                    # the plugin tree mounted at /opt/r9700: this launcher's own checkout` \
  MODEL=${MODEL-/models/DeepSeek-V4.1-Flash-1017-EXL3} SERVED_NAME=${SERVED_NAME-DeepSeek-V4.1} \
  DOCKER_ARGS="${DOCKER_ARGS-} $DS41 -v $HOME/exl3-tune:/exl3-tune -e EXLLAMAV3_TUNE_CACHE=/exl3-tune" \
                                        `# /ds41: where the composed dir's relative links (../../ds41/...) land;
                                        # /exl3-tune: exllamav3's autotune cache, kept across restarts (load-time warm-ups)` \
  GPUS=${GPUS-0,1,2,3,4,5,6,7} TP=${TP-8} OFFLOAD_GB=0 \
  MTP= `                                  # no MTP: the drafter is DSpark (DSPARK=1), not vLLM's mtp method` \
  MAXLEN=${MAXLEN-32768} NSEQ=${NSEQ-4} \
  NBT=${NBT-2048} `                       # prefill chunk (3.4k tok/s warm prefill at 2048; every soak / needle / GSM8K
                                        # number in notes/exl3-serving.md ran 2048; the topk / candidate buffers and
                                        # the sparse-MLA prefill workspace scale with it)` \
  KVMEM=${KVMEM-1.7} `                      # GiB a rank, fixed (vLLM's profile estimate is unreliable with load-time dequant):
                                        # the largest budget whose launch survives the drafter's graph capture (2.0 and
                                        # 2.2 run card 7 out of memory in the capture's dequant tier) and that keeps the
                                        # worst card ~0.7 GiB free (sysfs) through a 4-stream prefill burst and the
                                        # 15-minute soak (notes/exl3-serving.md section 3);
                                        # ~193k KV tokens a GiB (fp8 SWA + compressed + indexer caches, replicated). For
                                        # MAXLEN=131072 use KVMEM=0.8 (313,831 tokens; 1.0 leaves 0.2 GiB at 128k prompts, 1.5 fails capture).` \
  KVDTYPE=${KVDTYPE-fp8} `                # fp8_ds_mla records (the ROCm V4.1 attention's layout; the probe ran it)` \
  EAGER=${EAGER-0} `                      # HIP graphs (EAGER=1 for debugging; graph capture 70 s with adaptive verification)` \
  PREFIX_CACHE=${PREFIX_CACHE-0} `        # off (ROCm keeps SWA replay off anyway; divergence.py needs it off)` \
  EXTRA="--enable-expert-parallel --engram-config {\"cpu_offload\":true} --tokenizer-mode deepseek_v41
         --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 $SPEC $MP ${EXTRA-}" `# (later flags override serve.sh's
                                        # qwen3 parsers; JSON without spaces: serve.sh word-splits EXTRA)` \
  "${@}" \
  bash "$(dirname "$(realpath "$0")")/serve.sh" || exit $?
if [ "${WARM:-0}" = 1 ]; then
  for i in $(seq 1 120); do curl -s "localhost:${PORT:-8080}/v1/models" >/dev/null 2>&1 && break; sleep 10; done
  python3 - "${PORT:-8080}" "${SERVED_NAME-DeepSeek-V4.1}" <<'PY'
import json, sys, urllib.request
port, model = sys.argv[1], sys.argv[2]
words = " ".join(f"w{i}" for i in range(1500))
for n, m in ((8, 96), (2500, 4), (8, 96)):     # decode (the DSpark verify path JITs on its first steps) + one prefill
    body = {"model": model, "messages": [{"role": "user", "content": words[:n * 2] + " Say ok."}], "max_tokens": m,
            "ignore_eos": True, "chat_template_kwargs": {"thinking": False}}
    urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", json.dumps(body).encode(),
                                                  {"Content-Type": "application/json"}), timeout=600).read()
print("warm-up done")
PY
fi
