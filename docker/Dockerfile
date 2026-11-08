# Stock vLLM ROCm 10 nightly + the r9700_vllm plugin (editable; bind-mount the repo over /opt/r9700 to iterate).
ARG BASE=vllm/vllm-openai-rocm:v0.33.0-rocm100
FROM ${BASE}
# exllamav3's compiled extension for gfx1201 (EXL3 trellis kernels: the DeepSeek-V4.1-Flash EXL3 pack). MIT, built from the
# pinned tag with the image's own hipcc; only the extension is used (exllamav3_ext), so no python deps are pulled.
ARG EXL3_REF=v1.6.6
RUN git clone --depth 1 -b ${EXL3_REF} https://github.com/turboderp-org/exllamav3 /tmp/exllamav3 \
 && git config --global --add safe.directory /tmp/exllamav3 \
 && cd /tmp/exllamav3 && PYTORCH_ROCM_ARCH=gfx1201 MAX_JOBS=${MAX_JOBS:-28} pip wheel --no-build-isolation --no-deps -w /tmp/exl3-dist . \
 && pip install --no-deps /tmp/exl3-dist/exllamav3-*.whl && rm -rf /tmp/exllamav3 /tmp/exl3-dist
# the extension links torch's libc10 without an rpath: import torch first (as the plugin does)
RUN python3 -c "import torch, exllamav3_ext; print('exllamav3_ext ok')"
COPY . /opt/r9700
RUN cd /opt/r9700/kernels && ./build.sh && cp libr9k.so /opt/r9700/r9700_vllm/kernels/ \
 && pip install --no-deps --no-build-isolation -e /opt/r9700 \
 && python3 -c "from importlib.metadata import entry_points as e; print([x for x in e(group='vllm.general_plugins')])"
