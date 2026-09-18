# Stock vLLM ROCm 10 nightly + the r9700_vllm plugin (editable; bind-mount the repo over /opt/r9700 to iterate).
ARG BASE=vllm/vllm-openai-rocm:nightly-rocm100-dee37d89115db4c94a820a79a78a7828e141c910
FROM ${BASE}
COPY . /opt/r9700
RUN cd /opt/r9700/kernels && ./build.sh && cp libr9k.so /opt/r9700/r9700_vllm/kernels/ \
 && pip install --no-deps --no-build-isolation -e /opt/r9700 \
 && python3 -c "from importlib.metadata import entry_points as e; print([x for x in e(group='vllm.general_plugins')])"
