# Stock vLLM ROCm 10 nightly + the r9700_vllm plugin (editable; bind-mount the repo over /opt/r9700 to iterate).
ARG BASE=vllm/vllm-openai-rocm:nightly-rocm100-e9757321527ca1ecd514c07c1418dd2c53da3d19
FROM ${BASE}
COPY . /opt/r9700
ARG BUILD_KERNELS=1
RUN if [ "$BUILD_KERNELS" = 1 ]; then cd /opt/r9700/kernels && ./build.sh && cp libr9k.so /opt/r9700/r9700_vllm/kernels/; fi \
 && pip install --no-deps --no-build-isolation -e /opt/r9700 \
 && python3 -c "from importlib.metadata import entry_points as e; print([x for x in e(group='vllm.general_plugins')])"
