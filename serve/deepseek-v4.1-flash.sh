#!/usr/bin/env bash
# Experimental stage-1 correctness profile; not yet qualified for serving.
# Build with: bash docker/build-deepseek.sh. Preview with: DRYRUN=1 bash <this file>.
set -e
serve_dir=$(dirname "$(realpath "$0")")
exec env \
  REPO="${REPO:-$(dirname "$serve_dir")}" \
  IMG="${IMG:-r9700/vllm:deepseek-v41-18f8f960}" \
  MODELS_DIR="${MODELS_DIR:-/mnt/ai/models/deepseek}" \
  MODEL=/models/DeepSeek-V4.1-Flash-Quark-MXFP4 \
  NAME="${NAME:-deepseek41-flash}" SERVED_MODEL_NAME=deepseek-v4.1-flash \
  HOST="${HOST:-127.0.0.1}" PORT="${PORT:-8080}" \
  DOCKER_SUDO="${DOCKER_SUDO-}" REPLACE=0 BUILD_KERNELS=0 \
  MODEL_PREFLIGHT=r9700_vllm.models.deepseek_audit \
  GPUS=0,1,2,3,4,5,6,7 TP=8 `# TP8; no expert parallelism.` \
  MAXLEN=8192 NSEQ=1 NBT=512 `# Small correctness run; input + output context.` \
  UTIL=0.94 OFFLOAD_GB=10 `# Trial: 10 GiB/rank = 80 GiB total; not a measured minimum.` \
  KV_DTYPE=auto KVMEM= PREFIX_CACHE=0 `# Let the DeepSeek attention backend choose its cache format.` \
  MTP= DRAFT= SPEC= `# No DSpark/MTP or separate drafter during bring-up.` \
  EAGER=1 CGMODE=NONE CGSIZES= ATTN= `# Stock attention dispatch; no graph capture.` \
  REASONING_PARSER=deepseek_v41 TOOL_CALL_PARSER=deepseek_v41 \
  LMONLY=--language-model-only `# Do not allocate the vision tower.` \
  EXTRA='--quantization r9700_deepseek_quark --tokenizer-mode deepseek_v41 --generation-config vllm --disable-custom-all-reduce --no-async-scheduling --max-parallel-loading-workers 1 --engram-config={"cpu_offload":true,"dp_shared_memory":false,"use_thp":true}' \
  VLLM_PLUGINS=r9700_deepseek R9K_DEEPSEEK=1 R9K_PLATFORM=0 `# Only the scoped correctness adapter; stock RCCL.` \
  R9K_EXPERT_CACHE_SLOTS=0 R9K_TARGET_LMHEAD=stock R9K_DRAFT_LMHEAD=stock \
  VLLM_ROCM_USE_AITER=0 VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD=1 \
  CHAT_TEMPLATE= OVERLAYS= WRAP= PROF=0 \
  "$@" bash "$serve_dir/serve.sh"
