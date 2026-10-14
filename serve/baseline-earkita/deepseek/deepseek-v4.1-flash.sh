#!/usr/bin/env bash
# Experimental stage-1 correctness profile; not yet qualified for serving.
# Build with: bash docker/build-deepseek.sh. Preview with: DRYRUN=1 bash <this file>.
set -e
serve_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ "${DRYRUN:-0}" != 1 ]; then
  mkdir -p "${REPO:-$(dirname "$serve_dir")}/.runtime/cache/torch-kernels"
fi
exec env \
  REPO="${REPO:-$(dirname "$serve_dir")}" \
  IMG="${IMG:-r9700/vllm:deepseek-v41-5555a2d8-parser}" \
  MODELS_DIR="${MODELS_DIR:-/mnt/ai/models/deepseek}" \
  MODEL=/models/DeepSeek-V4.1-Flash-Quark-MXFP4 \
  NAME="${NAME:-deepseek41-flash}" SERVED_MODEL_NAME=deepseek-v4.1-flash \
  HOST="${HOST:-127.0.0.1}" PORT="${PORT:-8080}" \
  DOCKER_SUDO="${DOCKER_SUDO-}" REPLACE=0 BUILD_KERNELS=0 \
  MODEL_PREFLIGHT=r9700_vllm.models.deepseek_audit \
  GPUS=0,1,2,3,4,5,6,7 TP=8 `# TP8 non-expert layers; EP8 routed experts (48 complete experts/GPU).` \
  MAXLEN=-1 `# Auto: vLLM chooses context from model/KV capacity; client limits belong in LiteLLM.` \
  NSEQ=4 NBT=512 `# Four concurrent requests share KV; prefill chunk size in tokens.` \
  UTIL=0.95 OFFLOAD_GB=10.5 `# Reference bring-up: 95% GPU budget; exact CPU expert offload per GPU in GiB.` \
  KV_DTYPE=auto KVMEM= `# Let the DeepSeek attention backend choose its cache format and capacity.` \
  PREFIX_CACHE=1 `# Reuse common prefixes in the shared GPU KV pool; ROCm retains SWA blocks.` \
  MTP= DRAFT= SPEC= `# No DSpark/MTP or separate drafter during bring-up.` \
  EAGER=1 CGMODE=NONE CGSIZES= ATTN= `# Stock attention dispatch; no graph capture.` \
  REASONING_PARSER=deepseek_v41 TOOL_CALL_PARSER=deepseek_v41 \
  LMONLY= `# Load the vision tower; TP8 also shards its attention/MLP weights.` \
  EXTRA='--enable-expert-parallel --safetensors-load-strategy lazy --quantization r9700_deepseek_quark --tokenizer-mode deepseek_v41 --generation-config vllm --disable-custom-all-reduce --no-async-scheduling --engram-config={"cpu_offload":true,"dp_shared_memory":false,"use_thp":true} --mm-encoder-tp-mode weights --mm-encoder-attn-backend TRITON_ATTN --limit-mm-per-prompt {"image":4}' `# Lazy maps shards; bounded staging below avoids whole-shard RAM copies. Up to 4 images/request.` \
  DOCKER_ARGS="${DOCKER_ARGS:-} --oom-score-adj=500 -e PYTORCH_KERNEL_CACHE_PATH=/opt/r9700/.runtime/cache/torch-kernels" `# No Docker RAM cap; run the separate host-memory guard (4 GiB).` \
  R9K_DEEPSEEK_MOE=w4a4 `# Packed prefill/decode, 1-2048 rows; reject unsupported calls.` \
  R9K_DEEPSEEK_WO_A_QDQ="${R9K_DEEPSEEK_WO_A_QDQ:-1}" `# Preserve Quark MXFP8 activation quantization in grouped WO_A projection.` \
  R9K_DEEPSEEK_OFFLOAD_MATRICES=1 `# Only routed w13/w2 matrices; scales stay in VRAM.` \
  R9K_DEEPSEEK_STAGE_BUFFER=1 `# Reserve/reuse one projection buffer before automatic KV sizing (540 MiB/rank with resident scales).` \
  R9K_DEEPSEEK_STAGE_MIN_TOKENS=256 `# Whole-projection RAM→GPU copy for large eager batches; included in startup memory profiling.` \
  R9K_DEEPSEEK_PREFILL_SEQS=1 `# Legacy gather path: one context per scratch chunk; NSEQ=4 and NBT=512 stay unchanged.` \
  R9K_DEEPSEEK_PAGED_PREFILL="${R9K_DEEPSEEK_PAGED_PREFILL:-0}" `# Reference upstream gather/combine with #56720/#56638; 1 opts into experimental paged prefill.` \
  R9K_DEEPSEEK_LOAD_BUFFER_MIB=64 `# Reusable CPU-to-GPU staging: 64 MiB/rank, 512 MiB total for TP8.` \
  R9K_DEEPSEEK_LOAD=stream `# Stream tensors from each loaded shard; CPU copies into existing offloaded storage.` \
  VLLM_PLUGINS=r9700_deepseek R9K_DEEPSEEK=1 R9K_PLATFORM=0 `# Only the scoped correctness adapter; stock RCCL.` \
  R9K_EXPERT_CACHE_SLOTS=0 R9K_TARGET_LMHEAD=stock R9K_DRAFT_LMHEAD=stock \
  VLLM_ROCM_USE_AITER=0 VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD=1 \
  CHAT_TEMPLATE= OVERLAYS= WRAP= PROF=0 \
  "$@" bash "$serve_dir/serve.sh"
