#!/bin/bash
# GLM-5.3-Flash Quark W4A4 baseline candidate: stock vLLM, TP8, no speculative decoding.
# Profile settings and requirements: notes/glm-overview.md. No performance claim.
# MODEL may be a Hub ID or a container path under /models; MODELS_DIR is the host mount.
# Like the other profiles, arguments are env assignments, e.g. NSEQ=16 PORT=8081.
# Experimental C1 W4A4: pass R9K_GLM_MOE=w4a4 BUILD_KERNELS=1 (stock by default).
# Opt-in C1 graph: also pass EAGER=0 CGMODE=FULL_DECODE_ONLY CGSIZES=1 KVMEM=1.25.
# See notes/glm-testing.md for validation requirements.
exec env \
  IMG="${IMG:-r9700/vllm:dev}" \
  REPO="${REPO:-$(dirname "$(dirname "$(realpath "$0")")")}" \
  MODEL="${MODEL:-amd/GLM-5.3-Flash-Quark-MXFP4}" \
  MODEL_PREFLIGHT=r9700_vllm.models.glm5_next \
  SERVED_MODEL_NAME=glm-5.3-flash NAME="${NAME:-glm53-flash-baseline}" REPLACE=0 \
  GPUS="${GPUS:-0,1,2,3,4,5,6,7}" TP=8 MAXLEN=65536 NSEQ=1 \
  OFFLOAD_GB=0 MTP= DRAFT= SPEC= CHAT_TEMPLATE= OVERLAYS= \
  EAGER=1 PREFIX_CACHE=0 CGMODE= CGSIZES= ATTN= DRAFT_ATTN= \
  REASONING_PARSER=glm45 TOOL_CALL_PARSER=glm47 \
  VLLM_PLUGINS=r9700_glm R9K_GLM_BASELINE=1 R9K_PLATFORM=0 BUILD_KERNELS=0 \
  R9K_EXPERT_CACHE_SLOTS=0 R9K_TARGET_LMHEAD=stock R9K_DRAFT_LMHEAD=stock \
  VLLM_ROCM_USE_AITER=1 UTIL="${UTIL:-0.90}" \
  EXTRA="--disable-custom-all-reduce ${EXTRA:-}" \
  "$@" bash "$(dirname "$(realpath "$0")")/serve.sh"
