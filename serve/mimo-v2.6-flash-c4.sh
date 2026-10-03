#!/bin/bash
# MiMo-V2.6-Flash-MOPD: TP8, text/tools, C4, BF16 KV and DFlash K7.
# SPEC=0 disables DFlash for the matched control. MAXLEN includes input + output.
# KVMEM=<GiB per GPU> fixes the shared KV budget; KVMEM= restores automatic sizing.
# A fixed KV budget overrides UTIL for cache sizing, independently of MAXLEN/NSEQ.
set -e
serve_dir=$(dirname "$(realpath "$0")")
SPEC=${SPEC:-7}
NSEQ=${NSEQ:-4}
MAXLEN=${MAXLEN:-auto}
AR8=${AR8:-1} # 1: exact TP8 P2P reduction <=256 KiB; RCCL above. 0: RCCL only.
for arg in "$@"; do case "$arg" in SPEC=*) SPEC=${arg#*=} ;; AR8=*) AR8=${arg#*=} ;; NSEQ=*) NSEQ=${arg#*=} ;; MAXLEN=*) MAXLEN=${arg#*=} ;; esac; done
case "$SPEC" in 0|7) ;; *) echo 'SPEC must be 0 or 7' >&2; exit 2 ;; esac
case "$AR8" in 0|1) ;; *) echo 'AR8 must be 0 or 1' >&2; exit 2 ;; esac
case "$NSEQ" in 1|2|4) ;; *) echo 'NSEQ must be 1, 2 or 4' >&2; exit 2 ;; esac
[ "$MAXLEN" = auto ] && MAXLEN=-1
draft=/models/MiMo-V2.6-Flash-MOPD/dflash
[ "$SPEC" = 0 ] && draft=
exec env \
  REPO="${REPO:-$(dirname "$serve_dir")}" \
  IMG="${IMG:-r9700/vllm:mimo-e97573215}" \
  MODELS_DIR="${MODELS_DIR:-$HOME/models/mimo}" \
  MODEL=/models/MiMo-V2.6-Flash-MOPD \
  DRAFT="$draft" SPEC="$SPEC" SPEC_METHOD=dflash MTP= \
  NAME="${NAME:-mimo26-flash-c4}" SERVED_MODEL_NAME=mimo-v2.6-flash-mopd \
  HOST="${HOST:-0.0.0.0}" PORT="${PORT:-8080}" \
  DOCKER_SUDO="${DOCKER_SUDO-}" REPLACE=0 BUILD_KERNELS=0 \
  MODEL_PREFLIGHT=r9700_vllm.models.mimo \
  GPUS=0,1,2,3,4,5,6,7 TP=8 `# Both target and draft use all eight GPUs.` \
  MAXLEN="$MAXLEN" `# auto: maximum ONE request, input + output including thinking.` \
  NSEQ="$NSEQ" `# Active requests share the KV pool; no full-context-per-session reservation.` \
  NBT=8192 UTIL=0.94 OFFLOAD_GB=0 `# Prefill budget; weights and KV stay on GPU.` \
  KVMEM="${KVMEM-7.625}" `# GiB per GPU shared by all requests; empty restores auto sizing.` \
  KV_DTYPE=bfloat16 DRAFT_KV_DTYPE=bfloat16 PREFIX_CACHE=1 \
  EAGER=1 CGMODE=NONE CGSIZES= `# Establish correctness before HIP graphs.` \
  ATTN=TRITON_ATTN_DIFFKV DRAFT_ATTN=TRITON_ATTN \
  SPEC_EXTRA='"draft_tensor_parallel_size":8,"enforce_eager":true' \
  REASONING_PARSER=mimo TOOL_CALL_PARSER=mimo LMONLY=--language-model-only \
  EXTRA='--quantization r9700_mimo_mxfp4 --trust-remote-code --generation-config vllm --disable-custom-all-reduce --no-async-scheduling --block-size 16 --prefix-cache-retention-interval 1280 --max-parallel-loading-workers 1' \
  VLLM_PLUGINS=r9700,r9700_mimo R9K_MIMO=1 R9K_PLATFORM=1 \
  R9K_DISABLE=quant,models,gdn,attn,mtp R9K_FOLD=0 R9K_EXPERT_CACHE_SLOTS=0 \
  R9K_TARGET_LMHEAD=stock R9K_DRAFT_LMHEAD=stock R9K_ARN="$AR8" R9K_AR4=0 \
  R9K_ARN_1S_KB=16 R9K_ARN_MAX_KB=256 `# Plugin AR8; stock vLLM custom AR stays disabled.` \
  VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ROCM_USE_AITER=0 VLLM_DISABLE_PYNCCL=1 \
  VLLM_DIFFKV_FULL_ATTN_SEGMENTS=64 VLLM_DIFFKV_PREFILL_BLOCK_M=64 \
  VLLM_DIFFKV_PREFILL_NUM_WARPS=4 VLLM_DIFFKV_PREFILL_NUM_STAGES=2 \
  VLLM_DIFFKV_PREFILL_TILE=32 \
  P2P=1 HWQ=1 MWAITX=1 PIDNS=--pid=host \
  DOCKER_ARGS='-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e SAFETENSORS_FAST_GPU=1 -e NCCL_SHM_DISABLE=1 -e NCCL_SOCKET_IFNAME=lo' \
  CHAT_TEMPLATE= OVERLAYS= WRAP= PROF=0 \
  "$@" MAXLEN="$MAXLEN" NSEQ="$NSEQ" DRAFT="$draft" bash "$serve_dir/serve.sh"
