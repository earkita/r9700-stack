#!/usr/bin/env bash
# Experimental SGLang main aaf5f397 + LMCache dev 32cbdcb for GLM-5.3-Flash.
# Separate from the qualified v0.5.20/MTP3 profile and vLLM K4.
# Start with text, TP8, BF16 KV by default, Triton DSA (index_kpool=4), gfx1201
# packed-MXFP4 MoE fallback, reference Torch mHC, eager, no draft.
# AITER_FORCE_A8W4 also routes dense MXFP4 layers to the BF16 fallback on
# gfx1201; without it, long prefill selects an unsupported FP4 GEMM.
# DRYRUN=1 prints the exact command. This profile does not stop or replace vLLM.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
image=${IMG:-r9700/sglang:glm53-aaf5f397-lmcache-32cbdcb-syncgtt}
model=${MODEL_HOST:-/mnt/ai/models/glm/GLM-5.3-Flash-Quark-MXFP4}
name=${NAME:-glm53-sglang-main-lmcache}
port=${PORT:-8081}
maxlen=${MAXLEN:-8192}
nseq=${NSEQ:-1}
mem_fraction=${MEM_FRACTION:-0.82}
prefill_chunk=${PREFILL_CHUNK:-2048} # New tokens per chunk; test 2048/4096/8192 separately, 16384 only after memory qualification.
max_prefill_tokens=${MAX_PREFILL_TOKENS:-$prefill_chunk} # Scheduler prefill budget; defaults to the chosen chunk size.
prefill_moe=${R9K_SGLANG_PREFILL_MOE:-packed} # packed=reference; grouped prefill is a separately qualified experiment.
indexer_rows=${R9K_SGLANG_INDEXER_PREFILL_ROWS:-0} # 0=reference; 512/1024/2048 bound KPool logits workspace, without shrinking the scheduler chunk.
observe_requests=${OBSERVE_REQUESTS:-0} # 1=Prometheus metrics and per-request scheduler timing logs; separate from profiler runs.
aiter_a8w4=${AITER_FORCE_A8W4:-1}
gate_scale_group=${R9K_SGLANG_GATE_SCALE_GROUP:-0} # 0=S0, 1=experimental grouped E8M0 gate/up loads.
bf16_dense_gemv=${R9K_SGLANG_BF16_DENSE_GEMV:-0} # 0=S0, 1=experimental M=1 dense GEMV after load-time MXFP4 dequant.
down_bn=${R9K_SGLANG_DOWN_BN:-8} # 8=S0; 64=experimental wider down-projection tile, separate MoE experiment.
moe_backend=${R9K_SGLANG_MOE_BACKEND:-packed} # packed=S0; hip_w4a4=experimental C1 native WMMA, packed prefill.
mhc_sinkhorn=${R9K_SGLANG_MHC_SINKHORN:-0} # 0=reference Torch; 1=decode-only upstream Triton Sinkhorn experiment.
mhc_backend=${R9K_SGLANG_MHC_BACKEND:-} # torch | sinkhorn | vllm_tilelang; full pre is a separate numerical variant.
mhc_post=${R9K_SGLANG_MHC_POST:-torch} # triton=fused FP32 post at M>=128; decode/verify retain Torch. Requires adapter image.
stable_tool_prefix=${R9K_SGLANG_GLM_STABLE_TOOL_PREFIX:-0} # Experimental: retain tool definitions for final JSON with tool_choice=none; preserve prefix cache, without permitting calls.
eagle_sampling=${EAGLE_REJECTION_SAMPLING:-0} # 1=real HIP p/q sampling, requires the complete #37134 backport image; 0 preserves legacy behavior.
decode_graphs=${DECODE_GRAPHS:-0} # 1=experimental full decode graph for C1; prefill stays eager.
hwq=${HWQ:-} # Unset preserves S0. Explicit 1/2/4/8 controls HIP queues as a separate experiment.
server_seed=${SERVER_SEED:-} # Set identically across comparison legs; request sampling seed is separate.
mamba_cache_size=${MAMBA_CACHE_SIZE:-} # Optional KDA/Mamba state slots, not tokens or sequence count; unset=upstream budget split.
kv_cache_dtype=${KV_CACHE_DTYPE:-bfloat16} # fp8_e4m3 is experimental on gfx1201; BF16 remains the qualified rollback.
lmcache_port=${LMCACHE_PORT:-5556}
lmcache_http_port=${LMCACHE_HTTP_PORT:-5557}
lmcache_name=${LMCACHE_NAME:-glm53-sglang-main-cache}
lmcache_ram_gb=${LMCACHE_RAM_GB:-128}
lmcache_disk_gb=${LMCACHE_DISK_GB:-512}
lmcache_root=${LMCACHE_ROOT:-/mnt/ai/cache/sglang-aaf5f397-32cbdcb-syncgtt} # Fresh namespace; the failed async run is never reused.
lmcache_allocator=${LMCACHE_ALLOCATOR:-pinned-gtt} # Proven vLLM ROCm mode; lazy is diagnostic only.
cache_allocator_env=()
cache_allocator_args=()
case "$lmcache_allocator" in
  pinned-gtt)
    cache_allocator_env=(-e HSA_USERPTR_FOR_PAGED_MEM=0)
    cache_allocator_args=(--no-l1-use-lazy)
    ;;
  lazy) ;;
  *) echo 'LMCACHE_ALLOCATOR must be pinned-gtt or lazy.' >&2; exit 2 ;;
esac
mtp_steps=${MTP_STEPS:-0} # 0=target-only; 1..5=in-checkpoint EAGLE chain, steps+1 verify tokens; 4/5 require separate qualification.
if [[ $mtp_steps != 0 ]]; then
  echo 'This SGLang main LMCache connector does not yet support speculative decoding; use MTP_STEPS=0.' >&2
  exit 2
fi
mtp_on=0
mtp_args=()
case "$mtp_steps" in
  0) ;;
  1|2|3|4|5)
    mtp_on=1
    mtp_args=(--speculative-algorithm EAGLE --speculative-num-steps "$mtp_steps"
      --speculative-eagle-topk 1 --speculative-num-draft-tokens "$((mtp_steps + 1))")
    ;;
  *) echo 'MTP_STEPS must be 0, 1, 2, 3, 4 or 5.' >&2; exit 2 ;;
esac
case "$eagle_sampling" in
  0) ;;
  1)
    (( mtp_on == 1 )) || { echo 'EAGLE_REJECTION_SAMPLING=1 requires MTP_STEPS>0.' >&2; exit 2; }
    mtp_args+=(--speculative-use-rejection-sampling)
    ;;
  *) echo 'EAGLE_REJECTION_SAMPLING must be 0 or 1.' >&2; exit 2 ;;
esac
case "$mhc_post" in
  torch|triton) ;;
  *) echo 'R9K_SGLANG_MHC_POST must be torch or triton.' >&2; exit 2 ;;
esac
case "$kv_cache_dtype" in
  bfloat16|fp8_e4m3) ;;
  *) echo 'KV_CACHE_DTYPE must be bfloat16 or fp8_e4m3.' >&2; exit 2 ;;
esac
case "$stable_tool_prefix" in
  0|1) ;;
  *) echo 'R9K_SGLANG_GLM_STABLE_TOOL_PREFIX must be 0 or 1.' >&2; exit 2 ;;
esac
for value in "$prefill_chunk" "$max_prefill_tokens"; do
  [[ $value =~ ^[1-9][0-9]*$ ]] && (( value % 256 == 0 )) || {
    echo 'PREFILL_CHUNK and MAX_PREFILL_TOKENS must be positive multiples of 256.' >&2; exit 2;
  }
done
(( max_prefill_tokens >= prefill_chunk )) || {
  echo 'MAX_PREFILL_TOKENS must be at least PREFILL_CHUNK.' >&2; exit 2;
}
case "$prefill_moe" in
  packed|hip_w4a4_grouped) ;;
  *) echo 'R9K_SGLANG_PREFILL_MOE must be packed or hip_w4a4_grouped.' >&2; exit 2 ;;
esac
case "$indexer_rows" in
  0|512|1024|2048) ;;
  *) echo 'R9K_SGLANG_INDEXER_PREFILL_ROWS must be 0, 512, 1024 or 2048.' >&2; exit 2 ;;
esac
observe_args=()
case "$observe_requests" in
  0) ;;
  1) observe_args=(--enable-metrics --enable-request-time-stats-logging) ;;
  *) echo 'OBSERVE_REQUESTS must be 0 or 1.' >&2; exit 2 ;;
esac
mamba_args=()
trace_args=()
if [[ -n ${MTP_TRACE_DIR:-} ]]; then
  [[ $decode_graphs == 0 && ${MTP_TRACE_POSITION:-} =~ ^[0-9]+$ ]] || {
    echo 'MTP snapshots require eager execution and a nonnegative MTP_TRACE_POSITION.' >&2; exit 2;
  }
  mkdir -p "$MTP_TRACE_DIR"
  trace_args=(-v "$(realpath "$MTP_TRACE_DIR"):/mtp-trace"
    -e R9K_SGLANG_MTP_TRACE_DIR=/mtp-trace -e "R9K_SGLANG_MTP_TRACE_POSITION=$MTP_TRACE_POSITION")
fi
if [[ -n $mamba_cache_size ]]; then
  [[ $mamba_cache_size =~ ^[1-9][0-9]*$ ]] || { echo 'MAMBA_CACHE_SIZE must be a positive integer.' >&2; exit 2; }
  mamba_args=(--max-mamba-cache-size "$mamba_cache_size")
fi
seed_args=()
if [[ -n $server_seed ]]; then
  [[ $server_seed =~ ^[0-9]+$ ]] || { echo 'SERVER_SEED must be a nonnegative integer.' >&2; exit 2; }
  seed_args=(--random-seed "$server_seed")
fi
rocprof_attach=${R9K_SGLANG_ROCP_ATTACH:-0} # Diagnostic only: allow rocprofv3 to attach after startup.
[[ $gate_scale_group == 0 || $gate_scale_group == 1 ]] || {
  echo 'R9K_SGLANG_GATE_SCALE_GROUP must be 0 or 1.' >&2; exit 2;
}
[[ $bf16_dense_gemv == 0 || $bf16_dense_gemv == 1 ]] || {
  echo 'R9K_SGLANG_BF16_DENSE_GEMV must be 0 or 1.' >&2; exit 2;
}
[[ $down_bn == 8 || $down_bn == 64 ]] || {
  echo 'R9K_SGLANG_DOWN_BN must be 8 or 64.' >&2; exit 2;
}
[[ $moe_backend == packed || $moe_backend == hip_w4a4 ]] || {
  echo 'R9K_SGLANG_MOE_BACKEND must be packed or hip_w4a4.' >&2; exit 2;
}
[[ $mhc_sinkhorn == 0 || $mhc_sinkhorn == 1 ]] || {
  echo 'R9K_SGLANG_MHC_SINKHORN must be 0 or 1.' >&2; exit 2;
}
if [[ -z $mhc_backend ]]; then
  mhc_backend=torch
  [[ $mhc_sinkhorn == 0 ]] || mhc_backend=sinkhorn
fi
case "$mhc_backend" in
  torch|sinkhorn|vllm_tilelang) ;;
  *) echo 'R9K_SGLANG_MHC_BACKEND must be torch, sinkhorn or vllm_tilelang.' >&2; exit 2 ;;
esac
if [[ $mhc_sinkhorn == 1 && $mhc_backend != sinkhorn ]]; then
  echo 'Conflicting mHC backend and legacy SINKHORN=1 setting.' >&2; exit 2
fi
graph_args=(--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled)
case "$decode_graphs" in
  0) ;;
  1)
    [[ $nseq == 1 ]] || { echo 'DECODE_GRAPHS=1 is prepared for NSEQ=1 only.' >&2; exit 2; }
    graph_args=(--cuda-graph-config '{"decode":{"backend":"full","bs":[1],"max_bs":1},"prefill":{"backend":"disabled"}}')
    ;;
  *) echo 'DECODE_GRAPHS must be 0 or 1.' >&2; exit 2 ;;
esac
hwq_env=()
case "$hwq" in
  '') ;;
  1|2|4|8) hwq_env=(-e "GPU_MAX_HW_QUEUES=$hwq") ;;
  *) echo 'HWQ must be unset, 1, 2, 4 or 8.' >&2; exit 2 ;;
esac
[[ $rocprof_attach == 0 || $rocprof_attach == 1 ]] || {
  echo 'R9K_SGLANG_ROCP_ATTACH must be 0 or 1.' >&2; exit 2;
}
rocprof_env=()
if [[ $rocprof_attach == 1 ]]; then
  rocprof_env=(-e ROCP_TOOL_ATTACH=1)
fi
cache_dir=${CACHE_DIR:-$root/.runtime/cache/sglang}
lmcache_config_dir=$root/.runtime/sglang-main-lmcache
lmcache_config=$lmcache_config_dir/engine.yaml
for value in "$lmcache_port" "$lmcache_http_port" "$lmcache_ram_gb" "$lmcache_disk_gb"; do
  [[ $value =~ ^[1-9][0-9]*$ ]] || {
    echo 'LMCache ports, RAM and disk sizes must be positive integers.' >&2; exit 2;
  }
done
[[ $lmcache_port != "$lmcache_http_port" && $lmcache_port != "$port" && $lmcache_http_port != "$port" ]] || {
  echo 'SGLang, LMCache request and LMCache HTTP ports must differ.' >&2; exit 2;
}
[[ $lmcache_root = /* ]] || { echo 'LMCACHE_ROOT must be an absolute path.' >&2; exit 2; }
lmcache_disk_spec="{\"type\":\"fs_native\",\"base_path\":\"/cache/disk\",\"max_capacity_gb\":$lmcache_disk_gb,\"eviction\":{\"eviction_policy\":\"LRU\",\"trigger_watermark\":0.9,\"eviction_ratio\":0.1}}"

python3 - "$model" <<'PY'
import json
import pathlib
import sys

model = pathlib.Path(sys.argv[1]).resolve(strict=True)
config = json.loads((model / "config.json").read_text())
assert config.get("model_type") == "glm5_next", "wrong model architecture"
assert config.get("quantization_config", {}).get("quant_method") == "quark", "expected Quark checkpoint"
index = json.loads((model / "model.safetensors.index.json").read_text())
shards = set(index["weight_map"].values())
missing = sorted(shard for shard in shards if not (model / shard).is_file())
assert not missing, f"missing {len(missing)} shards: {missing[:4]}"
print(f"GLM SGLang preflight: {len(shards)} shards, Quark MXFP4, {model}")
PY

args=(
  docker run --runtime=runc -d --name "$name" --ipc=host --network=host --pid=host
  --shm-size 32g --device=/dev/kfd --device=/dev/dri
  --group-add 44 --group-add 991 --ulimit memlock=-1
  --security-opt seccomp=unconfined
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  -e HSA_ENABLE_IPC_MODE_LEGACY=0
  -e R9K_SGLANG_GLM_LMCACHE_SYNC_EVENTS=1
  -e SGLANG_USE_AITER=1
  -e "AITER_FORCE_A8W4=$aiter_a8w4"
  -e "R9K_SGLANG_GATE_SCALE_GROUP=$gate_scale_group"
  -e "R9K_SGLANG_BF16_DENSE_GEMV=$bf16_dense_gemv"
  -e "R9K_SGLANG_DOWN_BN=$down_bn"
  -e "R9K_SGLANG_MOE_BACKEND=$moe_backend"
  -e "R9K_SGLANG_PREFILL_MOE=$prefill_moe"
  -e "R9K_SGLANG_INDEXER_PREFILL_ROWS=$indexer_rows"
  -e "R9K_SGLANG_MHC_SINKHORN=$mhc_sinkhorn"
  -e "R9K_SGLANG_MHC_BACKEND=$mhc_backend"
  -e "R9K_SGLANG_MHC_POST=$mhc_post"
  -e "R9K_SGLANG_GLM_STABLE_TOOL_PREFIX=$stable_tool_prefix"
  -e "R9K_SGLANG_GLM_MTP=$mtp_on"
  "${hwq_env[@]}"
  "${rocprof_env[@]}"
  "${trace_args[@]}"
  -e AITER_JIT_DIR=/root/.cache/sglang/aiter-jit
  -e SGLANG_OPT_USE_TILELANG_MHC_PRE=0
  -e SGLANG_OPT_USE_TILELANG_MHC_POST=0
  -e SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
  -e SGLANG_OPT_FUSE_MHC_POST_PRE=0
  -e SGLANG_OPT_FUSED_KDA_VERIFY=0 # Reference Triton verify first; fused verify needs separate qualification.
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
  -v "$(dirname "$model"):/models:ro"
  -v "$cache_dir:/root/.cache/sglang"
  -v "$lmcache_config:/lmcache/engine.yaml:ro"
  "$image"
  --model-path "/models/$(basename "$model")"
  --served-model-name glm-5.3-flash-sglang
  --host 127.0.0.1 --port "$port"
  --tp-size 8 --max-running-requests "$nseq"
  "${seed_args[@]}"
  "${mamba_args[@]}"
  "${mtp_args[@]}"
  --context-length "$maxlen"
  --chunked-prefill-size "$prefill_chunk" --max-prefill-tokens "$max_prefill_tokens"
  "${observe_args[@]}"
  --mem-fraction-static "$mem_fraction"
  --attention-backend dsa
  --dsa-prefill-backend triton --dsa-decode-backend triton
  --linear-attn-backend triton
  --kv-cache-dtype "$kv_cache_dtype"
  --enable-lmcache --lmcache-config-file /lmcache/engine.yaml
  --moe-runner-backend aiter
  --fp8-gemm-backend triton
  --disable-custom-all-reduce "${graph_args[@]}"
  --reasoning-parser glm45 --tool-call-parser glm47
  --language-only
)

cache_args=(
  docker run --runtime=runc -d --name "$lmcache_name" --ipc=host --network=host
  --shm-size 32g --device=/dev/kfd --device=/dev/dri
  --group-add 44 --group-add 991 --ulimit memlock=-1
  --security-opt seccomp=unconfined
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # Same AMD device map as the model workers.
  -e HSA_ENABLE_IPC_MODE_LEGACY=0 # Match the SGLang exporter; HIP IPC handles must use the same mode on both sides.
  -e R9K_SGLANG_GLM_LMCACHE_SYNC_EVENTS=1 # Export only completed local events on both endpoints.
  "${cache_allocator_env[@]}"
  -v "$lmcache_root:/cache"
  --entrypoint lmcache "$image" server
  --host 127.0.0.1 --port "$lmcache_port"
  --http-host 127.0.0.1 --http-port "$lmcache_http_port"
  --separate-object-groups # GLM has KV, KDA/Mamba and DSA indexer objects with distinct lifetimes.
  --l1-size-gb "$lmcache_ram_gb" --eviction-policy LRU
  "${cache_allocator_args[@]}"
  --l2-adapter "$lmcache_disk_spec"
)

printf '%q ' "${cache_args[@]}"
printf '\n'
printf '%q ' "${args[@]}"
printf '\n'
if [[ ${DRYRUN:-0} == 1 ]]; then
  exit 0
fi
docker image inspect "$image" >/dev/null
[[ $(docker image inspect "$image" --format '{{index .Config.Labels "org.r9700.lmcache.glm-sync-events"}}') == 1 ]] || {
  echo 'Requested synchronized ROCm LMCache events require the patched GLM image.' >&2; exit 2;
}
[[ $(docker image inspect "$image" --format '{{index .Config.Labels "org.r9700.lmcache.native-eviction-locks"}}') == 1 ]] || {
  echo 'Disk LRU requires the pinned LMCache eviction-lock patch.' >&2; exit 2;
}
if [[ $stable_tool_prefix == 1 ]]; then
  [[ $(docker image inspect "$image" --format '{{index .Config.Labels "io.r9700.sglang.glm47-stable-tool-prefix"}}') == experimental-v2 ]] || {
    echo 'Requested stable GLM tool prefix requires an image explicitly containing its adapter (experimental-v2).' >&2; exit 2;
  }
fi
if [[ $mhc_post == triton ]]; then
  [[ $(docker image inspect "$image" --format '{{index .Config.Labels "org.r9700.sglang.mhc-post"}}') == 1 ]] || {
    echo 'Requested fused mHC post requires an image explicitly containing its adapter.' >&2; exit 2;
  }
fi
if [[ $eagle_sampling == 1 ]]; then
  sampling_enabled=$(docker image inspect "$image" --format '{{index .Config.Labels "org.r9700.sglang.rocm-eagle-sampling-enabled"}}')
  sampling_revision=$(docker image inspect "$image" --format '{{index .Config.Labels "org.r9700.sglang.rocm-eagle-sampling"}}')
  # The independently qualified incremental image predates the build-ARG
  # capability label; its label identifies the exact complete merged patch.
  [[ $sampling_enabled == 1 || $sampling_revision == 7eedd57ab0970dab586fa8ab00d4e4de93392ff5 ]] || {
    echo 'Requested HIP EAGLE sampling requires an image explicitly containing its adapter.' >&2; exit 2;
  }
fi
if [[ $indexer_rows != 0 ]]; then
  [[ $(docker image inspect "$image" --format '{{index .Config.Labels "org.r9700.sglang.indexer-prefill-rows"}}') == 1 ]] || {
    echo 'Requested KPool row guard requires an image explicitly containing its adapter.' >&2; exit 2;
  }
fi
if [[ $prefill_moe == hip_w4a4_grouped ]]; then
  [[ $(docker image inspect "$image" --format '{{index .Config.Labels "org.r9700.sglang.grouped-prefill"}}') == 1 ]] || {
    echo 'Requested grouped MoE requires an image explicitly containing its adapter.' >&2; exit 2;
  }
fi
if docker container inspect "$name" >/dev/null 2>&1; then
  echo "Container $name already exists; inspect/stop it before starting another." >&2
  exit 1
fi
if docker container inspect "$lmcache_name" >/dev/null 2>&1; then
  echo "Cache container $lmcache_name already exists; inspect it before starting another." >&2
  exit 1
fi
guard=$root/bench/results/sglang-prefill-optimization-20261015/c4-mtp5-20261017/kernel-guard/status.json
if [[ -f $guard && ${R9K_GPU_RECOVERY_CONFIRMED:-0} != 1 ]]; then
  python3 - "$guard" <<'PY'
import json, sys
state = json.load(open(sys.argv[1]))
if state.get("automatic_gpu_reuse_allowed") is False:
    sys.exit("GPU recovery is still blocked after the last MES fault; gather fresh recovery evidence before setting R9K_GPU_RECOVERY_CONFIRMED=1")
PY
fi
mkdir -p "$cache_dir" "$lmcache_config_dir" "$lmcache_root/disk"
printf 'mp_host: 127.0.0.1\nmp_port: %s\n' "$lmcache_port" > "$lmcache_config"
if python3 - "$lmcache_port" "$lmcache_http_port" "$port" <<'PY'
import socket, sys
for port in map(int, sys.argv[1:]):
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
    except OSError:
        continue
    sys.exit(f"Port {port} is already in use; no containers started.")
PY
then
  :
else
  exit 1
fi
cache_id=$("${cache_args[@]}")
cache_ready=0
for _ in $(seq 1 60); do
  if python3 - "$lmcache_http_port" <<'PY'
import sys, urllib.request
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{int(sys.argv[1])}/healthcheck", timeout=0.5) as response:
        assert response.status == 200
except Exception:
    sys.exit(1)
PY
  then
    cache_ready=1
    break
  fi
  if [[ $(docker inspect "$cache_id" --format '{{.State.Running}}') != true ]]; then
    break
  fi
  sleep 1
done
if [[ $cache_ready != 1 ]]; then
  docker logs "$cache_id" >&2 || true
  docker stop "$cache_id" >/dev/null || true
  echo 'LMCache did not become ready; GLM was not started.' >&2
  exit 1
fi
if ! model_id=$("${args[@]}"); then
  docker stop "$cache_id" >/dev/null || true
  echo 'SGLang did not start; LMCache stopped.' >&2
  exit 1
fi
model_ready=0
for _ in $(seq 1 180); do
  if curl --silent --show-error --fail --max-time 2 "http://127.0.0.1:$port/ready" >/dev/null 2>&1; then
    model_ready=1
    break
  fi
  if [[ $(docker inspect "$model_id" --format '{{.State.Running}}') != true ]]; then
    break
  fi
  sleep 5
done
if [[ $model_ready != 1 ]]; then
  docker logs --tail 120 "$model_id" >&2 || true
  docker stop "$model_id" >/dev/null || true
  docker stop "$cache_id" >/dev/null || true
  echo 'SGLang did not become ready; candidate model and LMCache stopped.' >&2
  exit 1
fi
printf 'LMCache: %s\nSGLang: %s\n' "$cache_id" "$model_id"
