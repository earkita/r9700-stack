#!/bin/bash
# Use the same BetterBench version/corpus/statistics as the published Qwen runs.
# Required measured provenance prevents an unlabelled run becoming a baseline.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
: "${ROCM_VERSION:?Record the runtime ROCm version}"
: "${VLLM_VERSION:?Record the runtime vLLM version/commit}"
: "${MODEL_REVISION:?Record the checkpoint revision}"
: "${POWER_CAP_W:?Record the verified cap on every GPU}"
: "${SERVER_NSEQ:?Record the actual server max-num-seqs (1 for initial baseline)}"
BB=${BB:-betterbench}
OUT=${OUT:-$HERE/results/glm-5.3-flash-baseline.json}
mkdir -p "$(dirname "$OUT")"
# Observe the cap, do not change it. Refuse to label a run with an unverified limit.
GPU_IDS=${GPUS:-0,1,2,3,4,5,6,7}
IFS=, read -r -a GPU_ARRAY <<< "$GPU_IDS"
amd-smi static --gpu "${GPU_ARRAY[@]}" --asic --limit --json > "$OUT.gpus.json"
python3 - "$OUT.gpus.json" "$POWER_CAP_W" <<'PY'
import json, sys
gpus = json.load(open(sys.argv[1]))["gpu_data"]
if len(gpus) != 8:
    raise SystemExit("Expected 8 GPUs for this benchmark profile")
for gpu in gpus:
    asic = gpu["asic"]
    cap = gpu["limit"]["ppt0"]["socket_power_limit"]
    if (asic["target_graphics_version"] != "gfx1201" or "R9700" not in asic["market_name"]
            or cap["unit"] != "W" or float(cap["value"]) != float(sys.argv[2])):
        raise SystemExit(f"GPU {gpu['gpu']}: hardware/power cap mismatch; benchmark not started")
PY
exec "$BB" run --endpoint "${BASE:-http://localhost:8080}/v1" --model "${SERVED_MODEL_NAME:-glm-5.3-flash}" \
  --name "${LABEL:-glm-5.3-flash-baseline}" --config "$HERE/glm-5.3-flash.json" --passes 20 \
  --no-update-check --no-html --out "$OUT" \
  --note gpu_count=8 --note gpu_model=R9700 --note gfx=gfx1201 \
  --note "rocm=$ROCM_VERSION" --note "vllm=$VLLM_VERSION" \
  --note "r9700_stack=$(git -C "$HERE/.." rev-parse HEAD)" \
  --note "r9700_stack_dirty=$(git -C "$HERE/.." status --porcelain | wc -l)" \
  --note checkpoint=amd/GLM-5.3-Flash-Quark-MXFP4 --note "model_revision=$MODEL_REVISION" \
  --note "tp=${TP:-8}" --note "ep=${EP:-1}" --note context=65536 \
  --note "server_max_num_seqs=$SERVER_NSEQ" --note "power_cap_w=$POWER_CAP_W" \
  --note "attention=${ATTENTION_PATH:-upstream}" --note "collective=${COLLECTIVE:-RCCL}" \
  --note speculative=none "$@"
