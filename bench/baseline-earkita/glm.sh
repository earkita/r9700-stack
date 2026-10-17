#!/bin/bash
# One unattended run against an ALREADY RUNNING, dedicated benchmark server.
# Does not stop/start services. Install dependencies once with setup.sh.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python="$root/.runtime/baseline-earkita/venv/bin/python"
out=${RUN_DIR:-"$root/bench/results/earkita-glm-$(date -u +%Y%m%dT%H%M%SZ)"}
[[ -x "$python" ]] || { echo 'Run bench/baseline-earkita/setup.sh first.' >&2; exit 1; }
policy=()
[[ -z "${PCIE_POLICY:-}" ]] || policy=(--pcie-policy "$PCIE_POLICY")
corpus=()
[[ -z "${FROZEN_CORPUS:-}" ]] || corpus=(--frozen-corpus "$FROZEN_CORPUS")
"$python" "$root/bench/baseline-earkita/run.py" prepare \
  --out "$out" \
  --model-container "${MODEL_CONTAINER:-earkita-glm-bench}" \
  --cache-container "${CACHE_CONTAINER:-earkita-glm-cache-bench}" \
  --base "${BASE_URL:-http://127.0.0.1:8080}" \
  --cache-metrics "${CACHE_METRICS_URL:-http://127.0.0.1:9091}" \
  --cache-port "${CACHE_PORT:-5555}" \
  --protocol-template "${PROTOCOL_TEMPLATE:-glm.json}" \
  --speculative-profile "${SPECULATIVE_PROFILE:-dflash-k4}" \
  "${policy[@]}" "${corpus[@]}"
exec "$python" "$root/bench/baseline-earkita/run.py" run --out "$out"
