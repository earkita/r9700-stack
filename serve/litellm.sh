#!/bin/bash
# Optional CPU-only gateway for an already running vLLM server (Linux).
# Usage: bash serve/litellm.sh [NAME=value ...]; DRYRUN=1 prints no secrets.
set -euo pipefail
for arg in "$@"; do
  case "$arg" in
    NAME=*|IMG=*|HOST=*|PORT=*|BACKEND_BASE=*|BACKEND_MODEL=*|ENV_FILE=*|CONFIG=*|DRYRUN=*) export "$arg" ;;
    *) echo "Unsupported option: ${arg%%=*}" >&2; exit 2 ;;
  esac
done
ROOT=$(dirname "$(dirname "$(realpath "$0")")")
IMG=${IMG:-ghcr.io/berriai/litellm:v1.103.0@sha256:bd089afdcd35b894b14a93f9743cdc8b591f82da1a38dd43a010a7b0c9de5fd7}
NAME=${NAME:-r9700-litellm}
HOST=${HOST:-127.0.0.1}                      # Set 0.0.0.0 explicitly for LAN access.
PORT=${PORT:-4000}
BACKEND_BASE=${BACKEND_BASE:-http://127.0.0.1:8080} # Root URL, without /v1.
BACKEND_BASE=${BACKEND_BASE%/}
BACKEND_MODEL=${BACKEND_MODEL:-glm-5.3-flash}
ENV_FILE=${ENV_FILE:-$ROOT/secrets/litellm.env} # Master key and backend key (EMPTY if unused).
if [[ ${DRYRUN:-0} != 1 && ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE; create it as described in proxy/README.md." >&2
  exit 2
fi
if [[ ${DRYRUN:-0} != 1 ]]; then
  python3 - "$ENV_FILE" "$BACKEND_BASE" "$PORT" <<'PY'
from pathlib import Path
import sys
from urllib.parse import urlsplit

values = {}
for line in Path(sys.argv[1]).read_text().splitlines():
    if line.strip() and not line.lstrip().startswith('#'):
        key, sep, value = line.partition('=')
        if sep:
            values[key.strip()] = value.strip()
key = values.get('LITELLM_MASTER_KEY', '')
if not key.startswith('sk-') or len(key) < 20:
    raise SystemExit('ENV_FILE must contain a nonempty, strong LITELLM_MASTER_KEY starting with sk-.')
if not values.get('LITELLM_BACKEND_KEY'):
    raise SystemExit('ENV_FILE must set LITELLM_BACKEND_KEY (EMPTY for an unauthenticated backend).')
base = urlsplit(sys.argv[2])
if (base.scheme not in {'http', 'https'} or not base.hostname or base.path not in {'', '/'}
        or base.username or base.password or base.query or base.fragment):
    raise SystemExit('BACKEND_BASE must be an HTTP(S) root URL, without credentials or /v1.')
if not sys.argv[3].isdigit() or not 1 <= int(sys.argv[3]) <= 65535:
    raise SystemExit('PORT must be between 1 and 65535.')
PY
fi
CONFIG=${CONFIG:-$ROOT/proxy/glm.yaml}
[[ -f "$CONFIG" ]] || { echo "Missing proxy CONFIG" >&2; exit 2; }
CMD=(docker run -d --name "$NAME" --restart unless-stopped --network host
  --env-file "$ENV_FILE"
  -e "LITELLM_BACKEND_BASE=$BACKEND_BASE"
  -e "LITELLM_OPENAI_BASE=$BACKEND_BASE/v1"
  -e "LITELLM_OPENAI_MODEL=hosted_vllm/$BACKEND_MODEL"
  -e "LITELLM_ANTHROPIC_MODEL=anthropic/$BACKEND_MODEL"
  -e PYTHONPATH=/opt/r9700-proxy
  -e LITELLM_LOCAL_MODEL_COST_MAP=True
  -e DO_NOT_TRACK=true
  -v "$ROOT/proxy:/opt/r9700-proxy:ro"
  -v "$(realpath "$CONFIG"):/opt/r9700-config.yaml:ro"
  "$IMG" --config /opt/r9700-config.yaml --host "$HOST" --port "$PORT")
if [[ ${DRYRUN:-0} == 1 ]]; then
  printf '%q ' "${CMD[@]}"; printf '\n'
else
  exec "${CMD[@]}"
fi
