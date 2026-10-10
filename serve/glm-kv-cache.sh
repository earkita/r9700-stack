#!/bin/bash
# Experimental GLM prefix-cache service; starts no model and replaces no container.
# Same ROCm/PyTorch/LMCache image as GLM. Run only when ready for offload validation.
# Overrides: bash serve/glm-kv-cache.sh KV_CACHE_DIR=/mnt/ai/cache/ DRYRUN=1
set -euo pipefail
STACK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

KV_CACHE_DIR=${KV_CACHE_DIR:-/mnt/ai/cache/} # Host NVMe directory.
KV_CACHE_NAMESPACE=${KV_CACHE_NAMESPACE:-glm53-e975-fp8-dflash-k4-v2} # Isolate cache layouts/checkpoint revisions.
KV_CACHE_RAM_GB=${KV_CACHE_RAM_GB:-16} # GiB total across TP8, not per GPU; additional process overhead remains.
KV_CACHE_CHUNK=${KV_CACHE_CHUNK:-1152} # Observed GLM block span; revalidate if cache geometry changes.
KV_CACHE_PORT=${KV_CACHE_PORT:-5555} # Local ZMQ port; same value in the GLM profile.
KV_CACHE_METRICS_PORT=${KV_CACHE_METRICS_PORT:-9091}
KV_CACHE_IO_WORKERS=${KV_CACHE_IO_WORKERS:-4}
IMG=${IMG:-r9700/vllm:glm53-lmcache055} # Build docker/Dockerfile.glm-lmcache first; use the same image for GLM.
NAME=${NAME:-glm53-kv-cache}
GPUS=${GPUS:-0,1,2,3,4,5,6,7} # HIP ordinal mapping must match the model workers.
DOCKER_SUDO=${DOCKER_SUDO-}
DRYRUN=${DRYRUN:-0}

for arg in "$@"; do
  case "$arg" in
    KV_CACHE_DIR=*|KV_CACHE_NAMESPACE=*|KV_CACHE_RAM_GB=*|KV_CACHE_CHUNK=*|KV_CACHE_PORT=*|KV_CACHE_METRICS_PORT=*|KV_CACHE_IO_WORKERS=*|IMG=*|NAME=*|GPUS=*|DOCKER_SUDO=*|DRYRUN=*)
      export "$arg" ;;
    *) echo "Unknown setting: ${arg%%=*}" >&2; exit 2 ;;
  esac
done

case "$KV_CACHE_DIR" in /*) ;; *) echo 'KV_CACHE_DIR must be an absolute host path.' >&2; exit 2 ;; esac
[[ "$KV_CACHE_NAMESPACE" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || { echo 'Invalid KV_CACHE_NAMESPACE.' >&2; exit 2; }
# JSON encoding and numeric validation run on the host without importing GPU libraries.
l2_config=$(python3 - "$KV_CACHE_NAMESPACE" "$KV_CACHE_RAM_GB" "$KV_CACHE_CHUNK" "$KV_CACHE_PORT" "$KV_CACHE_METRICS_PORT" "$KV_CACHE_IO_WORKERS" <<'PY'
import json, math, sys
namespace, ram, chunk, port, metrics, workers = sys.argv[1:]
try:
    assert math.isfinite(float(ram)) and float(ram) >= 1
    assert int(chunk) > 0 and int(workers) > 0
    assert 0 < int(port) < 65536 and 0 < int(metrics) < 65536
    assert int(port) != int(metrics)
except (ValueError, AssertionError):
    sys.exit('Invalid cache sizes/ports: RAM >= 1 GiB; chunk/workers > 0; distinct ports 1..65535.')
print(json.dumps(dict(type='fs_native', base_path='/kv-cache/'+namespace,
                     num_workers=int(workers), use_odirect=False)))
PY
)

# Guard the inspected interfaces before LMCache initializes its GPU transfer path.
preflight='import importlib.metadata as m, runpy, sys
if "e97573215" not in m.version("vllm") or m.version("lmcache") != "0.5.5":
    sys.exit("GLM KV offload requires e97573215 / LMCache 0.5.5; use the same candidate image for GLM and cache.")
from r9700_vllm.compat.glm_lmcache_events import install
install()
runpy.run_module("lmcache.v1.multiprocess.server", run_name="__main__")'

# ROCm's non-legacy IPC also needs the producer's network namespace. Sharing
# only IPC/PID namespaces fails _new_shared_cuda even for a 16-KiB tensor.
# Keep these namespaces and HSA mode aligned with serve.sh.
CMD=(docker run -d --name "$NAME" --network=host --ipc=host --pid=host
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991
  --ulimit memlock=-1 -e "HIP_VISIBLE_DEVICES=$GPUS"
  -e HSA_ENABLE_IPC_MODE_LEGACY=0 # Match serve.sh; the nightly image defaults to incompatible legacy IPC.
  -v "$STACK_ROOT:/opt/r9700:ro" -e PYTHONPATH=/opt/r9700
  -v "$KV_CACHE_DIR:/kv-cache" --entrypoint python3 "$IMG" -c "$preflight"
  --host 127.0.0.1 --port "$KV_CACHE_PORT"
  --prometheus-port "$KV_CACHE_METRICS_PORT"
  --chunk-size "$KV_CACHE_CHUNK" --separate-object-groups
  --l1-size-gb "$KV_CACHE_RAM_GB" --l1-use-lazy --l1-init-size-gb 1
  --eviction-policy LRU --l2-store-policy default
  --l2-adapter "$l2_config")
if [ "$DRYRUN" = 1 ]; then
  printf '%q ' "${CMD[@]}"; echo; exit 0
fi
mkdir -p "$KV_CACHE_DIR/$KV_CACHE_NAMESPACE"
# No docker rm/restart: a name collision must fail instead of disturbing a service.
${DOCKER_SUDO} "${CMD[@]}"
