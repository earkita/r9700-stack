#!/bin/bash
# Experimental GLM prefix-cache service; starts no model and replaces no container.
# Same ROCm/PyTorch/LMCache image as GLM. Run only when ready for offload validation.
# Overrides: bash serve/glm-kv-cache.sh KV_CACHE_DIR=/mnt/ai/cache/ DRYRUN=1
set -euo pipefail
STACK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

KV_CACHE_DIR=${KV_CACHE_DIR:-/mnt/ai/cache/} # Host NVMe directory.
KV_CACHE_NAMESPACE=${KV_CACHE_NAMESPACE:-glm53-e975-fp8-dflash-k4-v2} # Isolate cache layouts/checkpoint revisions.
KV_CACHE_RAM_GB=${KV_CACHE_RAM_GB:-16} # GiB total across TP8, not per GPU; additional process overhead remains.
KV_CACHE_ALLOCATOR=${KV_CACHE_ALLOCATOR:-pinned-gtt} # Native HIP allocation without USERPTR; lazy/pinned retained for diagnostics.
KV_CACHE_DISK_GB=${KV_CACHE_DISK_GB:-0} # GiB per namespace; 0 disables L2 eviction (legacy default).
KV_CACHE_DISK_WATERMARK=${KV_CACHE_DISK_WATERMARK:-0.75} # Begin LRU eviction before the asynchronous budget is full.
KV_CACHE_DISK_EVICTION_RATIO=${KV_CACHE_DISK_EVICTION_RATIO:-0.2} # Fraction of tracked objects evicted per pass.
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
    KV_CACHE_DIR=*|KV_CACHE_NAMESPACE=*|KV_CACHE_RAM_GB=*|KV_CACHE_ALLOCATOR=*|KV_CACHE_DISK_GB=*|KV_CACHE_DISK_WATERMARK=*|KV_CACHE_DISK_EVICTION_RATIO=*|KV_CACHE_CHUNK=*|KV_CACHE_PORT=*|KV_CACHE_METRICS_PORT=*|KV_CACHE_IO_WORKERS=*|IMG=*|NAME=*|GPUS=*|DOCKER_SUDO=*|DRYRUN=*)
      export "$arg" ;;
    *) echo "Unknown setting: ${arg%%=*}" >&2; exit 2 ;;
  esac
done

allocator_env=()
case "$KV_CACHE_ALLOCATOR" in
  lazy) allocator_args=(--l1-use-lazy --l1-init-size-gb 1) ;;
  # Keep SHM disabled explicitly: a named shared mapping would still use
  # HIP registration instead of the native host allocator.
  pinned) allocator_args=(--no-l1-use-lazy --shm-name "") ;;
  pinned-gtt)
    allocator_args=(--no-l1-use-lazy --shm-name "")
    # Scope the ROCr allocator policy to the cache process, not the host or model.
    # Explicit HIP registration (lazy) cannot be redirected by this setting.
    allocator_env=(-e HSA_USERPTR_FOR_PAGED_MEM=0)
    ;;
  *) echo 'KV_CACHE_ALLOCATOR must be lazy, pinned or pinned-gtt.' >&2; exit 2 ;;
esac

case "$KV_CACHE_DIR" in /*) ;; *) echo 'KV_CACHE_DIR must be an absolute host path.' >&2; exit 2 ;; esac
[[ "$KV_CACHE_NAMESPACE" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || { echo 'Invalid KV_CACHE_NAMESPACE.' >&2; exit 2; }
# JSON encoding and numeric validation run on the host without importing GPU libraries.
l2_config=$(python3 - "$KV_CACHE_NAMESPACE" "$KV_CACHE_RAM_GB" "$KV_CACHE_CHUNK" "$KV_CACHE_PORT" "$KV_CACHE_METRICS_PORT" "$KV_CACHE_IO_WORKERS" "$KV_CACHE_DISK_GB" "$KV_CACHE_DISK_WATERMARK" "$KV_CACHE_DISK_EVICTION_RATIO" <<'PY'
import json, math, sys
namespace, ram, chunk, port, metrics, workers, disk, watermark, ratio = sys.argv[1:]
try:
    assert math.isfinite(float(ram)) and float(ram) >= 1
    assert int(chunk) > 0 and int(workers) > 0
    assert 0 < int(port) < 65536 and 0 < int(metrics) < 65536
    assert int(port) != int(metrics)
    assert math.isfinite(float(disk)) and float(disk) >= 0
    assert 0 < float(watermark) < 1 and 0 < float(ratio) < 1
except (ValueError, AssertionError):
    sys.exit('Invalid cache sizes/ports: RAM >= 1 GiB; chunk/workers > 0; distinct ports 1..65535.')
config = dict(type='fs_native', base_path='/kv-cache/'+namespace,
              num_workers=int(workers), use_odirect=False)
if float(disk):
    config.update(max_capacity_gb=float(disk), eviction=dict(eviction_policy='LRU',
                  trigger_watermark=float(watermark), eviction_ratio=float(ratio)))
print(json.dumps(config))
PY
)

# Guard the inspected interfaces before LMCache initializes its GPU transfer path.
preflight='import importlib.metadata as m, runpy, sys
if "e97573215" not in m.version("vllm") or m.version("lmcache") != "0.5.5":
    sys.exit("GLM KV offload requires e97573215 / LMCache 0.5.5; use the same candidate image for GLM and cache.")
import inspect, json
adapter = json.loads(sys.argv[sys.argv.index("--l2-adapter")+1])
if adapter.get("max_capacity_gb", 0):
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import NativeConnectorL2Adapter
    if "protected = set(self._locked_keys)" not in inspect.getsource(NativeConnectorL2Adapter.delete):
        sys.exit("Disk LRU requires the patched LMCache native eviction lock guard; rebuild docker/Dockerfile.glm-lmcache.")
from r9700_vllm.compat.glm_lmcache_events import install
install()
runpy.run_module("lmcache.v1.multiprocess.server", run_name="__main__")'

# ROCm's non-legacy IPC also needs the producer's network namespace. Sharing
# only IPC/PID namespaces fails _new_shared_cuda even for a 16-KiB tensor.
# Keep these namespaces and HSA mode aligned with serve.sh.
CMD=(docker run -d --name "$NAME" --network=host --ipc=host --pid=host
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991
  --ulimit memlock=-1 -e "HIP_VISIBLE_DEVICES=$GPUS"
  "${allocator_env[@]}"
  -e HSA_ENABLE_IPC_MODE_LEGACY=0 # Match serve.sh; the nightly image defaults to incompatible legacy IPC.
  -v "$STACK_ROOT:/opt/r9700:ro" -e PYTHONPATH=/opt/r9700
  -v "$KV_CACHE_DIR:/kv-cache" --entrypoint python3 "$IMG" -c "$preflight"
  --host 127.0.0.1 --port "$KV_CACHE_PORT"
  --prometheus-port "$KV_CACHE_METRICS_PORT"
  --chunk-size "$KV_CACHE_CHUNK" --separate-object-groups
  --l1-size-gb "$KV_CACHE_RAM_GB" "${allocator_args[@]}"
  --eviction-policy LRU --l2-store-policy default
  --l2-adapter "$l2_config")
if [ "$DRYRUN" = 1 ]; then
  printf '%q ' "${CMD[@]}"; echo; exit 0
fi
mkdir -p "$KV_CACHE_DIR/$KV_CACHE_NAMESPACE"
if [ "$(python3 -c 'import sys; print(int(float(sys.argv[1]) > 0))' "$KV_CACHE_DISK_GB")" = 1 ] &&
   [ -n "$(find "$KV_CACHE_DIR/$KV_CACHE_NAMESPACE" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo 'Disk LRU requires a fresh empty namespace: this LMCache version does not inventory old files on startup. Nothing deleted.' >&2
  exit 2
fi
# No docker rm/restart: a name collision must fail instead of disturbing a service.
${DOCKER_SUDO} "${CMD[@]}"
