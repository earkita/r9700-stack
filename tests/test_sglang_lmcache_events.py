"""The GLM ROCm cache must acknowledge writes only after local GPU completion."""

import json
import os
from pathlib import Path
import shlex
import subprocess
from unittest.mock import MagicMock, Mock

import pytest

from r9700_sglang.lmcache_events import SynchronizedEvents


PROFILE = (
    Path(__file__).parents[1]
    / "serve/baseline-earkita/glm/glm-5.3-flash-sglang-main-lmcache.sh"
)


def test_completion_acknowledgement_follows_recorded_event():
    cuda = MagicMock()
    backend = SynchronizedEvents(cuda)
    backend.check_event_support("cuda:2")
    event = backend.create_event("cuda:2")
    cuda.Event.assert_called_once_with(interprocess=False)
    stream = object()
    backend.record_event(event, stream)
    event.record.assert_called_once_with(stream)

    event.synchronize.side_effect = RuntimeError("GPU transfer failed")
    with pytest.raises(RuntimeError, match="GPU transfer failed"):
        backend.export_event(event, "cuda:2")
    event.synchronize.side_effect = None
    handle = backend.export_event(event, "cuda:2")
    assert event.synchronize.call_count == 2
    completed = backend.import_event(handle, "cuda:2")
    assert backend.query_event(completed)
    backend.wait_event(completed, stream)
    backend.synchronize_event(completed, "cuda:2")


def test_mixed_event_protocol_fails_closed():
    with pytest.raises(RuntimeError, match="protocol mismatch"):
        SynchronizedEvents(Mock()).import_event(b"foreign-hip-event", "cuda:2")


def _dry_run(tmp_path, **overrides):
    model = tmp_path / "glm"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(
        json.dumps({"model_type": "glm5_next", "quantization_config": {"quant_method": "quark"}})
    )
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "shard.safetensors"}})
    )
    (model / "shard.safetensors").touch()
    env = dict(os.environ, DRYRUN="1", MODEL_HOST=str(model))
    env.update(overrides)
    return subprocess.run(["bash", str(PROFILE)], env=env, text=True, capture_output=True)


def test_cache_and_model_use_same_completion_protocol_and_pinned_gtt(tmp_path):
    result = _dry_run(tmp_path)
    assert result.returncode == 0, result.stderr
    cache, model = map(shlex.split, result.stdout.splitlines()[-2:])
    assert "R9K_SGLANG_GLM_LMCACHE_SYNC_EVENTS=1" in cache
    assert "R9K_SGLANG_GLM_LMCACHE_SYNC_EVENTS=1" in model
    assert "HSA_ENABLE_IPC_MODE_LEGACY=0" in cache
    assert "HSA_ENABLE_IPC_MODE_LEGACY=0" in model
    assert "HSA_USERPTR_FOR_PAGED_MEM=0" in cache
    assert "HSA_USERPTR_FOR_PAGED_MEM=0" not in model
    assert "--no-l1-use-lazy" in cache
    assert "--separate-object-groups" in cache
    assert "--speculative-algorithm" not in model
    assert "sglang-aaf5f397-32cbdcb-syncgtt:/cache" in " ".join(cache)


def test_lazy_allocator_is_explicit_and_mtp_is_rejected(tmp_path):
    result = _dry_run(tmp_path, LMCACHE_ALLOCATOR="lazy")
    assert result.returncode == 0, result.stderr
    cache = shlex.split(result.stdout.splitlines()[-2])
    assert "--no-l1-use-lazy" not in cache
    assert "HSA_USERPTR_FOR_PAGED_MEM=0" not in cache

    result = _dry_run(tmp_path, MTP_STEPS="3")
    assert result.returncode == 2
    assert "does not yet support speculative decoding" in result.stderr


def test_old_image_is_rejected_before_any_container_is_started(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$DOCKER_TEST_LOG"\n'
        'if [[ "$1 $2" == "image inspect" ]]; then\n'
        '  [[ "$*" == *--format* ]] && printf "<no value>\\n"\n'
        '  exit 0\nfi\nexit 99\n'
    )
    docker.chmod(0o755)
    log = tmp_path / "docker.log"
    result = _dry_run(
        tmp_path,
        DRYRUN="0",
        IMG="old-async-image",
        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
        DOCKER_TEST_LOG=str(log),
    )
    assert result.returncode == 2
    assert "synchronized ROCm LMCache events" in result.stderr
    assert all(not line.startswith("run ") for line in log.read_text().splitlines())
