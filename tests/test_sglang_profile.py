"""CPU-only dry-run checks: no Docker or GPU initialization."""
import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest

PROFILE = Path(__file__).parents[1] / "serve/baseline-earkita/glm/glm-5.3-flash-sglang.sh"


def launch(tmp_path, **overrides):
    model = tmp_path / "checkpoint"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps({"model_type": "glm5_next", "quantization_config": {"quant_method": "quark"}}))
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "shard.safetensors"}}))
    (model / "shard.safetensors").touch()
    env = dict(os.environ, MODEL_HOST=str(model), DRYRUN="1", PREFILL_CHUNK="2048", MAX_PREFILL_TOKENS="2048", MTP_STEPS="0", OBSERVE_REQUESTS="0", DECODE_GRAPHS="0",R9K_SGLANG_PREFILL_MOE="packed",R9K_SGLANG_INDEXER_PREFILL_ROWS="0")
    env.update(overrides)
    return subprocess.run(["bash", str(PROFILE)], env=env, text=True, capture_output=True)


def command(result):
    assert result.returncode == 0, result.stderr
    return shlex.split(result.stdout.splitlines()[-1])


def test_default_prefill_and_speculation_unchanged(tmp_path):
    args = command(launch(tmp_path))
    assert args.count("--runtime=runc")==1  # Mixed AMD/NVIDIA host: avoid the host's default runtime.
    assert args[args.index("--chunked-prefill-size") + 1] == "2048"
    assert args[args.index("--max-prefill-tokens") + 1] == "2048"
    assert "--speculative-algorithm" not in args
    assert args[args.index("--kv-cache-dtype") + 1] == "bfloat16"
    assert "--enable-metrics" not in args
    assert "R9K_SGLANG_MHC_POST=torch" in args
    assert "--speculative-use-rejection-sampling" not in args


def test_fp8_kv_is_explicit_and_does_not_change_speculation(tmp_path):
    args = command(launch(tmp_path, KV_CACHE_DTYPE="fp8_e4m3", MTP_STEPS="3"))
    assert args[args.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
    assert args[args.index("--speculative-num-steps") + 1] == "3"
    assert args[args.index("--speculative-num-draft-tokens") + 1] == "4"


def test_unsupported_kv_dtype_fails_before_docker(tmp_path):
    result = launch(tmp_path, KV_CACHE_DTYPE="fp8_e5m2")
    assert result.returncode == 2
    assert "KV_CACHE_DTYPE must be" in result.stderr


@pytest.mark.parametrize("chunk", [2048, 4096, 8192, 16384])
def test_explicit_chunk_and_request_observation(tmp_path, chunk):
    args = command(launch(tmp_path, PREFILL_CHUNK=str(chunk), MAX_PREFILL_TOKENS=str(chunk), OBSERVE_REQUESTS="1"))
    assert args[args.index("--chunked-prefill-size") + 1] == str(chunk)
    assert args[args.index("--max-prefill-tokens") + 1] == str(chunk)
    assert "--enable-request-time-stats-logging" in args


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_draft_chain_has_correct_verify_budget(tmp_path, depth):
    args = command(launch(tmp_path, MTP_STEPS=str(depth)))
    assert args[args.index("--speculative-num-steps") + 1] == str(depth)
    assert args[args.index("--speculative-num-draft-tokens") + 1] == str(depth + 1)


@pytest.mark.parametrize("settings", [{"PREFILL_CHUNK": "2049"}, {"PREFILL_CHUNK": "8192"}, {"MTP_STEPS": "6"}, {"OBSERVE_REQUESTS": "yes"}])
def test_invalid_configuration_is_rejected(tmp_path, settings):
    assert launch(tmp_path, **settings).returncode != 0


def test_indexer_workspace_limit_does_not_shrink_prefill_chunk(tmp_path):
    args=command(launch(tmp_path,R9K_SGLANG_INDEXER_PREFILL_ROWS="1024",PREFILL_CHUNK="8192",MAX_PREFILL_TOKENS="8192"))
    assert "R9K_SGLANG_INDEXER_PREFILL_ROWS=1024" in args
    assert "R9K_SGLANG_PREFILL_MOE=packed" in args
    assert args[args.index("--chunked-prefill-size")+1]=="8192"


@pytest.mark.parametrize("settings",[{"R9K_SGLANG_PREFILL_MOE":"auto"},{"R9K_SGLANG_INDEXER_PREFILL_ROWS":"8192"}])
def test_unknown_operator_configuration_is_rejected(tmp_path,settings):
    assert launch(tmp_path,**settings).returncode!=0


@pytest.mark.parametrize("feature",[{"R9K_SGLANG_INDEXER_PREFILL_ROWS":"1024"},{"R9K_SGLANG_PREFILL_MOE":"hip_w4a4_grouped"},{"R9K_SGLANG_MHC_POST":"triton"},{"MTP_STEPS":"3","EAGLE_REJECTION_SAMPLING":"1"},{"R9K_SGLANG_GLM_STABLE_TOOL_PREFIX":"1"}])
def test_requested_adapter_missing_from_image_fails_before_docker_run(tmp_path,feature):
    # A CPU-only Docker stub ensures an older image cannot silently ignore the
    # positive feature flag and start a misleading comparison on the GPU.
    binaries=tmp_path/"bin";binaries.mkdir()
    docker=binaries/"docker"
    docker.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$DOCKER_TEST_LOG"\nif [[ "$1 $2" == "image inspect" ]]; then printf "<no value>\\n"; exit 0; fi\nexit 99\n')
    docker.chmod(0o755);log=tmp_path/"docker.log"
    result=launch(tmp_path,DRYRUN="0",IMG="adapter-absent",PATH=str(binaries)+os.pathsep+os.environ["PATH"],DOCKER_TEST_LOG=str(log),**feature)
    assert result.returncode==2
    assert "requires an image explicitly containing its adapter" in result.stderr
    assert all(not line.startswith("run ") for line in log.read_text().splitlines())


def test_explicit_sampling_and_post_do_not_change_draft_depth(tmp_path):
    args=command(launch(tmp_path,MTP_STEPS="3",EAGLE_REJECTION_SAMPLING="1",R9K_SGLANG_MHC_POST="triton"))
    assert args.count("--speculative-use-rejection-sampling")==1
    assert args[args.index("--speculative-num-draft-tokens")+1]=="4"
    assert "R9K_SGLANG_MHC_POST=triton" in args


def test_stable_tool_prefix_is_explicit_and_does_not_enable_speculation(tmp_path):
    args=command(launch(tmp_path,R9K_SGLANG_GLM_STABLE_TOOL_PREFIX="1"))
    assert args.count("R9K_SGLANG_GLM_STABLE_TOOL_PREFIX=1")==1
    assert "--speculative-algorithm" not in args


@pytest.mark.parametrize("settings",[{"EAGLE_REJECTION_SAMPLING":"1"},{"EAGLE_REJECTION_SAMPLING":"auto"},{"R9K_SGLANG_MHC_POST":"auto"},{"R9K_SGLANG_GLM_STABLE_TOOL_PREFIX":"auto"}])
def test_invalid_sampling_or_post_rejected(tmp_path,settings):
    assert launch(tmp_path,**settings).returncode!=0
