import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("prefill",Path(__file__).parents[1]/"bench/sglang_glm_prefill.py")
mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_warm_rate_counts_only_processed_tokens_and_boundary_recomputation():
    meta = dict(prompt_tokens=131200,cached_tokens=130816)
    result = mod.metrics(meta,2.,131200,131072)
    assert result["uncached_tokens"] == 384
    assert result["recomputed_prefix_tokens"] == 256
    assert result["uncached_tokens_per_ttft"] == 192
    assert result["gpu_prefill_seconds"] is None
    assert result["scheduler_prefill_wall_seconds"] is None


def test_cold_requires_zero_hits_and_real_input_count():
    result = mod.metrics(dict(prompt_tokens=8192,cached_tokens=0),20.,8192,0)
    assert result["uncached_tokens_per_ttft"] == 409.6
    with pytest.raises(ValueError):
        mod.metrics(dict(prompt_tokens=8192,cached_tokens=2048),20.,8192,0)


def test_reused_append_or_missing_cache_evidence_is_rejected():
    with pytest.raises(ValueError,match="tail"):
        mod.metrics(dict(prompt_tokens=132096,cached_tokens=131584),1.,132096,131072)
    with pytest.raises(ValueError,match="Cache"):
        mod.metrics(dict(prompt_tokens=132096),1.,132096,131072)


def test_scheduler_wall_is_not_reported_as_gpu_time():
    result = mod.metrics(dict(prompt_tokens=8192,cached_tokens=0,forward_entry_time=10.,prefill_finished_time=18.),9.,8192,0)
    assert result["scheduler_prefill_wall_seconds"] == 8.
    assert result["gpu_prefill_seconds"] is None


def test_a_streamed_token_does_not_turn_abort_or_missing_runtime_data_into_pass():
    for meta in (dict(completion_tokens=1,num_retractions=0,finish_reason=dict(type="abort")),
                 dict(completion_tokens=1,finish_reason=dict(type="length"))):
        with pytest.raises(RuntimeError):
            mod.require_completed(meta)
    mod.require_completed(dict(completion_tokens=1,num_retractions=0,finish_reason=dict(type="length")))


def test_one_chunk_replay_miss_is_explicit_recomputation_not_gpu_hit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(mod.__file__).parent))
    meta = dict(prompt_tokens=8192,cached_tokens=0,cached_tokens_details=None)
    plan = dict(cache_tier="device",max_prefix_recompute_tokens=8192)
    with pytest.raises(ValueError,match="not observed"):
        mod.cache_evidence(meta,"warm",8192,plan)
    plan["allow_warm_cache_miss"] = True
    evidence = mod.cache_evidence(meta,"warm",8192,plan)
    assert evidence["cache_source"] == "none" and not evidence["cache_hit_observed"]
    assert mod.metrics(meta,5.,8192,8192)["recomputed_prefix_tokens"] == 8192
    # The opt-in is for identical replay only; it cannot excuse an appended
    # turn missing its required prefix or override the recomputation bound.
    with pytest.raises(ValueError,match="not observed"):
        mod.cache_evidence(meta,"append-128",8192,plan)
    with pytest.raises(RuntimeError,match="base-prefix"):
        mod.cache_evidence(meta,"warm",32768,plan)


def test_replay_miss_policy_keeps_unknown_and_host_sources_fail_closed(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(mod.__file__).parent))
    plan = dict(cache_tier="device",max_prefix_recompute_tokens=8192,allow_warm_cache_miss=True)
    with pytest.raises(ValueError,match="source evidence"):
        mod.cache_evidence(dict(cached_tokens=4096),"warm",8192,plan)
    with pytest.raises(RuntimeError,match="GPU-only"):
        mod.cache_evidence(dict(cached_tokens=4096,cached_tokens_details=dict(device=0,host=4096)),
                           "warm",8192,plan)
    assert mod.cache_evidence(dict(cached_tokens=4096,cached_tokens_details=dict(device=4096,host=0)),
                              "warm",8192,plan)["cache_source"] == "device"
