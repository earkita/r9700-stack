"""Bounded trace preparation must preserve complete long prefixes."""
import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/"bench"))
spec=importlib.util.spec_from_file_location("prefill_profile",Path(__file__).parents[1]/"bench/sglang_glm_prefill_profile.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_tail_bound_includes_prefix_recomputation():
    case=dict(input_ids=[1]*131072,tail_ids=[2]*8192,max_prefix_recompute_tokens=2048)
    mod.validate_cases([case],270592,10240)
    with pytest.raises(ValueError,match="recomputation"):
        mod.validate_cases([case],270592,8192)


def test_long_prefix_is_never_shortened_for_context_fit():
    case=dict(input_ids=[1]*262144,tail_ids=[2]*8192,max_prefix_recompute_tokens=2048)
    with pytest.raises(ValueError,match="never shorten"):
        mod.validate_cases([case],262400,10240)
    mod.validate_cases([case],270592,10240)


def test_unbounded_cold_trace_is_rejected():
    with pytest.raises(ValueError,match="unbounded"):
        mod.validate_cases([dict(input_ids=[1]*131072)],270592,10240)
    mod.validate_cases([dict(input_ids=[1]*8192)],270592,10240)


def test_one_token_abort_is_not_a_completed_trace_request():
    with pytest.raises(RuntimeError,match="abort"):
        mod.require_completed(dict(completion_tokens=1,finish_reason=dict(type="abort")))
    mod.require_completed(dict(completion_tokens=1,num_retractions=0,finish_reason=dict(type="length")))
    mod.require_completed(dict(completion_tokens=1,num_retractions=0,finish_reason=dict(type="stop")))


def test_capture_keeps_device_correlation_without_retaining_tensor_shapes():
    request = mod.profile_request(dict(trace_dir_container="/traces",
                                      profile_options=dict(with_stack=False, record_shapes=False)), "cold8192")
    assert request["activities"] == ["CPU", "GPU"]
    assert request["record_shapes"] is False and request["with_stack"] is False
    assert request["profile_id"] == "cold8192" and request["merge_profiles"] is False


@pytest.mark.parametrize("options", [None, {}, dict(with_stack=False, record_shapes=True),
                                    dict(with_stack=True, record_shapes=False),
                                    dict(with_stack=False, record_shapes=0)])
def test_unqualified_tensor_reference_tracing_fails_before_request(options):
    with pytest.raises(ValueError, match="shape/stack"):
        mod.profile_request(dict(trace_dir_container="/traces", profile_options=options), "cold8192")
