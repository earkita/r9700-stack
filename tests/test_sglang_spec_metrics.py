import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('spec_decode',Path(__file__).parents[1]/'bench/sglang_spec_decode.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_first_burst_is_not_assumed_one_token():
    result=mod.counter_metrics([(10.,3),(10.1,7),(10.2,11)])
    assert result['post_first_update_tokens']==8
    assert result['decode_tps']==pytest.approx(40.)
    assert result['ttft_seconds']==10.


def test_single_token_target_has_same_definition():
    assert mod.counter_metrics([(2.,1),(2.1,2),(2.2,3)])['decode_tps']==pytest.approx(10.)


def test_non_increasing_counts_are_rejected():
    with pytest.raises(ValueError):mod.counter_metrics([(1.,3),(2.,2)])


def test_natural_eos_keeps_actual_tokens_separate_from_the_budget():
    result=mod.completion_observation(dict(completion_tokens=197,num_retractions=0,
                                          finish_reason=dict(type='stop')),256)
    assert result==dict(actual_tokens=197,budget_tokens=256,budget_truncated=False,natural_eos=True)


def test_budget_cut_is_explicitly_truncated():
    result=mod.completion_observation(dict(completion_tokens=256,num_retractions=0,
                                          finish_reason=dict(type='length')),256)
    assert result['budget_truncated'] and not result['natural_eos']


@pytest.mark.parametrize('meta',[
    dict(completion_tokens=1,num_retractions=0,finish_reason=dict(type='abort')),
    dict(completion_tokens=257,num_retractions=0,finish_reason=dict(type='length')),
    dict(completion_tokens=200,num_retractions=0,finish_reason=dict(type='length')),
    dict(completion_tokens=True,num_retractions=0,finish_reason=dict(type='stop')),
    dict(completion_tokens=200,finish_reason=dict(type='stop')),
    dict(completion_tokens=200,num_retractions=1,finish_reason=dict(type='stop')),
])
def test_invalid_runtime_completion_is_not_a_performance_pass(meta):
    with pytest.raises(ValueError):mod.completion_observation(meta,256)


def test_terminal_native_marker_is_valid_only_at_natural_eos():
    assert mod.audit_native_markers('answer<|user|>\n','stop')
    assert mod.audit_native_markers('partial implementation','length')==[]
    for text,finish in [('answer<|user|>more','stop'),('answer<|user|>','length'),
                        ('answer<|assistant|>','stop'),('answer<|user|><|user|>','stop')]:
        with pytest.raises(ValueError):mod.audit_native_markers(text,finish)
