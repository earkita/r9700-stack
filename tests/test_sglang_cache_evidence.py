"""Cache misses remain explicit; missing source evidence never becomes a hit."""
import importlib.util
from pathlib import Path

import pytest

path=Path(__file__).parents[1]/"bench/sglang_cache_evidence.py"
spec=importlib.util.spec_from_file_location("cache_evidence",path)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_zero_hit_replay_requires_explicit_miss_policy():
    with pytest.raises(ValueError,match="not observed"):
        mod.native_cache_source(0,None)
    result=mod.native_cache_source(0,None,allow_miss=True)
    assert result==dict(cache_hit_observed=False,cache_source="none",
                        device_cached_tokens=0,host_cached_tokens=0)
    assert mod.native_cache_source(0,dict(device=0,host=0),allow_miss=True)==result


@pytest.mark.parametrize("count,details",[(1,None),(32,{"device":16,"host":0}),
                                        (0,{"device":1,"host":0}),(True,None),
                                        (4,{"device":True,"host":3})])
def test_unknown_or_conflicting_sources_fail_closed(count,details):
    with pytest.raises(ValueError):
        mod.native_cache_source(count,details,allow_miss=True)


def test_host_and_mixed_cache_never_masquerade_as_gpu_only():
    assert mod.native_cache_source(8,dict(device=8,host=0))["cache_source"]=="device"
    assert mod.native_cache_source(8,dict(device=0,host=8))["cache_source"]=="host"
    assert mod.native_cache_source(8,dict(device=4,host=4))["cache_source"]=="mixed"
