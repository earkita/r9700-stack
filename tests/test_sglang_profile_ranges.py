"""CPU-only checks: diagnostic annotations must not change call semantics."""
import importlib.util
from pathlib import Path
import sys
import types

import pytest

spec = importlib.util.spec_from_file_location(
    "profile_ranges", Path(__file__).parents[1] / "r9700_sglang/profile_ranges.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_disabled_preserves_callable_identity_and_avoids_torch(monkeypatch):
    monkeypatch.delenv("R9K_SGLANG_PROFILE_RANGES", raising=False)
    monkeypatch.setitem(sys.modules, "torch.profiler", None)
    def original(value):
        return value
    assert mod.profile_range("cache.lookup")(original) is original


def test_enabled_preserves_arguments_result_and_range_lifetime(monkeypatch):
    monkeypatch.setenv("R9K_SGLANG_PROFILE_RANGES", "1")
    events = []
    class Range:
        def __init__(self, name):
            events.append(("create", name))
        def __enter__(self):
            events.append("enter")
        def __exit__(self, *exc):
            events.append(("exit", exc[0]))
    monkeypatch.setitem(sys.modules, "torch.profiler", types.SimpleNamespace(record_function=Range))
    result = object()
    @mod.profile_range("cache.restore")
    def original(value, *, flag):
        """Original API."""
        assert value is result and flag is True
        events.append("call")
        return value
    assert original(result, flag=True) is result
    assert original.__name__ == "original" and original.__doc__ == "Original API."
    assert events == [("create", "r9700::cache.restore"), "enter", "call", ("exit", None)]


def test_exception_is_not_suppressed(monkeypatch):
    monkeypatch.setenv("R9K_SGLANG_PROFILE_RANGES", "1")
    exited = []
    class Range:
        def __init__(self, name):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *exc):
            exited.append(exc[0])
    monkeypatch.setitem(sys.modules, "torch.profiler", types.SimpleNamespace(record_function=Range))
    @mod.profile_range("kpool.write")
    def fail():
        raise RuntimeError("original failure")
    with pytest.raises(RuntimeError, match="original failure"):
        fail()
    assert exited == [RuntimeError]


def test_invalid_enable_setting_fails_clearly(monkeypatch):
    monkeypatch.setenv("R9K_SGLANG_PROFILE_RANGES", "yes")
    with pytest.raises(ValueError, match="0 or 1"):
        mod.profile_range("cache.lookup")


def test_layer_and_target_draft_identity_without_tensor_inspection(monkeypatch):
    monkeypatch.setenv("R9K_SGLANG_PROFILE_RANGES", "1")
    labels = []
    class Range:
        def __init__(self, name):
            labels.append(name)
        def __enter__(self):
            pass
        def __exit__(self, *exc):
            pass
    monkeypatch.setitem(sys.modules, "torch.profiler", types.SimpleNamespace(record_function=Range))
    @mod.profile_range("layer.forward",attributes=("layer_id","is_nextn"))
    def forward(self):
        return 42
    assert forward(types.SimpleNamespace(layer_id=45,is_nextn=True)) == 42
    assert labels == ["r9700::layer.forward[layer_id=45,is_nextn=True]"]
    with pytest.raises(ValueError,match="Python int/bool"):
        forward(types.SimpleNamespace(layer_id=object(),is_nextn=False))
