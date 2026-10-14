"""CPU checks for scoped paged-prefill dispatch and profiling allocation."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from r9700_vllm.compat.deepseek_prefill import (
    enabled_for_config, wrap_build, wrap_init, wrap_prefill, wrap_workspace,
)


def config(model="deepseek_v41", eager=True, transfer=None):
    return NS(model_config=NS(hf_config=NS(model_type=model), enforce_eager=eager),
              kv_transfer_config=transfer)


def test_opt_in_and_scope(monkeypatch):
    monkeypatch.delenv("R9K_DEEPSEEK_PAGED_PREFILL", raising=False)
    assert not enabled_for_config(config())
    monkeypatch.setenv("R9K_DEEPSEEK_PAGED_PREFILL", "1")
    assert enabled_for_config(config())
    assert not enabled_for_config(config("glm4_moe"))
    for cfg in (config(eager=False), config(transfer=object())):
        with pytest.raises(RuntimeError, match="eager without KV"):
            enabled_for_config(cfg)
    monkeypatch.setenv("R9K_DEEPSEEK_PAGED_PREFILL", "yes")
    with pytest.raises(ValueError):
        enabled_for_config(config())


def test_profile_does_not_reserve_context_gather(monkeypatch):
    monkeypatch.setenv("R9K_DEEPSEEK_PAGED_PREFILL", "1")
    obj = NS()
    wrap_init(Mock(), config)(obj)
    original = Mock(return_value=[])
    shapes = wrap_workspace(original)
    shapes(obj, 524288, 512, "query")
    assert original.call_args.args[-1] is False
    obj._r9700_paged_prefill = False
    shapes(obj, 524288, 512, "query")
    assert original.call_args.args[-1] is True


@pytest.mark.parametrize("enabled", ["0", "1"])
@pytest.mark.parametrize("keyword", [False, True])
def test_replay_builder_outside_global_config(monkeypatch, enabled, keyword):
    monkeypatch.setenv("R9K_DEEPSEEK_PAGED_PREFILL", enabled)
    cfg, obj = config(), NS()
    def original(self, vllm_config):
        self.vllm_config = vllm_config
    getter = Mock(side_effect=AssertionError("Current vLLM config is not set"))
    wrapped = wrap_init(original, getter)
    if keyword:
        wrapped(obj, vllm_config=cfg)
    else:
        wrapped(obj, cfg)
    getter.assert_not_called()
    assert obj._r9700_paged_prefill is (enabled == "1")


def test_disabled_constructor_without_any_config(monkeypatch):
    monkeypatch.setenv("R9K_DEEPSEEK_PAGED_PREFILL", "0")
    obj = NS()
    getter = Mock(side_effect=AssertionError("No current config"))
    wrap_init(Mock(), getter)(obj)
    getter.assert_not_called()
    assert obj._r9700_paged_prefill is False


def test_metadata_only_adds_prefill_indices():
    result = NS(num_prefill_tokens=7, prefill_swa_indices=Mock(),
                prefill_swa_lens="lengths", decode_swa_ragged_indices="decode")
    builder = NS(_r9700_paged_prefill=True)
    ragged = Mock(return_value=("slots", "offsets"))
    assert wrap_build(Mock(return_value=result), ragged)(builder) is result
    assert result.prefill_swa_ragged_indices == "slots"
    assert result.decode_swa_ragged_indices == "decode"
    result.num_prefill_tokens = 0
    ragged.reset_mock()
    wrap_build(Mock(return_value=result), ragged)(builder)
    ragged.assert_not_called()


def test_disabled_preserves_upstream_call():
    original, reader = Mock(), Mock()
    args = (NS(_r9700_paged_prefill=False), *[object() for _ in range(7)])
    wrap_prefill(original, reader)(*args)
    original.assert_called_once_with(*args, None)
    reader.assert_not_called()


def test_rejects_mxfp8_output_before_reading():
    with pytest.raises(RuntimeError, match="geometry/output"):
        wrap_prefill(Mock(), Mock())(NS(_r9700_paged_prefill=True),
            None, None, None, None, None, None, None, mxfp8_out=object())
