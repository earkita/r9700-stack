import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).parents[1] / "r9700_sglang/mtp_names.py"
spec = importlib.util.spec_from_file_location("mtp_names", path)
names = importlib.util.module_from_spec(spec)
spec.loader.exec_module(names)


def mapped(key):
    for old, new in sorted(names.nextn_name_map(45).items(), key=lambda x: len(x[0]), reverse=True):
        if old in key:
            return key.replace(old, new, 1)
    return key


@pytest.mark.parametrize("root", ["model.layers", "model.language_model.layers"])
def test_checkpoint_exclusions_match_actual_nextn_modules(root):
    assert mapped(f"{root}.45.eh_proj") == "model.eh_proj"
    assert mapped(f"{root}.45.mlp.gate") == "model.decoder.mlp.gate"
    assert mapped(f"{root}.45.self_attn.kv_b_proj") == "model.decoder.self_attn.kv_b_proj"
    assert mapped(f"{root}.45.shared_head.norm") == "model.shared_head.norm"
    assert mapped("lm_head") == "model.shared_head.head"


def test_per_layer_overrides_keep_target_and_draft_distinct():
    assert names.checkpoint_name("model.decoder.self_attn.q_a_proj", 45) == "model.language_model.layers.45.self_attn.q_a_proj"
    assert names.checkpoint_name("model.layers.3.self_attn.q_a_proj", 45) == "model.language_model.layers.3.self_attn.q_a_proj"
    assert names.checkpoint_name("model.eh_proj", 45) == "model.eh_proj"
    assert mapped("model.language_model.layers.44.mlp.gate") == "model.language_model.layers.44.mlp.gate"


def test_invalid_layer_count_fails():
    with pytest.raises(ValueError):
        names.nextn_name_map(0)


def test_native_nextn_expert_format_is_checked_completely():
    import copy
    item = dict(weight=dict(dtype="fp8_e4m3", qscheme="per_block", block_size=[128, 128], is_dynamic=False),
                input_tensors=dict(dtype="fp8_e4m3", qscheme="per_group", group_size=128, is_dynamic=True))
    table = {f"model.language_model.layers.45.mlp.experts.{i}.{p}":copy.deepcopy(item)
             for i in range(2) for p in ("gate_proj", "up_proj", "down_proj")}
    assert names.nextn_expert_fp8_config(dict(layer_quant_config=table), 45) == item
    broken = copy.deepcopy(table)
    broken.pop(next(iter(broken)))
    with pytest.raises(ValueError):
        names.nextn_expert_fp8_config(dict(layer_quant_config=broken), 45)
    broken = copy.deepcopy(table)
    next(iter(broken.values()))['weight']['block_size'] = [32, 32]
    with pytest.raises(ValueError):
        names.nextn_expert_fp8_config(dict(layer_quant_config=broken), 45)
