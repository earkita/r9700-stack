import pytest
pytest.importorskip('vllm')
from r9700_vllm.compat.deepseek_dspark import quant_mapper


def test_unqualified_speculation_fails_before_weight_loading(monkeypatch):
    from types import SimpleNamespace
    from r9700_vllm.compat.deepseek_dspark import check_speculation
    cfg = SimpleNamespace(method='dspark', num_speculative_tokens=5,
                          enable_adaptive_verification=False, draft_sample_method='probabilistic')
    monkeypatch.setenv('R9K_DEEPSEEK_DSPARK', '0')
    with pytest.raises(RuntimeError, match='requires R9K'):
        check_speculation(cfg)
    monkeypatch.setenv('R9K_DEEPSEEK_DSPARK', '1')
    check_speculation(cfg)
    cfg.enable_adaptive_verification = True
    with pytest.raises(RuntimeError, match='static 5'):
        check_speculation(cfg)
    cfg.method = 'mtp'
    with pytest.raises(RuntimeError, match='only'):
        check_speculation(cfg)


def test_draft_rules_follow_constructor_prefix_not_modulelist_index():
    mapper=quant_mapper()
    expected={
        'mtp.0.main_proj':'model.main_proj',
        'mtp.0.ffn.gate':'model.layers.40.ffn.gate',
        'mtp.1.attn.wkv':'model.layers.41.attn.wkv',
        'mtp.2.ffn.experts.0.w1':'model.layers.42.ffn.experts.0.w1',
        'mtp.2.confidence_head.proj':'model.confidence_head.proj',
        'mtp.2.markov_head.head':'model.markov_head.markov_w2',
    }
    assert {s:mapper.map_name(s) for s in expected}==expected


def test_quark_draft_fp8_and_exclusions_survive_mapping():
    from test_deepseek_quark import checkpoint_quant
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
    from vllm.model_executor.layers.linear import ReplicatedLinear
    cfg=checkpoint_quant()
    fp8=next(iter(cfg['layer_quant_config'].values()))
    cfg['layer_quant_config']={'mtp.0.main_proj':fp8, 'mtp.1.attn.wq_b':fp8}
    cfg['exclude']=['mtp.0.ffn.gate','mtp.2.confidence_head.proj']
    quant=QuarkConfig.from_config(cfg)
    quant.apply_vllm_mapper(quant_mapper())
    assert 'model.layers.40.ffn.gate' in quant.quant_config['exclude']
    assert 'model.confidence_head.proj' in quant.quant_config['exclude']
    for name in ('model.main_proj','model.layers.41.attn.wq_b'):
        assert quant._find_matched_config(name,ReplicatedLinear)['weight']['block_size']==[32,32]
