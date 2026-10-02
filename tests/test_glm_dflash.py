"""CPU gates for completed auxiliary states and disjoint draft cache regions."""
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm.models.glm5next.common import model as glm
from vllm.model_executor.models.interfaces import supports_eagle3
from r9700_vllm.models.glm_dflash import (
    GlmDFlashModel, R9kGlmDFlashForCausalLM, R9kGlmDFlashForConditionalGeneration,
)


class Layer(torch.nn.Module):
    n = 2

    def __init__(self, state=None):
        super().__init__()
        self.state = state

    def forward(self, pos, hidden, residual, post, comb):
        return self.state if self.state is not None else (hidden, None, None, None)

    def hc_post(self, hidden, residual, post, comb):
        return torch.einsum("sij,sih->sjh", comb, residual) + post * hidden.unsqueeze(1)


def model(monkeypatch, layers):
    m = object.__new__(GlmDFlashModel)
    torch.nn.Module.__init__(m)
    m.layers = torch.nn.ModuleList(layers)
    m._active_layers = tuple(layers)
    m.start_layer, m.end_layer = 0, len(layers)
    m.is_sequence_parallel = False
    m.norm = torch.nn.Identity()
    monkeypatch.setattr(glm, "get_pp_group", lambda: NS(is_first_rank=True, is_last_rank=True))
    monkeypatch.setattr(glm, "hc_contract", lambda x, n: x.mean(1))
    return m


def run(m, hidden):
    return m(None, torch.arange(len(hidden)), None, hidden)


def test_aux_completes_mhc_and_keeps_target_unchanged(monkeypatch):
    hidden = torch.tensor([[3., 4.]])
    state = (hidden, torch.tensor([[[5., 6.], [7., 8.]]]),
             torch.tensor([[[.5], [1.5]]]), torch.tensor([[[.75, .25], [.25, .75]]]))
    originals = [t.clone() for t in state]
    first = Layer(state)
    m = model(monkeypatch, [first, Layer()])
    baseline = run(m, hidden)
    m._set_aux_hidden_state_layers((1, 2))
    actual, aux = run(m, hidden)
    expected = (torch.einsum("sij,sih->sjh", state[3].double(), state[1].double())
                + state[2].double() * hidden.double().unsqueeze(1)).mean(1)
    torch.testing.assert_close(aux[0].double(), expected)
    torch.testing.assert_close(aux[1], hidden)
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    for t, copy in zip(state, originals):
        torch.testing.assert_close(t, copy, rtol=0, atol=0)
    # Changing taps must remove old hooks, not duplicate the captures.
    m._set_aux_hidden_state_layers((2,))
    assert len(run(m, hidden)[1]) == 1


def test_aux_gathers_and_trims_sequence_parallel_padding(monkeypatch):
    h = torch.arange(20).reshape(5, 4).float()
    m = model(monkeypatch, [Layer()])
    m.is_sequence_parallel = True
    m._set_aux_hidden_state_layers((1,))
    padded = torch.cat([h, torch.zeros(1, 4)])
    monkeypatch.setattr(glm, "sp_shard", lambda x: padded[:3])
    monkeypatch.setattr(glm, "sp_all_gather", lambda x: padded)
    result, aux = run(m, h)
    torch.testing.assert_close(result, h)
    torch.testing.assert_close(aux[0], h)


def test_wrapper_interfaces(monkeypatch):
    assert supports_eagle3(R9kGlmDFlashForCausalLM)
    assert supports_eagle3(R9kGlmDFlashForConditionalGeneration)
    taps = (6, 15, 25, 34, 43)
    for cls in (R9kGlmDFlashForCausalLM, R9kGlmDFlashForConditionalGeneration):
        wrapper = object.__new__(cls)
        torch.nn.Module.__init__(wrapper)
        core = model(monkeypatch, [Layer() for _ in range(45)])
        if cls is R9kGlmDFlashForCausalLM:
            wrapper.model = core
        else:
            wrapper.language_model = NS(model=core, embed_input_ids=lambda _: None,
                                        forward=lambda input_ids, positions: None)
            wrapper._language_model_names = ["language_model"]
        wrapper.set_aux_hidden_state_layers(taps)
        assert core.aux_hidden_state_layers == taps


@pytest.mark.parametrize("family,sharded", [("glm5_next", True), ("qwen3", False)])
def test_draft_projection_is_scoped_and_shards_before_loading(monkeypatch, family, sharded):
    import vllm.model_executor.models.qwen3_dflash as draft
    import vllm.model_executor.layers.linear as linear
    from r9700_vllm.compat.glm_dflash import install_draft_projection

    class FakeDraft:
        def __init__(self, **kwargs):
            self.use_aux_hidden_state = True
            self.quant_config = None
            self.fc = NS(weight=torch.empty(4096, 20480, device="meta"))

    calls = []
    def row_parallel(**kwargs):
        calls.append(kwargs)
        return NS(weight=torch.empty(4096, 2560, device="meta"))
    monkeypatch.setattr(draft, "DFlashQwen3Model", FakeDraft)
    monkeypatch.setattr(linear, "RowParallelLinear", row_parallel)
    install_draft_projection()
    cfg = NS(model_config=NS(hf_config=NS(model_type=family), dtype=torch.bfloat16))
    result = FakeDraft(vllm_config=cfg, prefix="draft.model")
    assert bool(calls) == sharded
    assert result.fc.weight.shape == (4096, 2560 if sharded else 20480)
    if sharded:
        assert calls[0]["input_is_parallel"] is False
        assert calls[0]["reduce_results"] is True
        assert calls[0]["return_bias"] is False
        assert calls[0]["prefix"] == "draft.model.fc"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.uint8])
@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("manager,target_dtype", [(640, torch.bfloat16), (1152, torch.uint8)])
def test_draft_cache_budget_and_disjoint_regions(monkeypatch, dtype, drop, tail, manager, target_dtype):
    from vllm.v1.core import kv_cache_utils as kv
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, MambaSpec, SlidingWindowSpec, KpoolTailSpec
    from r9700_vllm.compat.glm_dflash import install_cache_adapters
    cfg = NS(max_in_flight_tokens=1024,
             model_config=NS(hf_config=NS(model_type="glm5_next"), max_model_len=8192),
             parallel_config=NS(pipeline_parallel_size=1, decode_context_parallel_size=1,
                                prefill_context_parallel_size=1),
             attention_config=NS(hisparse_config=None),
             scheduler_config=NS(max_num_batched_tokens=1024),
             speculative_config=NS(method="dflash", use_eagle_block_drop=lambda: drop),
             cache_config=NS(num_gpu_blocks_override=None, prefix_cache_retention_interval=0,
                             mamba_cache_mode="all"))
    specs = {}
    for i in (3, 7):
        specs[f"layers.{i}.attn"] = MLAAttentionSpec(manager, num_kv_heads=1, head_size=512, dtype=target_dtype)
        specs[f"layers.{i}.indexer"] = MLAAttentionSpec(manager, num_kv_heads=1, head_size=132,
                                                dtype=torch.uint8, tokens_per_state=4)
    for i in (0, 1, 2, 4, 5, 6):
        specs[f"layers.{i}.mamba"] = MambaSpec(manager, shapes=((1024,),), dtypes=(torch.float32,))
    if tail:
        for i in (3, 7):
            specs[f"layers.{i}.tail"] = KpoolTailSpec(
                4, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, sliding_window=4)
    target = kv._get_kv_cache_groups_glm5_next(cfg, specs)
    original_bytes = kv._get_kv_cache_bytes_per_block(target)
    original = kv.get_kv_cache_config_from_groups(cfg, target, 100 * original_bytes)
    target_required = kv._max_memory_usage_bytes_from_groups(cfg, target)
    for i in range(5):
        specs[f"draft.layers.{i}.attn"] = SlidingWindowSpec(64, num_kv_heads=1, head_size=128,
                                                           dtype=dtype, sliding_window=2048)
    page = specs["draft.layers.0.attn"].page_size_bytes
    for name in ("_get_kv_cache_groups_glm5_next", "_get_kv_cache_bytes_per_block",
                 "get_kv_cache_config_from_groups", "_max_memory_usage_bytes_from_groups"):
        monkeypatch.setattr(kv, name, getattr(kv, name))
    install_cache_adapters(kv)
    groups = kv._get_kv_cache_groups_glm5_next(cfg, specs)
    assert groups[-1].is_eagle_group == drop
    assert groups[-1].kv_cache_spec.block_size == 64
    total = original_bytes + 5 * page
    assert kv._pool_bytes_per_block(groups) == total
    result = kv.get_kv_cache_config_from_groups(vllm_config=cfg, kv_cache_groups=groups,
                                               available_memory=100 * total + 1)
    assert result.num_blocks == 100
    assert all(t.size == 100 * total for t in result.kv_cache_tensors)
    for old, new in zip(original.kv_cache_tensors, result.kv_cache_tensors):
        assert (old.layers, old.offset, old.block_stride, old.layer_stride) == (
            new.layers, new.offset, new.block_stride, new.layer_stride)
    draft = result.kv_cache_tensors[-5:]
    assert [t.offset for t in draft] == [100 * (original_bytes + i * page) for i in range(5)]
    assert all(t.layer_stride == 100 * page and t.block_stride == page for t in draft)
    assert draft[-1].offset + draft[-1].layer_stride == 100 * total
    # Admission must include independently held sliding-window draft blocks.
    draft_blocks = (2048 - 1 + 1024 + 63) // 64 + 1
    assert kv._max_memory_usage_bytes_from_groups(cfg, groups) == (
        target_required // original_bytes + draft_blocks) * total
    # Target-only callers retain the exact original budget and aliases.
    assert kv._max_memory_usage_bytes_from_groups(cfg, target) == target_required
    assert kv.get_kv_cache_config_from_groups(cfg, target, 100 * original_bytes) == original
