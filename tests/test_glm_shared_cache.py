"""Shared pool admission, layout and physical block isolation on pinned vLLM."""
from types import SimpleNamespace as NS
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('vllm')

@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.uint8])
@pytest.mark.parametrize('drop', [False, True])
def test_shared_cache(monkeypatch, dtype, drop):
    from vllm.v1.core import kv_cache_utils as kv
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, MambaSpec, SlidingWindowSpec
    from r9700_vllm.compat.glm_shared_cache import install_shared_cache_adapters
    cfg = NS(max_in_flight_tokens=8192,
        model_config=NS(hf_config=NS(model_type='glm5_next'), max_model_len=65536),
        parallel_config=NS(pipeline_parallel_size=1, decode_context_parallel_size=1, prefill_context_parallel_size=1),
        attention_config=NS(hisparse_config=None),
        scheduler_config=NS(max_num_batched_tokens=4096),
        speculative_config=NS(method='dflash', use_eagle_block_drop=lambda: drop),
        cache_config=NS(num_gpu_blocks_override=None, prefix_cache_retention_interval=0, mamba_cache_mode='all'))
    specs = {}
    for i in range(2):
        specs[f'mla{i}'] = MLAAttentionSpec(1152, num_kv_heads=1, head_size=512, dtype=torch.uint8)
        specs[f'idx{i}'] = MLAAttentionSpec(1152, num_kv_heads=1, head_size=132, dtype=torch.uint8, tokens_per_state=4)
    for i in range(6):
        specs[f'mamba{i}'] = MambaSpec(1152, shapes=((1024,),), dtypes=(torch.float32,))
    target = kv._get_kv_cache_groups_glm5_next(cfg, specs)
    per_block = kv._get_kv_cache_bytes_per_block(target)
    required = kv._max_memory_usage_bytes_from_groups(cfg, target)
    for i in range(5):
        specs[f'draft{i}'] = SlidingWindowSpec(64, num_kv_heads=1, head_size=128, dtype=dtype, sliding_window=2048)
    for attr in ('_get_kv_cache_groups_glm5_next', '_glm5_next_tensor_layout'):
        monkeypatch.setattr(kv, attr, getattr(kv, attr))
    install_shared_cache_adapters(kv)
    groups = kv._get_kv_cache_groups_glm5_next(cfg, specs)
    drafts = groups[len(target):]
    assert len(drafts) == 3
    assert all(g.is_eagle_group == drop and g.kv_cache_spec.block_size == 1152 for g in drafts)
    assert kv._get_kv_cache_bytes_per_block(groups) == per_block
    assert kv._max_memory_usage_bytes_from_groups(cfg, groups) == required + 3 * 10 * per_block
    result = kv.get_kv_cache_config_from_groups(cfg, groups, 16 * per_block)
    assert result.num_blocks == 16
    tensors = {t.layers[0]: t for t in result.kv_cache_tensors}
    assert set(tensors) == set(specs)
    for t in tensors.values():
        assert t.size == 16 * per_block
        assert t.offset + 16 * t.block_stride <= t.size
    assert tensors['draft0'].offset == tensors['mla0'].offset
    # Distinct pool IDs give disjoint bytes despite aliased layer views.
    # Run on the GPU when available to exercise byte addressing there too.
    storage = torch.zeros(16 * per_block, dtype=torch.uint8,
                          device='cuda' if torch.cuda.is_available() else 'cpu')
    for name, block_id, value in [('mla0', 1, 17), ('draft0', 2, 29), ('mamba0', 3, 41)]:
        t = tensors[name]
        start = t.offset + block_id * t.block_stride
        storage[start:start+t.block_stride].fill_(value)
    for name, block_id, value in [('mla0', 1, 17), ('draft0', 2, 29), ('mamba0', 3, 41)]:
        t = tensors[name]
        start = t.offset + block_id * t.block_stride
        assert bool(torch.all(storage[start:start+t.block_stride] == value))

@pytest.mark.parametrize('family,method,drop,expected', [
    ('glm5_next','dflash',False,1152), ('glm5_next','dflash',True,2304),
    ('qwen3','dflash',False,2304), ('glm5_next','eagle',False,2304)])
def test_apc_boundary_scope(family, method, drop, expected):
    from r9700_vllm.compat.glm_apc import install_boundary_adapter
    class Scheduler:
        def _mamba_block_aligned_split(self, request, count, local=0, external=0):
            return count
    install_boundary_adapter(Scheduler)
    installed = Scheduler._mamba_block_aligned_split
    install_boundary_adapter(Scheduler)
    assert Scheduler._mamba_block_aligned_split is installed
    s = Scheduler()
    s.vllm_config = NS(model_config=NS(hf_config=NS(model_type=family)), speculative_config=NS(method=method))
    s.cache_config = NS(block_size=1152)
    s.use_eagle_block_drop = drop
    request = NS(num_tokens=64512, num_computed_tokens=62208)
    assert s._mamba_block_aligned_split(request,2304) == expected
    request.num_computed_tokens=63360
    assert s._mamba_block_aligned_split(request,1152) == 1152
    request.num_computed_tokens=61056
    assert s._mamba_block_aligned_split(request,2304,1152,0) == expected

@pytest.mark.parametrize('length', [1152,32256,32257,32768,64511,64512,64513])
def test_actual_scheduler_materializes_reusable_boundary(monkeypatch, length):
    from vllm.v1.core.sched.scheduler import Scheduler
    from r9700_vllm.compat.glm_apc import install_boundary_adapter
    monkeypatch.setattr(Scheduler, '_mamba_block_aligned_split', Scheduler._mamba_block_aligned_split)
    install_boundary_adapter(Scheduler)
    s = NS(vllm_config=NS(model_config=NS(hf_config=NS(model_type='glm5_next')), speculative_config=NS(method='dflash')),
        cache_config=NS(block_size=1152), use_eagle_block_drop=False, hash_block_size=16,
        mamba_has_prefill_checkpoint_blocks=False, max_num_scheduled_tokens=4096,
        scheduler_config=NS(long_prefill_token_threshold=0), mamba_partial_cache_hit=False,
        mamba_shared_prefix_checkpoint=False)
    req = NS(num_tokens=length, num_prompt_tokens=length, num_computed_tokens=0, shared_prefix_boundary=0)
    ends = [0]
    while req.num_computed_tokens < length:
        count = Scheduler._mamba_block_aligned_split(s, req, min(4096,length-req.num_computed_tokens))
        assert count > 0
        req.num_computed_tokens += count
        ends.append(req.num_computed_tokens)
    assert (length-1)//1152*1152 in ends
