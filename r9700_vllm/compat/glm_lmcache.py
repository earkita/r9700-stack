"""Opt-in LMCache 0.5.5 connector for GLM Flash on pinned e97573215 ROCm.

Only external-cache metadata/views are adapted. Circular kpool scratch is
rebuilt during the aligned suffix prefill, just as for vLLM's GPU APC.
"""
from importlib.metadata import version
from types import SimpleNamespace

import torch
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestState
from lmcache.integration.vllm.utils import vllm_layout_hints

from .glm_cache_geometry import (
    logical_mla_view, select_block_ids, select_cache_groups, select_scheduler_output,
)


class GlmLMCacheMPConnector(LMCacheMPConnector):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        if (version('lmcache') != '0.5.5' or 'e97573215' not in version('vllm')
                or not torch.version.hip
                or vllm_config.model_config.hf_config.model_type != 'glm5_next'
                or kv_cache_config is None):
            raise RuntimeError('GLM offload requires ROCm, GLM Flash, e97573215 and LMCache 0.5.5')
        pc = vllm_config.parallel_config
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        if (pc.tensor_parallel_size != 8 or pc.pipeline_parallel_size != 1
                or pc.decode_context_parallel_size != 1
                or extra.get('lmcache.mp.lazy_offload', False)):
            raise RuntimeError('GLM offload supports TP8/PP1/DCP1 with eager stores only')
        self._original_group_count = len(kv_cache_config.kv_cache_groups)
        filtered, self._kept_groups, self._scratch_layers = select_cache_groups(kv_cache_config)
        from .glm_lmcache_events import install
        install()
        super().__init__(vllm_config, role, filtered)
        # Every persisted boundary must complete an index_kpool=4 pool. A
        # non-empty suffix then seeds the scratch before any decode reads it.
        if any(span % 4 for span in self._group_tokens_per_block):
            raise RuntimeError('GLM offload boundaries must align to index_kpool=4')

    def _select(self, blocks):
        return select_block_ids(blocks, self._kept_groups, self._original_group_count)

    def register_kv_caches(self, kv_caches):
        layout = vllm_layout_hints(self._vllm_config)['kv_layout']
        views = {n: t for n, t in kv_caches.items() if n not in self._scratch_layers}
        for group in self._kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            leaves = getattr(spec, 'kv_cache_specs', None) or {n: spec for n in group.layer_names}
            for name, leaf in leaves.items():
                if type(leaf).__name__ == 'MLAAttentionSpec':
                    views[name] = logical_mla_view(views[name], leaf, layout)
        super().register_kv_caches(views)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        selected = self._select(blocks.get_block_ids())
        super().update_state_after_alloc(
            request, SimpleNamespace(get_block_ids=lambda: selected), num_external_tokens)

    def build_connector_meta(self, scheduler_output):
        return super().build_connector_meta(select_scheduler_output(
            scheduler_output, self._kept_groups, self._original_group_count))

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        # A full-prompt hit normally leaves one token to recompute. That can
        # start inside a kpool. Conservatively recompute this uncommon case
        # instead of restoring scratch from an unrelated request.
        result = super().get_num_new_matched_tokens(request, num_computed_tokens)
        tracker = self.request_trackers.get(request.request_id)
        if (tracker is not None and tracker.num_lmcache_hit_tokens > 0
                and tracker.num_lmcache_hit_tokens == len(request.all_token_ids)):
            self.scheduler_adapter.free_lookup_locks(
                token_ids=tracker.get_token_ids(), start=0,
                end=tracker.num_lmcache_hit_tokens, request_id=request.request_id,
                cache_salt=tracker.cache_salt)
            self.scheduler_adapter.cleanup_lookup_result(request.request_id)
            tracker.num_stored_tokens = 0
            tracker.num_lmcache_hit_tokens = 0
            tracker.num_vllm_hit_tokens = 0
            tracker.state = LMCacheMPRequestState.BYPASS_LMCACHE
            return 0, False
        return result
