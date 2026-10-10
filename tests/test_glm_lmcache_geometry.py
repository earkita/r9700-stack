"""CPU checks for the opt-in external cache's GLM block mapping."""
from dataclasses import dataclass
from types import SimpleNamespace as NS
import unittest
import pytest

from r9700_vllm.compat.glm_cache_geometry import (
    logical_mla_view, select_cache_groups, select_block_ids, select_scheduler_output,
)


class KpoolTailSpec:
    prefix_cacheable = False


@dataclass
class Config:
    kv_cache_groups: list


class Geometry(unittest.TestCase):
    def test_upstream_store_gate_with_and_without_circular_scratch(self):
        pytest.importorskip('vllm')
        pytest.importorskip('lmcache')
        from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata

        def tracker(groups):
            row = NS(request_id='probe', all_token_ids=list(range(4246)),
                     allocated_block_ids=dict(enumerate(groups)),
                     num_scheduled_tokens=3456, num_stored_tokens=0,
                     num_vllm_hit_tokens=0, num_lmcache_hit_tokens=0,
                     cache_salt='', request_configs=None)
            row.num_allocated_blocks = lambda: {i: len(b) for i, b in row.allocated_block_ids.items()}
            row.get_token_ids = lambda: row.all_token_ids
            row.increase_num_stored_tokens = lambda n: setattr(row, 'num_stored_tokens', row.num_stored_tokens+n)
            return row

        # Actual GLM circular manager allocates one block for the whole request.
        broken = tracker(([1, 2, 3], [99], [4, 5, 6]))
        self.assertIsNone(LMCacheMPRequestMetadata.GetStoreMetadata(broken, 1152, [1152, 8, 1152]))
        fixed = tracker(([1, 2, 3], [4, 5, 6]))
        metadata = LMCacheMPRequestMetadata.GetStoreMetadata(fixed, 1152, [1152, 1152])
        self.assertEqual((metadata.op.start, metadata.op.end), (0, 3456))
        self.assertEqual(fixed.num_stored_tokens, 3456)

    def test_only_scratch_is_excluded(self):
        attention = NS(prefix_cacheable=True)
        groups = [NS(kv_cache_spec=attention, layer_names=['mla'], enable_kv_transfer=True),
                  NS(kv_cache_spec=KpoolTailSpec(), layer_names=['tail'], enable_kv_transfer=True),
                  NS(kv_cache_spec=attention, layer_names=['state'], enable_kv_transfer=True)]
        source = Config(groups)
        result, kept, dropped = select_cache_groups(source)
        self.assertEqual(kept, (0, 2))
        self.assertEqual(dropped, {'tail'})
        self.assertEqual(source.kv_cache_groups, groups)
        self.assertEqual(len(result.kv_cache_groups), 2)
        self.assertEqual(select_block_ids(([1, 2], [99], [3, 4]), kept, 3), ([1, 2], [3, 4]))
        with self.assertRaises(ValueError):
            select_block_ids(([1],), kept, 3)

    def test_scheduler_copy_including_empty_deltas(self):
        ids = ([1, 2], [99], [3, 4])
        source = NS(scheduled_new_reqs=[NS(block_ids=ids)],
                    scheduled_cached_reqs=NS(new_block_ids=[ids, None, ()]))
        result = select_scheduler_output(source, (0, 2), 3)
        self.assertEqual(result.scheduled_new_reqs[0].block_ids, ([1, 2], [3, 4]))
        self.assertEqual(result.scheduled_cached_reqs.new_block_ids, [([1, 2], [3, 4]), None, ()])
        self.assertEqual(source.scheduled_new_reqs[0].block_ids, ids)
        self.assertEqual(source.scheduled_cached_reqs.new_block_ids, [ids, None, ()])

    def test_mla_and_indexer_byte_roundtrip(self):
        torch = pytest.importorskip('torch')
        for compression, kernel_states, width in [(1, 128, 512), (4, 32, 132)]:
            for layout, heads in [('NHD', False), ('NHD', True), ('HND', True)]:
                source = torch.arange(3*9*kernel_states*width, dtype=torch.int64).to(torch.uint8)
                shape = ((27, kernel_states, 1, width) if layout == 'NHD'
                         else (27, 1, kernel_states, width)) if heads else (27, kernel_states, width)
                source = source.reshape(shape)
                spec = NS(block_size=1152, tokens_per_state=compression,
                          page_size_bytes=1152//compression*width)
                view = logical_mla_view(source, spec, layout)
                expected = list(shape)
                expected[0] = 3
                expected[2 if heads and layout == 'HND' else 1] = 1152//compression
                self.assertEqual(tuple(view.shape), tuple(expected))
                self.assertEqual(view.data_ptr(), source.data_ptr())
                # Persist/retrieve logical block 1, with adjacent blocks as canaries.
                saved = view[1].clone()
                view[1].zero_()
                view[1].copy_(saved)
                self.assertTrue(torch.equal(view[1].reshape(-1), source[9:18].reshape(-1)))
                self.assertTrue(torch.equal(source.reshape(-1),
                    torch.arange(source.numel(), dtype=torch.int64).to(torch.uint8)))
                spec.page_size_bytes += 1
                with self.assertRaises(ValueError): logical_mla_view(source, spec, layout)


if __name__ == '__main__':
    unittest.main()
