"""CPU contract/isolation tests in the pinned DeepSeek image."""
import unittest
import os
from functools import wraps
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    import torch
    import vllm
except ImportError as exc:
    raise unittest.SkipTest('Run inside the DeepSeek vLLM image') from exc

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from r9700_vllm.compat.deepseek_indexer import make_indexer_impl, wrap_forward, wrap_init, install


# Stand-in to verify private global binding, including keyword defaults.
def rocm_fp8_paged_mqa_logits(value):
    return value + 1


def rocm_fp8_mqa_logits(value):
    return value


def sample_upstream(value=2, *, bias=3):
    return rocm_fp8_paged_mqa_logits(rocm_fp8_mqa_logits(value)) + bias


class DeepseekIndexerContract(unittest.TestCase):
    def test_cache_view_preserves_storage_and_rejects_invalid_layouts(self):
        from r9700_vllm.attn.deepseek_indexer import cache_view
        for page in (64, 128):
            stride = ((page * 132 + 511) // 512) * 512 * 3
            raw = torch.zeros(512 + 4 * stride, dtype=torch.uint8)
            cache = raw.as_strided((4,page,1,132),(stride,132,132,1),512)
            flat = cache_view(cache)
            self.assertEqual(flat.data_ptr(), cache.data_ptr())
            self.assertEqual(flat.stride(), (stride,1))
            self.assertEqual(flat.untyped_storage().data_ptr(), raw.data_ptr())
        for shape, strides, offset in (
            ((2,64,1,132),(9000,133,132,1),0),  # gaps inside a page
            ((2,64,1,132),(9001,132,132,1),0),  # unaligned FP32 scales
            ((2,64,1,132),(8448,132,132,1),1),  # unaligned base
            ((2,64,1,132),(8444,132,132,1),0),  # overlapping pages
            ((2,32,1,132),(4224,132,132,1),0),  # unaudited page width
        ):
            cache = torch.empty(40000,dtype=torch.uint8).as_strided(shape,strides,offset)
            with self.assertRaisesRegex(ValueError,'shape=.*stride='):
                cache_view(cache)

    def setUp(self):
        env = patch.dict(os.environ, R9K_DEEPSEEK_GRAPHS="0")
        env.start()
        self.addCleanup(env.stop)
        self.cfg = VllmConfig(device_config=DeviceConfig(device='cpu'))
        self.cfg.model_config = SimpleNamespace(
            enforce_eager=True, hf_config=SimpleNamespace(model_type='deepseek_v41'))
        self.cfg.scheduler_config = SimpleNamespace(max_num_seqs=4)
        context = patch('vllm.forward_context.get_forward_context',
                        return_value=SimpleNamespace(attn_metadata={}))
        context.start()
        self.addCleanup(context.stop)
        self.original, self.implementation = Mock(return_value='stock'), Mock(return_value='adapted')
        self.forward = wrap_forward(self.original, self.implementation)
        self.layer = SimpleNamespace(
            use_fp4_cache=False,use_pcp=False,dcp_world_size=1,compress_ratio=2,
            head_dim=128,quant_block_size=128,scale_fmt='ue8m0',topk_tokens=512,
            max_model_len=4096,max_total_seq_len=4096,topk_indices_buffer=None,
            skip_k_cache_insert=True,candidate_blocks=None,candidate_block_size=128,
            candidate_write=False,k_cache=SimpleNamespace(prefix='test.index',kv_cache=None))
        with set_current_vllm_config(self.cfg):
            wrap_init(lambda self: None)(self.layer)
        self.q = torch.empty(1,32,128,dtype=torch.float8_e4m3fn)

    def test_profile_reserves_reader_layout_without_runtime_config(self):
        from r9700_vllm.compat.deepseek_indexer import profile_workspace
        from vllm.config import get_current_vllm_config
        with self.assertRaises(AssertionError):
            get_current_vllm_config()
        self.layer.max_model_len = 1048576
        self.layer.max_total_seq_len = 4 * 1048576
        hidden = torch.empty(512, 1)
        workspace = Mock()
        with patch('vllm.v1.worker.workspace.current_workspace_manager', return_value=workspace), \
             patch('vllm.envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB', 1), \
             patch('vllm.forward_context.get_forward_context', return_value=SimpleNamespace(attn_metadata=None)):
            self.assertIsNone(self.forward(self.layer, hidden, self.q, None, None))
        self.implementation.assert_not_called()
        self.assertEqual(workspace.get_simultaneous.call_args_list[0].args,
                         (((4*1048576,128),torch.float8_e4m3fn),
                          ((4*1048576,4),torch.uint8)))
        self.assertEqual(workspace.get_simultaneous.call_args_list[1].args,
                         (((4,1048576),torch.float32),))
        self.assertEqual(self.layer._r9700_ds_decode_rows, 4)

    def test_prefill_bound_uses_all_four_compressed_contexts(self):
        for ratio in (1,2):
            self.layer.max_model_len=1048576//ratio
            self.layer.max_total_seq_len=40*1048576//ratio
            with set_current_vllm_config(self.cfg):
                wrap_init(lambda self: None)(self.layer)
            self.assertEqual(self.layer.max_total_seq_len,4*1048576//ratio)

    def test_profile_decode_bound_includes_speculation(self):
        self.cfg.speculative_config = SimpleNamespace(num_speculative_tokens=7)
        with set_current_vllm_config(self.cfg):
            wrap_init(lambda self: None)(self.layer)
        self.assertEqual(self.layer._r9700_ds_decode_rows,32)

    def test_private_binding_keeps_upstream_unchanged(self):
        adapted = make_indexer_impl(sample_upstream, lambda value: value * 10,
                                    lambda value: value * 2)
        self.assertEqual(adapted(),43)
        self.assertEqual(sample_upstream(),6)
        self.assertIs(sample_upstream.__globals__['rocm_fp8_paged_mqa_logits'],rocm_fp8_paged_mqa_logits)
        self.assertIs(sample_upstream.__globals__['rocm_fp8_mqa_logits'],rocm_fp8_mqa_logits)

    def test_decorated_eager_body_binding(self):
        @wraps(sample_upstream)
        def graph_wrapper(*args, **kwargs):
            raise AssertionError('Graph wrapper must not run in this eager-only adapter')
        adapted = make_indexer_impl(graph_wrapper, lambda value: value * 10,
                                    lambda value: value)
        self.assertEqual(adapted(),23)

    def test_other_models_delegate(self):
        for model in ('glm5_next','qwen3','mimo_v2'):
            self.cfg.model_config.hf_config.model_type = model
            with set_current_vllm_config(self.cfg):
                wrap_init(lambda self: None)(self.layer)
            self.assertEqual(self.forward(self.layer,None,self.q,None,None),'stock')
        self.implementation.assert_not_called()

    def test_forward_after_construction_context_exits(self):
        from vllm.config import get_current_vllm_config
        with self.assertRaises(AssertionError):
            get_current_vllm_config()
        self.assertEqual(self.forward(self.layer,None,self.q,None,None), "adapted")

    def test_candidate_and_compression_arguments_preserved(self):
        self.assertEqual(self.forward(self.layer,None,self.q,None,None),'adapted')
        self.original.assert_not_called()
        kw = self.implementation.call_args.kwargs
        self.assertEqual(kw['compress_ratio'],2)
        self.assertTrue(kw['skip_k_cache_insert'])
        self.assertEqual(kw['candidate_block_size'],128)

    def test_reject_unqualified_features(self):
        for field,value in (('use_fp4_cache',True),('compress_ratio',4),
                            ('use_pcp',True),('dcp_world_size',2),('scale_fmt','float32')):
            with self.subTest(field=field), patch.object(self.layer,field,value):
                with self.assertRaisesRegex(RuntimeError,'Unsupported'):
                    self.forward(self.layer,None,self.q,None,None)
        self.layer._r9700_ds_eager = False
        with self.assertRaisesRegex(RuntimeError,'enforce_eager'):
            self.forward(self.layer,None,self.q,None,None)
        self.implementation.assert_not_called()

    def test_graph_dispatch_requires_explicit_optin(self):
        self.layer._r9700_ds_eager = False
        self.layer._r9700_ds_graphs = True
        self.assertEqual(self.forward(self.layer,None,self.q,None,None),'adapted')

    def test_reject_wrong_version(self):
        from vllm.platforms import current_platform
        with patch.object(current_platform,'is_rocm',return_value=True), \
             patch('r9700_vllm.compat.gate.vllm_commit',return_value='wrong'):
            with self.assertRaisesRegex(RuntimeError,'18f8f960'):
                install()

if __name__ == '__main__':
    unittest.main()
