"""Opt-in indexer qualification on the pinned DeepSeek gfx1201 image."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

if os.environ.get('R9700_DEEPSEEK_GPU_TEST') != '1':
    raise unittest.SkipTest('Explicit GPU test opt-in required')

import torch
import vllm
from vllm.config import CUDAGraphMode, DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import override_forward_context
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata, DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerPrefillMetadata, DeepseekV32IndexerPrefillChunkMetadata,
)
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
from r9700_vllm.attn.deepseek_indexer import paged_logits
from r9700_vllm.compat.deepseek_indexer import install


def fixture(page, count=2048, layout="dense"):
    g = torch.Generator().manual_seed(970041)
    k = torch.randn(count, 128, generator=g).to(torch.bfloat16)
    k[0] = 0
    k[1] *= 64
    # Independent CPU quantization and packing, not the GPU writer under test.
    scales = torch.exp2(torch.ceil(torch.log2(k.float().abs().amax(-1).clamp_min(1e-4) / 448)))
    codes = (k.float() / scales[:, None]).to(torch.float8_e4m3fn)
    n = count // page
    cache = torch.zeros(n, page * 132, dtype=torch.uint8)
    raw = codes.view(torch.uint8).reshape(n, page // 16, 16, 8, 16)
    cache[:, :page*128] = raw.permute(0, 1, 3, 2, 4).reshape(n, -1)
    cache[:, page*128:] = scales.reshape(n, page).view(torch.uint8)
    cache = cache.reshape(n, page, 1, 132).cuda()
    if layout != "dense":
        from vllm.v1.kv_cache_interface import (
            KVCacheTensor, MLAAttentionSpec, create_kv_cache_views,
        )
        from vllm.v1.kv_cache_layout import KVCacheLayout
        spec = MLAAttentionSpec(block_size=128, num_kv_heads=1,
                                head_size=132, dtype=torch.uint8,
                                tokens_per_state=128 // page, alignment=512)
        page_bytes = spec.page_size_bytes
        layers = 3 if layout == "packed" else 1
        offset = 512  # Nonzero backing offset, as with packed cache groups.
        placement = KVCacheTensor(
            size=offset + n * layers * page_bytes,
            layers=[f"layer{i}" for i in range(layers)],
            layer_stride=page_bytes, block_stride=layers * page_bytes,
            offset=offset)
        backing = torch.full((placement.size,), 91, dtype=torch.int8, device='cuda')
        views = create_kv_cache_views(backing, spec, n, KVCacheLayout.BLNHC, placement)
        target = views[-1].squeeze(1).unsqueeze(2)
        target.copy_(cache)
        cache = target
    return k, codes, scales, cache


def logits_reference(q, codes, scales, weights, table, lens, page, maximum):
    b, steps, _, _ = q.shape
    out = torch.full((b*steps, maximum), -float('inf'))
    for i in range(b):
        for j in range(steps):
            length = int(lens[i, j] if lens.ndim == 2 else lens[i] - steps + j + 1)
            pos = torch.arange(length)
            slots = table[i, pos // page].long() * page + pos % page
            score = q[i, j].float() @ codes[slots].float().T
            out[i*steps+j, :length] = (score.relu() * weights[i*steps+j, :, None]).sum(0) * scales[slots]
    return out


class DeepseekIndexerGPU(unittest.TestCase):
    def setUp(self):
        self.assertIn('18f8f960', vllm.__version__)
        self.assertTrue(torch.version.hip)
        self.assertIn('gfx1201', torch.cuda.get_device_properties(0).gcnArchName)
        self.cfg = VllmConfig(device_config=DeviceConfig(device='cuda'))
        self.cfg.model_config = SimpleNamespace(
            dtype=torch.bfloat16, enforce_eager=True,
            hf_config=SimpleNamespace(model_type='deepseek_v41'))
        self.cfg.scheduler_config = SimpleNamespace(max_num_seqs=4)
        ctx = set_current_vllm_config(self.cfg)
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        init_workspace_manager(torch.device('cuda'))
        self.addCleanup(reset_workspace_manager)
        self.g = torch.Generator().manual_seed(41)

    def test_full_context_c4_profile_and_locked_workspace(self):
        from r9700_vllm.compat.deepseek_indexer import profile_workspace
        from vllm.v1.worker.workspace import current_workspace_manager
        layer = SimpleNamespace(max_total_seq_len=4*1048576, head_dim=128,
                                max_model_len=1048576, _r9700_ds_decode_rows=4,
                                topk_indices_buffer=None)
        hidden = torch.empty(512, 1, device='cuda')
        with set_current_vllm_config(None):
            profile_workspace(layer, hidden)
        workspace = current_workspace_manager()
        workspace.lock()
        # Actual C4 gather/decode calls fit the reservation without growth.
        keys, scales = workspace.get_simultaneous(
            ((4*1048576,128),torch.float8_e4m3fn),
            ((4*1048576,4),torch.uint8))
        self.assertEqual(keys.numel()+scales.numel(), 528*1024*1024)
        del keys, scales
        (logits,) = workspace.get_simultaneous(((4,1048576),torch.float32))
        self.assertEqual(logits.numel()*4,16*1024*1024)
        workspace.unlock()

    def test_mla_prefill_budget_preserves_four_request_chunk_plan(self):
        from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention
        from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata
        from vllm.v1.worker.workspace import current_workspace_manager
        install()
        maximum=1048576
        layer=SimpleNamespace(PREFILL_CHUNK_SIZE=1)
        q=torch.empty(512,8,512,device='cuda',dtype=torch.bfloat16)
        shapes=DeepseekV41ROCMAiterMLAAttention._prefill_workspace_shapes(
            layer,maximum+128+512,512,q)
        (scratch,)=current_workspace_manager().get_simultaneous(*shapes)
        self.assertLess(scratch.numel()*scratch.element_size(),1.01*2**30)
        for ratio in (1,2):
            for lengths in ([maximum]*4,[1024,maximum,8192,65536]):
                meta=SimpleNamespace(num_prefills=4,
                    prefill_seq_lens_cpu=torch.tensor(lengths),
                    prefill_query_lens_cpu=torch.tensor([128]*4),
                    prefill_max_model_len=maximum,prefill_window_size=128,
                    prefill_max_num_batched_tokens=512)
                plan=DeepseekSparseSWAMetadata.get_prefill_chunk_plan(
                    meta,ratio,1,has_compressed=True)
                covered=[]
                for start,end,compressed,width in plan:
                    covered.extend(range(start,end))
                    self.assertLessEqual((end-start)*width,maximum//ratio+128+512)
                    self.assertGreaterEqual(compressed,max(lengths[start:end])//ratio)
                self.assertEqual(covered,[0,1,2,3])
        self.assertEqual(DeepseekV41ROCMAiterMLAAttention.PREFILL_CHUNK_SIZE,1)

    def test_writer_and_prefill_gather(self):
        for ratio, page in ((1, 128), (2, 64)):
            k, codes, scales, packed = fixture(page)
            actual = torch.zeros_like(packed).squeeze(2)
            slots = torch.arange(k.shape[0], device='cuda')
            ops.indexer_k_quant_and_cache_triton(k.cuda(), actual, slots, 128, 'ue8m0')
            torch.testing.assert_close(actual, packed.squeeze(2), atol=0, rtol=0)
            table = torch.arange(k.shape[0]//page, dtype=torch.int32, device='cuda')[None]
            gathered = torch.empty_like(codes, device='cuda')
            gathered_scale = torch.empty(k.shape[0], 4, dtype=torch.uint8, device='cuda')
            ops.cp_gather_indexer_k_quant_cache_triton(
                actual, gathered, gathered_scale, table,
                torch.tensor([0,k.shape[0]], dtype=torch.int32, device='cuda'),
                torch.zeros(k.shape[0], dtype=torch.int32, device='cuda'))
            torch.testing.assert_close(gathered.cpu().float(), codes.float(), atol=0, rtol=0)
            torch.testing.assert_close(gathered_scale.view(torch.float32).cpu().flatten(), scales, atol=0, rtol=0)

    def test_model_fused_writer(self):
        from vllm.models.deepseek_v41.common.ops import indexer_k_norm_rope_store
        # Exact-valued normalization and nontrivial GPT-J rotation isolate layout,
        # group boundaries, skipped slots and the actual model writer.
        for ratio, page in ((1,128),(2,64)):
            k = torch.where(torch.arange(37*128).reshape(37,128)%3 == 0, -1., 1.).to(torch.bfloat16)
            norm = torch.linspace(.25,2,128).to(torch.bfloat16)
            positions = torch.arange(37,dtype=torch.int64)
            slots = torch.arange(page-19,page+18,dtype=torch.int64)
            slots[5] = -1
            cs = torch.cat([torch.zeros(37,32),torch.ones(37,32)],-1)
            actual = torch.zeros(2,page,132,dtype=torch.uint8,device='cuda')
            indexer_k_norm_rope_store(k.cuda(),positions.cuda(),cs.cuda(),norm.cuda(),
                                     0.,actual,slots.cuda(),ratio,False)
            expected = torch.zeros(2,page*132,dtype=torch.uint8)
            for row in range(37):
                if slots[row] < 0 or (int(positions[row])+1)%ratio:
                    continue
                x = (k[row].float()*norm.float()).to(torch.bfloat16).float()
                pairs = x[64:].view(32,2).clone()
                x[64::2],x[65::2] = -pairs[:,1],pairs[:,0]
                scale = torch.exp2(torch.ceil(torch.log2(x.abs().max().clamp_min(1e-4)/448)))
                codes = (x/scale).to(torch.float8_e4m3fn).view(torch.uint8)
                block, offset = divmod(int(slots[row]),page)
                dim = torch.arange(128)
                address = (offset//16)*128*16 + (dim//16)*256 + (offset%16)*16 + dim%16
                expected[block,address] = codes
                expected[block,page*128+offset*4:page*128+(offset+1)*4] = scale.reshape(1).view(torch.uint8)
            torch.testing.assert_close(actual.cpu().reshape(2,-1),expected,atol=0,rtol=0)

    def test_paged_logits(self):
        for ratio, page, layout in ((r, p, l) for r, p in ((1,128),(2,64))
                                    for l in ("dense", "padded", "packed")):
            _, codes, scales, cache = fixture(page, layout=layout)
            self.assertEqual(cache.stride(1), 132)
            if layout == "packed" or (layout == "padded" and page == 64):
                self.assertFalse(cache.is_contiguous())
            for b, steps in ((1,1), (2,1), (1,8), (2,8)):
                q = torch.randn(b,steps,32,128,generator=self.g).to(torch.float8_e4m3fn)
                weights = torch.randn(b*steps,32,generator=self.g)
                table = torch.arange(2048//page, dtype=torch.int32).flip(0).repeat(b,1)
                for per_row in (False, True):
                    lens = torch.tensor([129, 1025][:b],dtype=torch.int32)
                    if per_row:
                        lens = lens[:,None] - steps + torch.arange(steps) + 1
                    expected = logits_reference(q,codes,scales,weights,table,lens,page,1152)
                    actual = paged_logits(q.cuda(),cache,weights.cuda(),lens.cuda(),table.cuda(),None,1152,compress_ratio=ratio)
                    torch.testing.assert_close(actual.cpu(),expected,atol=.003,rtol=.0001)

    def test_graph_replay_with_poisoned_unused_block(self):
        for page, ratio in ((128, 1), (64, 2)):
            _, codes, scales, cache = fixture(page)
            cache[0].fill_(255)  # Unused block must never influence real rows.
            q = torch.randn(4, 1, 32, 128, device='cuda').to(torch.float8_e4m3fn)
            weights = torch.rand(4, 32, device='cuda')
            table = torch.arange(1, 2048//page, device='cuda', dtype=torch.int32).repeat(4, 1)
            lens = torch.tensor([128, 256, 0, 0], device='cuda', dtype=torch.int32)
            def call():
                return paged_logits(q, cache, weights, lens, table, None, 512, compress_ratio=ratio)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): call()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = call()
            for lengths in ([257, 383, 0, 0], [100, 300, 400, 500]):
                lens.copy_(torch.tensor(lengths, device='cuda', dtype=torch.int32))
                q.copy_(torch.randn_like(q.float()).to(q.dtype))
                graph.replay()
                expected = logits_reference(q.cpu(), codes, scales, weights.cpu(), table.cpu(), lens.cpu(), page, 512)
                torch.testing.assert_close(output.cpu(), expected, atol=.003, rtol=.0001)
            del graph, output

    def test_full_indexer_prefill_decode_candidates(self):
        from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
        stock_reader = ops.rocm_fp8_paged_mqa_logits
        stock_prefill = ops.rocm_fp8_mqa_logits
        install()
        self.assertIs(ops.rocm_fp8_paged_mqa_logits, stock_reader)
        self.assertIs(ops.rocm_fp8_mqa_logits, stock_prefill)
        fallback = patch.object(ops, 'fp8_mqa_logits_torch',
                                side_effect=AssertionError('DeepSeek must not use prefill reference'))
        fallback.start()
        self.addCleanup(fallback.stop)
        for ratio, page, layout in ((r, p, l) for r, p in ((1,128),(2,64))
                                    for l in ("dense", "padded", "packed")):
            _, codes, scales, cache = fixture(page, layout=layout)
            b, steps, maximum, topk = 2, 1, 1152, 512
            q = torch.randn(b,steps,32,128,generator=self.g).to(torch.float8_e4m3fn)
            weights = torch.rand(b,32,generator=self.g)
            table = torch.arange(2048//page,dtype=torch.int32).flip(0).repeat(b,1)
            lens = torch.tensor([769,1025],dtype=torch.int32)
            expected = logits_reference(q,codes,scales,weights,table,lens,page,maximum)
            # Keep the two contexts disjoint in the gathered prefill layout.
            total = int(lens.sum())
            starts = torch.tensor([0,int(lens[0])],dtype=torch.int32,device='cuda')
            ends = starts + lens.cuda()
            chunk = DeepseekV32IndexerPrefillChunkMetadata(
                block_table=table.cuda(),cu_seqlen_ks=starts,cu_seqlen_ke=ends,
                cu_seq_lens=torch.cat([starts,ends[-1:]]),
                token_to_seq=torch.repeat_interleave(torch.arange(b,dtype=torch.int32),lens).cuda(),
                total_seq_lens=total,token_start=0,token_end=b,num_reqs=b)
            for prefill in (False,True):
                decode = DeepSeekV32IndexerDecodeMetadata(
                    table.cuda(),lens.cuda(),torch.ones(b,dtype=torch.int32,device='cuda'),
                    False,torch.empty(0,device='cuda'))
                meta = DeepseekV32IndexerMetadata(
                    lens.cuda(),maximum*ratio,torch.full((b,),-1,dtype=torch.int64,device='cuda'),
                    0 if prefill else b,0 if prefill else b,b if prefill else 0,b if prefill else 0,
                    decode=None if prefill else decode,
                    prefill=DeepseekV32IndexerPrefillMetadata([chunk]) if prefill else None)
                candidates = torch.empty(b,4,dtype=torch.int32,device='cuda')
                output = torch.empty(b,topk,dtype=torch.int32,device='cuda')
                layer = SparseAttnIndexer(
                    SimpleNamespace(prefix='test.index',kv_cache=cache.squeeze(2)),
                    128,'ue8m0',topk,128,maximum,total,output,
                    skip_k_cache_insert=True,compress_ratio=ratio)
                context = SimpleNamespace(attn_metadata={'test.index':meta},
                                          cudagraph_runtime_mode=CUDAGraphMode.NONE)
                for candidate_mode in ('none','write','read'):
                    layer.candidate_blocks = None if candidate_mode == 'none' else candidates
                    layer.candidate_block_size = 128
                    layer.candidate_write = candidate_mode == 'write'
                    with set_current_vllm_config(None), override_forward_context(context):
                        result = layer.forward_hip(torch.zeros(b,16,device='cuda'),q[:,0].cuda(),None,weights.cuda()).cpu()
                    target = expected.clone()
                    if candidate_mode == 'read':
                        for row in range(b):
                            allowed = candidates[row].cpu()
                            keep = torch.isin(torch.arange(maximum)//128,allowed)
                            # Upstream candidate mask also retains the last visible token.
                            keep[int(lens[row])-1] = True
                            target[row,~keep] = -float('inf')
                    for row in range(b):
                        indices = result[row]
                        self.assertTrue(((indices >= 0) & (indices < lens[row])).all())
                        self.assertEqual(indices.unique().numel(), topk)
                        selected = target[row,indices].sort(descending=True).values
                        best = target[row].topk(topk).values
                        torch.testing.assert_close(selected,best,atol=.2 if prefill else .003,rtol=.005 if prefill else .0001)

if __name__ == '__main__':
    unittest.main()
