"""DeepSeek TP8 W4A4 numerical qualification; no checkpoint load."""
import os, unittest
if os.environ.get("R9700_DEEPSEEK_GPU_TEST") != "1":
    raise unittest.SkipTest("Explicit GPU test opt-in required")
import torch
from types import SimpleNamespace
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig, RoutingMethodType, ocp_mx_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import (
    OCP_MXQuantizationEmulationTritonExperts as Upstream,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend, make_mxfp4_moe_kernel
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager
from r9700_vllm.moe.deepseek_w4a4 import apply_grouped, encode_qdq, gemm, launch_config

class Candidate(Upstream):
    def apply(self,output,hidden_states,w1,w2,topk_weights,topk_ids,activation,
              global_num_experts,expert_map,a1q_scale,a2_scale,workspace13,workspace2,
              expert_tokens_meta,apply_router_weight_on_input):
        return apply_grouped(self,output,hidden_states,w1,w2,topk_weights,topk_ids,activation,
                             expert_map=expert_map,global_experts=global_num_experts)

class DeepseekMoEGPU(unittest.TestCase):
    def setUp(self):
        init_workspace_manager(torch.device('cuda'))
        self.addCleanup(reset_workspace_manager)
        torch.manual_seed(41005)

    def test_projection_and_transport(self):
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import quant_dequant_mxfp4
        from test_glm_numerics import unpack_reference
        for n,k in ((576,5120),(5120,288)):
            w=torch.randint(256,(8,n,k//2),device='cuda',dtype=torch.uint8)
            s=torch.randint(108,135,(8,n,k//32),device='cuda',dtype=torch.uint8)
            ids=torch.tensor([[7,2,0,5,1,6]],device='cuda',dtype=torch.int32)
            x=quant_dequant_mxfp4(torch.randn(6,k,device='cuda',dtype=torch.bfloat16))
            x[0].zero_()
            q,scales=encode_qdq(x)
            torch.testing.assert_close((q.float().view(-1,32)*scales.flatten()[:,None]).view_as(x),x.float(),rtol=0,atol=0)
            out=torch.empty(6,n,device='cuda',dtype=torch.bfloat16)
            gemm(q,scales,w,s,out,moe_align_block_size(ids,16,8,None),6,1,grouped=True)
            ref=torch.stack([x[i].float() @ unpack_reference(w[e],s[e]).T for i,e in enumerate(ids[0].tolist())]).bfloat16()
            torch.testing.assert_close(out,ref,atol=.02,rtol=.01)
        self.assertEqual(launch_config(5120,288)[1],1)

    def test_full_moe_and_host_weights(self):
        from vllm.models.deepseek_v41.common.engram import _allocate_huge_page_storage
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
        e,h,i,k=384,5120,288,6
        w1=torch.randint(256,(e,2*i,h//2),device='cuda',dtype=torch.uint8)
        w2=torch.randint(256,(e,h,i//2),device='cuda',dtype=torch.uint8)
        s1=torch.randint(117,129,(e,2*i,h//32),device='cuda',dtype=torch.uint8)
        s2=torch.randint(117,129,(e,h,i//32),device='cuda',dtype=torch.uint8)
        cfg=FusedMoEConfig(num_experts=e,experts_per_token=k,hidden_dim=h,
            intermediate_size=i,num_local_experts=e,num_logical_experts=e,
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            activation=MoEActivation.SILU,in_dtype=torch.bfloat16,device='cuda:0',
            routing_method=RoutingMethodType.Renormalize)
        kernels=[]
        for cls in (Upstream,Candidate):
            quant=ocp_mx_moe_quant_config(quant_dtype='mxfp4',weight_dtype='mxfp4',
                w1_scale=s1,w2_scale=s2,gemm1_clamp_limit=10.)
            with set_current_vllm_config(VllmConfig()):
                kernels.append(make_mxfp4_moe_kernel(moe_quant_config=quant,moe_config=cfg,
                    mxfp4_backend=Mxfp4MoeBackend.EMULATION,experts_cls=cls,routing_tables=None))
        # Exercise the real model-scoped dispatcher after construction context exits.
        from unittest.mock import patch
        from r9700_vllm.compat.deepseek_moe import install
        with patch.dict(os.environ, R9K_DEEPSEEK_MOE='w4a4'):
            install()
        construction=VllmConfig()
        construction.parallel_config.tensor_parallel_size=8
        construction.model_config=SimpleNamespace(enforce_eager=True,
            hf_config=SimpleNamespace(model_type='deepseek_v41'))
        with set_current_vllm_config(construction):
            quant=ocp_mx_moe_quant_config(quant_dtype='mxfp4',weight_dtype='mxfp4',
                w1_scale=s1,w2_scale=s2,gemm1_clamp_limit=10.)
            dispatched=make_mxfp4_moe_kernel(moe_quant_config=quant,moe_config=cfg,
                mxfp4_backend=Mxfp4MoeBackend.EMULATION,experts_cls=Upstream,routing_tables=None)
        for host in (False,True):
            if host:
                def offload(w):
                    cpu=_allocate_huge_page_storage(w.numel()).view(w.shape)
                    cpu.copy_(w)
                    return get_accelerator_view_from_cpu_tensor(cpu)
                w1,w2=offload(w1),offload(w2)
            for rows in (1,2,8,16,17,32,128,512):
                with self.subTest(host=host,rows=rows):
                    x=(torch.randn(rows,h,device='cuda')*32).bfloat16()
                    x[0,:32]=0
                    ids=torch.stack([torch.randperm(e,device='cuda')[:k] for _ in range(rows)]).int()
                    ids[-1]=ids[0]
                    if rows == 512:
                        # Many tiles for each of six experts; exercise block
                        # boundaries and per-row scales, not just one tile.
                        ids[:]=ids[0].clone()
                    weights=torch.rand(rows,k,device='cuda');weights*=1.5/weights.sum(-1,keepdim=True)
                    args=dict(hidden_states=x,w1=w1,w2=w2,topk_weights=weights,topk_ids=ids,
                        activation=MoEActivation.SILU,global_num_experts=e,expert_map=None,
                        apply_router_weight_on_input=False)
                    ref=kernels[0].apply(**args).clone()
                    out=kernels[1].apply(**args)
                    # Fail if this supposed fast-path test silently takes fallback.
                    with patch.object(Upstream,'_dequantize_weights',side_effect=AssertionError('unexpected fallback')):
                        via_dispatch=dispatched.apply(**args)
                    torch.testing.assert_close(via_dispatch,out,atol=0,rtol=0)
                    self.assertTrue(torch.isfinite(out).all())
                    torch.testing.assert_close(out,ref,atol=.02,rtol=.01)

    def test_ep_projection_geometry_and_nonlocal_routes(self):
        # Full EP projection widths with a small local bank; no checkpoint or
        # full-model allocation. Distinct ownership maps include an idle rank.
        from unittest.mock import patch
        from r9700_vllm.moe.deepseek_staging import stage_threshold
        from vllm.models.deepseek_v41.common.engram import _allocate_huge_page_storage
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
        e,local,h,i,k=8,2,5120,2304,6
        w1=torch.randint(256,(local,2*i,h//2),device='cuda',dtype=torch.uint8)
        w2=torch.randint(256,(local,h,i//2),device='cuda',dtype=torch.uint8)
        s1=torch.full((local,2*i,h//32),120,device='cuda',dtype=torch.uint8)
        s2=torch.full((local,h,i//32),120,device='cuda',dtype=torch.uint8)
        cfg=FusedMoEConfig(num_experts=e,experts_per_token=k,hidden_dim=h,
            intermediate_size=i,num_local_experts=local,num_logical_experts=e,
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            activation=MoEActivation.SILU,in_dtype=torch.bfloat16,device='cuda:0',
            routing_method=RoutingMethodType.Renormalize)
        kernels=[]
        for cls in (Upstream,Candidate):
            quant=ocp_mx_moe_quant_config(quant_dtype='mxfp4',weight_dtype='mxfp4',
                w1_scale=s1,w2_scale=s2,gemm1_clamp_limit=10.)
            with set_current_vllm_config(VllmConfig()):
                kernels.append(make_mxfp4_moe_kernel(moe_quant_config=quant,moe_config=cfg,
                    mxfp4_backend=Mxfp4MoeBackend.EMULATION,experts_cls=cls,routing_tables=None))
        for storage in ("gpu","uva","mixed"):
            if storage == "mixed":
                from r9700_vllm.moe.deepseek_residency import extension, make_partially_resident
                def mixed(w):
                    page=extension().page_size(w.device.index,0)
                    flags=torch.arange((w.nbytes+page-1)//page)%2 == 0
                    view=make_partially_resident(w,flags)
                    view._vllm_is_uva_offloaded=True
                    view._r9700_host_bytes=int((~flags).sum())*page
                    return view
                w1,w2=mixed(w1),mixed(w2)
            if storage == "uva":
                def offload(w):
                    cpu=_allocate_huge_page_storage(w.numel()).view(w.shape)
                    cpu.copy_(w)
                    view=get_accelerator_view_from_cpu_tensor(cpu)
                    view._vllm_is_uva_offloaded=True
                    return view
                w1,w2=offload(w1),offload(w2)
            for rows in (1,2,4,32,128,256,512,1024,2048):
                for owned in ((1,5),(6,7)):
                    emap=torch.full((e,),-1,device='cuda',dtype=torch.int32)
                    emap[list(owned)]=torch.arange(local,device='cuda',dtype=torch.int32)
                    ids=torch.arange(k,device='cuda',dtype=torch.int32).repeat(rows,1)
                    x=torch.randn(rows,h,device='cuda',dtype=torch.bfloat16)
                    weights=torch.full((rows,k),1/k,device='cuda')
                    args=dict(hidden_states=x,w1=w1,w2=w2,topk_weights=weights,topk_ids=ids,
                        activation=MoEActivation.SILU,global_num_experts=e,expert_map=emap,
                        apply_router_weight_on_input=False)
                    ref=kernels[0].apply(**args).clone()
                    with patch.dict(os.environ,R9K_DEEPSEEK_STAGE_MIN_TOKENS='0'):
                        stage_threshold.cache_clear()
                        out=kernels[1].apply(**args).clone()
                    torch.testing.assert_close(out,ref,atol=.02,rtol=.01)
                    self.assertTrue(torch.isfinite(out).all())
                    if owned==(6,7):self.assertEqual(torch.count_nonzero(out).item(),0)
                    with patch.dict(os.environ,R9K_DEEPSEEK_STAGE_MIN_TOKENS='256'):
                        stage_threshold.cache_clear()
                        staged=kernels[1].apply(**args)
                    torch.testing.assert_close(staged,out,atol=0,rtol=0)
                    if rows in (256, 1024):
                        with patch.dict(os.environ,R9K_DEEPSEEK_STAGE_MIN_TOKENS='256',R9K_DEEPSEEK_STAGE_BUFFER='1'):
                            stage_threshold.cache_clear()
                            buffered=kernels[1].apply(**args)
                        torch.testing.assert_close(buffered,out,atol=0,rtol=0)
                    if rows == 4:
                        # Capture and replay with LIVE inputs/routes. Staging
                        # is bypassed during capture; weights remain immutable.
                        stream=torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(2): kernels[1].apply(**args)
                        torch.cuda.current_stream().wait_stream(stream)
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,stream=stream):
                            captured=kernels[1].apply(**args)
                        for shift in (1,3):
                            x.add_(0.125)
                            ids.copy_((ids+shift)%e)
                            graph.replay()
                            expected=kernels[0].apply(**args)
                            torch.testing.assert_close(captured,expected,atol=.02,rtol=.01)
                        del graph, captured
        stage_threshold.cache_clear()

    def test_cross_rank_sum_and_dspark_top3(self):
        from unittest.mock import patch
        from r9700_vllm.moe.deepseek_staging import stage_threshold
        e,h,i=8,5120,2304
        w1=torch.randint(256,(e,2*i,h//2),device='cuda',dtype=torch.uint8)
        w2=torch.randint(256,(e,h,i//2),device='cuda',dtype=torch.uint8)
        s1=torch.full((e,2*i,h//32),120,device='cuda',dtype=torch.uint8)
        s2=torch.full((e,h,i//32),120,device='cuda',dtype=torch.uint8)
        for topk in (3,6):
            def make(cls,first,last):
                config=FusedMoEConfig(num_experts=e,experts_per_token=topk,hidden_dim=h,
                    intermediate_size=i,num_local_experts=last-first,num_logical_experts=e,
                    moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
                    activation=MoEActivation.SILU,in_dtype=torch.bfloat16,device='cuda:0',
                    routing_method=RoutingMethodType.Renormalize)
                quant=ocp_mx_moe_quant_config(quant_dtype='mxfp4',weight_dtype='mxfp4',
                    w1_scale=s1[first:last],w2_scale=s2[first:last],gemm1_clamp_limit=10.)
                with set_current_vllm_config(VllmConfig()):
                    return make_mxfp4_moe_kernel(moe_quant_config=quant,moe_config=config,
                        mxfp4_backend=Mxfp4MoeBackend.EMULATION,experts_cls=cls,routing_tables=None)
            reference=make(Upstream,0,e)
            ranks=[make(Candidate,n,n+2) for n in range(0,e,2)]
            for rows in (1,4,24,128):
                x=torch.randn(rows,h,device='cuda',dtype=torch.bfloat16)
                ids=torch.stack([torch.randperm(e,device='cuda')[:topk] for _ in range(rows)]).int()
                weights=torch.rand(rows,topk,device='cuda');weights/=weights.sum(-1,keepdim=True)
                common=dict(hidden_states=x,topk_ids=ids,topk_weights=weights,
                    activation=MoEActivation.SILU,global_num_experts=e,apply_router_weight_on_input=False)
                expected=reference.apply(w1=w1,w2=w2,expert_map=None,**common).clone()
                partial=[]
                for rank,kernel in enumerate(ranks):
                    first=rank*2
                    emap=torch.full((e,),-1,device='cuda',dtype=torch.int32)
                    emap[first:first+2]=torch.arange(2,device='cuda',dtype=torch.int32)
                    partial.append(kernel.apply(w1=w1[first:first+2],w2=w2[first:first+2],expert_map=emap,**common).clone().float())
                actual=torch.stack(partial).sum(0).bfloat16()
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual,expected,atol=.04,rtol=.02)

if __name__=='__main__':unittest.main()
