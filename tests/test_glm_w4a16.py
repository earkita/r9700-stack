"""Bounded W4A16 numerical, graph replay, and configuration-isolation gates."""
from copy import deepcopy
from types import SimpleNamespace
import pytest


def test_config_copy_preserves_weights_fp8_kv_and_original():
    from r9700_vllm.compat.glm_w4a16 import weight_only_config
    mx = dict(dtype="fp4", group_size=32, scale_format="e8m0", qscheme="per_group")
    q = dict(global_quant_config=dict(weight=mx, input_tensors=mx | dict(is_dynamic=True)),
             layer_quant_config={"dense": dict(weight={"dtype":"fp8_e4m3"},
                                              input_tensors={"dtype":"fp8_e4m3"}),
                                 "kv": dict(output_tensors={"dtype":"fp8_e4m3"})},
             layer_type_quant_config={}, exclude=["lm_head"])
    original = deepcopy(q)
    result, count = weight_only_config(q)
    assert count == 1 and result["global_quant_config"]["input_tensors"] is None
    assert result["global_quant_config"]["weight"] == q["global_quant_config"]["weight"]
    assert result["layer_quant_config"] == q["layer_quant_config"]
    assert q == original
    assert weight_only_config(result) == (result, 0)


def test_rejects_non_mx_fp4():
    from r9700_vllm.compat.glm_w4a16 import weight_only_config
    with pytest.raises(ValueError, match="group-32"):
        weight_only_config(dict(global_quant_config=dict(weight=dict(dtype="fp4", group_size=16))))


def test_patch_scopes_metadata_to_glm(monkeypatch):
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
    from vllm.model_executor.layers.quantization.quark.quark_moe import QuarkOCP_MX_MoEMethod
    from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import OCP_MXQuantizationEmulationTritonExperts as Experts
    from r9700_vllm.compat.glm_w4a16 import patch
    # Restore class methods after this test, since patching is process-global.
    monkeypatch.setattr(QuarkConfig, "maybe_update_config", QuarkConfig.maybe_update_config)
    monkeypatch.setattr(Experts, "__init__", Experts.__init__)
    monkeypatch.setattr(Experts, "apply", Experts.apply)
    monkeypatch.setattr(QuarkOCP_MX_MoEMethod, "get_fused_moe_quant_config", lambda *args: "unmodified")
    assert patch()
    mx = dict(dtype="fp4", group_size=32, scale_format="e8m0", qscheme="per_group")
    q = dict(global_quant_config=dict(weight=mx, input_tensors=mx | dict(is_dynamic=True)))
    for family in ("glm5_next", "qwen", "dflash"):
        config = QuarkConfig(deepcopy(q))
        config.maybe_update_config("unused", SimpleNamespace(model_type=family))
        key = config._get_scheme_cls_from_config(config.quant_config["global_quant_config"])[1]
        assert (key is None) == (family == "glm5_next")
        assert bool(getattr(config, "_r9k_glm_w4a16", False)) == (family == "glm5_next")
    assert q["global_quant_config"]["input_tensors"] is not None
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    ex=SimpleNamespace(model_type="glm5_next",ocp_mx_scheme="w_mxfp4",mxfp4_backend=Mxfp4MoeBackend.EMULATION)
    layer=SimpleNamespace(w13_weight_scale=None,w2_weight_scale=None,w13_bias=None,w2_bias=None,swiglu_limit=10.)
    quant=QuarkOCP_MX_MoEMethod.get_fused_moe_quant_config(ex,layer)
    assert quant.gemm1_clamp_limit == 10.
    ex.model_type="qwen"
    assert QuarkOCP_MX_MoEMethod.get_fused_moe_quant_config(ex,layer) == "unmodified"


@pytest.mark.parametrize("rows,input_scale,capture", [(1,1,False),(5,4,True),(10,32,True),(32,4,True)])
def test_full_moe_against_weight_only_upstream(rows, input_scale, capture):
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEParallelConfig, RoutingMethodType, mxfp4_w4a16_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import OCP_MXQuantizationEmulationTritonExperts as Upstream
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend, make_mxfp4_moe_kernel
    from vllm.v1.worker.workspace import init_workspace_manager
    from r9700_vllm.moe.w4a16 import apply
    class Candidate(Upstream):
        def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
                  global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2,
                  expert_tokens_meta, apply_router_weight_on_input):
            return apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation)
    init_workspace_manager(torch.accelerator.current_device_index())
    torch.manual_seed(1616 + rows)
    e, h, n, topk = 288, 4096, 256, 8
    w1 = torch.randint(256,(e,n*2,h//2),device="cuda",dtype=torch.uint8)
    w2 = torch.randint(256,(e,h,n//2),device="cuda",dtype=torch.uint8)
    s1 = torch.randint(117,129,(e,n*2,h//32),device="cuda",dtype=torch.uint8)
    s2 = torch.randint(117,129,(e,h,n//32),device="cuda",dtype=torch.uint8)
    x = (torch.randn(rows,h,device="cuda")*input_scale).bfloat16()
    ids = torch.stack([torch.randperm(e,device="cuda")[:topk] for _ in range(rows)]).int()
    weights = torch.rand(rows,topk,device="cuda")
    weights *= 2.5/weights.sum(1,keepdim=True)
    config = FusedMoEConfig(num_experts=e, experts_per_token=topk, hidden_dim=h,
        intermediate_size=n, num_local_experts=e, num_logical_experts=e,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(), activation=MoEActivation.SILU,
        in_dtype=x.dtype, device="cuda:0", routing_method=RoutingMethodType.Renormalize)
    kernels=[]
    for cls in (Upstream,Candidate):
        q = mxfp4_w4a16_moe_quant_config(w1_scale=s1,w2_scale=s2,gemm1_clamp_limit=10.)
        with set_current_vllm_config(VllmConfig()):
            kernels.append(make_mxfp4_moe_kernel(moe_quant_config=q,moe_config=config,
                mxfp4_backend=Mxfp4MoeBackend.EMULATION,experts_cls=cls,routing_tables=None))
    def run(kernel):
        return kernel.apply(hidden_states=x,w1=w1,w2=w2,topk_weights=weights,topk_ids=ids,
            activation=MoEActivation.SILU,global_num_experts=e,expert_map=None,apply_router_weight_on_input=False)
    with set_current_vllm_config(VllmConfig()):
        if capture:
            stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): run(kernels[1])
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream): captured=run(kernels[1])
        for replay in range(3 if capture else 1):
            if replay:
                x.copy_(torch.randn_like(x)*input_scale)
                ids.copy_(torch.stack([torch.randperm(e,device="cuda")[:topk] for _ in range(rows)]).int())
                ids[-1].copy_(ids[0])
                weights.copy_(torch.rand_like(weights)); weights *= 2.5/weights.sum(1,keepdim=True)
            reference=run(kernels[0]).clone()
            if capture:
                graph.replay(); actual=captured.clone()
            else: actual=run(kernels[1]).clone()
            assert torch.isfinite(actual).all()
            # BF16 projection boundaries amplify rare FP32 reduction-rounding
            # differences in upstream, especially after routed cancellation.
            # Check every pointwise-disagreeing row against independent FP64,
            # plus first/last rows even when upstream agrees.
            from test_glm_numerics import unpack_reference
            mismatched=(~torch.isclose(actual,reference,rtol=1e-2,atol=1e-2)).any(1)
            checked=sorted({0,rows-1,*mismatched.nonzero().flatten().tolist()})
            for row in checked:
                parts=[]
                for slot,expert in enumerate(ids[row].tolist()):
                    gu=(x[row].double() @ unpack_reference(w1[expert],s1[expert]).double().T).bfloat16().float()
                    gate,up=gu.chunk(2,-1)
                    mid=(torch.nn.functional.silu(gate.clamp(max=10))*up.clamp(-10,10)).bfloat16()
                    down=mid.double() @ unpack_reference(w2[expert],s2[expert]).double().T
                    parts.append((down*weights[row,slot].double()).bfloat16())
                exact=torch.stack(parts).float().sum(0).bfloat16()
                error=(actual[row].float()-exact.float())
                rms=exact.float().square().mean().sqrt()
                rel_l2=(error.norm()/exact.float().norm()).item()
                normalized_max=(error.abs().max()/rms).item()
                # Routed summation can cancel large BF16 values to near zero;
                # elementwise relative error there is not a stable criterion.
                # Bound norm error to 1/4 BF16 epsilon and every absolute error
                # to 2 BF16 epsilons of the row RMS, also against independent FP64.
                assert rel_l2 < torch.finfo(torch.bfloat16).eps/4
                assert normalized_max < 2*torch.finfo(torch.bfloat16).eps
                print('FP64 bounds',dict(rows=rows,replay=replay,row=row,
                      relative_l2=rel_l2,normalized_max=normalized_max),flush=True)
            relative_l2=((actual.float()-reference.float()).norm()/reference.float().norm()).item()
            assert relative_l2 < torch.finfo(torch.bfloat16).eps/4
            print('W4A16 reference',dict(rows=rows,replay=replay,fp64_rows=checked,
                  upstream_relative_l2=relative_l2),flush=True)


@pytest.mark.parametrize("n,k,divisor",[(512,4096,8),(4096,256,1)])
def test_bf16_projection_against_fp64_without_activation_rounding(n,k,divisor):
    import torch
    from r9700_vllm.moe.packed_gemv import gemv
    from test_glm_numerics import unpack_reference
    torch.manual_seed(1617+n)
    slots,e=16,8
    w=torch.randint(256,(e,n,k//2),device="cuda",dtype=torch.uint8)
    scales=torch.randint(117,129,(e,n,k//32),device="cuda",dtype=torch.uint8)
    ids=torch.tensor([7,0,1,3,2,6,5,4]*2,device="cuda",dtype=torch.int32)
    x=torch.randn(slots//divisor,k,device="cuda",dtype=torch.bfloat16)
    weights=torch.rand(slots,device="cuda") if divisor==1 else None
    actual=gemv(x,w,scales,ids,divisor,weights)
    for slot,expert in enumerate(ids.tolist()):
        ref=x[slot//divisor].double() @ unpack_reference(w[expert],scales[expert]).double().T
        if weights is not None:ref*=weights[slot].double()
        torch.testing.assert_close(actual[slot],ref.bfloat16(),rtol=torch.finfo(torch.bfloat16).eps,atol=1e-3)
