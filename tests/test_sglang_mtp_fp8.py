# SPDX-License-Identifier: Apache-2.0
"""Standalone numerical probe for the original GLM draft's block-FP8 MoE.

Run inside the pinned image on an exclusive gfx1201 GPU. The independent
reference expands original block scales and reproduces activation FP8 QDQ,
BF16 intermediate rounding, GLM clamp, routing weights and expert reduction.
"""
import json


def main():
    import torch
    import torch.nn.functional as F
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.runtime_context import get_context
    from r9700_sglang.mtp_fp8 import NextNBlockFp8
    from sglang.srt.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    import tempfile

    rendezvous = tempfile.mktemp(prefix="glm-mtp-unit-")
    init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                 distributed_init_method="file://" + rendezvous,
                                 backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=1)

    override = get_context().override_server_args(moe_runner_backend="triton")
    override.install()
    torch.manual_seed(20261015)
    e, h, n, routes = 8, 4096, 256, 8
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter((torch.randn(e,2*n,h,device="cuda")*32).to(torch.float8_e4m3fn),requires_grad=False)
    layer.w2_weight = torch.nn.Parameter((torch.randn(e,h,n,device="cuda")*32).to(torch.float8_e4m3fn),requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(torch.rand(e,2*n//128,h//128,device="cuda")*.002+.0001,requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(torch.rand(e,h//128,n//128,device="cuda")*.002+.0001,requires_grad=False)
    scheme = NextNBlockFp8();scheme.process_weights_after_loading(layer)
    config = MoeRunnerConfig(num_experts=e,num_local_experts=e,hidden_size=h,
                             intermediate_size_per_partition=n,top_k=routes,
                             inplace=False,swiglu_limit=10,gate_up_interleaved=False)
    scheme.create_moe_runner(layer,config)
    def unpack(w,s):return w.float()*s.repeat_interleave(128,1).repeat_interleave(128,2)
    w1,w2=unpack(layer.w13_weight,layer.w13_weight_scale),unpack(layer.w2_weight,layer.w2_weight_scale)
    def qdq(x):
        from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8
        # Triton's approximate FP32 division is not bit-identical to Torch's
        # division. Near FP8 halfway points, a scale differing by one FP32 ULP
        # changes the rounded code. Validate the native scale independently,
        # validate every code via Torch RN-even conversion, then feed those
        # validated codes/scales to the independent dense arithmetic oracle.
        # This does NOT claim an independent bit-exact scale generator.
        original=x.shape
        x=x.contiguous().reshape(-1,x.shape[-1])
        quant,scale=per_token_group_quant_fp8(x,128)
        grouped=x.float().reshape(x.shape[0],-1,128)
        ideal=grouped.abs().amax(-1,keepdim=True).clamp(min=1e-10)/448.
        observed=scale[:,:,None]
        assert torch.all((observed-ideal).abs()<=ideal.abs()*torch.finfo(torch.float32).eps*2)
        rounded=(grouped*(1./observed)).clamp(-448,448).to(torch.float8_e4m3fn)
        assert torch.equal(rounded.view(torch.uint8),quant.reshape_as(rounded).view(torch.uint8))
        return (rounded.float()*observed).reshape(original)
    rows=[]
    for m in (1,2,4):
        x=torch.randn(m,h,device="cuda",dtype=torch.bfloat16)*.25
        ids=torch.stack([torch.randperm(e,device="cuda") for _ in range(m)]).int()
        weights=torch.rand(m,routes,device="cuda");weights/=weights.sum(-1,keepdim=True)
        dispatched=StandardDispatchOutput(hidden_states=x,hidden_states_scale=None,
                                         topk_output=StandardTopKOutput(weights,ids,None))
        actual=scheme.apply_weights(layer,dispatched).hidden_states.float()
        reference=torch.zeros_like(actual)
        qx=qdq(x)
        for row in range(m):
            partial=[]
            for j in range(routes):
                idx=ids[row,j]
                gu=(qx[row]@w1[idx].T).bfloat16().float()
                gate,up=gu.chunk(2,-1)
                mid=(F.silu(gate.clamp(max=10))*up.clamp(-10,10)).bfloat16()
                value=qdq(mid)@w2[idx].T
                partial.append((value*weights[row,j]).bfloat16().float())
            reference[row]=torch.stack(partial).sum(0).bfloat16().float()
        delta=(actual-reference).abs()
        rel=((actual-reference).norm()/reference.norm()).item()
        assert torch.isfinite(actual).all()
        assert rel<.02,(m,rel)
        assert delta.max().item()<.05,(m,delta.max().item())
        rows.append(dict(rows=m,relative_l2=rel,max_abs=delta.max().item(),mean_abs=delta.mean().item()))
    print(json.dumps(rows,indent=2),flush=True)


if __name__ == "__main__":
    main()
