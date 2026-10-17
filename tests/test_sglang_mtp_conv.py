# SPDX-License-Identifier: Apache-2.0
"""GLM's actual BF16 width-4 KDA convolution: verify, commit and continue."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3], choices=range(1, 8))
    parser.add_argument("--batches", type=int, nargs="+", default=[2], choices=[1, 2, 4])
    args = parser.parse_args()
    import torch
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update as conv
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import fused_mamba_state_scatter_with_mask as commit
    torch.manual_seed(20261015);results=[]
    for b,depth in ((batch,d) for batch in args.batches for d in args.depths):
        dim,t=3*8*128,depth+1
        slots=torch.arange(1,2*b,2,device='cuda',dtype=torch.int32)
        base=torch.randn(2*b+1,3,dim,device='cuda',dtype=torch.bfloat16)
        x=torch.randn(b,dim,t,device='cuda',dtype=torch.bfloat16)
        w=torch.randn(dim,4,device='cuda',dtype=torch.float32)*.1
        current=base.clone();outputs=[];checkpoints=[]
        for step in range(t):
            outputs.append(conv(x[:,:,step].contiguous(),current.transpose(-1,-2),w,
                                activation='silu',conv_state_indices=slots).clone())
            checkpoints.append(current[slots].clone())
        scratch=torch.zeros(2*b+1,t,3,dim,device='cuda',dtype=torch.bfloat16)
        verified=base.clone()
        actual=conv(x.clone(),verified.transpose(-1,-2),w,activation='silu',conv_state_indices=slots,
                    intermediate_conv_window=scratch.transpose(-1,-2),
                    intermediate_state_indices=torch.arange(b,device='cuda',dtype=torch.int32))
        expected=torch.stack(outputs,dim=-1)
        error=(actual.float()-expected.float()).abs().max().item()
        assert torch.equal(actual,expected),error
        for step in range(t):assert torch.equal(scratch[:b,step],checkpoints[step])
        for accepted in range(depth+1):
            lengths=torch.tensor([accepted if row==0 else depth-accepted if row==1
                                  else (accepted+row-1)%(depth+1) for row in range(b)],
                                 device='cuda',dtype=torch.int32)
            dst=verified.clone();commit(dst[None],scratch[None],slots,lengths)
            ref=base.clone()
            for row,last in enumerate(lengths.tolist()):ref[slots[row]]=checkpoints[last][row]
            # Only real slots are meaningful after the tentative verify.
            assert torch.equal(dst[slots],ref[slots])
            next_x=x[:,:,0].contiguous()
            actual_next=conv(next_x.clone(),dst.transpose(-1,-2),w,activation='silu',conv_state_indices=slots)
            ref_next=conv(next_x.clone(),ref.transpose(-1,-2),w,activation='silu',conv_state_indices=slots)
            assert torch.equal(actual_next,ref_next)
            assert torch.equal(dst[slots],ref[slots])
        results.append(dict(batch=b,depth=depth,output_max_abs=error,checkpoints_exact=True,
                            acceptance_cases=depth+1,continuation_exact=True))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
