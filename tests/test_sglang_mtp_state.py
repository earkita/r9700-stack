# SPDX-License-Identifier: Apache-2.0
"""Bounded gfx1201 MTP state probe inside the pinned SGLang image.

Based on SGLang's Apache-2.0 test/manual/test_kda_target_verify.py.
Verify reads committed FP32 KDA state; scatter commits only the accepted
prefix. Compare continuation after every forced acceptance/rejection length
against sequential decode, including non-contiguous slots and a masked row.
This is a state/kernel test, not model-quality or throughput qualification.
"""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3], choices=range(1, 8))
    parser.add_argument("--batches", type=int, nargs="+", default=[2], choices=[1, 2, 4])
    args = parser.parse_args()
    import torch
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update as run
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import fused_mamba_state_scatter_with_mask as commit

    torch.manual_seed(20261015)
    results = []
    for b, lower_bound, steps in ((batch,bound,d) for batch in args.batches for bound in (None,-5.) for d in args.depths):
        heads, kdim, vdim, t = 8, 128, 128, steps + 1
        slots = torch.arange(1, 2*b, 2, device="cuda", dtype=torch.int32)
        state = torch.randn(2*b+1, heads, kdim, vdim, device="cuda", dtype=torch.float32) * .1
        q, k = [torch.randn(1, b*t, heads, kdim, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        v = torch.randn(1, b*t, heads, vdim, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(b*t, heads*kdim, device="cuda", dtype=torch.bfloat16)
        beta = torch.randn(1, b*t, heads, device="cuda", dtype=torch.bfloat16)
        alog = torch.randn(heads, device="cuda")
        bias = torch.randn(heads*kdim, device="cuda")
        common = dict(A_log=alog, dt_bias=bias, initial_state_indices=slots,
                      use_qk_l2norm_in_kernel=True, softplus_beta=1.,
                      softplus_threshold=20., is_kda=True, lower_bound=lower_bound)
        current = state.clone(); checkpoints = []; outputs = []
        for step in range(t):
            idx = [row*t+step for row in range(b)]
            output = run(q=q[:,idx].contiguous(), k=k[:,idx].contiguous(), v=v[:,idx].contiguous(),
                         a=a[idx].contiguous(), b=beta[:,idx].contiguous(), initial_state_source=current,
                         cu_seqlens=torch.arange(b+1,device="cuda",dtype=torch.int32), **common)
            checkpoints.append(current[slots].clone()); outputs.append(output)
        committed = state.clone(); scratch = torch.zeros(2*b+1,t,heads,kdim,vdim,device="cuda")
        actual = run(q=q,k=k,v=v,a=a,b=beta,initial_state_source=committed,
                     cu_seqlens=torch.arange(0,b*t+1,t,device="cuda",dtype=torch.int32),
                     disable_state_update=True,intermediate_states_buffer=scratch,
                     intermediate_state_indices=torch.arange(b,device="cuda",dtype=torch.int32),
                     cache_steps=t,retrieve_parent_token=None,**common)
        expected = torch.cat([outputs[s][:,row:row+1] for row in range(b) for s in range(t)],dim=1)
        out_error = (actual.float()-expected.float()).abs().max().item()
        assert out_error < 1e-3, out_error
        assert torch.equal(committed,state), "Verify mutated committed state"
        state_error = max((scratch[:b,s]-checkpoints[s]).abs().max().item() for s in range(t))
        assert state_error < 1e-5, state_error
        # At least the target token is committed; 0 means reject all draft
        # tokens, `steps` means accept all. Another request chooses a different
        # acceptance length, so cross-request scratch/slot mixups are visible.
        for accepted in range(steps+1):
            lengths = torch.tensor([accepted if row == 0 else steps-accepted if row == 1
                                    else (accepted+row-1) % (steps+1) for row in range(b)],
                                   device="cuda",dtype=torch.int32)
            dst = state.clone(); commit(dst[None],scratch[None],slots,lengths)
            reference = state.clone()
            for row, last in enumerate(lengths.tolist()):reference[slots[row]]=checkpoints[last][row]
            assert torch.equal(dst,reference), "Wrong accepted checkpoint committed"
            # Subsequent decode must start from precisely the accepted prefix.
            idx = [row*t for row in range(b)]
            def advance(pool):
                return run(q=q[:,idx].contiguous(),k=k[:,idx].contiguous(),v=v[:,idx].contiguous(),
                           a=a[idx].contiguous(),b=beta[:,idx].contiguous(),initial_state_source=pool,
                           cu_seqlens=torch.arange(b+1,device="cuda",dtype=torch.int32),**common)
            assert torch.equal(advance(dst),advance(reference))
            assert torch.equal(dst,reference)
        masked = state.clone()
        commit(masked[None],scratch[None],slots,torch.full((b,),-1,device="cuda",dtype=torch.int32))
        assert torch.equal(masked,state)
        results.append(dict(batch=b,depth=steps,lower_bound=lower_bound,heads=heads,
                            output_max_abs=out_error,state_max_abs=state_error,
                            committed_unchanged=True,forced_acceptance_cases=steps+1,
                            continuation_exact=True,masked_rows_unchanged=True))
    print(json.dumps(results,indent=2),flush=True)


if __name__ == "__main__":
    main()
