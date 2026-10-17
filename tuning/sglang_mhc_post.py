#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exclusive one-GPU mHC post correctness and workspace/operator diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def protocol():
    return dict(scope="operator diagnostic, not whole-model speed or quality",
                hypothesis="Eliminate replicated FP32 [M,4,4,4096] products and separate reduction/add/cast launches",
                variable="Torch HC4 post versus fused Triton post; same BF16 inputs and FP32 mixes/products; FMA disabled",
                rows=[128,512,2048,4096,8192], scales=[0.,.1,2.], seed=20261016,
                order=["torch_a","triton","torch_b"], warmups=3,repetitions=11,
                correctness="bit-exact BF16 output; any mismatch rejects candidate before timings",
                rejection="Any nonfinite/mismatched output, changed source, GPU fault or missing PCIe evidence; no server integration",
                rollback="No serving dispatch added; restore the exact preserved MTP3 container after exclusive probe",
                sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [Path('tuning/sglang_mhc_post.py'),Path('r9700_sglang/mhc_post.py')]})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    plan=protocol();raw=json.dumps(plan,indent=2)+'\n'
    if args.prepare:
        args.out.mkdir(parents=True,exist_ok=False)
        (args.out/'protocol.json').write_text(raw)
        (args.out/'protocol.sha256').write_text(hashlib.sha256(raw.encode()).hexdigest()+'\n')
        return
    if ((args.out/'protocol.json').read_text()!=raw or
            (args.out/'protocol.sha256').read_text().strip()!=hashlib.sha256(raw.encode()).hexdigest() or
            (args.out/'results.jsonl').exists()):
        raise RuntimeError('Protocol/source changed or attempt already exists')
    import torch
    from r9700_sglang.mhc_post import post
    from sglang.kernels.ops.layernorm.mhc import _mhc_post_torch
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly one exclusively available GPU')
    torch.manual_seed(plan['seed'])
    def record(row):
        with (args.out/'results.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def measure(op,inputs):
        for _ in range(plan['warmups']):op(*inputs)
        torch.cuda.synchronize();samples=[]
        for _ in range(plan['repetitions']):
            torch.cuda.reset_peak_memory_stats()
            before=torch.cuda.memory_allocated()
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();value=op(*inputs);b.record();b.synchronize()
            samples.append(dict(gpu_interval_ms=a.elapsed_time(b),extra_allocated_bytes=torch.cuda.max_memory_allocated()-before))
            del value
        return dict(samples=samples,median_ms=statistics.median(x['gpu_interval_ms'] for x in samples),max_extra_allocated_bytes=max(x['extra_allocated_bytes'] for x in samples))
    for rows in plan['rows']:
        for scale in plan['scales']:
            inputs=((torch.randn(rows,4096,device='cuda')*scale).bfloat16(),
                    (torch.randn(rows,4,4096,device='cuda')*scale).bfloat16(),
                    torch.rand(rows,4,1,device='cuda')*2,
                    torch.randn(rows,4,4,device='cuda'))
            expected=_mhc_post_torch(*inputs);actual=post(*inputs)
            difference=(actual.float()-expected.float()).abs()
            result=dict(rows=rows,scale=scale,exact=torch.equal(actual,expected),max_abs=difference.max().item(),mismatches=(actual!=expected).sum().item(),finite=torch.isfinite(actual).all().item())
            record(result)
            if not result['exact'] or not result['finite']:
                raise RuntimeError('Numerical gate failed; original attempt retained, no timings or serving promotion')
            del expected,actual,difference
            if scale==.1:
                result=dict(rows=rows,scope=plan['scope'])
                for name,op in [('torch_a',_mhc_post_torch),('triton',post),('torch_b',_mhc_post_torch)]:result[name]=measure(op,inputs)
                record(result)
    (args.out/'complete.json').write_text(json.dumps(dict(status='pass',scope=plan['scope']))+'\n')


if __name__=='__main__':main()
