#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exclusive GPU gate for upstream PR37134's classic chain sampler.

Not DFlash target-only verification: passing zero draft distributions to this
sampler is not a valid replacement. Test the actual p/q contract separately.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def oracle(candidates, p, q, coins, final_coin, vocab):
    accepted=0
    for step,token in enumerate(candidates[1:]):
        prob=q[step].get(token,0.)
        if not (0<prob<=1 and coins[step]*prob<p[step].get(token,0.)):break
        accepted+=1
    row=dict(p[accepted])
    if accepted<len(candidates)-1:
        row={i:max(a-(q[accepted].get(i,0.) if 0<=q[accepted].get(i,0.)<=1 else 0),0) for i,a in row.items()}
    limit=final_coin*sum(row.values());cdf=0
    for index,prob in sorted(row.items()):
        cdf+=prob
        if cdf>limit:return accepted,index
    return accepted,vocab-1


def protocol():
    return dict(scope='Upstream p/q chain GPU numerical/state gate, not model sampling/performance',
                batches=[1,2,4],widths=[4,5,6,8],vocabs=[17,4097,154880],
                invalid_q=['zero','negative','nan','positive_inf','negative_inf','above_one'],
                execution=['eager','captured replay with changed coins'],
                reference='Independent sequential CPU p/q accept and residual CDF; dyadic probabilities avoid rounding-boundary ambiguity',
                sampling_note='Q is the real draft proposal; zeroQ must reject, never accept all positiveP',
                sources={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['tests/test_sglang_rejection_sampling.py','r9700_sglang/patches/sglang-v0.5.20-rocm-eagle-sampling.patch']})


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    raw=json.dumps(protocol(),indent=2)+'\n'
    if args.prepare:
        args.out.mkdir(parents=True,exist_ok=False);(args.out/'protocol.json').write_text(raw);(args.out/'protocol.sha256').write_text(hashlib.sha256(raw.encode()).hexdigest()+'\n');return
    assert (args.out/'protocol.json').read_text()==raw and (args.out/'protocol.sha256').read_text().strip()==hashlib.sha256(raw.encode()).hexdigest()
    assert not (args.out/'results.jsonl').exists()
    import torch
    from sglang.kernels.ops.speculative.reject_sampling import chain_speculative_sampling_triton
    assert torch.cuda.device_count()==1
    assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx1201'
    count=0
    def record(row):
        with (args.out/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    for bs in protocol()['batches']:
      for width in protocol()['widths']:
       for vocab in protocol()['vocabs']:
        ids=[0,1,vocab//2,vocab-1]
        reference_p=[dict(zip(ids,[.5,.25,.125,.125]))]*width
        reference_q=[dict(zip(ids,[.25,.5,.125,.125]))]*(width-1)
        pp=torch.zeros(bs,width,vocab,dtype=torch.float32)
        qq=torch.zeros(bs,width-1,vocab,dtype=torch.float32)
        pp[:,:,ids]=torch.tensor([.5,.25,.125,.125]);qq[:,:,ids]=torch.tensor([.25,.5,.125,.125])
        candidates=torch.ones(bs,width,dtype=torch.int64,device='cuda')
        retrieve=torch.arange(bs*width,device='cuda',dtype=torch.int64).reshape(bs,width).flip(1).contiguous()
        nxt=torch.arange(1,width+1,device='cuda',dtype=torch.int64).repeat(bs,1);nxt[:,-1]=-1
        sibling=torch.full_like(nxt,-1)
        p=pp.cuda();q=qq.cuda();coins=torch.empty(bs,width,device='cuda');final=torch.empty(bs,device='cuda')
        predicts=torch.empty(bs*width,dtype=torch.int32,device='cuda');accept_index=torch.empty(bs,width,dtype=torch.int32,device='cuda');accepted=torch.empty(bs,dtype=torch.int32,device='cuda')
        def run():
            predicts.fill_(-1);accept_index.fill_(-1)
            chain_speculative_sampling_triton(predicts,accept_index,accepted,candidates,retrieve,nxt,sibling,coins,final,p,q,1.,1.,True)
        coins.fill_(.25);final.fill_(.375);run();torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):run()
        for prefix in range(width):
            cc=torch.full((bs,width),.25);ff=torch.tensor([.0625,.375,.6875,.9375][:bs])
            lengths=[(prefix+b)%width for b in range(bs)]
            for b,k in enumerate(lengths):
                if k<width-1:cc[b,k]=.75
            coins.copy_(cc);final.copy_(ff)
            for mode in ['eager','graph']:
                run() if mode=='eager' else graph.replay()
                got=accepted.cpu().tolist();pred=predicts.cpu().tolist();index=accept_index.cpu().tolist();mapping=retrieve.cpu().tolist()
                for b in range(bs):
                    k,bonus=oracle([1]*width,reference_p,reference_q,cc[b].tolist(),ff[b].item(),vocab)
                    assert k==lengths[b] and got[b]==k
                    assert index[b][:k+1]==mapping[b][:k+1]
                    assert all(pred[mapping[b][j]]==1 for j in range(k))
                    assert pred[mapping[b][k]]==bonus
                count+=bs
        # Invalid q at the proposed token must reject; residual sanitization
        # must never produce an invalid token or propagate NaN/Inf.
        for name,value in [('zero',0.),('negative',-1.),('nan',float('nan')),('positive_inf',float('inf')),('negative_inf',float('-inf')),('above_one',2.)]:
            bad=torch.full_like(qq,value);q.copy_(bad);coins.fill_(.25);final.copy_(torch.full((bs,),.375));graph.replay()
            got=accepted.cpu().tolist();pred=predicts.cpu().tolist();mapping=retrieve.cpu().tolist()
            for b in range(bs):
                k,bonus=oracle([1]*width,reference_p,[dict.fromkeys(ids,value)]*(width-1),[.25]*width,.375,vocab)
                assert k==0 and got[b]==0 and pred[mapping[b][0]]==bonus
            count+=bs
        record(dict(batch=bs,width=width,vocab=vocab,status='pass',cumulative_request_checks=count))
    (args.out/'complete.json').write_text(json.dumps(dict(status='pass',request_checks=count,scope=protocol()['scope']))+'\n')


def test_independent_oracle_accept_reject_and_invalid_q():
    p=[{0:.5,1:.25,2:.125,3:.125}]*3
    q=[{0:.25,1:.5,2:.125,3:.125}]*2
    assert oracle([1,1,1],p,q,[.25,.25,.25],.9375,4)==(2,3)
    assert oracle([1,1,1],p,q,[.75,.25,.25],.9375,4)==(0,0)
    bad=[{i:float('nan') for i in range(4)}]*2
    assert oracle([1,1,1],p,bad,[.25]*3,.9375,4)==(0,3)


if __name__=='__main__':main()
