#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exclusive gfx1201 DFlash2 selector walk, realized Q, and worker accept gate.

No model throughput or attention/projection qualification. The exact checkpoint
has a selector; its p/q branch bypasses the absent HIP target-only sampler.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def oracle_walk(ids, scores, temperatures, greedy, turn):
    tokens=[];rows=[];coins=[]
    for b,sequence in enumerate(scores):
        pred=0;bt=[];bq=[];bu=[]
        for step,matrix in enumerate(sequence):
            logits=[x/temperatures[b] for x in matrix[pred]]
            top=max(logits);values=[math.exp(x-top) for x in logits];total=sum(values)
            probs=[x/total for x in values]
            chosen=max(range(len(logits)),key=lambda j:logits[j]) if greedy[b] else (turn+b+step)%len(logits)
            coin=sum(probs[:chosen])+probs[chosen]/2
            if greedy[b]:probs=[float(j==chosen) for j in range(len(logits))]
            bt.append(ids[b][step][chosen]);bq.append(probs);bu.append(coin);pred=chosen
        tokens.append(bt);rows.append(bq);coins.append(bu)
    return tokens,rows,coins


def protocol():
    return dict(scope='DFlash2 selector operator/worker sampling gate only; not projection/attention/model quality/performance',batches=[1,2,4],block=8,candidate_top_k=16,vocab=154880,turns=16,temperature_rows=[1.,.7,2.,1.],greedy_patterns=['all_sample','alternating'],execution=['eager','graph replay with changed coins/masks'],q_tolerance=dict(atol=2e-6,rtol=2e-5),accept='Actual worker sparseQscatter->DSPARK p/q; force all8prefixes with ratio1/2, ensureQbuffer cleared between changed candidateIDs and capacity reuse',seed=20261016,sources={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['tests/test_sglang_dflash2_selector.py','r9700_sglang/patches/sglang-v0.5.20-rocm-eagle-sampling.patch']})


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--prepare',action='store_true');args=ap.parse_args();raw=json.dumps(protocol(),indent=2)+'\n'
    if args.prepare:
        args.out.mkdir(parents=True,exist_ok=False);(args.out/'protocol.json').write_text(raw);(args.out/'protocol.sha256').write_text(hashlib.sha256(raw.encode()).hexdigest()+'\n');return
    assert (args.out/'protocol.json').read_text()==raw and (args.out/'protocol.sha256').read_text().strip()==hashlib.sha256(raw.encode()).hexdigest()
    assert not (args.out/'results.jsonl').exists()
    import torch
    from sglang.srt.models.dflash import CandidateSelector
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
    assert torch.cuda.device_count()==1 and torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx1201'
    torch.manual_seed(protocol()['seed']);count=0;stub=SimpleNamespace(top_k=16);worker=SimpleNamespace(_draft_probs_buf=None)
    def record(value):
        with (args.out/'results.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    for bs in [1,2,4,2]:
        k=16;length=7;vocab=154880
        scores_cpu=torch.randn(bs,length,k,k,dtype=torch.float32)*.7
        ids_cpu=(torch.arange(bs*length*k).view(bs,length,k)*137+2)%vocab
        ids_cpu[:,:,-1]=vocab-1
        scores=scores_cpu.cuda();ids=ids_cpu.cuda();temps=torch.tensor(protocol()['temperature_rows'][:bs],device='cuda');mask=torch.zeros(bs,device='cuda',dtype=torch.bool);uniforms=torch.full((bs,length),.5,device='cuda')
        def walk():return CandidateSelector.sample_path(stub,candidate_ids=ids,scores=scores,uniforms=uniforms,temperatures=temps,greedy_mask=mask)
        walk();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):graph_tokens,graph_q=walk()
        maximum=0.
        for pattern in protocol()['greedy_patterns']:
          for turn in range(protocol()['turns']):
            greedy=[pattern=='alternating' and b%2==1 for b in range(bs)]
            expected,q,coins=oracle_walk(ids_cpu.tolist(),scores_cpu.tolist(),protocol()['temperature_rows'][:bs],greedy,turn)
            uniforms.copy_(torch.tensor(coins,device='cuda'));mask.copy_(torch.tensor(greedy,device='cuda'))
            for mode in ['eager','graph']:
                if mode=='eager':got,real_q=walk()
                else:graph.replay();got,real_q=graph_tokens,graph_q
                assert got.cpu().tolist()==expected
                reference=torch.tensor(q,device='cuda',dtype=torch.float32)
                torch.testing.assert_close(real_q,reference,atol=2e-6,rtol=2e-5)
                maximum=max(maximum,(real_q-reference).abs().max().item());count+=bs
        # Use the real realized selector Q, not a fabricated zero scratch Q.
        mask.zero_();tokens,q=walk();q=q.clone();candidates=torch.cat([torch.zeros(bs,1,device='cuda',dtype=torch.int64),tokens.long()],dim=1)
        info=SimpleNamespace(need_top_k_sampling=False,need_top_p_sampling=False,temperatures=torch.ones(bs,1,device='cuda'))
        draft_input=SimpleNamespace(max_top_k=k,uniform_top_k_value=k)
        for shift in [0,1]:
            active_ids=ids if shift==0 else torch.where(ids==vocab-1,ids-1,ids+1)
            candidates[:,1:]=torch.stack([active_ids[b].gather(1,torch.tensor([row.index(t) for row,t in zip(ids_cpu[b].tolist(),tokens[b].cpu().tolist())],device='cuda')[:,None]).squeeze(1) for b in range(bs)])
            target=torch.zeros(bs,8,vocab,device='cuda');target[:,0:7].scatter_(-1,active_ids,.5*q);target[:,-1].scatter_(-1,active_ids[:,-1],.5*q[:,-1]);target[:,:,0]=.5;logits=target.log().view(bs*8,vocab)
            for prefix in range(8):
                lengths=[(prefix+b)%8 for b in range(bs)];coins=torch.full((bs,7),.25,device='cuda');final=torch.full((bs,),.25,device='cuda')
                for b,n in enumerate(lengths):
                    if n<7:coins[b,n]=.75
                def fixed_rand(shape,**kwargs):
                    if tuple(shape)==(bs,7):return coins
                    if tuple(shape)==(bs,):return final
                    raise AssertionError(('unexpected RNG call',shape))
                with patch('torch.rand',side_effect=fixed_rand) as rng:
                    accepted,bonus=DFlashWorkerV2._selector_sampling_accept(worker,candidates=candidates,next_token_logits=logits,candidate_ids=active_ids,q_rows=q,sampling_info=info,draft_input=draft_input)
                assert rng.call_count==2 and accepted.cpu().tolist()==lengths and bonus.cpu().tolist()==[0]*bs
                assert not bool(worker._draft_probs_buf.any()),'Selector Q leaked after finally cleanup'
                count+=bs
        record(dict(batch=bs,block=8,selector_tokens_exact=True,realized_q_max_abs=maximum,all_accept_prefixes_pass=True,changed_ids_and_capacity_reuse_clean=True,cumulative_checks=count))
    (args.out/'complete.json').write_text(json.dumps(dict(status='pass',request_checks=count,scope=protocol()['scope']))+'\n')


def test_oracle_uses_realized_predecessor_row_and_greedy_point_mass():
    ids=[[[10,11],[20,21]]];scores=[[[[0.,0.],[0.,0.]],[[10.,-10.],[-10.,10.]]]]
    tokens,q,_=oracle_walk(ids,scores,[1.],[False],1)
    assert tokens==[[11,20]] and q[0][1][1]>.99
    tokens,q,_=oracle_walk(ids,scores,[1.],[True],0)
    assert tokens==[[10,20]] and q==[[[1.,0.],[1.,0.]]]


if __name__=='__main__':main()
