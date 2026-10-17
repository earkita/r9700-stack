#!/usr/bin/env python3
"""Exclusive one-GPU KPool row-slicing correctness/workspace qualification.

The controller must save PCIe/HIP/BDF evidence before/during this probe and stop
all model work first. This is not a model TPS benchmark. Keep full unsliced
MQA/selection as the reference, sampled independent FP32 logits, unequal C1/C2/C4
lengths, original shared page mappings, and a padded-query boundary.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,required=True);out=p.parse_args().out
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if sha(out/"protocol.json") != (out/"protocol.sha256").read_text().strip():
        raise RuntimeError("Frozen protocol changed")
    plan=json.loads((out/"protocol.json").read_text())
    if (out/"attempts.jsonl").exists():raise RuntimeError("Refusing to overwrite attempts")
    for path,digest in plan["sources_container"].items():
        if sha(path)!=digest:raise RuntimeError("Probe source changed: "+path)
    hw_path=Path(__file__).parents[1]/"bench/baseline-earkita/hardware.py"
    if sha(hw_path)!=plan["hardware_sha256"]:raise RuntimeError("PCIe observer changed")
    spec=importlib.util.spec_from_file_location("hardware",hw_path);hw=importlib.util.module_from_spec(spec);spec.loader.exec_module(hw)
    def snapshot(label):
        value=hw.links(plan["hardware_mapping"])
        (out/(label+".pcie.json")).write_text(json.dumps(dict(utc_ns=time.time_ns(),links=value))+"\n")
        return value
    import torch
    from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
    from r9700_sglang.indexer_prefill import row_slices,select_rows
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]!="gfx1201":
        raise RuntimeError("Expected one isolated gfx1201 GPU")
    if not envs.SGLANG_DSA_FUSE_TOPK.get():raise RuntimeError("Shared page-table mapping must be enabled")
    torch.manual_seed(plan["seed"])
    for case in plan["cases"]:
        rows=case["queries_per_sequence"];keys=case["keys_per_sequence"];n=sum(rows);total_k=sum(keys);padded=n+5
        amplitude=case.get("amplitude",1.)
        feature_scale=torch.ones(128,device="cuda")
        if case.get("mixed_exponents",False):
            feature_scale[32:64]=1/32;feature_scale[64:]=1/1024
        q=(torch.randn(padded,32,128,device="cuda")*.25*amplitude*feature_scale).to(torch.float8_e4m3fn)
        k=(torch.randn(total_k,128,device="cuda")*.25*amplitude*feature_scale).to(torch.float8_e4m3fn)
        scales=(torch.rand(total_k,device="cuda")+.5)/amplitude
        weights=torch.randn(padded,32,device="cuda")*.1/amplitude
        assert torch.isfinite(q.float()).all() and torch.isfinite(k.float()).all(),"Synthetic FP8 operands must remain finite"
        starts=[];lengths=[];row_ids=[];offset=0
        for i,(count,key_count) in enumerate(zip(rows,keys)):
            starts.extend([offset]*count)
            lengths.extend([min(key_count,j*max(1,key_count//max(1,count-1))) for j in range(count)])
            row_ids.extend([i]*count);offset+=key_count
        ks=torch.tensor(starts,device="cuda",dtype=torch.int32)
        lengths=torch.tensor(lengths,device="cuda",dtype=torch.int32)
        ke=ks+lengths;seq=lengths*4+torch.arange(n,device="cuda",dtype=torch.int32)%4
        tables=torch.arange(len(rows)*(max(keys)*4+4),device="cuda",dtype=torch.int32).reshape(len(rows),-1)+10000
        row_index=torch.tensor(row_ids,device="cuda",dtype=torch.int32)
        state=SimpleNamespace(seq_lens_expanded=seq,pooled_seq_lens_expanded=lengths,ragged_q_ks=ks,ragged_q_ke=ke,
            ragged_paged_page_table=tables,ragged_paged_page_table_row_index=row_index)
        transform=case.get("transform","PAGED")
        offsets=(row_index*1000000+123).reshape(-1,1) if transform=="RAGGED" else None
        meta=SimpleNamespace(topk_transform_method=getattr(TopkTransformMethod,transform),
                             attn_metadata=SimpleNamespace(topk_indices_offset=offsets))
        mapping=dict(page_table=tables,page_table_row_index=row_index) if transform=="PAGED" else dict(topk_offsets=offsets)
        indexer=SimpleNamespace(index_kpool=4,index_topk=2048)
        def select(logits,pool_lens,**kwargs):
            return IndexerKPool._topk_from_kpool_logits(indexer,logits,pool_lens,**kwargs)
        indexer._topk_from_kpool_logits=select
        reference_link=snapshot(case["id"]+"-reference-before")
        torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        baseline_start=torch.cuda.memory_allocated()
        logits=fp8_mqa_logits(q[:n].contiguous(),k,scales,weights[:n].contiguous(),ks,ke,clean_logits=True)
        reference=select(logits,lengths,seq_lens=seq,**mapping,row_starts=ks,out_rows=padded)
        torch.cuda.synchronize();baseline_peak=torch.cuda.max_memory_allocated()
        hw.check_loaded([snapshot(case["id"]+"-reference-after")],reference_link,hw.expected_links(reference_link))
        logits_cpu=logits.cpu();reference_cpu=reference.cpu()
        oracle=[]
        for row in sorted({0,n//2,n-1}):
            # Independent FP32 per-head dot/ReLU/weighted sum, no AITER call.
            dense=((q[row].float()@k.float().T).relu()*weights[row,:,None]).sum(0)*scales
            valid=(torch.arange(total_k,device="cuda")>=ks[row])&(torch.arange(total_k,device="cuda")<ke[row])
            target=logits[row,valid];dense=dense[valid]
            err=(target-dense).float()
            oracle.append(dict(row=row,max_abs=err.abs().max().item() if err.numel() else 0.,relative_l2=(err.norm()/dense.norm().clamp_min(1e-10)).item()))
        del logits,reference,dense,target,err,valid
        torch.cuda.synchronize();torch.cuda.empty_cache()
        for limit in plan["row_limits"]:
            candidate_link=snapshot(case["id"]+f"-rows{limit}-before")
            torch.cuda.reset_peak_memory_stats();start=torch.cuda.memory_allocated()
            chunks=row_slices(n,total_k,limit)
            actual=select_rows(indexer,q,k,scales,weights,state,meta,chunks)
            torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated();actual_cpu=actual.cpu()
            hw.check_loaded([snapshot(case["id"]+f"-rows{limit}-after")],candidate_link,hw.expected_links(candidate_link))
            mismatch=int((actual_cpu!=reference_cpu).sum())
            del actual
            max_abs=0.;error_sq=ref_sq=0.
            for rowslice in chunks:
                chunk=fp8_mqa_logits(q[rowslice].contiguous(),k,scales,weights[rowslice].contiguous(),ks[rowslice],ke[rowslice],clean_logits=True).cpu()
                ref=logits_cpu[rowslice];finite=torch.isfinite(ref)
                assert torch.equal(torch.isfinite(chunk),finite)
                delta=chunk[finite]-ref[finite]
                if delta.numel():max_abs=max(max_abs,delta.abs().max().item());error_sq+=delta.double().square().sum().item();ref_sq+=ref[finite].double().square().sum().item()
                del chunk,ref,finite,delta
            row=dict(case=case["id"],transform=transform,amplitude=amplitude,mixed_exponents=case.get("mixed_exponents",False),row_limit=limit,queries=n,padded_queries=padded,pooled_keys=total_k,
                logits_max_abs=max_abs,logits_relative_l2=(error_sq/max(ref_sq,1e-20))**.5,topk_mismatches=mismatch,
                independent_oracle=oracle,baseline_peak_extra_bytes=baseline_peak-baseline_start,candidate_peak_extra_bytes=peak-start)
            with (out/"attempts.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
            tolerance=plan["tolerance"]
            if (mismatch or max_abs>tolerance["max_abs"] or row["logits_relative_l2"]>tolerance["relative_l2"]
                or any(x["max_abs"]>tolerance["oracle_max_abs"] or x["relative_l2"]>tolerance["oracle_relative_l2"] for x in oracle)):
                raise RuntimeError("Numerical/top-k gate failed; retain evidence, do not relax after measurement")
        del q,k,scales,weights,ks,ke,lengths,seq,tables,row_index,logits_cpu,reference_cpu,actual_cpu
        torch.cuda.empty_cache()
    (out/"status.json").write_text('{"status":"pass","scope":"oneGPU operator correctness/workspace only; full model still required"}\n')


if __name__=="__main__":main()
