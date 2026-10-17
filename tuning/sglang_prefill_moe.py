#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exclusive one-GPU grouped-W4A4 prefill candidate qualification, not model TPS.

Requires a frozen protocol and outside all-model/GPU coordination. The caller
saves participating GPU mapping and idle/loaded PCIe evidence. Never stops or
starts a serving model. All failed numerical attempts remain in the output.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(out/"protocol.json") != (out/"protocol.sha256").read_text().strip():
        raise RuntimeError("Frozen protocol changed")
    plan = json.loads((out/"protocol.json").read_text())
    if sha(__file__) != plan["runner_sha256"]:
        raise RuntimeError("Operator client changed")
    if (out/"attempts.jsonl").exists():
        raise RuntimeError("Refusing to overwrite operator attempts")
    hw_path=Path(__file__).parents[1]/"bench/baseline-earkita/hardware.py"
    if sha(hw_path)!=plan["hardware_sha256"]:
        raise RuntimeError("PCIe observer changed")
    spec=importlib.util.spec_from_file_location("hardware",hw_path)
    hw=importlib.util.module_from_spec(spec);spec.loader.exec_module(hw)
    def snapshot(label):
        value=hw.links(plan["hardware_mapping"])
        (out/(label+".pcie.json")).write_text(json.dumps(dict(utc_ns=time.time_ns(),links=value))+"\n")
        return value

    import torch
    from r9700_sglang.packed_moe import apply_sglang
    from r9700_sglang.hip_w4a4 import encode_qdq
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import quant_dequant_mxfp4
    from sglang_mxfp4_operator import reference_one_row

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1201":
        raise RuntimeError("Expected one isolated gfx1201 GPU")
    torch.manual_seed(plan["seed"])
    w1 = torch.randint(0,256,(288,512,2048),device="cuda",dtype=torch.uint8)
    w2 = torch.randint(0,256,(288,4096,128),device="cuda",dtype=torch.uint8)
    s1 = torch.randint(117,129,(288,512,128),device="cuda",dtype=torch.uint8)
    s2 = torch.randint(117,129,(288,4096,8),device="cuda",dtype=torch.uint8)
    layer = SimpleNamespace(w13_weight=w1,w2_weight=w2,w13_weight_scale=s1,w2_weight_scale=s2,
                            moe_runner_config=SimpleNamespace(swiglu_limit=10.,activation="silu"))
    os.environ.update(R9K_SGLANG_MOE_BACKEND="packed",R9K_SGLANG_GATE_SCALE_GROUP="0",R9K_SGLANG_DOWN_BN="8",R9K_FOLD="0")
    for rows in plan["rows"]:
        ids = torch.randint(0,288,(rows,8),device="cuda",dtype=torch.int32)
        # Concentrated routing tests real multirow tiles; even routes cover
        # padding, many experts and differing per-token activation scales.
        if plan["routing"] == "concentrated":
            ids %= 8
        routing = torch.softmax(torch.randn(rows,8,device="cuda"),dim=1)
        x = (torch.randn(rows,4096,device="cuda")*.25).bfloat16()
        x *= torch.linspace(.25,2.,rows,device="cuda")[:,None].bfloat16()
        dispatch = SimpleNamespace(hidden_states=x,topk_output=SimpleNamespace(topk_ids=ids,topk_weights=routing))
        sorted_ids,expert_ids,count = moe_align_block_size(ids,16,288)
        count_cpu = count.item()
        aligned = sorted_ids[:count_cpu].long()
        valid = aligned < ids.numel()
        assert torch.equal(ids.flatten()[aligned[valid]],expert_ids.repeat_interleave(16)[:count_cpu][valid])
        qdq = quant_dequant_mxfp4(x.contiguous());q,scale = encode_qdq(qdq)
        assert torch.equal(qdq,(q.float()*scale.repeat_interleave(32,dim=1)).bfloat16())

        def run(backend):
            os.environ["R9K_SGLANG_PREFILL_MOE"] = backend
            return apply_sglang(layer,dispatch).hidden_states
        before=snapshot(f"rows{rows}-numerical-packed-before")
        baseline = run("packed");torch.cuda.synchronize()
        baseline_link=snapshot(f"rows{rows}-numerical-packed-after")
        candidate_before=snapshot(f"rows{rows}-numerical-grouped-before")
        actual = run("hip_w4a4_grouped")
        torch.cuda.synchronize()
        candidate_link=snapshot(f"rows{rows}-numerical-grouped-after")
        hw.check_loaded([baseline_link],before,hw.expected_links(before))
        hw.check_loaded([candidate_link],candidate_before,hw.expected_links(candidate_before))
        delta = actual.float()-baseline.float()
        numerical = dict(rows=rows,routing=plan["routing"],max_abs=delta.abs().max().item(),relative_l2=(delta.norm()/baseline.float().norm()).item(),finite=bool(torch.isfinite(actual).all()),transport_exact=True,alignment_exact=True,oracle=[])
        for row in sorted({0,rows//2,rows-1}):
            oracle = reference_one_row(x[row:row+1],ids[row],routing[row],w1,s1,w2,s2)
            error = actual[row:row+1].float()-oracle.float()
            control_error = baseline[row:row+1].float()-oracle.float()
            numerical["oracle"].append(dict(row=row,max_abs=error.abs().max().item(),relative_l2=(error.norm()/oracle.float().norm()).item(),baseline_max_abs=control_error.abs().max().item(),baseline_relative_l2=(control_error.norm()/oracle.float().norm()).item(),oracle_finite=bool(torch.isfinite(oracle).all())))
        with (out/"attempts.jsonl").open("a") as f:f.write(json.dumps(dict(numerical=numerical))+"\n")
        tol = plan["tolerance"]
        for item in [numerical,*numerical["oracle"]]:
            if (item["max_abs"] > tol["max_abs"] or item["relative_l2"] > tol["relative_l2"]
                    or item.get("baseline_max_abs", 0) > tol["max_abs"]
                    or item.get("baseline_relative_l2", 0) > tol["relative_l2"]
                    or not item.get("oracle_finite", True) or not numerical["finite"]):
                raise RuntimeError("Numerical gate failed; do not relax tolerances after measurement")
        for phase_index,backend in enumerate(plan["order"]):
            tag=f"rows{rows}-phase{phase_index}-{backend}"
            before=snapshot(tag+"-before")
            for _ in range(plan["warmups"]):run(backend)
            torch.cuda.synchronize()
            loaded=[snapshot(tag+"-warmup-after")]
            values=[]
            torch.cuda.reset_peak_memory_stats()
            for repetition in range(plan["repetitions"]):
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();run(backend);end.record();end.synchronize();values.append(start.elapsed_time(end))
                # Outside the CUDA-event interval, within the planned series.
                loaded.append(snapshot(tag+f"-repetition{repetition}-after"))
            row = dict(rows=rows,backend=backend,gpu_ms=values,median_gpu_ms=statistics.median(values),peak_allocated_bytes=torch.cuda.max_memory_allocated())
            with (out/"attempts.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
            hw.check_loaded(loaded,before,hw.expected_links(before))
            print(json.dumps(row),flush=True)
    (out/"status.json").write_text('{"status":"pass","scope":"one-GPU full operator only, no model performance claim"}\n')


if __name__ == "__main__":
    main()
