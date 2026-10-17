#!/usr/bin/env python3
"""Isolated gfx1201 sparse-verify warp diagnostic, not model decode.

Run with one visible, idle GPU after the required PCIe preflight. Compare
the exact FP8 GLM kernel at the C1/C4 verify row counts, retaining numerical
outputs and individual graph replay timings for every candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import triton

from r9700_vllm.attn.glm_sparse_fp8 import _sparse_attn_prefill_ragged_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Output directory must be new")
    args.out.mkdir(parents=True)
    props = torch.cuda.get_device_properties(0)
    if torch.cuda.device_count() != 1 or props.gcnArchName.split(":")[0] != "gfx1201":
        raise SystemExit("Expected one isolated gfx1201 GPU")
    torch.manual_seed(517)
    protocol = dict(rows=[5, 20], warps=[4, 8], heads=16, dim=512,
                    selected_kv=2048, kv_rows=4096, fp8_dtype=str(current_platform.fp8_dtype()),
                    seed=517, graph_kernels=12, replays=8, scale=512**-.5)
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    kv = torch.randn(4096, 512, device="cuda").to(current_platform.fp8_dtype())
    kv_scale = torch.tensor(0.5, dtype=torch.float32, device="cuda")
    sink = torch.randn(16, dtype=torch.float32, device="cuda")
    for rows in protocol["rows"]:
        q = torch.randn(rows, 16, 512, dtype=torch.bfloat16, device="cuda")
        ids = torch.stack([torch.randperm(4096, device="cuda")[:2048]
                           for _ in range(rows)]).int().flatten()
        indptr = torch.arange(rows + 1, dtype=torch.int32, device="cuda") * 2048
        baseline = None
        for warps in protocol["warps"]:
            out = torch.empty_like(q)

            def invoke():
                _sparse_attn_prefill_ragged_kernel[(rows, 1)](
                    q, kv, kv_scale, ids, indptr, sink, out,
                    *q.stride(), *kv.stride(), *out.stride(),
                    16, 512, kv.shape[0], protocol["scale"],
                    FP8_KV=True, HAS_ATTN_SINK=True,
                    BLOCK_H=16, BLOCK_D=512, BLOCK_K=16,
                    num_warps=warps)

            row = dict(rows=rows, warps=warps, status="error")
            try:
                for _ in range(3):
                    invoke()
                torch.cuda.synchronize()
                if baseline is None:
                    baseline = out.clone()
                    row.update(max_abs_diff=0.0, exact_fraction=1.0)
                else:
                    delta = (out.float() - baseline.float()).abs()
                    row.update(max_abs_diff=delta.max().item(),
                               exact_fraction=(out == baseline).float().mean().item())
                    torch.testing.assert_close(out, baseline, rtol=0, atol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(protocol["graph_kernels"]):
                        invoke()
                timings = []
                for _ in range(protocol["replays"]):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record(); graph.replay(); end.record(); end.synchronize()
                    timings.append(start.elapsed_time(end) / protocol["graph_kernels"])
                row.update(status="pass", per_kernel_ms=timings,
                           median_kernel_ms=statistics.median(timings))
            except Exception as error:
                row["error"] = repr(error)
            with (args.out / "results.jsonl").open("a") as file:
                file.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
