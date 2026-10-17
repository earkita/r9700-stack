#!/usr/bin/env python3
"""Isolated diagnostic of GLM TP8 Mamba BF16 projections at decode widths.

Compare stock torch/hipBLASLt with the existing r9k router GEMV kernel.
This is a single-GPU microbenchmark, never a whole-model speed claim.
Run only after the PCIe preflight recorded beside --out.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F


SHAPES = (("mamba_qkv_tp8", 1024, 4096),
          ("mamba_o_tp8", 4096, 1024),
          ("sparse_kvb_tp8", 4096, 512))
SPLITS = (1, 2, 4, 8)


def load_kernel(path: str):
    lib = ctypes.CDLL(path)
    op = lib.r9k_router_gemm
    op.restype = ctypes.c_int
    op.argtypes = [ctypes.c_long] * 6 + [ctypes.c_int] * 6 + [ctypes.c_float, ctypes.c_long]
    return op


def bench(fn, repetitions: int, nodes: int):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(nodes):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / nodes)
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lib", default="/opt/r9700/r9700_vllm/kernels/libr9k.so")
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=30)
    args = parser.parse_args()
    if args.repetitions < 3 or args.nodes < 10 or args.out.exists():
        parser.error("need a new output directory, >=3 repetitions and >=10 nodes")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU for isolated microbenchmark")
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(":")[0] == "gfx1201"
    args.out.mkdir(parents=True)
    op = load_kernel(args.lib)
    torch.manual_seed(1907)
    rows = []
    for m in (5, 20):
        for name, n, k in SHAPES:
            x = (torch.randn((m, k), device="cuda") * .2).to(torch.bfloat16)
            w = (torch.randn((n, k), device="cuda") * .2).to(torch.bfloat16)
            target = F.linear(x, w)
            torch.cuda.synchronize()
            reference = target.float()

            def stock():
                return F.linear(x, w)

            candidates = [("stock", None, stock)]
            for split in SPLITS:
                output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

                def candidate(split=split, output=output):
                    rc = op(x.data_ptr(), x.stride(0), w.data_ptr(), w.stride(0),
                            output.data_ptr(), output.stride(0), m, n, k, split,
                            1, 0, 1.0, torch.cuda.current_stream().cuda_stream)
                    if rc:
                        raise RuntimeError(f"router GEMV failed, rc={rc}, m={m}, n={n}, k={k}, split={split}")
                    return output

                candidates.append(("router", split, candidate))
            for impl, split, fn in candidates:
                output = fn()
                torch.cuda.synchronize()
                delta = (output.float() - reference).abs()
                rel_l2 = (delta.square().sum() / reference.square().sum().clamp_min(1e-30)).sqrt().item()
                row = dict(name=name, m=m, n=n, k=k, implementation=impl,
                           split=split, max_abs_diff=delta.max().item(), rel_l2=rel_l2,
                           start_monotonic=time.monotonic())
                try:
                    torch.testing.assert_close(output, target, atol=.03, rtol=.03)
                    samples = bench(fn, args.repetitions, args.nodes)
                    row.update(status="pass", times_us=samples,
                               median_us=statistics.median(samples))
                except Exception as exc:
                    row.update(status="fail", error=repr(exc))
                row["end_monotonic"] = time.monotonic()
                rows.append(row)
                with (args.out / "results.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps({k: row.get(k) for k in
                                  ("name", "m", "implementation", "split", "status", "median_us", "rel_l2", "error")}),
                      flush=True)
    (args.out / "complete.json").write_text(json.dumps({
        "kind": "single-GPU BF16 GEMV diagnostic; not model decode",
        "gpu": props.name, "arch": props.gcnArchName, "torch": torch.__version__,
        "shapes": SHAPES, "splits": SPLITS, "rows": (5, 20),
        "repetitions": args.repetitions, "nodes": args.nodes}, indent=2) + "\n")
    if any(row["status"] != "pass" for row in rows):
        raise SystemExit("failed candidates; see results")


if __name__ == "__main__":
    main()
