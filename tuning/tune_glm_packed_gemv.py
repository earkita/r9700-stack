#!/usr/bin/env python3
"""Diagnostic sweep of the GLM packed W4A4 GEMV launch geometry.

Run on one idle gfx1201 GPU after the mandatory PCIe preflight. This isolates
the routed expert GEMVs; it is not a model decode benchmark. The stock geometry
is always included, and every candidate is checked against its BF16 output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from vllm.triton_utils import triton

from r9700_vllm.moe.packed_gemv import _gemv


CONFIGS = ((8, 256, 4), (8, 128, 4), (16, 256, 4),
           (16, 128, 4), (32, 256, 4), (16, 256, 8),
           (8, 256, 2), (16, 256, 2), (32, 256, 2),
           (64, 256, 4))
SHAPES = (("gate_up", 512, 4096, 8, False),
          ("down", 4096, 256, 1, True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=(5, 20))
    parser.add_argument("--experts", type=int, default=288)
    parser.add_argument("--replays", type=int, default=6)
    parser.add_argument("--kernels-per-graph", type=int, default=30)
    args = parser.parse_args()
    if args.out.exists() or any(row < 1 or row > 32 for row in args.rows):
        raise SystemExit("Output must be new; rows must be in 1..32")
    args.out.mkdir(parents=True)
    props = torch.cuda.get_device_properties(0)
    arch = props.gcnArchName.split(":")[0]
    if arch != "gfx1201" or torch.cuda.device_count() != 1:
        raise SystemExit(f"Expected exactly one isolated gfx1201 GPU, got {arch}")
    protocol = dict(kind="diagnostic microbenchmark; not model decode", arch=arch,
                    rows=args.rows, experts=args.experts, replays=args.replays,
                    kernels_per_graph=args.kernels_per_graph, shapes=SHAPES,
                    configs=CONFIGS, seed=2701, torch=torch.__version__)
    raw = json.dumps(protocol, indent=2) + "\n"
    (args.out / "protocol.json").write_text(raw)
    (args.out / "protocol.sha256").write_text(hashlib.sha256(raw.encode()).hexdigest() + "\n")
    torch.manual_seed(2701)

    def invoke(x, w, scales, ids, router, out, k, n, divisor, weighted, config):
        bn, bk_limit, warps = config
        bk = min(bk_limit, k // 2)
        _gemv[(ids.numel(), triton.cdiv(n, bn))](
            x, w, scales, ids, router, out, k, n, divisor, weighted,
            bn, bk, num_warps=warps, enable_fp_fusion=False)

    for rows in args.rows:
        slots = rows * 8
        ids = torch.randint(0, args.experts, (slots,), dtype=torch.int32, device="cuda")
        router = torch.rand(slots, dtype=torch.float32, device="cuda")
        for shape, n, k, divisor, weighted in SHAPES:
            x = torch.randn((slots // divisor, k), dtype=torch.bfloat16, device="cuda")
            w = torch.randint(0, 256, (args.experts, n, k // 2), dtype=torch.uint8, device="cuda")
            scales = torch.randint(117, 129, (args.experts, n, k // 32), dtype=torch.uint8, device="cuda")
            baseline = torch.empty((slots, n), dtype=torch.bfloat16, device="cuda")
            invoke(x, w, scales, ids, router, baseline, k, n, divisor, weighted, CONFIGS[0])
            torch.cuda.synchronize()
            for config in CONFIGS:
                # A 64-column gate/up tile combines a long K with too many
                # live products. Keep this extra candidate on short-K down.
                if shape == "gate_up" and config[0] == 64:
                    continue
                out = torch.empty_like(baseline)
                row = dict(shape=shape, rows=rows, config=config, start_monotonic=time.monotonic())
                try:
                    for _ in range(3):
                        invoke(x, w, scales, ids, router, out, k, n, divisor, weighted, config)
                    torch.cuda.synchronize()
                    delta = (out.float() - baseline.float()).abs()
                    row["max_abs_diff"] = delta.max().item()
                    row["exact_fraction"] = (out == baseline).float().mean().item()
                    torch.testing.assert_close(out, baseline, rtol=.01, atol=.01)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(args.kernels_per_graph):
                            invoke(x, w, scales, ids, router, out, k, n, divisor, weighted, config)
                    times = []
                    for _ in range(args.replays):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        times.append(start.elapsed_time(end) / args.kernels_per_graph)
                    row["per_kernel_ms"] = times
                    row["median_kernel_ms"] = statistics.median(times)
                    row["status"] = "pass"
                except Exception as exc:
                    row["status"] = "fail"
                    row["error"] = repr(exc)
                row["end_monotonic"] = time.monotonic()
                with (args.out / "results.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps({k: row.get(k) for k in
                                  ("shape", "rows", "config", "status", "median_kernel_ms", "error")}),
                      flush=True)


if __name__ == "__main__":
    main()
