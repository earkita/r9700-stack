#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated one-GPU GLM dense decode diagnostic; not model throughput."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


SHAPES = ((3072, 4096), (4096, 1536), (1024, 4096),
          (4096, 1024), (128, 4096), (288, 4096))
CONFIGS = ((4, 256), (8, 256), (16, 256), (8, 512), (16, 512))


def protocol():
    return dict(kind="single-GPU BF16 dense fallback C1 operator diagnostic",
                shapes=SHAPES, configs=CONFIGS, seed=20261015,
                order=["stock", "candidate", "stock"],
                warmups=3, repetitions=9, graph_nodes=64,
                tolerance=dict(relative_l2=0.02, max_abs=0.25),
                excluded="server startup, prefill, attention, MoE, collectives")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prepare", action="store_true")
    args = p.parse_args()
    spec = json.dumps(protocol(), indent=2) + "\n"
    digest = hashlib.sha256(spec.encode()).hexdigest() + "\n"
    if args.prepare:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "protocol.json").write_text(spec)
        (args.out / "protocol.sha256").write_text(digest)
        return
    if ((args.out / "protocol.json").read_text() != spec or
            (args.out / "protocol.sha256").read_text() != digest or
            (args.out / "results.jsonl").exists()):
        raise SystemExit("Frozen protocol missing/changed or output already exists")

    import torch
    import torch.nn.functional as F
    from r9700_sglang.bf16_dense import linear_one_row

    if torch.cuda.device_count() != 1:
        raise SystemExit("Expose exactly one idle GPU")
    props = torch.cuda.get_device_properties(0)
    if props.gcnArchName.split(":")[0] != "gfx1201":
        raise SystemExit("This diagnostic requires gfx1201")
    torch.manual_seed(protocol()["seed"])

    def measure(fn):
        for _ in range(protocol()["warmups"]):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(protocol()["graph_nodes"]):
                fn()
        graph.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(protocol()["repetitions"]):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000 / protocol()["graph_nodes"])
        return times

    result_file = args.out / "results.jsonl"
    for n, k in SHAPES:
        x = (torch.randn((1, k), device="cuda") * .2).bfloat16()
        w = (torch.randn((n, k), device="cuda") * .2).bfloat16()
        expected = F.linear(x, w)
        for bn, bk in CONFIGS:
            def stock():
                return F.linear(x, w)

            def candidate():
                return linear_one_row(x, w, block_n=bn, block_k=bk)

            actual = candidate()
            torch.cuda.synchronize()
            delta = actual.float() - expected.float()
            rel = (delta.norm() / expected.float().norm().clamp_min(1e-30)).item()
            max_abs = delta.abs().max().item()
            row = dict(n=n, k=k, block_n=bn, block_k=bk,
                       relative_l2=rel, max_abs=max_abs,
                       finite=bool(torch.isfinite(actual).all()))
            if not row["finite"] or rel > .02 or max_abs > .25:
                row.update(status="numeric_fail")
            else:
                for tag, fn in (("stock_a", stock), ("candidate", candidate),
                                ("stock_b", stock)):
                    samples = measure(fn)
                    row[tag] = dict(times_us=samples, median_us=statistics.median(samples))
                row["status"] = "pass"
            with result_file.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({key: row.get(key) for key in
                              ("n", "k", "block_n", "block_k", "status",
                               "relative_l2", "max_abs")}), flush=True)
    (args.out / "complete.json").write_text(json.dumps(dict(
        gpu=props.name, arch=props.gcnArchName, torch=torch.__version__,
        kind=protocol()["kind"]), indent=2) + "\n")


if __name__ == "__main__":
    main()
