#!/usr/bin/env python3
"""Join vLLM CPU execute annotations to GPU kernels using HIP correlation IDs.

This is a diagnostic, not an unprofiled decode benchmark. It needs worker
traces collected with CPU and CUDA activities. A step's GPU envelope is the
earliest-to-latest kernel launched by that step across all participating ranks.
"""

import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path
import re
from statistics import median


def read_rank(path):
    rank = int(re.search(r"rank(\d+)", path.name).group(1))
    events = json.loads(path.read_text())["traceEvents"]
    steps = sorted((event for event in events
                    if event.get("cat") == "user_annotation"
                    and event.get("name", "").startswith("execute_")),
                   key=lambda event: event["ts"])
    if not steps:
        raise ValueError(f"{path}: no CPU execute annotations; request CPU+CUDA activities")
    starts = [event["ts"] for event in steps]
    correlation_to_step = {}
    collective_calls = Counter()
    for event in events:
        if event.get("cat") == "user_annotation" and event.get("name", "").startswith("nccl:"):
            collective_calls[event["name"]] += 1
        if event.get("cat") != "cuda_runtime":
            continue
        correlation = event.get("args", {}).get("correlation")
        if correlation is None:
            continue
        index = bisect_right(starts, event["ts"]) - 1
        if index >= 0 and event["ts"] < steps[index]["ts"] + steps[index]["dur"]:
            correlation_to_step[correlation] = index
    kernels = [[] for _ in steps]
    unassigned = 0
    for event in events:
        if event.get("cat") != "kernel":
            continue
        index = correlation_to_step.get(event.get("args", {}).get("correlation"))
        if index is None:
            unassigned += 1
        else:
            kernels[index].append(event)
    rows = []
    for step, launches in zip(steps, kernels):
        if not launches:
            rows.append({"name": step["name"], "cpu_ms": step["dur"] / 1000,
                         "kernel_count": 0})
            continue
        begin = min(event["ts"] for event in launches)
        end = max(event["ts"] + event["dur"] for event in launches)
        moe = [event for event in launches if "_gemv" in event["name"]]
        comm = [event for event in launches if "nccl" in event["name"].lower()]
        rows.append({
            "name": step["name"], "cpu_ms": step["dur"] / 1000,
            "gpu_begin_us": begin, "gpu_end_us": end,
            "gpu_envelope_ms": (end - begin) / 1000,
            "host_to_gpu_lag_ms": (begin - step["ts"]) / 1000,
            "kernel_count": len(launches),
            "moe_kernel_sum_ms": sum(event["dur"] for event in moe) / 1000,
            "nccl_kernel_sum_ms": sum(event["dur"] for event in comm) / 1000,
            "moe_kernel_count": len(moe), "nccl_kernel_count": len(comm),
            "nccl_episodes": sorted((event["ts"], event["ts"] + event["dur"])
                                    for event in comm),
        })
    return {"rank": rank, "path": str(path), "steps": rows,
            "total_kernels": sum(len(x) for x in kernels) + unassigned,
            "unassigned_kernels": unassigned,
            "collective_cpu_annotations": dict(collective_calls)}


def combine(ranks):
    nsteps = {len(rank["steps"]) for rank in ranks}
    if len(nsteps) != 1:
        raise ValueError(f"ranks have unequal step counts: {nsteps}")
    combined = []
    for index in range(nsteps.pop()):
        rows = [rank["steps"][index] for rank in ranks]
        names = {row["name"] for row in rows}
        if len(names) != 1:
            raise ValueError(f"step {index} has different scheduler annotations: {names}")
        if any(not row["kernel_count"] for row in rows):
            continue
        begin = min(row["gpu_begin_us"] for row in rows)
        end = max(row["gpu_end_us"] for row in rows)
        episodes = [row["nccl_episodes"] for row in rows]
        paired = len({len(x) for x in episodes}) == 1
        skew = []
        if paired:
            for group in zip(*episodes):
                starts = [event[0] for event in group]
                ends = [event[1] for event in group]
                skew.append({"entry_us": max(starts) - min(starts),
                             "exit_us": max(ends) - min(ends)})
        combined.append({
            "step": index, "name": rows[0]["name"],
            "all_rank_gpu_envelope_ms": (end - begin) / 1000,
            "rank_gpu_envelope_ms": [row["gpu_envelope_ms"] for row in rows],
            "rank_cpu_execute_ms": [row["cpu_ms"] for row in rows],
            "rank_moe_kernel_sum_ms": [row["moe_kernel_sum_ms"] for row in rows],
            "rank_nccl_kernel_sum_ms": [row["nccl_kernel_sum_ms"] for row in rows],
            "nccl_episode_count": len(skew) if paired else None,
            "nccl_entry_skew_median_us": median(x["entry_us"] for x in skew) if skew else None,
            "nccl_exit_skew_median_us": median(x["exit_us"] for x in skew) if skew else None,
        })
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    paths = list(args.trace_dir.glob("*rank*.pt.trace.json"))
    if len(paths) != 8:
        raise SystemExit(f"expected eight worker traces, got {len(paths)}")
    ranks = sorted((read_rank(path) for path in paths), key=lambda row: row["rank"])
    if [rank["rank"] for rank in ranks] != list(range(8)):
        raise SystemExit("expected worker ranks 0..7")
    steps = combine(ranks)
    result = {"trace_dir": str(args.trace_dir), "ranks": ranks, "steps": steps,
              "summary": {"paired_steps": len(steps),
                          "median_all_rank_gpu_envelope_ms": median(
                              row["all_rank_gpu_envelope_ms"] for row in steps)}}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
