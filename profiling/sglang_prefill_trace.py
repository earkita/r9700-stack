#!/usr/bin/env python3
"""Correlate opt-in GLM host ranges with asynchronous HIP work, offline.

CPU range durations measure host execution/launch overhead. GPU busy time is
the union of correlated device events, not their sum or their launch range's
CPU duration. Nested ranges are inclusive and must not be added together.
No throughput, critical-path speedup or unprofiled latency claim is derived.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re


def occupied(intervals):
    total, end = 0., float("-inf")
    for start, stop in sorted(intervals):
        if stop > end:
            total += stop - max(start, end)
            end = stop
    return total


def summarize(events):
    ranges = [e for e in events if e.get("ph") == "X"
              and e.get("cat") == "user_annotation"
              and e.get("name", "").startswith("r9700::")]
    # HIP uses CUDA-compatible Kineto categories and correlation IDs.
    # IDs must be unique in a rank trace. Ambiguities remain unassigned.
    launches = defaultdict(list)
    for event in events:
        if event.get("cat") in ("cuda_runtime", "hip_runtime"):
            key = event.get("args", {}).get("correlation")
            if key is not None:
                launches[key].append(event)
    device = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
              and e.get("dur", 0) > 0]
    groups = defaultdict(list)
    for event in ranges:
        groups[event["name"]].append(event)
    result = {}
    for name, annotations in sorted(groups.items()):
        gpu = []
        for event in device:
            calls = launches.get(event.get("args", {}).get("correlation"), [])
            if len(calls) != 1:
                continue
            launch = calls[0]
            if any(launch.get("pid") == span.get("pid")
                   and launch.get("tid") == span.get("tid")
                   and span["ts"] <= launch["ts"] < span["ts"] + span["dur"]
                   for span in annotations):
                gpu.append(event)
        intervals = [(e["ts"], e["ts"] + e["dur"]) for e in gpu]
        result[name] = dict(
            calls=len(annotations),
            host_inclusive_us=sum(e["dur"] for e in annotations),
            host_occupied_us=occupied((e["ts"], e["ts"]+e["dur"]) for e in annotations),
            correlated_device_events=len(gpu),
            correlated_gpu_busy_us=occupied(intervals) if gpu else None,
            correlated_gpu_envelope_us=(max(b for _, b in intervals)-min(a for a, _ in intervals)) if gpu else None,
            device_event_sum_us=sum(e["dur"] for e in gpu) if gpu else None,
        )
    ambiguous = sum(len(launches.get(e.get("args", {}).get("correlation"), [])) != 1 for e in device)
    return dict(ranges=result, total_device_events=len(device),
                kernel_events=sum(e.get("cat") == "kernel" for e in device),
                missing_or_ambiguous_launch_correlations=ambiguous,
                total_gpu_busy_us=occupied((e["ts"],e["ts"]+e["dur"]) for e in device) if device else None,
                first_gpu_us=min((e["ts"] for e in device),default=None),
                last_gpu_us=max((e["ts"]+e["dur"] for e in device),default=None))


def read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        trace = json.load(f)
    match = re.search(r"TP-(\d+)(?:\D|$)", path.name)
    if match is None:
        raise ValueError(f"No explicit TP rank in trace filename: {path}")
    report = summarize(trace["traceEvents"])
    if not report["ranges"]:
        raise ValueError(f"No GLM range annotations in {path}; cannot infer cache subcomponent cost")
    return dict(rank=int(match[1]),path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                baseTimeNanoseconds=trace.get("baseTimeNanoseconds"),**report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("traces", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allow-cpu-only", action="store_true",
                   help="Explicit offline host-only audit; missing GPU time remains unknown")
    args = p.parse_args()
    if args.out.exists():
        raise RuntimeError("Refusing to overwrite a diagnostic report")
    ranks = sorted((read(path) for path in args.traces),key=lambda r:r["rank"])
    if [r["rank"] for r in ranks] != list(range(8)):
        raise ValueError("Expected exactly one trace from each TP rank 0..7")
    bases = [r["baseTimeNanoseconds"] for r in ranks]
    shared_axis = None not in bases and len(set(bases)) == 1
    missing = [r["rank"] for r in ranks if not r["kernel_events"]]
    report = dict(scope="profiled diagnostic only; not an unprofiled throughput measurement",
                  status="incomplete_missing_gpu_activity" if missing else "device_activity_present",
                  missing_kernel_ranks=missing,
                  shared_timestamp_base_verified=shared_axis,
                  cross_rank_wall_envelope_us=(max(r["last_gpu_us"] for r in ranks)-min(r["first_gpu_us"] for r in ranks)) if shared_axis and not missing else None,
                  ranks=ranks,
                  limitations=["Nested range costs are inclusive; do not add them together",
                               "CPU duration is not GPU execution time; device work may finish after the host range",
                               "A range without correlated GPU work has unknown device cost, not measured zero",
                               "Trace envelope is not a proved critical path or unprofiled request latency",
                               "Different trace bases forbid cross-rank timestamp comparison here",
                               "Successful request/export does not prove GPU tracing; absent activity is unknown, never zero"])
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    if missing and not args.allow_cpu_only:
        raise SystemExit(f"GPU kernel activity missing from TP ranks {missing}; incomplete report saved")


if __name__ == "__main__":
    main()
