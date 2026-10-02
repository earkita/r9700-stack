#!/usr/bin/env python3
"""Summarize a torch-profiler trace: GPU time, launch counts and internal gaps.

Usage: torchprof-summary.py DIRECTORY_OR_TRACE [STEPS]
Directory selection retains the existing rank-0 preference. Pass an explicit
worker trace to compare ranks. GPU spans exclude time before the first event
and after the last; they are not end-to-end request latency measurements.
"""
import collections
import glob
import json
from pathlib import Path
import re
import sys


def interval_coverage(intervals):
    """Return span and occupied time, counting overlapping streams only once."""
    intervals = sorted((start, end) for start, end in intervals if end > start)
    if not intervals:
        return 0, 0
    first = start = intervals[0][0]
    end = intervals[0][1]
    occupied = 0
    for next_start, next_end in intervals[1:]:
        if next_start > end:
            occupied += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return end - first, occupied + end - start


def main(source, steps=1):
    if steps < 1:
        raise SystemExit("STEPS must be positive")
    paths = ([source] if Path(source).is_file() else
             sorted(glob.glob(source + "/*rank0*.json") or glob.glob(source + "/*.json")))
    if not paths:
        raise SystemExit(f"No torch-profiler JSON traces in {source}")
    path = paths[-1]
    events = json.loads(Path(path).read_text())["traceEvents"]
    times = collections.Counter()
    counts = collections.Counter()
    intervals = collections.defaultdict(list)
    launches = collections.Counter()
    for event in events:
        if event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset") or "dur" not in event:
            continue
        name = re.sub(r"\(.*", "", event["name"])[:90]
        times[name] += event["dur"]
        counts[name] += 1
        device = str(event.get("args", {}).get("device", event.get("pid", "unknown")))
        if event["cat"] == "kernel":
            launches[device] += 1
        if "ts" in event:
            intervals[device].append((event["ts"], event["ts"] + event["dur"]))
    total = sum(times.values())
    print(f"{path}\nSummed GPU event time {total/1e3:.1f} ms total, "
          f"{total/1e3/steps:.1f} ms/step over {steps} supplied steps")
    for device, spans in sorted(intervals.items()):
        span, occupied = interval_coverage(spans)
        idle_pct = 100 * (span - occupied) / span if span else 0
        print(f"GPU {device}: {launches[device]} kernel launches "
              f"({launches[device]/steps:.1f}/step), internal span {span/1e3:.1f} ms, "
              f"occupied {occupied/1e3:.1f} ms, gaps {idle_pct:.1f}%")
    for name, us in times.most_common(40):
        fraction = 100 * us / total if total else 0
        print(f"{fraction:5.1f}% {us/1e3/steps:8.2f} ms/step "
              f"{counts[name]/steps:7.1f}/step  {name}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
