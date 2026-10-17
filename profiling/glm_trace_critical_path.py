"""Analyze an existing eight-rank GLM CUDA/HIP trace without touching GPUs.

Usage: python3 profiling/glm_trace_critical_path.py TRACE_DIRECTORY > summary.json
The selected window matches the 2026-10-14 diagnostic report by default.
GPU kernel timestamps can show overlap and arrival skew, but cannot identify
NCCL operation types, tensor sizes, or exact model-step boundaries.
"""

import argparse
import glob
import json
import statistics as stats
from pathlib import Path


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def union_us(intervals):
    length = 0.0
    end = float("-inf")
    for start, stop in sorted(intervals):
        if start >= end:
            length += stop - start
            end = stop
        elif stop > end:
            length += stop - end
            end = stop
    return length


def overlap_us(left, right):
    i = j = 0
    total = 0.0
    left, right = sorted(left), sorted(right)
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def merged(intervals):
    result = []
    for start, stop in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(stop, result[-1][1]))
        else:
            result.append((start, stop))
    return result


def load_trace(path):
    data = json.loads(Path(path).read_text())
    kernels = sorted((e for e in data["traceEvents"] if e.get("cat") == "kernel"
                      and e.get("dur", 0) > 0), key=lambda e: e["ts"])
    return data["baseTimeNanoseconds"], kernels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir")
    parser.add_argument("--start-ms", type=float, default=600)
    parser.add_argument("--end-ms", type=float, default=1600)
    args = parser.parse_args()
    paths = sorted(glob.glob(f"{args.trace_dir}/dp0_pp0_tp*_rank*.pt.trace.json"))
    if len(paths) != 8:
        raise SystemExit(f"expected eight TP traces, found {len(paths)}")
    traces = [load_trace(path) for path in paths]
    bases = [base for base, _ in traces]
    if len(set(bases)) != 1:
        raise SystemExit("trace baseTimeNanoseconds differ; no cross-rank comparison")
    t0 = min(kernels[0]["ts"] for _, kernels in traces)
    lo, hi = t0 + args.start_ms * 1000, t0 + args.end_ms * 1000
    if lo >= hi:
        raise SystemExit("window must have positive duration")
    rank_reports, collectives, nccl_by_rank, moe_by_rank = [], [], [], []
    activity_changes = []
    for rank, (_, kernels) in enumerate(traces):
        nccl_all = [e for e in kernels if e["name"].startswith("ncclDevKernel")]
        collectives.append(nccl_all)
        window = [e for e in kernels if e["ts"] < hi and e["ts"] + e["dur"] > lo]
        intervals = [(max(lo, e["ts"]), min(hi, e["ts"] + e["dur"])) for e in window]
        nccl = [(max(lo, e["ts"]), min(hi, e["ts"] + e["dur"])) for e in window
                if e["name"].startswith("ncclDevKernel")]
        moe = [(max(lo, e["ts"]), min(hi, e["ts"] + e["dur"])) for e in window
               if e["name"].startswith("_gemv.")]
        nccl_by_rank.append(nccl)
        moe_by_rank.append(moe)
        for start, stop in merged(intervals):
            activity_changes.extend(((start, 1), (stop, -1)))
        rank_reports.append({
            "rank": rank, "all_gpu_kernel_count": len(kernels),
            "all_nccl_kernel_count": len(nccl_all),
            "window_kernel_sum_ms": round(sum(b - a for a, b in intervals) / 1000, 3),
            "window_gpu_occupied_ms": round(union_us(intervals) / 1000, 3),
            "window_gpu_idle_ms": round(((hi - lo) - union_us(intervals)) / 1000, 3),
            "window_nccl_sum_ms": round(sum(b - a for a, b in nccl) / 1000, 3),
            "window_moe_gemv_sum_ms": round(sum(b - a for a, b in moe) / 1000, 3),
            "same_rank_nccl_moe_overlap_ms": round(overlap_us(nccl, moe) / 1000, 3),
        })
    counts = {len(events) for events in collectives}
    activity_ms = {str(rank): 0.0 for rank in range(9)}
    active, previous = 0, lo
    for when, delta in sorted(activity_changes):
        activity_ms[str(active)] += (when - previous) / 1000
        active += delta
        previous = when
    activity_ms[str(active)] += (hi - previous) / 1000
    cross_rank_nccl_moe_ms = [round(overlap_us(nccl_by_rank[r], merged(
        interval for peer in range(8) if peer != r for interval in moe_by_rank[peer])) / 1000, 3)
        for r in range(8)]
    episodes = []
    if len(counts) == 1:
        # Ordinal pairing is an inference; an op marker is needed to prove it.
        for index, events in enumerate(zip(*collectives)):
            starts = [e["ts"] for e in events]
            if not all(lo <= start < hi for start in starts):
                continue
            durations = [e["dur"] for e in events]
            ends = [start + duration for start, duration in zip(starts, durations)]
            episodes.append({"ordinal": index,
                             "start_skew_us": max(starts) - min(starts),
                             "end_skew_us": max(ends) - min(ends),
                             "mean_duration_us": stats.mean(durations),
                             "earliest_rank": starts.index(min(starts)),
                             "latest_rank": starts.index(max(starts)),
                             "earliest_duration_us": durations[starts.index(min(starts))],
                             "latest_duration_us": durations[starts.index(max(starts))],
                             "starts_ms_from_first_kernel": [round((s - t0) / 1000, 3) for s in starts],
                             "durations_us": [round(d, 3) for d in durations]})
    stats_out = {}
    if episodes:
        for field in ("start_skew_us", "end_skew_us", "mean_duration_us"):
            values = [episode[field] for episode in episodes]
            stats_out[field] = {"median": round(stats.median(values), 3),
                                "p90": round(quantile(values, .90), 3),
                                "max": round(max(values), 3)}
        stats_out["earliest_kernel_longer_fraction"] = round(stats.mean(
            episode["earliest_duration_us"] > episode["latest_duration_us"]
            for episode in episodes), 4)
    result = {
        "source": paths, "baseTimeNanoseconds": bases[0],
        "window_ms_from_first_kernel": [args.start_ms, args.end_ms],
        "limitations": ["No model-step markers", "NCCL kernel name lacks operation/dtype/count",
                        "ordinal NCCL pairing inferred from equal event counts",
                        "GPU-only activity omits CPU scheduling and API latency"],
        "ranks": rank_reports, "paired_nccl_kernel_count": len(episodes),
        "wall_ms_by_active_gpu_count": {key: round(value, 3) for key, value in activity_ms.items()},
        "nccl_overlap_with_other_rank_moe_ms": cross_rank_nccl_moe_ms,
        "paired_nccl_stats_us": stats_out,
        "representative_median_episode": min(episodes, key=lambda x: abs(
            x["start_skew_us"] - stats.median(e["start_skew_us"] for e in episodes))) if episodes else None,
        "largest_entry_skew_episode": max(episodes, key=lambda x: x["start_skew_us"]) if episodes else None,
        "largest_mean_duration_episode": max(episodes, key=lambda x: x["mean_duration_us"]) if episodes else None,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
