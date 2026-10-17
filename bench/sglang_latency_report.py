#!/usr/bin/env python3
"""Offline client-visible latency audit of native cumulative-token captures.

A speculative SSE update can deliver several tokens at once. Count one gap to
that burst and zero client-visible gaps within it; their GPU generation times
are unknown. Do not label the update gap as every token's ITL or as a GPU step.
This report changes neither an existing measurement protocol nor its results.
"""
import argparse
import hashlib
import json
from pathlib import Path


def percentile(values, q):
    ordered = sorted(values)
    if not ordered or not 0 <= q <= 1:
        raise ValueError("Invalid percentile input")
    position = (len(ordered)-1)*q
    lower = int(position)
    upper = min(lower+1,len(ordered)-1)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)


def latencies(events):
    if len(events)<2:
        raise ValueError("Need multiple cumulative token updates")
    visible = []
    gaps = []
    for (ta,ca),(tb,cb) in zip(events,events[1:]):
        if tb < ta or not isinstance(cb,int) or not isinstance(ca,int) or cb <= ca:
            raise ValueError("Counters/timestamps must increase")
        gap = (tb-ta)*1000
        gaps.append(gap)
        visible.extend([gap]+[0.]*(cb-ca-1))
    return dict(post_first_update_tokens=len(visible),
                client_visible_itl_ms={f"p{p}":percentile(visible,p/100) for p in (50,95,99)},
                stream_update_gap_ms={f"p{p}":percentile(gaps,p/100) for p in (50,95,99)},
                mean_ms_per_post_first_update_token=sum(visible)/len(visible),
                note="Tokens in one SSE burst share an arrival timestamp. Zero gaps are client observations, not GPU time; sub-burst generation timing unknown. First burst excluded consistently with native decode TPS.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("Refusing to overwrite a latency report")
    results = []
    for line in args.requests.read_text().splitlines():
        row=json.loads(line)
        events=row.get("metrics",{}).get("events")
        if events:
            results.append(dict(
                index=row.get("index"),
                excluded_population=row.get("excluded_population"),
                excluded_warmup=row.get("excluded_warmup", row.get("excluded_population")),
                latency=latencies(events),
            ))
    args.out.write_text(json.dumps(dict(source=str(args.requests),source_sha256=hashlib.sha256(args.requests.read_bytes()).hexdigest(),scope="offline API-arrival audit; not GPU kernel/step latency or completed-answer quality",requests=results),indent=2)+"\n")


if __name__ == "__main__":
    main()
