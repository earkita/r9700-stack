#!/usr/bin/env python3
"""Summarize fixed-K speculative counter deltas on an otherwise idle server."""
import argparse
import json
from pathlib import Path
import re


def parse(text):
    result = {}
    for line in text.splitlines():
        m = re.match(r'vllm:spec_decode_(\w+)_total\{([^}]*)\}\s+(\S+)', line)
        if m:
            name, labels, value = m.groups()
            pos = re.search(r'position="(\d+)"', labels)
            key = name + (":" + pos[1] if pos else "")
            result[key] = result.get(key, 0) + float(value)
    return result


def summarize(before, after):
    a, b = parse(before), parse(after)
    required = {"num_drafts", "num_draft_tokens", "num_accepted_tokens"}
    if not required <= b.keys() or a.keys() - b.keys():
        raise ValueError("Missing speculative counters; require the same speculative server")
    delta = {k: v - a.get(k, 0) for k, v in b.items()}
    if any(v < 0 for v in delta.values()):
        raise ValueError("Counters decreased; the server may have restarted")
    rounds = delta.get("num_drafts", 0)
    proposed = delta.get("num_draft_tokens", 0)
    accepted = delta.get("num_accepted_tokens", 0)
    positions = []
    previous = rounds
    for k in sorted((k for k in delta if k.startswith("num_accepted_tokens_per_pos:")),
                    key=lambda k: int(k.split(":")[1])):
        count = delta[k]
        positions.append(dict(position=int(k.split(":")[1]) + 1, accepted=count,
                              fraction_of_rounds=count / rounds if rounds else None,
                              fraction_of_previous=count / previous if previous else None))
        previous = count
    return dict(rounds=rounds, proposed=proposed, accepted=accepted,
                acceptance=accepted / proposed if proposed else None,
                accepted_per_round=accepted / rounds if rounds else None,
                positions=positions,
                scope="Process-wide counter delta; includes all requests in the interval. "
                      "Per-position fractions assume a fixed K; end-of-request truncation affects them.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    text = json.dumps(summarize(args.before.read_text(), args.after.read_text()), indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
