#!/usr/bin/env python3
"""Rescore saved NIAH responses offline, preserving original artifacts/status.

An optional full rerun must cover identical requests (except cache salts).
It supplies reproduction evidence, never replaces the original strict result.
"""
import argparse
import hashlib
import json
from pathlib import Path

from glm_quality import is_niah, niah_scores, summarize


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rescore(rows, cases, rerun=None):
    def index(items):
        result = {}
        for row in items:
            key = (row["case"], row["phase"], row["seed"], row.get("repetition", 0))
            if key in result:
                raise ValueError("Duplicate request identity")
            result[key] = row
        return result

    original = index(rows)
    repeated = index(rerun) if rerun is not None else None
    if repeated is not None:
        if original.keys() != repeated.keys():
            raise ValueError("Require a full rerun, not selected failing examples")
        for key, row in original.items():
            other = repeated[key]
            if (other.get("error") or other.get("status") == "api_or_measurement_error"
                    or (other.get("response") or {}).get("finish_reason") != "stop"):
                raise ValueError("Reproduction assessment requires a completed, error-free full rerun")
            a = {k: v for k, v in row["request"].items() if k != "cache_salt"}
            b = {k: v for k, v in repeated[key]["request"].items() if k != "cache_salt"}
            if a != b:
                raise ValueError("Rerun changed prompt, sampling, budget or request options")
    result = []
    for key, row in original.items():
        case = cases[row["case"]]
        if not is_niah(case):
            raise ValueError("NIAH report requires NIAH cases")
        def score(item):
            return niah_scores(item.get("response"), case,
                "UNKNOWN" if item.get("error") or item.get("status") == "api_or_measurement_error" else None)
        scores = score(row)
        reproduction = "not_applicable" if scores["strict_exact_match"] == "PASS" else "not_assessed"
        if repeated is not None and scores["strict_exact_match"] == "FAIL":
            other = score(repeated[key])
            if other["strict_exact_match"] == "PASS":
                reproduction = "non-reproducible"
            elif other["strict_exact_match"] == "FAIL":
                reproduction = "strict_failure_recurred"  # Does not establish the same cause.
        result.append(dict(row, niah=scores, reproduction=reproduction))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--full-rerun", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    cases = {c["id"]: c for c in json.loads(args.corpus.read_text())}
    rows = rescore(read_rows(args.requests), cases,
                   read_rows(args.full_rerun) if args.full_rerun else None)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = summarize(rows)
    report["evidence"] = {name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                          for name, path in (("original", args.requests), ("corpus", args.corpus),
                                             ("full_rerun", args.full_rerun)) if path}
    report["reproduction_scope"] = "Observed in the supplied full request-matched rerun; no causal attribution."
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["niah_groups"], indent=2))


if __name__ == "__main__":
    main()
