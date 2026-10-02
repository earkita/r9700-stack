#!/usr/bin/env python3
"""Bounded GLM answer-quality probes; separate from BetterBench throughput.

Corpus: JSON list of {id, messages, expected, scoring: exact|json}.
See notes/glm-testing.md. This runner never changes the serving runtime.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
import urllib.request
import uuid

from api_smoke import collect_stream


def read_url(base, route, body=None, timeout=30):
    request = urllib.request.Request(base + route,
        None if body is None else json.dumps(body).encode(),
        {"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=timeout)


def metrics(base):
    with read_url(base, "/metrics") as response:
        raw = response.read().decode()
    result = {}
    for name in ("num_requests_running", "num_requests_waiting",
                 "prefix_cache_queries_total", "prefix_cache_hits_total"):
        values = [float(line.rsplit(" ", 1)[1]) for line in raw.splitlines()
                  if line.startswith("vllm:" + name + "{")]
        if not values:
            raise RuntimeError("Missing metric: " + name)
        result[name] = sum(values)
    cache = [line for line in raw.splitlines() if line.startswith("vllm:cache_config_info{")]
    if len(cache) != 1:
        raise RuntimeError("Require one observed cache configuration")
    result["prefix_cache"] = 'enable_prefix_caching="True"' in cache[0]
    result["cache_config"] = cache[0]
    return result


def classify(response, case):
    if response.get("finish_reason") != "stop" or not response.get("content", "").strip():
        return "incomplete"
    answer = response["content"].strip()
    if case["scoring"] == "json":
        try:
            # Reject duplicate keys: an overwritten wrong field is not a pass.
            def unique(pairs):
                obj = {}
                for key, value in pairs:
                    if key in obj:
                        raise ValueError("Duplicate JSON key")
                    obj[key] = value
                return obj
            answer = json.loads(answer, object_pairs_hook=unique)
        except (ValueError, TypeError):
            return "wrong_completed"
    return "correct_completed" if answer == case["expected"] else "wrong_completed"


def token_counts(usage):
    total = usage.get("completion_tokens")
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    # API accounting remainder, not a separately tokenized visible answer.
    remainder = total - reasoning if total is not None and reasoning is not None else None
    return {"prompt": usage.get("prompt_tokens"), "completion": total,
            "reasoning": reasoning, "non_reasoning_by_subtraction": remainder}


NIAH_KEY_PATTERN = r"R9700-NIAH-[0-9]{3}-[0-9A-F]{6}"


def is_niah(case):
    # Recognize frozen v1 corpora without changing their contents or hashes.
    return case.get("task") == "niah" or bool(
        re.fullmatch(r"niah_[0-9]+_depth[0-9]{3}", case.get("id", "")))


def niah_scores(response, case, backend_runtime_error=None):
    """Separate retrieval evidence from exact-answer and format compliance.

    For these synthetic keys, retrieval means an intact key in generated
    content or reasoning, not a semantic judge or an assertion of correctness
    of surrounding prose. Record the channels so reasoning-only retrieval
    cannot be mistaken for a compliant final answer.
    """
    result = dict(strict_exact_match="UNKNOWN", semantic_needle_retrieval="UNKNOWN",
                  output_format_compliance="UNKNOWN", generation_completed="UNKNOWN",
                  backend_runtime_error=backend_runtime_error or ("NO" if response else "UNKNOWN"),
                  retrieval_channels=[])
    if response is None:
        return result
    pattern = case.get("key_pattern", NIAH_KEY_PATTERN)
    expected = case["expected"]
    if case["scoring"] != "exact" or not re.fullmatch(pattern, expected):
        raise ValueError("NIAH requires an exact key matching key_pattern")
    channels = {name: response.get(name) or "" for name in ("content", "reasoning", "reasoning_content")}
    needle = re.compile(r"(?<![\w-])" + re.escape(expected) + r"(?![\w-])")
    found = [name for name, text in channels.items() if needle.search(text)]
    result.update(
        strict_exact_match="PASS" if classify(response, case) == "correct_completed" else "FAIL",
        semantic_needle_retrieval="PASS" if found else "FAIL",
        output_format_compliance="PASS" if re.fullmatch(pattern, channels["content"].strip()) else "FAIL",
        generation_completed="PASS" if response.get("finish_reason") == "stop" else "FAIL",
        retrieval_channels=found)
    return result


def summarize_niah(rows):
    fields = ("strict_exact_match", "semantic_needle_retrieval", "output_format_compliance",
              "generation_completed", "backend_runtime_error")
    return dict(attempts=len(rows), **{field: dict(Counter(row["niah"][field] for row in rows))
                                     for field in fields})


def stream(base, body, timeout):
    start = time.monotonic()
    timing = {"ttft_ms": None, "first_content_ms": None}

    def observed(lines):
        for raw in lines:
            if raw.startswith(b"data:") and raw[5:].strip() != b"[DONE]":
                event = json.loads(raw[5:])
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                        now = (time.monotonic() - start) * 1000
                        if timing["ttft_ms"] is None:
                            timing["ttft_ms"] = now
                        if delta.get("content") and timing["first_content_ms"] is None:
                            timing["first_content_ms"] = now
            yield raw

    with read_url(base, "/v1/chat/completions", body, timeout) as response:
        result = collect_stream(observed(response))
    timing["end_to_end_ms"] = (time.monotonic() - start) * 1000
    return result, timing


def summarize(rows):
    groups = {}
    for phase in sorted({row["phase"] for row in rows}):
        selected = [row for row in rows if row["phase"] == phase]
        counts = Counter(row["status"] for row in selected)
        completed = counts["correct_completed"] + counts["wrong_completed"]
        ttfts = [row["timing"]["ttft_ms"] for row in selected if row.get("timing", {}).get("ttft_ms") is not None]
        groups[phase] = {
            "attempts": len(selected), "counts": dict(counts),
            "completion_rate": completed / len(selected),
            "correct_completed_over_attempts": counts["correct_completed"] / len(selected),
            "completed_denominator": completed,
            "accuracy_among_completed": counts["correct_completed"] / completed if completed else None,
            "ttft_median_ms": statistics.median(ttfts) if ttfts else None,
        }
        niah = [row for row in selected if "niah" in row]
        if niah:
            groups[phase]["niah"] = summarize_niah(niah)
    niah_groups = {}
    for row in rows:
        if "niah" in row:
            key = f"{(row.get('tokens') or {}).get('prompt', 'unknown')}/{row['phase']}"
            niah_groups.setdefault(key, []).append(row)
    return {"groups": groups,
            "niah_groups": {key: summarize_niah(value) for key, value in niah_groups.items()},
            "scope": "Bounded API quality probes, not a full quality suite or BetterBench score"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8080")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=["pilot", "off", "on"], required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--effort", choices=["low", "high", "max"], default="high")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 5678])
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.repetitions < 1 or args.timeout <= 0:
        parser.error("Budget, repetitions and timeout must be positive")
    cases = json.loads(args.corpus.read_text())
    if len({c["id"] for c in cases}) != len(cases):
        parser.error("Duplicate corpus ids")
    if args.ids:
        if set(args.ids) - {c["id"] for c in cases}:
            parser.error("Unknown case id")
        cases = [c for c in cases if c["id"] in args.ids]
    for case in cases:
        if case["scoring"] not in ("exact", "json"):
            parser.error("Unknown scoring rule")
    if not cases:
        parser.error("Empty corpus")
    args.out.mkdir(parents=True, exist_ok=False)
    base = args.base.rstrip("/")
    before = metrics(base)
    if before["prefix_cache"] != (args.phase == "on"):
        raise RuntimeError("Observed APC setting does not match requested leg")
    with read_url(base, "/v1/models") as response:
        models = json.load(response)["data"]
    model = next(m for m in models if m["id"] == args.model)
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "top_k": -1,
                "min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                "repetition_penalty": 1.0, "max_tokens": args.max_tokens,
                "ignore_eos": False, "chat_template_kwargs": {"reasoning_effort": args.effort}}
    prompt_counts = {}
    for case in cases:
        with read_url(base, "/tokenize", {"model": args.model, "messages": case["messages"],
                      "add_generation_prompt": True,
                      "chat_template_kwargs": sampling["chat_template_kwargs"]}) as response:
            prompt_counts[case["id"]] = json.load(response)["count"]
        if prompt_counts[case["id"]] + args.max_tokens > model["max_model_len"]:
            raise RuntimeError("Prompt + budget exceeds context: " + case["id"])
    manifest = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "sampling": sampling, "prompt_counts": prompt_counts, "metrics_before": before,
                "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "model": model}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    for repetition in range(args.repetitions):
        for case in cases:
            for seed in args.seeds:
                salt = "glm-quality-" + uuid.uuid4().hex
                for phase in (["on_cold", "on_warm"] if args.phase == "on" else [args.phase]):
                    row = {"case": case["id"], "phase": phase, "seed": seed,
                           "repetition": repetition, "expected": case["expected"], "scoring": case["scoring"]}
                    body = {"model": args.model, "messages": case["messages"], **sampling,
                            "seed": seed, "cache_salt": salt, "stream": True,
                            "stream_options": {"include_usage": True}}
                    row["request"] = body
                    fatal = None
                    try:
                        before = metrics(base)
                        if before["num_requests_running"] or before["num_requests_waiting"]:
                            raise RuntimeError("Require idle endpoint for isolated cache observations")
                        result, timing = stream(base, body, args.timeout)
                        row.update(response=result, timing=timing, status=classify(result, case),
                                   tokens=token_counts(result.get("usage") or {}))
                        if row["tokens"]["prompt"] != prompt_counts[case["id"]]:
                            raise RuntimeError("Tokenized and observed prompt counts differ")
                        after = metrics(base)
                        if args.phase == "on":
                            for _ in range(120):
                                if after["prefix_cache_queries_total"] - before["prefix_cache_queries_total"] >= row["tokens"]["prompt"]:
                                    break
                                time.sleep(0.1)
                                after = metrics(base)
                        hits = after["prefix_cache_hits_total"] - before["prefix_cache_hits_total"]
                        queries = after["prefix_cache_queries_total"] - before["prefix_cache_queries_total"]
                        row.update(metrics_before=before, metrics_after=after, cached_tokens=hits, cache_queries=queries)
                        if args.phase == "on" and queries != row["tokens"]["prompt"]:
                            raise RuntimeError("Cache query counters not isolated or not updated")
                        if phase != "on_warm" and hits != 0:
                            raise RuntimeError("Unexpected cache hit in off/cold request")
                        row["warm_hit_observed"] = hits > 0 if phase == "on_warm" else None
                    except Exception as exc:
                        fatal = exc
                        row.update(status="api_or_measurement_error", error=f"{type(exc).__name__}: {exc}")
                    if is_niah(case):
                        row["niah"] = niah_scores(row.get("response"), case,
                            backend_runtime_error="UNKNOWN" if fatal else None)
                    rows.append(row)
                    with (args.out / "requests.jsonl").open("a") as output:
                        output.write(json.dumps(row) + "\n")
                        output.flush()
                    (args.out / "summary.json").write_text(json.dumps(summarize(rows), indent=2) + "\n")
                    print(json.dumps({k: row[k] for k in ("case", "phase", "seed", "status", "niah", "tokens", "cached_tokens", "timing", "error") if k in row}), flush=True)
                    if fatal:
                        raise fatal


if __name__ == "__main__":
    main()
