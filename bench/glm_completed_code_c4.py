#!/usr/bin/env python3
"""Frozen C4 natural-EOS coding comparison for the GLM sparse-verify opt-in.

Never starts a server. Four independent 131072-token requests share each batch;
the generated code is retained and checked after inference has finished.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import threading
import time

from glm_completed_code import (ROOT, HARDWARE, container_info, get, hardware,
                                post, runtime_metrics, save, sha, stream)
from sglang_glm_completed_code import evaluate, extract_code


SOURCES = (Path(__file__).resolve(), ROOT / "bench/glm_completed_code.py",
           ROOT / "bench/api_smoke.py", ROOT / "bench/sglang_glm_completed_code.py",
           HARDWARE)
WARP_KEY = "R9K_GLM_SPARSE_VERIFY_WARPS"
COUNTERS = ("spec_decode_num_draft_tokens_total",
            "spec_decode_num_accepted_tokens_total",
            "num_preemptions_total")


def counters(path):
    raw = Path(path).read_text()
    result = {}
    for name in COUNTERS:
        prefix = "vllm:" + name
        values = [float(line.rsplit(" ", 1)[1]) for line in raw.splitlines()
                  if line.startswith(prefix + "{") or line.startswith(prefix + " ")]
        result[name] = sum(values) if values else None
    return result


def overlap_seconds(attempts):
    """Intersection of intervals from first text update through each end."""
    if len(attempts) != 4 or any("timing" not in row for row in attempts):
        return 0.0
    beginning = max(row["started_at"] + row["timing"]["ttft_ms"] / 1000
                    for row in attempts)
    ending = min(row["ended_at"] for row in attempts)
    return max(0.0, ending - beginning)


def prepare(args):
    out = args.out.resolve()
    out.relative_to(ROOT / "bench/results")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.corpus, out / "corpus.json")
    corpus = json.loads((out / "corpus.json").read_text())
    if corpus.get("task") != "dependency_waves" or len(corpus.get("messages", [])) != 2:
        raise RuntimeError("Require frozen dependency_waves corpus")
    model = next(m for m in get(args.base, "/v1/models")["data"]
                 if m["id"] == args.model)
    count = post(args.base, "/tokenize", dict(
        model=args.model, messages=corpus["messages"], add_generation_prompt=True,
        chat_template_kwargs={"reasoning_effort": "high"}))["count"]
    if count != 131072 or count + args.max_tokens > model["max_model_len"]:
        raise RuntimeError(f"Actual context or completion room changed: {count}")
    runtime = container_info(args.container)
    if runtime["env"].get(WARP_KEY, "4") != "4":
        raise RuntimeError("Prepare on baseline 4-warp K4")
    plan = dict(schema=1, name="glm-k4-c4-natural-eos-sparse-verify-v1",
                scope="completed coding C4 128K; not 256K or multi-turn agent",
                experiment=f"{WARP_KEY} 4/8/4; same image, args and model",
                sources={str(path.relative_to(ROOT)): sha(path) for path in SOURCES},
                corpus_sha256=sha(out / "corpus.json"),
                base=args.base, model=args.model, image_id=runtime["image_id"],
                runtime_args=runtime["args"],
                runtime_env_without_warp={k: v for k, v in runtime["env"].items()
                                          if k != WARP_KEY},
                input_tokens=count, max_model_len=model["max_model_len"],
                concurrency=4, warmup_batches=1, measured_batches=3,
                sampling=dict(temperature=1.0, top_p=0.95, top_k=-1,
                              min_p=0.0, presence_penalty=0.0,
                              frequency_penalty=0.0, repetition_penalty=1.0,
                              max_tokens=args.max_tokens, ignore_eos=False,
                              chat_template_kwargs={"reasoning_effort": "high"}),
                seed=1234,
                cache_salt="glm-sparse-c4-" + hashlib.sha256(str(out).encode()).hexdigest()[:16],
                decode_metric="continuous output-token delta from first to last text update, excluding TTFT and first burst",
                overlap_min_seconds=0.5)
    save(out / "protocol.json", plan)
    (out / "protocol.sha256").write_text(sha(out / "protocol.json") + "\n")
    save(out / "prepared-runtime.json", runtime)
    print(json.dumps(dict(out=str(out), input_tokens=count,
                          image_id=runtime["image_id"])), flush=True)


def run(args):
    out = args.out.resolve()
    if sha(out / "protocol.json") != (out / "protocol.sha256").read_text().strip():
        raise RuntimeError("Frozen protocol changed")
    plan = json.loads((out / "protocol.json").read_text())
    if sha(out / "corpus.json") != plan["corpus_sha256"]:
        raise RuntimeError("Frozen corpus changed")
    for path, digest in plan["sources"].items():
        if sha(ROOT / path) != digest:
            raise RuntimeError(f"Frozen source changed: {path}")
    runtime = container_info(args.container)
    if (runtime["image_id"] != plan["image_id"]
            or runtime["args"] != plan["runtime_args"]
            or {k: v for k, v in runtime["env"].items() if k != WARP_KEY}
               != plan["runtime_env_without_warp"]
            or runtime["env"].get(WARP_KEY, "4") != args.warps):
        raise RuntimeError("Runtime identity differs beyond declared warp setting")
    leg = out / args.leg
    leg.mkdir(parents=True, exist_ok=False)
    save(leg / "runtime.json", runtime)
    hw = hardware()
    mapping = hw.hip_mapping(args.container)
    idle = hw.capture(leg / "pcie-before", mapping)
    expected = hw.expected_links(idle)
    save(leg / "pcie-expected.json", expected)
    corpus = json.loads((out / "corpus.json").read_text())
    current = post(plan["base"], "/tokenize", dict(
        model=plan["model"], messages=corpus["messages"], add_generation_prompt=True,
        chat_template_kwargs={"reasoning_effort": "high"}))["count"]
    if current != plan["input_tokens"]:
        raise RuntimeError("Actual token count changed")
    samples, done = [], threading.Event()
    def watch():
        while not done.is_set():
            samples.append(hw.links(mapping))
            done.wait(1)
    monitor = threading.Thread(target=watch)
    monitor.start()
    batches, all_rows = [], []
    try:
        for batch in range(plan["warmup_batches"] + plan["measured_batches"]):
            before_path = leg / f"batch-{batch:02d}.before.prom"
            before = runtime_metrics(plan["base"], before_path)
            if before["num_requests_running"] != 0 or before["num_requests_waiting"] != 0:
                raise RuntimeError("Endpoint not idle before isolated C4 batch")
            before_counters = counters(before_path)
            barrier = threading.Barrier(4)
            def one(slot):
                stem = leg / f"batch-{batch:02d}-slot-{slot}"
                body = dict(model=plan["model"], messages=corpus["messages"],
                            **plan["sampling"], seed=plan["seed"],
                            cache_salt=f'{plan["cache_salt"]}-{args.leg}-slot-{slot}',
                            stream=True, stream_options={"include_usage": True,
                                                         "continuous_usage_stats": True})
                row = dict(batch=batch, slot=slot, warmup=batch == 0, status="error")
                try:
                    barrier.wait(timeout=30)
                    row["started_at"] = time.perf_counter()
                    response, timing = stream(plan["base"], body, stem)
                    row["ended_at"] = time.perf_counter()
                    row.update(timing=timing, usage=response["usage"],
                               finish_reason=response["finish_reason"],
                               content_chars=len(response["content"]),
                               reasoning_chars=len(response["reasoning"]))
                    if response["usage"]["prompt_tokens"] != plan["input_tokens"]:
                        raise RuntimeError("Actual prompt token count changed")
                    if response["finish_reason"] != "stop":
                        row["status"] = "incomplete"
                    else:
                        row["status"] = "completed_unchecked"
                except Exception as error:
                    row["error"] = repr(error)
                return row
            with ThreadPoolExecutor(max_workers=4) as pool:
                rows = list(pool.map(one, range(4)))
            # CPU oracle runs only after all four GPU responses have finished.
            for row in rows:
                if row["status"] != "completed_unchecked":
                    continue
                stem = leg / f'batch-{batch:02d}-slot-{row["slot"]}'
                response = json.loads(Path(str(stem) + ".response.json").read_text())
                try:
                    code = extract_code("</think>" + response["content"])
                    Path(str(stem) + ".py").write_text(code + "\n")
                    row["execution"] = evaluate(code, runtime["image_id"])
                    row["status"] = ("correct_completed" if row["execution"]["exit_code"] == 0
                                     else "wrong_completed")
                except (ValueError, SyntaxError) as error:
                    row.update(status="wrong_completed", format_error=repr(error))
            after_path = leg / f"batch-{batch:02d}.after.prom"
            after = runtime_metrics(plan["base"], after_path)
            after_counters = counters(after_path)
            deltas = {key: (after_counters[key] - before_counters[key]
                            if None not in (after_counters[key], before_counters[key]) else None)
                      for key in COUNTERS}
            hits = {key: (after[key] - before[key]
                          if None not in (after[key], before[key]) else None)
                    for key in ("prefix_cache_hits_total", "external_prefix_cache_hits_total")}
            overlap = overlap_seconds(rows)
            batch_row = dict(batch=batch, warmup=batch == 0, overlap_seconds=overlap,
                             cache_hits=hits, counter_deltas=deltas,
                             acceptance=(deltas[COUNTERS[1]] / deltas[COUNTERS[0]]
                                         if deltas[COUNTERS[0]] else None),
                             all_correct=all(r["status"] == "correct_completed" for r in rows),
                             total_output_tokens=sum((r.get("usage") or {}).get("completion_tokens", 0)
                                                     for r in rows),
                             batch_wall_seconds=(max(r.get("ended_at", 0) for r in rows)
                                                 - min(r.get("started_at", 0) for r in rows)))
            batch_row["aggregate_tps_including_ttft"] = (
                batch_row["total_output_tokens"] / batch_row["batch_wall_seconds"]
                if batch_row["batch_wall_seconds"] > 0 else None)
            batches.append(batch_row)
            all_rows.extend(rows)
            with (leg / "attempts.jsonl").open("a") as file:
                for row in rows:
                    file.write(json.dumps(row) + "\n")
            with (leg / "batches.jsonl").open("a") as file:
                file.write(json.dumps(batch_row) + "\n")
            print(json.dumps(dict(batch=batch, all_correct=batch_row["all_correct"],
                                  overlap_seconds=overlap, cache_hits=hits,
                                  acceptance=batch_row["acceptance"],
                                  decode_tps=[(r.get("timing") or {}).get("decode_tps") for r in rows])),
                  flush=True)
    finally:
        done.set(); monitor.join()
        save(leg / "pcie-loaded.json", samples)
        hw.check_loaded(samples, idle, expected)
    measured = [r for r in all_rows if not r["warmup"]]
    warm_batches = [b for b in batches if not b["warmup"]]
    valid = (len(measured) == 12 and all(b["all_correct"] for b in batches)
             and all(b["overlap_seconds"] >= plan["overlap_min_seconds"] for b in warm_batches)
             and all((b["cache_hits"]["prefix_cache_hits_total"] or 0)
                     + (b["cache_hits"]["external_prefix_cache_hits_total"] or 0)
                     >= 4 * (plan["input_tokens"] - 2048) for b in warm_batches)
             and all(b["counter_deltas"]["num_preemptions_total"] in (0, None)
                     for b in warm_batches))
    save(leg / "summary.json", dict(qualified=valid, measured=len(measured),
         correct_completed=sum(r["status"] == "correct_completed" for r in measured),
         decode_tps_median=statistics.median(r["timing"]["decode_tps"] for r in measured
                                             if r["status"] == "correct_completed") if valid else None,
         ttft_ms_median=statistics.median(r["timing"]["ttft_ms"] for r in measured
                                          if r["status"] == "correct_completed") if valid else None,
         overlap_seconds=[b["overlap_seconds"] for b in warm_batches],
         acceptance=[b["acceptance"] for b in warm_batches],
         aggregate_tps_including_ttft=[b["aggregate_tps_including_ttft"] for b in warm_batches],
         all_attempts_retained=True))
    if not valid:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--corpus", type=Path, required=True)
    prep.add_argument("--container", default="glm53-k4-lmcache-20261017")
    prep.add_argument("--base", default="http://127.0.0.1:8080")
    prep.add_argument("--model", default="glm-5.3-flash")
    prep.add_argument("--max-tokens", type=int, default=4096)
    trial = subs.add_parser("run")
    trial.add_argument("--out", type=Path, required=True)
    trial.add_argument("--leg", required=True)
    trial.add_argument("--container", default="glm53-k4-lmcache-20261017")
    trial.add_argument("--warps", choices=["4", "8"], required=True)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else run)(args)


if __name__ == "__main__":
    main()
