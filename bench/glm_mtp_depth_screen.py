#!/usr/bin/env python3
"""Matched vLLM GLM NextN MTP3/4/5 C1 coding screen; no server lifecycle.

The 8K screen is a quality gate, not a 128K/C4 promotion benchmark. Keep
natural EOS, the full answer, actual token counts and every failed attempt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import threading

from glm_completed_code import (ROOT, HARDWARE, container_info, get, hardware,
                                post, runtime_metrics, save, sha, stream)
from sglang_glm_completed_code import evaluate, extract_code


SOURCES = (Path(__file__).resolve(), ROOT / "bench/glm_completed_code.py",
           ROOT / "bench/api_smoke.py", ROOT / "bench/sglang_glm_completed_code.py",
           ROOT / "serve/baseline-earkita/glm/glm-5.3-flash-c4-mtp3-lmcache.sh", HARDWARE)
SEEDS = (1234, 5678, 9012)
METRICS = ("spec_decode_num_draft_tokens_total",
           "spec_decode_num_accepted_tokens_total", "num_preemptions_total")


def metric_counters(path):
    result = {}
    for name in METRICS:
        prefix = "vllm:" + name
        values = [float(line.rsplit(" ", 1)[1]) for line in Path(path).read_text().splitlines()
                  if line.startswith(prefix + "{") or line.startswith(prefix + " ")]
        result[name] = sum(values) if values else None
    return result


def option(args, flag):
    return args[args.index(flag) + 1]


def prepare(args):
    out = args.out.resolve()
    out.relative_to(ROOT / "bench/results")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.corpus, out / "corpus.json")
    corpus = json.loads((out / "corpus.json").read_text())
    if corpus.get("task") != "dependency_waves" or len(corpus.get("messages", [])) != 2:
        raise RuntimeError("Require a frozen dependency_waves coding corpus")
    import subprocess
    image = json.loads(subprocess.check_output(
        ["docker", "image", "inspect", args.image], text=True))[0]["Id"]
    plan = dict(schema=1, name="glm-vllm-mtp345-c1-8k-natural-eos-v1",
                scope="8K C1 quality/decode screen only; no 128K/C4 promotion claim",
                image_id=image, corpus_sha256=sha(out / "corpus.json"),
                sources={str(p.relative_to(ROOT)): sha(p) for p in SOURCES},
                prompt_tokens=8192, max_model_len=131072, concurrency=1,
                warmup_seed=42, measured_seeds=list(SEEDS),
                sampling=dict(temperature=1.0, top_p=0.95, top_k=-1,
                              min_p=0.0, presence_penalty=0.0,
                              frequency_penalty=0.0, repetition_penalty=1.0,
                              max_tokens=4096, ignore_eos=False,
                              chat_template_kwargs={"reasoning_effort": "high"}),
                cache_policy="GPU APC only; one warmup, identical prompt/salt for measured requests",
                min_cache_hit_fraction=0.80,
                cache_salt="glm-mtp345-" + hashlib.sha256(str(out).encode()).hexdigest()[:16],
                decode_metric="actual continuous output-token delta after first text update / elapsed time",
                qualification="all three completed code answers pass 169-case oracle; measured APC hits >=80% of prompt; speculative and preemption counters present; no preemption")
    save(out / "protocol.json", plan)
    (out / "protocol.sha256").write_text(sha(out / "protocol.json") + "\n")
    print(json.dumps(dict(out=str(out), image_id=image,
                          corpus_sha256=plan["corpus_sha256"])), flush=True)


def run(args):
    out = args.out.resolve()
    if sha(out / "protocol.json") != (out / "protocol.sha256").read_text().strip():
        raise RuntimeError("Frozen protocol changed")
    plan = json.loads((out / "protocol.json").read_text())
    if sha(out / "corpus.json") != plan["corpus_sha256"]:
        raise RuntimeError("Frozen corpus changed")
    for path, digest in plan["sources"].items():
        if sha(ROOT / path) != digest:
            raise RuntimeError("Frozen source changed: " + path)
    runtime = container_info(args.container)
    cmd = runtime["args"]
    spec = json.loads(option(cmd, "--speculative-config"))
    if (runtime["image_id"] != plan["image_id"]
            or spec != {"method": "mtp", "num_speculative_tokens": args.depth}
            or option(cmd, "--max-num-seqs") != "1"
            or option(cmd, "--max-model-len") != str(plan["max_model_len"])
            or option(cmd, "--port") != str(args.port)
            or "--kv-transfer-config" in cmd
            or runtime["env"].get("R9K_GLM_DFLASH") != "0"):
        raise RuntimeError("Runtime differs from frozen GPU-APC MTP depth screen")
    base = f"http://127.0.0.1:{args.port}"
    model = next(m for m in get(base, "/v1/models")["data"]
                 if m["id"] == "glm-5.3-flash")
    if model["max_model_len"] != plan["max_model_len"]:
        raise RuntimeError("Effective model context changed")
    corpus = json.loads((out / "corpus.json").read_text())
    count = post(base, "/tokenize", dict(model="glm-5.3-flash",
                   messages=corpus["messages"], add_generation_prompt=True,
                   chat_template_kwargs={"reasoning_effort": "high"}))["count"]
    if count != plan["prompt_tokens"]:
        raise RuntimeError(f"Actual input count differs: {count}")
    leg = out / f"mtp{args.depth}"
    leg.mkdir(parents=True, exist_ok=False)
    save(leg / "runtime.json", runtime)
    hw = hardware()
    mapping = hw.hip_mapping(args.container)
    idle = hw.capture(leg / "pcie-before", mapping)
    expected = hw.expected_links(idle)
    save(leg / "pcie-expected.json", expected)
    samples, done = [], threading.Event()
    def monitor():
        while not done.is_set():
            samples.append(hw.links(mapping))
            done.wait(1)
    watcher = threading.Thread(target=monitor)
    watcher.start()
    rows = []
    try:
        for index, seed in enumerate((plan["warmup_seed"], *plan["measured_seeds"])):
            stem = leg / f"attempt-{index:02d}"
            before_file = Path(str(stem) + ".before.prom")
            before = runtime_metrics(base, before_file)
            before_counters = metric_counters(before_file)
            if before["num_requests_running"] or before["num_requests_waiting"]:
                raise RuntimeError("Endpoint must be idle before each attempt")
            body = dict(model="glm-5.3-flash", messages=corpus["messages"],
                        **plan["sampling"], seed=seed, cache_salt=plan["cache_salt"],
                        stream=True, stream_options={"include_usage": True,
                                                     "continuous_usage_stats": True})
            row = dict(index=index, seed=seed, warmup=index == 0, status="error")
            try:
                response, timing = stream(base, body, stem)
                row.update(timing=timing, usage=response["usage"],
                           finish_reason=response["finish_reason"],
                           content_chars=len(response["content"]),
                           reasoning_chars=len(response["reasoning"]))
                if response["usage"]["prompt_tokens"] != plan["prompt_tokens"]:
                    raise RuntimeError("Actual prompt count changed")
                if response["finish_reason"] == "stop":
                    code = extract_code("</think>" + response["content"])
                    Path(str(stem) + ".py").write_text(code + "\n")
                    row["execution"] = evaluate(code, runtime["image_id"])
                    row["status"] = ("correct_completed" if row["execution"]["exit_code"] == 0
                                     else "wrong_completed")
                else:
                    row["status"] = "incomplete"
            except Exception as exc:
                row["error"] = repr(exc)
            after_file = Path(str(stem) + ".after.prom")
            after = runtime_metrics(base, after_file)
            after_counters = metric_counters(after_file)
            row["cache_hits"] = (after["prefix_cache_hits_total"] - before["prefix_cache_hits_total"]
                                 if None not in (after["prefix_cache_hits_total"], before["prefix_cache_hits_total"])
                                 else None)
            row["counter_deltas"] = {
                k: (after_counters[k] - before_counters[k]
                    if None not in (after_counters[k], before_counters[k]) else None)
                for k in METRICS}
            if row["counter_deltas"]["num_preemptions_total"] not in (0, None):
                row.update(status="runtime_unqualified", error="GPU preemption")
            draft = row["counter_deltas"][METRICS[0]]
            accepted = row["counter_deltas"][METRICS[1]]
            row["acceptance"] = accepted / draft if draft else None
            rows.append(row)
            with (leg / "attempts.jsonl").open("a") as file:
                file.write(json.dumps(row) + "\n")
            print(json.dumps(dict(index=index, status=row["status"],
                                  decode_tps=(row.get("timing") or {}).get("decode_tps"),
                                  ttft_ms=(row.get("timing") or {}).get("ttft_ms"),
                                  cache_hits=row["cache_hits"],
                                  acceptance=row["acceptance"])), flush=True)
    finally:
        done.set(); watcher.join()
        save(leg / "pcie-loaded.json", samples)
        hw.check_loaded(samples, idle, expected)
    measured = [r for r in rows if not r["warmup"]]
    cache_hit_floor = math.ceil(plan["prompt_tokens"] * plan["min_cache_hit_fraction"])
    qualified = (len(measured) == 3
                 and all(r["status"] == "correct_completed" for r in measured)
                 and all((r["cache_hits"] or 0) >= cache_hit_floor for r in measured)
                 and all(r["counter_deltas"]["num_preemptions_total"] == 0
                         and r["counter_deltas"][METRICS[0]] is not None
                         and r["counter_deltas"][METRICS[1]] is not None
                         for r in measured))
    save(leg / "summary.json", dict(qualified=qualified, attempts=len(rows),
         measured=len(measured), cache_hit_floor=cache_hit_floor, correct_completed=sum(
             r["status"] == "correct_completed" for r in measured),
         decode_tps_median=(statistics.median(r["timing"]["decode_tps"] for r in measured)
                            if qualified else None),
         ttft_ms_median=(statistics.median(r["timing"]["ttft_ms"] for r in measured)
                         if qualified else None),
         wall_seconds_median=(statistics.median(r["timing"]["wall_seconds"] for r in measured)
                              if qualified else None), all_attempts_retained=True))
    if not qualified:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--out", required=True, type=Path)
    prep.add_argument("--corpus", required=True, type=Path)
    prep.add_argument("--image", default="r9700/vllm:glm53-lmcache055-mtp5004")
    trial = subs.add_parser("run")
    trial.add_argument("--out", required=True, type=Path)
    trial.add_argument("--depth", required=True, type=int, choices=(3, 4, 5))
    trial.add_argument("--container", required=True)
    trial.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else run)(args)


if __name__ == "__main__":
    main()
