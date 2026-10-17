#!/usr/bin/env python3
"""Frozen, natural-EOS GLM coding diagnostic with retained answers.

Prepare copies an existing long-context source archive. Run never starts or
restarts the model. Results from different legs are comparable only when the
protocol, source hashes, GPU topology and effective runtime match.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

from api_smoke import collect_stream
from sglang_glm_completed_code import evaluate, extract_code


ROOT = Path(__file__).resolve().parents[1]
HARDWARE = ROOT / "bench/baseline-earkita/hardware.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def post(base, route, body):
    request = urllib.request.Request(base + route, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def get(base, route):
    with urllib.request.urlopen(base + route, timeout=30) as response:
        return json.load(response)


def runtime_metrics(base, stem):
    with urllib.request.urlopen(base + "/metrics", timeout=30) as response:
        raw = response.read().decode()
    Path(stem).write_text(raw)
    result = {}
    for name in ("num_requests_running", "num_requests_waiting",
                 "prefix_cache_hits_total", "external_prefix_cache_hits_total",
                 "num_preemptions_total"):
        prefix = "vllm:" + name
        values = []
        for line in raw.splitlines():
            if line.startswith(prefix + "{") or line.startswith(prefix + " "):
                values.append(float(line.rsplit(" ", 1)[1]))
        result[name] = sum(values) if values else None
    return result


def hardware():
    spec = importlib.util.spec_from_file_location("earkita_hardware", HARDWARE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def container_info(name):
    raw = subprocess.check_output(["docker", "inspect", name], text=True)
    info = json.loads(raw)[0]
    if not info["State"]["Running"]:
        raise RuntimeError("Serving container is not running")
    return dict(name=name, image_id=info["Image"], args=info["Config"]["Cmd"],
                env={item.split("=", 1)[0]: item.split("=", 1)[1]
                     for item in info["Config"]["Env"] if "=" in item})


def prepare(args):
    out = args.out.resolve()
    out.relative_to(ROOT / "bench/results")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.corpus, out / "corpus.json")
    corpus = json.loads((out / "corpus.json").read_text())
    if corpus.get("task") != "dependency_waves" or len(corpus.get("messages", [])) != 2:
        raise RuntimeError("Require the frozen dependency_waves source archive")
    model_info = get(args.base, "/v1/models")
    model = next(item for item in model_info["data"] if item["id"] == args.model)
    count = post(args.base, "/tokenize", dict(model=args.model,
                 messages=corpus["messages"], add_generation_prompt=True,
                 chat_template_kwargs={"reasoning_effort": "high"}))["count"]
    if not 125000 <= count <= 135000 or count + args.max_tokens > model["max_model_len"]:
        raise RuntimeError(f"128K coding context/budget mismatch: {count}")
    runtime = container_info(args.container)
    if runtime["env"].get("R9K_GLM_DENSE_FP8", "0") != "0":
        raise RuntimeError("Prepare the K4 baseline with dense FP8 optimization off")
    plan = dict(schema=2, name="glm-k4-natural-eos-code-128k-v2",
                scope="focused coding diagnostic; not a full agent benchmark",
                corpus_sha256=sha(out / "corpus.json"),
                sources={str(path.relative_to(ROOT)): sha(path)
                         for path in (Path(__file__).resolve(), ROOT / "bench/api_smoke.py",
                                      ROOT / "bench/sglang_glm_completed_code.py", HARDWARE)},
                base=args.base, model=args.model, model_image_id=runtime["image_id"],
                input_tokens=count, max_model_len=model["max_model_len"],
                sampling=dict(temperature=1.0, top_p=0.95, top_k=-1,
                              min_p=0.0, presence_penalty=0.0,
                              frequency_penalty=0.0, repetition_penalty=1.0,
                              max_tokens=args.max_tokens, ignore_eos=False,
                              chat_template_kwargs={"reasoning_effort": "high"}),
                seed=1234, warmup=1, measured=3,
                cache_salt="glm-natural-code-" + hashlib.sha256(str(out).encode()).hexdigest()[:16],
                experiment="R9K_GLM_DENSE_FP8 0 vs 1; same K4 image/checkpoint/sampling; natural EOS and full code oracle",
                metric="(last continuous completion count - first count)/(last text update - first text update); TTFT separate")
    save(out / "protocol.json", plan)
    (out / "protocol.sha256").write_text(sha(out / "protocol.json") + "\n")
    save(out / "prepared-runtime.json", runtime)
    print(json.dumps(dict(out=str(out), input_tokens=count, image=runtime["image_id"])), flush=True)


def decode_observation(events, usage):
    """Only actual cumulative counters between text-bearing stream updates."""
    counts = []
    for elapsed, count in events:
        if type(count) is not int or count < 0:
            raise ValueError("Missing/invalid continuous completion counter")
        if counts and count < counts[-1][1]:
            raise ValueError("Continuous completion counter decreased")
        if count and (not counts or count > counts[-1][1]):
            counts.append((elapsed, count))
    if len(counts) < 2 or any(b[1] <= a[1] or b[0] <= a[0]
                              for a, b in zip(counts, counts[1:])):
        raise ValueError("Missing/inconsistent continuous completion counters")
    reported = usage.get("completion_tokens")
    if type(reported) is not int or reported < counts[-1][1] or reported - counts[-1][1] > 2:
        raise ValueError("Final usage differs from streamed counters")
    elapsed = counts[-1][0] - counts[0][0]
    return dict(first_update_ms=counts[0][0] * 1000,
                first_update_tokens=counts[0][1],
                last_update_tokens=counts[-1][1],
                final_usage_minus_last_update=reported - counts[-1][1],
                decode_seconds=elapsed,
                decode_tps=(counts[-1][1] - counts[0][1]) / elapsed,
                update_gaps_ms=[(b[0] - a[0]) * 1000 for a, b in zip(counts, counts[1:])],
                scope="API cumulative usage; excludes entire first streamed burst and TTFT; batched chunks are not per-token ITL")


def stream(base, body, stem):
    save(str(stem) + ".request.json", body)
    start = time.perf_counter()
    events = []
    request = urllib.request.Request(base + "/v1/chat/completions",
                                     json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response, \
            Path(str(stem) + ".sse.jsonl").open("w") as raw:
        def lines():
            for line in response:
                elapsed = time.perf_counter() - start
                text = line.decode(errors="replace")
                raw.write(json.dumps(dict(seconds=elapsed, line=text)) + "\n")
                if text.startswith("data:") and text[5:].strip() != "[DONE]":
                    event = json.loads(text[5:])
                    choices = event.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                            usage = event.get("usage") or {}
                            events.append((elapsed, usage.get("completion_tokens")))
                yield line
        result = collect_stream(lines())
    wall = time.perf_counter() - start
    save(str(stem) + ".response.json", result)
    return result, dict(wall_seconds=wall,
                        ttft_ms=events[0][0] * 1000 if events else None,
                        **decode_observation(events, result.get("usage") or {}))


def run(args):
    out = args.out.resolve()
    plan_path = out / "protocol.json"
    if sha(plan_path) != (out / "protocol.sha256").read_text().strip():
        raise RuntimeError("Frozen protocol changed")
    plan = json.loads(plan_path.read_text())
    if sha(out / "corpus.json") != plan["corpus_sha256"]:
        raise RuntimeError("Frozen corpus changed")
    for relpath, digest in plan["sources"].items():
        if sha(ROOT / relpath) != digest:
            raise RuntimeError(f"Runner/evaluator changed: {relpath}")
    leg = out / args.leg
    leg.mkdir(parents=True, exist_ok=False)
    runtime = container_info(args.container)
    if runtime["image_id"] != plan["model_image_id"]:
        raise RuntimeError("Serving image differs from frozen protocol")
    if runtime["env"].get("R9K_GLM_DENSE_FP8", "0") != args.dense:
        raise RuntimeError("Dense FP8 effective setting differs from declared leg")
    save(leg / "runtime.json", runtime)
    corpus = json.loads((out / "corpus.json").read_text())
    current = post(plan["base"], "/tokenize", dict(model=plan["model"],
                   messages=corpus["messages"], add_generation_prompt=True,
                   chat_template_kwargs={"reasoning_effort": "high"}))["count"]
    if current != plan["input_tokens"]:
        raise RuntimeError("Actual prompt count changed")
    hw = hardware()
    mapping = hw.hip_mapping(args.container)
    idle = hw.capture(leg / "pcie-before", mapping)
    expected = hw.expected_links(idle)
    save(leg / "pcie-expected.json", expected)
    samples, done = [], threading.Event()
    def watch():
        while not done.is_set():
            samples.append(hw.links(mapping))
            done.wait(1)
    rows = []
    thread = threading.Thread(target=watch)
    thread.start()
    try:
        for index in range(plan["warmup"] + plan["measured"]):
            before = runtime_metrics(plan["base"], str(stem := leg / f"attempt-{index:02d}") + ".before.prom")
            if before["num_requests_running"] != 0 or before["num_requests_waiting"] != 0:
                raise RuntimeError("Endpoint is not idle before the isolated request")
            body = dict(model=plan["model"], messages=corpus["messages"],
                        **plan["sampling"], seed=plan["seed"],
                        cache_salt=plan["cache_salt"] + "-" + args.leg,
                        stream=True, stream_options={"include_usage": True,
                                                     "continuous_usage_stats": True})
            row = dict(index=index, warmup=index < plan["warmup"], status="error", metrics_before=before)
            try:
                response, timing = stream(plan["base"], body, stem)
                row.update(timing=timing, usage=response["usage"],
                           finish_reason=response["finish_reason"],
                           content_chars=len(response["content"]),
                           reasoning_chars=len(response["reasoning"]))
                if response["usage"]["prompt_tokens"] != plan["input_tokens"]:
                    raise RuntimeError("Actual prompt count differs from protocol")
                if response["finish_reason"] != "stop":
                    row["status"] = "incomplete"
                else:
                    try:
                        code = extract_code("</think>" + response["content"])
                        Path(str(stem) + ".py").write_text(code + "\n")
                        row["execution"] = evaluate(code, runtime["image_id"])
                        row["status"] = ("correct_completed" if row["execution"]["exit_code"] == 0
                                         else "wrong_completed")
                    except (ValueError, SyntaxError) as error:
                        row.update(status="wrong_completed", format_error=repr(error))
            except Exception as error:
                row["error"] = repr(error)
            try:
                after = runtime_metrics(plan["base"], str(stem) + ".after.prom")
                row["metrics_after"] = after
                row["prefix_hits_delta"] = (after["prefix_cache_hits_total"] - before["prefix_cache_hits_total"]
                                            if None not in (after["prefix_cache_hits_total"], before["prefix_cache_hits_total"])
                                            else None)
                if not row["warmup"] and row["prefix_hits_delta"] < plan["input_tokens"] - 2048:
                    row.update(status="cache_unqualified", error="Measured request lacks full GPU prefix reuse")
                if (None not in (after["num_preemptions_total"], before["num_preemptions_total"])
                        and after["num_preemptions_total"] > before["num_preemptions_total"]):
                    row.update(status="runtime_unqualified", error="GPU preemption during request")
            except Exception as error:
                row.update(status="runtime_unqualified", error=f"Metrics observation failed: {error!r}")
            rows.append(row)
            with (leg / "attempts.jsonl").open("a") as file:
                file.write(json.dumps(row) + "\n")
            timing = row.get("timing") or {}
            print(json.dumps(dict(index=index, status=row["status"],
                                  ttft_ms=timing.get("ttft_ms"),
                                  decode_tps=timing.get("decode_tps"),
                                  wall_seconds=timing.get("wall_seconds"),
                                  prefix_hits_delta=row.get("prefix_hits_delta"),
                                  error=row.get("error"))), flush=True)
    finally:
        done.set()
        thread.join()
        save(leg / "pcie-loaded.json", samples)
        hw.check_loaded(samples, idle, expected)
    measured = [r for r in rows if not r["warmup"]]
    valid = [r for r in measured if r["status"] == "correct_completed"]
    summary = dict(scope=plan["scope"], attempts=len(rows),
                   measured=len(measured), correct_completed=len(valid),
                   decode_tps_median=statistics.median(r["timing"]["decode_tps"] for r in valid) if valid else None,
                   ttft_ms_median=statistics.median(r["timing"]["ttft_ms"] for r in valid) if valid else None,
                   wall_seconds_median=statistics.median(r["timing"]["wall_seconds"] for r in valid) if valid else None,
                   all_attempts_retained=True, comparable_performance=False)
    save(leg / "summary.json", summary)
    if len(valid) != len(measured):
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
    trial.add_argument("--dense", choices=["0", "1"], required=True)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else run)(args)


if __name__ == "__main__":
    main()
