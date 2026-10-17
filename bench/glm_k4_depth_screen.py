#!/usr/bin/env python3
"""Run DFlash K4 against the frozen vLLM MTP3/4/5 C1/8K coding screen.

This does not manage the server. It requires a dedicated K4 runtime with the
same image, C1, GPU APC, context limit, checkpoint and KV geometry as MTP.
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

from glm_completed_code import ROOT, container_info, get, hardware, post, runtime_metrics, save, sha, stream
from glm_mtp_depth_screen import METRICS, SOURCES, metric_counters, option
from sglang_glm_completed_code import evaluate, extract_code


THIS_SOURCE = Path(__file__).resolve()
K4_PROFILE = ROOT / "serve/baseline-earkita/glm/glm-5.3-flash-c4-lmcache.sh"


def prepare(args):
    source = args.reference.resolve()
    original = json.loads((source / "protocol.json").read_text())
    if sha(source / "protocol.json") != (source / "protocol.sha256").read_text().strip():
        raise RuntimeError("Reference MTP protocol changed")
    if original["name"] != "glm-vllm-mtp345-c1-8k-natural-eos-v1":
        raise RuntimeError("Unexpected MTP protocol")
    if sha(source / "corpus.json") != original["corpus_sha256"]:
        raise RuntimeError("Reference corpus changed")
    out = args.out.resolve()
    out.relative_to(ROOT / "bench/results")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source / "corpus.json", out / "corpus.json")
    protocol = dict(original)
    protocol.update(name="glm-vllm-dflash-k4-c1-8k-natural-eos-v1",
                    reference_protocol_sha256=sha(source / "protocol.json"),
                    reference_path=str(source.relative_to(ROOT)),
                    sources={str(p.relative_to(ROOT)): sha(p)
                             for p in (*SOURCES, THIS_SOURCE, K4_PROFILE)},
                    speculative_config="DFlash K4; graph size 5",
                    comparison_scope="Matched C1/8K request/image/cache geometry; K4 versus MTP runtime implementation; not 128K/C4 or LMCache")
    save(out / "protocol.json", protocol)
    (out / "protocol.sha256").write_text(sha(out / "protocol.json") + "\n")
    print(json.dumps(dict(out=str(out), reference=protocol["reference_path"],
                          image_id=protocol["image_id"])), flush=True)


def run(args):
    out = args.out.resolve()
    if sha(out / "protocol.json") != (out / "protocol.sha256").read_text().strip():
        raise RuntimeError("Frozen K4 protocol changed")
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
            or spec.get("method") != "dflash"
            or spec.get("num_speculative_tokens") != 4
            or option(cmd, "--max-num-seqs") != "1"
            or option(cmd, "--max-model-len") != str(plan["max_model_len"])
            or option(cmd, "--port") != str(args.port)
            or option(cmd, "--max-num-batched-tokens") != "2048"
            or option(cmd, "--kv-cache-memory") != "4429185024"
            or option(cmd, "--kv-cache-dtype") != "fp8_e4m3"
            or "--kv-transfer-config" in cmd
            or runtime["env"].get("R9K_GLM_DFLASH") != "1"):
        raise RuntimeError("Runtime differs from frozen GPU-APC K4 screen")
    graph = json.loads(option(cmd, "--compilation-config"))
    if (graph.get("cudagraph_mode") != "FULL_DECODE_ONLY"
            or graph.get("cudagraph_capture_sizes") != [5]):
        raise RuntimeError("K4 decode graph differs from frozen screen")
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

    leg = out / "k4"
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
        done.set()
        watcher.join()
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
         measured=len(measured), cache_hit_floor=cache_hit_floor,
         correct_completed=sum(r["status"] == "correct_completed" for r in measured),
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
    prep.add_argument("--reference", required=True, type=Path)
    trial = subs.add_parser("run")
    trial.add_argument("--out", required=True, type=Path)
    trial.add_argument("--container", required=True)
    trial.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else run)(args)


if __name__ == "__main__":
    main()
