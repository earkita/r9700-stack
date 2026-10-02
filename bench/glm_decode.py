#!/usr/bin/env python3
"""Bounded C1 decode check using the unmodified BetterBench streaming client.

Fresh cache salt per request: APC stays enabled but every sample is cold.
This fixed-length throughput workload is separate from answer-quality tests.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import time
import uuid

from betterbench.client import stream_chat_sync
from glm_quality import metrics, read_url


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://localhost:8080")
    p.add_argument("--model", default="glm-5.3-flash")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument("--spec-metrics", action="store_true", help="Save per-request speculative acceptance by position")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    version = importlib.metadata.version("betterbench")
    assert version == "0.6.0", version
    hardware = json.loads(subprocess.check_output(
        ["amd-smi", "static", "--gpu", *map(str, range(8)), "--asic", "--limit", "--json"]))
    assert len(hardware["gpu_data"]) == 8
    for gpu in hardware["gpu_data"]:
        assert gpu["asic"]["target_graphics_version"] == "gfx1201"
        assert float(gpu["limit"]["ppt0"]["socket_power_limit"]["value"]) == 225
    (args.out / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
    rows = []
    for name, target in [("context_8k", 8192), ("context_32k", 32768)]:
        messages = json.loads((args.corpus / (name + ".jsonl")).read_text())["messages"]
        with read_url(args.base, "/tokenize", dict(model=args.model, messages=messages,
                      add_generation_prompt=True, chat_template_kwargs={"reasoning_effort": "high"})) as r:
            count = json.load(r)["count"]
        assert count == target, (name, count, target)
        for i in range(args.passes + 1):
            before = metrics(args.base)
            assert not before["num_requests_running"] and not before["num_requests_waiting"]
            if args.spec_metrics:
                with read_url(args.base, "/metrics") as r:
                    spec_before = r.read().decode()
            extra = dict(cache_salt="glm-decode-" + uuid.uuid4().hex, ignore_eos=True,
                         min_p=0.0, presence_penalty=0.0, frequency_penalty=0.0,
                         repetition_penalty=1.0, chat_template_kwargs={"reasoning_effort": "high"})
            result = stream_chat_sync(args.base + "/v1", args.model, messages,
                max_tokens=args.output_tokens, temperature=1.0, top_p=.95, top_k=-1,
                seed=1234, category=name, prompt_id=f"{name}-{i}", extra_body=extra)
            for _ in range(120):
                after = metrics(args.base)
                if after["prefix_cache_queries_total"] - before["prefix_cache_queries_total"] >= count:
                    break
                time.sleep(.1)
            row = dict(label=args.label, category=name, warmup=i == 0,
                       sampling=dict(temperature=1.0, top_p=.95, top_k=-1, seed=1234),
                       request_extra=extra, messages=messages, result=result.as_dict(),
                       metrics_before=before, metrics_after=after,
                       cached_tokens=after["prefix_cache_hits_total"] - before["prefix_cache_hits_total"])
            if args.spec_metrics:
                from glm_spec_metrics import summarize
                with read_url(args.base, "/metrics") as r:
                    spec_after = r.read().decode()
                row["speculative"] = summarize(spec_before, spec_after)
            with (args.out / "requests.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            assert result.ok and result.prompt_tokens == target, result.as_dict()
            assert result.completion_tokens == args.output_tokens and result.finish_reason == "length"
            assert row["cached_tokens"] == 0
            assert abs(result.decode_tps - (args.output_tokens - 1) / (sum(result.update_gaps_ms) / 1000)) < 1e-8
            rows.append(row)
            print(json.dumps(dict(category=name, warmup=i == 0, decode_tps=result.decode_tps,
                                  ttft_ms=result.ttft_ms, chunking=result.chunking)), flush=True)
    summary = dict(label=args.label, betterbench=version, concurrency=1,
                   formula="(completion_tokens - 1) / (last content-bearing SSE - first content-bearing SSE)",
                   limits="Native BetterBench convention; for batched speculation first burst may contain multiple tokens. No per-token ITL claim for batched output. Three samples do not qualify tail latency.",
                   groups={})
    for name in ("context_8k", "context_32k"):
        group = [r["result"] for r in rows if r["category"] == name and not r["warmup"]]
        summary["groups"][name] = {field: dict(mean=statistics.mean(r[field] for r in group),
                    median=statistics.median(r[field] for r in group), min=min(r[field] for r in group),
                    max=max(r[field] for r in group)) for field in ("decode_tps", "ttft_ms", "pp_tps")}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
