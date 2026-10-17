#!/usr/bin/env python3
"""Standalone warm-prefix C1 diagnostic using BetterBench 0.6.0.

One cold population/warmup followed by five timed identical-prefix requests.
This is a separate protocol from the cold 8K A/B; never compare their TTFTs.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

from betterbench.client import stream_chat_sync
from sglang_glm_decode import hardware_module, post, read


def capture_loaded(out, mapping):
    hw = hardware_module()
    samples = []
    for index in range(3):
        time.sleep(1)
        samples.append(hw.capture(out / f"pcie-loaded-{index}", mapping))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    raw = (out / "protocol.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != (out / "protocol.sha256").read_text().strip():
        raise RuntimeError("Protocol hash changed")
    plan = json.loads(raw)
    if importlib.metadata.version("betterbench") != "0.6.0":
        raise RuntimeError("Require pinned BetterBench 0.6.0")
    if (out / "requests.jsonl").exists():
        raise RuntimeError("Refusing to overwrite attempts")
    corpus = out / plan["corpus"]["file"]
    if hashlib.sha256(corpus.read_bytes()).hexdigest() != plan["corpus"]["sha256"]:
        raise RuntimeError("Corpus hash changed")
    image = subprocess.check_output(
        ["docker", "inspect", plan["container"], "--format", "{{.Image}}"], text=True).strip()
    if image != plan["image_id"]:
        raise RuntimeError("Runtime image changed")
    case = json.loads(corpus.read_text())
    base, model = plan["base"], plan["model"]
    options = plan["workload"]
    count = post(base + "/tokenize", dict(model=model, messages=case["messages"],
                 add_generation_prompt=True,
                 chat_template_kwargs={"reasoning_effort": options["reasoning_effort"]}))["count"]
    limit = next(m["max_model_len"] for m in read(base + "/v1/models")["data"] if m["id"] == model)
    if count not in (8192, 32768, 130816, 261888):
        raise RuntimeError("Frozen C1 protocol requires exactly 8192/32768/130816/261888 input tokens")
    # Pinned SGLang: max_req_len=context_len-1, then scheduler subtracts one.
    if count != plan["corpus"]["expected_prompt_tokens"] or count + options["output_tokens"] + 2 > limit:
        raise RuntimeError("Tokenization/context differs from protocol")
    hw = hardware_module()
    mapping = hw.hip_mapping(plan["container"])
    idle = json.loads((out / "pcie-idle/links.json").read_text())
    if mapping != json.loads((out / "pcie-idle/mapping.json").read_text()):
        raise RuntimeError("Device mapping changed")
    expected = json.loads((out / "pcie-expected.json").read_text())
    rows = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        for index in range(6):
            start = datetime.datetime.now(datetime.timezone.utc).isoformat()
            future = pool.submit(capture_loaded, out, mapping) if index == 0 else None
            result = stream_chat_sync(base + "/v1", model, case["messages"],
                       max_tokens=options["output_tokens"], temperature=options["temperature"],
                       top_p=options["top_p"], top_k=-1, seed=options["seed"],
                       category="long_warm_prefix", prompt_id=str(index), timeout=1800,
                       extra_body=dict(cache_salt=plan["cache_salt"], ignore_eos=True,
                         min_p=0.0, presence_penalty=0.0, frequency_penalty=0.0,
                         repetition_penalty=1.0,
                         chat_template_kwargs={"reasoning_effort": options["reasoning_effort"]}))
            logs = subprocess.check_output(["docker", "logs", "--since", start,
                                           plan["container"]], stderr=subprocess.STDOUT, text=True)
            (out / f"request-{index}-server.log").write_text(logs)
            hits = [int(n) for n in re.findall(r"#cached-token: (\d+)", logs)]
            decode_lines = [line for line in logs.splitlines() if "Decode batch," in line]
            row = dict(index=index, warmup=index == 0, result=result.as_dict(),
                       cached_tokens_observed=max(hits) if hits else None,
                       graph_replay_observed=all("cuda graph: True" in line for line in decode_lines)
                                             if decode_lines else None)
            rows.append(row)
            with (out / "requests.jsonl").open("a") as file:
                file.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if future:
                hw.check_loaded(future.result(), idle, expected)
                (out / "pcie-status.txt").write_text("PASS eight-GPU idle/loaded preflight\n")
            if not result.ok or result.prompt_tokens != count or result.completion_tokens != options["output_tokens"] or result.finish_reason != "length":
                raise RuntimeError("Request failed frozen output budget; raw attempt retained")
            if row["cached_tokens_observed"] is None or (index == 0 and row["cached_tokens_observed"] != 0) or (index and row["cached_tokens_observed"] <= 0):
                raise RuntimeError("Declared cold/warm prefix state not established in per-request logs")
            if plan["runtime"]["graphs"] and row["graph_replay_observed"] is not True:
                raise RuntimeError("Graph replay not established in recorded decode batches")
    summary = dict(samples=5, scope="standalone C1 long-context warm-prefix diagnostic, not a cold/K4 A/B", measured={})
    for field in ("decode_tps", "ttft_ms", "wall_ms"):
        values = [row["result"][field] for row in rows[1:]]
        summary["measured"][field] = dict(median=statistics.median(values), minimum=min(values),
                                          maximum=max(values), values=values)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
