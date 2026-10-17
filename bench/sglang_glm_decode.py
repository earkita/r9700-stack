#!/usr/bin/env python3
"""Run a frozen, C1 BetterBench diagnostic against the experimental SGLang GLM.

The output is a standalone runtime observation. It is not a K4 comparison.
Create protocol.json and a frozen corpus in --out before invoking this script.
"""

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import uuid

from betterbench.client import stream_chat_sync


def post(url, body):
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def read(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def hardware_module():
    path = Path(__file__).parent / "baseline-earkita" / "hardware.py"
    spec = importlib.util.spec_from_file_location("earkita_hardware", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pci_loaded(out, mapping):
    hw = hardware_module()
    return [hw.capture(out / f"pcie-loaded-{i}", mapping) for i in range(3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    parser.add_argument("--model", default="glm-5.3-flash-sglang")
    args = parser.parse_args()
    out = args.out
    protocol = json.loads((out / "protocol.json").read_text())
    if (out / "requests.jsonl").exists():
        raise RuntimeError("Refusing to overwrite existing benchmark attempts")
    if importlib.metadata.version("betterbench") != protocol["client"]["version"]:
        raise RuntimeError("BetterBench version differs from frozen protocol")
    running_image = subprocess.check_output(
        ["docker", "inspect", "glm53-sglang-v0520", "--format", "{{.Image}}"], text=True
    ).strip()
    if running_image != protocol["image_id"]:
        raise RuntimeError(f"Container image differs from protocol: {running_image}")

    case = json.loads((out / protocol["corpus"]["file"]).read_text())
    messages = case["messages"]
    options = protocol["workload"]
    tokenized = post(args.base + "/tokenize", dict(
        model=args.model, messages=messages, add_generation_prompt=True,
        chat_template_kwargs={"reasoning_effort": options["reasoning_effort"]},
    ))
    count = tokenized["count"]
    model = next(m for m in read(args.base + "/v1/models")["data"] if m["id"] == args.model)
    if count != protocol["corpus"]["expected_prompt_tokens"]:
        raise RuntimeError(f"Prompt tokens: {count}, expected {protocol['corpus']['expected_prompt_tokens']}")
    if count + options["output_tokens"] > model["max_model_len"]:
        raise RuntimeError("Prompt and output exceed runtime context limit")

    hw = hardware_module()
    mapping = hw.hip_mapping("glm53-sglang-v0520")
    idle = json.loads((out / "pcie-idle" / "links.json").read_text())
    if mapping != json.loads((out / "pcie-idle" / "mapping.json").read_text()):
        raise RuntimeError("HIP/BDF mapping changed since idle preflight")
    expected = json.loads((out / "pcie-expected.json").read_text())
    rows = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        for index in range(1 + options.get("repetitions", 3)):
            warmup = index == 0
            salt = "sglang-bench-" + uuid.uuid4().hex
            extra = dict(
                cache_salt=salt, ignore_eos=options["ignore_eos"],
                chat_template_kwargs={"reasoning_effort": options["reasoning_effort"]},
                min_p=0.0, presence_penalty=0.0, frequency_penalty=0.0,
                repetition_penalty=1.0,
            )
            pci_future = pool.submit(pci_loaded, out, mapping) if warmup else None
            result = stream_chat_sync(
                args.base + "/v1", args.model, messages,
                max_tokens=options["output_tokens"],
                temperature=options["temperature"], top_p=options["top_p"],
                top_k=options["top_k"], seed=options["seed"],
                category="context_8k", prompt_id=f"context_8k-{index}",
                extra_body=extra, timeout=1800,
            )
            row = dict(index=index, warmup=warmup, prompt_sha256=protocol["corpus"]["sha256"],
                       request_options=dict(model=args.model, max_tokens=options["output_tokens"],
                                            temperature=options["temperature"], top_p=options["top_p"],
                                            top_k=options["top_k"], seed=options["seed"], **extra),
                       result=result.as_dict())
            with (out / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            print(json.dumps(dict(index=index, warmup=warmup, ok=result.ok,
                                  prompt_tokens=result.prompt_tokens,
                                  completion_tokens=result.completion_tokens,
                                  finish_reason=result.finish_reason,
                                  ttft_ms=result.ttft_ms,
                                  decode_tps=result.decode_tps, error=result.error)), flush=True)
            if pci_future is not None:
                hw.check_loaded(pci_future.result(), idle, expected)
                (out / "pcie-status.txt").write_text("PASS: all eight endpoints and upstream links checked during warmup\n")
            if not result.ok or result.prompt_tokens != count or \
                    result.completion_tokens != options["output_tokens"] or \
                    result.finish_reason != "length":
                raise RuntimeError(f"Attempt {index} failed frozen workload; raw result preserved")

    measured = [row["result"] for row in rows if not row["warmup"]]
    summary = dict(protocol="protocol.json", samples=len(measured),
                   comparable_to_K4=False, measured={})
    for field in ("decode_tps", "ttft_ms", "pp_tps", "wall_ms"):
        values = [row[field] for row in measured]
        summary["measured"][field] = dict(median=statistics.median(values),
                                           minimum=min(values), maximum=max(values),
                                           values=values)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
