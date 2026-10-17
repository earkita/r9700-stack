#!/usr/bin/env python3
"""Bounded C1 32K/128K SGLang diagnostic; one attempt each, no A/B claim."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import uuid

from betterbench.client import stream_chat_sync


def post(url, body):
    request = urllib.request.Request(url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def hardware():
    spec = importlib.util.spec_from_file_location(
        "hardware", Path(__file__).parent / "baseline-earkita" / "hardware.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def loaded(hw, out, mapping):
    values = []
    for index in range(4):
        values.append(hw.capture(out / f"pcie-loaded-{index}", mapping))
        time.sleep(2)
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--base", default="http://127.0.0.1:8081")
    args = p.parse_args()
    out = args.out
    raw = (out / "protocol.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != (out / "protocol.sha256").read_text().strip():
        raise RuntimeError("Long diagnostic protocol hash changed")
    spec = json.loads(raw)
    if (out / "requests.jsonl").exists():
        raise RuntimeError("Refusing to overwrite attempts")
    image = subprocess.check_output(
        ["docker", "inspect", spec["container"], "--format", "{{.Image}}"],
        text=True).strip()
    if image != spec["image_id"]:
        raise RuntimeError("Running image differs from frozen protocol")
    models = urllib.request.urlopen(args.base + "/v1/models", timeout=30)
    with models:
        model = next(x for x in json.load(models)["data"] if x["id"] == spec["model"])
    hw = hardware()
    mapping = hw.hip_mapping(spec["container"])
    idle = json.loads((out / "pcie-idle/links.json").read_text())
    expected = json.loads((out / "pcie-expected.json").read_text())
    if mapping != json.loads((out / "pcie-idle/mapping.json").read_text()):
        raise RuntimeError("HIP mapping changed after idle preflight")

    with ThreadPoolExecutor(max_workers=1) as pool:
        for index, case in enumerate(spec["cases"]):
            messages = json.loads((out / case["file"]).read_text())["messages"]
            count = post(args.base + "/tokenize", dict(
                model=spec["model"], messages=messages, add_generation_prompt=True,
                chat_template_kwargs={"reasoning_effort": spec["reasoning_effort"]}))['count']
            if count != case["input_tokens"] or count + case["output_tokens"] > model["max_model_len"]:
                raise RuntimeError(f"Case {case['name']} tokenization/context mismatch: {count}")
            extra = dict(cache_salt="sglang-long-" + uuid.uuid4().hex,
                         ignore_eos=True, min_p=0.0, presence_penalty=0.0,
                         frequency_penalty=0.0, repetition_penalty=1.0,
                         chat_template_kwargs={"reasoning_effort": spec["reasoning_effort"]})
            future = pool.submit(loaded, hw, out, mapping) if index == 0 else None
            result = stream_chat_sync(
                args.base + "/v1", spec["model"], messages,
                max_tokens=case["output_tokens"], temperature=spec["temperature"],
                top_p=spec["top_p"], top_k=spec["top_k"], seed=spec["seed"],
                category=case["name"], prompt_id=case["name"],
                extra_body=extra, timeout=1800)
            row = dict(case=case, request_options=extra, result=result.as_dict())
            with (out / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(dict(case=case["name"], ok=result.ok,
                                  prompt_tokens=result.prompt_tokens,
                                  completion_tokens=result.completion_tokens,
                                  ttft_ms=result.ttft_ms,
                                  decode_tps=result.decode_tps,
                                  error=result.error)), flush=True)
            if future:
                hw.check_loaded(future.result(), idle, expected)
                (out / "pcie-status.txt").write_text("PASS: eight GPUs loaded links\n")
            if (not result.ok or result.prompt_tokens != count or
                    result.completion_tokens != case["output_tokens"] or
                    result.finish_reason != "length"):
                raise RuntimeError(f"Case {case['name']} failed; raw attempt preserved")


if __name__ == "__main__":
    main()
