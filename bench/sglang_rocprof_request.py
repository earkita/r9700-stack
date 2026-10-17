#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Frozen prime/profiled request for the isolated SGLang ROCprofv3 trace."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import urllib.request

from betterbench.client import stream_chat_sync


def hardware():
    path = Path(__file__).parent / "baseline-earkita" / "hardware.py"
    spec = importlib.util.spec_from_file_location("earkita_hardware", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def loaded_samples(hw, out, mapping):
    result = []
    for index in range(3):
        result.append(hw.capture(out / f"pcie-loaded-{index}", mapping))
        time.sleep(2)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase", choices=("prime", "profiled"), required=True)
    args = p.parse_args()
    out = args.out
    raw = (out / "protocol.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != (out / "protocol.sha256").read_text().strip():
        raise RuntimeError("Protocol hash differs")
    protocol = json.loads(raw)
    if (out / f"{args.phase}.json").exists():
        raise RuntimeError("Refusing to overwrite request attempt")
    image = subprocess.check_output(
        ["docker", "inspect", "glm53-sglang-rocprof-20261015", "--format", "{{.Image}}"],
        text=True).strip()
    if image != protocol["image_id"]:
        raise RuntimeError("S0 image differs from frozen protocol")
    corpus = (out / protocol["prompt_file"]).read_bytes()
    if hashlib.sha256(corpus).hexdigest() != protocol["prompt_sha256"]:
        raise RuntimeError("Corpus hash differs")
    messages = json.loads(corpus)["messages"]
    body = dict(model=protocol["model"], messages=messages,
                add_generation_prompt=True,
                chat_template_kwargs={"reasoning_effort": protocol["reasoning_effort"]})
    request = urllib.request.Request("http://127.0.0.1:8081/tokenize",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        count = json.load(response)["count"]
    if count != protocol["expected_prompt_tokens"]:
        raise RuntimeError(f"Expected 8192 prompt tokens, got {count}")
    hw = hardware()
    mapping = hw.hip_mapping("glm53-sglang-rocprof-20261015")
    idle = json.loads((out / "pcie-idle/links.json").read_text())
    if mapping != json.loads((out / "pcie-idle/mapping.json").read_text()):
        raise RuntimeError("HIP/BDF mapping changed")
    expected = json.loads((out / "pcie-expected.json").read_text())
    token_budget = protocol["prime_output_tokens" if args.phase == "prime"
                            else "profiled_output_tokens"]
    extra = dict(cache_salt=protocol["cache_salt"], ignore_eos=True,
                 chat_template_kwargs={"reasoning_effort": protocol["reasoning_effort"]},
                 min_p=0.0, presence_penalty=0.0, frequency_penalty=0.0,
                 repetition_penalty=1.0)
    with ThreadPoolExecutor(max_workers=1) as pool:
        loaded = pool.submit(loaded_samples, hw, out, mapping) if args.phase == "prime" else None
        result = stream_chat_sync(
            "http://127.0.0.1:8081/v1", protocol["model"], messages,
            max_tokens=token_budget, temperature=protocol["temperature"],
            top_p=protocol["top_p"], top_k=protocol["top_k"], seed=protocol["seed"],
            category=f"rocprof_{args.phase}", prompt_id=f"rocprof_{args.phase}",
            extra_body=extra, timeout=1800)
        (out / f"{args.phase}.json").write_text(json.dumps(result.as_dict(), indent=2) + "\n")
        if loaded is not None:
            hw.check_loaded(loaded.result(), idle, expected)
            (out / "pcie-status.txt").write_text("PASS: all TP8 links verified under prime load\n")
    if not result.ok or result.prompt_tokens != count or result.completion_tokens != token_budget:
        raise RuntimeError(f"{args.phase} request failed; raw result retained")
    print(json.dumps(dict(phase=args.phase, prompt_tokens=result.prompt_tokens,
                          completion_tokens=result.completion_tokens,
                          ttft_ms=result.ttft_ms, decode_tps=result.decode_tps)))


if __name__ == "__main__":
    main()
