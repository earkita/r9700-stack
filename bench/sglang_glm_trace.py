#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded C1 8K SGLang CPU+GPU diagnostic trace; never a speed comparison."""

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


def api(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET")
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read().decode()


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    args = parser.parse_args()
    out = args.out
    protocol_raw = (out / "protocol.json").read_bytes()
    if hashlib.sha256(protocol_raw).hexdigest() != (out / "protocol.sha256").read_text().strip():
        raise RuntimeError("Trace protocol hash differs")
    protocol = json.loads(protocol_raw)
    for name in ("prime.json", "profiled.json", "pcie-loaded-0"):
        if (out / name).exists():
            raise RuntimeError(f"Refusing to overwrite {name}")
    running_image = subprocess.check_output(
        ["docker", "inspect", "glm53-sglang-v0520", "--format", "{{.Image}}"],
        text=True).strip()
    if running_image != protocol["image_id"]:
        raise RuntimeError("Container image differs from frozen trace protocol")
    case = json.loads((out / protocol["corpus"]["file"]).read_text())
    messages = case["messages"]
    model = protocol["model"]
    opts = protocol["workload"]
    tokenized = json.loads(api(args.base + "/tokenize", dict(
        model=model, messages=messages, add_generation_prompt=True,
        chat_template_kwargs={"reasoning_effort": opts["reasoning_effort"]})))
    if tokenized["count"] != protocol["corpus"]["expected_prompt_tokens"]:
        raise RuntimeError("Prompt tokenization differs from frozen protocol")
    hw = hardware()
    mapping = hw.hip_mapping("glm53-sglang-v0520")
    idle_mapping = json.loads((out / "pcie-idle" / "mapping.json").read_text())
    if mapping != idle_mapping:
        raise RuntimeError("HIP/BDF mapping changed since idle PCIe preflight")
    idle = json.loads((out / "pcie-idle" / "links.json").read_text())
    expected = json.loads((out / "pcie-expected.json").read_text())
    extra = dict(cache_salt=opts["cache_salt"], ignore_eos=True,
                 chat_template_kwargs={"reasoning_effort": opts["reasoning_effort"]},
                 min_p=0.0, presence_penalty=0.0, frequency_penalty=0.0,
                 repetition_penalty=1.0)
    with ThreadPoolExecutor(max_workers=1) as pool:
        samples_future = pool.submit(loaded_samples, hw, out, mapping)
        prime = stream_chat_sync(
            args.base + "/v1", model, messages,
            max_tokens=opts["prime_output_tokens"], temperature=opts["temperature"],
            top_p=opts["top_p"], top_k=opts["top_k"], seed=opts["seed"],
            category="trace_prime", prompt_id="trace_prime",
            extra_body=extra, timeout=1800)
        (out / "prime.json").write_text(json.dumps(prime.as_dict(), indent=2) + "\n")
        hw.check_loaded(samples_future.result(), idle, expected)
        (out / "pcie-status.txt").write_text("PASS: TP8 PCIe mapping, topology and loaded links\n")
    if not prime.ok or prime.prompt_tokens != tokenized["count"]:
        raise RuntimeError("Prime request failed; preserved raw response")
    profile = dict(output_dir=protocol["profile_output_dir"],
                   profile_id="glm-sglang-c1-8k-decode-20261015",
                   num_steps=opts["profile_steps"], activities=["CPU", "GPU"],
                   profile_by_stage=True, with_stack=False, record_shapes=False)
    api(args.base + "/start_profile", profile)
    try:
        profiled = stream_chat_sync(
            args.base + "/v1", model, messages,
            max_tokens=opts["profiled_output_tokens"], temperature=opts["temperature"],
            top_p=opts["top_p"], top_k=opts["top_k"], seed=opts["seed"],
            category="trace_profiled", prompt_id="trace_profiled",
            extra_body=extra, timeout=1800)
        (out / "profiled.json").write_text(json.dumps(profiled.as_dict(), indent=2) + "\n")
    except BaseException:
        # A short/failed request can leave the bounded profiler active.
        try:
            api(args.base + "/stop_profile")
        except Exception:
            pass
        raise
    # num_steps automatically ends each stage; stop_profile after that returns
    # HTTP 500 in SGLang v0.5.20 even though all trace files were saved.
    if not profiled.ok or profiled.prompt_tokens != tokenized["count"] or \
            profiled.completion_tokens != opts["profiled_output_tokens"]:
        raise RuntimeError("Profiled request failed frozen workload; raw response retained")
    print(json.dumps(dict(prime_ttft_ms=prime.ttft_ms,
                          profiled_ttft_ms=profiled.ttft_ms,
                          profiled_decode_tps=profiled.decode_tps,
                          trace_dir=protocol["profile_output_dir"])))


if __name__ == "__main__":
    main()
