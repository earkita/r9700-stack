#!/usr/bin/env python3
"""Frozen, bounded SGLang quality diagnostics with retained transcripts.

Separate from BetterBench throughput. Never starts or changes the runtime.
Prepare protocol.json, its SHA256 and a JSON-list corpus before invocation.
The caller supplies the mandatory PCIe idle and loaded evidence.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from glm_quality import classify, niah_scores, stream, token_counts, read_url


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    args = parser.parse_args()
    out = args.out
    if digest(out / "protocol.json") != (out / "protocol.sha256").read_text().strip():
        raise RuntimeError("Frozen protocol hash changed")
    plan = json.loads((out / "protocol.json").read_text())
    if digest(Path(__file__)) != plan["runner_sha256"]:
        raise RuntimeError("Quality client differs from frozen protocol")
    corpus = out / plan["corpus"]["file"]
    if digest(corpus) != plan["corpus"]["sha256"]:
        raise RuntimeError("Frozen quality corpus changed")
    if (out / "requests.jsonl").exists():
        raise RuntimeError("Refusing to overwrite previous attempts")
    image = subprocess.check_output(
        ["docker", "inspect", plan["container"], "--format", "{{.Image}}"],
        text=True).strip()
    if image != plan["image_id"]:
        raise RuntimeError("Running image differs from protocol")
    environment = json.loads(subprocess.check_output(
        ["docker", "inspect", plan["container"], "--format", "{{json .Config.Env}}"],
        text=True))
    environment = dict(item.split("=", 1) for item in environment)
    if environment.get("R9K_SGLANG_MHC_BACKEND") != plan["backend"]:
        raise RuntimeError("Observed mHC backend differs from protocol")
    with read_url(args.base, "/get_server_info") as response:
        info = json.load(response)
    (out / "server-info.json").write_text(json.dumps(info, indent=2) + "\n")
    runtime = info.get("server_args", info)
    for field, expected in plan.get("expected_server_args", {}).items():
        if runtime.get(field) != expected:
            raise RuntimeError(f"Observed runtime field differs from protocol: {field}")
    graph = runtime.get("cuda_graph_config", {}).get("decode", {}).get("backend")
    if graph != ("full" if plan["graphs"] else "disabled"):
        raise RuntimeError("Observed graph mode differs from protocol")
    if runtime.get("context_length") != plan["context_length"]:
        raise RuntimeError("Observed context limit differs from protocol")
    if "mamba_cache_size" in plan and runtime.get("max_mamba_cache_size") != plan["mamba_cache_size"]:
        raise RuntimeError("Observed Mamba state-slot budget differs from protocol")
    with read_url(args.base, "/v1/models") as response:
        model = next(m for m in json.load(response)["data"] if m["id"] == plan["model"])
    rows = []
    for case in json.loads(corpus.read_text()):
        body = dict(model=plan["model"], messages=case["messages"],
                    max_tokens=case["max_tokens"], temperature=plan["temperature"],
                    top_p=plan["top_p"], top_k=-1, seed=plan["seed"],
                    ignore_eos=False, stream=True,
                    stream_options={"include_usage": True},
                    cache_salt=case["cache_salt"],
                    chat_template_kwargs={"reasoning_effort": plan["reasoning_effort"]})
        row = dict(id=case["id"], request=body, expected=case["expected"],
                   status="error", response=None, timing=None, error=None)
        try:
            with read_url(args.base, "/tokenize", dict(
                    model=plan["model"], messages=case["messages"],
                    add_generation_prompt=True,
                    chat_template_kwargs=body["chat_template_kwargs"])) as response:
                count = json.load(response)["count"]
            row["tokenized_prompt_tokens"] = count
            if count != case["input_tokens"] or count + case["max_tokens"] + 2 > model["max_model_len"]:
                raise RuntimeError("Tokenization or context budget differs from frozen case")
            result, timing = stream(args.base, body, plan["timeout_seconds"])
            row.update(response=result, timing=timing, status=classify(result, case),
                       tokens=token_counts(result.get("usage") or {}))
            usage = result.get("usage") or {}
            if row["tokens"]["reasoning"] is None and usage.get("reasoning_tokens") is not None:
                row["tokens"].update(reasoning=usage["reasoning_tokens"],
                    non_reasoning_by_subtraction=usage["completion_tokens"] - usage["reasoning_tokens"],
                    reasoning_source="SGLang usage.reasoning_tokens")
            if row["tokens"]["prompt"] != count:
                raise RuntimeError("API usage differs from tokenized prompt")
        except Exception as error:
            row.update(status="error", error=repr(error))
        if case.get("task") == "niah":
            row["niah"] = niah_scores(row["response"], case,
                                     "UNKNOWN" if row["error"] else None)
        rows.append(row)
        with (out / "requests.jsonl").open("a") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({k: row[k] for k in ("id", "status", "error")}), flush=True)
    summary = dict(attempts=len(rows),
                   counts={name: sum(r["status"] == name for r in rows) for name in
                           ("correct_completed", "wrong_completed", "incomplete", "error")},
                   scope="bounded diagnostics only, not broad quality equivalence")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if any(row["status"] != "correct_completed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
