#!/usr/bin/env python3
"""Bounded, explicitly separate prefill/cache trace on a diagnostic deployment.

Requires a frozen protocol and an image with opt-in GLM host annotations. Does
not launch/restart services. Prime long contexts outside the profiler; profile
only a bounded cold request or independent appended tail. Preserve all ranks,
requests and cache-source evidence. Profiled latency is never a speed score.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import threading
import urllib.request

from sglang_glm_prefill import metrics, request, require_completed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def post(base, path, body=None):
    req = urllib.request.Request(base+path, b"" if body is None else json.dumps(body).encode(),
                                 {"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read().decode()


def validate_cases(cases, context, maximum_profiled_new_tokens):
    if not 1 <= maximum_profiled_new_tokens <= 16384:
        raise ValueError("Keep the profiled request bounded to at most 16384 uncached tokens")
    for case in cases:
        base, tail = case["input_ids"], case.get("tail_ids", [])
        if not base or len(base)+len(tail)+1 > context:
            raise ValueError("Full prefix plus tail/output does not fit; never shorten the base")
        if not tail and len(base) > maximum_profiled_new_tokens:
            raise ValueError("Long cold request would make an unbounded trace")
        if tail and len(tail)+case["max_prefix_recompute_tokens"] > maximum_profiled_new_tokens:
            raise ValueError("Tail plus allowed prefix recomputation exceeds trace bound")


def profile_request(plan, label):
    # Shape tracing retains tensor references and can introduce extra copies.
    # This initial bounded diagnostic has no separate VRAM qualification for
    # that mode; preserve kernels/ranges/correlation without retaining shapes.
    options = plan.get("profile_options")
    if (not isinstance(options, dict) or set(options) != {"with_stack", "record_shapes"}
            or any(options[key] is not False for key in options)):
        raise ValueError("Declare shape/stack tracing off in the bounded profile protocol")
    return dict(output_dir=plan["trace_dir_container"], profile_id=label,
                activities=["CPU", "GPU"], with_stack=False, record_shapes=False,
                merge_profiles=False, profile_by_stage=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    out = p.parse_args().out
    if sha(out/"protocol.json") != (out/"protocol.sha256").read_text().strip():
        raise RuntimeError("Frozen protocol changed")
    plan = json.loads((out/"protocol.json").read_text())
    profile_request(plan, "preflight")
    for file, digest in plan["sources"].items():
        if sha(file) != digest:
            raise RuntimeError(f"Trace client changed: {file}")
    if (out/"attempts.jsonl").exists():
        raise RuntimeError("Refusing to overwrite trace attempts")
    corpus = out/plan["corpus"]["file"]
    if sha(corpus) != plan["corpus"]["sha256"]:
        raise RuntimeError("Frozen trace corpus changed")
    identity = json.loads(subprocess.check_output(["docker","inspect",plan["container"]],text=True))[0]
    if not identity["State"]["Running"] or identity["Image"] != plan["image_id"]:
        raise RuntimeError("Diagnostic image/runtime changed")
    if "R9K_SGLANG_PROFILE_RANGES=1" not in identity["Config"]["Env"]:
        raise RuntimeError("Scoped diagnostic ranges must be explicitly enabled")
    if not any(m["Destination"] == plan["trace_dir_container"] and m.get("RW")
               and Path(m["Source"]).resolve() == (out/"traces").resolve()
               for m in identity["Mounts"]):
        raise RuntimeError("Missing writable, isolated trace output mount")
    with urllib.request.urlopen(plan["base"]+"/get_server_info",timeout=30) as f:
        info = json.load(f)
    runtime = info.get("server_args",info)
    for key,value in plan["expected_server_args"].items():
        if runtime.get(key) != value:
            raise RuntimeError(f"Diagnostic runtime differs: {key}")
    (out/"server-info.json").write_text(json.dumps(info,indent=2)+"\n")
    cases = json.loads(corpus.read_text())["cases"]
    validate_cases(cases,runtime["context_length"],plan["maximum_profiled_new_tokens"])
    spec = importlib.util.spec_from_file_location("hardware",Path(__file__).parent/"baseline-earkita/hardware.py")
    hw = importlib.util.module_from_spec(spec);spec.loader.exec_module(hw)
    mapping = hw.hip_mapping(plan["container"])
    idle = hw.capture(out/"pcie-before",mapping);expected = hw.expected_links(idle)
    done = threading.Event();samples = []
    def watch():
        while not done.is_set():
            samples.append(hw.links(mapping));done.wait(.5)
    watcher = threading.Thread(target=watch);watcher.start()
    try:
        for case in cases:
            label = case["id"]
            if not label.replace("-", "").replace("_", "").isalnum():
                raise ValueError("Trace case ID must be a safe filename component")
            base, tail = case["input_ids"], case.get("tail_ids", [])
            common = dict(stream=True,cache_salt=plan["cache_salt"]+"-"+label,
                          sampling_params=plan["sampling"]|dict(max_new_tokens=1,ignore_eos=False))
            (out/(label+"-flush-before.txt")).write_text(post(plan["base"],"/flush_cache?timeout=5"))
            prime,first,wall = request(plan["base"],common|dict(input_ids=base),out/(label+"-unprofiled-prime"))
            require_completed(prime["meta_info"])
            if prime["meta_info"]["prompt_tokens"] != len(base) or prime["meta_info"].get("cached_tokens") != 0:
                raise RuntimeError("Prime must be a genuine cold full-prefix request")
            if not tail:
                (out/(label+"-flush-cold.txt")).write_text(post(plan["base"],"/flush_cache?timeout=5"))
            profile = profile_request(plan, label)
            (out/(label+"-profile-request.json")).write_text(json.dumps(profile)+"\n")
            row = dict(case=label,status="error",scope="profiled diagnostic; not unprofiled performance")
            started = False
            try:
                # A start-response timeout does not prove that the workers did
                # not arm the profiler. Attempt cleanup after any start call.
                started=True
                (out/(label+"-profile-start.txt")).write_text(post(plan["base"],"/start_profile",profile))
                result,first,wall = request(plan["base"],common|dict(input_ids=base+tail),out/(label+"-profiled"))
                meta = result["meta_info"]
                row.update(meta_info=meta,profiled_observations=metrics(meta,first,len(base+tail),len(base) if tail else 0))
                cached = meta["cached_tokens"]
                if tail:
                    details = meta.get("cached_tokens_details") or {}
                    if cached < len(base)-case["max_prefix_recompute_tokens"] or details.get("device") != cached or details.get("host") != 0:
                        raise RuntimeError("Actual GPU prefix reuse not established")
                require_completed(meta)
                row["status"] = "pass"
            except BaseException as error:
                row["error"] = repr(error)
                raise
            finally:
                try:
                    if started:
                        (out/(label+"-profile-stop.txt")).write_text(post(plan["base"],"/stop_profile"))
                except BaseException as error:
                    row["status"] = "error"
                    row["stop_error"] = repr(error)
                    raise
                finally:
                    with (out/"attempts.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
            paths = sorted((out/"traces").glob(label+"-TP-*.trace.json.gz"))
            subprocess.run(["python3", "profiling/sglang_prefill_trace.py",*map(str,paths),"--out",str(out/(label+"-trace-report.json"))],check=True)
            print("PASS diagnostic trace",label,flush=True)
    finally:
        done.set();watcher.join();(out/"loaded-links.json").write_text(json.dumps(samples)+"\n")
        subprocess.run(["docker","logs",plan["container"]],stdout=(out/"server.log").open("w"),stderr=subprocess.STDOUT)
        hw.check_loaded(samples,idle,expected)


if __name__ == "__main__":
    main()
