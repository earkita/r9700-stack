#!/usr/bin/env python3
"""Frozen C1 cold, GPU-prefix-hit and new-tail prefill measurements.

One output token isolates TTFT from sustained decode; this is a cache/prefill
mechanics test, not completed-answer quality. A corpus supplies exact native
input IDs and separate tail branches for each repetition. An appended request
must reuse the base prefix, never a previously computed tail. Scheduler timing
is reported separately from client TTFT and is not called pure GPU time.
No service launch, profiling or configuration changes occur here.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import urllib.request


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_evidence(path, value):
    """Publish a complete record before moving on; partial .tmp is not success."""
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("w") as file:
        file.write(json.dumps(value,ensure_ascii=False)+"\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)
    directory = os.open(path.parent,os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def watch_links(read_links, journal_path, done, samples, errors, boot_id):
    """Keep completed observations if the process dies before its finally block.

    fsync is an OS durability request, not a guarantee against faulty hardware
    or storage losing power. This journal alone never qualifies a whole series.
    """
    try:
        with journal_path.open("x") as journal:
            while not done.is_set():
                sample = read_links()
                record = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              monotonic_seconds=time.monotonic(),boot_id=boot_id,links=sample)
                journal.write(json.dumps(record)+"\n")
                journal.flush()
                os.fsync(journal.fileno())
                samples.append(sample)
                done.wait(.5)
    except BaseException as error:
        errors.append(repr(error))
        done.set()


def require_completed(meta):
    if (meta.get("completion_tokens") != 1 or meta.get("num_retractions") != 0
            or (meta.get("finish_reason") or {}).get("type") not in ("stop", "length")):
        raise RuntimeError("Unexpected completion budget, abort, retraction or missing runtime evidence")


def cache_evidence(meta, phase, base_tokens, plan):
    from sglang_cache_evidence import native_cache_source

    if phase == "cold":
        return native_cache_source(meta["cached_tokens"], meta.get("cached_tokens_details"),
                                   allow_miss=True)
    if meta["cached_tokens"] < base_tokens - plan["max_prefix_recompute_tokens"]:
        raise RuntimeError("Required base-prefix hit not established")
    # An identical replay can miss completely when its length equals one
    # scheduler chunk. This is measured recomputation, never a cached speed.
    evidence = native_cache_source(
        meta["cached_tokens"], meta.get("cached_tokens_details"),
        allow_miss=phase == "warm" and plan.get("allow_warm_cache_miss") is True)
    if plan.get("cache_tier") == "device" and evidence["cache_source"] not in ("device", "none"):
        raise RuntimeError("Required GPU-only cache source not established")
    return evidence


def metrics(meta, first_seconds, input_tokens, prefix_tokens):
    if meta["prompt_tokens"] != input_tokens or first_seconds <= 0:
        raise ValueError("Actual input count or TTFT invalid")
    cached = meta.get("cached_tokens")
    if not isinstance(cached, int) or not 0 <= cached <= input_tokens:
        raise ValueError("Cache evidence absent or inconsistent")
    if cached > prefix_tokens:
        raise ValueError("Previously computed tail was reused; not a new-tail measurement")
    # These scheduler wall timestamps include inter-chunk waiting and metadata;
    # they are not the sum of GPU kernel time or lookup/restore subcomponents.
    start, end = meta.get("forward_entry_time"), meta.get("prefill_finished_time")
    scheduler = end - start if start is not None and end is not None else None
    if scheduler is not None and scheduler < 0:
        raise ValueError("Invalid scheduler timestamps")
    return dict(ttft_seconds=first_seconds, cached_tokens=cached,
                cache_fraction=cached / input_tokens,
                uncached_tokens=input_tokens-cached,
                recomputed_prefix_tokens=prefix_tokens-cached,
                uncached_tokens_per_ttft=(input_tokens-cached)/first_seconds,
                cached_tokens_details=meta.get("cached_tokens_details"),
                scheduler_prefill_wall_seconds=scheduler,
                gpu_prefill_seconds=None, lookup_seconds=None,
                restore_seconds=None, kda_kpool_update_seconds=None,
                component_timing_note="requires separate operator/host trace; absent counters are unknown")


def request(base, body, stem):
    marker = stem.with_suffix(".attempt.json")
    if marker.exists() or stem.with_suffix(".request.json").exists():
        raise RuntimeError("Refusing to overwrite a previously started request")
    save_evidence(stem.with_suffix(".request.json"),body)
    attempt = dict(status="started",boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                   started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save_evidence(marker,attempt)
    req = urllib.request.Request(base+"/generate", json.dumps(body).encode(), {"Content-Type":"application/json"})
    first = None
    last = None
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as response, stem.with_suffix(".sse").open("w",buffering=1) as raw:
            for line in response:
                elapsed = time.perf_counter()-start
                text = line.decode()
                raw.write(json.dumps(dict(seconds=elapsed, line=text))+"\n")
                if not text.startswith("data:") or text[5:].strip() == "[DONE]":
                    continue
                last = json.loads(text[5:])
                if last.get("meta_info", {}).get("completion_tokens", 0) and first is None:
                    first = elapsed
        wall = time.perf_counter()-start
        if last is None or first is None:
            raise RuntimeError("No generated-token observation")
        save_evidence(stem.with_suffix(".response.json"),last)
        # Only transport/response persistence, not a quality or cache PASS.
        save_evidence(marker,attempt|dict(status="response_saved",wall_seconds=wall))
        return last, first, wall
    except BaseException as error:
        save_evidence(marker,attempt|dict(status="error",error=repr(error)))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    out = p.parse_args().out
    if sha(out/"protocol.json") != (out/"protocol.sha256").read_text().strip():
        raise RuntimeError("Frozen protocol changed")
    plan = json.loads((out/"protocol.json").read_text())
    if plan.get("warmups_per_case") != 1:
        raise RuntimeError("Declare one excluded complete warmup cycle per context")
    if sha(__file__) != plan["runner_sha256"]:
        raise RuntimeError("Measurement runner changed")
    if sha(Path(__file__).with_name("sglang_cache_evidence.py")) != plan["cache_evidence_sha256"]:
        raise RuntimeError("Cache-evidence helper changed")
    if (out/"requests.jsonl").exists() or (out/"run-started.json").exists():
        raise RuntimeError("Refusing to overwrite attempts")
    corpus = out/plan["corpus"]["file"]
    if sha(corpus) != plan["corpus"]["sha256"]:
        raise RuntimeError("Frozen corpus changed")
    image = subprocess.check_output(["docker","inspect",plan["container"],"--format","{{.Image}}"],text=True).strip()
    if image != plan["image_id"]:
        raise RuntimeError("Serving image differs")
    with urllib.request.urlopen(plan["base"]+"/get_server_info",timeout=30) as response:
        info = json.load(response)
    runtime = info.get("server_args",info)
    for key,value in plan["expected_server_args"].items():
        if runtime.get(key) != value:
            raise RuntimeError(f"Runtime differs: {key}")
    (out/"server-info.json").write_text(json.dumps(info,indent=2)+"\n")
    cases = json.loads(corpus.read_text())["cases"]
    for case in cases:
        ids = case["input_ids"]
        assert len(ids) == case["base_tokens"]
        assert len(ids)+1 <= runtime["context_length"], "Full base does not fit"
        assert len(case.get("branches",[])) == plan["repetitions"]
        for branch in case["branches"]:
            for size,tail in branch.items():
                assert len(tail) == int(size)
                assert len(ids)+len(tail)+1 <= runtime["context_length"], "Full base plus tail does not fit; do not silently shorten it"

    spec = importlib.util.spec_from_file_location("hardware",Path(__file__).parent/"baseline-earkita/hardware.py")
    hw = importlib.util.module_from_spec(spec);spec.loader.exec_module(hw)
    mapping = hw.hip_mapping(plan["container"])
    idle = hw.capture(out/"pcie-idle",mapping);expected = hw.expected_links(idle)
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    save_evidence(out/"run-started.json",dict(boot_id=boot_id,pid=os.getpid(),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(out/"protocol.json")))
    samples = [];observer_errors = [];done = threading.Event()
    watcher = threading.Thread(target=watch_links,args=(lambda:hw.links(mapping),
        out/"loaded-links.journal.jsonl",done,samples,observer_errors,boot_id));watcher.start()
    rows = []
    try:
        for case in cases:
            ids = case["input_ids"]
            for repetition in range(-1, plan["repetitions"]):
                flush = urllib.request.Request(plan["base"]+"/flush_cache?timeout=5",b"",method="POST")
                with urllib.request.urlopen(flush,timeout=10) as response:
                    (out/f'{case["id"]}-{repetition}.flush.txt').write_bytes(response.read())
                phases = [("cold", ids, 0), ("warm", ids, len(ids))]
                phases += [("append-"+size, ids+tail, len(ids)) for size,tail in case["branches"][max(0,repetition)].items()]
                for phase, input_ids, prefix in phases:
                    if observer_errors:
                        raise RuntimeError(f"PCIe observer failed: {observer_errors}")
                    label = f'{case["id"]}-{repetition}-{phase}'
                    row = dict(case=case["id"],repetition=repetition,phase=phase,
                               excluded_warmup=repetition == -1,status="error")
                    try:
                        body = dict(input_ids=input_ids,stream=True,
                            cache_salt=plan["cache_salt"]+"-"+case["id"],
                            sampling_params=plan["sampling"]|{"max_new_tokens":1,"ignore_eos":False})
                        result,first,wall = request(plan["base"],body,out/label)
                        meta = result["meta_info"]
                        row.update(meta_info=meta,metrics=metrics(meta,first,len(input_ids),prefix),wall_seconds=wall)
                        require_completed(meta)
                        row["cache_evidence"] = cache_evidence(meta,phase,len(ids),plan)
                        row["status"] = "pass"
                    except BaseException as error:
                        row["error"] = repr(error)
                        raise
                    finally:
                        with (out/"requests.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
                    if repetition >= 0:
                        rows.append(row)
                    print(json.dumps({k:row[k] for k in ("case","repetition","phase","metrics")}),flush=True)
    finally:
        request_failed = sys.exc_info()[0] is not None
        done.set();watcher.join()
        (out/"loaded-links.json").write_text(json.dumps(samples)+"\n")
        subprocess.run(["docker","logs",plan["container"]],stdout=(out/"server.log").open("w"),stderr=subprocess.STDOUT)
        try:
            if observer_errors:
                raise RuntimeError(f"PCIe observer failed: {observer_errors}")
            hw.check_loaded(samples,idle,expected)
        except BaseException as error:
            save_evidence(out/"hardware-gate-error.json",dict(error=repr(error)))
            if not request_failed:
                raise
    summary = {}
    for case in cases:
        for phase in sorted({row["phase"] for row in rows if row["case"]==case["id"]}):
            selected = [row for row in rows if row["case"]==case["id"] and row["phase"]==phase]
            values = [row["metrics"]["ttft_seconds"] for row in selected]
            sources = [row["cache_evidence"]["cache_source"] for row in selected]
            summary[case["id"]+"/"+phase] = dict(samples=len(values),ttft_median=statistics.median(values),ttft_min=min(values),ttft_max=max(values),values=values,
                cache_source_counts={source:sources.count(source) for source in sorted(set(sources))},
                cache_hits_observed=sum(row["cache_evidence"]["cache_hit_observed"] for row in selected))
    (out/"summary.json").write_text(json.dumps(dict(scope=plan["scope"],results=summary),indent=2)+"\n")


if __name__ == "__main__":
    main()
