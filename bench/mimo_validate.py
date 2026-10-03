#!/usr/bin/env python3
"""Frozen MiMo concurrent benchmark and independent completion/retrieval gates.

Run --freeze with the local checkpoint tokenizer, then --phase on each runtime.
Bench timings use BetterBench 0.6.0 unchanged. All attempts are retained.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import urllib.request


def api(path, body=None, base="http://127.0.0.1:8080"):
    req = urllib.request.Request(
        base + path,
        None if body is None else json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def counters(base):
    with urllib.request.urlopen(base + "/metrics") as r:
        raw = r.read().decode()
    out = {}
    for line in raw.splitlines():
        if not line.startswith("vllm:"):
            continue
        name = line.split("{")[0].split(" ")[0][5:]
        if any(
            x in name
            for x in (
                "prefix_cache",
                "spec_decode",
                "preemption",
                "num_requests_running",
                "num_requests_waiting",
            )
        ):
            key = name
            if "accepted_tokens_per_pos" in name:
                m = re.search('position="([0-9]+)"', line)
                if m:
                    key += "_" + m[1]
            try:
                out[key] = out.get(key, 0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return out


def freeze(a):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        a.model_path, local_files_only=True, trust_remote_code=False
    )
    a.out.mkdir(parents=True, exist_ok=False)
    cases = []
    for kind, lengths in [
        ("speed", a.bench_lengths),
        ("quality", a.quality_lengths),
    ]:
        for length in lengths:
            for client in range(max(a.concurrency)):
                key = f"MIMO-KEY-{length}-{client}-C83A"
                records = [
                    f"Record {i:06d}: routine inspection completed, archive unchanged, maintenance scheduled.\n"
                    for i in range(12000)
                ]

                def make(n, pad=0):
                    pos = n // 2
                    body = (
                        "".join(records[:pos])
                        + f"CRITICAL NEEDLE: secret code {key}.\n"
                        + "".join(records[pos:n])
                        + "Padding:"
                        + " x" * pad
                    )
                    task = (
                        "Write a detailed Python scheduler implementation and explain its design in at least 1500 words."
                        if kind == "speed"
                        else "Return only the secret code from the CRITICAL NEEDLE record."
                    )
                    return [
                        {
                            "role": "system",
                            "content": f"Archive task for client {client}. Follow the final instruction.",
                        },
                        {"role": "user", "content": body + "\nFINAL TASK: " + task},
                    ]

                def count(msg):
                    return len(
                        tok.apply_chat_template(
                            msg,
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=True,
                            return_dict=False,
                        )
                    )

                lo, hi = 0, len(records)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if count(make(mid)) <= length:
                        lo = mid
                    else:
                        hi = mid - 1
                n = lo
                pad = length - count(make(n))
                msg = make(n, pad)
                for _ in range(8):
                    delta = length - count(msg)
                    if not delta:
                        break
                    pad += delta
                    msg = make(n, pad)
                assert count(msg) == length
                cases.append(
                    dict(
                        id=f"{kind}-{length}-{client}",
                        kind=kind,
                        prompt_tokens=length,
                        client=client,
                        messages=msg,
                        expected=key,
                    )
                )
    text = json.dumps(cases)
    (a.out / "corpus.json").write_text(text)
    plan = dict(
        model="mimo-v2.6-flash-mopd",
        corpus_sha256=hashlib.sha256(text.encode()).hexdigest(),
        temperature=1.0,
        top_p=0.95,
        top_k=-1,
        seed=1234,
        thinking=True,
        ignore_eos=False,
        output_tokens=256,
        quality_output_tokens=4096,
        lengths=a.bench_lengths,
        quality_lengths=a.quality_lengths,
        concurrency=a.concurrency,
        warmup_rounds=1,
        measured_rounds=3,
        cache="unique prefix_salt per phase-independent case/round; cold; verify zero hits",
        decode="BetterBench 0.6.0 (completion_tokens-1)/(last-first content-bearing SSE), includes reasoning; report batched chunks",
        comparison="Record runtime identities and the declared experimental variable before either leg; match other runtime settings and hardware. Quality is independent from speed truncation.",
    )
    (a.out / "protocol.json").write_text(json.dumps(plan, indent=2) + "\n")
    print("Frozen corpus and protocol:", a.out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--model-path")
    p.add_argument("--corpus", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase", choices=["bench", "quality", "apc", "smoke"])
    p.add_argument("--base", default="http://127.0.0.1:8080")
    p.add_argument("--bench-lengths", type=int, nargs="+", default=[1024, 32768, 130816],
                   help="Freeze the same length selection for both comparison legs")
    p.add_argument("--bench-cache-namespace", default="",
                   help="Fresh salt namespace for a cold performance comparison")
    p.add_argument("--concurrency", type=int, nargs="+", choices=[1, 2, 4], default=[1, 2],
                   help="Freeze the same concurrency selection for compared profiles")
    p.add_argument("--quality-lengths", type=int, nargs="+", default=[8192, 32768, 126976])
    p.add_argument(
        "--quality-cache-namespace",
        default="",
        help="Fresh salt namespace for a complete cold quality rerun; prompts stay unchanged",
    )
    a = p.parse_args()
    if a.freeze:
        return freeze(a)
    a.out.mkdir(parents=True, exist_ok=False)
    model = "mimo-v2.6-flash-mopd"
    lock = threading.Lock()

    def save(row):
        with lock:
            with (a.out / "attempts.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")

    from api_smoke import collect_stream

    def request(label, body):
        save(dict(event="request", id=label, body=body))
        try:
            start = time.perf_counter()
            req = urllib.request.Request(
                a.base + "/v1/chat/completions",
                json.dumps(body).encode(),
                {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=1800) as r:
                result = collect_stream(r) if body.get("stream") else json.load(r)
            save(
                dict(
                    event="response",
                    id=label,
                    result=result,
                    wall=time.perf_counter() - start,
                )
            )
            return result
        except Exception as e:
            save(dict(event="error", id=label, error=str(e)))
            raise

    common = dict(
        model=model,
        temperature=1.0,
        top_p=0.95,
        top_k=-1,
        seed=1234,
        max_tokens=4096,
        ignore_eos=False,
        chat_template_kwargs={"enable_thinking": True},
    )
    if a.phase == "smoke":
        checks = []
        tool = dict(
            type="function",
            function=dict(
                name="lookup_file",
                description="Read file",
                parameters=dict(
                    type="object",
                    properties=dict(path=dict(type="string")),
                    required=["path"],
                ),
            ),
        )
        for thinking in [False, True]:
            for stream in [False, True]:
                prefix = f"thinking{thinking}-stream{stream}"
                body = common | dict(
                    chat_template_kwargs={"enable_thinking": thinking},
                    stream=stream,
                    messages=[
                        dict(
                            role="user",
                            content="What is 19 + 4? Return only the number.",
                        )
                    ],
                )
                if stream:
                    body["stream_options"] = {"include_usage": True}
                r = request(prefix, body)
                msg = r if stream else r["choices"][0]["message"]
                finish = (
                    r["finish_reason"] if stream else r["choices"][0]["finish_reason"]
                )
                assert (
                    re.fullmatch(r"23\.?", msg["content"].strip()) and finish == "stop"
                ), r
                if not thinking:
                    assert not (msg.get("reasoning") or msg.get("reasoning_content"))
                checks.append(prefix)
                body.update(
                    tools=[tool],
                    tool_choice="auto",
                    messages=[
                        dict(
                            role="user",
                            content="Use lookup_file to read README.md, then report the status code recorded in that file.",
                        )
                    ],
                )
                r = request(prefix + "-tool", body)
                calls = (
                    r["tool_calls"]
                    if stream
                    else r["choices"][0]["message"]["tool_calls"]
                )
                tool_finish = (
                    r["finish_reason"] if stream else r["choices"][0]["finish_reason"]
                )
                assert (
                    tool_finish == "tool_calls" and len(calls) == 1 and calls[0]["id"]
                ), r
                c = calls[0] if stream else calls[0]["function"]
                assert c["name"] == "lookup_file" and json.loads(c["arguments"]) == {
                    "path": "README.md"
                }, r
                assistant = (
                    dict(
                        role="assistant",
                        content=r["content"] or None,
                        reasoning=r["reasoning"],
                        tool_calls=[
                            dict(
                                id=calls[0]["id"],
                                type="function",
                                function=dict(name=c["name"], arguments=c["arguments"]),
                            )
                        ],
                    )
                    if stream
                    else r["choices"][0]["message"]
                )
                body.update(
                    stream=False,
                    messages=body["messages"]
                    + [
                        assistant,
                        dict(
                            role="tool",
                            tool_call_id=calls[0]["id"],
                            content="README.md\nStatus code: LOCAL_OK",
                        ),
                    ],
                )
                body.pop("stream_options", None)
                follow = request(prefix + "-result", body)
                assert (
                    follow["choices"][0]["finish_reason"] == "stop"
                    and "LOCAL_OK" in follow["choices"][0]["message"]["content"]
                ), follow
                checks.append(prefix + "-tool-roundtrip")
        (a.out / "checks.json").write_text(json.dumps(checks, indent=2))
        print("Smoke PASS", len(checks), flush=True)
        return
    cases = json.loads(a.corpus.read_text())
    for case in cases:
        count = api(
            "/tokenize",
            dict(
                model=model,
                messages=case["messages"],
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": True},
            ),
            a.base,
        )["count"]
        assert count == case["prompt_tokens"], (case["id"], count)
    if a.phase == "bench":
        import betterbench, betterbench.client as client

        assert betterbench.__version__ == "0.6.0"
        add = client._Timeline.add
        finalize = client._finalize

        def keep_add(self, content, reasoning):
            add(self, content, reasoning)
            if not hasattr(self, "captured_content"):
                self.captured_content = []
                self.captured_reasoning = []
            self.captured_content.append(content or "")
            self.captured_reasoning.append(reasoning or "")

        def keep_finalize(result, tl, *args, **kwargs):
            r = finalize(result, tl, *args, **kwargs)
            r.captured_content = "".join(getattr(tl, "captured_content", []))
            r.captured_reasoning = "".join(getattr(tl, "captured_reasoning", []))
            return r

        client._Timeline.add = keep_add
        client._finalize = keep_finalize
        summaries = []
        for length in a.bench_lengths:
            for concurrency in a.concurrency:
                chosen = [
                    c
                    for c in cases
                    if c["kind"] == "speed" and c["prompt_tokens"] == length
                ][:concurrency]
                assert len(chosen) == concurrency, (length, concurrency)
                for round_ in range(4):
                    before = counters(a.base)
                    barrier = threading.Barrier(concurrency)
                    samples = []
                    stop = threading.Event()

                    def monitor():
                        while not stop.wait(0.1):
                            samples.append(
                                dict(time=time.perf_counter(), metrics=counters(a.base))
                            )

                    watcher = threading.Thread(target=monitor, daemon=True)
                    watcher.start()

                    def run(case):
                        extra = dict(
                            chat_template_kwargs={"enable_thinking": True},
                            ignore_eos=False,
                            cache_salt=f"{a.bench_cache_namespace}mimo-frozen-{length}-{concurrency}-{round_}-{case['client']}",
                        )
                        barrier.wait()
                        t0 = time.perf_counter()
                        r = client.stream_chat_sync(
                            a.base + "/v1",
                            model,
                            case["messages"],
                            max_tokens=256,
                            temperature=1.0,
                            top_p=0.95,
                            top_k=-1,
                            seed=1234,
                            extra_body=extra,
                            timeout=1800,
                        )
                        row = dict(
                            id=case["id"],
                            round=round_,
                            warmup=round_ == 0,
                            concurrency=concurrency,
                            start=t0,
                            end=time.perf_counter(),
                            result=r.as_dict(),
                            request_extra=extra,
                        )
                        save(row)
                        assert (
                            r.ok
                            and r.prompt_tokens == length
                            and r.completion_tokens > 0
                        ), row
                        return row

                    try:
                        with ThreadPoolExecutor(max_workers=concurrency) as pool:
                            rows = list(pool.map(run, chosen))
                    finally:
                        stop.set()
                        watcher.join()
                    time.sleep(0.3)
                    after = counters(a.base)
                    assert after.get("num_preemptions_total", 0) == before.get(
                        "num_preemptions_total", 0
                    )
                    assert after.get("prefix_cache_hits_total", 0) == before.get(
                        "prefix_cache_hits_total", 0
                    ), (before, after)
                    starts = [r["start"] + r["result"]["ttft_ms"] / 1000 for r in rows]
                    ends = [
                        t + sum(r["result"]["update_gaps_ms"]) / 1000
                        for t, r in zip(starts, rows)
                    ]
                    overlap = max(0, min(ends) - max(starts))
                    group = dict(
                        length=length,
                        concurrency=concurrency,
                        round=round_,
                        before=before,
                        after=after,
                        rows=rows,
                        decode_intersection_s=overlap,
                        scheduler_samples=samples,
                        aggregate_e2e_tps=sum(
                            r["result"]["completion_tokens"] for r in rows
                        )
                        / (max(r["end"] for r in rows) - min(r["start"] for r in rows)),
                    )
                    if concurrency > 1:
                        assert overlap > 0, "Requests did not decode concurrently"
                    summaries.append(group)
                    (a.out / "groups.json").write_text(json.dumps(summaries, indent=2))
                    print(
                        "BENCH",
                        length,
                        concurrency,
                        round_,
                        [round(r["result"]["decode_tps"], 2) for r in rows],
                        flush=True,
                    )
        return
    if a.phase == "quality":
        from glm_quality import niah_scores

        scores = []
        for length in a.quality_lengths:
            chosen = [
                c
                for c in cases
                if c["kind"] == "quality" and c["prompt_tokens"] == length
            ]
            for concurrency in a.concurrency:
                assert len(chosen) >= concurrency, (length, concurrency)
                before = counters(a.base)
                barrier = threading.Barrier(concurrency)

                def run(case):
                    barrier.wait()
                    r = request(
                        case["id"] + f"-c{concurrency}",
                        common
                        | dict(
                            messages=case["messages"],
                            stream=True,
                            stream_options={"include_usage": True},
                            cache_salt=f"{a.quality_cache_namespace}quality-{length}-{concurrency}-{case['client']}",
                        ),
                    )
                    score = dict(
                        id=case["id"],
                        concurrency=concurrency,
                        **niah_scores(
                            r,
                            case
                            | dict(
                                scoring="exact",
                                key_pattern=r"MIMO-KEY-[0-9]+-[0-9]+-C83A",
                            ),
                        ),
                    )
                    generated = (r.get("content") or "") + (r.get("reasoning") or "")
                    foreign = [
                        c["expected"] for c in chosen if c["client"] != case["client"]
                    ]
                    score["cross_request_contamination"] = any(
                        key in generated for key in foreign
                    )
                    save(dict(event="quality", **score))
                    return score

                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    scores.extend(pool.map(run, chosen[:concurrency]))
                after = counters(a.base)
                save(
                    dict(
                        event="quality_metrics",
                        length=length,
                        concurrency=concurrency,
                        before=before,
                        after=after,
                    )
                )
                assert after.get("num_preemptions_total", 0) == before.get(
                    "num_preemptions_total", 0
                )
                assert after.get("prefix_cache_hits_total", 0) == before.get(
                    "prefix_cache_hits_total", 0
                )
                (a.out / "quality.json").write_text(json.dumps(scores, indent=2))
                print("QUALITY", length, concurrency, "recorded", flush=True)
        failures = [
            s
            for s in scores
            if s["semantic_needle_retrieval"] != "PASS"
            or s["generation_completed"] != "PASS"
            or s["cross_request_contamination"]
        ]
        assert not failures, (
            failures
        )  # Preserve the full matrix before failing the gate.
        return
    if a.phase == "apc":
        chosen = [
            c for c in cases if c["kind"] == "quality" and c["prompt_tokens"] == 32768
        ]
        allrows = []
        for concurrency in (1, 2):
            for repeat in range(2):
                before = counters(a.base)
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    rows = list(
                        pool.map(
                            lambda c: request(
                                c["id"] + f"-apc-c{concurrency}-{repeat}",
                                common
                                | dict(
                                    messages=c["messages"],
                                    stream=True,
                                    stream_options={"include_usage": True},
                                    cache_salt=f"mimo-apc-c{concurrency}-"
                                    + str(c["client"]),
                                ),
                            ),
                            chosen[:concurrency],
                        )
                    )
                after = counters(a.base)
                hits = (
                    after["prefix_cache_hits_total"] - before["prefix_cache_hits_total"]
                )
                allrows.append(
                    dict(
                        concurrency=concurrency,
                        repeat=repeat,
                        before=before,
                        after=after,
                        hits=hits,
                    )
                )
                (a.out / "apc.json").write_text(json.dumps(allrows, indent=2))
                assert all(
                    r["finish_reason"] == "stop" and c["expected"] in r["content"]
                    for r, c in zip(rows, chosen[:concurrency])
                )
                assert hits > 0 if repeat else hits == 0
                print("APC", concurrency, repeat, hits, flush=True)


if __name__ == "__main__":
    main()
