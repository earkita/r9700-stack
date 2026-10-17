#!/usr/bin/env python3
"""Frozen two-call/tool-result API contract, not a model quality benchmark.

Requires an exclusive prepared server, pinned image/runtime/protocol, and all
eight GPU PCIe evidence. Does not build an image or launch/restart services.
CPU parser tests separately cover multiple calls coalesced in a final chunk;
model-produced streaming chunk boundaries are not guaranteed to reproduce it.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import threading
import time
import urllib.request

from api_smoke import collect_stream


TOOLS = [{"type": "function", "function": {
    "name": name, "description": "Read " + name + "; no arguments.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}
    for name in ("read_first", "read_second")]
VALUES = {"read_first": 17, "read_second": 25}


class CompletionTruncated(ValueError):
    pass


def require_termination(response):
    finish = response.get("finish_reason")
    if finish == "length":
        raise CompletionTruncated("Completion exhausted its declared token budget")
    if finish not in ("stop", "tool_calls"):
        raise RuntimeError("Missing, aborted or unknown finish reason: " + str(finish))


def strict_json(value):
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = item
        return result
    return json.loads(value, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def normalize(result, streamed):
    if streamed:
        return result
    if len(result.get("choices", [])) != 1:
        raise ValueError("Expected exactly one completion choice")
    choice = result["choices"][0]
    message = choice["message"]
    return dict(content=message.get("content") or "",
                reasoning=message.get("reasoning_content") or "",
                tool_calls=[dict(id=call["id"], name=call["function"]["name"],
                                 arguments=call["function"]["arguments"])
                            for call in message.get("tool_calls") or []],
                finish_reason=choice["finish_reason"], usage=result.get("usage"))


def tool_results(response, tag):
    require_termination(response)
    calls = response.get("tool_calls") or []
    if response.get("finish_reason") != "tool_calls" or len(calls) != 2:
        raise ValueError("Expected two completed tool calls")
    identities = [call.get("id") for call in calls]
    if any(not isinstance(item, str) or not item for item in identities) or len(set(identities)) != 2:
        raise ValueError("Tool call IDs must be nonempty and distinct")
    if {call.get("name") for call in calls} != set(VALUES):
        raise ValueError("Missing, repeated or unknown tool name")
    results = []
    for call in calls:
        if strict_json(call["arguments"]) != {}:
            raise ValueError("No-argument tool received arguments")
        results.append(dict(role="tool", tool_call_id=call["id"], content=json.dumps(
            dict(value=VALUES[call["name"]], tag=tag))))
    return results


def check_final(response, tag):
    require_termination(response)
    if response.get("finish_reason") != "stop" or response.get("tool_calls"):
        raise ValueError("Final answer did not finish normally without further tools")
    answer = strict_json(response["content"])
    if (not isinstance(answer, dict) or set(answer) != {"total", "tag"}
            or type(answer["total"]) is not int or answer != {"total": 42, "tag": tag}):
        raise ValueError("Wrong final total, tag or JSON format")


def post(base, body, stem):
    stem.with_suffix(".request.json").write_text(json.dumps(body, indent=2)+"\n")
    req = urllib.request.Request(base+"/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        if body["stream"]:
            with stem.with_suffix(".sse").open("wb") as raw:
                def retained():
                    for line in response:
                        raw.write(line)
                        raw.flush()
                        yield line
                result = collect_stream(retained())
        else:
            result = json.load(response)
    stem.with_suffix(".response.json").write_text(json.dumps(result, indent=2)+"\n")
    return normalize(result, body["stream"])


def attempt(plan, seed, streamed, out, transport=post):
    # Exact request templates must be identical in the control/candidate legs.
    tag = "two-call-"+str(seed)+("-stream" if streamed else "-plain")
    messages = [{"role": "user", "content":
        "Call both read_first and read_second once, together in your first response. "
        "Do not guess their values. After receiving both tool results, return only "
        'a JSON object with their sum in "total" and the returned tag in "tag".'}]
    common = dict(model=plan["model"], temperature=0.0, top_p=1.0, top_k=-1,
                  max_tokens=512, seed=seed, stream=streamed,
                  cache_salt=plan["cache_salt"]+"-"+tag,
                  chat_template_kwargs={"reasoning_effort": "high", "clear_thinking": False})
    if streamed:
        common["stream_options"] = {"include_usage": True}
    start = time.perf_counter()
    record = dict(seed=seed, stream=streamed, tag=tag, status="incomplete")
    try:
        first = transport(plan["base"], common | dict(messages=messages, tools=TOOLS,
            tool_choice="auto", parallel_tool_calls=True), out/"calls")
        results = tool_results(first, tag)
        assistant = dict(role="assistant", content=first["content"] or None,
            tool_calls=[dict(id=call["id"], type="function", function=dict(
                name=call["name"], arguments=call["arguments"])) for call in first["tool_calls"]])
        if first.get("reasoning"):
            assistant["reasoning_content"] = first["reasoning"]
        final = transport(plan["base"], common | dict(messages=messages+[assistant]+results,
            tools=TOOLS, tool_choice="none", response_format={"type": "json_schema", "json_schema": {
                "name": "two_tool_result", "strict": True, "schema": {
                    "type": "object", "properties": {"total": {"type": "integer"},
                    "tag": {"type": "string"}}, "required": ["total", "tag"],
                    "additionalProperties": False}}}), out/"result")
        check_final(final, tag)
        record.update(status="pass", tool_calls=2, final=final)
    except CompletionTruncated as error:
        record.update(status="incomplete", error=repr(error))
    except (ValueError, KeyError, TypeError) as error:
        record.update(status="contract_failure", error=repr(error))
    except Exception as error:
        record.update(status="error", error=repr(error))
    record["wall_seconds"] = time.perf_counter()-start
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if (out/"requests.jsonl").exists():
        raise RuntimeError("Refusing to replace previous attempts")
    if digest(out/"protocol.json") != (out/"protocol.sha256").read_text().strip():
        raise RuntimeError("Frozen protocol changed")
    plan = json.loads((out/"protocol.json").read_text())
    for file, expected in plan["sources"].items():
        if digest(file) != expected:
            raise RuntimeError("Frozen source changed: "+file)
    if plan["sources"].get(str(Path(__file__).relative_to(Path.cwd()))) != digest(__file__):
        raise RuntimeError("Runner must be frozen")
    if plan["seeds"] != [1234, 5678, 9012]:
        raise RuntimeError("Declare the entire three-seed contract cohort")
    identity = json.loads(subprocess.check_output(["docker", "inspect", plan["container"]],text=True))[0]
    if not identity["State"]["Running"] or identity["Image"] != plan["image_id"]:
        raise RuntimeError("Wrong or stopped model image")
    with urllib.request.urlopen(plan["base"]+"/get_server_info",timeout=30) as response:
        info = json.load(response)
    runtime = info.get("server_args",info)
    if any(runtime.get(key) != value for key,value in plan["expected_server_args"].items()):
        raise RuntimeError("Runtime differs from frozen protocol")
    (out/"server-info.json").write_text(json.dumps(info,indent=2)+"\n")
    spec = importlib.util.spec_from_file_location("hardware",Path(__file__).parent/"baseline-earkita/hardware.py")
    hw = importlib.util.module_from_spec(spec); spec.loader.exec_module(hw)
    mapping = hw.hip_mapping(plan["container"]); idle = hw.capture(out/"pcie-before",mapping)
    from sglang_glm_prefill import watch_links
    samples, errors, rows = [], [], []
    done = threading.Event()
    worker = threading.Thread(target=watch_links,args=(lambda:hw.links(mapping),
        out/"loaded-links.journal.jsonl",done,samples,errors,
        Path("/proc/sys/kernel/random/boot_id").read_text().strip()))
    worker.start()
    try:
        for streamed in (False,True):
            for seed in plan["seeds"]:
                folder = out/(str(seed)+("-stream" if streamed else "-plain"))
                folder.mkdir()
                row = attempt(plan,seed,streamed,folder)
                rows.append(row)
                with (out/"requests.jsonl").open("a") as file:
                    file.write(json.dumps(row)+"\n");file.flush()
                print(json.dumps(row),flush=True)
                if row["status"] == "error":
                    raise RuntimeError("Transport/runtime failure; inspect before further model requests")
    finally:
        done.set(); worker.join()
        if errors:
            raise RuntimeError("PCIe observer failed: "+repr(errors))
        hw.check_loaded(samples,idle,hw.expected_links(idle))
    passed = len(rows)==6 and all(row["status"]=="pass" for row in rows)
    (out/"status.json").write_text(json.dumps(dict(status="pass" if passed else "failed",
        scope="Two-call and tool-result API contract only, not model quality/performance",
        attempts=len(rows),pass_count=sum(row["status"]=="pass" for row in rows),
        final_schema="Applied only after both tool calls were validated; tool_choice=none",
        chunk_boundary="Model API chunking is not forced; separate CPU regression covers coalesced final calls"),indent=2)+"\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
