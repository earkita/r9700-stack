#!/usr/bin/env python3
"""API contract/soak checks, not a throughput benchmark (use BetterBench).

Checks normal and streamed reasoning/content, streamed tool JSON, and a long
generation. --duration 1800 repeats for at least 30 minutes. Server/GPU logs
must be checked separately; an HTTP response cannot prove kernel correctness.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def collect_stream(lines):
    content, reasoning, calls = [], [], {}
    done = False
    finish = None
    usage = None
    for raw in lines:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
            break
        event = json.loads(payload)
        if "error" in event:
            raise RuntimeError(f"Stream error: {event['error']}")
        usage = event.get("usage") or usage
        for choice in event.get("choices", []):
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta", {})
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning") or delta.get("reasoning_content") or "")
            for call in delta.get("tool_calls", []):
                target = calls.setdefault(call["index"], {"id": "", "name": "", "arguments": ""})
                target["id"] += call.get("id") or ""
                function = call.get("function") or {}
                target["name"] += function.get("name") or ""
                target["arguments"] += function.get("arguments") or ""
    if not done or not finish:
        raise RuntimeError("Truncated SSE stream (missing DONE or finish reason)")
    return {"content": "".join(content), "reasoning": "".join(reasoning),
            "tool_calls": list(calls.values()), "finish_reason": finish, "usage": usage}


def post(base, body, transcript=None):
    request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = collect_stream(response) if body.get("stream") else json.load(response)
    if transcript:
        with open(transcript, "a") as f:
            f.write(json.dumps({"time": time.time(), "request": body, "response": result}) + "\n")
    return result


def check_tool(name, arguments):
    if name != "lookup_file" or json.loads(arguments) != {"path": "README.md"}:
        raise RuntimeError("Incorrect tool name or JSON arguments")


def cycle(base, model, long_tokens, transcript=None):
    common = {"model": model, "temperature": 0.0, "max_tokens": 4096,
              "messages": [{"role": "user", "content": "Find the smallest positive integer with remainders "
                            "2 modulo 3, 3 modulo 5, and 2 modulo 7. Reason carefully, then give the answer."}],
              "chat_template_kwargs": {"reasoning_effort": "high"}}
    result = post(base, common, transcript)
    message = result["choices"][0]["message"]
    if not message.get("content") or not (message.get("reasoning") or message.get("reasoning_content")):
        raise RuntimeError("Missing non-stream content/reasoning")
    stream = {"stream": True, "stream_options": {"include_usage": True}}
    result = post(base, common | stream, transcript)
    if not result["content"] or not result["reasoning"]:
        raise RuntimeError("Missing streamed content/reasoning")
    tool = {"type": "function", "function": {"name": "lookup_file", "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                           "required": ["path"], "additionalProperties": False}}}
    request = common | {"messages": [{"role": "user", "content": "Use lookup_file to read README.md."}],
                        "tools": [tool], "tool_choice": "auto"}
    result = post(base, request, transcript)
    choice = result["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    if choice["finish_reason"] != "tool_calls" or len(calls) != 1 or not calls[0].get("id"):
        raise RuntimeError("Missing non-stream tool call")
    check_tool(calls[0]["function"]["name"], calls[0]["function"]["arguments"])
    result = post(base, request | stream, transcript)
    calls = result["tool_calls"]
    if result["finish_reason"] != "tool_calls" or len(calls) != 1 or not calls[0]["id"]:
        raise RuntimeError("Missing streamed tool call")
    check_tool(calls[0]["name"], calls[0]["arguments"])
    result = post(base, common | stream | {"messages": [{"role": "user", "content":
                  "Write a detailed guide to implementing a compiler, with examples for every phase."}],
                  "max_tokens": long_tokens, "ignore_eos": True}, transcript)
    if ((result.get("usage") or {}).get("completion_tokens") != long_tokens
            or not (result["content"] or result["reasoning"])):
        raise RuntimeError("Long generation did not return the requested token budget/content")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8080")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--duration", type=float, default=0, help="seconds; 10800 for 3 hours")
    parser.add_argument("--long-tokens", type=int, default=4096)
    parser.add_argument("--transcript", type=Path, help="append requests/responses for output inspection")
    args = parser.parse_args()
    if args.duration < 0 or args.long_tokens < 1:
        parser.error("duration must be nonnegative and long-tokens positive")
    if args.transcript:
        args.transcript.parent.mkdir(parents=True, exist_ok=True)
    base = args.base.rstrip("/")
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as response:
        if args.model not in {m["id"] for m in json.load(response)["data"]}:
            raise RuntimeError("Requested model is not served")
    start = time.monotonic()
    count = 0
    while True:
        cycle(base, args.model, args.long_tokens, args.transcript)
        count += 1
        elapsed = time.monotonic() - start
        print(json.dumps({"cycles": count, "elapsed_s": elapsed, "api_contract": "pass"}), flush=True)
        if elapsed >= args.duration:
            break


if __name__ == "__main__":
    main()
