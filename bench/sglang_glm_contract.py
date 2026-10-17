#!/usr/bin/env python3
"""Bounded direct-SGLang answer, streaming and two-turn tool contract smoke.

This is not a quality benchmark or proof of model equivalence. Preserve every
request/response and keep it separate from timed decode comparisons.
"""

import argparse
import json
from pathlib import Path
import re
import urllib.request

from api_smoke import collect_stream


def post(base, body, output, name):
    (output / f"{name}.request.json").write_text(json.dumps(body, indent=2) + "\n")
    req = urllib.request.Request(
        base + "/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        if body.get("stream"):
            with (output / f"{name}.sse").open("wb") as raw:
                def retained():
                    for line in response:
                        raw.write(line)
                        yield line
                result = collect_stream(retained())
        else:
            result = json.load(response)
    (output / f"{name}.json").write_text(json.dumps(
        {"request": body, "response": result}, indent=2, ensure_ascii=False) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://127.0.0.1:8081")
    p.add_argument("--model", default="glm-5.3-flash-sglang")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    common = {"model": args.model, "temperature": 0.0, "top_p": 1.0,
              "top_k": -1, "max_tokens": 512,
              "chat_template_kwargs": {"reasoning_effort": "high"}}
    answer = post(args.base, common | {"messages": [
        {"role": "user", "content": "What is 7 plus 5? Give the number."}]},
        args.out, "answer")
    text = answer["choices"][0]["message"].get("content") or ""
    if not re.search(r"\b12\b", text) or answer["choices"][0]["finish_reason"] != "stop":
        raise RuntimeError("Wrong or incomplete arithmetic answer")

    tool = {"type": "function", "function": {"name": "add",
            "description": "Add two integers", "parameters": {"type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"], "additionalProperties": False}}}
    prompt = {"role": "user", "content": "Use the add tool to calculate 7 + 5, then tell me the result."}
    first = post(args.base, common | {"messages": [prompt], "tools": [tool],
                                     "tool_choice": "auto"}, args.out, "tool_call")
    choice = first["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    if choice["finish_reason"] != "tool_calls" or len(calls) != 1:
        raise RuntimeError("No completed tool call")
    call = calls[0]
    if call["function"]["name"] != "add" or \
            json.loads(call["function"]["arguments"]) != {"a": 7, "b": 5}:
        raise RuntimeError("Wrong tool or arguments")
    final = post(args.base, common | {"messages": [
        prompt, choice["message"], {"role": "tool", "tool_call_id": call["id"],
                                    "content": "12"}], "tools": [tool]}, args.out, "tool_result")
    final_text = final["choices"][0]["message"].get("content") or ""
    if not re.search(r"\b12\b", final_text) or final["choices"][0]["finish_reason"] != "stop":
        raise RuntimeError("Tool result did not produce a completed answer")

    streamed = post(args.base, common | {"messages": [
        {"role": "user", "content": "Say OK."}], "stream": True,
        "stream_options": {"include_usage": True}}, args.out, "stream")
    if not streamed["content"] or streamed["finish_reason"] != "stop":
        raise RuntimeError("Streaming response incomplete")

    streamed_call = post(args.base, common | {"messages": [prompt], "tools": [tool],
                         "tool_choice": "auto", "stream": True,
                         "stream_options": {"include_usage": True}}, args.out, "stream_tool_call")
    calls = streamed_call.get("tool_calls") or []
    if streamed_call["finish_reason"] != "tool_calls" or len(calls) != 1:
        raise RuntimeError("No completed streamed tool call")
    call = calls[0]
    if not call["id"] or call["name"] != "add" or json.loads(call["arguments"]) != {"a": 7, "b": 5}:
        raise RuntimeError("Wrong streamed tool or arguments")
    assistant = {"role": "assistant", "content": streamed_call["content"] or None,
                 "tool_calls": [{"id": call["id"], "type": "function", "function": {
                     "name": call["name"], "arguments": call["arguments"]}}]}
    if streamed_call["reasoning"]:
        assistant["reasoning_content"] = streamed_call["reasoning"]
    streamed_final = post(args.base, common | {"messages": [prompt, assistant,
                          {"role": "tool", "tool_call_id": call["id"], "content": "12"}],
                          "tools": [tool], "stream": True,
                          "stream_options": {"include_usage": True}}, args.out, "stream_tool_result")
    if not re.search(r"\b12\b", streamed_final["content"]) or streamed_final["finish_reason"] != "stop":
        raise RuntimeError("Streamed tool result did not produce a completed answer")
    (args.out / "status.json").write_text(json.dumps({
        "status": "pass", "scope": "bounded answer/stream/two-turn tool contract only",
        "requests": 6}, indent=2) + "\n")
    print("PASS bounded answer/stream/two-turn tool contract")


if __name__ == "__main__":
    main()
