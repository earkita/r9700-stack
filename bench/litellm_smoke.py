#!/usr/bin/env python3
"""Live local gateway contract checks; not an accuracy or throughput benchmark.

LITELLM_MASTER_KEY is read from the environment, never written to evidence.
Saves all requests/responses before checking assertions; no selected retries.
"""
import argparse
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from api_smoke import collect_stream


def anthropic_stream(lines):
    events, blocks = [], {}
    stopped, reason = False, None
    for raw in lines:
        if not raw.startswith(b"data:"):
            continue
        event = json.loads(raw[5:].strip())
        events.append(event)
        kind = event.get("type")
        if kind == "error":
            raise RuntimeError(event)
        if kind == "content_block_start":
            blocks[event["index"]] = dict(event["content_block"])
        elif kind == "content_block_delta":
            block = blocks[event["index"]]
            delta = event["delta"]
            for key in ("text", "thinking", "signature", "partial_json"):
                if key in delta:
                    block[key] = block.get(key, "") + delta[key]
        elif kind == "message_delta":
            reason = event["delta"].get("stop_reason") or reason
        elif kind == "message_stop":
            stopped = True
    for block in blocks.values():
        if block.get("type") == "tool_use" and "partial_json" in block:
            block["input"] = json.loads(block.pop("partial_json"))
    return dict(content=list(blocks.values()), stop_reason=reason,
                stream_completed=stopped, events=events)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:4000")
    ap.add_argument("--backend", default="http://127.0.0.1:8080")
    ap.add_argument("--backend-model", default="glm-5.3-flash")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    key = os.environ["LITELLM_MASTER_KEY"]
    args.out.mkdir(parents=True, exist_ok=False)
    checks = []

    def call(label, path, body, direct=False):
        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        if not direct:
            headers.update(Authorization="Bearer " + key, **{"x-api-key": key})
        elif os.environ.get("LITELLM_BACKEND_KEY", "EMPTY") != "EMPTY":
            backend_key = os.environ["LITELLM_BACKEND_KEY"]
            headers.update(Authorization="Bearer " + backend_key, **{"x-api-key": backend_key})
        req = urllib.request.Request((args.backend if direct else args.base) + path,
                                     json.dumps(body).encode(), headers)
        start = time.perf_counter()
        row = dict(label=label, path=path, direct=direct, request=body)
        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                if body.get("stream"):
                    result = (anthropic_stream(response) if path == "/v1/messages"
                              else collect_stream(response))
                else:
                    result = json.load(response)
            row["response"] = result
        except Exception as error:
            row["error"] = str(error)
            raise
        finally:
            row["elapsed_s"] = time.perf_counter() - start
            with (args.out / "requests.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
        return result

    def passed(name):
        checks.append(name)
        (args.out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        print("PASS", name, flush=True)

    try:
        urllib.request.urlopen(args.base + "/v1/models", timeout=10)
    except urllib.error.HTTPError as e:
        assert e.code in (401, 403), e.code
    else:
        raise AssertionError("Unauthenticated model listing accepted")
    passed("authentication required")

    prompt = "What is 19 + 4? Reply with only the number."
    common = dict(model="glm-5.3-flash", temperature=0, max_tokens=1024,
                  messages=[dict(role="user", content=prompt)],
                  chat_template_kwargs=dict(reasoning_effort="high"))
    tool = dict(type="function", function=dict(name="lookup_file", description="Read a file",
                parameters=dict(type="object", properties=dict(path=dict(type="string")),
                                required=["path"], additionalProperties=False)))
    for stream in (False, True):
        body = common | dict(stream=stream)
        if stream:
            body["stream_options"] = dict(include_usage=True)
        res = call(f"openai-text-{stream}", "/v1/chat/completions", body)
        text = res["content"] if stream else res["choices"][0]["message"]["content"]
        finish = res["finish_reason"] if stream else res["choices"][0]["finish_reason"]
        assert re.fullmatch(r"23\.?", text.strip()) and finish == "stop", res
        assert res["usage"]["completion_tokens"] > 0
        passed(f"OpenAI answer, stream={stream}")
        body.update(messages=[dict(role="user", content="Use lookup_file to read README.md.")],
                    tools=[tool], tool_choice="auto", max_tokens=4096)
        res = call(f"openai-tool-{stream}", "/v1/chat/completions", body)
        finish = res["finish_reason"] if stream else res["choices"][0]["finish_reason"]
        calls = res["tool_calls"] if stream else res["choices"][0]["message"]["tool_calls"]
        assert finish == "tool_calls" and len(calls) == 1
        c = calls[0] if stream else calls[0]["function"]
        assert c["name"] == "lookup_file" and json.loads(c["arguments"]) == {"path": "README.md"}
        if not stream:
            assistant = res["choices"][0]["message"]
            body.update(messages=body["messages"] + [assistant, dict(role="tool",
                        tool_call_id=calls[0]["id"], content="The file contains the code LOCAL_OK. Report that code.")])
            follow = call("openai-tool-result", "/v1/chat/completions", body)
            assert "LOCAL_OK" in follow["choices"][0]["message"]["content"]
            assert follow["choices"][0]["finish_reason"] == "stop"
            passed("OpenAI tool round trip")
        passed(f"OpenAI tool call, stream={stream}")

    common = dict(model="glm-5.3-flash-high", max_tokens=4096, temperature=1,
                  thinking=dict(type="enabled", budget_tokens=2048),
                  messages=[dict(role="user", content="Find the smallest positive integer with remainders "
                                 "2 modulo 3, 3 modulo 5, and 2 modulo 7. Reason carefully, then answer.")])
    for stream in (False, True):
        res = call(f"anthropic-thinking-{stream}", "/v1/messages", common | dict(stream=stream))
        content = res["content"]
        assert res["stop_reason"] == "end_turn", res
        assert any(b["type"] == "thinking" and b.get("thinking") for b in content), res
        assert any(b["type"] == "text" and re.search(r"\b23\b", b["text"]) for b in content), res
        if stream:
            assert res["stream_completed"]
        passed(f"Anthropic thinking/answer, stream={stream}")
        body = common | dict(stream=stream, messages=[dict(role="user", content="Use lookup_file to read README.md.")],
                             tools=[dict(name="lookup_file", description="Read a file", input_schema=tool["function"]["parameters"])])
        res = call(f"anthropic-tool-{stream}", "/v1/messages", body)
        calls = [b for b in res["content"] if b["type"] == "tool_use"]
        assert res["stop_reason"] == "tool_use" and len(calls) == 1, res
        assert calls[0]["name"] == "lookup_file" and calls[0]["input"] == {"path": "README.md"}, res
        if stream:
            assert res["stream_completed"]
        # Echo the actual thinking/signature/tool blocks, as a harness does.
        body.update(stream=False, messages=body["messages"] + [dict(role="assistant", content=res["content"]),
            dict(role="user", content=[dict(type="tool_result", tool_use_id=calls[0]["id"],
                 content="The file contains the code LOCAL_OK. Report that code.")])])
        follow = call(f"anthropic-tool-result-{stream}", "/v1/messages", body)
        assert follow["stop_reason"] == "end_turn" and any(
            b["type"] == "text" and "LOCAL_OK" in b["text"] for b in follow["content"]), follow
        passed(f"Anthropic tool round trip, streamed call={stream}")

    for tools in ([], [dict(name="lookup_file", description="Read a file", input_schema=tool["function"]["parameters"])]):
        body = dict(model="glm-5.3-flash-high", messages=common["messages"], tools=tools)
        proxied = call("count-proxy", "/v1/messages/count_tokens", body)
        direct = call("count-direct", "/v1/messages/count_tokens", body | dict(model=args.backend_model), direct=True)
        assert proxied["input_tokens"] == direct["input_tokens"] > 0, (proxied, direct)
        passed(f"Local token count matches backend, tools={bool(tools)}")


if __name__ == "__main__":
    main()
