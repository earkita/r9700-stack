import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("api_smoke", Path(__file__).resolve().parents[1] / "bench/api_smoke.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def events(*items):
    return [("data: " + (item if isinstance(item, str) else json.dumps(item))).encode() for item in items]


class Stream(unittest.TestCase):
    def test_tool_fragments(self):
        stream = events(
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1",
              "function": {"name": "lookup_file", "arguments": '{"path":'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0,
              "function": {"arguments": '"README.md"}'}}]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"completion_tokens": 12}}, "[DONE]")
        result = api.collect_stream(stream)
        call = result["tool_calls"][0]
        api.check_tool(call["name"], call["arguments"])
        self.assertEqual(call["id"], "call_1")
        self.assertEqual(result["usage"]["completion_tokens"], 12)

    def test_reasoning_variants(self):
        for key in ("reasoning", "reasoning_content"):
            result = api.collect_stream(events({"choices": [{"delta": {key: "thinking"}}]},
                {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]}, "[DONE]"))
            self.assertEqual(result["reasoning"], "thinking")
            self.assertEqual(result["content"], "answer")

    def test_truncation_and_errors(self):
        for stream in (events("[DONE]"), events({"error": "backend failed"}),
                       events({"choices": [{"delta": {}, "finish_reason": "stop"}]})):
            with self.assertRaises(RuntimeError):
                api.collect_stream(stream)


if __name__ == "__main__":
    unittest.main()
