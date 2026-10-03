"""Run in the pinned LiteLLM image without GPU or network."""

import asyncio
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "proxy"))
try:
    import mimo_hooks
except ModuleNotFoundError as exc:
    if exc.name not in ("httpx", "litellm"):
        raise
    mimo_hooks = None


@unittest.skipIf(mimo_hooks is None, "Requires pinned LiteLLM image")
class MiMoCounter(unittest.TestCase):
    def test_native_messages_preserves_flag_only_for_mimo(self):
        for model in ("mimo-v2.6-flash-mopd", "claude-test"):
            for thinking in (False, True):
                params = dict(
                    max_tokens=256,
                    extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
                )
                with patch.dict(
                    os.environ, LITELLM_ANTHROPIC_MODEL="anthropic/mimo-v2.6-flash-mopd"
                ):
                    result = mimo_hooks.optional_params(
                        params, model=model, custom_llm_provider="anthropic"
                    )
                self.assertEqual(result["max_tokens"], 256)
                if model.startswith("mimo-"):
                    self.assertEqual(
                        result["chat_template_kwargs"]["enable_thinking"], thinking
                    )
                else:
                    self.assertNotIn("chat_template_kwargs", result)

    def test_preserves_alias_thinking_and_tools(self):
        for thinking in (False, True):
            client = AsyncMock()
            response = unittest.mock.Mock()
            response.json.return_value = {"input_tokens": 42}
            client.post.return_value = response
            client.__aenter__.return_value = client
            params = dict(
                model="anthropic/mimo-v2.6-flash-mopd",
                api_key="test-backend-key",
                extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
            )
            tools = [dict(name="read", input_schema={"type": "object"})]
            with (
                patch.dict(
                    os.environ,
                    LITELLM_ANTHROPIC_MODEL=params["model"],
                    LITELLM_BACKEND_BASE="http://127.0.0.1:8080",
                ),
                patch.object(mimo_hooks.httpx, "AsyncClient", return_value=client),
            ):
                r = asyncio.run(
                    mimo_hooks.count_tokens(
                        None,
                        "mimo-v2.6-flash-mopd",
                        [{"role": "user", "content": "hello"}],
                        None,
                        {"litellm_params": params},
                        "alias",
                        tools,
                        "system",
                    )
                )
            self.assertEqual(r.total_tokens, 42)
            args = client.post.call_args
            self.assertEqual(
                args.args[0], "http://127.0.0.1:8080/v1/messages/count_tokens"
            )
            self.assertEqual(
                args.kwargs["json"]["chat_template_kwargs"]["enable_thinking"], thinking
            )
            self.assertEqual(args.kwargs["json"]["tools"], tools)
            self.assertEqual(args.kwargs["json"]["system"], "system")


if __name__ == "__main__":
    unittest.main()
