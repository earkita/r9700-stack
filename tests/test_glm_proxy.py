"""GLM Messages transport gates; run in the pinned LiteLLM image."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "proxy"))
try:
    import glm_hooks
    import httpx
except ModuleNotFoundError as exc:
    if exc.name != "httpx" and not exc.name.startswith("litellm"):
        raise
    glm_hooks = None


@unittest.skipIf(glm_hooks is None, "Run in the pinned LiteLLM image")
class GLMProxy(unittest.IsolatedAsyncioTestCase):
    def test_low_template_survives_client_high_without_mutating_request(self):
        params = {
            "max_tokens": 2048,
            "output_config": {"effort": "high"},
            "extra_body": {"chat_template_kwargs": {"reasoning_effort": "low"}},
        }
        before = copy.deepcopy(params)
        with patch.dict(os.environ, LITELLM_ANTHROPIC_MODEL="anthropic/glm-5.3-flash"):
            result = glm_hooks.optional_params(
                params, model="glm-5.3-flash", custom_llm_provider="anthropic"
            )
        self.assertEqual(result["chat_template_kwargs"], {"reasoning_effort": "low"})
        self.assertEqual(result["output_config"]["effort"], "low")
        self.assertEqual(params, before)

    def test_unrelated_routes_keep_upstream_filtering(self):
        params = {"max_tokens": 2048,
                  "extra_body": {"chat_template_kwargs": {"reasoning_effort": "low"}}}
        with patch.dict(os.environ, LITELLM_ANTHROPIC_MODEL="anthropic/glm-5.3-flash"):
            for model, provider in [("mimo-v2", "anthropic"),
                                    ("glm-other", "anthropic"),
                                    ("glm-5.3-flash", "openai")]:
                with self.subTest(model=model, provider=provider):
                    result = glm_hooks.optional_params(
                        params, model=model, custom_llm_provider=provider
                    )
                    self.assertNotIn("chat_template_kwargs", result)

    async def test_count_uses_local_tokenizer_and_low_template_with_tools(self):
        deployment = {"litellm_params": {
            "model": "anthropic/glm-5.3-flash", "api_key": "test-only",
            "extra_body": {"chat_template_kwargs": {"reasoning_effort": "low"}},
        }}
        response = httpx.Response(200, json={"input_tokens": 37},
                                  request=httpx.Request("POST", "http://localhost"))
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.post.return_value = response
        with patch.dict(os.environ, LITELLM_ANTHROPIC_MODEL="anthropic/glm-5.3-flash",
                        LITELLM_BACKEND_BASE="http://127.0.0.1:8080"), \
                patch.object(glm_hooks.httpx, "AsyncClient", return_value=client):
            result = await glm_hooks.count_tokens(
                None, "glm-5.3-flash", [{"role": "user", "content": "OK"}],
                None, deployment, "glm-5.3-flash-fast", tools=[], system="Be concise.")
        self.assertEqual(result.total_tokens, 37)
        args, kwargs = client.post.call_args
        self.assertEqual(args[0], "http://127.0.0.1:8080/v1/messages/count_tokens")
        self.assertEqual(kwargs["json"]["chat_template_kwargs"], {"reasoning_effort": "low"})
        self.assertEqual(kwargs["json"]["system"], "Be concise.")
        self.assertEqual(kwargs["json"]["tools"], [])

    async def test_high_alias_keeps_original_counter(self):
        original = AsyncMock(return_value="unchanged")
        with patch.object(glm_hooks, "_original", original):
            result = await glm_hooks.count_tokens(
                None, "glm-5.3-flash", [], None, {}, "glm-5.3-flash-high")
        self.assertEqual(result, "unchanged")
        original.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
