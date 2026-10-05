"""DeepSeek Messages transport gates; run in the pinned LiteLLM image."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "proxy"))
try:
    import deepseek_hooks
    import httpx
except ModuleNotFoundError as exc:
    if exc.name != "httpx" and not exc.name.startswith("litellm"):
        raise
    deepseek_hooks = None


@unittest.skipIf(deepseek_hooks is None, "Run in the pinned LiteLLM image")
class DeepSeekProxy(unittest.IsolatedAsyncioTestCase):
    def test_low_template_survives_client_high_without_mutating_request(self):
        params = {
            "max_tokens": 2048,
            "output_config": {"effort": "high"},
            "thinking": {"type": "disabled"},
            "extra_body": {"chat_template_kwargs": {
                "reasoning_effort": "low", "enable_thinking": True}},
        }
        before = copy.deepcopy(params)
        with patch.dict(os.environ, LITELLM_ANTHROPIC_MODEL="anthropic/deepseek-v4.1-flash"):
            result = deepseek_hooks.optional_params(
                params, model="deepseek-v4.1-flash", custom_llm_provider="anthropic"
            )
        self.assertEqual(result["chat_template_kwargs"], {
            "reasoning_effort": "low", "enable_thinking": True})
        self.assertEqual(result["output_config"]["effort"], "low")
        self.assertEqual(result["thinking"], {"type": "adaptive"})
        self.assertEqual(params, before)

    def test_config_and_client_roles_agree(self):
        import json
        import yaml
        root = Path(__file__).resolve().parents[1]
        cfg = yaml.safe_load((root / 'proxy/deepseek.yaml').read_text())
        routes = {r['model_name']: r for r in cfg['model_list']}
        client = json.loads((root / 'serve/templates/deepseek-v4.1-flash.settings.local.json').read_text())
        for role in ('ANTHROPIC_SMALL_FAST_MODEL', 'ANTHROPIC_DEFAULT_HAIKU_MODEL'):
            name = client['env'][role]
            self.assertEqual(name, 'deepseek-v4.1-flash-fast')
            self.assertEqual(routes[name]['litellm_params']['extra_body']['chat_template_kwargs'],
                             {'enable_thinking': True, 'reasoning_effort': 'low'})
        self.assertEqual(routes[client['model']]['litellm_params']['extra_body']['chat_template_kwargs']['reasoning_effort'], 'high')

    def test_unrelated_routes_keep_upstream_filtering(self):
        params = {"max_tokens": 2048,
                  "extra_body": {"chat_template_kwargs": {"reasoning_effort": "low"}}}
        with patch.dict(os.environ, LITELLM_ANTHROPIC_MODEL="anthropic/deepseek-v4.1-flash"):
            for model, provider in [("mimo-v2", "anthropic"),
                                    ("deepseek-v4.1-other", "anthropic"),
                                    ("deepseek-v4.1-flash", "openai")]:
                with self.subTest(model=model, provider=provider):
                    result = deepseek_hooks.optional_params(
                        params, model=model, custom_llm_provider=provider
                    )
                    self.assertNotIn("chat_template_kwargs", result)

    async def test_count_uses_local_tokenizer_and_low_template_with_tools(self):
        deployment = {"litellm_params": {
            "model": "anthropic/deepseek-v4.1-flash", "api_key": "test-only",
            "extra_body": {"chat_template_kwargs": {"reasoning_effort": "low"}},
        }}
        response = httpx.Response(200, json={"input_tokens": 37},
                                  request=httpx.Request("POST", "http://localhost"))
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.post.return_value = response
        with patch.dict(os.environ, LITELLM_ANTHROPIC_MODEL="anthropic/deepseek-v4.1-flash",
                        LITELLM_BACKEND_BASE="http://127.0.0.1:8080"), \
                patch.object(deepseek_hooks.httpx, "AsyncClient", return_value=client):
            result = await deepseek_hooks.count_tokens(
                None, "deepseek-v4.1-flash", [{"role": "user", "content": "OK"}],
                None, deployment, "deepseek-v4.1-flash-fast", tools=[], system="Be concise.")
        self.assertEqual(result.total_tokens, 37)
        args, kwargs = client.post.call_args
        self.assertEqual(args[0], "http://127.0.0.1:8080/v1/messages/count_tokens")
        self.assertEqual(kwargs["json"]["chat_template_kwargs"], {"reasoning_effort": "low"})
        self.assertEqual(kwargs["json"]["system"], "Be concise.")
        self.assertEqual(kwargs["json"]["tools"], [])

    async def test_high_alias_keeps_original_counter(self):
        original = AsyncMock(return_value="unchanged")
        with patch.object(deepseek_hooks, "_original", original):
            result = await deepseek_hooks.count_tokens(
                None, "deepseek-v4.1-flash", [], None, {}, "deepseek-v4.1-flash-high")
        self.assertEqual(result, "unchanged")
        original.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
