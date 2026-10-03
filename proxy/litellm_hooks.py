"""Compatibility for the dedicated local proxy, isolated from the vLLM plugin.

LiteLLM 1.103.0 hardcodes api.anthropic.com in its count-tokens transport,
ignoring the deployment's api_base. This process serves one local backend;
route its Anthropic token counter there as well. Do not load this hook into
a proxy mixing local deployments and remote Anthropic accounts.
"""
import os
from urllib.parse import urlsplit

from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.anthropic.count_tokens.transformation import AnthropicCountTokensConfig


def local_count_endpoint(_self):
    base = os.environ["LITELLM_BACKEND_BASE"].rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("LITELLM_BACKEND_BASE must be an HTTP(S) backend URL")
    return base + "/v1/messages/count_tokens"


AnthropicCountTokensConfig.get_anthropic_count_tokens_endpoint = local_count_endpoint
local_backend = CustomLogger()
