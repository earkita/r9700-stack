"""MiMo-only thinking and count_tokens adapters for LiteLLM 1.103.0.

The native Messages filter and token counter discard deployment template flags.
Forward them to vLLM so the fast alias generates and counts a non-thinking prefix.
Response bodies and tool streams are not modified.
"""

import importlib.metadata
import os

import httpx
from litellm.llms.anthropic.count_tokens.token_counter import AnthropicTokenCounter
from litellm.llms.anthropic.experimental_pass_through.messages.utils import (
    AnthropicMessagesRequestUtils,
)
from litellm.types.utils import TokenCountResponse
from litellm_hooks import local_backend, local_count_endpoint  # noqa: F401

if importlib.metadata.version("litellm") != "1.103.0":
    raise RuntimeError("Revalidate MiMo token counting before upgrading LiteLLM")

_original = AnthropicTokenCounter.count_tokens


async def count_tokens(
    self,
    model_to_use,
    messages,
    contents,
    deployment=None,
    request_model="",
    tools=None,
    system=None,
):
    params = (deployment or {}).get("litellm_params", {})
    kwargs = params.get("extra_body", {}).get("chat_template_kwargs", {})
    if (
        params.get("model") != os.environ.get("LITELLM_ANTHROPIC_MODEL")
        or not model_to_use.startswith("mimo-")
        or "enable_thinking" not in kwargs
    ):
        return await _original(
            self,
            model_to_use,
            messages,
            contents,
            deployment,
            request_model,
            tools,
            system,
        )
    body = dict(model=model_to_use, messages=messages, chat_template_kwargs=kwargs)
    if tools is not None:
        body["tools"] = tools
    if system is not None:
        body["system"] = system
    key = params["api_key"]
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            local_count_endpoint(None),
            json=body,
            headers={"x-api-key": key, "Authorization": "Bearer " + key},
        )
        response.raise_for_status()  # no fallback to an unrelated tokenizer
        result = response.json()
    return TokenCountResponse(
        total_tokens=result["input_tokens"],
        request_model=request_model,
        model_used=model_to_use,
        tokenizer_type="local_vllm_api",
        original_response=result,
    )


AnthropicTokenCounter.count_tokens = count_tokens

_optional_params = (
    AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param
)


def optional_params(params, *, model=None, drop_params=False, custom_llm_provider=None):
    result = _optional_params(
        params,
        model=model,
        drop_params=drop_params,
        custom_llm_provider=custom_llm_provider,
    )
    backend = os.environ.get("LITELLM_ANTHROPIC_MODEL", "").removeprefix("anthropic/")
    if (
        model == backend
        and model.startswith("mimo-")
        and custom_llm_provider == "anthropic"
    ):
        template = params.get("extra_body", {}).get("chat_template_kwargs", {})
        if "enable_thinking" in template:
            result["chat_template_kwargs"] = dict(template)
    return result


AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param = (
    staticmethod(optional_params)
)
