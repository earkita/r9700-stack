"""Pinned-image parser regression, CPU only; uses the checkpoint's reference encoder.

DEEPSEEK_TEST_MODEL=/models/DeepSeek-V4.1-Flash-Quark-MXFP4 pytest -q <this file>
No model weights or GPU allocations are needed. PR #58295 needs an adaptation
because single-pass parsing already processes content through _events_to_delta.
"""
import importlib.util
import copy
import json
import os
from pathlib import Path

import pytest

MODEL = os.environ.get('DEEPSEEK_TEST_MODEL')
if not MODEL:
    pytest.skip('Set DEEPSEEK_TEST_MODEL for pinned-image CPU parser tests', allow_module_level=True)

from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.deepseekv41_engine_tool_parser import DeepSeekV41EngineToolParser


@pytest.fixture(scope='module')
def runtime():
    model = Path(MODEL)
    spec = importlib.util.spec_from_file_location('checkpoint_encoding', model / 'encoding/encoding.py')
    encoding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoding)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tools = [dict(type='function', function=dict(name='lookup_file', parameters=dict(
        type='object', properties=dict(path=dict(type='string')), required=['path']))) ]
    request = ChatCompletionRequest(model='deepseek-v4.1-flash',
        messages=[dict(role='user', content='Read README.md')], tools=tools)
    return encoding, tokenizer, request


def stream(parser, tokenizer, request, text, chunk):
    ids = tokenizer.encode(text, add_special_tokens=False)
    previous, consumed, deltas = '', 0, []
    for end in list(range(chunk, len(ids), chunk)) + [len(ids)]:
        current = tokenizer.decode(ids[:end], skip_special_tokens=False)
        if current.endswith('\ufffd'):
            continue
        delta = parser.extract_tool_calls_streaming(previous, current,
            current[len(previous):], ids[:consumed], ids[:end], ids[consumed:end], request)
        previous, consumed = current, end
        if delta:
            deltas.append(delta)
    final = parser.finish_streaming()
    if final:
        deltas.append(final)
    return deltas


@pytest.mark.parametrize('content', ['Let me check.', 'Let me check.\n', 'Let me check.\n\n', ''])
@pytest.mark.parametrize('chunk', [None, 1, 7, 10000])
def test_reference_tool_roundtrip(runtime, content, chunk):
    encoding, tokenizer, request = runtime
    call = dict(role='assistant', content=content, tool_calls=[dict(id='call_1',
        type='function', function=dict(name='lookup_file', arguments='{"path":"README.md"}'))])
    text = encoding.encode_messages([dict(role='user', content='Read README.md'), call],
        thinking_mode='chat').split('<｜Assistant｜>', 1)[1].removeprefix('</think>').removesuffix(encoding.eos_token)
    parser = DeepSeekV41EngineToolParser(tokenizer, tools=request.tools)
    if chunk is None:
        result = parser.extract_tool_calls(text, request)
        assert (result.content or '') == content
        assert result.tool_calls[0].function.name == 'lookup_file'
        assert json.loads(result.tool_calls[0].function.arguments) == {'path': 'README.md'}
    else:
        deltas = stream(parser, tokenizer, request, text, chunk)
        assert ''.join(d.content or '' for d in deltas) == content
        calls = {}
        for delta in deltas:
            for tc in delta.tool_calls or []:
                value = calls.setdefault(tc.index, dict(name='', arguments=''))
                if tc.function:
                    value['name'] += tc.function.name or ''
                    value['arguments'] += tc.function.arguments or ''
        assert len(calls) == 1 and calls[0]['name'] == 'lookup_file'
        assert json.loads(calls[0]['arguments']) == {'path': 'README.md'}


@pytest.mark.parametrize('text', ['Done.\n', 'Done.\n\n', 'First.\n\nSecond.\n'])
@pytest.mark.parametrize('chunk', [1, 7, 10000])
def test_plain_text_keeps_whitespace(runtime, text, chunk):
    _, tokenizer, request = runtime
    parser = DeepSeekV41EngineToolParser(tokenizer, tools=request.tools)
    deltas = stream(parser, tokenizer, request, text, chunk)
    assert ''.join(d.content or '' for d in deltas) == text
    assert not any(d.tool_calls for d in deltas)


@pytest.mark.parametrize('effort', ['low', 'high', 'max', 1, 75, 100])
@pytest.mark.parametrize('thinking', [False, True])
@pytest.mark.parametrize('with_tools', [False, True])
def test_prompt_matches_checkpoint_reference(runtime, effort, thinking, with_tools):
    """Compare the actual vLLM wrapper with the independent checkpoint encoder.

    Include a completed tool turn and reasoning: checking only the first user
    prompt would miss context loss when the client sends back tool results.
    """
    from vllm.tokenizers.deepseek_v41 import get_deepseek_v41_tokenizer

    encoding, tokenizer, request = runtime
    messages = [dict(role='system', content='Use the supplied information.'),
                dict(role='user', content='Read README.md')]
    tools = [tool.model_dump(exclude_none=True) for tool in request.tools] if with_tools else None
    if with_tools:
        messages.extend([
            dict(role='assistant', content='Checking.',
                 reasoning_content='I need the file before answering.',
                 tool_calls=[dict(id='call_1', type='function', function=dict(
                     name='lookup_file', arguments='{"path":"README.md"}'))]),
            dict(role='tool', tool_call_id='call_1', content='Project: ORION.'),
        ])
    else:
        messages.extend([
            dict(role='assistant', reasoning_content='An earlier thought.', content='Ready.'),
            dict(role='user', content='Now answer in Polish: project ORION.'),
        ])
    reference_messages = copy.deepcopy(messages)
    if tools:
        reference_messages[0]['tools'] = tools
    expected = encoding.encode_messages(reference_messages,
        thinking_mode='thinking' if thinking else 'chat',
        reasoning_effort=effort, drop_thinking=True)
    actual = get_deepseek_v41_tokenizer(tokenizer).apply_chat_template(
        messages, tools=tools, tokenize=False, enable_thinking=thinking,
        reasoning_effort=effort)
    assert actual == expected
    if with_tools and thinking:
        assert 'I need the file before answering.' in actual
