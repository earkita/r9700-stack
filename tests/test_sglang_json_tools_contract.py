"""CPU checks for a secret that is available only through the actual tool result."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

bench=Path(__file__).parents[1]/"bench";sys.path.insert(0,str(bench))
spec=importlib.util.spec_from_file_location("json_tools_contract",bench/"sglang_glm_json_tools.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def call():
    return dict(content="",reasoning="Read the actual value",finish_reason="tool_calls",tool_calls=[
        dict(id="read1",name="read_file",arguments='{"path":"note.txt"}')])


@pytest.mark.parametrize("kind",["json_object","json_schema"])
@pytest.mark.parametrize("streamed",[False,True])
@pytest.mark.parametrize("thinking",[False,True])
def test_schema_stays_on_both_auto_turns_and_secret_only_arrives_in_tool_result(tmp_path,kind,streamed,thinking):
    case=dict(id="case",format=kind,stream=streamed,thinking=thinking,seed=1234,secret="unknown-secret-a6f5")
    bodies=[]
    def transport(base,body,stem):
        bodies.append(copy.deepcopy(body))
        assert body["tool_choice"]=="auto" and body["response_format"]["type"]==kind
        assert body["chat_template_kwargs"]["enable_thinking"]==thinking
        if len(bodies)==1:
            assert case["secret"] not in json.dumps(body)
            return call()
        assert body["response_format"]==bodies[0]["response_format"]
        assert body["messages"][-2]["reasoning_content"]=="Read the actual value"
        result=body["messages"][-1]
        assert result["role"]=="tool" and result["tool_call_id"]=="read1"
        return dict(content=result["content"],reasoning="",tool_calls=[],finish_reason="stop")
    result=mod.attempt(dict(base="base",model="model",cache_salt="fixed"),case,tmp_path,transport)
    assert result["status"]=="pass" and result["tool_calls"]==1 and len(bodies)==2
    assert "wall_seconds" in result


@pytest.mark.parametrize("failure,expected",[
    ("missing","missing_tool_call"),("loop","looped_tool_call"),
    ("wrong","wrong_completed"),("initial_length","incomplete"),
    ("final_length","incomplete"),("bad_args","contract_failure"),
    ("duplicate_json","contract_failure"),("abort","error")])
def test_failure_cannot_pass_or_be_retried_until_it_looks_correct(tmp_path,failure,expected):
    case=dict(id="case",format="json_schema",stream=True,thinking=True,seed=1234,secret="secret")
    requests=[]
    def transport(base,body,stem):
        requests.append(body)
        if len(requests)==1:
            if failure=="missing":return dict(content='{"word":"guessed"}',tool_calls=[],finish_reason="stop")
            if failure=="initial_length":return dict(content="unfinished",tool_calls=[],finish_reason="length")
            if failure=="bad_args":
                response=call();response["tool_calls"][0]["arguments"]='{"path":"wrong.txt"}';return response
            return call()
        if failure=="loop":return call()
        if failure=="final_length":return dict(content="{",tool_calls=[],finish_reason="length")
        if failure=="abort":return dict(content="",tool_calls=[],finish_reason="abort")
        content='{"word":"wrong"}' if failure=="wrong" else '{"word":"wrong","word":"secret"}'
        return dict(content=content,tool_calls=[],finish_reason="stop")
    result=mod.attempt(dict(base="base",model="model",cache_salt="fixed"),case,tmp_path,transport)
    assert result["status"]==expected and "wall_seconds" in result
    assert len(requests)<=2
