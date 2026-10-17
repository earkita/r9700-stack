"""CPU fixtures: lost calls, wrong tool IDs/results, truncation and round trips."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

bench = Path(__file__).parents[1]/"bench"
sys.path.insert(0,str(bench))
spec = importlib.util.spec_from_file_location("multitool",bench/"sglang_glm_multitool.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


def calls():
    return dict(finish_reason="tool_calls",content="",reasoning="inspect both",
        tool_calls=[dict(id="id-first",name="read_first",arguments="{}"),
                    dict(id="id-second",name="read_second",arguments="{}")])


@pytest.mark.parametrize("corrupt",["lost", "duplicate_id", "duplicate_name", "arguments", "partial"])
def test_corrupt_two_call_response_cannot_pass(corrupt):
    first = calls()
    if corrupt == "lost": first["tool_calls"].pop()
    if corrupt == "duplicate_id": first["tool_calls"][1]["id"] = "id-first"
    if corrupt == "duplicate_name": first["tool_calls"][1]["name"] = "read_first"
    if corrupt == "arguments": first["tool_calls"][1]["arguments"] = '{"value":25}'
    if corrupt == "partial": first["tool_calls"][1]["arguments"] = '{'
    with pytest.raises(ValueError): mod.tool_results(first,"tag")


def test_results_follow_actual_ids_and_call_order():
    first = calls(); first["tool_calls"].reverse()
    results = mod.tool_results(first,"unique")
    assert [r["tool_call_id"] for r in results] == ["id-second","id-first"]
    assert [json.loads(r["content"]) for r in results] == [
        {"value":25,"tag":"unique"},{"value":17,"tag":"unique"}]


@pytest.mark.parametrize("content",[
    '{"total":41,"tag":"tag"}', '{"total":42,"tag":"other"}',
    '{"total":42,"tag":"tag","extra":true}',
    '{"total":42,"total":41,"tag":"tag"}',
    '{"total":42.0,"tag":"tag"}', '{"total":NaN,"tag":"tag"}',
    'Result: {"total":42,"tag":"tag"}'])
def test_self_report_or_malformed_json_cannot_pass(content):
    with pytest.raises(ValueError):
        mod.check_final(dict(finish_reason="stop",content=content,tool_calls=[]),"tag")


@pytest.mark.parametrize("streamed",[False,True])
def test_round_trip_preserves_both_results_and_constrains_only_final(tmp_path,streamed):
    bodies=[]
    def transport(base,body,stem):
        bodies.append(copy.deepcopy(body))
        if len(bodies)==1:
            assert body["parallel_tool_calls"] is True
            assert body["tool_choice"]=="auto" and "response_format" not in body
            return calls()
        assert body["tool_choice"]=="none"
        assert body["response_format"]["json_schema"]["schema"]["properties"]["total"]=={"type":"integer"}
        assert body["messages"][-3]["reasoning_content"]=="inspect both"
        results=body["messages"][-2:]
        assert [r["tool_call_id"] for r in results]==["id-first","id-second"]
        values=[json.loads(r["content"]) for r in results]
        return dict(content=json.dumps(dict(total=sum(v["value"] for v in values),tag=values[0]["tag"])),
                    reasoning="",tool_calls=[],finish_reason="stop")
    result=mod.attempt(dict(model="model",base="base",cache_salt="fixed"),1234,streamed,tmp_path,transport)
    assert result["status"]=="pass" and result["tool_calls"]==2
    assert len(bodies)==2 and all(b["stream"]==streamed for b in bodies)


@pytest.mark.parametrize("finish, expected",[("length","incomplete"),("abort","error"),(None,"error")])
def test_truncation_and_runtime_abort_are_not_normal_contract_failures(tmp_path,finish,expected):
    invoked=[]
    def transport(base,body,stem):
        invoked.append(stem)
        return calls() | {"finish_reason":finish}
    result=mod.attempt(dict(model="model",base="base",cache_salt="fixed"),1234,True,tmp_path,transport)
    assert result["status"]==expected and len(invoked)==1


def test_nonstream_normalization_preserves_both_calls():
    source=calls()
    message=dict(content="",reasoning_content=source["reasoning"],tool_calls=[
        dict(id=c["id"],type="function",function=dict(name=c["name"],arguments=c["arguments"]))
        for c in source["tool_calls"]])
    response=mod.normalize(dict(choices=[dict(message=message,finish_reason="tool_calls")]),False)
    assert response["tool_calls"]==source["tool_calls"]
    assert response["reasoning"]==source["reasoning"]
