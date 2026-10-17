"""CPU fixtures for task completion, actual token counters and tool budgets."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

bench=Path(__file__).parents[1]/"bench"
sys.path.insert(0,str(bench))
spec=importlib.util.spec_from_file_location("agent_task",bench/"sglang_glm_agent_task.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def passed(code,image):
    return {"exit_code":0,"stdout":json.dumps({"status":"pass","cases":169}),"stderr":""}


def test_self_report_and_zero_exit_cannot_pass_unexecuted_tests():
    ws=mod.Workspace("image",lambda *_:{"exit_code":0,"stdout":"","stderr":""})
    final='{"status":"complete","tests_passed":169}'
    assert not ws.complete(final)
    ws.call("read_file",{"path":"dependency.py"})
    ws.call("write_file",{"path":"dependency.py","content":"import sys; sys.exit(0)"})
    feedback,_=ws.call("run_tests",{})
    assert feedback["passed"] is False
    assert not ws.complete(final)


def test_latest_written_version_must_be_independently_tested():
    ws=mod.Workspace("image",passed)
    ws.call("read_file",{"path":"dependency.py"})
    ws.call("write_file",{"path":"dependency.py","content":"one"})
    ws.call("run_tests",{})
    assert ws.complete('{"status":"complete","tests_passed":169}')
    ws.call("write_file",{"path":"dependency.py","content":"two"})
    assert not ws.complete('{"status":"complete","tests_passed":169}')
    ws.call("run_tests",{})
    with pytest.raises(ValueError,match="Duplicate"):
        ws.complete('{"status":"complete","status":"complete","tests_passed":169}')


def test_virtual_tools_cannot_read_or_modify_other_paths():
    ws=mod.Workspace("image",passed)
    assert "error" in ws.call("read_file",{"path":"/etc/passwd"})[0]
    assert "error" in ws.call("write_file",{"path":"../dependency.py","content":"bad"})[0]
    assert "error" in ws.call("run_tests",{"shell":"true"})[0]
    assert ws.code==mod.INITIAL_CODE and ws.used==set()
    assert "error" in ws.call("write_file",{"path":"dependency.py","content":"replacement"})[0]
    assert ws.code==mod.INITIAL_CODE


def test_stream_decode_excludes_ttft_and_first_speculative_burst():
    observed=mod.stream_metrics([[0,0],[12,4],[12,4],[12.1,8],[12.2,10],[12.3,10]],{"completion_tokens":10})
    assert observed["ttft_seconds"]==12
    assert observed["post_first_update_tokens"]==6
    assert observed["decode_tps"]==pytest.approx(30)
    assert observed["events"]==[[12,4],[12.1,8],[12.2,10]]
    with pytest.raises(ValueError,match="decreased"):
        mod.stream_metrics([[1,3],[2,2]],{"completion_tokens":2})
    with pytest.raises(ValueError,match="inconsistent"):
        mod.stream_metrics([[1,2]],{"completion_tokens":4})


def test_complete_multitool_task_replays_results_and_checks_context(tmp_path):
    calls=[("read_file",{"path":"dependency.py"}),
           ("write_file",{"path":"dependency.py","content":"def dependency_waves(tasks): return []"}),
           ("run_tests",{})]
    bodies=[]
    def fake(base,body,stem):
        index=len(bodies);bodies.append(body)
        if index<3:
            name,args=calls[index]
            result={"finish_reason":"tool_calls","content":"","reasoning":"r",
                    "tool_calls":[{"id":f"id{index}","name":name,"arguments":json.dumps(args)}]}
        else:
            result={"finish_reason":"stop","content":'{"status":"complete","tests_passed":169}',
                    "reasoning":"r","tool_calls":[]}
        result["usage"]={"prompt_tokens":131072+index*100,"completion_tokens":10}
        return result,{"wall_seconds":1.0}
    plan=dict(image_id="image",max_turns=8,total_output_tokens=4096,per_turn_tokens=2048,
              max_tool_calls=8,max_tool_feedback_bytes=3000,model="model",base="base",
              cache_salt="fixed",sampling={"temperature":1.0,"top_p":.95,"top_k":-1},
              context_length=270592)
    result=mod.attempt(plan,{"messages":[{"role":"user","content":"repair"}],"input_tokens":131072},1234,tmp_path/"attempt",fake,passed)
    assert result["status"]=="correct_completed" and result["tool_calls"]==3
    assert result["completion_tokens"]==40 and result["model_wall_seconds"]==4
    assert bodies[-1]["messages"][-1]["role"]=="tool"
    assert json.loads(bodies[-1]["messages"][-1]["content"])["passed"] is True
    assert bodies[-1]["seed"]==1237


def test_truncation_is_not_completed_answer(tmp_path):
    def truncated(*args):
        return {"usage":{"prompt_tokens":8,"completion_tokens":10},"finish_reason":"length"},{"wall_seconds":2}
    plan=dict(image_id="image",max_turns=8,total_output_tokens=10,per_turn_tokens=10,
              max_tool_calls=8,max_tool_feedback_bytes=3000,model="model",base="base",
              cache_salt="fixed",sampling={},context_length=100)
    result=mod.attempt(plan,{"messages":[],"input_tokens":8},1,tmp_path/"attempt",truncated,passed)
    assert result["status"]=="incomplete" and not result["final_version_tested"]


def test_parallel_tool_reply_is_not_silently_executed(tmp_path):
    def parallel(*args):
        call={"id":"id","name":"read_file","arguments":'{"path":"dependency.py"}'}
        return {"usage":{"prompt_tokens":8,"completion_tokens":10},"finish_reason":"tool_calls",
                "tool_calls":[call,dict(call,id="id2")]},{"wall_seconds":2}
    plan=dict(image_id="image",max_turns=8,total_output_tokens=10,per_turn_tokens=10,
              max_tool_calls=8,max_tool_feedback_bytes=3000,model="model",base="base",
              cache_salt="fixed",sampling={},context_length=100)
    result=mod.attempt(plan,{"messages":[],"input_tokens":8},1,tmp_path/"attempt",parallel,passed)
    assert result["status"]=="protocol_or_answer_failure" and result["tool_calls"]==0


def test_json_schema_is_enabled_only_after_independently_verified_latest_version(tmp_path):
    calls=[('read_file',{'path':'dependency.py'}),
           ('write_file',{'path':'dependency.py','content':'broken'}),
           ('run_tests',{}),
           ('write_file',{'path':'dependency.py','content':'repaired'}),
           ('run_tests',{})]
    bodies=[]
    def executor(code,image):
        return passed(code,image) if code=='repaired' else {'exit_code':1,'stdout':'','stderr':'AssertionError'}
    def transport(base,body,stem):
        index=len(bodies);bodies.append(body)
        if index<len(calls):
            name,args=calls[index]
            response=dict(content='',reasoning='',finish_reason='tool_calls',
                          tool_calls=[dict(id=f'call{index}',name=name,arguments=json.dumps(args))])
        else:
            response=dict(content='{"status":"complete","tests_passed":169}',reasoning='',
                          finish_reason='stop',tool_calls=[])
        return response|{'usage':{'prompt_tokens':8192+index*100,'completion_tokens':10}},{'wall_seconds':1.}
    plan=dict(image_id='image',max_turns=8,total_output_tokens=4096,per_turn_tokens=2048,
              max_tool_calls=8,max_tool_feedback_bytes=3000,model='model',base='base',
              cache_salt='fixed',sampling={},context_length=20000,final_json_schema=True)
    result=mod.attempt(plan,{'messages':[],'input_tokens':8192},1234,tmp_path/'attempt',transport,executor)
    assert result['status']=='correct_completed' and result['tool_calls']==5
    assert result['final_format_mode']=='guided_after_verified_tests'
    assert all('response_format' not in body and body['tool_choice']=='auto' for body in bodies[:-1])
    assert bodies[-1]['tool_choice']=='none' and bodies[-1]['response_format']==mod.FINAL_RESPONSE_FORMAT
    schema=bodies[-1]['response_format']['json_schema']['schema']
    assert schema['additionalProperties'] is False and schema['required']==['status','tests_passed']
