"""Repository regressions, virtual file isolation and independent test contracts."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/'bench'))
import sglang_glm_agent_task as agent
import sglang_repository_tasks as tasks


def reference(task_id):
    task=tasks.task_bundle(task_id)
    path=ROOT/task['path']
    name='native_cache_source' if task_id=='cache_replay_evidence' else 'counter_metrics'
    return tasks._function_source(path,name)


def cpu_evaluate(task_id, code):
    # Only fixture-owned known source runs here. Runtime model-generated source
    # uses the separate CPU-only Docker evaluator, never the host Python.
    result=subprocess.run([sys.executable,'-I','-S','-c',tasks.HARNESS],
                          input=json.dumps(dict(task=task_id,code=code)),text=True,capture_output=True,timeout=5)
    return dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)


@pytest.mark.parametrize('task_id',['cache_replay_evidence','decode_counter_accounting'])
def test_reproduced_regression_fails_but_actual_repository_reference_passes(task_id):
    bundle=tasks.task_bundle(task_id)
    assert cpu_evaluate(task_id,bundle['initial_code'])['exit_code']!=0
    passed=cpu_evaluate(task_id,reference(task_id))
    assert passed['exit_code']==0 and json.loads(passed['stdout'])==dict(status='pass',cases=bundle['cases'])
    assert bundle['source_hashes'] and 'def test_' in next(iter(bundle['readonly_files'].values()))


def test_representative_tests_are_required_readonly_and_latest_version_is_retested():
    bundle=tasks.task_bundle('cache_replay_evidence')
    def evaluator(code,image):return cpu_evaluate(bundle['id'],code)
    workspace=agent.Workspace('unused',evaluator,bundle)
    path=bundle['path'];test_path=next(iter(bundle['readonly_files']))
    workspace.call('read_file',{'path':path})
    assert 'error' in workspace.call('write_file',{'path':path,'content':reference(bundle['id'])})[0]
    workspace.call('read_file',{'path':test_path})
    assert 'error' in workspace.call('write_file',{'path':test_path,'content':'skip all tests'})[0]
    assert 'error' in workspace.call('read_file',{'path':'/etc/passwd'})[0]
    workspace.call('write_file',{'path':path,'content':reference(bundle['id'])})
    assert workspace.call('run_tests',{})[0]==dict(passed=True,cases=19)
    assert workspace.complete('{"status":"complete","tests_passed":19}')
    workspace.call('write_file',{'path':path,'content':bundle['initial_code']})
    assert not workspace.ready_to_complete()
    assert workspace.call('run_tests',{})[0]['passed'] is False
    assert not workspace.complete('{"status":"complete","tests_passed":19}')


def test_task_counts_cannot_be_borrowed_from_an_unrelated_evaluator():
    task=tasks.task_bundle('decode_counter_accounting')
    wrong=lambda *_:dict(exit_code=0,stdout='{"status":"pass","cases":169}',stderr='')
    workspace=agent.Workspace('unused',wrong,task)
    for path in task['required_reads']:workspace.call('read_file',{'path':path})
    workspace.call('write_file',{'path':task['path'],'content':reference(task['id'])})
    assert workspace.call('run_tests',{})[0]['passed'] is False
    assert not workspace.complete('{"status":"complete","tests_passed":11}')


@pytest.mark.parametrize('obey_final_tool_choice',[True,False])
def test_full_repository_tool_roundtrip_uses_task_paths_and_gates_final_schema(tmp_path,obey_final_tool_choice):
    task=tasks.task_bundle('decode_counter_accounting');target=task['path'];test_path=task['required_reads'][1]
    calls=[('read_file',{'path':target}),('read_file',{'path':test_path}),
           ('write_file',{'path':target,'content':reference(task['id'])}),('run_tests',{})]
    bodies=[]
    def transport(base,body,stem):
        index=len(bodies);bodies.append(body)
        if index<len(calls):
            name,args=calls[index]
            response=dict(content='',reasoning='',finish_reason='tool_calls',tool_calls=[dict(id=f'c{index}',name=name,arguments=json.dumps(args))])
        else:
            response=dict(content='{"status":"complete","tests_passed":11}',reasoning='',finish_reason='stop',tool_calls=[])
            if not obey_final_tool_choice:
                response=dict(content='',reasoning='',finish_reason='tool_calls',
                              tool_calls=[dict(id='unexpected',name='read_file',arguments=json.dumps(dict(path=target)))])
        return response|{'usage':{'prompt_tokens':8192+index*100,'completion_tokens':10}},{'wall_seconds':1.}
    plan=dict(image_id='unused',max_turns=8,total_output_tokens=4096,per_turn_tokens=2048,
              max_tool_calls=8,max_tool_feedback_bytes=3000,model='model',base='base',cache_salt='fixed',
              sampling={},context_length=20000,final_json_schema=True)
    row=agent.attempt(plan,dict(messages=[],input_tokens=8192),1234,tmp_path/'attempt',transport,
                      lambda code,image:cpu_evaluate(task['id'],code),repository_task=task)
    assert row['status']==('correct_completed' if obey_final_tool_choice else 'protocol_or_answer_failure')
    assert row['task_id']==task['id'] and row['tool_calls']==4
    if not obey_final_tool_choice:
        assert 'tools were disabled' in row['error']
    assert all('response_format' not in body for body in bodies[:-1])
    assert bodies[-1]['tool_choice']=='none'
    assert bodies[-1]['response_format']['json_schema']['schema']['properties']['tests_passed']['enum']==[11]
    assert bodies[0]['tools'][0]['function']['parameters']['properties']['path']['enum']==[target,test_path]


def test_virtual_repository_sources_never_execute_when_task_is_selected():
    with pytest.raises(ValueError,match='Unknown'):
        tasks.task_bundle('../../etc/passwd')


def test_evaluator_timeout_retains_partial_output_and_cleans_up_its_own_container(monkeypatch):
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        if len(calls)==1:
            raise subprocess.TimeoutExpired(command,15,output=b'partial output')
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(tasks.subprocess,'run',run)
    result=tasks.evaluate_repository('decode_counter_accounting','while True: pass','fixed-image')
    assert result['exit_code'] is None and result['stdout']=='partial output' and 'timeout' in result['stderr']
    json.dumps(result)
    original_name=calls[0][calls[0].index('--name')+1]
    assert calls[1]==['docker','rm','-f',original_name]
