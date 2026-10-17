"""Process-death and observer-failure regressions from the interrupted long run."""
import importlib.util
import json
from pathlib import Path
import selectors
import subprocess
import sys
import threading

import pytest

SOURCE=Path(__file__).parents[1]/'bench/sglang_glm_prefill.py'
spec=importlib.util.spec_from_file_location('prefill_recovery',SOURCE)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_sigkill_preserves_started_request_and_partial_stream_without_a_pass(tmp_path):
    # Kill a real child between streamed events. Its finally blocks cannot run,
    # reproducing the exact evidence gap of a dead benchmark controller.
    code='''import importlib.util,sys,time
from pathlib import Path
spec=importlib.util.spec_from_file_location("prefill",sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Response:
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def __iter__(self):
  yield b'data: {"meta_info":{"completion_tokens":1}}\\n'
  print("STREAM_SAVED",flush=True)
  time.sleep(60)
m.urllib.request.urlopen=lambda *args,**kwargs:Response()
m.request("http://unused",{"input_ids":[1],"stream":True},Path(sys.argv[2]))
'''
    stem=tmp_path/'attempt'
    proc=subprocess.Popen([sys.executable,'-c',code,str(SOURCE),str(stem)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout,selectors.EVENT_READ)
            assert selector.select(timeout=5),'Child did not reach the streamed event'
            assert proc.stdout.readline().strip()=='STREAM_SAVED'
        proc.kill();proc.wait(timeout=5)
        marker=json.loads(stem.with_suffix('.attempt.json').read_text())
        assert marker['status']=='started' and marker['boot_id']
        assert json.loads(stem.with_suffix('.request.json').read_text())['input_ids']==[1]
        assert 'completion_tokens' in stem.with_suffix('.sse').read_text()
        assert not stem.with_suffix('.response.json').exists()
        assert not (tmp_path/'summary.json').exists()
        with pytest.raises(RuntimeError,match='previously started'):
            mod.request('http://unused',{},stem)
    finally:
        if proc.poll() is None:proc.kill();proc.wait(timeout=5)
        proc.stdout.close();proc.stderr.close()


def test_observer_failure_retains_prior_sample_and_cannot_silently_disappear(tmp_path):
    samples=[];errors=[];done=threading.Event();calls=0
    def read():
        nonlocal calls
        calls+=1
        if calls==2:raise OSError('PCI sysfs became unreadable')
        return {'paths':{'0000:e3:00.0':['0000:e3:00.0']},'devices':{}}
    path=tmp_path/'loaded-links.journal.jsonl'
    worker=threading.Thread(target=mod.watch_links,args=(read,path,done,samples,errors,'frozen-boot'))
    worker.start();worker.join(timeout=3)
    assert not worker.is_alive() and done.is_set()
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==len(samples)==1 and rows[0]['links']==samples[0]
    assert rows[0]['boot_id']=='frozen-boot' and rows[0]['monotonic_seconds']>0
    assert errors and 'PCI sysfs became unreadable' in errors[0]
