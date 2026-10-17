"""CPU-only concurrency and measurement gates for the planned C4 experiment."""
import copy
import json
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/"bench"))
import sglang_glm_interference as mod


LANES=("short0","short1","short2","long")


def bodies():
    return {name:dict(rid="owned-"+name,input_ids=[1]*8,stream=True,
                     sampling_params=dict(ignore_eos=False,max_new_tokens=20))
            for name in LANES}


def test_long_arrives_only_after_three_first_tokens_and_generation_is_joined(tmp_path):
    long_sent=threading.Event();active=set();lock=threading.Lock();seen=[]
    def transport(body,stem,on_send,on_first):
        lane=stem.name
        with lock:active.add(lane)
        on_send()
        if lane=="long":
            with lock:assert active==set(LANES)
            long_sent.set()
        else:
            on_first()
            assert long_sent.wait(2)
        with lock:active.remove(lane);seen.append(lane)
        return dict(lane=lane)
    def forbidden_abort(ids):raise AssertionError("Successful cohort must not abort")
    result=mod.run_cohort(bodies(),tmp_path,transport,forbidden_abort,first_timeout=2)
    assert result["status"]=="responses_saved" and set(seen)==set(LANES) and not active
    events=result["arrival_events"]
    long_index=next(i for i,e in enumerate(events) if e["kind"]=="long_send")
    assert {e["lane"] for e in events[:long_index] if e["kind"]=="first"}==set(LANES[:3])
    assert not any(e["kind"]=="finish" for e in events[:long_index])
    assert json.loads((tmp_path/"cohort-transport.json").read_text())["status"]=="responses_saved"


@pytest.mark.parametrize("failure",["early_short","long_failure"])
def test_failure_aborts_only_owned_ids_and_joins_other_lanes(tmp_path,failure):
    aborted=threading.Event();calls=[];finished=set();lock=threading.Lock()
    def abort_owned(ids):calls.append(ids);aborted.set()
    def transport(body,stem,on_send,on_first):
        lane=stem.name
        try:
            on_send()
            if failure=="early_short" and lane=="short0":return {}
            on_first()
            if lane=="long":raise RuntimeError("failed long request")
            assert aborted.wait(2),"Error was not observed while other lanes were live"
            raise RuntimeError("owned request aborted")
        finally:
            with lock:finished.add(lane)
    result=mod.run_cohort(bodies(),tmp_path,transport,abort_owned,first_timeout=2)
    assert result["status"]=="error" and len(calls)==1
    assert set(calls[0])=={"owned-"+name for name in LANES}
    assert finished==set(LANES if failure=="long_failure" else LANES[:3])
    assert result["active_short_at_long_send"]==(failure=="long_failure")


def test_first_token_barrier_times_out_without_retrying(tmp_path):
    aborted=threading.Event();calls=[]
    def transport(body,stem,on_send,on_first):
        calls.append(stem.name);on_send();assert aborted.wait(2);return {}
    result=mod.run_cohort(bodies(),tmp_path,transport,lambda ids:aborted.set(),first_timeout=.02)
    assert result["status"]=="error" and sorted(calls)==list(LANES[:3])
    assert "TimeoutError" in result["errors"]["cohort"]


def test_long_send_rechecks_short_still_active_after_first_token_barrier():
    gate=mod.ArrivalGate(LANES[:3])
    for lane in LANES[:3]:gate.event("first",lane)
    gate.wait_for_first_tokens(.1)
    gate.event("finish","short1")
    with pytest.raises(RuntimeError,match="not active"):gate.event("long_send","long")


def test_duplicate_owned_request_id_rejected_before_any_transport(tmp_path):
    requests=bodies();requests["long"]["rid"]=requests["short0"]["rid"]
    with pytest.raises(ValueError,match="distinct owned"):
        mod.run_cohort(requests,tmp_path,None,None)
    assert not list(tmp_path.iterdir())


class Response:
    def __init__(self,packets):self.packets=packets
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def __iter__(self):
        return iter([("data: "+json.dumps(packet)+"\n").encode() for packet in self.packets])


def native_packets(body):
    meta=dict(id=body["rid"],prompt_tokens=8,num_retractions=0,finish_reason={"type":"stop"})
    return [dict(text="generated",meta_info=meta|dict(completion_tokens=count)) for count in (4,8,10)]


def test_native_decode_excludes_ttft_and_entire_first_speculative_burst(tmp_path,monkeypatch):
    body=bodies()["short0"];clock=iter([0.,2.,3.,5.,6.]);wall=iter([100.,101.,103.]);callbacks=[]
    monkeypatch.setattr(mod.time,"monotonic",lambda:next(clock))
    monkeypatch.setattr(mod.time,"time",lambda:next(wall))
    monkeypatch.setattr(mod.urllib.request,"urlopen",lambda *a,**k:Response(native_packets(body)))
    row=mod.stream_native("http://unused",body,tmp_path/"short0",lambda:callbacks.append("send"),lambda:callbacks.append("first"))
    assert callbacks==["send","first"]
    assert row["metrics"]["first_update_tokens"]==4 and row["metrics"]["post_first_update_tokens"]==6
    assert row["metrics"]["ttft_seconds"]==2 and row["metrics"]["decode_seconds"]==3
    assert row["metrics"]["decode_tps"]==2 and row["wall_seconds"]==6
    assert row["latency"]["post_first_update_tokens"]==6
    assert row["wall_events"]==[(100.,4),(101.,8),(103.,10)]
    assert row["completion"]["natural_eos"] is True
    assert json.loads((tmp_path/"short0.attempt.json").read_text())["status"]=="response_saved"
    with pytest.raises(RuntimeError,match="overwrite"):
        mod.stream_native("http://unused",body,tmp_path/"short0",None,None)


@pytest.mark.parametrize("failure",["id","context","decrease","abort","retraction","final_count"])
def test_bad_native_counters_or_runtime_cannot_qualify(tmp_path,monkeypatch,failure):
    body=bodies()["short0"];packets=native_packets(body)
    if failure=="id":packets[-1]["meta_info"]["id"]="another-session"
    if failure=="context":packets[-1]["meta_info"]["prompt_tokens"]=9
    if failure=="decrease":packets[-1]["meta_info"]["completion_tokens"]=3
    if failure=="abort":packets[-1]["meta_info"]["finish_reason"]={"type":"abort"}
    if failure=="retraction":packets[-1]["meta_info"]["num_retractions"]=1
    if failure=="final_count":packets[-1]["meta_info"]["completion_tokens"]=True
    monkeypatch.setattr(mod.urllib.request,"urlopen",lambda *a,**k:Response(packets))
    with pytest.raises((RuntimeError,ValueError)):
        mod.stream_native("http://unused",body,tmp_path/"short0",lambda:None,lambda:None)
    assert json.loads((tmp_path/"short0.attempt.json").read_text())["status"]=="error"
    assert (tmp_path/"short0.sse").exists()


def test_send_barrier_failure_is_durably_saved_before_http(tmp_path,monkeypatch):
    def reject():raise RuntimeError("short lane already finished")
    def forbidden(*a,**k):raise AssertionError("HTTP must not run")
    monkeypatch.setattr(mod.urllib.request,"urlopen",forbidden)
    with pytest.raises(RuntimeError,match="already finished"):
        mod.stream_native("http://unused",bodies()["short0"],tmp_path/"short0",reject,lambda:None)
    assert json.loads((tmp_path/"short0.attempt.json").read_text())["status"]=="error"


def test_scheduler_overlap_distinguishes_progress_and_starvation():
    results={name:dict(wall_events=[(99.,4),(100.5,8),(101.5,12),(104.,15)],latency={}) for name in LANES[:3]}
    results["short1"]["wall_events"]=[(99.,4),(104.,12)]
    results["short2"]["wall_events"]=[(95.,4),(96.,8)]
    results["long"]=dict(meta_info=dict(forward_entry_time=100.,prefill_finished_time=102.))
    report=mod.scheduler_interference(results)
    assert report["scheduler_prefill_wall_seconds"]==2
    assert report["short"]["short0"]["updates_during_prefill"]==2
    assert report["short"]["short0"]["tokens_between_observed_updates_during_prefill"]==4
    assert report["short"]["short1"]["updates_during_prefill"]==0
    assert report["short"]["short1"]["max_gap_touching_prefill_ms"]==5000
    assert report["short"]["short2"]["arrival_window_overlaps_prefill"] is False
    assert report["short"]["short2"]["max_gap_touching_prefill_ms"] is None


def test_c4_speculative_state_is_checked_for_both_comparison_legs():
    mtp={name:dict(meta_info=dict(spec_verify_ct=3)) for name in LANES}
    plain={name:dict(meta_info={}) for name in LANES}
    mod.require_speculation(mtp,"EAGLE")
    mod.require_speculation(plain,None)
    with pytest.raises(RuntimeError,match="not established"):
        mod.require_speculation(plain,"EAGLE")
    with pytest.raises(RuntimeError,match="Unexpected MTP"):
        mod.require_speculation(mtp,None)


@pytest.mark.parametrize("times",[[(100.,4),(99.,8)],[(100.,4),(101.,3)],[(float("nan"),4)],[(100.,True)]])
def test_wall_clock_or_counter_anomalies_are_unknown_not_speed_evidence(times):
    results={name:dict(wall_events=times,latency={}) for name in LANES[:3]}
    results["long"]=dict(meta_info=dict(forward_entry_time=100.,prefill_finished_time=102.))
    with pytest.raises(ValueError,match="invalid"):mod.scheduler_interference(results)


def test_independent_tag_checks_and_truncations_are_not_counted_as_completed(tmp_path):
    cases={name:dict(session_tag="private-"+name) for name in LANES}
    transport=dict(status="responses_saved",results={name:dict(completion=dict(natural_eos=True)) for name in LANES})
    for name in LANES:
        (tmp_path/(name+".response.json")).write_text(json.dumps(dict(text="</think>def dependency_waves(tasks): return []\nSESSION_TAG="+repr(cases[name]["session_tag"]))))
    transport["results"]["short1"]["completion"]["natural_eos"]=False
    seen=[]
    def evaluator(code,image):
        seen.append(code)
        assert "assert globals().get('SESSION_TAG') == 'private-" in code
        return dict(exit_code=0 if "private-long" not in code else 1,stdout='{"status":"pass","cases":169}',stderr="")
    result=mod.evaluate_cohort(transport,cases,tmp_path,"cpu-image",evaluator)
    assert result["status"]=="failed" and len(seen)==3
    assert result["lanes"]["short1"]["status"]=="incomplete"
    assert result["lanes"]["long"]["status"]=="wrong_completed"
    assert result["lanes"]["short0"]["status"]=="correct_completed"
    bad=copy.deepcopy(cases);bad["long"]["session_tag"]=bad["short0"]["session_tag"]
    with pytest.raises(ValueError,match="distinct session"):
        mod.evaluate_cohort(transport,bad,tmp_path,"cpu-image",evaluator)
