import importlib.util
import json
from pathlib import Path
import subprocess
import sys

spec = importlib.util.spec_from_file_location("trace",Path(__file__).parents[1]/"profiling/sglang_prefill_trace.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def span(name,start,duration,tid=1):
    return dict(ph="X",cat="user_annotation",name="r9700::"+name,ts=start,dur=duration,pid=7,tid=tid)


def launch(correlation,start,tid=1):
    return dict(cat="cuda_runtime",ts=start,dur=1,pid=7,tid=tid,args=dict(correlation=correlation))


def kernel(correlation,start,duration):
    return dict(cat="kernel",ts=start,dur=duration,args=dict(correlation=correlation))


def test_async_kernels_are_attributed_by_launch_not_gpu_timestamp():
    result = mod.summarize([span("cache.restore",0,10), launch(1,5),kernel(1,100,40)])
    cost = result["ranges"]["r9700::cache.restore"]
    assert cost["host_inclusive_us"] == 10
    assert cost["correlated_gpu_busy_us"] == 40
    assert cost["correlated_device_events"] == 1


def test_overlap_is_union_nested_ranges_inclusive_and_other_thread_excluded():
    result = mod.summarize([span("layer",0,20),span("moe",2,10),launch(1,4),launch(2,6),
                            launch(3,5,tid=2),kernel(1,100,10),kernel(2,105,20),kernel(3,140,40)])
    for name in ("layer","moe"):
        cost = result["ranges"]["r9700::"+name]
        assert cost["correlated_gpu_busy_us"] == 25
        assert cost["device_event_sum_us"] == 30
        assert cost["correlated_device_events"] == 2


def test_missing_and_ambiguous_correlation_is_unknown():
    result = mod.summarize([span("cache.lookup",0,10),launch(1,2),launch(1,4),kernel(1,20,3),kernel(2,30,4)])
    assert result["missing_or_ambiguous_launch_correlations"] == 2
    assert result["ranges"]["r9700::cache.lookup"]["correlated_gpu_busy_us"] is None


def test_cpu_only_trace_has_unknown_device_time_not_zero():
    result = mod.summarize([span("cache.lookup",0,10)])
    assert result["total_device_events"] == result["kernel_events"] == 0
    assert result["total_gpu_busy_us"] is None
    assert result["ranges"]["r9700::cache.lookup"]["host_inclusive_us"] == 10


def test_missing_gpu_rank_fails_closed_but_keeps_cpu_evidence(tmp_path):
    paths=[]
    for rank in range(8):
        path=tmp_path/f"cold8192-TP-{rank}.trace.json"
        events=[span("cache.lookup",0,10)]
        if rank != 7:events += [launch(1,5),kernel(1,100,40)]
        path.write_text(json.dumps(dict(baseTimeNanoseconds=123,traceEvents=events)))
        paths.append(str(path))
    out=tmp_path/"report.json"
    result=subprocess.run([sys.executable,mod.__file__,*paths,"--out",str(out)],capture_output=True,text=True)
    assert result.returncode != 0 and "ranks [7]" in result.stderr
    report=json.loads(out.read_text())
    assert report["status"] == "incomplete_missing_gpu_activity"
    assert report["missing_kernel_ranks"] == [7]
    assert report["ranks"][7]["total_gpu_busy_us"] is None
    assert report["cross_rank_wall_envelope_us"] is None
    # Host-only review must be explicit and cannot overwrite the first audit.
    cpu=tmp_path/"cpu-only.json"
    result=subprocess.run([sys.executable,mod.__file__,*paths,"--out",str(cpu),"--allow-cpu-only"],capture_output=True,text=True)
    assert result.returncode == 0
    assert json.loads(cpu.read_text())["status"] == "incomplete_missing_gpu_activity"
