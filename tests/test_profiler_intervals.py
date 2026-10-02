"""Coverage must not double-count simultaneous GPU streams as busy time."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "torchprof_summary", Path(__file__).parents[1] / "profiling" / "torchprof-summary.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


@pytest.mark.parametrize("intervals,expected", [
    ([], (0, 0)),
    ([(2, 2), (4, 3)], (0, 0)),
    ([(20, 30), (0, 10)], (30, 20)),
    ([(0, 10), (5, 15)], (15, 15)),
    ([(0, 20), (5, 10), (25, 30)], (30, 25)),
    ([(0, 10), (10, 20)], (20, 20)),
])
def test_interval_coverage(intervals, expected):
    assert summary.interval_coverage(intervals) == expected


def test_trace_keeps_gpu_spans_separate(tmp_path, capsys):
    trace = tmp_path / "worker.json"
    trace.write_text(json.dumps({"traceEvents": [
        {"cat": "kernel", "name": "a", "ts": 0, "dur": 10, "args": {"device": 0}},
        {"cat": "kernel", "name": "a", "ts": 20, "dur": 10, "args": {"device": 0}},
        {"cat": "kernel", "name": "b", "ts": 1000, "dur": 20, "args": {"device": 1}},
    ]}))
    summary.main(str(trace), 2)
    lines = capsys.readouterr().out.splitlines()
    gpu0 = next(line for line in lines if line.startswith("GPU 0:"))
    gpu1 = next(line for line in lines if line.startswith("GPU 1:"))
    assert "2 kernel launches (1.0/step)" in gpu0 and "gaps 33.3%" in gpu0
    assert "1 kernel launches (0.5/step)" in gpu1 and "gaps 0.0%" in gpu1
