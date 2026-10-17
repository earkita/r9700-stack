import importlib.util
import json
import sys
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location("latency",Path(__file__).parents[1]/"bench/sglang_latency_report.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_speculative_burst_is_not_one_token_per_update():
    result=mod.latencies([(2.,3),(2.2,7),(2.4,11)])
    assert result["post_first_update_tokens"]==8
    assert result["client_visible_itl_ms"]["p50"]==0
    assert result["stream_update_gap_ms"]["p50"]==pytest.approx(200)
    assert result["mean_ms_per_post_first_update_token"]==pytest.approx(50)


def test_single_token_updates_and_invalid_counters():
    result=mod.latencies([(2.,1),(2.1,2),(2.2,3)])
    assert result["client_visible_itl_ms"]["p95"]==pytest.approx(100)
    with pytest.raises(ValueError):
        mod.latencies([(2.,3),(2.1,2)])


def test_report_preserves_both_excluded_requests_and_legacy_population(tmp_path, monkeypatch):
    source=tmp_path/'requests.jsonl'
    rows=[dict(index=i, excluded_population=i==0, excluded_warmup=i<2,
               metrics=dict(events=[(2.,3),(2.2,7)])) for i in range(3)]
    rows.append(dict(index=3, excluded_population=True,
                     metrics=dict(events=[(2.,3),(2.2,7)])))
    source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    output=tmp_path/'report.json'
    monkeypatch.setattr(sys,'argv',['report','--requests',str(source),'--out',str(output)])
    mod.main()
    report=json.loads(output.read_text())['requests']
    assert [row['excluded_warmup'] for row in report]==[True,True,False,True]
    assert [row['excluded_population'] for row in report]==[True,False,False,True]
    assert all(row['latency']['post_first_update_tokens']==4 for row in report)
