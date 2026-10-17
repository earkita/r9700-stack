import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location("telemetry",Path(__file__).parents[1]/"bench/gpu_telemetry.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_smi_id_does_not_imply_hip_ordinal_and_missing_metric_stays_unknown():
    raw=dict(gpu_data=[dict(gpu=0,usage="N/A"),dict(gpu=4,usage={"gfx":100})])
    devices=[dict(gpu=0,bdf="0000:c3:00.0"),dict(gpu=4,bdf="0000:e3:00.0")]
    mapping=[dict(hip=0,worker_rank=0,bdf="0000:e3:00.0"),dict(hip=4,worker_rank=4,bdf="0000:c3:00.0")]
    result=mod.annotate(raw,devices,mapping)
    assert result[0]["smi_id"]==4 and result[0]["hip"]==0
    assert result[1]["metrics"]["usage"]=="N/A"


def test_missing_participating_gpu_cannot_pass():
    with pytest.raises(ValueError,match="every participating"):
        mod.annotate(dict(gpu_data=[]),[dict(gpu=0,bdf="0000:c3:00.0")],[dict(hip=0,bdf="0000:c3:00.0")])


def test_irq_totals_follow_bdf_across_cpu_columns():
    interrupts = """ 362: 10 20 30 IR-PCI-MSI-0000:e3:00.0 0-edge amdgpu
 366: 2 3 5 IR-PCI-MSI-0000:c3:00.0 0-edge amdgpu
 999: 7 8 9 IR-PCI-MSI-0000:cc:00.0 0-edge amdgpu
"""
    mapping = [dict(hip=0, bdf="0000:e3:00.0"), dict(hip=4, bdf="0000:c3:00.0")]
    assert mod.irq_counts(interrupts, mapping) == {"0000:c3:00.0": 10, "0000:e3:00.0": 60}


def test_irq_missing_is_unknown_and_duplicate_is_rejected():
    mapping = [dict(hip=4, bdf="0000:c3:00.0")]
    assert mod.irq_counts("", mapping) == {"0000:c3:00.0": None}
    line = " 366: 1 2 IR-PCI-MSI-0000:c3:00.0 0-edge amdgpu\n"
    with pytest.raises(ValueError, match="Duplicate"):
        mod.irq_counts(line + line, mapping)


def test_irq_sample_keeps_errors_explicit():
    mapping = [dict(hip=4, bdf="0000:c3:00.0")]
    sample = mod.irq_sample(mapping, lambda: " 366: 2 3 IR-PCI-MSI-0000:c3:00.0 0-edge amdgpu\n")
    assert sample["status"] == "complete"
    assert sample["irq_by_bdf"] == {"0000:c3:00.0": 5}
    bad = mod.irq_sample(mapping, lambda: " 366: bad IR-PCI-MSI-0000:c3:00.0 0-edge amdgpu\n")
    assert bad["status"] == "error" and "ValueError" in bad["error"]


def test_irq_sampler_writes_independent_stream(tmp_path, monkeypatch):
    samples = iter([{"status": "complete", "irq_by_bdf": {"0000:c3:00.0": value}}
                    for value in (10, 11, 11)])
    monkeypatch.setattr(mod, "irq_sample", lambda mapping: next(samples))

    class StopAfterThree:
        reads = 0

        def is_set(self):
            self.reads += 1
            return self.reads > 3

        def wait(self, interval):
            pass

    mod.irq_sampler(tmp_path, [dict(hip=4, bdf="0000:c3:00.0")], StopAfterThree(), 1.)
    rows = [__import__("json").loads(line) for line in (tmp_path / "irq-samples.jsonl").read_text().splitlines()]
    assert [row["irq_by_bdf"]["0000:c3:00.0"] for row in rows] == [10, 11, 11]
