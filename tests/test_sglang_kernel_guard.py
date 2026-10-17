"""The journal guard must stop reuse before repeating a failed GPU series."""

import importlib.util
import json
from pathlib import Path
import queue
import sys
import threading
from types import SimpleNamespace

bench = Path(__file__).parents[1] / "bench"
sys.path.insert(0, str(bench))
spec = importlib.util.spec_from_file_location("kernel_guard", bench / "sglang_kernel_guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class FakeFollower:
    def __init__(self):
        self.lines = queue.Queue()
        self.stdout = self
        self.ended = False

    def __iter__(self):
        while True:
            line = self.lines.get(timeout=2)
            if line is None:
                return
            yield line

    def poll(self):
        return 0 if self.ended else None

    def terminate(self):
        self.ended = True
        self.lines.put(None)

    def wait(self, timeout=None):
        return 0


class FakeChild:
    def __init__(self):
        self.interrupted = threading.Event()

    def poll(self):
        return -2 if self.interrupted.is_set() else None

    def send_signal(self, sig):
        self.interrupted.set()

    def wait(self, timeout=None):
        assert self.interrupted.wait(2), "live MES did not interrupt the controller"
        return -2


MES = "Oct 16 21:25:17 kernel: amdgpu 0000:c3:00.0: MES(1) failed to respond to msg=INVALIDATE_TLBS\n"


def test_prior_mes_on_current_boot_blocks_new_controller(monkeypatch, tmp_path):
    follower = FakeFollower()
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **k: follower)
    monkeypatch.setattr(guard.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=MES))
    assert guard.run(["would-start-a-gpu-controller"], tmp_path / "evidence") == 86
    status = json.loads((tmp_path / "evidence/status.json").read_text())
    assert status["status"] == "blocked"
    assert status["reason"] == "kernel_gpu_fault_this_boot"


def test_live_mes_interrupts_owned_controller(monkeypatch, tmp_path):
    follower = FakeFollower()
    child = FakeChild()

    def popen(command, **kwargs):
        if command[0] == "journalctl":
            return follower
        follower.lines.put(MES)
        return child

    monkeypatch.setattr(guard.subprocess, "Popen", popen)
    monkeypatch.setattr(guard.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    assert guard.run(["owned-controller"], tmp_path / "evidence") == 86
    status = json.loads((tmp_path / "evidence/status.json").read_text())
    assert status["status"] == "blocked"
    assert status["gpu_fault_observed"] is True
    assert status["automatic_gpu_reuse_allowed"] is False
