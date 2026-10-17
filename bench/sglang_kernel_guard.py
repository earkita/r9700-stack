"""Fail closed around an exclusive GPU benchmark when amdgpu reports a fault.

Use as ``python3 bench/sglang_kernel_guard.py --evidence DIR -- COMMAND ...``.
The child controller remains responsible for stopping only its own runtime.
This guard never resets GPUs, starts another model, or changes K4.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

from sglang_runtime_health import kernel_gpu_fault_lines, pcie_warning_lines


def _save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def run(command, evidence):
    evidence.mkdir(parents=True, exist_ok=False)
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    _save(evidence / "guard.json", {"boot_id": boot_id, "command": command,
                                     "started_utc": datetime.now(timezone.utc).isoformat()})

    # Start following before the full-boot scan, so the two observations
    # overlap. A prior fatal event on this boot blocks *all* further attempts.
    follower = subprocess.Popen(
        ["journalctl", "-k", "-b", "--follow", "-n", "0", "--no-pager", "-o", "short-iso"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    stop = threading.Event()
    fatal = threading.Event()
    child = None
    findings = []

    def observe():
        with (evidence / "kernel-live.log").open("w") as raw:
            for line in follower.stdout:
                raw.write(line)
                raw.flush()
                kind = "gpu_fault" if kernel_gpu_fault_lines(line) else (
                    "pcie_warning" if pcie_warning_lines(line) else None)
                if kind:
                    record = {"kind": kind, "line": line.rstrip("\n"),
                              "observed_utc": datetime.now(timezone.utc).isoformat()}
                    findings.append(record)
                    with (evidence / "kernel-events.jsonl").open("a") as events:
                        events.write(json.dumps(record) + "\n")
                        events.flush()
                        os.fsync(events.fileno())
                    if kind == "gpu_fault":
                        fatal.set()
                        if child is not None and child.poll() is None:
                            child.send_signal(signal.SIGINT)
            if not stop.is_set():
                fatal.set()  # Losing the observer makes reuse unsafe.

    thread = threading.Thread(target=observe, daemon=True)
    thread.start()
    try:
        scan = subprocess.run(["journalctl", "-k", "-b", "--no-pager", "-o", "short-iso"],
                              capture_output=True, text=True, check=True)
        (evidence / "kernel-before.log").write_text(scan.stdout)
        prior = kernel_gpu_fault_lines(scan.stdout)
        if prior or fatal.is_set():
            _save(evidence / "status.json", {"status": "blocked", "reason": "kernel_gpu_fault_this_boot",
                                               "prior_fault_lines": prior, "live_events": findings})
            return 86
        if follower.poll() is not None:
            raise RuntimeError("kernel journal follower exited before child launch")
        child = subprocess.Popen(command)
        if fatal.is_set() and child.poll() is None:
            child.send_signal(signal.SIGINT)
        interrupt_deadline = None
        while True:
            try:
                code = child.wait(timeout=2)
                break
            except subprocess.TimeoutExpired:
                if fatal.is_set() and interrupt_deadline is None:
                    child.send_signal(signal.SIGINT)
                    interrupt_deadline = time.monotonic() + 45
                if interrupt_deadline is not None and time.monotonic() >= interrupt_deadline:
                    child.terminate()
                    try:
                        code = child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        code = child.wait(timeout=10)
                    break
        status = "blocked" if fatal.is_set() else ("pass" if code == 0 else "child_failed")
        _save(evidence / "status.json", {"status": status, "child_exit_code": code,
                                           "gpu_fault_observed": fatal.is_set(), "live_events": findings,
                                           "automatic_gpu_reuse_allowed": status == "pass"})
        return 86 if fatal.is_set() else code
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
        _save(evidence / "status.json", {"status": "blocked", "reason": "journal_observation_failed",
                                           "error": repr(error), "automatic_gpu_reuse_allowed": False})
        return 86
    finally:
        stop.set()
        follower.terminate()
        try:
            follower.wait(timeout=5)
        except subprocess.TimeoutExpired:
            follower.kill()
            follower.wait()
        thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a child command is required")
    return run(command, args.evidence)


if __name__ == "__main__":
    sys.exit(main())
