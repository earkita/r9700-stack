#!/usr/bin/env python3
"""Run the frozen, serial MTP3/MTP4 quality screen on an exclusive 8-GPU host.

Invoke under sglang_kernel_guard.py. This controller owns only its two
experiment containers; it never starts or stops K4 or a production service.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
import urllib.request

from sglang_runtime_health import kernel_gpu_fault_lines


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def protocol(out):
    raw = (out / "protocol.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != (out / "protocol.sha256").read_text().strip():
        raise RuntimeError("Frozen MTP4 protocol changed")
    plan = json.loads(raw)
    if plan["order"] != [3, 4] or len(plan["quality_seeds"]) != 10:
        raise RuntimeError("Expected the declared MTP3/MTP4 ten-seed cohort")
    if len(set(plan["quality_seeds"])) != 10:
        raise RuntimeError("Quality seeds are not distinct")
    if digest(out / plan["corpus"]["file"]) != plan["corpus"]["sha256"]:
        raise RuntimeError("Frozen input IDs changed")
    for path, expected in plan["sources"].items():
        if digest(path) != expected:
            raise RuntimeError(f"Frozen source changed: {path}")
    if digest(__file__) != plan["controller_sha256"]:
        raise RuntimeError("Frozen controller changed")
    left, right = [list(plan["commands"][str(depth)]) for depth in (3, 4)]
    for command, depth in ((left, 3), (right, 4)):
        if command[command.index("--speculative-num-steps") + 1] != str(depth):
            raise RuntimeError("MTP depth differs from the protocol leg")
        if command[command.index("--speculative-num-draft-tokens") + 1] != str(depth + 1):
            raise RuntimeError("MTP verify width differs from the depth")
        for field in ("--name", "--speculative-num-steps", "--speculative-num-draft-tokens"):
            command[command.index(field) + 1] = "<variable>"
    if left != right:
        raise RuntimeError("More than the declared MTP depth differs between legs")
    return plan


def hardware_module():
    source = Path(__file__).parent / "baseline-earkita/hardware.py"
    spec = importlib.util.spec_from_file_location("mtp4_hardware", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ready(name, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8081/get_server_info", timeout=10) as response:
                return json.load(response)
        except Exception:
            running = subprocess.check_output(
                ["docker", "inspect", name, "--format", "{{.State.Running}}"], text=True
            ).strip()
            if running != "true":
                raise RuntimeError(f"{name} exited before readiness")
            time.sleep(2)
    raise TimeoutError(f"{name} did not become ready")


def flush():
    request = urllib.request.Request(
        "http://127.0.0.1:8081/flush_cache?timeout=5", b"", method="POST"
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.read().decode()


def quality_plan(master, name, args):
    expected = dict(master["expected_server_args"])
    expected.update(speculative_num_steps=args, speculative_num_draft_tokens=args + 1)
    return {
        "container": name,
        "image_id": master["image_id"],
        "base": "http://127.0.0.1:8081",
        "expected_server_args": expected,
        "corpus": master["corpus"],
        "runner_sha256": master["sources"]["bench/sglang_glm_completed_code.py"],
        "stream_client_sha256": master["sources"]["bench/sglang_spec_decode.py"],
        "seeds": master["quality_seeds"],
        "sampling": master["sampling"] | {
            "max_new_tokens": master["quality_budget"],
            "ignore_eos": False,
            "skip_special_tokens": False,
            "no_stop_trim": True,
        },
        "cache_salt": "mtp4-quality-followup-v1",
    }


def run(out):
    plan = protocol(out)
    hw = hardware_module()
    kernel = subprocess.check_output(
        ["journalctl", "-k", "-b", "--no-pager"], text=True, timeout=30
    )
    if kernel_gpu_fault_lines(kernel):
        raise RuntimeError("Current boot contains a GPU fault; do not launch MTP4")
    image = subprocess.check_output(
        ["docker", "image", "inspect", plan["image"], "--format", "{{.Id}}"], text=True
    ).strip()
    if image != plan["image_id"]:
        raise RuntimeError("Qualified image identity changed")
    active = subprocess.check_output(["docker", "ps", "--format", "{{.Names}}"], text=True).splitlines()
    for container in active:
        devices = json.loads(subprocess.check_output(
            ["docker", "inspect", container, "--format", "{{json .HostConfig.Devices}}"], text=True
        )) or []
        if any(device.get("PathOnHost") == "/dev/kfd" for device in devices):
            raise RuntimeError(f"GPU container {container} is active; this series requires exclusive GPUs")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8081)) == 0:
            raise RuntimeError("Port 8081 is occupied; refuse to contact an unrelated model")
    for depth in plan["order"]:
        command = plan["commands"][str(depth)]
        name = command[command.index("--name") + 1]
        if subprocess.run(["docker", "inspect", name], capture_output=True).returncode == 0:
            raise RuntimeError(f"Refusing to reuse experiment container {name}")

    # Capture every participating endpoint and upstream bridge before startup.
    prior = Path("bench/results/sglang-prefill-optimization-20261015/mtp345-sampling37134-model-v1/mtp3/pcie-before/mapping.json")
    expected_mapping = json.loads(prior.read_text())
    host_idle = hw.capture(out / "pcie-before-runtime", expected_mapping)
    (out / "boot-id.txt").write_text(Path("/proc/sys/kernel/random/boot_id").read_text())
    reports = {}
    for depth in plan["order"]:
        kernel = subprocess.check_output(
            ["journalctl", "-k", "-b", "--no-pager"], text=True, timeout=30
        )
        if kernel_gpu_fault_lines(kernel):
            raise RuntimeError("GPU fault appeared before the next MTP variant")
        leg = out / f"mtp{depth}"
        leg.mkdir()
        command = plan["commands"][str(depth)]
        name = command[command.index("--name") + 1]
        idle = hw.capture(leg / "pcie-before", expected_mapping)
        if idle["paths"] != host_idle["paths"]:
            raise RuntimeError("PCIe topology changed before a variant")
        model_started = False
        watcher_stop = threading.Event()
        loaded = []
        watcher = None
        telemetry = None
        telemetry_error = None
        try:
            with (leg / "container-id.txt").open("w") as output:
                subprocess.run(command, check=True, stdout=output)
            model_started = True
            info = ready(name)
            actual_mapping = hw.hip_mapping(name)
            save(leg / "mapping.json", actual_mapping)
            if actual_mapping != expected_mapping:
                raise RuntimeError("HIP/worker/BDF mapping differs from frozen preflight")
            server_args = info.get("server_args", info)
            for key, expected in quality_plan(plan, name, depth)["expected_server_args"].items():
                if server_args.get(key) != expected:
                    raise RuntimeError(f"Runtime {key}={server_args.get(key)!r}, expected {expected!r}")
            save(leg / "server-info.json", info)

            def watch_links():
                while not watcher_stop.is_set():
                    loaded.append(hw.links(actual_mapping))
                    watcher_stop.wait(0.5)

            watcher = threading.Thread(target=watch_links, daemon=True)
            watcher.start()
            telemetry = subprocess.Popen(
                ["python3", "bench/gpu_telemetry.py", "--out", str(leg / "telemetry"),
                 "--mapping", str(leg / "mapping.json"), "--until-pid", str(os.getpid()),
                 "--interval", "5"],
                stdout=(leg / "telemetry.log").open("w"), stderr=subprocess.STDOUT,
            )
            with (leg / "contract.log").open("w") as log:
                subprocess.run(
                    ["python3", "bench/sglang_glm_contract.py", "--out", str(leg / "contract")],
                    check=True, stdout=log, stderr=subprocess.STDOUT,
                )
            quality = leg / "quality"
            quality.mkdir()
            shutil.copyfile(out / "corpus.json", quality / "corpus.json")
            (quality / "flush.txt").write_text(flush())
            frozen = json.dumps(quality_plan(plan, name, depth), indent=2) + "\n"
            (quality / "protocol.json").write_text(frozen)
            (quality / "protocol.sha256").write_text(hashlib.sha256(frozen.encode()).hexdigest() + "\n")
            with (quality / "run.log").open("w") as log:
                result = subprocess.run(
                    ["python3", "bench/sglang_glm_completed_code.py", "--out", str(quality)],
                    stdout=log, stderr=subprocess.STDOUT,
                )
            summary = json.loads((quality / "summary.json").read_text())
            summary["runner_exit_code"] = result.returncode
            reports[str(depth)] = summary
            save(leg / "result.json", summary)
        finally:
            watcher_stop.set()
            if watcher is not None:
                watcher.join(timeout=5)
                save(leg / "loaded-links.json", loaded)
            if telemetry is not None:
                telemetry.terminate()
                try:
                    telemetry.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    telemetry.kill()
                    telemetry.wait(timeout=10)
                evidence = leg / "telemetry"
                smi = evidence / "samples.jsonl"
                irq = evidence / "irq-samples.jsonl"
                if not smi.exists() or not irq.exists():
                    telemetry_error = "GPU telemetry did not produce both SMI and IRQ evidence"
                elif not any(json.loads(line).get("status") == "pass" for line in smi.read_text().splitlines()):
                    telemetry_error = "GPU telemetry has no complete SMI sample"
                elif not any(json.loads(line).get("status") == "complete" for line in irq.read_text().splitlines()):
                    telemetry_error = "GPU telemetry has no complete per-BDF IRQ sample"
            if model_started:
                with (leg / "server.log").open("w") as output:
                    subprocess.run(["docker", "logs", name], stdout=output, stderr=subprocess.STDOUT)
                subprocess.run(["docker", "stop", "-t", "45", name], check=True)
            if watcher is not None:
                hw.check_loaded(loaded, idle, hw.expected_links(idle))
            if telemetry_error is not None:
                raise RuntimeError(telemetry_error)
    save(out / "comparison.json", {
        "scope": plan["scope"], "reports": reports,
        "qualification": "screen only; no speed claim or long-context qualification",
    })
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    protocol(args.out)
    if not args.validate_only:
        run(args.out)


if __name__ == "__main__":
    main()
