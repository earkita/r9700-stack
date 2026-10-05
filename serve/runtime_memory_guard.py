#!/usr/bin/env python3
"""Watch host headroom and stop only the specified container."""
import argparse
import json
import subprocess
import time
from pathlib import Path


def memory_snapshot(*, include_gpu=True):
    """Read host/driver counters without attaching to workers or touching GPUs.

    GTT includes normal host-visible allocations too: changes are evidence
    for investigation, not an automatic diagnosis of VRAM spill.
    """
    keys = {"MemAvailable", "MemFree", "SwapFree", "SwapCached", "Cached",
            "AnonPages", "Shmem", "Mlocked", "Unevictable", "PageTables",
            "SUnreclaim"}
    memory = {line.split(":")[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if line.split(":")[0] in keys}
    devices = []
    paths = sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_used")) if include_gpu else []
    for path in paths:
        device = {"card": path.parent.parent.name,
                  "pci": path.parent.resolve().name}
        for field in ("vram_used", "vram_total", "gtt_used", "gtt_total", "preempt_used"):
            try:
                device[field] = int((path.parent / ("mem_info_" + field)).read_text())
            except (FileNotFoundError, PermissionError):
                device[field] = None
        devices.append(device)
    vmstat = {}
    for line in Path("/proc/vmstat").read_text().splitlines():
        key, value = line.split()
        if key in ("pgmajfault", "pswpin", "pswpout", "pgscan_direct", "pgscan_kswapd"):
            vmstat[key] = int(value)
    return {"host": memory, "gpu_used": [d["vram_used"] for d in devices],
            "gpu_memory": devices, "vmstat": vmstat}


def inspect_container(container):
    result = subprocess.run(
        ["docker", "inspect", container], capture_output=True, text=True,
        timeout=5,
    )
    if result.returncode:
        raise RuntimeError(f"Cannot inspect container {container}: {result.stderr.strip()}")
    return json.loads(result.stdout)[0]


def validate_oom_priority(info):
    config = info["HostConfig"]
    if config["OomScoreAdj"] < 500:
        raise ValueError("Guard requires --oom-score-adj >= 500")


def stop_container(container_id, grace_seconds, emit):
    # Pin the immutable ID: a later container reusing the name is never killed.
    # SIGINT during engine startup did not stop workers promptly in attempt 5.
    for signal in ("SIGTERM", "SIGKILL"):
        emit({"action": "stop", "signal": signal, "container_id": container_id})
        try:
            subprocess.run(["docker", "kill", f"--signal={signal}", container_id],
                           capture_output=True, text=True, timeout=5)
        except subprocess.TimeoutExpired:
            emit({"action": "kill_timeout", "signal": signal})
        deadline = time.monotonic() + (grace_seconds if signal == "SIGTERM" else 10)
        while time.monotonic() < deadline:
            try:
                info = inspect_container(container_id)
            except subprocess.TimeoutExpired:
                emit({"action": "inspect_timeout", "container_id": container_id})
                continue
            if not info["State"]["Running"]:
                emit({"action": "stopped", "exit_code": info["State"]["ExitCode"]})
                return
            time.sleep(.2)
    raise RuntimeError("Container did not exit after bounded TERM/KILL shutdown")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("container")
    parser.add_argument("--minimum-available-gib", type=float, default=4)
    parser.add_argument("--grace-seconds", type=float, default=2)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    if args.minimum_available_gib <= 0 or not 0 <= args.grace_seconds <= 5:
        parser.error("Positive memory headroom and 0–5 seconds grace are required")
    info = inspect_container(args.container)
    validate_oom_priority(info)
    container_id = info["Id"]
    args.log.parent.mkdir(parents=True, exist_ok=True)

    def emit(record):
        record = {"time": time.time(), **record}
        with args.log.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if "action" in record:
            print(json.dumps(record), flush=True)

    emit({"action": "watch", "container_id": container_id,
          "minimum_available_gib": args.minimum_available_gib})
    while True:
        # The protection path must not wait on GPU-driver sysfs reads. Keep
        # optional GPU diagnostics outside the host-headroom guard.
        snapshot = memory_snapshot(include_gpu=False)
        emit(snapshot)
        if snapshot["host"]["MemAvailable"] < args.minimum_available_gib * 2**30:
            stop_container(container_id, args.grace_seconds, emit)
            return
        try:
            info = inspect_container(container_id)
        except subprocess.TimeoutExpired:
            emit({"action": "inspect_timeout", "container_id": container_id})
            time.sleep(1)
            continue
        if not info["State"]["Running"]:
            emit({"action": "exited", "exit_code": info["State"]["ExitCode"],
                  "oom_killed": info["State"]["OOMKilled"]})
            return
        time.sleep(1)


if __name__ == "__main__":
    main()
