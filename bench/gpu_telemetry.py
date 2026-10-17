#!/usr/bin/env python3
"""Read-only AMD telemetry with explicit SMI-to-HIP/BDF mapping.

No workload, model request, device reset or clock/power mutation. Sampled VRAM
maxima are observations, not exact allocator peaks. Missing metrics stay N/A.
Can follow a verified controller process; its exit never authorizes a restart.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
import time


COMMAND = ["amd-smi","metric","--mem-usage","--usage","--power","--temperature","--clock","--json"]
IRQ_BDF = re.compile(r"\bIR-PCI-MSI-(?P<bdf>[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])\b")


def irq_counts(interrupts, mapping):
    """Read per-GPU MSI totals without assuming IRQ number or HIP ordinal.

    None means that /proc/interrupts lacks a matching line. A flat counter is
    only an observation; it is not by itself evidence of a GPU hang.
    """
    expected = {device["bdf"].lower() for device in mapping}
    if len(expected) != len(mapping):
        raise ValueError("Ambiguous HIP-to-BDF mapping")
    result = {bdf: None for bdf in expected}
    for line in interrupts.splitlines():
        found = IRQ_BDF.search(line)
        if not found or "amdgpu" not in line[found.end():].split():
            continue
        bdf = found.group("bdf").lower()
        if bdf not in expected:
            continue
        if result[bdf] is not None:
            raise ValueError(f"Duplicate amdgpu MSI line for {bdf}")
        fields = line[:found.start()].split()
        if len(fields) < 2 or not fields[0].endswith(":"):
            raise ValueError(f"Malformed amdgpu MSI line for {bdf}")
        result[bdf] = sum(int(value) for value in fields[1:])
    return dict(sorted(result.items()))


def annotate(raw, devices, mapping):
    by_id = {d["gpu"]:d["bdf"].lower() for d in devices}
    by_bdf = {d["bdf"].lower():d for d in mapping}
    if len(by_id) != len(devices) or len(by_bdf) != len(mapping):
        raise ValueError("Ambiguous SMI or HIP identity")
    result = []
    for row in raw["gpu_data"]:
        bdf = by_id[row["gpu"]]
        if bdf in by_bdf:
            result.append(dict(smi_id=row["gpu"],**by_bdf[bdf],metrics=row))
    if len(result) != len(mapping) or len({r["bdf"] for r in result}) != len(mapping):
        raise ValueError("Not every participating HIP device has exactly one SMI observation")
    return sorted(result,key=lambda r:r["hip"])


def process_identity(pid):
    try:
        # Ignore spaces/parentheses in comm; state and starttime follow its last ')'.
        data = Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()
        return None if data[0] in ("Z","X") else data[19]
    except FileNotFoundError:
        return None


def irq_sample(mapping, read_interrupts=None):
    """Take one host-only sample, independent of amd-smi progress."""
    read_interrupts = read_interrupts or (lambda: Path("/proc/interrupts").read_text())
    row = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               monotonic_seconds=time.monotonic())
    try:
        row["irq_by_bdf"] = irq_counts(read_interrupts(), mapping)
        row["status"] = ("complete" if all(value is not None for value in row["irq_by_bdf"].values())
                         else "missing_gpu_irq")
    except Exception as error:
        row["status"] = "error"
        row["error"] = repr(error)
    return row


def irq_sampler(out, mapping, stop, interval):
    """Keep IRQ evidence flowing even when an amd-smi query stalls."""
    path = out / "irq-samples.jsonl"
    with path.open("a") as stream:
        while not stop.is_set():
            stream.write(json.dumps(irq_sample(mapping)) + "\n")
            stream.flush()
            stop.wait(interval)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--mapping",type=Path,required=True)
    p.add_argument("--until-pid",type=int,required=True)
    p.add_argument("--interval",type=float,default=5.)
    args = p.parse_args()
    if args.interval < 2:
        raise ValueError("Use an observational interval of at least two seconds")
    identity = process_identity(args.until_pid)
    if identity is None:
        raise RuntimeError("Controller is not live; do not infer permission to start work")
    args.out.mkdir()
    mapping = json.loads(args.mapping.read_text())
    devices = json.loads(subprocess.check_output(["amd-smi","list","--json"],text=True,timeout=15))
    (args.out/"devices.json").write_text(json.dumps(devices,indent=2)+"\n")
    plan = dict(scope="read-only sampled telemetry; not a workload or exact allocation peak",
                controller_pid=args.until_pid,controller_starttime=identity,interval_seconds=args.interval,
                irq_interval_seconds=1.,command=COMMAND,mapping=mapping,
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/"protocol.json").write_text(json.dumps(plan,indent=2)+"\n")
    stop = threading.Event()
    irq_errors = []

    def run_irq_sampler():
        try:
            irq_sampler(args.out,mapping,stop,1.)
        except Exception as error:
            irq_errors.append(repr(error))

    irq_thread = threading.Thread(target=run_irq_sampler,daemon=True)
    irq_thread.start()
    try:
        while process_identity(args.until_pid) == identity:
            start = time.monotonic()
            row = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),monotonic_seconds=start,status="error")
            try:
                response = subprocess.run(COMMAND,text=True,capture_output=True,timeout=15,check=True)
                row["devices"] = annotate(json.loads(response.stdout),devices,mapping)
                row["status"] = "pass"
                row["containers"] = subprocess.check_output(["docker","ps","--filter","name=glm53-sglang","--format","{{.Names}}"],text=True,timeout=10).splitlines()
            except Exception as error:
                row["error"] = repr(error)
            row["query_seconds"] = time.monotonic()-start
            with (args.out/"samples.jsonl").open("a") as stream:
                stream.write(json.dumps(row)+"\n")
            time.sleep(max(0,args.interval-(time.monotonic()-start)))
    finally:
        stop.set()
        irq_thread.join(timeout=2)
        (args.out/"complete.json").write_text(
            json.dumps(dict(status="controller terminal or sampler stopped; no restart performed",
                            irq_sampler_stopped=not irq_thread.is_alive(),
                            irq_sampler_error=irq_errors[0] if irq_errors else None)) + "\n")


if __name__ == "__main__":
    main()
