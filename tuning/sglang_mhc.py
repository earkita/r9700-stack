#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded mHC attribution: operator timings only, never model throughput."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time


def protocol(backend="sinkhorn"):
    result = dict(kind="single-GPU mHC Torch vs upstream Triton Sinkhorn diagnostic",
                rows=[1, 5, 20], hidden=4096, hc=4, sinkhorn_iterations=20,
                seed=20261015, warmups=3, repetitions=11,
                order=["torch_a", "triton", "torch_b"], execution="eager",
                tolerance=dict(mix_atol=2e-6, mix_rtol=2e-5,
                               input_relative_l2=0.002, input_max_abs=0.03125),
                excluded="prefill, attention, MoE, collectives; no model-speed claim")
    if backend != "sinkhorn":
        result.update(kind="single-GPU mHC Torch vs vLLM TileLang full pre diagnostic",
                      backend=backend, order=["torch_a", backend, "torch_b"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--backend", choices=("sinkhorn", "vllm_tilelang"), default="sinkhorn")
    args = parser.parse_args()
    spec = json.dumps(protocol(args.backend), indent=2) + "\n"
    digest = hashlib.sha256(spec.encode()).hexdigest() + "\n"
    if args.prepare:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "protocol.json").write_text(spec)
        (args.out / "protocol.sha256").write_text(digest)
        return
    if ((args.out / "protocol.json").read_text() != spec or
            (args.out / "protocol.sha256").read_text() != digest or
            (args.out / "results.jsonl").exists()):
        raise RuntimeError("Missing/changed frozen protocol or existing results")
    import torch
    from sglang.kernels.ops.layernorm.mhc import _mhc_pre_torch
    from r9700_sglang.mhc import pre, pre_tilelang
    candidate = pre if args.backend == "sinkhorn" else pre_tilelang
    candidate_name = "triton" if args.backend == "sinkhorn" else args.backend
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one idle GPU")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1201":
        raise RuntimeError("This diagnostic requires gfx1201")
    torch.manual_seed(protocol()["seed"])

    def measure(op, inputs):
        for _ in range(3):
            op(**inputs)
        torch.cuda.synchronize()
        samples = []
        for _ in range(11):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            t = time.perf_counter()
            start.record()
            op(**inputs)
            end.record()
            end.synchronize()
            samples.append(dict(gpu_interval_ms=start.elapsed_time(end),
                                wall_ms=(time.perf_counter() - t) * 1000))
        return dict(samples=samples,
                    median_gpu_interval_ms=statistics.median(s["gpu_interval_ms"] for s in samples),
                    median_wall_ms=statistics.median(s["wall_ms"] for s in samples))

    for rows in (1, 5, 20):
        for scale in (0.0, 0.1, 2.0):
            inputs = dict(residual=(torch.randn(rows, 4, 4096, device="cuda") * scale).bfloat16(),
                          fn=torch.randn(24, 16384, device="cuda") * 0.02,
                          hc_scale=torch.tensor([1., 2., .5], device="cuda"),
                          hc_base=torch.randn(24, device="cuda"),
                          rms_eps=1e-5, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
                          hc_post_mult_value=2.0, sinkhorn_repeat=20)
            expected, actual = _mhc_pre_torch(**inputs), candidate(**inputs)
            errors = []
            for i, (ref, got) in enumerate(zip(expected, actual)):
                delta = (got.float() - ref.float()).abs()
                rel = (delta.norm() / ref.float().norm().clamp_min(1e-30)).item()
                errors.append(dict(max_abs=delta.max().item(), relative_l2=rel,
                                   exact_fraction=(ref == got).float().mean().item()))
                assert torch.isfinite(got).all()
                if i < 2:
                    torch.testing.assert_close(got, ref, atol=2e-6, rtol=2e-5)
                else:
                    assert rel <= .002 and delta.max().item() <= .03125
            result = dict(rows=rows, scale=scale, numeric=errors)
            if scale == .1:
                for tag, op in (("torch_a", _mhc_pre_torch), (candidate_name, candidate), ("torch_b", _mhc_pre_torch)):
                    result[tag] = measure(op, inputs)
            with (args.out / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
