#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated SGLang GLM MoE experiments on one idle gfx1201 GPU.

Run inside the pinned image with this repository mounted at /work. Select
one candidate at a time; this diagnostic never substitutes for model decode.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from r9700_sglang.packed_moe import apply_sglang
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import quant_dequant_mxfp4


def dequant_reference(weight, scales):
    """Tensor counterpart of the independent CPU E2M1/E8M0 oracle."""
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                        0, -.5, -1, -1.5, -2, -3, -4, -6],
                       device=weight.device, dtype=torch.float32)
    n, packed_k = weight.shape
    value = torch.empty((n, packed_k * 2), device=weight.device)
    value[:, ::2] = lut[(weight & 15).long()]
    value[:, 1::2] = lut[(weight >> 4).long()]
    scale = torch.exp2(scales.float() - 127).repeat_interleave(32, dim=1)
    return (value * scale).bfloat16()


def reference_one_row(hidden, ids, routing, w1, s1, w2, s2):
    """Dequantize selected experts, then use dense BLAS as a separate oracle."""
    x = quant_dequant_mxfp4(hidden)
    parts = []
    for expert, weight in zip(ids.tolist(), routing.tolist()):
        gu = F.linear(x, dequant_reference(w1[expert], s1[expert])).bfloat16()
        gate, up = gu.float().chunk(2, dim=1)
        mid = (F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)).bfloat16()
        mid = quant_dequant_mxfp4(mid)
        # Packed GEMV and HIP fold the FP32 route weight into the FP32 down
        # accumulator BEFORE its single BF16 store. A BF16 linear output here
        # would add a rounding point that neither implementation has.
        down = F.linear(mid.float(), dequant_reference(w2[expert], s2[expert]).float())
        parts.append((down * weight).bfloat16())
    return torch.stack(parts).sum(dim=0).bfloat16()


def output(layer, dispatch, grouped, down_bn=8, backend="packed"):
    os.environ["R9K_SGLANG_GATE_SCALE_GROUP"] = "1" if grouped else "0"
    os.environ["R9K_SGLANG_DOWN_BN"] = str(down_bn)
    os.environ["R9K_SGLANG_MOE_BACKEND"] = backend
    return apply_sglang(layer, dispatch).hidden_states


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 5, 20])
    p.add_argument("--experiment", choices=("gate-scale", "down-bn", "hip-w4a4"),
                   default="gate-scale")
    p.add_argument("--repetitions", type=int, default=9)
    p.add_argument("--prepare", action="store_true",
                   help="Freeze protocol without initializing or using a GPU")
    args = p.parse_args()
    if args.repetitions < 3 or any(row < 1 for row in args.rows):
        raise SystemExit("Need at least three repeats and positive row counts")
    protocol = dict(kind="one-GPU routed MoE operator diagnostic, not model decode",
                    rows=args.rows, topk=8, local_experts=288,
                    gate_up=[288, 512, 2048], down=[288, 4096, 128],
                    order=["S0", "S1", "S0"], repetitions=args.repetitions,
                    warmups=3, seed=20261015, input_dtype="bfloat16",
                    weights="packed E2M1", scales="E8M0", activation="Quark MXFP4 QDQ",
                    tolerance=dict(max_abs=0.25, relative_l2=0.02),
                    torch=torch.__version__)
    if args.experiment == "down-bn":
        protocol["experiment"] = "down-bn 8→64→8, grouped gate/up scales disabled"
    elif args.experiment == "hip-w4a4":
        if args.rows != [1]:
            raise SystemExit("HIP W4A4 C1 qualification requires --rows 1")
        protocol["experiment"] = "C1 packed GEMV→gfx1201 HIP W4A4→packed GEMV"
    content = json.dumps(protocol, indent=2) + "\n"
    digest = hashlib.sha256(content.encode()).hexdigest() + "\n"
    if args.prepare:
        if args.out.exists():
            raise SystemExit("Refusing to overwrite existing diagnostic")
        args.out.mkdir(parents=True)
        (args.out / "protocol.json").write_text(content)
        (args.out / "protocol.sha256").write_text(digest)
        print("Frozen operator protocol at", args.out)
        return
    if ((args.out / "protocol.json").read_text() != content or
            (args.out / "protocol.sha256").read_text() != digest):
        raise SystemExit("Frozen protocol differs from invocation")
    if (args.out / "results.jsonl").exists() or (args.out / "numerical.jsonl").exists():
        raise SystemExit("Refusing to overwrite existing diagnostic attempts")
    if torch.cuda.device_count() != 1 or not torch.cuda.get_device_name(0).endswith("R9700"):
        raise SystemExit("Expected exactly one isolated Radeon R9700")
    torch.manual_seed(protocol["seed"])
    w1 = torch.randint(0, 256, (288, 512, 2048), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (288, 4096, 128), dtype=torch.uint8, device="cuda")
    s1 = torch.randint(117, 129, (288, 512, 128), dtype=torch.uint8, device="cuda")
    s2 = torch.randint(117, 129, (288, 4096, 8), dtype=torch.uint8, device="cuda")
    layer = SimpleNamespace(w13_weight=w1, w2_weight=w2,
                            w13_weight_scale=s1, w2_weight_scale=s2,
                            moe_runner_config=SimpleNamespace(swiglu_limit=10.0,
                                                              activation="silu"))
    for rows in args.rows:
        ids = torch.randint(0, 288, (rows, 8), dtype=torch.int32, device="cuda")
        routing = torch.softmax(torch.randn((rows, 8), device="cuda"), dim=1)
        hidden = (torch.randn((rows, 4096), device="cuda") * .25).bfloat16()
        dispatch = SimpleNamespace(hidden_states=hidden,
                                   topk_output=SimpleNamespace(topk_ids=ids,
                                                               topk_weights=routing))
        baseline = output(layer, dispatch, False)
        candidate = output(layer, dispatch,
                           args.experiment == "gate-scale",
                           64 if args.experiment == "down-bn" else 8,
                           "hip_w4a4" if args.experiment == "hip-w4a4" else "packed")
        torch.cuda.synchronize()
        delta = candidate.float() - baseline.float()
        numerical = dict(max_abs=delta.abs().max().item(),
                         relative_l2=(delta.norm() / baseline.float().norm()).item(),
                         exact_fraction=(candidate == baseline).float().mean().item(),
                         finite=bool(torch.isfinite(candidate).all()),
                         nonzero=bool(candidate.abs().sum() > 0))
        if rows == 1:
            oracle = reference_one_row(hidden, ids[0], routing[0], w1, s1, w2, s2)
            oracle_delta = candidate.float() - oracle.float()
            numerical["oracle_max_abs"] = oracle_delta.abs().max().item()
            numerical["oracle_relative_l2"] = (oracle_delta.norm() / oracle.float().norm()).item()
            if args.experiment == "hip-w4a4":
                from r9700_sglang.hip_w4a4 import encode_qdq
                qdq = quant_dequant_mxfp4(hidden.contiguous())
                encoded, group_scales = encode_qdq(qdq)
                decoded = (encoded.float() * group_scales.repeat_interleave(32, dim=1)).bfloat16()
                numerical["transport_exact"] = bool(torch.equal(qdq, decoded))
        with (args.out / "numerical.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(rows=rows, **numerical)) + "\n")
        if not numerical["finite"] or not numerical["nonzero"] or \
                numerical.get("transport_exact", True) is False or \
                numerical["max_abs"] > protocol["tolerance"]["max_abs"] or \
                numerical["relative_l2"] > protocol["tolerance"]["relative_l2"]:
            raise RuntimeError(f"S1 numerical gate failed for rows={rows}: {numerical}")
        for variant in protocol["order"]:
            grouped = variant == "S1" and args.experiment == "gate-scale"
            down_bn = 64 if variant == "S1" and args.experiment == "down-bn" else 8
            backend = "hip_w4a4" if variant == "S1" and args.experiment == "hip-w4a4" else "packed"
            for _ in range(protocol["warmups"]):
                output(layer, dispatch, grouped, down_bn, backend)
            torch.cuda.synchronize()
            gpu_ms = []
            wall_ms = []
            for _ in range(args.repetitions):
                start, end = (torch.cuda.Event(enable_timing=True),
                              torch.cuda.Event(enable_timing=True))
                wall_start = time.perf_counter()
                start.record()
                output(layer, dispatch, grouped, down_bn, backend)
                end.record()
                end.synchronize()
                wall_ms.append((time.perf_counter() - wall_start) * 1000)
                gpu_ms.append(start.elapsed_time(end))
            record = dict(rows=rows, variant=variant, gpu_ms=gpu_ms,
                          wall_ms=wall_ms, median_gpu_ms=statistics.median(gpu_ms),
                          median_wall_ms=statistics.median(wall_ms))
            with (args.out / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps({k: record[k] for k in ("rows", "variant", "median_gpu_ms", "median_wall_ms")}), flush=True)


if __name__ == "__main__":
    main()
