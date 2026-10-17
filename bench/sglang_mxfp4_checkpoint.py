#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check packed-weight interpretation against slices of the real GLM checkpoint.

CPU-only: reads three small rows through safetensors mmap and compares the
stdlib oracle with an independent Torch broadcast implementation. No model
server, GPU allocation or full-weight copies are needed.
"""

import argparse
import json
from pathlib import Path

from safetensors import safe_open
import torch

from r9700_sglang.mxfp4_reference import unpack_row


def torch_dequant(packed, scales):
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                        0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
    values = torch.empty((packed.numel() * 2,), dtype=torch.float32)
    values[::2] = lut[(packed & 15).long()]
    values[1::2] = lut[(packed >> 4).long()]
    return values * torch.exp2(scales.float() - 127).repeat_interleave(32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit("Refusing to overwrite checkpoint diagnostic")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    names = [f"model.language_model.layers.22.mlp.experts.0.{projection}"
             for projection in ("gate_proj", "up_proj", "down_proj")]
    records = []
    for name in names:
        shard = index["weight_map"][name + ".weight"]
        if shard != index["weight_map"][name + ".weight_scale"]:
            raise RuntimeError("Weight and scale are in different shards")
        with safe_open(args.model / shard, framework="pt", device="cpu") as f:
            weight = f.get_slice(name + ".weight")
            scale = f.get_slice(name + ".weight_scale")
            shape, scale_shape = weight.get_shape(), scale.get_shape()
            if shape[1] != scale_shape[1] * 16:
                raise RuntimeError("Checkpoint has unexpected MXFP4 scale geometry")
            for row in (0, 1, 2):
                packed = weight[row:row + 1, :32].flatten()
                scales = scale[row:row + 1, :2].flatten()
                if bool((scales == 255).any()):
                    raise RuntimeError("Reserved E8M0 scale encountered")
                actual = torch.tensor(unpack_row(packed.tolist(), scales.tolist()))
                expected = torch_dequant(packed, scales)
                if not torch.equal(actual, expected):
                    raise RuntimeError(f"Oracle disagrees for {name}, row {row}")
                records.append(dict(name=name, shard=shard, row=row,
                                    packed_shape=shape, scale_shape=scale_shape,
                                    scale_min=int(scales.min()), scale_max=int(scales.max()),
                                    nonzero=int(torch.count_nonzero(actual)),
                                    max_abs_diff=float((actual - expected).abs().max())))
    args.out.write_text(json.dumps(records, indent=2) + "\n")
    print(f"PASS: {len(records)} real-checkpoint row slices agree exactly")


if __name__ == "__main__":
    main()
