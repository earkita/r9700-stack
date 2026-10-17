#!/usr/bin/env python3
"""Compare scale loading for the GLM packed W4A4 gate/up GEMV.

Diagnostic one-GPU graph replay only. It is neither a model decode test nor
evidence of a model-level speedup. Run only on an otherwise idle gfx1201 GPU
after recording the mandatory PCIe preflight and loaded-link observations.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import time

import torch
from vllm.triton_utils import triton

from r9700_vllm.moe.packed_gemv import _gemv


def invoke(x, w, scales, ids, out, grouped):
    n, k = 512, 4096
    _gemv[(ids.numel(), triton.cdiv(n, 8))](
        x, w, scales, ids, None, out, k, n, 8, False, 8, 256, grouped,
        num_warps=4, enable_fp_fusion=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--replays', type=int, default=9)
    parser.add_argument('--kernels-per-graph', type=int, default=30)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('output directory already exists')
    args.out.mkdir(parents=True)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx1201':
        raise SystemExit('expected exactly one isolated gfx1201 GPU')
    protocol = dict(kind='one-GPU microbenchmark, not model decode',
                    rows=[5, 20, 32], order=['baseline', 'grouped', 'baseline'],
                    replays=args.replays, kernels_per_graph=args.kernels_per_graph,
                    seed=2702, torch=torch.__version__)
    raw = json.dumps(protocol, indent=2) + '\n'
    (args.out / 'protocol.json').write_text(raw)
    (args.out / 'protocol.sha256').write_text(hashlib.sha256(raw.encode()).hexdigest() + '\n')
    torch.manual_seed(2702)
    for rows in protocol['rows']:
        slots = rows * 8
        ids = torch.randint(0, 288, (slots,), dtype=torch.int32, device='cuda')
        x = torch.randn((rows, 4096), dtype=torch.bfloat16, device='cuda')
        w = torch.randint(0, 256, (288, 512, 2048), dtype=torch.uint8, device='cuda')
        scales = torch.randint(117, 129, (288, 512, 128), dtype=torch.uint8, device='cuda')
        reference = torch.empty((slots, 512), dtype=torch.bfloat16, device='cuda')
        candidate = torch.empty_like(reference)
        invoke(x, w, scales, ids, reference, False)
        invoke(x, w, scales, ids, candidate, True)
        torch.cuda.synchronize()
        delta = (candidate.float() - reference.float()).abs()
        numerical = dict(max_abs_diff=delta.max().item(),
                         exact_fraction=(candidate == reference).float().mean().item())
        torch.testing.assert_close(candidate, reference, rtol=.01, atol=.01)
        for variant in protocol['order']:
            grouped = variant == 'grouped'
            out = candidate if grouped else reference
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.kernels_per_graph):
                    invoke(x, w, scales, ids, out, grouped)
            for _ in range(2):
                graph.replay()
            torch.cuda.synchronize()
            times = []
            for _ in range(args.replays):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end) / args.kernels_per_graph)
            record = dict(rows=rows, variant=variant, per_kernel_ms=times,
                          median_kernel_ms=median(times), numerical=numerical,
                          timestamp_monotonic=time.monotonic())
            with (args.out / 'results.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            print(json.dumps({k: record[k] for k in ('rows', 'variant', 'median_kernel_ms')}), flush=True)


if __name__ == '__main__':
    main()
