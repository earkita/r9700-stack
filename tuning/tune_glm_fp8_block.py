#!/usr/bin/env python3
"""Tune only the existing Quark block-FP8 GEMM launch configuration on gfx1201.

Run on an idle GPU in the pinned runtime image. No requantization or model
patching. Writes every candidate measurement/error and fresh-seed numerical
checks under --out. Output configs still need matched serving validation.
"""
import argparse
import itertools
import json
from pathlib import Path
import random
import statistics


DEFAULT = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128,
               GROUP_SIZE_M=32, num_warps=4, num_stages=2)


def candidates(m):
    # Conservative search: one 128-wide quantization group per K iteration,
    # and whole groups along N. Smaller divisors could be explored separately.
    for bm, bn, group, warps, stages in itertools.product(
            (16, 32, 64) if m <= 20 else (32, 64, 128),
            (128, 256), (1, 8), (4, 8), (1, 2)):
        yield dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=128,
                   GROUP_SIZE_M=group, num_warps=warps, num_stages=stages)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape', required=True, help='N,K')
    p.add_argument('--rows', required=True, help='Comma-separated M values')
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    n, k = map(int, a.shape.split(','))
    assert (n, k) in ((3072, 4096), (4096, 1536))
    rows = list(map(int, a.rows.split(',')))
    a.out.mkdir(parents=True, exist_ok=False)

    import torch
    import triton
    from vllm.model_executor.layers.quantization.utils import fp8_utils as f

    device = torch.cuda.get_device_properties(0)
    assert device.gcnArchName.split(':')[0] == 'gfx1201'
    torch.backends.cuda.matmul.allow_tf32 = False
    original = f.get_w8a8_block_fp8_configs
    records = a.out / 'attempts.jsonl'

    def save(row):
        with records.open('a') as out:
            out.write(json.dumps(row) + '\n')

    def data(m, seed):
        torch.manual_seed(seed)
        x = torch.randn(m, k, device='cuda').to(torch.float8_e4m3fn)
        w = torch.randn(n, k, device='cuda').to(torch.float8_e4m3fn)
        xs = torch.rand(m, k // 128, device='cuda') * .2 + .05
        ws = torch.rand(n // 128, k // 128, device='cuda') * .2 + .05
        reference = ((x.float() * xs.repeat_interleave(128, 1)) @
                     (w.float() * ws.repeat_interleave(128, 0).repeat_interleave(128, 1)).T)
        return (x, w, xs, ws), reference

    def run(tensors, cfg):
        m = tensors[0].shape[0]
        f.get_w8a8_block_fp8_configs = lambda *args, **kwargs: {m: cfg}
        try:
            return f.w8a8_triton_block_scaled_mm(*tensors, [128, 128], torch.bfloat16)
        finally:
            f.get_w8a8_block_fp8_configs = original

    def check(tensors, ref, cfg):
        y = run(tensors, cfg).float()
        torch.cuda.synchronize()
        diff = y - ref
        rel = (diff.norm() / ref.norm().clamp_min(1e-12)).item()
        peak = (diff.abs().max() / ref.abs().max().clamp_min(1e-12)).item()
        ok = bool(torch.isfinite(y).all()) and rel <= .005 and peak <= .015
        return dict(pass_=ok, relative_l2=rel, max_error_over_ref_peak=peak)

    def measure(tensors, cfg):
        return triton.testing.do_bench_cudagraph(lambda: run(tensors, cfg), rep=30)

    result = {}
    for m in rows:
        tensors, ref = data(m, 1234)
        base_check = check(tensors, ref, DEFAULT)
        save(dict(M=m, kind='default-correctness', **base_check))
        assert base_check['pass_'], base_check
        order = list(candidates(m))
        random.Random(1234 + m).shuffle(order)
        screened = []
        for index, cfg in enumerate([DEFAULT] + order):
            record = dict(M=m, kind='screen', index=index, config=cfg)
            try:
                correctness = check(tensors, ref, cfg)
                record.update(correctness)
                if correctness['pass_']:
                    ms = measure(tensors, cfg)
                    record['ms'] = ms
                    screened.append((ms, cfg))
            except Exception as error:
                record['error'] = str(error)
                # Compilation errors are recoverable; device runtime errors
                # invalidate subsequent timings and must stop this worker.
                if isinstance(error, (torch.OutOfMemoryError, torch.AcceleratorError)):
                    save(record)
                    raise
            save(record)
        _, winner = min(screened, key=lambda row: row[0])
        validations = []
        for seed in (5678, 9012):
            fresh, reference = data(m, seed)
            validation = check(fresh, reference, winner)
            save(dict(M=m, kind='fresh-validation', seed=seed, config=winner, **validation))
            validations.append(validation['pass_'])
            del fresh, reference
        if not all(validations):
            winner = DEFAULT
        paired = []
        for repetition in range(5):
            pair = {}
            labels = ('default', 'candidate') if repetition % 2 == 0 else ('candidate', 'default')
            for label in labels:
                pair[label] = measure(tensors, DEFAULT if label == 'default' else winner)
            paired.append(pair)
            save(dict(M=m, kind='paired', repetition=repetition, **pair))
        base_ms = statistics.median(x['default'] for x in paired)
        win_ms = statistics.median(x['candidate'] for x in paired)
        promoted = all(validations) and win_ms <= .95 * base_ms
        selected = winner if promoted else DEFAULT
        result[str(m)] = dict(config=selected, promoted=promoted, default_ms=base_ms,
                              candidate_ms=win_ms, speedup=base_ms / win_ms,
                              effective_tflops=2*m*n*k/(win_ms*1e9))
        (a.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(N=n, K=k, M=m, **result[str(m)])), flush=True)
        del tensors, ref

    name = f'N={n},K={k},device_name={f.get_device_name_as_file_name()},dtype=fp8_w8a8,block_shape=[128,128].json'
    (a.out / name).write_text(json.dumps({m: row['config'] for m, row in result.items()}, indent=2) + '\n')


if __name__ == '__main__':
    main()
