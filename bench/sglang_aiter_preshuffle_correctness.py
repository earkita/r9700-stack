"""One-GPU numerical check for the gfx1201 AITER paged-MQA KPool reader.

This is a bounded compatibility diagnostic, not a throughput benchmark.
"""

import json

import torch
from aiter import dtypes
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits


def packed_cache(keys: torch.Tensor, scales: torch.Tensor, preshuffle: bool):
    # SGLang's writer uses (token_tile, col_tile, token_in_tile, col_in_tile).
    num_blocks = keys.shape[0]
    key_bytes = keys.view(torch.uint8)
    if preshuffle:
        key_bytes = (
            key_bytes.view(num_blocks, 4, 16, 8, 16)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
    raw = torch.empty((num_blocks, 64 * 132), dtype=torch.uint8, device=keys.device)
    raw[:, : 64 * 128] = key_bytes.reshape(num_blocks, -1)
    raw[:, 64 * 128 :] = scales.view(torch.uint8).reshape(num_blocks, -1)
    return raw.view(num_blocks, 64, 1, 132)


def main():
    torch.manual_seed(20261017)
    device = torch.device("cuda:0")
    q = torch.randn((1, 1, 32, 128), device=device, dtype=torch.float16).to(dtypes.fp8)
    keys = torch.randn((2, 64, 128), device=device, dtype=torch.float16).to(dtypes.fp8)
    scales = torch.rand((2, 64), device=device, dtype=torch.float32) * 0.4 + 0.8
    weights = torch.rand((1, 1, 32), device=device, dtype=torch.float32)
    block_table = torch.tensor([[1, 0]], device=device, dtype=torch.int32)
    results = []
    for length in (64, 96, 128):
        context_lens = torch.tensor([length], device=device, dtype=torch.int32)
        outputs = []
        for preshuffle in (False, True):
            out = torch.full((1, 1, 128), float("nan"), device=device)
            deepgemm_fp8_paged_mqa_logits(
                q,
                packed_cache(keys, scales, preshuffle),
                weights,
                out,
                context_lens,
                block_table,
                128,
                Preshuffle=preshuffle,
                KVBlockSize=64,
                ChunkK=64,
                TotalCuCount=1,
                WavePerEU=1,
            )
            torch.cuda.synchronize()
            outputs.append(out[0, 0, :length])
        if not torch.isfinite(outputs[1]).all():
            raise AssertionError(f"preshuffled logits contain non-finite values at {length}")
        torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
        results.append({"context_tokens": length, "max_abs_diff": 0.0})
    print(json.dumps({"status": "pass", "cases": results}))


if __name__ == "__main__":
    main()
