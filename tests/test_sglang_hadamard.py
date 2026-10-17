"""Numerical smoke for the SGLang gfx1201 indexer transform.

Run in the pinned GPU image; this is not a throughput benchmark.
"""

import json

import torch

from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation


def butterfly(x):
    width = x.shape[-1]
    result = x.float()
    step = 1
    while step < width:
        blocks = result.reshape(*result.shape[:-1], -1, 2, step)
        result = torch.cat((blocks[..., 0, :] + blocks[..., 1, :],
                            blocks[..., 0, :] - blocks[..., 1, :]), dim=-1).reshape_as(result)
        step *= 2
    return (result / width**0.5).to(x.dtype)


def main():
    torch.manual_seed(123)
    x = torch.randn(3, 32, 128, device="cuda", dtype=torch.bfloat16)
    actual = rotate_activation(x)
    expected = butterfly(x)
    error = (actual.float() - expected.float()).abs()
    report = {"max_abs_error": error.max().item(), "mean_abs_error": error.mean().item()}
    print(json.dumps(report))
    assert torch.allclose(actual.float(), expected.float(), atol=0.016, rtol=0.01), report


if __name__ == "__main__":
    main()
