"""Numerical probe for the gfx1201 paged FP8 logits used by KPool=4.

Run inside the pinned SGLang image; this is not a performance benchmark.
"""

import json

import torch

from sglang.kernels.ops.attention.dsa.paged_mqa_logits import aiter_paged_mqa_logits


def main():
    torch.manual_seed(615)
    device = "cuda"
    rows, heads, dim, page_size, pages = 2, 32, 128, 64, 2
    q = (torch.randn(rows, heads, dim, device=device) * 0.25).to(torch.float8_e4m3fn)
    k = (torch.randn(pages, page_size, dim, device=device) * 0.25).to(torch.float8_e4m3fn)
    scales = torch.full((pages, page_size, 1), 0.75, device=device, dtype=torch.float32)
    # SGLang stores every page's 64 FP8 keys first, then 64 FP32 scales.
    packed = torch.cat(
        (k.view(torch.uint8).reshape(pages, -1),
         scales.view(torch.uint8).reshape(pages, -1)),
        dim=-1,
    ).view(pages, page_size, 1, dim + 4).contiguous()
    weights = torch.rand(rows, heads, device=device, dtype=torch.float32)
    lengths = torch.tensor([6, 9], device=device, dtype=torch.int32)
    pages_for_rows = torch.tensor([[0], [1]], device=device, dtype=torch.int32)
    actual = aiter_paged_mqa_logits(
        q, packed, weights, lengths, pages_for_rows, page_size,
        preshuffle=False, kv_block_size=page_size,
    )
    torch.cuda.synchronize()
    errors = []
    for row, length in enumerate(lengths.tolist()):
        key = k[row, :length].float() * scales[row, :length].float()
        expected = (torch.einsum("hd,kd->hk", q[row].float(), key).clamp_min(0) * weights[row, :, None]).sum(0)
        errors.append((actual[row, :length] - expected).abs())
    error = torch.cat(errors)
    report = {"max_abs_error": error.max().item(), "mean_abs_error": error.mean().item()}
    print(json.dumps(report))
    assert error.max().item() < 0.1, report


if __name__ == "__main__":
    main()
