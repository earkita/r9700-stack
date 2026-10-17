"""Standalone GPU reference check for SGLang's Triton DSA KPool tail.

Run inside the pinned SGLang image with ``python3 /work/test_sglang_dsa_tail.py``.
This is a correctness probe, not a performance benchmark.
"""

import json

import torch

from sglang.kernels.ops.attention.dsa.triton_sparse_mla import triton_sparse_mla_fwd
from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
    triton_sparse_mla_decode_splitk,
)


def reference(q_nope, q_rope, kv, indices, scale):
    q = torch.cat((q_nope, q_rope), dim=-1).float()
    outputs = []
    for row in range(q.shape[0]):
        selected = indices[row, 0]
        selected = selected[selected >= 0].long()
        keys = kv[selected, 0].float()
        scores = q[row] @ keys.T * scale
        probs = scores.softmax(dim=-1)
        outputs.append(probs @ keys[:, : q_nope.shape[-1]])
    return torch.stack(outputs).bfloat16()


def check(rope_dim):
    torch.manual_seed(427)
    device = "cuda"
    rows, heads, value_dim, pages = 2, 8, 512, 4096
    kv = (torch.randn(pages, 1, value_dim + rope_dim, device=device) * 0.1).bfloat16()
    q_nope = (torch.randn(rows, heads, value_dim, device=device) * 0.1).bfloat16()
    q_rope = (torch.randn(rows, heads, rope_dim, device=device) * 0.1).bfloat16()
    indices = torch.full((rows, 1, 2051), -1, device=device, dtype=torch.int32)
    indices[0, 0, :1536] = torch.randperm(pages, device=device)[:1536].int()
    indices[1, 0, :2048] = torch.randperm(pages, device=device)[:2048].int()
    indices[:, 0, 2048:] = torch.tensor(
        [[3100, 3101, 3102], [3200, 3201, 3202]], device=device, dtype=torch.int32
    )
    scale = (value_dim + rope_dim) ** -0.5
    expected = reference(q_nope, q_rope, kv, indices, scale)
    result = {}
    for name, kernel in (
        ("prefill", triton_sparse_mla_fwd),
        ("decode", triton_sparse_mla_decode_splitk),
    ):
        actual = kernel(
            q_nope=q_nope,
            q_rope=q_rope,
            kv=kv,
            indices=indices,
            sm_scale=scale,
            d_v=value_dim,
        ).squeeze(0)
        torch.cuda.synchronize()
        error = (actual.float() - expected.float()).abs()
        result[name] = {"max_abs_error": error.max().item(), "mean_abs_error": error.mean().item()}
        assert torch.allclose(actual.float(), expected.float(), atol=0.03, rtol=0.03), result
    return result


def main():
    print(json.dumps({rope_dim: check(rope_dim) for rope_dim in (0, 64)}, indent=2))


if __name__ == "__main__":
    main()
