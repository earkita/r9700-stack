# SPDX-License-Identifier: Apache-2.0
"""Small-M BF16 linear for GLM's load-time MXFP4 dense fallback.

The checkpoint's packed weights are dequantized once by SGLang at load time.
This kernel reads that BF16 matrix directly and is opt-in for one-row decode;
prefill and all other shapes keep SGLang's existing linear path.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv(X, W, Y, K: tl.constexpr, N: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr):
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    partial = tl.zeros((BN, BK), tl.float32)
    for start in range(0, K, BK):
        offsets = start + ks
        x = tl.load(X + offsets, offsets < K, other=0).to(tl.float32)
        w = tl.load(W + cols[:, None] * K + offsets[None, :],
                    (cols[:, None] < N) & (offsets[None, :] < K),
                    other=0).to(tl.float32)
        partial += w * x[None, :]
    values = tl.sum(partial, axis=1)
    tl.store(Y + cols, values, cols < N)


def linear_one_row(x: torch.Tensor, weight: torch.Tensor, bias=None,
                   block_n: int = 8, block_k: int = 256,
                   num_warps: int = 4) -> torch.Tensor:
    """Return ``x @ weight.T`` for the qualified one-row contiguous layout."""
    if (x.ndim != 2 or x.shape[0] != 1 or weight.ndim != 2 or
            x.shape[1] != weight.shape[1] or x.dtype != torch.bfloat16 or
            weight.dtype != torch.bfloat16 or not x.is_contiguous() or
            not weight.is_contiguous() or x.device.type != "cuda" or
            x.device != weight.device or block_n not in (4, 8, 16, 32) or
            block_k not in (128, 256, 512, 1024)):
        raise ValueError("BF16 GEMV requires contiguous one-row BF16 CUDA inputs")
    n, k = weight.shape
    out = torch.empty((1, n), device=x.device, dtype=torch.bfloat16)
    _gemv[(triton.cdiv(n, block_n),)](
        x, weight, out, k, n, block_n, block_k, num_warps=num_warps,
        enable_fp_fusion=False)
    if bias is not None:
        out = out + bias
    return out
