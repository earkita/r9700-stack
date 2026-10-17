# SPDX-License-Identifier: Apache-2.0
"""Experimental HC4 residual mixing without the [M,4,4,H] temporary.

This module is not dispatched by the server. Qualify its arithmetic against
the pinned Torch implementation before adding an opt-in serving adapter.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _post(X, R, POST, COMB, Y, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    output = tl.program_id(1)
    column = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    mask = column < H
    x = tl.load(X + row * H + column, mask, 0).to(tl.float32)
    p = tl.load(POST + row * 4 + output)
    # Torch materializes each FP32 product before reducing input streams.
    # Disable FMA and keep those rounding boundaries in the fused version.
    c0 = tl.load(COMB + row * 16 + output)
    c1 = tl.load(COMB + row * 16 + 4 + output)
    c2 = tl.load(COMB + row * 16 + 8 + output)
    c3 = tl.load(COMB + row * 16 + 12 + output)
    r0 = tl.load(R + row * 4 * H + column, mask, 0).to(tl.float32)
    r1 = tl.load(R + row * 4 * H + H + column, mask, 0).to(tl.float32)
    r2 = tl.load(R + row * 4 * H + 2 * H + column, mask, 0).to(tl.float32)
    r3 = tl.load(R + row * 4 * H + 3 * H + column, mask, 0).to(tl.float32)
    mixed = ((c0 * r0 + c1 * r1) + c2 * r2) + c3 * r3
    value = p * x + mixed
    tl.store(Y + row * 4 * H + output * H + column, value, mask)


def post(x, residual, post_layer_mix, comb_res_mix):
    """Return the pinned Torch post layout, with FP32 arithmetic/BF16 output."""
    if (x.ndim != 2 or x.shape[1] != 4096 or x.shape[0] < 1 or
            residual.shape != (x.shape[0], 4, 4096) or
            post_layer_mix.shape != (x.shape[0], 4, 1) or
            comb_res_mix.shape != (x.shape[0], 4, 4) or
            x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16 or
            post_layer_mix.dtype != torch.float32 or comb_res_mix.dtype != torch.float32 or
            not x.is_cuda or any(t.device != x.device or not t.is_contiguous()
                                for t in (x, residual, post_layer_mix, comb_res_mix))):
        raise ValueError("mHC post requires contiguous GLM BF16 HC4/H4096 and FP32 mixing")
    arch = torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0]
    if arch != "gfx1201":
        raise ValueError("Experimental mHC post requires gfx1201")
    result = torch.empty_like(residual)
    _post[(x.shape[0], 4, triton.cdiv(4096, 256))](
        x, residual, post_layer_mix, comb_res_mix, result, 4096, 256,
        num_warps=4, enable_fp_fusion=False)
    return result
