# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output-tiled packed MXFP4 GEMV, adapted from my-llm's RDNA4 kernel.

Input is already rounded by Quark's MXFP4 QDQ. Read only the routed experts,
unpack fragments in registers, accumulate in FP32 and store BF16. This does
not allocate dequantized weight tensors or change the activation format.
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _e2m1(code):
    p = code & 7
    # For p>=2, IEEE exponent p//2 and the low bit encode the E2M1 value.
    mag = tl.where(p < 2, p.to(tl.float32) * .5,
                   ((p // 2 + 126) << 23 | (p % 2) << 22).to(tl.float32, bitcast=True))
    return tl.where((code & 8) != 0, -mag, mag)


@triton.jit
def _gemv(X, W, S, IDS, RW, OUT, K: tl.constexpr, N: tl.constexpr,
          DIV: tl.constexpr, WEIGHTED: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    route = tl.program_id(0)
    expert = tl.load(IDS + route).to(tl.int64)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    pk = tl.arange(0, BK)
    acc = tl.zeros((BN,), tl.float32)
    for start in range(0, K // 2, BK):
        packed_k = start + pk
        mask = (cols[:, None] < N) & (packed_k[None, :] < K // 2)
        packed = tl.load(W + expert * N * (K // 2) + cols[:, None] * (K // 2)
                         + packed_k[None, :], mask=mask, other=0).to(tl.int32)
        scale = tl.load(S + expert * N * (K // 32) + cols[:, None] * (K // 32)
                        + packed_k[None, :] // 16, mask=mask, other=127).to(tl.int32)
        scale = (scale << 23).to(tl.float32, bitcast=True)
        even = tl.load(X + (route // DIV) * K + 2 * packed_k, packed_k < K // 2, 0).to(tl.float32)
        odd = tl.load(X + (route // DIV) * K + 2 * packed_k + 1, packed_k < K // 2, 0).to(tl.float32)
        products = (even[None, :] * _e2m1(packed & 15)
                    + odd[None, :] * _e2m1(packed >> 4)) * scale
        acc += tl.sum(products, 1)
    if WEIGHTED:
        acc *= tl.load(RW + route)
    tl.store(OUT + route * N + cols, acc, cols < N)


def gemv(x, w, scales, ids, divisor, weights=None, block_n=8):
    # The caller gates the GLM shape and scale range 117..128. Unlike a
    # general MXFP4 decoder this intentionally excludes reserved E8M0 codes.
    n, k = w.shape[1], w.shape[2] * 2
    out = torch.empty((ids.numel(), n), device=x.device, dtype=torch.bfloat16)
    _gemv[(ids.numel(), triton.cdiv(n, block_n))](
        x, w, scales, ids, weights, out, k, n, divisor, weights is not None,
        block_n, min(256, k // 2), num_warps=4, enable_fp_fusion=False)
    return out


def apply(experts, output, hidden, w1, w2, weights, ids, activation):
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import quant_dequant_mxfp4
    rows, topk = ids.shape
    gu = gemv(quant_dequant_mxfp4(hidden), w1, experts.w1_scale_val, ids, topk)
    mid = torch.empty((ids.numel(), w2.shape[2] * 2), device=hidden.device, dtype=torch.bfloat16)
    experts.activation(activation, mid, gu)
    down = gemv(quant_dequant_mxfp4(mid), w2, experts.w2_scale_val, ids, 1, weights)
    ops.moe_sum(down.view(rows, topk, w2.shape[1]), output)
