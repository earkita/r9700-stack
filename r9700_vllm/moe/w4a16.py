"""Experimental GLM MXFP4 weights with BF16 activations, including clamp=10.

Reuse the GLM packed weight reader, whose GEMV already accepts BF16 inputs.
Unlike W4A4, neither projection applies activation QDQ. FP8 dense layers and
the KV format are independent of this path.
"""
import torch
from .packed_gemv import gemv


def apply(experts, output, hidden, w1, w2, weights, ids, activation):
    import vllm._custom_ops as ops
    rows, topk = ids.shape
    gu = gemv(hidden, w1, experts.w1_scale_val, ids, topk)
    mid = torch.empty((ids.numel(), w2.shape[2] * 2), device=hidden.device, dtype=torch.bfloat16)
    experts.activation(activation, mid, gu)
    down = gemv(mid, w2, experts.w2_scale_val, ids, 1, weights)
    ops.moe_sum(down.view(rows, topk, w2.shape[1]), output)
