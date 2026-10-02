"""Quark MXFP4 MoE using the existing grouped HIP WMMA C1 geometry.

Quark still rounds activations to MXFP4. FP8 is only a lossless transport for
those rounded values, with a separate power-of-two scale per 32 elements.
Weights stay in their original packed layout; the kernel unpacks only the
selected experts' fragments in registers. No full BF16 weight allocation.
"""
from __future__ import annotations

import ctypes

import torch
from vllm.triton_utils import tl, triton

from ..kernels import moe as K


@triton.jit
def _encode_qdq(X, Q, S):
    group = tl.program_id(0)
    col = tl.arange(0, 32)
    x = tl.load(X + group * 32 + col).to(tl.float32)
    amax = tl.max(tl.abs(x), 0)
    # Normalize by a power of two, never by a new FP8 quantization scale.
    # MXFP4 values in a group have <= 2 significant bits. Their normalized
    # range fits E4M3 exactly; BF16 subnormals also fit after normalization.
    bits = amax.to(tl.int32, bitcast=True) & 0x7F800000
    bits = tl.maximum(bits, 0x00800000)
    scale = tl.where(amax == 0, 1.0, bits.to(tl.float32, bitcast=True))
    tl.store(Q + group * 32 + col, x / scale)
    tl.store(S + group, scale)


def encode_qdq(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Losslessly encode finite, group-32 MXFP4-QDQ BF16 values for WMMA.

    This is not a quantizer for arbitrary BF16 input. The caller must first
    apply the pinned Quark QDQ primitive; tests check exact reconstruction.
    """
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] % 32 or not x.is_contiguous():
        raise ValueError("Expected contiguous [rows,K] BF16 MXFP4-QDQ values, K divisible by 32")
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((x.shape[0], x.shape[1] // 32), device=x.device, dtype=torch.float32)
    _encode_qdq[(scales.numel(),)](x, q, scales, num_warps=1)
    return q, scales


_GEMM = {}


def kernel(grouped=False):
    if grouped not in _GEMM:
        f = getattr(K.lib(), "r9k_moe_mxfp4a4_batch" if grouped else "r9k_moe_mxfp4a4_c1")
        f.restype = ctypes.c_int
        f.argtypes = [ctypes.c_long] * 9 + [ctypes.c_int] * 7 + [ctypes.c_long]
        _GEMM[grouped] = f
    return _GEMM[grouped]


def gemm(q, scales, weight, weight_scales, output, alignment, slots, row_divisor, router_weights=None,
         grouped=False):
    """The C1 entry needs isolated slots; grouped supports independent row scales."""
    n, k = weight.shape[1], weight.shape[2] * 2
    sid, eid, ntpp = alignment
    if (weight.dtype != torch.uint8 or weight_scales.dtype != torch.uint8
            or weight_scales.shape != (weight.shape[0], n, k // 32)
            or not weight.is_contiguous() or not weight_scales.is_contiguous()):
        raise ValueError("Expected original contiguous Quark packed weights and E8M0 scales")
    # Conservative legal configs for TP8 gate/up and down; no tuning claim.
    wv, sk = (2, 4) if k >= 1024 else (4, 2)
    rc = kernel(grouped)(q.data_ptr(), scales.data_ptr(), weight.data_ptr(), weight_scales.data_ptr(),
                  output.data_ptr(), sid.data_ptr(), eid.data_ptr(), ntpp.data_ptr(),
                  router_weights.data_ptr() if router_weights is not None else 0,
                  eid.numel(), slots, row_divisor, k, n, wv, sk, K._stream())
    if rc:
        raise RuntimeError(f"r9k_moe_mxfp4a4_c1 failed: {rc}")


def apply_c1(experts, output, hidden_states, w1, w2, topk_weights, topk_ids, activation):
    """Called only after the scoped adapter's shape/parallel/quantization gate."""
    assert hidden_states.shape[0] == 1
    return _apply(experts, output, hidden_states, w1, w2, topk_weights, topk_ids, activation)


def apply_isolated_rows(experts, output, hidden_states, w1, w2, topk_weights, topk_ids, activation):
    """Reuse the C1 kernel for 2–8 rows with one valid routed slot per block.

    Activation rows are repeated per routed slot, so both projections use
    row_divisor=1. Every block has only one activation scale, preserving the
    existing kernel's C1 invariant even when experts repeat across tokens.
    """
    assert 2 <= hidden_states.shape[0] <= 8
    return _apply(experts, output, hidden_states, w1, w2, topk_weights, topk_ids, activation)


def apply_grouped(experts, output, hidden_states, w1, w2, topk_weights, topk_ids, activation):
    """Share expert weights across real rows, retaining their individual scales."""
    return _apply(experts, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, grouped=True)


def _apply(experts, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, grouped=False):
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import quant_dequant_mxfp4

    rows, topk = topk_ids.shape
    slots = topk_ids.numel()
    if rows == 1 or grouped:
        alignment = moe_align_block_size(topk_ids, 16, w1.shape[0], None)
    else:
        # Static slot geometry; expert IDs remain a live view of the routes.
        # Padding repeats each block's first activation inside the HIP kernel.
        cache = getattr(experts, "_r9k_isolated_alignment", None)
        if cache is None:
            cache = experts._r9k_isolated_alignment = {}
        key = (rows, topk, hidden_states.device)
        if key not in cache:
            sid = torch.full((slots, 16), slots, device=topk_ids.device, dtype=torch.int32)
            sid[:, 0] = torch.arange(slots, device=topk_ids.device, dtype=torch.int32)
            ntpp = torch.full((1,), slots * 16, device=topk_ids.device, dtype=torch.int32)
            cache[key] = (sid.flatten(), ntpp)
        sid, ntpp = cache[key]
        alignment = (sid, topk_ids.flatten(), ntpp)
    q, scales = encode_qdq(quant_dequant_mxfp4(hidden_states))
    if rows > 1 and not grouped:
        q = q.repeat_interleave(topk, dim=0)
        scales = scales.repeat_interleave(topk, dim=0)
    gate_up = torch.empty((slots, w1.shape[1]), device=hidden_states.device, dtype=torch.bfloat16)
    gemm(q, scales, w1, experts.w1_scale_val, gate_up, alignment, slots,
         topk if grouped else (slots if rows == 1 else 1), grouped=grouped)
    intermediate = torch.empty((slots, w2.shape[2] * 2), device=hidden_states.device, dtype=torch.bfloat16)
    # Reuse pinned TritonExperts.activation: GLM's clamp=10, BF16 rounding,
    # then exactly the same activation QDQ as the baseline.
    experts.activation(activation, intermediate, gate_up)
    q2, scales2 = encode_qdq(quant_dequant_mxfp4(intermediate))
    down = torch.empty((slots, w2.shape[1]), device=hidden_states.device, dtype=torch.bfloat16)
    gemm(q2, scales2, w2, experts.w2_scale_val, down, alignment, slots, 1, topk_weights, grouped=grouped)
    ops.moe_sum(down.view(rows, topk, w2.shape[1]), output)
