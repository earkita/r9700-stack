# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output-tiled packed MXFP4 GEMV for SGLang on gfx1201.

Input is already rounded by Quark's MXFP4 QDQ. Read only the routed experts,
unpack fragments in registers, accumulate in FP32 and store BF16. This does
not allocate dequantized weight tensors or change the activation format.
"""
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _e2m1(code):
    p = code & 7
    # For p>=2, IEEE exponent p//2 and the low bit encode the E2M1 value.
    mag = tl.where(p < 2, p.to(tl.float32) * .5,
                   ((p // 2 + 126) << 23 | (p % 2) << 22).to(tl.float32, bitcast=True))
    return tl.where((code & 8) != 0, -mag, mag)


@triton.jit
def _gemv(X, W, S, IDS, RW, OUT, K: tl.constexpr, N: tl.constexpr,
          DIV: tl.constexpr, WEIGHTED: tl.constexpr, BN: tl.constexpr,
          BK: tl.constexpr, GROUP_SCALE: tl.constexpr = False):
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
        if GROUP_SCALE:
            # One E8M0 scale covers 16 packed bytes (32 MXFP4 elements).
            # Load it once per group, then broadcast in registers. Keep the
            # original path available for matched model-level comparisons.
            groups = start // 16 + tl.arange(0, BK // 16)
            grouped = tl.load(S + expert * N * (K // 32)
                              + cols[:, None] * (K // 32) + groups[None, :],
                              mask=(cols[:, None] < N) & (groups[None, :] < K // 32),
                              other=127).to(tl.int32)
            indices = tl.broadcast_to((tl.arange(0, BK) // 16)[None, :], (BN, BK))
            scale = tl.gather(grouped, indices, 1)
        else:
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


def gemv(x, w, scales, ids, divisor, weights=None, block_n=8, num_warps=4,
         group_scale=False):
    # The caller gates the GLM shape and scale range 117..128. Unlike a
    # general MXFP4 decoder this intentionally excludes reserved E8M0 codes.
    n, k = w.shape[1], w.shape[2] * 2
    out = torch.empty((ids.numel(), n), device=x.device, dtype=torch.bfloat16)
    _gemv[(ids.numel(), triton.cdiv(n, block_n))](
        x, w, scales, ids, weights, out, k, n, divisor, weights is not None,
        block_n, min(256, k // 2), group_scale,
        num_warps=num_warps, enable_fp_fusion=False)
    return out


def apply_sglang(layer, dispatch_output):
    """Correctness fallback for GLM's Quark MoE on gfx1201.

    The pinned AITER FP4×FP4 path returned all-zero output on a real-shape
    synthetic test. This path keeps packed weights and original E8M0 scales;
    it is deliberately limited to the standard TP dispatcher.
    """
    prefill_backend = os.environ.get("R9K_SGLANG_PREFILL_MOE", "packed")
    if prefill_backend not in ("packed", "hip_w4a4_grouped"):
        raise ValueError("R9K_SGLANG_PREFILL_MOE must be packed or hip_w4a4_grouped")
    if prefill_backend == "hip_w4a4_grouped" and dispatch_output.hidden_states.shape[0] >= 128:
        from r9700_sglang.prefill_moe import apply_sglang as apply_prefill
        return apply_prefill(layer, dispatch_output)
    backend = os.environ.get("R9K_SGLANG_MOE_BACKEND", "packed")
    if backend not in ("packed", "hip_w4a4"):
        raise ValueError("R9K_SGLANG_MOE_BACKEND must be packed or hip_w4a4")
    if backend == "hip_w4a4" and dispatch_output.hidden_states.shape[0] == 1:
        from r9700_sglang.hip_w4a4 import apply_sglang as apply_hip_w4a4
        return apply_hip_w4a4(layer, dispatch_output)

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import quant_dequant_mxfp4

    topk = dispatch_output.topk_output
    ids = topk.topk_ids.contiguous()
    weights = topk.topk_weights.contiguous()
    hidden = dispatch_output.hidden_states
    if hidden.ndim != 2 or ids.ndim != 2 or weights.shape != ids.shape:
        raise ValueError("gfx1201 packed MoE requires standard [rows, topk] routing")
    if ids.shape[1] != 8 or hidden.shape[1] != 4096:
        raise ValueError("gfx1201 packed MoE is qualified only for GLM TP8 topk8")
    if layer.w13_weight.dtype != torch.uint8 or layer.w2_weight.dtype != torch.uint8:
        raise ValueError("gfx1201 packed MoE requires original MXFP4 weights")

    rows, topk = ids.shape
    w1, w2 = layer.w13_weight, layer.w2_weight
    limit = layer.moe_runner_config.swiglu_limit
    if limit != 10.0 or layer.moe_runner_config.activation != "silu":
        raise ValueError("gfx1201 packed MoE requires GLM SiLU with clamp=10")
    # Opt-in first experiment. The checkpoint stores one E8M0 scale per 32
    # values (16 packed bytes); only gate/up's redundant loads change. The
    # default preserves S0 exactly, including down and the output arithmetic.
    group_scale = os.environ.get("R9K_SGLANG_GATE_SCALE_GROUP", "0")
    if group_scale not in ("0", "1"):
        raise ValueError("R9K_SGLANG_GATE_SCALE_GROUP must be 0 or 1")
    down_bn = os.environ.get("R9K_SGLANG_DOWN_BN", "8")
    if down_bn not in ("8", "64"):
        raise ValueError("R9K_SGLANG_DOWN_BN must be 8 or 64")
    gu = gemv(quant_dequant_mxfp4(hidden.contiguous()), w1,
              layer.w13_weight_scale, ids.flatten(), topk,
              group_scale=group_scale == "1")
    gate, up = gu.float().split(gu.shape[1] // 2, dim=-1)
    mid = (F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)).to(torch.bfloat16)
    down = gemv(quant_dequant_mxfp4(mid.contiguous()), w2,
                layer.w2_weight_scale, ids.flatten(), 1, weights.flatten(),
                block_n=int(down_bn))
    return StandardCombineInput(down.view(rows, topk, w2.shape[1]).sum(dim=1))
