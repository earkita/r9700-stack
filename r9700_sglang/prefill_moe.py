# SPDX-License-Identifier: Apache-2.0
"""Opt-in prefill candidate using the existing multirow gfx1201 W4A4 kernel.

Preserve original Quark weights/scales and MXFP4 activation QDQ. The existing
vLLM multirow entry reuses expert weights across 16 real routed rows, retaining
each row's group-32 scale. This is not the per-tensor-scale W4A8 prefill entry.
Decode/verify stay on the independently qualified packed path. GPU numerical
and model qualification are required before using this experimental candidate.
"""
import os

import torch
import torch.nn.functional as F

from .hip_w4a4 import _gemm, encode_qdq


def apply_sglang(layer, dispatch_output):
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import quant_dequant_mxfp4

    hidden = dispatch_output.hidden_states
    topk = dispatch_output.topk_output
    ids = topk.topk_ids.contiguous()
    weights = topk.topk_weights.contiguous()
    if (hidden.ndim != 2 or hidden.shape[0] < 128 or hidden.shape[1] != 4096
            or hidden.dtype != torch.bfloat16 or ids.shape != (hidden.shape[0], 8)
            or ids.dtype != torch.int32 or weights.shape != ids.shape
            or weights.dtype != torch.float32):
        raise ValueError("Grouped prefill requires GLM TP8, >=128 BF16 rows and topk8 FP32 routes")
    if (layer.w13_weight.shape != (288, 512, 2048)
            or layer.w2_weight.shape != (288, 4096, 128)
            or layer.moe_runner_config.swiglu_limit != 10.0
            or layer.moe_runner_config.activation != "silu"):
        raise ValueError("Grouped prefill requires audited original GLM TP8 expert layouts")
    if os.environ.get("R9K_FOLD", "0") != "0":
        raise ValueError("Grouped W4A4 prefill forbids exponent folding")
    if torch.cuda.get_device_properties(hidden.device).gcnArchName.split(":")[0] != "gfx1201":
        raise ValueError("Grouped W4A4 prefill requires gfx1201")

    slots = ids.numel()
    alignment = moe_align_block_size(ids, 16, 288)
    q, scales = encode_qdq(quant_dequant_mxfp4(hidden.contiguous()))
    gate_up = torch.empty((slots, 512), device=hidden.device, dtype=torch.bfloat16)
    _gemm(q, scales, layer.w13_weight, layer.w13_weight_scale,
          gate_up, alignment, slots, 8, grouped=True)
    gate, up = gate_up.float().chunk(2, dim=-1)
    mid = (F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)).bfloat16()
    q2, scales2 = encode_qdq(quant_dequant_mxfp4(mid.contiguous()))
    down = torch.empty((slots, 4096), device=hidden.device, dtype=torch.bfloat16)
    _gemm(q2, scales2, layer.w2_weight, layer.w2_weight_scale,
          down, alignment, slots, 1, weights.flatten(), grouped=True)
    return StandardCombineInput(down.view(hidden.shape[0], 8, 4096).sum(dim=1))
