# SPDX-License-Identifier: Apache-2.0
"""GLM C1 adapter for the repository's gfx1201 HIP W4A4 MoE kernel.

The kernel is shared with the vLLM plugin, but its input contract is checked
here against the pinned SGLang Quark layout. Larger batches retain the packed
GEMV implementation until the HIP grouped path is qualified separately.
"""
from __future__ import annotations

import ctypes
import os
from functools import lru_cache

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _encode_qdq(X, Q, S):
    """Power-of-two FP8 encoding of already-MXFP4-QDQ BF16 groups of 32."""
    group = tl.program_id(0)
    col = tl.arange(0, 32)
    x = tl.load(X + group * 32 + col).to(tl.float32)
    amax = tl.max(tl.abs(x), 0)
    bits = amax.to(tl.int32, bitcast=True) & 0x7F800000
    bits = tl.maximum(bits, 0x00800000)
    scale = tl.where(amax == 0, 1.0, bits.to(tl.float32, bitcast=True))
    tl.store(Q + group * 32 + col, x / scale)
    tl.store(S + group, scale)


def encode_qdq(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode finite Quark-MXFP4-QDQ BF16, not arbitrary BF16 input."""
    if (x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] % 32
            or not x.is_contiguous()):
        raise ValueError("Expected contiguous [rows,K] BF16 QDQ, K divisible by 32")
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((x.shape[0], x.shape[1] // 32), device=x.device,
                         dtype=torch.float32)
    _encode_qdq[(scales.numel(),)](x, q, scales, num_warps=1)
    return q, scales


@lru_cache(maxsize=2)
def _kernel(grouped=False):
    path = os.environ.get("R9K_SGLANG_W4A4_LIB", "/opt/r9700_sglang/libr9k_moe.so")
    lib = ctypes.CDLL(path)
    fn = getattr(lib, "r9k_moe_mxfp4a4_batch" if grouped else "r9k_moe_mxfp4a4_c1")
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_long] * 9 + [ctypes.c_int] * 7 + [ctypes.c_long]
    return fn


def _alignment(ids: torch.Tensor):
    """One live routed slot in each 16-row HIP block; padding is never stored."""
    slots = ids.numel()
    sorted_ids = torch.full((slots, 16), slots, device=ids.device, dtype=torch.int32)
    sorted_ids[:, 0] = torch.arange(slots, device=ids.device, dtype=torch.int32)
    count = torch.full((1,), slots * 16, device=ids.device, dtype=torch.int32)
    return sorted_ids.flatten(), ids.flatten(), count


def _gemm(q, scales, weight, weight_scales, output, alignment,
          slots, row_divisor, router_weights=None, grouped=False):
    n, k = weight.shape[1], weight.shape[2] * 2
    sid, eid, count = alignment
    if (weight.dtype != torch.uint8 or weight_scales.dtype != torch.uint8
            or weight_scales.shape != (weight.shape[0], n, k // 32)
            or not weight.is_contiguous() or not weight_scales.is_contiguous()):
        raise ValueError("Expected original contiguous Quark MXFP4 weights and E8M0 scales")
    wv, sk = (2, 4) if k >= 1024 else (4, 2)
    rc = _kernel(grouped)(q.data_ptr(), scales.data_ptr(), weight.data_ptr(),
                   weight_scales.data_ptr(), output.data_ptr(), sid.data_ptr(),
                   eid.data_ptr(), count.data_ptr(),
                   router_weights.data_ptr() if router_weights is not None else 0,
                   eid.numel() if grouped else slots, slots, row_divisor, k, n, wv, sk,
                   torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"r9k_moe_mxfp4a4_c1 failed: {rc}")


def apply_sglang(layer, dispatch_output):
    """Compute one token's eight routed experts using original Quark layouts."""
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import quant_dequant_mxfp4

    topk = dispatch_output.topk_output
    ids = topk.topk_ids.contiguous()
    weights = topk.topk_weights.contiguous()
    hidden = dispatch_output.hidden_states
    if (hidden.shape != (1, 4096) or hidden.dtype != torch.bfloat16
            or ids.shape != (1, 8) or ids.dtype != torch.int32
            or weights.shape != (1, 8) or weights.dtype != torch.float32):
        raise ValueError("HIP W4A4 adapter requires GLM TP8 C1, topk8 BF16/FP32 routing")
    if (layer.w13_weight.shape != (288, 512, 2048)
            or layer.w2_weight.shape != (288, 4096, 128)
            or layer.moe_runner_config.swiglu_limit != 10.0
            or layer.moe_runner_config.activation != "silu"):
        raise ValueError("HIP W4A4 adapter requires audited GLM TP8 Quark MoE shapes")
    if os.environ.get("R9K_FOLD", "0") != "0":
        raise ValueError("HIP W4A4 candidate requires R9K_FOLD=0")
    if not torch.cuda.get_device_properties(hidden.device).gcnArchName.startswith("gfx1201"):
        raise ValueError("HIP W4A4 candidate requires gfx1201")

    alignment = _alignment(ids)
    q, scales = encode_qdq(quant_dequant_mxfp4(hidden.contiguous()))
    gate_up = torch.empty((8, 512), device=hidden.device, dtype=torch.bfloat16)
    _gemm(q, scales, layer.w13_weight, layer.w13_weight_scale,
          gate_up, alignment, 8, 8)
    gate, up = gate_up.float().split(256, dim=-1)
    mid = (F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)).to(torch.bfloat16)
    q2, scales2 = encode_qdq(quant_dequant_mxfp4(mid.contiguous()))
    down = torch.empty((8, 4096), device=hidden.device, dtype=torch.bfloat16)
    _gemm(q2, scales2, layer.w2_weight, layer.w2_weight_scale,
          down, alignment, 8, 1, weights.flatten())
    return StandardCombineInput(down.view(1, 8, 4096).sum(dim=1))
