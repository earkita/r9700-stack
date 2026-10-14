"""Opt-in Quark WO_A activation QDQ for the pinned gfx1201 BF16 BMM path.

Ordinary MXFP8 linear calls pass through DeepseekQuarkLinearMethod. WO_A
uses a grouped einsum instead and bypasses that method. Only that pre-rotated,
BF16 emulation path is changed; native quantized attention output is untouched.
Full-model quality is not established by the primitive contract tests.
"""
import inspect
import os
from functools import wraps


def quark_wo_a_input(wo_a, attn_out):
    import torch
    from vllm.model_executor.kernels.linear.mxfp8.emulation import EmulationMxfp8LinearKernel
    from vllm.model_executor.layers.quantization.quark.schemes import QuarkOCP_MX
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic, kMxfp8Static

    scheme = getattr(wo_a, "scheme", None)
    if not (isinstance(attn_out, torch.Tensor)
            and isinstance(scheme, QuarkOCP_MX)
            and scheme.weight_quant_key == kMxfp8Static
            and scheme.activation_quant_key == kMxfp8Dynamic
            and isinstance(scheme.ocp_mx_linear, EmulationMxfp8LinearKernel)):
        return attn_out
    if attn_out.dtype != torch.bfloat16 or attn_out.ndim != 3 or attn_out.shape[-1] % 32:
        raise ValueError("DeepSeek WO_A QDQ requires BF16 [tokens,heads,dim] with dim%32=0")
    if attn_out.numel() == 0:
        return attn_out
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        _mxfp8_e4m3_quantize_torch, dequant_mxfp8_to_bf16, mxfp8_e4m3_quantize,
    )
    flat = attn_out.reshape(attn_out.shape[0], -1).contiguous()
    quantize = mxfp8_e4m3_quantize if flat.is_cuda else _mxfp8_e4m3_quantize_torch
    values, scales = quantize(flat)
    return dequant_mxfp8_to_bf16(values, scales).view_as(attn_out)


def install():
    mode = os.environ.get("R9K_DEEPSEEK_WO_A_QDQ", "0")
    if mode == "0":
        return
    if mode != "1":
        raise ValueError("R9K_DEEPSEEK_WO_A_QDQ must be 0 or 1")
    # Called only by DeepSeek's vLLM-version / ROCm / gfx1201 registration gate.
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention
    from vllm.logger import init_logger
    cls = DeepseekV41ROCMAiterMLAAttention
    original = cls._o_proj
    if getattr(original, "_r9700_wo_a_qdq", False):
        return
    if tuple(inspect.signature(original).parameters) != ("self", "attn_out", "positions"):
        raise RuntimeError("DeepSeek WO_A QDQ: unsupported _o_proj interface")

    @wraps(original)
    def o_proj(self, attn_out, positions):
        return original(self, quark_wo_a_input(self.wo_a, attn_out), positions)

    o_proj._r9700_wo_a_qdq = True
    cls._o_proj = o_proj
    init_logger("vllm.r9700_vllm").info(
        "r9700: experimental DeepSeek Quark WO_A MXFP8 activation QDQ enabled")
