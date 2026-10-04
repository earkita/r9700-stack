"""Opt-in DeepSeek Quark correctness fallback on the separately pinned image.

Upstream owns checkpoint loading and every MoE path. Only the MXFP8 linear
emulation fallback needs activation QDQ: upstream's native MX kernel performs
it, but EmulationMxfp8LinearKernel feeds BF16 activations directly to F.linear.
"""

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.kernels.linear.mxfp8.emulation import EmulationMxfp8LinearKernel
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig, QuarkLinearMethod
from vllm.model_executor.layers.quantization.quark.schemes import QuarkOCP_MX
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch, dequant_mxfp8_to_bf16, mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic, kMxfp8Static


class DeepseekQuarkLinearMethod(QuarkLinearMethod):
    def apply(self, layer, x, bias=None):
        if isinstance(layer.scheme.ocp_mx_linear, EmulationMxfp8LinearKernel):
            if x.dtype != torch.bfloat16 or x.shape[-1] % 32:
                raise ValueError("DeepSeek MXFP8 emulation requires BF16, group-32 inputs")
            # Reuse upstream's ROCm quantizer (including zero-block handling).
            # The torch reference is for CPU tests. Flatten to the ROCm 2-D path.
            shape = x.shape
            flat = x.reshape(-1, shape[-1]).contiguous()
            quantize = mxfp8_e4m3_quantize if x.is_cuda else _mxfp8_e4m3_quantize_torch
            values, scales = quantize(flat)
            x = dequant_mxfp8_to_bf16(values, scales).view(shape)
        return super().apply(layer, x, bias)


@register_quantization_config("r9700_deepseek_quark")
class DeepseekQuarkConfig(QuarkConfig):
    @classmethod
    def get_name(cls):
        return "r9700_deepseek_quark"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if user_quant != cls.get_name():
            return None
        if (getattr(hf_config, "model_type", None) not in ("deepseek_v41", "deepseek_v41_text")
                or hf_quant_cfg.get("quant_method") != "quark"):
            raise ValueError("r9700_deepseek_quark requires a DeepSeek V4.1 Quark checkpoint")
        return cls.get_name()

    def get_quant_method(self, layer, prefix):
        cfg = get_current_vllm_config()
        if getattr(cfg.model_config.hf_config, "model_type", None) not in (
                "deepseek_v41", "deepseek_v41_text"):
            raise ValueError("DeepSeek Quark adapter cannot be used by other models")
        method = super().get_quant_method(layer, prefix)
        scheme = getattr(layer, "scheme", None)
        if (isinstance(method, QuarkLinearMethod) and isinstance(scheme, QuarkOCP_MX)
                and scheme.weight_quant_key == kMxfp8Static
                and scheme.activation_quant_key == kMxfp8Dynamic):
            return DeepseekQuarkLinearMethod(self)
        return method
