# SPDX-License-Identifier: Apache-2.0
"""Original Quark GLM NextN block-FP8 experts through upstream Triton MoE.

Keep serialized E4M3 weights and FP32 128x128 scales. No dequantization,
requantization or AITER shuffle. Target MXFP4 experts are unaffected.
"""
import torch

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.quark.schemes import QuarkMoEScheme


class NextNBlockFp8(QuarkMoEScheme):
    @classmethod
    def get_min_capability(cls):
        return 89

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **attrs):
        helper = Fp8MoEMethod(Fp8Config(is_checkpoint_fp8_serialized=True,
                                      weight_block_size=[128, 128]))
        helper.create_weights(layer, num_experts, hidden_size,
                              intermediate_size_per_partition, params_dtype, **attrs)
        # Quark serializes dequantization multipliers as weight_scale, whereas
        # upstream FP8 calls the same multipliers weight_scale_inv.
        for prefix in ("w13", "w2"):
            param = layer._parameters.pop(prefix + "_weight_scale_inv")
            layer.register_parameter(prefix + "_weight_scale", param)

    def process_weights_after_loading(self, layer):
        if torch.cuda.get_device_properties(layer.w13_weight.device).gcnArchName.split(":")[0] != "gfx1201":
            raise RuntimeError("GLM NextN block FP8 is qualified for gfx1201 only")
        for prefix in ("w13", "w2"):
            w, s = getattr(layer, prefix + "_weight"), getattr(layer, prefix + "_weight_scale")
            if w.dtype != torch.float8_e4m3fn or s.dtype != torch.float32:
                raise ValueError("GLM NextN must preserve E4M3fn weights and FP32 block scales")
            if s.shape != (w.shape[0], (w.shape[1]+127)//128, (w.shape[2]+127)//128):
                raise ValueError("GLM NextN block-scale shape mismatch")

    def create_moe_runner(self, layer, config):
        # GLM uses SiLU(clamp_max(gate, 10)) * clamp(up, -10, 10).
        # The stock gemm1_clamp_limit branch clamps *after* SiLU, which is
        # different. Keep swiglu_limit and use the scoped HIP activation patch.
        if (config.swiglu_limit != 10 or config.gemm1_alpha is not None
                or config.gemm1_clamp_limit is not None):
            raise ValueError("GLM NextN requires the original limit-only SiLU clamp=10")
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, config)

    def apply_weights(self, layer, dispatch_output):
        quant = TritonMoeQuantInfo(
            w13_weight=layer.w13_weight, w2_weight=layer.w2_weight,
            use_fp8_w8a8=True, w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale, block_shape=[128, 128],
            a13_scale=None, a2_scale=None)
        return self.runner.run(dispatch_output, quant)


@torch.compile
def swiglu_clamped_hip(x, limit):
    gate, up = x.float().chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate.clamp(max=limit))
            * up.clamp(-limit, limit)).to(x.dtype)
