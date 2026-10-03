"""Native vLLM FP8 checkpoints with MXFP4 MoE weights on gfx1201.

MiMo-V2.6 declares ``quant_method=fp8`` and ``store_dtype=mxfp4``. Stock vLLM
therefore uses its native :class:`Mxfp4MoEMethod`, rather than the
compressed-tensors override used by the Qwen profiles. On R9700 that path
otherwise selects the generic MXFP4 emulation backend. This registered config
changes only native MXFP4 RoutedExperts and preserves stock FP8 handling for
dense linears, attention and ignored layers. The adapter gates gfx1201 use.
"""

from __future__ import annotations

import os

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

from ..moe import prep
from ..moe.experts import R9700Mxfp4Experts, r9k_available

logger = init_logger("vllm." + __name__)


def _moe_disabled() -> bool:
    disabled = {item.strip() for item in os.environ.get("R9K_DISABLE", "").split(",")}
    return "moe" in disabled


class R9kNativeMxfp4MoEMethod(Mxfp4MoEMethod):
    """Native MXFP4 loader plus libr9k grouped MXFP4 x FP8 expert GEMMs."""

    def __init__(self, moe):
        # Keep stock sizing and checkpoint loaders. Only the post-load layout
        # conversion and expert implementation differ.
        super().__init__(moe)
        self.experts_cls = R9700Mxfp4Experts

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if (
            getattr(layer, "w13_bias", None) is not None
            or getattr(layer, "w2_bias", None) is not None
        ):
            raise NotImplementedError("r9700 native MXFP4 MoE does not support bias")

        w13_param = layer.w13_weight
        w2_param = layer.w2_weight
        w13 = prep.permute_in_place(w13_param.data)
        w2 = prep.permute_in_place(w2_param.data)
        device = torch.device("cuda", torch.cuda.current_device())
        s13 = prep.pack_scales_gpu(layer.w13_weight_scale.data, device)
        s2 = prep.pack_scales_gpu(layer.w2_weight_scale.data, device)

        layer.w13_weight = prep.as_param(w13, w13_param)
        layer.w2_weight = prep.as_param(w2, w2_param)
        layer.w13_weight_scale = Parameter(s13, requires_grad=False)
        layer.w2_weight_scale = Parameter(s2, requires_grad=False)

        from ..kernels.moe import fold_decide

        name = getattr(layer, "layer_name", "") or "moe"
        layer._r9k_fold = (
            fold_decide(s13, name + ".w13", logger.info),
            fold_decide(s2, name + ".w2", logger.info),
        )

        from vllm.model_executor.layers.fused_moe.config import (
            mxfp4_w4a16_moe_quant_config,
        )
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
            make_mxfp4_moe_kernel,
        )

        self.moe_quant_config = mxfp4_w4a16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        )
        self.moe_kernel = make_mxfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=R9700Mxfp4Experts,
            mxfp4_backend=self.mxfp4_backend,
            routing_tables=layer._expert_routing_tables(),
        )
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)
        prep.maybe_attach_cache(self, layer)
        logger.info_once(
            "r9700: native FP8/MXFP4 MoE -> R9700Mxfp4Experts "
            "(libr9k grouped MXFP4xFP8)"
        )


@register_quantization_config("r9700_mimo_mxfp4")
class R9kFp8Config(Fp8Config):
    """Explicit MiMo config; stock FP8 handling for dense layers."""

    @classmethod
    def get_name(cls):
        return "r9700_mimo_mxfp4"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if user_quant == cls.get_name():
            if (
                getattr(hf_config, "model_type", None) != "mimo_v2"
                or hf_quant_cfg.get("store_dtype") != "mxfp4"
            ):
                raise ValueError("MiMo quantization requires mimo_v2/MXFP4")
            return cls.get_name()
        return None

    def get_quant_method(self, layer, prefix):
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
        if cfg.model_config.hf_config.model_type != "mimo_v2":
            raise RuntimeError("r9700_mimo_mxfp4 requires mimo_v2")
        method = super().get_quant_method(layer, prefix)
        if (
            isinstance(layer, RoutedExperts)
            and self.store_dtype == "mxfp4"
            and isinstance(method, Mxfp4MoEMethod)
        ):
            if not r9k_available() or _moe_disabled():
                raise RuntimeError(
                    "Requested MiMo native MXFP4 kernels are unavailable or disabled"
                )
            return R9kNativeMxfp4MoEMethod(layer.moe_config)
        return method
