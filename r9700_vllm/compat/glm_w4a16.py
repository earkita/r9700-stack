"""Opt-in W4A16 experiment for GLM5Next Quark MXFP4 on the audited pin.

Change only MXFP4 activation metadata on a private QuarkConfig copy. Both
prefill fallback and shared experts then use weight-only upstream semantics;
packed decode uses the matching BF16-input implementation. Checkpoint files,
FP8 dense-layer configuration, KV settings and other model families stay intact.
"""
from copy import deepcopy


def weight_only_config(config):
    result = deepcopy(config)
    configs = [result.get("global_quant_config", {})]
    for name in ("layer_quant_config", "layer_type_quant_config"):
        configs.extend((result.get(name) or {}).values())
    changed = 0
    for c in configs:
        w, a = c.get("weight"), c.get("input_tensors")
        if not isinstance(w, dict) or w.get("dtype") != "fp4":
            continue
        if (w.get("group_size") != 32 or w.get("scale_format") != "e8m0"
                or w.get("qscheme") != "per_group"):
            raise ValueError("GLM W4A16 requires group-32 MXFP4/E8M0 weights")
        if a is None:
            continue
        if (not isinstance(a, dict) or a.get("dtype") != "fp4"
                or a.get("group_size") != 32 or a.get("scale_format") != "e8m0"
                or not a.get("is_dynamic")):
            raise ValueError("GLM W4A16 only replaces dynamic MXFP4 activation quantization")
        c["input_tensors"] = None
        changed += 1
    return result, changed


def eligible(experts, hidden, w1, w2, weights, ids, activation,
             global_num_experts, expert_map, a1q_scale, a2_scale, router_input):
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import OCP_MX_Scheme
    return (
        getattr(experts, "_r9k_glm_w4a16", False)
        and experts.ocp_mx_scheme == OCP_MX_Scheme.w_mxfp4
        and experts.quant_dtype is None
        and activation == MoEActivation.SILU and experts.activation_config.clamp_limit == 10.0
        and getattr(experts, "_lora_context", None) is None
        and getattr(experts, "w1_bias", None) is None and getattr(experts, "w2_bias", None) is None
        and hidden.ndim == 2 and hidden.shape[1] == 4096 and 1 <= hidden.shape[0] <= 32
        and hidden.dtype == torch.bfloat16
        and w1.shape == (288, 512, 2048) and w2.shape == (288, 4096, 128)
        and w1.dtype == w2.dtype == torch.uint8
        and experts.w1_scale_val.shape == (288, 512, 128)
        and experts.w2_scale_val.shape == (288, 4096, 8)
        and experts.w1_scale_val.dtype == experts.w2_scale_val.dtype == torch.uint8
        and ids.shape == weights.shape == (hidden.shape[0], 8)
        and ids.dtype == torch.int32 and weights.dtype == torch.float32
        and all(t.is_contiguous() for t in (hidden, w1, w2, weights, ids,
                                            experts.w1_scale_val, experts.w2_scale_val))
        and global_num_experts in (-1, 288) and expert_map is None
        and a1q_scale is None and a2_scale is None and not router_input
    )


def patch():
    import torch
    from vllm.platforms import current_platform
    from .gate import vllm_commit
    if (not current_platform.is_rocm() or not vllm_commit().startswith("e97573215")
            or torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0] != "gfx1201"):
        raise RuntimeError("GLM W4A16 experiment requires gfx1201 and vLLM e97573215")
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
    from vllm.model_executor.layers.quantization.quark.quark_moe import QuarkOCP_MX_MoEMethod
    from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import (
        OCP_MXQuantizationEmulationTritonExperts as Experts,
    )
    if getattr(Experts.apply, "_r9k_glm_w4a16", False):
        return True
    if getattr(Experts.apply, "_r9k_glm_w4a4", False):
        raise RuntimeError("Cannot switch activation precision inside a running process")
    from vllm.logger import init_logger
    from ..moe.w4a16 import apply as apply_packed
    log = init_logger("vllm.r9700_vllm")
    original_update = QuarkConfig.maybe_update_config
    original_quant = QuarkOCP_MX_MoEMethod.get_fused_moe_quant_config
    original_init, original_apply = Experts.__init__, Experts.apply

    def update(self, model_name, hf_config=None, revision=None):
        original_update(self, model_name, hf_config, revision)
        if getattr(hf_config, "model_type", None) != "glm5_next":
            return
        self.quant_config, count = weight_only_config(self.quant_config)
        self._r9k_glm_w4a16 = True
        log.info("r9700: experimental GLM W4A16: %d MXFP4 activation configs disabled; weights/FP8/KV unchanged", count)

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        from vllm.config.vllm import get_current_vllm_config_or_none
        cfg = get_current_vllm_config_or_none()
        self._r9k_glm_w4a16 = bool(cfg is not None
            and getattr(cfg.model_config.hf_config, "model_type", None) == "glm5_next"
            and getattr(cfg.quant_config, "_r9k_glm_w4a16", False))

    def quant_config(self, layer):
        if self.model_type == "glm5_next" and self.ocp_mx_scheme == "w_mxfp4":
            from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
            from vllm.model_executor.layers.fused_moe.config import mxfp4_w4a16_moe_quant_config
            if self.mxfp4_backend != Mxfp4MoeBackend.EMULATION:
                raise RuntimeError("GLM W4A16 experiment requires the audited emulation/packed backend")
            # The pinned upstream weight-only branch omits GLM's clamp.
            # Retain all activation settings for both prefill and decode.
            return mxfp4_w4a16_moe_quant_config(
                w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
                w1_bias=layer.w13_bias, w2_bias=layer.w2_bias,
                gemm1_alpha=getattr(layer, "swiglu_alpha", None),
                gemm1_beta=getattr(layer, "swiglu_beta", None),
                gemm1_clamp_limit=getattr(layer, "swiglu_limit", None))
        return original_quant(self, layer)

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
              global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2,
              expert_tokens_meta, apply_router_weight_on_input):
        if expert_tokens_meta is None and eligible(self, hidden_states, w1, w2, topk_weights,
                topk_ids, activation, global_num_experts, expert_map, a1q_scale, a2_scale,
                apply_router_weight_on_input):
            if not hasattr(self, "_r9k_w4a16_scales_valid"):
                ranges = [torch.aminmax(s) for s in (self.w1_scale_val, self.w2_scale_val)]
                self._r9k_w4a16_scales_valid = all(117 <= lo.item() <= hi.item() <= 128 for lo, hi in ranges)
            if self._r9k_w4a16_scales_valid:
                log.info_once("r9700: GLM packed W4A16 active, BF16 activations, clamp=10, max rows=32")
                return apply_packed(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation)
        return original_apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
            global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2,
            expert_tokens_meta, apply_router_weight_on_input)

    apply._r9k_glm_w4a16 = True
    QuarkConfig.maybe_update_config = update
    QuarkOCP_MX_MoEMethod.get_fused_moe_quant_config = quant_config
    Experts.__init__, Experts.apply = init, apply
    log.info("r9700: experimental GLM W4A16 adapter installed")
    return True
