"""Pinned GLM TP8 W4A4 adapter: C1 by default, opt-in packed batches up to 32."""
from __future__ import annotations


def eligible(experts, hidden_states, w1, w2, topk_weights, topk_ids, activation,
             global_num_experts, expert_map, a1q_scale, a2_scale, apply_router_weight_on_input,
             max_rows=1):
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import OCP_MX_Scheme

    return (
        experts.ocp_mx_scheme == OCP_MX_Scheme.w_mxfp4_a_mxfp4
        and activation == MoEActivation.SILU and experts.activation_config.clamp_limit == 10.0
        and getattr(experts, "_lora_context", None) is None
        and getattr(experts, "w1_bias", None) is None and getattr(experts, "w2_bias", None) is None
        and hidden_states.ndim == 2 and hidden_states.shape[1] == 4096
        and 1 <= hidden_states.shape[0] <= max_rows and hidden_states.dtype == torch.bfloat16
        and w1.shape == (288, 512, 2048) and w2.shape == (288, 4096, 128)
        and w1.dtype == w2.dtype == torch.uint8
        and experts.w1_scale_val.shape == (288, 512, 128)
        and experts.w2_scale_val.shape == (288, 4096, 8)
        and experts.w1_scale_val.dtype == experts.w2_scale_val.dtype == torch.uint8
        and topk_ids.shape == topk_weights.shape == (hidden_states.shape[0], 8)
        and topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
        and all(t.is_contiguous() for t in (hidden_states, w1, w2, topk_ids, topk_weights,
                                            experts.w1_scale_val, experts.w2_scale_val))
        and global_num_experts in (-1, 288) and expert_map is None
        and a1q_scale is None and a2_scale is None and not apply_router_weight_on_input
    )


def patch() -> bool:
    import os
    if os.environ.get("R9K_GLM_MOE", "stock") == "stock":
        return False
    if os.environ["R9K_GLM_MOE"] != "w4a4":
        raise ValueError("R9K_GLM_MOE must be stock or w4a4")
    max_rows = int(os.environ.get("R9K_GLM_W4A4_MAX_ROWS", "1"))
    if max_rows not in (1, 8, 32):
        raise ValueError("R9K_GLM_W4A4_MAX_ROWS must be 1, 8 or 32")
    batch = os.environ.get("R9K_GLM_W4A4_BATCH", "isolated")
    if batch not in ("isolated", "grouped", "packed"):
        raise ValueError("R9K_GLM_W4A4_BATCH must be isolated, grouped or packed")
    if max_rows == 32 and batch != "packed":
        raise ValueError("R9K_GLM_W4A4_MAX_ROWS=32 requires R9K_GLM_W4A4_BATCH=packed")
    import torch
    from vllm.platforms import current_platform
    from .gate import check, vllm_commit
    if (not current_platform.is_rocm()
            or torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0] != "gfx1201"
            or not vllm_commit().startswith("e97573215")):
        raise RuntimeError("GLM W4A4 candidate requires gfx1201 and audited vLLM e97573215")
    from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import (
        OCP_MXQuantizationEmulationTritonExperts as Experts,
    )
    if getattr(Experts.apply, "_r9k_glm_w4a4", False):
        return True
    from ..moe.w4a4 import apply_c1, apply_isolated_rows, apply_grouped, kernel
    from ..moe.packed_gemv import apply as apply_packed
    kernel()  # Fail at startup if the requested compiled entry point is absent.
    if batch == "grouped":
        kernel(grouped=True)
    check("glm_w4a4_c1")
    from vllm.logger import init_logger
    log = init_logger("vllm.r9700_vllm")
    original = Experts.apply
    original_init = Experts.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # vLLM sets this context during model construction, not eager forward.
        # Capture identity now; never query the construction context in apply.
        from vllm.config.vllm import get_current_vllm_config_or_none
        cfg = get_current_vllm_config_or_none()
        model = cfg.model_config if cfg is not None else None
        self._r9k_is_glm = model is not None and getattr(model.hf_config, "model_type", None) == "glm5_next"

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
              global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2,
              expert_tokens_meta, apply_router_weight_on_input):
        if expert_tokens_meta is None and eligible(self, hidden_states, w1, w2, topk_weights, topk_ids, activation,
                    global_num_experts, expert_map, a1q_scale, a2_scale, apply_router_weight_on_input, max_rows):
            if not hasattr(self, "_r9k_w4a4_validated"):
                is_glm = self._r9k_is_glm
                # One-time startup validation, outside steady-state decode.
                # Bound to the audited checkpoint range, excluding reserved
                # codes and BF16 overflow/subnormal differences in weight math.
                ranges = [torch.aminmax(s) for s in (self.w1_scale_val, self.w2_scale_val)] if is_glm else []
                self._r9k_w4a4_validated = is_glm and all(
                    117 <= lo.item() <= hi.item() <= 128 for lo, hi in ranges)
            if self._r9k_w4a4_validated:
                log.info_once("r9700: GLM W4A4 candidate active; scale gate passed; max rows=%d, batch=%s", max_rows, batch)
                impl = (apply_c1 if hidden_states.shape[0] == 1 else
                        apply_grouped if batch == "grouped" else
                        apply_packed if batch == "packed" else apply_isolated_rows)
                return impl(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation)
        return original(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
                        global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2,
                        expert_tokens_meta, apply_router_weight_on_input)

    apply._r9k_glm_w4a4 = True
    Experts.__init__ = init
    Experts.apply = apply
    log.info(
        "r9700: experimental GLM TP8 W4A4, max rows=%d, batch=%s; upstream fallback for other calls", max_rows, batch)
    return True
