"""Opt-in DeepSeek TP8 packed W4A4 prefill/decode, with bounded geometry."""
import os
from functools import wraps


def eligible(experts, x, w1, w2, weights, ids, activation, global_experts,
             expert_map, a1_scale, a2_scale, router_on_input):
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    ep = expert_map is not None
    draft = getattr(experts, "_r9700_ds_dspark", False) and (
        global_experts == 128 or (global_experts == -1 and not ep and w1.shape[0] == 128))
    global_count, topk = (128, 3) if draft else (384, 6)
    e, intermediate = (global_count // 8, 2304) if ep else (global_count, 288)
    return (
        getattr(experts, "_r9700_ds_moe", False)
        and experts.ocp_mx_scheme == "w_mxfp4_a_mxfp4"
        and activation == MoEActivation.SILU
        and experts.activation_config.clamp_limit == 10.0
        and getattr(experts, "_lora_context", None) is None
        and getattr(experts, "w1_bias", None) is None
        and getattr(experts, "w2_bias", None) is None
        and x.ndim == 2 and 1 <= x.shape[0] <= 2048 and x.shape[1] == 5120
        and x.dtype == torch.bfloat16
        and w1.shape == (e,2*intermediate,2560) and w2.shape == (e,5120,intermediate//2)
        and w1.dtype == w2.dtype == torch.uint8
        and experts.w1_scale_val.shape == (e,2*intermediate,160)
        and experts.w2_scale_val.shape == (e,5120,intermediate//32)
        and experts.w1_scale_val.dtype == experts.w2_scale_val.dtype == torch.uint8
        and ids.shape == weights.shape == (x.shape[0],topk)
        and ids.dtype == torch.int32 and weights.dtype == torch.float32
        and all(t.is_contiguous() for t in (x,w1,w2,weights,ids,
                    experts.w1_scale_val,experts.w2_scale_val))
        and global_experts in (-1,global_count)
        and (not ep or (global_experts == global_count and expert_map.shape == (global_count,)
                        and expert_map.dtype == torch.int32 and expert_map.is_contiguous()))
        and a1_scale is None and a2_scale is None and not router_on_input
    )


def install():
    mode = os.environ.get("R9K_DEEPSEEK_MOE", "stock")
    if mode == "stock":
        return
    if mode != "w4a4":
        raise ValueError("R9K_DEEPSEEK_MOE must be stock or w4a4")
    # Called only by the version/ROCm/gfx1201-gated DeepSeek registration.
    from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import (
        OCP_MXQuantizationEmulationTritonExperts as Experts,
    )
    from vllm.config.vllm import get_current_vllm_config_or_none
    from vllm.logger import init_logger
    from r9700_vllm.moe.deepseek_w4a4 import apply_grouped, kernel
    if getattr(Experts.apply, "_r9700_ds_moe", False):
        return
    kernel(grouped=True)
    original, original_init = Experts.apply, Experts.__init__
    log = init_logger("vllm.r9700_vllm")

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        cfg = get_current_vllm_config_or_none()
        self._r9700_ds_moe = bool(cfg and cfg.model_config
            and getattr(cfg.model_config.hf_config,"model_type",None)
                in ("deepseek_v41","deepseek_v41_text")
            and cfg.parallel_config.tensor_parallel_size == 8)
        if self._r9700_ds_moe:
            from .deepseek_dspark import check_speculation
            check_speculation(cfg.speculative_config)
        self._r9700_ds_dspark = bool(self._r9700_ds_moe
            and getattr(cfg.speculative_config, "method", None) == "dspark"
            and os.environ.get("R9K_DEEPSEEK_DSPARK", "0") == "1")
        if (self._r9700_ds_moe and not cfg.model_config.enforce_eager
                and os.environ.get("R9K_DEEPSEEK_GRAPHS", "0") != "1"):
            raise RuntimeError("DeepSeek HIP graphs require explicit R9K_DEEPSEEK_GRAPHS=1 qualification")

    @wraps(original)
    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
              global_num_experts, expert_map, a1q_scale, a2_scale, workspace13,
              workspace2, expert_tokens_meta, apply_router_weight_on_input):
        if expert_tokens_meta is None and eligible(
            self,hidden_states,w1,w2,topk_weights,topk_ids,activation,
            global_num_experts,expert_map,a1q_scale,a2_scale,apply_router_weight_on_input
        ):
            log.info_once("r9700: DeepSeek packed W4A4 prefill/decode active (1-2048 rows), EP=%s", expert_map is not None)
            return apply_grouped(self,output,hidden_states,w1,w2,topk_weights,topk_ids,activation,
                                 expert_map=expert_map, global_experts=global_num_experts)
        if getattr(self, "_r9700_ds_moe", False):
            raise RuntimeError(
                "DeepSeek packed W4A4 call is outside validated TP8 geometry "
                "(1-2048 BF16 rows, H5120/top6/clamp10; TP I288/E384 or EP I2304/E48; opt-in DSpark E128/top3); "
                "refusing full-weight emulation because its memory peak is unsafe")
        return original(self,output,hidden_states,w1,w2,topk_weights,topk_ids,activation,
                        global_num_experts,expert_map,a1q_scale,a2_scale,workspace13,
                        workspace2,expert_tokens_meta,apply_router_weight_on_input)

    apply._r9700_ds_moe = True
    Experts.__init__, Experts.apply = initialize, apply
