"""Opt-in Quark dense decode adapter, pinned GLM TP8/gfx1201 only.

Keep original weights/scales for stock prefill and unsupported inputs.
The candidate packs the original FP8 bytes once; it never requantizes weights.
"""
from functools import wraps

SHAPES = {(3072, 4096), (4096, 1536)}


def target_model(cfg):
    return bool(cfg and getattr(cfg.model_config.hf_config, "model_type", None)
                in ("glm5_next", "glm5_next_text")
                and cfg.parallel_config.tensor_parallel_size == 8)


def target_prefix(prefix):
    # Transformers backend traverses its model with prefix="model".
    # Native wrappers may instead place model under language_model.
    import re
    return re.search(r"(?:^|\.)(?:model\.language_model|language_model\.model)\.layers\.\d+\.", prefix) is not None


def prepare(scheme, layer, pack):
    import torch
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym, kFp8Static128BlockSym,
    )
    w, s = layer.weight, layer.weight_scale
    if not (getattr(scheme, "_r9k_glm_dense", False)
            and scheme.activation_quant_key == kFp8Dynamic128Sym
            and scheme.weight_quant_key == kFp8Static128BlockSym
            and scheme.weight_scale_dtype == torch.float32
            and scheme.input_dtype == scheme.out_dtype == torch.bfloat16
            and tuple(w.shape) in SHAPES and w.dtype == torch.float8_e4m3fn
            and w.is_cuda and w.stride(1) == 1 and w.stride(0) >= w.shape[1]
            and s.is_contiguous()
            and s.dtype == torch.float32
            and tuple(s.shape) == (w.shape[0]//128, w.shape[1]//128)
            and type(scheme.fp8_linear).__name__ == "TritonFp8BlockScaledMMKernel"
            and not scheme.fp8_linear.quant_fp8.use_aiter):
        return False
    layer.register_buffer("_r9k_glm_dense_packed", pack(w), persistent=False)
    return True


def wrap_apply(original, gemm):
    @wraps(original)
    def apply(self, layer, x, bias=None):
        import torch
        packed = getattr(layer, "_r9k_glm_dense_packed", None)
        if (packed is None or not isinstance(x, torch.Tensor) or x.ndim != 2
                or not 0 < x.shape[0] <= 32 or x.dtype != torch.bfloat16
                or x.device != packed.device or x.shape[1] != layer.weight.shape[1]
                or x.stride(1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16):
            return original(self, layer, x, bias)
        y = gemm(x, packed, layer.weight_scale, *layer.weight.shape)
        return y if bias is None else y + bias
    return apply


def patch():
    import os
    mode = os.environ.get("R9K_GLM_DENSE_FP8", "0")
    if mode == "0":
        return False
    if mode != "1":
        raise ValueError("R9K_GLM_DENSE_FP8 must be 0 or 1")
    import inspect
    import torch
    from vllm.platforms import current_platform
    from .gate import vllm_commit
    if (not current_platform.is_rocm() or not vllm_commit().startswith("e97573215")
            or torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0] != "gfx1201"):
        raise RuntimeError("GLM dense FP8 requires gfx1201 and vLLM e97573215")
    from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_fp8 import QuarkW8A8Fp8PerBlock as Scheme
    if getattr(Scheme, "_r9k_glm_dense_installed", False):
        return True
    if list(inspect.signature(Scheme.apply_weights).parameters) != ["self", "layer", "x", "bias"]:
        raise RuntimeError("Quark apply interface changed; revalidate GLM dense FP8")
    from ..kernels.glm_fp8 import gemm, bind
    from ..kernels.fp8 import permute_fp8
    bind()  # Reject a stale library before weight loading.
    init, process, apply = Scheme.__init__, Scheme.process_weights_after_loading, Scheme.apply_weights

    @wraps(init)
    def scoped_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        from vllm.config.vllm import get_current_vllm_config_or_none
        cfg = get_current_vllm_config_or_none()
        self._r9k_glm_dense = target_model(cfg)

    @wraps(process)
    def after_load(self, layer):
        process(self, layer)
        # Exclude vision and drafters even if constructed under target config.
        if not target_prefix(getattr(layer, "prefix", "")):
            return
        from vllm.logger import init_logger
        if prepare(self, layer, permute_fp8):
            init_logger("vllm.r9700_vllm").info_once(
                "r9700: GLM dense FP8 candidate enabled for %s, <=32 rows; stock prefill retained",
                tuple(layer.weight.shape))
        elif tuple(layer.weight.shape) in SHAPES:
            init_logger("vllm.r9700_vllm").warning_once(
                "r9700: GLM dense FP8 candidate skipped for %s: model=%s, "
                "backend=%s, input=%s, output=%s, weight=%s, scale=%s, "
                "weight_stride=%s, scale_shape=%s, aiter=%s",
                tuple(layer.weight.shape), self._r9k_glm_dense,
                type(self.fp8_linear).__name__, self.input_dtype, self.out_dtype,
                layer.weight.dtype, layer.weight_scale.dtype, layer.weight.stride(),
                tuple(layer.weight_scale.shape), self.fp8_linear.quant_fp8.use_aiter)
    Scheme.__init__ = scoped_init
    Scheme.process_weights_after_loading = after_load
    Scheme.apply_weights = wrap_apply(apply, gemm)
    Scheme._r9k_glm_dense_installed = True
    return True
