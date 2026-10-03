"""Explicitly enabled MiMo support for the audited e97573215 ROCm runtime."""

from copy import copy
import functools
import inspect


def layer_cache_config(cache, sliding_window):
    """Preserve global attention without mutating model-wide SWA settings."""
    cache = copy(cache)
    if sliding_window <= -1:
        cache.sliding_window = None
    return cache


def register():
    from vllm.platforms import current_platform
    from .gate import vllm_commit

    if not current_platform.is_rocm() or not vllm_commit().startswith("e97573215"):
        raise RuntimeError("MiMo adapter requires ROCm vLLM e97573215")
    # Register a unique quantization name, not an override of fp8.
    from ..quant import mimo  # noqa: F401
    from vllm.model_executor.models import mimo_v2

    original = mimo_v2.MiMoV2Attention.__init__
    if getattr(original, "_r9k_mimo", False):
        return
    signature = inspect.signature(original)
    for name in ("cache_config", "sliding_window_size", "head_dim", "v_head_dim"):
        if name not in signature.parameters:
            raise RuntimeError(f"Unsupported MiMo attention interface: missing {name}")

    @functools.wraps(original)
    def init(self, *args, **kwargs):
        import torch
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
        if cfg.model_config.hf_config.model_type != "mimo_v2":
            return original(self, *args, **kwargs)
        arch = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).gcnArchName.split(":")[0]
        if arch != "gfx1201":
            raise RuntimeError("MiMo adapter requires gfx1201")
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        cache = layer_cache_config(
            bound.arguments["cache_config"] or cfg.cache_config,
            bound.arguments["sliding_window_size"],
        )
        bound.arguments["cache_config"] = cache
        return original(*bound.args, **bound.kwargs)

    init._r9k_mimo = True
    mimo_v2.MiMoV2Attention.__init__ = init
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.TRITON_ATTN_DIFFKV,
        "r9700_vllm.attn.mimo_diffkv.TritonAttentionDiffKVBackend",
    )
