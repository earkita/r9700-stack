"""Pinned RDNA tiled-cache reader, installed by the opt-in GLM plugin only."""
from functools import wraps


def _wrap(original, replacement):
    @wraps(original)
    def dispatch(q, cache, weights, context_lens, table, schedule_metadata,
                 max_model_len, *, compress_ratio=1):
        import torch
        if (compress_ratio == 1 and q.shape[-2:] == (32, 128)
                and q.dtype == torch.float8_e4m3fn and cache.dtype == torch.uint8
                and cache.ndim == 4 and cache.shape[1] in (32, 64)
                and cache.shape[2:] == (1, 132) and cache.is_contiguous()):
            return replacement(q, cache, weights, context_lens, table,
                               schedule_metadata, max_model_len)
        return original(q, cache, weights, context_lens, table, schedule_metadata,
                        max_model_len, compress_ratio=compress_ratio)
    dispatch._r9700_glm_paged_logits = True
    return dispatch


def patch() -> bool:
    import inspect
    import os
    import torch
    from vllm.platforms import current_platform
    if os.environ.get("R9K_GLM_PAGED_LOGITS", "tiled") == "stock" or not current_platform.is_rocm():
        return False
    if torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0] != "gfx1201":
        return False
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
    if getattr(ops.rocm_fp8_paged_mqa_logits, "_r9700_glm_paged_logits", False):
        return True
    from .gate import check, vllm_commit
    if (not vllm_commit().startswith("e97573215")
            or "deepgemm_fp8_paged_mqa_logits_stage1" not in inspect.getsource(ops.rocm_fp8_paged_mqa_logits)):
        raise RuntimeError("GLM tiled logits adapter requires audited vLLM e97573215; revalidate the new pin")
    check("glm_paged_logits")
    from r9700_vllm.attn.glm_paged_logits import paged_logits
    ops.rocm_fp8_paged_mqa_logits = _wrap(ops.rocm_fp8_paged_mqa_logits, paged_logits)
    from vllm.logger import init_logger
    init_logger("vllm.r9700_vllm").info("r9700: GLM RDNA decode logits read 16x16 tiled FP8 cache")
    return True
