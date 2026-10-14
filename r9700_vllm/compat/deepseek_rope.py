"""Correct the pinned portable decode kernel's inverse-RoPE completion flag.

On gfx1201 the portable ragged kernel returns before the fused inverse-RoPE
epilogue. Upstream nevertheless reports every row rotated. Report zero so
DeepSeek V4.1's existing mixed-batch caller applies its standalone rotation
exactly once. No global attention function or GLM/Qwen path is replaced.
"""
from functools import wraps
import inspect


def wrap_decode(original):
    @wraps(original)
    def decode(self, q, positions, kv_cache, swa_metadata, attn_metadata,
               swa_only, output, output_mxfp8=None):
        if output_mxfp8 is not None:
            raise RuntimeError("DeepSeek gfx1201 inverse-RoPE fix requires BF16 output")
        original(self, q, positions, kv_cache, swa_metadata, attn_metadata,
                 swa_only, output, output_mxfp8)
        # The portable gfx1201 kernel writes unrotated rows. forward_mqa
        # already rotates output[rotated:] (including any prefill rows).
        return 0
    decode._r9700_portable_inverse_rope = True
    return decode


def install():
    import torch
    from vllm.platforms import current_platform
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention
    from vllm.logger import init_logger
    from .gate import vllm_commit
    from .deepseek_version import supported_commit

    if (not current_platform.is_rocm() or not supported_commit(vllm_commit())
            or torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0] != "gfx1201"
            or ops._ON_GFX942 or ops._ON_GFX950):
        raise RuntimeError("DeepSeek inverse-RoPE adapter requires pinned vLLM 18f8f960/5555a2d8 / gfx1201")
    cls = DeepseekV41ROCMAiterMLAAttention
    original = cls._forward_decode
    if getattr(original, "_r9700_portable_inverse_rope", False):
        return
    if tuple(inspect.signature(original).parameters) != (
            "self", "q", "positions", "kv_cache", "swa_metadata", "attn_metadata",
            "swa_only", "output", "output_mxfp8"):
        raise RuntimeError("DeepSeek decode interface changed; revalidate inverse-RoPE completion")
    cls._forward_decode = wrap_decode(original)
    init_logger("vllm.r9700_vllm").info(
        "r9700: DeepSeek gfx1201 portable decode uses standalone inverse RoPE")
