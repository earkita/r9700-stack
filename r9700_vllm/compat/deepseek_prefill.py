"""Opt-in paged sparse prefill for DeepSeek V4.1 on pinned gfx1201 vLLM.

Reuse upstream prefill slot mapping and its portable two-cache Triton reader.
The AITER equivalent uses gfx950-specific Gluon instructions: do not spoof its
architecture gate. Decode dispatch and the caller's inverse RoPE stay intact.
No full-context BF16 gather is allocated, including during memory profiling.
"""
from functools import wraps
import inspect


def enabled_for_config(cfg):
    import os
    enabled = os.environ.get("R9K_DEEPSEEK_PAGED_PREFILL", "0")
    if enabled not in ("0", "1"):
        raise ValueError("R9K_DEEPSEEK_PAGED_PREFILL must be 0 or 1")
    if enabled == "0" or getattr(cfg.model_config.hf_config, "model_type", None) not in (
            "deepseek_v41", "deepseek_v41_text"):
        return False
    if not cfg.model_config.enforce_eager or cfg.kv_transfer_config is not None:
        raise RuntimeError("DeepSeek paged prefill requires eager without KV transfer")
    return True


def wrap_init(original, config_getter):
    @wraps(original)
    def initialize(self, *args, **kwargs):
        import os
        flag = os.environ.get("R9K_DEEPSEEK_PAGED_PREFILL", "0")
        if flag not in ("0", "1"):
            raise ValueError("R9K_DEEPSEEK_PAGED_PREFILL must be 0 or 1")
        original(self, *args, **kwargs)
        enabled = False
        if flag == "1":
            # Replay metadata builders can be created during execute_model,
            # outside the global config context. Their constructor retains the
            # explicit config. Disabled adapters must not require any context.
            cfg = getattr(self, "vllm_config", None)
            enabled = enabled_for_config(cfg if cfg is not None else config_getter())
        self._r9700_paged_prefill = enabled
    return initialize


def wrap_workspace(original):
    @wraps(original)
    def shapes(self, M, num_prefill_tokens, q, gather=True):
        if getattr(self, "_r9700_paged_prefill", False):
            gather = False
        return original(self, M, num_prefill_tokens, q, gather)
    return shapes


def wrap_build(original, build_ragged):
    @wraps(original)
    def build(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if getattr(self, "_r9700_paged_prefill", False) and result.num_prefill_tokens:
            if result.prefill_swa_indices is None or result.prefill_swa_lens is None:
                raise RuntimeError("DeepSeek paged prefill requires SWA slot metadata")
            (result.prefill_swa_ragged_indices,
             result.prefill_swa_ragged_indptr) = build_ragged(
                result.prefill_swa_indices.reshape(result.num_prefill_tokens, -1),
                result.prefill_swa_lens)
        return result
    return build


def wrap_prefill(original, reader):
    @wraps(original)
    def prefill(self, q, positions, compressed_k_cache, swa_k_cache, output,
                attn_metadata, swa_metadata, mxfp8_out=None):
        if not getattr(self, "_r9700_paged_prefill", False):
            return original(self, q, positions, compressed_k_cache, swa_k_cache,
                            output, attn_metadata, swa_metadata, mxfp8_out)
        if (mxfp8_out is not None or output is None
                or self.head_dim != 512 or self.nope_head_dim != 448
                or self.rope_head_dim != 64 or self.compress_ratio not in (0, 1, 2)
                or self.kv_cache_dtype != "fp8_ds_mla"):
            raise RuntimeError("Unsupported DeepSeek paged prefill geometry/output")
        n = swa_metadata.num_prefill_tokens
        main_indices = swa_metadata.prefill_swa_ragged_indices
        main_indptr = swa_metadata.prefill_swa_ragged_indptr
        if main_indices is None or main_indptr is None:
            raise RuntimeError("Missing DeepSeek paged prefill SWA indices")
        extra_indices = extra_indptr = None
        if attn_metadata is None:
            compressed_k_cache = None
        else:
            if compressed_k_cache is None:
                raise RuntimeError("Missing DeepSeek compressed cache")
            extra_indices, extra_indptr = self._prefill_topk_ragged(
                swa_metadata, attn_metadata, compressed_k_cache)
        # Despite its upstream name this reader accepts any number of queries.
        # Causality is encoded by upstream's SWA and compressed slot streams.
        # Return unrotated BF16 rows, just like the original prefill path.
        reader(q=q[:n], main_cache=swa_k_cache, main_indices=main_indices,
               main_indptr=main_indptr, scale=self.scale,
               attn_sink=self.attn_sink[:q.shape[1]], nope_head_dim=448,
               rope_head_dim=64, extra_cache=compressed_k_cache,
               extra_indices=extra_indices, extra_indptr=extra_indptr,
               out=output[:n])
    prefill._r9700_paged_prefill = True
    return prefill


def install():
    import torch
    from vllm.config import get_current_vllm_config
    from vllm.platforms import current_platform
    from vllm.models.deepseek_v41.amd import rocm
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
    from .gate import vllm_commit
    from .deepseek_version import supported_commit

    if (not current_platform.is_rocm() or not supported_commit(vllm_commit())
            or torch.cuda.get_device_properties(torch.cuda.current_device())
            .gcnArchName.split(":")[0] != "gfx1201" or ops._ON_GFX950 or ops._ON_GFX942):
        raise RuntimeError("DeepSeek paged prefill requires ROCm gfx1201 / pinned vLLM 18f8f960/5555a2d8")
    cls = rocm.DeepseekV41ROCMAiterMLAAttention
    builder = rocm.DeepseekV4ROCMAiterSparseSWAMetadataBuilder
    if getattr(cls._forward_prefill, "_r9700_paged_prefill", False):
        return
    contracts = [
        (cls._forward_prefill, ("self", "q", "positions", "compressed_k_cache",
          "swa_k_cache", "output", "attn_metadata", "swa_metadata", "mxfp8_out")),
        (cls._prefill_workspace_shapes, ("self", "M", "num_prefill_tokens", "q", "gather")),
        (cls._prefill_topk_ragged, ("self", "swa_metadata", "attn_metadata", "compressed_k_cache")),
        (ops._rocm_sparse_attn_decode_ragged_triton, ("q", "main_cache", "main_indices",
          "main_indptr", "scale", "attn_sink", "nope_head_dim", "rope_head_dim",
          "extra_cache", "extra_indices", "extra_indptr", "out", "extra_cache_nan_free",
          "adaptive_splits", "inv_rope_positions", "inv_rope_cos_sin_cache", "out_mxfp8")),
    ]
    for fn, parameters in contracts:
        if tuple(inspect.signature(fn).parameters) != parameters:
            raise RuntimeError(f"DeepSeek paged prefill interface changed: {fn.__name__}")
    cls.__init__ = wrap_init(cls.__init__, get_current_vllm_config)
    builder.__init__ = wrap_init(builder.__init__, get_current_vllm_config)
    builder.build = wrap_build(builder.build, ops.build_ragged_indices_from_dense)
    cls._prefill_workspace_shapes = wrap_workspace(cls._prefill_workspace_shapes)
    cls._forward_prefill = wrap_prefill(cls._forward_prefill,
                                      ops._rocm_sparse_attn_decode_ragged_triton)
    from vllm.logger import init_logger
    init_logger("vllm.r9700_vllm").info(
        "r9700: DeepSeek paged prefill adapter installed (opt-in); portable "
        "Triton reads SWA/compressed cache directly; decode unchanged")
