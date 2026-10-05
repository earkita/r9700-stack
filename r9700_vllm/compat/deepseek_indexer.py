"""Model-scoped sparse indexer on ROCm gfx1201 / vLLM 18f8f960.

Reuse upstream insertion, gathering, candidate selection and top-k. Bind the
paged reader and fused prefill logits in a private namespace: no global AITER flags,
architecture spoofing or changes to the GLM/Qwen paths.
"""
from functools import wraps
import inspect
import os
from types import FunctionType


def profile_workspace(layer, hidden_states):
    """Reserve the actual reader layout, without a per-head logits tensor.

    ModelRunner V2 may profile without a current VllmConfig context. Capture
    scheduler bounds at construction rather than falling back to prefill rows.
    Prefill gathering and transient logits retain upstream accounting.
    """
    import torch
    from vllm import envs
    from vllm.v1.worker.workspace import current_workspace_manager
    workspace = current_workspace_manager()
    workspace.get_simultaneous(
        ((layer.max_total_seq_len, layer.head_dim), torch.float8_e4m3fn),
        ((layer.max_total_seq_len, 4), torch.uint8),
    )
    rows = min(hidden_states.shape[0], layer._r9700_ds_decode_rows)
    workspace.get_simultaneous(((rows, layer.max_model_len), torch.float32))
    # Same transient prefill-logits reservation as the pinned upstream path.
    torch.empty(envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024,
                dtype=torch.uint8, device=hidden_states.device)
    return layer.topk_indices_buffer


def make_indexer_impl(upstream, reader, prefill_reader):
    # The launcher auto-enables the breakable-graph decorator even in eager
    # mode. Bind the body; outer model capture owns optional HIP graphs.
    upstream = inspect.unwrap(upstream)
    namespace = dict(upstream.__globals__)
    namespace["rocm_fp8_paged_mqa_logits"] = reader
    namespace["rocm_fp8_mqa_logits"] = prefill_reader
    result = FunctionType(upstream.__code__, namespace, upstream.__name__,
                          upstream.__defaults__, upstream.__closure__)
    result.__kwdefaults__ = upstream.__kwdefaults__
    return result


def wrap_init(original):
    @wraps(original)
    def initialize(self, *args, **kwargs):
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        self._r9700_ds_indexer = getattr(
            cfg.model_config.hf_config, "model_type", None
        ) in ("deepseek_v41", "deepseek_v41_text")
        self._r9700_ds_eager = cfg.model_config.enforce_eager
        self._r9700_ds_graphs = os.environ.get("R9K_DEEPSEEK_GRAPHS", "0") == "1"
        if self._r9700_ds_indexer:
            self._r9700_ds_decode_rows = cfg.scheduler_config.max_num_seqs * (
                1 + (getattr(cfg.speculative_config, "num_speculative_tokens", 0) or 0))
        original(self, *args, **kwargs)
        if self._r9700_ds_indexer:
            # No PCP/DCP in this adapter: gathered keys cannot exceed the
            # sum of max_num_seqs compressed contexts. max_model_len is
            # already divided by compress_ratio by the upstream constructor.
            self.max_total_seq_len = min(
                self.max_total_seq_len,
                cfg.scheduler_config.max_num_seqs * self.max_model_len)
    return initialize


def wrap_forward(original, implementation):
    @wraps(original)
    def forward(self, hidden_states, q_quant, k, weights):
        import torch
        from vllm.utils.torch_utils import _encode_layer_name

        if not self._r9700_ds_indexer:
            return original(self, hidden_states, q_quant, k, weights)
        if not self._r9700_ds_eager and not getattr(self, "_r9700_ds_graphs", False):
            raise RuntimeError("DeepSeek gfx1201 indexer requires enforce_eager or explicit R9K_DEEPSEEK_GRAPHS=1 qualification")
        if (self.use_fp4_cache or self.use_pcp or self.dcp_world_size != 1
                or self.compress_ratio not in (1, 2)
                or self.head_dim != 128 or self.quant_block_size != 128
                or self.scale_fmt != "ue8m0"
                or not isinstance(q_quant, torch.Tensor)
                or q_quant.dtype != torch.float8_e4m3fn
                or q_quant.shape[-2:] != (32, 128)):
            raise RuntimeError("Unsupported DeepSeek gfx1201 sparse indexer configuration")
        from vllm.forward_context import get_forward_context
        if not isinstance(get_forward_context().attn_metadata, dict):
            return profile_workspace(self, hidden_states)
        return implementation(
            hidden_states, _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache, q_quant, k, weights,
            self.quant_block_size, self.scale_fmt, self.topk_tokens,
            self.head_dim, self.max_model_len, self.max_total_seq_len,
            self.topk_indices_buffer,
            skip_k_cache_insert=self.skip_k_cache_insert,
            compress_ratio=self.compress_ratio,
            candidate_blocks=self.candidate_blocks,
            candidate_block_size=self.candidate_block_size,
            candidate_write=self.candidate_write,
        )
    forward._r9700_deepseek_indexer = True
    return forward


def install():
    import torch
    from vllm.platforms import current_platform
    from .gate import vllm_commit

    if (not current_platform.is_rocm()
            or not vllm_commit().startswith("18f8f960")
            or torch.cuda.get_device_properties(torch.cuda.current_device())
            .gcnArchName.split(":")[0] != "gfx1201"):
        raise RuntimeError("DeepSeek indexer requires ROCm gfx1201 / vLLM 18f8f960")
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
    from vllm._aiter_ops import rocm_aiter_ops

    if getattr(SparseAttnIndexer.forward_hip, "_r9700_deepseek_indexer", False) is True:
        return
    # Separate from scheduler NBT/NSEQ: this bounds how many requests' full
    # compressed KV rows are gathered at once during prefill. Upstream already
    # loops over this chunk plan; decode and token scheduling are unchanged.
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention
    prefill_seqs = int(os.environ.get("R9K_DEEPSEEK_PREFILL_SEQS", "4"))
    if not 1 <= prefill_seqs <= 4:
        raise ValueError("R9K_DEEPSEEK_PREFILL_SEQS must be between 1 and 4")
    if list(inspect.signature(
            DeepseekV41ROCMAiterMLAAttention._prefill_workspace_shapes).parameters) != [
                "self", "M", "num_prefill_tokens", "q", "gather"]:
        raise RuntimeError("DeepSeek prefill workspace interface changed; revalidate adapter")
    DeepseekV41ROCMAiterMLAAttention.PREFILL_CHUNK_SIZE = prefill_seqs
    if rocm_aiter_ops.is_enabled() or rocm_aiter_ops.is_rdna_aiter_enabled():
        raise RuntimeError("DeepSeek correctness indexer requires VLLM_ROCM_USE_AITER=0")
    upstream = inspect.unwrap(ops.rocm_aiter_sparse_attn_indexer)
    parameters = list(inspect.signature(upstream).parameters)
    if (parameters[-4:] != ["compress_ratio", "candidate_blocks",
                           "candidate_block_size", "candidate_write"]
            or "rocm_fp8_paged_mqa_logits" not in upstream.__code__.co_names
            or "rocm_fp8_mqa_logits" not in upstream.__code__.co_names
            or list(inspect.signature(SparseAttnIndexer.forward_hip).parameters)
            != ["self", "hidden_states", "q_quant", "k", "weights"]):
        raise RuntimeError("DeepSeek sparse indexer interface changed; revalidate adapter")
    from r9700_vllm.attn.deepseek_indexer import paged_logits
    from r9700_vllm.attn.deepseek_indexer_prefill import (
        check_kernel_interface, prefill_logits,
    )
    check_kernel_interface()
    implementation = make_indexer_impl(upstream, paged_logits, prefill_logits)
    SparseAttnIndexer.__init__ = wrap_init(SparseAttnIndexer.__init__)
    SparseAttnIndexer.forward_hip = wrap_forward(SparseAttnIndexer.forward_hip,
                                               implementation)
    from vllm.logger import init_logger
    init_logger("vllm.r9700_vllm").info(
        "r9700: DeepSeek gfx1201 sparse indexer; fused AITER Triton prefill, "
        "16x16 tiled FP8 decode reader, compression=1/2")
