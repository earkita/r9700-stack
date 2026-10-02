"""GLM kpool dispatch and cache-geometry compatibility for the pinned runtime.

e97573215 checks the CDNA-only AITER predicate before calling its generic kpool
implementation. The downstream rocm_fp8[_paged]_mqa_logits dispatchers already
accept is_rdna_aiter_enabled(). Adapt only the GLM module's predicate binding;
never tell other AITER callers that RDNA4 supports CK kernels. The GLM cache
also needs token-sized page declarations so block tables match its views.
No mathematical kernel is copied or replaced.
"""
from __future__ import annotations


def indexer_workspace_pools(max_model_len: int, max_num_seqs: int, index_kpool: int) -> int:
    """Bound gathered pool entries by scheduled sequences, not token entries.

    Keep the upstream 40-context chunk ceiling. Round each sequence up before
    summing so unequal/non-divisible sequence lengths cannot underallocate.
    """
    if min(max_model_len, max_num_seqs, index_kpool) < 1:
        raise ValueError("GLM indexer workspace dimensions must be positive")
    return min(40, max_num_seqs) * ((max_model_len + index_kpool - 1) // index_kpool)


def install_workspace_sizing(indexer_cls, log):
    """Change only GLM's workspace request, before its first profiling run."""
    from functools import wraps
    original = indexer_cls.__init__
    if getattr(original, "_r9700_workspace_sizing", False):
        return

    @wraps(original)
    def init(self, vllm_config, *args, **kwargs):
        original(self, vllm_config, *args, **kwargs)
        previous = self.max_total_seq_len
        pools = indexer_workspace_pools(
            vllm_config.model_config.max_model_len,
            vllm_config.scheduler_config.max_num_seqs,
            self.index_kpool,
        )
        # The constructor creates the op but does not allocate its shared
        # workspace. Both profiling and real prefill read the op's value.
        self.max_total_seq_len = pools
        self.indexer_op.max_total_seq_len = pools
        log.info_once(
            "r9700: GLM indexer gather workspace: entries=%d (was %d), "
            "max_num_seqs=%d, index_kpool=%d",
            pools, previous, vllm_config.scheduler_config.max_num_seqs,
            self.index_kpool,
        )

    init._r9700_workspace_sizing = True
    indexer_cls.__init__ = init


class _GlmAiterGate:
    def __init__(self, original):
        self.original = original

    def is_enabled(self):
        return self.original.is_enabled() or self.original.is_rdna_aiter_enabled()

    def __getattr__(self, name):
        return getattr(self.original, name)


def patch() -> bool:
    import inspect
    import os
    import torch
    from vllm.logger import init_logger
    from vllm.platforms import current_platform

    if os.environ.get("R9K_GLM_INDEXER", "rdna4") == "stock" or not current_platform.is_rocm():
        return False
    if torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0] != "gfx1201":
        return False
    from vllm.models.glm5next.amd import sparse_indexer
    if isinstance(sparse_indexer.rocm_aiter_ops, _GlmAiterGate):
        return True
    source = inspect.getsource(sparse_indexer.SparseAttnIndexerKpool.forward_hip)
    from .gate import check, vllm_commit
    if not vllm_commit().startswith("e97573215") or source.count("if not rocm_aiter_ops.is_enabled():") != 1:
        raise RuntimeError("GLM RDNA4 indexer adapter requires audited vLLM e97573215; revalidate the new pin")
    check("glm_rdna_indexer")
    from vllm.models.glm5next.common.attention import Glm5NextIndexerCache, Indexer
    from r9700_vllm.attn.glm_indexer import get_glm_indexer_backend
    workspace = os.environ.get("R9K_GLM_INDEXER_WORKSPACE", "bounded")
    if workspace not in ("bounded", "stock"):
        raise ValueError("R9K_GLM_INDEXER_WORKSPACE must be bounded or stock")
    if workspace == "bounded":
        init_source = inspect.getsource(Indexer.__init__)
        if init_source.count("self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)") != 1:
            raise RuntimeError("GLM indexer workspace sizing changed; revalidate the pinned adapter")
        install_workspace_sizing(Indexer, init_logger("vllm.r9700_vllm"))
    if os.environ.get("R9K_GLM_CACHE_GEOMETRY", "aligned") != "stock":
        Glm5NextIndexerCache.get_attn_backend = get_glm_indexer_backend
    sparse_indexer.rocm_aiter_ops = _GlmAiterGate(sparse_indexer.rocm_aiter_ops)
    init_logger("vllm.r9700_vllm").info(
        "r9700: GLM indexer accepts upstream RDNA4 Triton; cache geometry=%s",
        os.environ.get("R9K_GLM_CACHE_GEOMETRY", "aligned"),
    )
    return True
