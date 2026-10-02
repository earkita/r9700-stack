"""GLM kpool dispatch and cache-geometry compatibility for the pinned runtime.

e97573215 checks the CDNA-only AITER predicate before calling its generic kpool
implementation. The downstream rocm_fp8[_paged]_mqa_logits dispatchers already
accept is_rdna_aiter_enabled(). Adapt only the GLM module's predicate binding;
never tell other AITER callers that RDNA4 supports CK kernels. The GLM cache
also needs token-sized page declarations so block tables match its views.
No mathematical kernel is copied or replaced.
"""
from __future__ import annotations


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
    from vllm.models.glm5next.common.attention import Glm5NextIndexerCache
    from r9700_vllm.attn.glm_indexer import get_glm_indexer_backend
    if os.environ.get("R9K_GLM_CACHE_GEOMETRY", "aligned") != "stock":
        Glm5NextIndexerCache.get_attn_backend = get_glm_indexer_backend
    sparse_indexer.rocm_aiter_ops = _GlmAiterGate(sparse_indexer.rocm_aiter_ops)
    init_logger("vllm.r9700_vllm").info(
        "r9700: GLM indexer accepts upstream RDNA4 Triton; cache geometry=%s",
        os.environ.get("R9K_GLM_CACHE_GEOMETRY", "aligned"),
    )
    return True
