"""GLM adapters still needed on the audited vLLM ROCm 10 77154ca6 image.

This runtime already implements GLM auxiliary DFlash states, kpool tail rings,
Mamba APC block units, the GLM indexer backend and generic mixed-page KV
packing. Do not stack our e97573215 versions on those upstream fixes.
"""
from __future__ import annotations


def patch() -> None:
    import inspect
    import os
    import torch

    from vllm.logger import init_logger
    from vllm.models.glm5next.common import model as glm
    from vllm.models.glm5next.common.attention import Glm5NextIndexerCache, Glm5NextTailCache
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.mla.indexer import Glm5NextIndexerBackend
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    from .gate import vllm_commit

    if (not vllm_commit().startswith("77154ca6") or not current_platform.is_rocm()
            or torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0] != "gfx1201"):
        raise RuntimeError("GLM latest adapter requires vLLM 77154ca6, ROCm and gfx1201")
    # Fail if any upstream fix we depend on disappears or changes its contract.
    if (not issubclass(glm.Glm5NextModel, glm.EagleModelMixin)
            or Glm5NextIndexerCache.get_attn_backend(Glm5NextIndexerCache) is not Glm5NextIndexerBackend
            or "mamba_block_size" not in inspect.getsource(MambaHybridModelState.add_request)
            or "num_speculative_tokens" not in inspect.getsource(Glm5NextTailCache.get_kv_cache_spec)):
        raise RuntimeError("vLLM GLM/DFlash cache interfaces changed; revalidate the 77154ca6 adapter")

    from .glm_indexer import patch as patch_indexer
    patch_indexer()
    from .glm_paged_logits import patch as patch_logits
    patch_logits()
    from .glm_head_gemv import patch as patch_head
    patch_head()
    if os.environ.get("R9K_GLM_DFLASH") == "1":
        from .glm_dflash import install_draft_projection
        if os.environ.get("R9K_GLM_DFLASH_SHARD_FC") == "1":
            install_draft_projection()
    from .glm_moe import patch as patch_moe
    patch_moe()
    from .glm_fp8_dense import patch as patch_dense
    patch_dense()
    from .glm_sparse_fp8 import patch as patch_sparse
    patch_sparse()
    init_logger("vllm.r9700_vllm").info(
        "r9700: GLM 77154ca6 uses upstream DFlash states, APC, kpool tail and KV packing")
