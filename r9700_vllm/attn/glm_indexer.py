"""GLM kpool=4 cache geometry; all metadata builders and kernels stay upstream.

ROCm's generic indexer advertises MultipleOf(16), so the manager's 640-token
block wins selection, while Glm5NextIndexerCache views storage as 128-token
blocks (32 pools). The metadata builder already sees the 128-token spec: using
an unsplit 640-token block table then addresses wrong blocks and eventually
reads beyond the table on long prompts. Advertise the actual token widths.
"""
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend


class GlmKpool4IndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # The cache spec picks the largest pool page (32 or 64) that tiles
        # block_size / kpool. Backend negotiation must use token units.
        return [128, 256]


def get_glm_indexer_backend(cache):
    if cache._index_kpool != 4:
        raise NotImplementedError("GLM RDNA4 cache geometry adapter is audited for index_kpool=4")
    return GlmKpool4IndexerBackend
