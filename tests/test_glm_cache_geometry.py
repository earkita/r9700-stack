"""Run in the pinned runtime; CPU-only regression for the long-prefill fault."""
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.worker.utils import select_common_block_size
from r9700_vllm.attn.glm_indexer import GlmKpool4IndexerBackend, get_glm_indexer_backend


@pytest.mark.parametrize("manager,expected", [(128, 128), (256, 256), (640, 128), (768, 256), (1152, 128), (1280, 256)])
def test_indexer_metadata_and_storage_choose_same_page(manager, expected):
    selected = select_common_block_size(manager, [GlmKpool4IndexerBackend])
    # Independent copy of the upstream cache-spec page rule, in token units.
    pool_storage = manager // 4
    storage = 4 * (64 if pool_storage % 64 == 0 else 32)
    assert selected == storage == expected
    # Expanded block IDs and compressed physical slots agree at every boundary.
    factor = manager // selected
    manager_ids = [7, 2, 9]
    table = [b * factor + p for b in manager_ids for p in range(factor)]
    for pos in range(3, len(manager_ids) * manager, 4):
        pool_pos = pos // 4
        page = selected // 4
        actual = table[pool_pos // page] * page + pool_pos % page
        reference = manager_ids[pos // manager] * (manager // 4) + (pos % manager) // 4
        assert actual == reference


def test_original_rocm_declaration_reproduces_block_table_overrun():
    class OriginalRocmIndexer:
        @staticmethod
        def get_supported_kernel_block_sizes():
            return [1, MultipleOf(16)]

    selected = select_common_block_size(640, [OriginalRocmIndexer])
    assert selected == 640
    # A max-context table has ceil(65536/640)=103 manager entries, but the
    # storage-sized metadata advances one entry each 128 tokens. At token
    # 16383 it tries entry 127 even though only 26 manager blocks are needed.
    manager_table_width = (65536 + selected - 1) // selected
    metadata_index = (16383 // 4) // 32
    assert metadata_index >= manager_table_width


def test_geometry_is_scoped_to_kpool4():
    assert get_glm_indexer_backend(SimpleNamespace(_index_kpool=4)) is GlmKpool4IndexerBackend
    with pytest.raises(NotImplementedError):
        get_glm_indexer_backend(SimpleNamespace(_index_kpool=16))
