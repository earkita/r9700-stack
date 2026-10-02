"""CPU checks for pooled workspace bounds and constructor integration."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

from r9700_vllm.compat.glm_indexer import indexer_workspace_pools, install_workspace_sizing


class Workspace(unittest.TestCase):
    def test_c1_c4_and_upstream_chunk_ceiling(self):
        self.assertEqual(indexer_workspace_pools(524288, 1, 4), 131072)
        self.assertEqual(indexer_workspace_pools(524288, 4, 4), 524288)
        self.assertEqual(indexer_workspace_pools(65536, 4, 4), 65536)
        self.assertEqual(indexer_workspace_pools(65536, 64, 4), 655360)

    def test_partial_pools_are_rounded_per_sequence(self):
        # Four independent final partial pools need four entries, not one.
        capacity = indexer_workspace_pools(9, 4, 4)
        self.assertEqual(capacity, 12)
        for lengths in ((9, 9, 9, 9), (1, 4, 8, 9), (9,)):
            required = sum((n + 3) // 4 for n in lengths)
            self.assertGreaterEqual(capacity, required)

    def test_reject_zero_dimensions(self):
        for dimensions in ((0, 4, 4), (65536, 0, 4), (65536, 4, 0)):
            with self.assertRaises(ValueError):
                indexer_workspace_pools(*dimensions)

    def test_constructor_updates_profiling_and_forward_op_once(self):
        calls = []

        class Indexer:
            def __init__(self, vllm_config, *, marker):
                calls.append(marker)
                self.index_kpool = 4
                self.max_total_seq_len = vllm_config.model_config.max_model_len * 40
                self.indexer_op = NS(max_total_seq_len=self.max_total_seq_len)

        log = Mock()
        install_workspace_sizing(Indexer, log)
        wrapped = Indexer.__init__
        install_workspace_sizing(Indexer, log)
        self.assertIs(Indexer.__init__, wrapped)
        cfg = NS(model_config=NS(max_model_len=65536), scheduler_config=NS(max_num_seqs=4))
        model = Indexer(vllm_config=cfg, marker="unchanged")
        self.assertEqual(calls, ["unchanged"])
        self.assertEqual(model.max_total_seq_len, 65536)
        self.assertEqual(model.indexer_op.max_total_seq_len, 65536)
        log.info_once.assert_called_once()


if __name__ == "__main__":
    unittest.main()
