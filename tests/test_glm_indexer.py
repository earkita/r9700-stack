"""GLM indexer dispatch gates; numerical kernels remain upstream-owned."""
import unittest
from types import SimpleNamespace

from r9700_vllm.compat.glm_indexer import _GlmAiterGate


class Gate(unittest.TestCase):
    def test_rdna_gate_does_not_enable_global_ck(self):
        original = SimpleNamespace(is_enabled=lambda: False, is_rdna_aiter_enabled=lambda: True,
                                   is_mla_enabled=lambda: False)
        local = _GlmAiterGate(original)
        self.assertTrue(local.is_enabled())
        self.assertFalse(original.is_enabled())
        self.assertFalse(local.is_mla_enabled())

    def test_disabled_and_cdna(self):
        for cdna in (False, True):
            original = SimpleNamespace(is_enabled=lambda: cdna, is_rdna_aiter_enabled=lambda: False)
            self.assertEqual(_GlmAiterGate(original).is_enabled(), cdna)


if __name__ == "__main__":
    unittest.main()
