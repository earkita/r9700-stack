"""CPU audit failures must be detected before reserving model memory."""

import json
from pathlib import Path
import struct
import tempfile
import unittest

from r9700_vllm.models.deepseek_audit import (
    GIB, check_scales, component, memory_estimates, read_headers,
)


class DeepseekAudit(unittest.TestCase):
    def read(self, tensors, payload=b"\0" * 4):
        metadata = json.dumps(tensors).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.safetensors"
            path.write_bytes(struct.pack("<Q", len(metadata)) + metadata + payload)
            return read_headers(path)

    def test_valid_header(self):
        self.assertIn("w", self.read({"w": {
            "dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}))

    def test_truncated_tensor(self):
        with self.assertRaisesRegex(ValueError, "Truncated tensor"):
            self.read({"w": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 6]}})

    def test_wrong_element_count(self):
        with self.assertRaisesRegex(ValueError, "Invalid byte count"):
            self.read({"w": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 4]}})

    def test_overlap(self):
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            self.read({name: {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}
                       for name in ("a", "b")})

    def test_draft_not_counted_as_backbone(self):
        self.assertEqual(component("mtp.0.ffn.experts.0.w1.weight"), "draft")
        self.assertEqual(component("layers.0.ffn.experts.0.w1.weight"), "routed_experts")
        self.assertEqual(component("layers.1.engram.embed.weight"), "engram_tables")

    def test_fp8_scales_are_32_by_32(self):
        tensors = {
            "layers.0.attn.wkv.weight": {"dtype": "F8_E4M3", "shape": [512, 5120]},
            "layers.0.attn.wkv.weight_scale": {"dtype": "F8_E8M0", "shape": [16, 160]},
        }
        check_scales(tensors)
        tensors["layers.0.attn.wkv.weight_scale"]["shape"] = [4, 40]
        with self.assertRaisesRegex(ValueError, "Invalid scale layout"):
            check_scales(tensors)

    def test_packed_expert_scale_geometry(self):
        tensors = {
            "layers.0.ffn.experts.0.w2.weight": {"dtype": "U8", "shape": [5120, 1152]},
            "layers.0.ffn.experts.0.w2.weight_scale": {"dtype": "U8", "shape": [5120, 72]},
        }
        check_scales(tensors)
        del tensors["layers.0.ffn.experts.0.w2.weight_scale"]
        with self.assertRaisesRegex(ValueError, "Missing scale"):
            check_scales(tensors)

    def test_offload_is_total_not_per_rank(self):
        groups = {"engram_tables": 189 * GIB, "routed_experts": 269 * GIB,
                  "other": 9 * GIB, "draft": 7 * GIB}
        report = memory_estimates(groups, [189 * GIB], 8, 80)
        self.assertEqual(report["expert_offload_gib_per_rank"], 10)
        self.assertEqual(report["ideal_weight_gib_per_gpu_after_offload"], 24.75)
        self.assertEqual(report["host_payload_gib_with_exact_engram"], 269)
        self.assertEqual(report["engram_gib_if_each_tp_allocation_rounds_to_power_of_two"], 256)
        with self.assertRaisesRegex(ValueError, "Offload exceeds"):
            memory_estimates(groups, [], 8, 300)


if __name__ == "__main__":
    unittest.main()
