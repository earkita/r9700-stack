"""Reject damaged checkpoint headers before loading any GPU weights."""

import json
from pathlib import Path
import struct
import tempfile
import unittest

from r9700_vllm.models.mimo import headers


class CheckpointHeaders(unittest.TestCase):
    def read_tensor(self, shape, offsets, payload):
        metadata = json.dumps(
            {
                "weight": {
                    "dtype": "BF16",
                    "shape": shape,
                    "data_offsets": offsets,
                }
            }
        ).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.safetensors"
            path.write_bytes(struct.pack("<Q", len(metadata)) + metadata + payload)
            return headers(path)

    def test_valid_payload(self):
        self.assertEqual(
            self.read_tensor([2], [0, 4], b"\0" * 4)["weight"]["shape"], [2]
        )

    def test_truncated_payload(self):
        with self.assertRaisesRegex(ValueError, "Truncated tensor"):
            self.read_tensor([2], [0, 4], b"\0" * 2)

    def test_shape_does_not_match_bytes(self):
        with self.assertRaisesRegex(ValueError, "Invalid tensor byte count"):
            self.read_tensor([3], [0, 4], b"\0" * 4)


if __name__ == "__main__":
    unittest.main()
