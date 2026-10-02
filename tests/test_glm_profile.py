"""CPU gates for GLM detection and launcher isolation; no model/GPU claims."""
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import unittest

from r9700_vllm.models.glm5_next import inspect_checkpoint, is_glm5_next

ROOT = Path(__file__).resolve().parents[1]


def checkpoint():
    spec = dict(dtype="fp4", group_size=32, qscheme="per_group", ch_axis=-1,
                scale_format="e8m0", is_dynamic=False)
    return dict(architectures=["Glm5NextForConditionalGeneration"], model_type="glm5_next",
                text_config=dict(model_type="glm5_next_text", hidden_size=4096,
                                 moe_intermediate_size=2048, num_attention_heads=64),
                quantization_config=dict(quant_method="quark", global_quant_config=dict(
                    weight=spec, input_tensors={**spec, "is_dynamic": True}),
                    export=dict(pack_method="reorder", weight_format="real_quantized")))


class Detection(unittest.TestCase):
    def test_structural_detection(self):
        for c in ({"architectures": ["Glm5NextForConditionalGeneration"]},
                  {"architectures": ["Glm5NextForCausalLM"]}, {"model_type": "glm5_next"},
                  {"text_config": {"model_type": "glm5_next_text"}}):
            self.assertTrue(is_glm5_next(c))
        for c in ({}, {"_name_or_path": "amd/GLM-5.3-Flash-Quark-MXFP4"},
                  {"architectures": ["Qwen4ExpForConditionalGeneration"]},
                  {"model_type": "glm_moe_dsa"}):
            self.assertFalse(is_glm5_next(c))

    def test_metadata_is_not_mutated(self):
        c = checkpoint()
        c["quantization_config"]["layer_quant_config"] = {"dense": {"weight": {"dtype": "fp8_e4m3"}}}
        before = copy.deepcopy(c)
        result = inspect_checkpoint(c)
        self.assertEqual(c, before)
        self.assertEqual(result["expert_intermediate_per_rank"], 256)
        self.assertEqual(result["layer_quant_overrides"], 1)
        self.assertIn("MXFP4", result["activation_format"])

    def test_reject_changed_quantization(self):
        for field, value in (("dtype", "fp8_e4m3"), ("group_size", 16),
                             ("scale_format", "float32"), ("is_dynamic", False)):
            c = checkpoint()
            c["quantization_config"]["global_quant_config"]["input_tensors"][field] = value
            with self.assertRaises(ValueError):
                inspect_checkpoint(c)
        c = checkpoint()
        c["quantization_config"]["quant_method"] = "compressed-tensors"
        with self.assertRaises(ValueError):
            inspect_checkpoint(c)

    def test_invalid_tp(self):
        for tp in (0, -1, 3):
            with self.assertRaises(ValueError):
                inspect_checkpoint(checkpoint(), tp)


if __name__ == "__main__":
    unittest.main()
