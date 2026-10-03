"""CPU gates for GLM detection and launcher isolation; no model/GPU claims."""
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
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


class Launcher(unittest.TestCase):
    def launch(self, profile, *assignments, **env):
        clean = {k: v for k, v in os.environ.items() if not k.startswith(("R9K_", "VLLM_"))}
        clean.update(DRYRUN="1", REPO=str(ROOT), **env)
        out = subprocess.check_output(["bash", str(ROOT / "serve" / profile), *assignments], env=clean, text=True)
        return shlex.split(out)

    def test_dflash_is_separate_and_explicit(self):
        args = self.launch("glm-5.3-flash-c2.sh", MODEL="/models/local GLM")
        spec = json.loads(args[args.index("--speculative-config") + 1])
        self.assertEqual(spec["method"], "dflash")
        self.assertEqual(spec["num_speculative_tokens"], 4)
        self.assertEqual(spec["attention_backend"], "TRITON_ATTN")
        self.assertTrue(spec["disable_eagle_block_drop"])
        self.assertEqual(spec["draft_tensor_parallel_size"], 8)
        self.assertEqual(spec["revision"], "bf582e4eacc1810f76656d1811693ff6c6737d2a")
        self.assertEqual(spec["kv_cache_dtype"], "fp8_e4m3")
        self.assertEqual(args[args.index("--kv-cache-dtype") + 1], "fp8_e4m3")
        self.assertEqual(args[args.index("--max-model-len") + 1], "-1")
        self.assertEqual(args[args.index("--kv-cache-memory") + 1], "4429185024")
        self.assertIn("r9700/vllm:glm53-plugin-e97573215", args)
        self.assertIn("R9K_GLM_FP8_SPARSE=1", args)
        self.assertIn("R9K_GLM_DFLASH=1", args)
        self.assertIn("R9K_GLM_W4A4_MAX_ROWS=32", args)
        self.assertIn("R9K_GLM_W4A4_BATCH=packed", args)
        self.assertIn("R9K_GLM_DFLASH_SHARD_FC=1", args)
        self.assertNotIn("--no-enable-prefix-caching", args)

    def test_dflash_window_override(self):
        args = self.launch("glm-5.3-flash-c2.sh", "SPEC=1", "CGSIZES=2")
        spec = json.loads(args[args.index("--speculative-config") + 1])
        graphs = json.loads(args[args.index("--compilation-config") + 1])
        self.assertEqual(spec["num_speculative_tokens"], 1)
        self.assertEqual(graphs["cudagraph_capture_sizes"], [2])

    def test_dflash_concurrency_profiles(self):
        for count in (2, 4):
            args = self.launch(f"glm-5.3-flash-c{count}.sh")
            spec = json.loads(args[args.index("--speculative-config") + 1])
            graphs = json.loads(args[args.index("--compilation-config") + 1])
            self.assertEqual(args[args.index("--max-num-seqs") + 1], str(count))
            self.assertEqual(graphs["cudagraph_capture_sizes"], list(range(5, 5*count+1, 5)))
            self.assertEqual(args[args.index("--kv-cache-dtype") + 1], "fp8_e4m3")
            self.assertEqual(spec["kv_cache_dtype"], "fp8_e4m3")
            self.assertEqual(spec["num_speculative_tokens"], 4)
            self.assertEqual(args[args.index("--max-model-len") + 1], "-1")
            self.assertEqual(args[args.index("--kv-cache-memory") + 1], "4429185024")
            self.assertNotIn("--no-enable-prefix-caching", args)
            self.assertIn("r9700/vllm:glm53-plugin-e97573215", args)

    def test_cache_dtype_and_image_isolate_compile_cache(self):
        def cache_mount(args):
            return next(x for x in args if x.endswith(":/root/.cache/vllm"))
        bf16 = self.launch("glm-5.3-flash-c2.sh", "KV_DTYPE=bfloat16")
        fp8 = self.launch("glm-5.3-flash-c2.sh")
        patched = self.launch("glm-5.3-flash-c2.sh", "KV_DTYPE=bfloat16", "IMG=patched-image")
        self.assertNotEqual(cache_mount(bf16), cache_mount(fp8))
        self.assertNotEqual(cache_mount(bf16), cache_mount(patched))

    def test_vision_limits_and_text_fallback(self):
        for count in (2, 4):
            args = self.launch(f"glm-5.3-flash-c{count}.sh")
            self.assertNotIn("--language-model-only", args)
            self.assertEqual(args[args.index("--mm-encoder-attn-backend") + 1], "FLASH_ATTN")
            self.assertIn("FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE", args)
            limits = json.loads(args[args.index("--limit-mm-per-prompt") + 1])
            processor = json.loads(args[args.index("--mm-processor-kwargs") + 1])
            self.assertEqual(limits, {"image": 100, "video": 0})
            self.assertEqual(processor["max_image_tokens"], 2048)
            self.assertEqual(processor["max_pixels"], 2 * 28 * 28 * processor["max_image_tokens"])
            # GLM's square dummy image reports only 2025 tokens. The scheduler
            # budget must also cover nonsquare images with the full 2048.
            self.assertGreaterEqual(int(args[args.index("--max-num-batched-tokens") + 1]),
                                    processor["max_image_tokens"])
        self.assertIn("--language-model-only", self.launch("glm-5.3-flash.sh"))

    def test_optional_fp8_configs_are_individual_readonly_mounts(self):
        plain = self.launch("glm-5.3-flash-c4.sh", "FP8_CONFIG_DIR=")
        tuned = self.launch("glm-5.3-flash-c4.sh",
                            f"FP8_CONFIG_DIR={ROOT / 'tuning/configs/glm'}")
        mounts = [x for x in tuned if "/quantization/utils/configs/" in x]
        self.assertEqual(len(mounts), 2)
        self.assertTrue(all(x.endswith(".json:ro") for x in mounts))
        self.assertFalse(any("/quantization/utils/configs/" in x for x in plain))
        # Tuning must not change any model/scheduler/sampling arguments.
        image = "r9700/vllm:glm53-plugin-e97573215"
        self.assertEqual(plain[plain.index(image):], tuned[tuned.index(image):])

    def test_fp8_config_content_invalidates_compile_cache(self):
        def cache_mount(args):
            return next(x for x in args if x.endswith(":/root/.cache/vllm"))
        with tempfile.TemporaryDirectory(prefix="fp8 configs ") as folder:
            source = next((ROOT / "tuning/configs/glm").glob("N=*.json"))
            config = Path(folder) / source.name
            config.write_text(source.read_text())
            before = self.launch("glm-5.3-flash-c4.sh", f"FP8_CONFIG_DIR={folder}")
            data = json.loads(config.read_text())
            data["1"]["num_warps"] = 4
            config.write_text(json.dumps(data))
            after = self.launch("glm-5.3-flash-c4.sh", f"FP8_CONFIG_DIR={folder}")
            self.assertNotEqual(cache_mount(before), cache_mount(after))

    def test_baseline_and_local_model(self):
        args = self.launch("glm-5.3-flash.sh", MODEL="/models/local GLM", MODELS_DIR="/tmp/model store",
                           MTP="3", DRAFT="/models/draft", R9K_PLATFORM="1")
        self.assertIn("/models/local GLM", args)
        self.assertIn("/tmp/model store:/models", args)
        for key, value in (("--tensor-parallel-size", "8"), ("--max-model-len", "65536"),
                           ("--max-num-seqs", "1"), ("--reasoning-parser", "glm45"),
                           ("--tool-call-parser", "glm47"), ("--served-model-name", "glm-5.3-flash")):
            self.assertEqual(args[args.index(key) + 1], value)
        for x in ("--enforce-eager", "--disable-custom-all-reduce", "R9K_PLATFORM=0", "VLLM_PLUGINS=r9700_glm"):
            self.assertIn(x, args)
        for x in ("--speculative-config", "--cpu-offload-gb", "--chat-template"):
            self.assertNotIn(x, args)

    def test_qwen_defaults_preserved(self):
        args = self.launch("serve.sh")
        for key, value in (("--served-model-name", "Qwen3.8"), ("--tensor-parallel-size", "2"),
                           ("--reasoning-parser", "qwen3"), ("--tool-call-parser", "qwen3_coder")):
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertEqual(json.loads(args[args.index("--speculative-config") + 1]),
                         {"method": "mtp", "num_speculative_tokens": 3})


if __name__ == "__main__":
    unittest.main()
