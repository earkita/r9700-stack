"""CPU gates for MiMo profile isolation and checkpoint preflight."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import unittest
from unittest.mock import patch
from r9700_vllm import register_mimo

ROOT = Path(__file__).resolve().parents[1]


class MiMoProfile(unittest.TestCase):
    def command(self, *args):
        env = dict(os.environ, DRYRUN="1")
        return shlex.split(
            subprocess.check_output(
                ["bash", str(ROOT / "serve/mimo-v2.6-flash.sh"), *args],
                env=env,
                text=True,
            )
        )

    def test_control_and_draft(self):
        for k in (0, 7):
            cmd = self.command(f"SPEC={k}")
            for flag, value in [
                ("--tensor-parallel-size", "8"),
                ("--max-num-seqs", "2"),
                ("--max-model-len", "131072"),
                ("--kv-cache-dtype", "bfloat16"),
                ("--quantization", "r9700_mimo_mxfp4"),
            ]:
                self.assertEqual(cmd[cmd.index(flag) + 1], value)
            self.assertIn("--disable-custom-all-reduce", cmd)
            self.assertIn("--no-async-scheduling", cmd)
            self.assertIn("--language-model-only", cmd)
            if k:
                spec = json.loads(cmd[cmd.index("--speculative-config") + 1])
                self.assertEqual(spec["num_speculative_tokens"], 7)
                self.assertEqual(spec["draft_tensor_parallel_size"], 8)
                self.assertEqual(spec["kv_cache_dtype"], "bfloat16")
            else:
                self.assertNotIn("--speculative-config", cmd)

    def test_opt_in_only(self):
        with patch.dict(os.environ, {}, clear=True):
            register_mimo()  # does not import vLLM, access GPUs or patch other models

    def test_exact_ar8_switch(self):
        for enabled in (0, 1):
            cmd = self.command(f"AR8={enabled}")
            self.assertIn(f"R9K_ARN={enabled}", cmd)
            self.assertIn("R9K_ARN_MAX_KB=256", cmd)
            self.assertIn("R9K_ARN_1S_KB=16", cmd)
            self.assertIn("R9K_AR4=0", cmd)
            self.assertIn("--disable-custom-all-reduce", cmd)

    def test_hybrid_cache_does_not_mutate_global_config(self):
        from types import SimpleNamespace
        from r9700_vllm.compat.mimo import layer_cache_config

        original = SimpleNamespace(sliding_window=128)
        self.assertIsNone(layer_cache_config(original, -1).sliding_window)
        self.assertEqual(layer_cache_config(original, 128).sliding_window, 128)
        self.assertEqual(original.sliding_window, 128)

    def test_template(self):
        obj = json.loads(
            (ROOT / "serve/templates/mimo-v2.6-flash.settings.local.json").read_text()
        )
        self.assertEqual(obj["apiKeyHelper"], "printenv LITELLM_MASTER_KEY")
        self.assertEqual(obj["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "131072")
        self.assertEqual(
            obj["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"], "mimo-v2.6-flash-fast"
        )
        self.assertNotIn("/home/", json.dumps(obj))


if __name__ == "__main__":
    unittest.main()
