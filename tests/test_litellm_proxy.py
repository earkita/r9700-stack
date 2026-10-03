"""Optional proxy gates; run inside the pinned LiteLLM image (no GPU needed)."""
import importlib.metadata
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proxy"))
try:
    import litellm_hooks
    from litellm.llms.anthropic.count_tokens.transformation import AnthropicCountTokensConfig
except ModuleNotFoundError as exc:
    if not (exc.name == "litellm" or exc.name.startswith("litellm.")):
        raise
    litellm_hooks = None


@unittest.skipIf(litellm_hooks is None, "Run in the optional pinned LiteLLM image")
class LocalCountTokens(unittest.TestCase):
    def test_pinned_transport_routes_locally(self):
        self.assertEqual(importlib.metadata.version("litellm"), "1.103.0")
        with patch.dict(os.environ, LITELLM_BACKEND_BASE="http://127.0.0.1:8080/"):
            config = AnthropicCountTokensConfig()
            self.assertEqual(config.get_anthropic_count_tokens_endpoint(),
                             "http://127.0.0.1:8080/v1/messages/count_tokens")

    def test_missing_or_invalid_backend_fails_without_remote_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(KeyError):
                litellm_hooks.local_count_endpoint(None)
        for base in ("", "file:///tmp/model", "http:///missing-host"):
            with patch.dict(os.environ, LITELLM_BACKEND_BASE=base):
                with self.assertRaises(ValueError):
                    litellm_hooks.local_count_endpoint(None)


class Launcher(unittest.TestCase):
    def test_dryrun_preserves_paths_and_hides_credentials(self):
        with tempfile.TemporaryDirectory(prefix="proxy test ") as td:
            path = Path(td) / "private.env"
            path.write_text("LITELLM_MASTER_KEY=do-not-print-this-value\n")
            env = {"PATH": os.environ["PATH"], "DRYRUN": "1", "ENV_FILE": str(path)}
            output = subprocess.check_output(["bash", str(ROOT / "serve/litellm.sh")], env=env, text=True)
            args = shlex.split(output)
            self.assertNotIn("do-not-print-this-value", output)
            self.assertEqual(args[args.index("--env-file") + 1], str(path))
            self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
            self.assertIn("LITELLM_BACKEND_BASE=http://127.0.0.1:8080", args)
            self.assertFalse(any(a in args for a in ("--privileged", "--device", "--gpus")))
            self.assertNotIn("LITELLM_BACKEND_KEY=EMPTY", args)  # must honor the file's backend key

    def test_mimo_config_mount(self):
        output = subprocess.check_output(
            ['bash', str(ROOT/'serve/litellm.sh'), 'DRYRUN=1',
             f'CONFIG={ROOT}/proxy/mimo.yaml', 'BACKEND_MODEL=mimo-v2.6-flash-mopd'], text=True)
        args = shlex.split(output)
        self.assertIn(str(ROOT/'proxy/mimo.yaml')+':/opt/r9700-config.yaml:ro', args)
        self.assertIn('LITELLM_OPENAI_MODEL=hosted_vllm/mimo-v2.6-flash-mopd', args)
        self.assertEqual(args[args.index('--config')+1], '/opt/r9700-config.yaml')

    def test_unknown_option_rejected(self):
        r = subprocess.run(["bash", str(ROOT / "serve/litellm.sh"), "UNSUPPORTED=secret"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertNotIn("secret", r.stderr)

    def test_empty_key_cannot_start_unauthenticated_proxy(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "env"
            path.write_text("LITELLM_MASTER_KEY=\nLITELLM_BACKEND_KEY=EMPTY\n")
            r = subprocess.run(["bash", str(ROOT / "serve/litellm.sh"), f"ENV_FILE={path}"],
                               env={"PATH": os.environ["PATH"]}, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("LITELLM_MASTER_KEY", r.stderr)

    @unittest.skipIf(yaml is None, "PyYAML is supplied by the LiteLLM image")
    def test_config_has_only_local_routes_and_no_secret(self):
        config = yaml.safe_load((ROOT / "proxy/litellm.yaml").read_text())
        self.assertEqual(config["general_settings"]["master_key"], "os.environ/LITELLM_MASTER_KEY")
        for model in config["model_list"]:
            params = model["litellm_params"]
            for key in ("api_key", "api_base", "model"):
                self.assertTrue(params[key].startswith("os.environ/"))
        self.assertFalse(config["litellm_settings"]["drop_params"])
        self.assertEqual(config["router_settings"]["num_retries"], 0)


if __name__ == "__main__":
    unittest.main()
