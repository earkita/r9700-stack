"""Credential lookup works from other projects without shell evaluation/leaks."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
KEY = "sk-test-placeholder-0123456789"


class ClaudeKeyHelper(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="claude key ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "serve").mkdir()
        (self.root / "secrets").mkdir()
        (self.root / "project").mkdir()
        shutil.copyfile(ROOT / "serve/claude-litellm-key.py", self.root / "serve/claude-litellm-key.py")
        self.command = json.loads((ROOT / "serve/templates/mimo-v2.6-flash.settings.local.json").read_text())["apiKeyHelper"]

    def run_helper(self, **extra):
        return subprocess.run(
            ["/bin/sh", "-c", self.command],
            cwd=self.root / "project",
            env={"PATH": os.environ["PATH"], "R9700_STACK_ROOT": str(self.root), **extra},
            capture_output=True, text=True,
        )

    def test_file_lookup_from_other_project_never_sources_file(self):
        marker = self.root / "executed"
        (self.root / "secrets/litellm.env").write_text(
            f"UNRELATED=$(touch '{marker}')\nLITELLM_MASTER_KEY={KEY}\n"
        )
        result = self.run_helper()
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, KEY + "\n", ""))
        self.assertFalse(marker.exists())

    def test_environment_key_does_not_need_file(self):
        result = self.run_helper(LITELLM_MASTER_KEY=KEY)
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, KEY + "\n", ""))

    def test_missing_or_invalid_key_fails_without_disclosure(self):
        missing = self.run_helper()
        self.assertNotEqual(missing.returncode, 0)
        self.assertEqual(missing.stdout, "")
        invalid = self.run_helper(LITELLM_MASTER_KEY="private-invalid-value")
        self.assertNotEqual(invalid.returncode, 0)
        self.assertEqual(invalid.stdout, "")
        self.assertNotIn("private-invalid-value", invalid.stderr)

    def test_custom_credentials_file(self):
        path = self.root / "custom.env"
        path.write_text(f"LITELLM_MASTER_KEY={KEY}\n")
        result = self.run_helper(R9700_LITELLM_ENV_FILE=str(path))
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, KEY + "\n", ""))
