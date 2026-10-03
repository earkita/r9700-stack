#!/usr/bin/env python3
"""Print only the local proxy key for Claude Code's apiKeyHelper."""

import os
from pathlib import Path
import sys


def main():
    key = os.environ.get("LITELLM_MASTER_KEY", "").strip()
    if not key:
        default = Path(__file__).resolve().parents[1] / "secrets/litellm.env"
        path = Path(os.environ.get("R9700_LITELLM_ENV_FILE") or default)
        try:
            # Read Docker-style KEY=value data; never execute/source the file.
            for line in path.read_text().splitlines():
                name, separator, value = line.partition("=")
                if separator and name.strip() == "LITELLM_MASTER_KEY":
                    key = value.strip()
        except (OSError, UnicodeError):
            print(f"Cannot read LiteLLM credentials from {path}. Set LITELLM_MASTER_KEY or R9700_LITELLM_ENV_FILE.", file=sys.stderr)
            return 1
    if not key.startswith("sk-") or len(key) < 20 or any(c.isspace() for c in key):
        print("Missing or invalid LITELLM_MASTER_KEY. Use the key configured for your local LiteLLM proxy.", file=sys.stderr)
        return 1
    print(key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
