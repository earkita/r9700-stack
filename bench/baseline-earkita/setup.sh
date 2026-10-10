#!/bin/bash
# Installs benchmark dependencies only under the checkout; starts no services.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
venv="$root/.runtime/baseline-earkita/venv"
export PIP_CACHE_DIR="$root/.runtime/baseline-earkita/pip-cache"
export TMPDIR="$root/.runtime/baseline-earkita/tmp"
mkdir -p "$TMPDIR"
python3 -m venv "$venv"
"$venv/bin/python" -m pip install 'git+https://github.com/GGZ14/BetterBench.git@d00ad5ec8098c06584a88ec3468bacd37d5ed098'
"$venv/bin/python" -c 'import importlib.metadata as m; assert m.version("betterbench")=="0.6.0"'
echo "Ready: $venv/bin/python"
