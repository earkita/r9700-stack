#!/usr/bin/env bash
# DeepSeek candidate only; GLM/MiMo keep their existing base and image tags.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE='vllm/vllm-openai-rocm:nightly-rocm100-18f8f96025b556071eb627076f94df560fbd3a22@sha256:ccf5ae13df46441de7945890fa93ffd869aadbd2987ebd01dd918cfaf6f34b49'
exec docker build --progress=plain -f "$ROOT/docker/Dockerfile" \
  --build-arg "BASE=$BASE" -t r9700/vllm:deepseek-v41-18f8f960 "$ROOT"
