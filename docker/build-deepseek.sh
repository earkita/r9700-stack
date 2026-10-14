#!/usr/bin/env bash
# DeepSeek candidate only; GLM/MiMo keep their existing base and image tags.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE='vllm/vllm-openai-rocm:nightly-rocm100-5555a2d8f7116b2b4877f66516f4128176cf722a@sha256:e7d4d8643501172acc5e9a6fcc3739680359613c2904ec977cde9e721e798e82'
docker build --progress=plain -f "$ROOT/docker/Dockerfile" \
  --build-arg "BASE=$BASE" -t r9700/vllm:deepseek-v41-5555a2d8 "$ROOT"
exec docker build --progress=plain -f "$ROOT/docker/Dockerfile.deepseek-parser" \
  --build-arg BASE=r9700/vllm:deepseek-v41-5555a2d8 \
  --build-arg VLLM_COMMIT=5555a2d8 \
  -t r9700/vllm:deepseek-v41-5555a2d8-parser "$ROOT"
