# RDNA4 sparse MLA FP8 port

These patches retain the historical source-patched reference and upstream
submission material. The current GLM profile uses a process-local plugin on
unmodified vLLM; see [current GLM profiles](../../notes/glm-overview.md).

Base: vLLM `e9757321527ca1ecd514c07c1418dd2c53da3d19`.

The source patch is independent of the r9700 plugin. It supports standard
E4M3 cache storage with a BF16 query for rope-free sparse MLA on RDNA4. It
does not enable E5M2, DeepSeek's packed FP8 layout or draft-cache quantization.
The implementation ports the relevant behavior of my-llm's v0.29 patch
`0018-fix-route-RDNA4-FP8-sparse-MLA-through-Triton.patch` to the pinned source,
including its newer sink handling and gfx950 Opus dispatch.

The second patch tunes short FP8 batches on RDNA4 to eight warps, reducing
register spilling for H=16/D=512 without changing BLOCK_K or the arithmetic.

For an upstream source checkout, apply both patches in manifest order with `git apply` and
copy `tests/test_rocm_sparse_mla_fp8.py` from this repository to an appropriate
upstream attention-test directory. That test has no r9700 imports, model
weights or launcher dependencies. Keep the two together in an upstream PR;
the local profiles and baseline manifest belong only in r9700-stack.

For this stack, `docker/Dockerfile.glm-fp8` runs `apply.py`. The installer checks
every affected source file against `manifest.json` before and after applying
the patches; no fuzzy matching or live mutation of the baseline container is
used. A future vLLM revision needs an explicit reviewed port and fresh hashes.

GPU validation (with no concurrent model workload):

```bash
docker run --rm --entrypoint python \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e VLLM_PLUGINS= -e VLLM_ROCM_USE_AITER=1 \
  -v "$PWD:/opt/r9700" -w /opt/r9700 \
  r9700/vllm:glm53-fp8-w8-e97573215 \
  -m pytest -q tests/test_rocm_sparse_mla_fp8.py
```

Compare against a dequantized reference with the same FP8 bytes. This isolates
reader correctness from FP8 quantization error. Separately run the unchanged
BF16 attention gates and the [model-level testing protocol](../../notes/glm-testing.md)
before selecting a runtime default.
