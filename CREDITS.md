# Credits

Almost everything here is original work, licensed Apache-2.0 (see [LICENSE](LICENSE)). Incorporated components
are acknowledged below; [NOTICE](NOTICE) is the authoritative statement of what the licence does
and does not cover.

## Third-party components

| What | Source | Used in | License |
|---|---|---|---|
| Device-side expert LRU cache kernels (vendored unmodified) | davetha -- https://github.com/davetha/r9700-lru-expert-cache (3743f13) | `kernels/third_party/davetha/` | **Apache-2.0** -- its own LICENSE and NOTICE are preserved in that directory |
| Chat template `qwen-fixed-v22.3.jinja` (copied verbatim) | GGZ14/vllm-mxfp4 -- https://github.com/GGZ14/vllm-mxfp4 (92eed82) | `serve/templates/` | used with the author's permission, with credit; no upstream LICENSE file |
| AMD GLM kpool tail kernels (adapted for speculative rollback) | vLLM `e9757321527ca1ecd514c07c1418dd2c53da3d19`, `vllm/models/glm5next/amd/ops/kpool_compress.py`; ring fix from local `my-llm` | `r9700_vllm/attn/glm_tail.py` | **Apache-2.0**; upstream copyright header preserved |
| Packed MXFP4 output-tiled GEMV (adapted) | Local `my-llm` vLLM patches `0022-perf-add-packed-RDNA4-MXFP4-decode-GEMV.patch` and `0026-perf-add-tiled-RDNA4-MXFP4-decode-GEMV.patch` | `r9700_vllm/moe/packed_gemv.py` | **Apache-2.0**; source copyright header preserved |
| Scaled FP8 sparse MLA reader and GLM adapter | vLLM `e9757321527ca1ecd514c07c1418dd2c53da3d19`, `rocm_aiter_mla_sparse` backend/ops; earlier port from local `my-llm` patch 0018 | `r9700_vllm/attn/glm_sparse_fp8.py`, `r9700_vllm/compat/glm_sparse_fp8.py` | **Apache-2.0**; upstream copyright headers preserved |

The chat template is worth about +14% speculative-decoding acceptance over the checkpoint's own template. It is
the one file in this repository not covered by our Apache-2.0 grant: permission was given for **this** project,
so it does not automatically pass to forks.

## Upstream, used unmodified and not redistributed

vLLM, ROCm, PyTorch and Triton (Apache-2.0 / MIT / BSD) remain runtime dependencies;
their distributions are not vendored. The bounded GLM kernel adaptations above
are the source-code exception. This project runs as a plugin on stock builds.

## Model checkpoints

Qwen3.8-27B-NVFP4, Qwen3.8-Flash-Next and the DFlash2 drafter are third-party weights under their own model-card
terms. They are not part of this repository.

---

Provenance of the kernels themselves -- including which constants are generated from published format
specifications rather than authored, and how each replacement was written -- is recorded in
[notes/independence.md](notes/independence.md). `tools/gen_kmag.py --check` regenerates the folded-unpack table
from the OCP e2m1 and e4m3 specifications and verifies it against the checked-in copy.

### MiMo BF16 DiffKV and native MXFP4 adapter

The MiMo backend/ops retain vLLM's Apache-2.0 headers. They adapt vLLM
`dee37d89115db4c94a820a79a78a7828e141c910` and the local `my-llm` recipe
`vllm_mimov26r9700_v0.1` patches `0003-route-native-mxfp4-moe-to-r9700.patch`
and `0004-port-diffbot-diffkv-fp8-and-verify-to-rocm.patch`. This integration
uses a unique quantization name, BF16-only KV and pinned MiMo-scoped adapters;
it does not copy the CUDA Diffbot kernels or replace the Xiaomi checkpoint.
