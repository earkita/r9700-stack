# R9700 adapter for SGLang

This directory contains our isolated compatibility patches for the pinned
SGLang v0.5.20 source (`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`).
`docker/Dockerfile.sglang-glm` applies them to that exact revision when building
`r9700/sglang:glm53-v0.5.20-gfx1201`. The upstream source checkout in
`.runtime/sglang-v0.5.20` is a local work area, not part of the deliverable.

The launch profile is `serve/baseline-earkita/glm/glm-5.3-flash-sglang.sh`.
It uses a separate container and port (8081) from vLLM. On 8× R9700, the
pinned AITER MXFP4 MoE path returned zeros in a synthetic real-shape test and
the full model repeated a prompt. The gfx1201 adapter now routes this MoE
through packed GEMV, retaining the checkpoint's MXFP4 weights and E8M0 scales.
A bounded full-model smoke test passed a completed chat answer, streaming and
a two-turn tool call. This initial smoke alone did not qualify long-context quality or performance.
Later bounded C1 results are listed below; the profile remains **experimental**,
with concurrency and broad quality/stability still unqualified.
Raw attempts and PCIe evidence are in ignored
`bench/results/glm-sglang-v0520-smoke-20261015/`.

The first frozen C1 8K throughput diagnostic measured 4.413 tok/s median
decode over three requests. Its exact protocol and failed first attempt are
documented in the [SGLang profile note](../serve/baseline-earkita/glm/glm-5.3-flash-sglang.md).

The repository also contains isolated, opt-in decode experiments. The
one-row BF16 dense GEMV is enabled with `R9K_SGLANG_BF16_DENSE_GEMV=1` after
the checkpoint's MXFP4 dense weights have been dequantized once at load. A
matched SGLang-only C1 8K test found a 2.6% median decode gain. The packed
MoE down-projection tile can be changed from 8 to 64 with
`R9K_SGLANG_DOWN_BN=64`; it was faster in an isolated MoE probe but 1.14%
slower in median C1 8K model decode, so BN8 remains the default. The native
gfx1201 HIP W4A4 C1 path can be selected with
`R9K_SGLANG_MOE_BACKEND=hip_w4a4`, while prefill remains on packed GEMV. Its
first full-operator test was numerically identical to packed GEMV but slower
at C1, so it is not enabled by default. Each experiment is independent;
none is a qualified replacement for the production vLLM profile.

The decode-only `R9K_SGLANG_MHC_SINKHORN=1` experiment replaces the reference
Torch Sinkhorn loop with the pinned upstream Triton operator. The projection,
residual mixing, MoE, graphs and speculation settings stay unchanged. Its
isolated numerical/timing probe is `tuning/sglang_mhc.py`; this operator result
must not be reported as whole-model decode speed. The default remains `0`.
The explicit selector is `R9K_SGLANG_MHC_BACKEND=torch|sinkhorn|vllm_tilelang`.
The last variant reuses the pinned vLLM package's independent fused mHC pre
operator and needs its own model comparison, because its projection reduction
order differs. `DECODE_GRAPHS=1` and `HWQ=1` are independently prepared launch
overrides; neither is part of the mHC-only experiment.

The qualified C1 experiments now include full mHC pre and full decode graphs.
Separate five-sample cold 8K A/Bs observed 6.069→8.266 tok/s for full mHC,
then 8.266→20.041 tok/s for graphs. These are controlled SGLang comparisons,
not comparisons with production K4.

The long-context adapter also avoids expanding a full page-table row per
prefill token and avoids materializing query×history int64 score indices.
`kpool.py` keeps the original Torch top-k selection, with exact integer
mapping and bit-exact FP32 shift/mask kernels. Standalone correctness,
allocation, changing-metadata graph replay and integrated top-k tests passed.
Image `r9700/sglang:glm53-v0.5.20-gfx1201-kpool-workspace-v3` passed bounded
Python, 32K/256K exact-key, streaming and two-turn tool tests at TP8/C1.
It uses 32 KDA state slots, BF16 KV and unchanged FP32 recurrent states.
The final-image five-sample warm-prefix matrix measured median decode
31.914/31.863/31.657/21.258 tok/s at actual inputs
8,192/32,768/130,816/261,888 tokens respectively, excluding TTFT.
A separate appended-query exact-key test reused 261,824 cached tokens and
completed correctly. Full ranges, cold observations and limitations are in the
profile note; do not compare these warm rates to earlier cold protocols or
treat limited probes as broad quality equivalence.

Keep version-specific SGLang changes here. Do not copy these patches into
`r9700_vllm`, which belongs to the vLLM integration. Before comparing model
performance, follow `serve/baseline-earkita/glm/testing.md`, including the
PCIe evidence and identical comparison protocol.
