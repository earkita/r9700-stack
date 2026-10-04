# DeepSeek V4.1 Flash: stage 0 audit

Status (2026-10-04): **not qualified to serve on gfx1201**. No model load,
GPU kernel test, service replacement or throughput benchmark has been performed.
GLM remains running. This commit adds only a CPU audit and the implementation plan.

Checkpoint: `amd/DeepSeek-V4.1-Flash-Quark-MXFP4`.
The [AMD model card](https://huggingface.co/amd/DeepSeek-V4.1-Flash-Quark-MXFP4)
qualifies gfx950/MI350–355, not gfx1201. Its deployment image differs from ours.
Local config and safetensors headers, rather than the card's summary of original
precision, are the source of the numbers below.

## Reproduce the structural audit

```bash
python3 -m r9700_vllm.models.deepseek_audit "$MODEL_DIR" --expert-offload-gib 80
python3 -m unittest discover -s tests -p 'test_deepseek_audit.py'
```

The offload argument is a **total across TP8**, not a per-rank allocation.
JSON goes to stdout; save local evidence in ignored `bench/results/`.
The audit reads headers only, checks index membership, contiguous byte ranges,
file bounds, byte counts and scale geometry. It does not verify weight checksums,
numerical values, routing or runtime compatibility. `runtime_qualified` stays false.

Observed: 48 shards, 96,085 tensors, 474.575 GiB tensor payload, 474.585 GiB files.
Index `total_size` matches the file sum including headers.
Config SHA256: `bae1de0165f08a5bbf803360972dca113104f09e2a57b56a9447172495bd4bb7`.
Index SHA256: `2f8d2d6d4591049ca0dbe851f9f063d3a002e8c555979956627a09f9cfedab8b`.

## Layout and memory

40 backbone layers, hidden 5120, 384 routed experts/layer, top-6, intermediate 2304.
At TP8 the expert intermediate dimension is 288. Routing is `sqrtsoftplus`,
`noaux_tc`, with scale 1.5; preserve it and the SwiGLU clamp of 10.
Attention: 64 heads, head dimension 512 (448 NoPE + 64 RoPE), sliding window 128,
compressed/cache-sharing attention. The declared model limit is 1,048,576 tokens;
this is not a hardware capacity measurement or a launch target.

* Routed and shared experts: packed U8 MXFP4 weights, group-32 E8M0 scales,
  **MXFP4 activations (W4A4)**. Full `w1/w3` shape `[2304,2560]`, scales
  `[2304,160]`; `w2` `[5120,1152]`, scales `[5120,72]`.
* 226 dense quantization overrides: FP8 E4M3 weights with **32×32 blocks** and
  E8M0 scales, dynamic FP8 group-32 activations. Preserve the activation scheme
  from config too; do not assume every generic MXFP8 implementation is equivalent.
* Engram has two huge FP8 lookup tables with per-row group-32 scales. Projection
  and gate tensors are separate and stay on the accelerator.
* `mtp.*` includes three draft layers and must be accounted separately.

| Stored component | GiB | Stage-1 placement intent |
|---|---:|---|
| Engram tables + scales | 188.833 | Host RAM, mapped/pinned once |
| Engram projections/gates | 0.293 | GPU |
| Backbone routed experts | 268.945 | GPU + deterministic host offload |
| Shared experts | 0.700 | GPU |
| Other dense/router/embeddings/norms | 7.557 | GPU |
| Vision + aligner | 0.904 | Excluded in text-only bring-up |
| DSpark/MTP | 7.342 | Excluded until stage 9 |

Text weights excluding CPU Engram, vision and draft total **277.497 GiB**.
Ideal uniform TP8 division is **34.687 GiB/GPU**, already too large. Actual
VRAM capacity from sysfs is **31.859 GiB/GPU**; host MemTotal is **341.448 GiB**.
All eight GPU links report PCIe 5.0 x16; this is link state, not measured bandwidth.

At utilization 0.94, the weight-only lower bound for offload is 37.914 GiB total.
With an *assumed* additional 4/6/8 GiB per GPU, it becomes 69.914/85.914/101.914 GiB.
These bounds omit nonuniform/replicated weights and backend-specific padding.
The true minimum must be found after correctness with measured load peaks.

**80 GiB total (10 GiB/rank) is a trial budget, not a proven minimum:**
ideal resident weights are 24.687 GiB/GPU, leaving 5.261 GiB within the 0.94 budget
for replication, temporary tensors, activations, allocator overhead and KV.
Quark's MoE emulation alone materializes roughly **3.164 GiB/GPU** of BF16
expert weights for one layer; additional dequantization intermediates may raise
the peak. It also reads all experts when dequantizing, not only the top six.
This is a correctness fallback, not a predicted fast offload implementation.

Host payload with exact Engram allocation and 80 GiB experts is **268.833 GiB**,
leaving roughly 72.615 GiB for OS, processes and loading overhead. Avoid a second
full host copy. If each separate Engram weight/scale allocation rounds to a power
of two, Engram instead needs **264 GiB**, and with 80 GiB experts exceeds physical
RAM before any process overhead. This is an allocation scenario derived from the
headers, not measured DeepSeek RSS. Exact pinning is a prerequisite, not optional
tuning. Do not depend on swap to hold pinned model weights.

## Confirmed blocker in pinned vLLM

Audit target: vLLM `e9757321527ca1ecd514c07c1418dd2c53da3d19`, the existing image.
It contains `DeepseekV41ForCausalLM`, ROCm attention, Engram, DSpark, tokenizer,
reasoning and tool parsers. Registration alone does not establish compatibility.

`vllm/models/deepseek_v41/quant_config.py` automatically routes this Quark config
to `deepseek_v4_fp8`. `from_config` then unconditionally selects `[128,128]`,
discarding the checkpoint's `[32,32]` overrides. The MoE dispatch chooses
`Mxfp4MoEMethod`, not the Quark activation-QDQ W4A4 path.
CPU-only reproduction using the full local config returned:

```text
overrides: 226
auto: deepseek_v4_fp8
explicit_quark: deepseek_v4_fp8
translated_weight_block: [128, 128]
```

Merely passing `--quantization quark` does not prevent this override. The stock
Quark W8A8 per-block matcher also only accepts 128×128/group-128, so bypassing
the override is insufficient. Implement a version- and model-scoped quantization
adapter retaining all Quark semantics; validate tiny layers against explicit QDQ
before any full model load. Do not silently reinterpret the checkpoint as W4A16
or the MiMo MXFP4×FP8 scheme.

## Reuse and missing work

| Area | Reuse | Required gate/adaptation |
|---|---|---|
| Model, routing, mHC | Pinned vLLM DeepSeek V4.1 implementation | Check all ROCm dependencies on gfx1201; Qwen HC is not interchangeable |
| Expert correctness | Quark OCP MX emulation + existing QDQ helpers | Scoped quant dispatch, TP8 geometry, clamp/routing tests, temporary-memory budget |
| Expert optimization | `moe/w4a4.py`, packed GEMV and HIP entry points | Validate hidden 5120, I=288, E=384, top-6; retain strict GLM gates |
| Dense attention FP8 | Existing upstream quantization building blocks | Preserve block-32/E8M0 weights AND activation scales; no silent requantization |
| Host allocation | `utils/hostmem.py:pinned_empty`, UVA helpers | Engram allocates via `torch.empty(pin_memory=True)`, which `exact_pinning()` does not intercept; add a scoped allocation adapter |
| Offload/cache | Existing vLLM expert offloader and `moe/cache.py` concepts | Prove Quark layout compatibility, single-copy ownership, per-rank byte budget before LRU/hot-cold |
| Attention/indexer | Upstream ROCm sparse Triton fallback | Validate CSA2/cache sharing and 448+64 geometry; inspect gfx950-only/AITER gates and `fp8_ds_mla` cache assumptions |
| Collectives | RCCL first; existing custom AR later | Isolated A/B only after baseline |
| Serving/bench | Shared `serve.sh`, existing benchmark protocol | Explicit settings; its generic 34 GiB/rank offload default is not this model's budget |

No existing kernels or model registrations are changed by stage 0.
Neither the dense path nor sparse attention is yet numerically qualified here.

## Lessons from the 8×5090 reference

Inspected [DeepSeek-V4.1-Flash-Accel](https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel/tree/4a4e88ec2cdcf2fc7c802e1f6b7c943d897770c8)
at `4a4e88ec2cdcf2fc7c802e1f6b7c943d897770c8`; it pins a different vLLM revision
(`8c1d1c2974ee42757ee2e93cc898932edfd9d265`), backend and checkpoint.
Reuse exact host allocation, single-copy offload and measured placement ideas.
Its CUDA VMM page remapping is not a HIP implementation. Likewise, its Marlin
padding savings with EP are backend-specific: our emulation aligns to 32, so
TP8 I=288 does not incur Marlin's 288→384 padding. EP has no proven memory win yet.

Routing calibration must cover general/coding workloads and unseen evaluation
prompts, separating prefill/decode. Count estimated host bytes separately from
measured PCIe traffic. Adapt profiling to arbitrary rows/top-k; do not inherit
the reference's fixed speculative batch assumptions. All backbone layers run
during prefill; do not assume offloading later layers eliminates prefill traffic.
Its reported output throughput includes request time; do not compare it directly
with our decode-only rate. Preserve our frozen A/B methodology.

## Implementation gates and commit boundaries

1. **Correctness prerequisites:** scoped quant adapter; exact Engram allocation;
   QDQ/scale, TP8 sharding, UVA lookup and sparse-attention microtests. Separate
   correctness commits from performance work. Fail clearly on unsupported layout.
2. **Stage 1 load:** one explicit profile calling `serve.sh`; TP8, 8K then 16K,
   C1, eager, RCCL, Engram CPU, minimal measured expert offload, no vision/draft/
   speculation/experimental fusions. Validate text, finite tensors, all eight
   workers and memory balance. Save a reversible GLM launch configuration first.
3. **Stage 2 baseline:** 64K C1, OpenAI API and 30–60 minute stability test. Record
   per-request decode, step latency, TTFT/prefill, VRAM/RAM, PCIe; save protocol
   and results under `bench/results/`. No performance claims before this gate.
4. **Stages 3–4:** existing optimizations one at a time, then qualify deterministic
   offload at the measured budget. The offload needed to load already exists in
   stage 1; stage 4 studies its cost rather than assuming weights can all fit.
5. **Stages 5–6:** routing counters without placement changes, then hot/cold
   placement vs naive on held-out prompts. Measure hit rate and actual traffic.
6. **Stages 7–8:** TP/EP comparisons, then 64K/128K/256K attention capacity tests.
7. **Stages 9–10:** DSpark K3/K5 (optional K1/K7), acceptance by position, draft
   cost and memory; only then select a qualified profile.

There is intentionally no serving command or throughput result yet. Stage 0
establishes storage geometry and a memory budget; full runtime footprint and
the final offload amount remain measurements to obtain in stage 1.
