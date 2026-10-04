# DeepSeek V4.1 Flash: audit and correctness candidate

Status (2026-10-04): **not yet qualified to serve on gfx1201**. GPU microtests
have passed. The first full TP8 load reached memory profiling but failed on a
BF16 MoE workspace allocation. A second attempt loaded all eight ranks but
failed at the sparse indexer architecture gate. Stage-1 serving is not qualified;
there is no throughput baseline.

Selected DeepSeek build (approved 2026-10-04):
`vllm/vllm-openai-rocm:nightly-rocm100-18f8f96025b556071eb627076f94df560fbd3a22`,
digest `sha256:ccf5ae13df46441de7945890fa93ffd869aadbd2987ebd01dd918cfaf6f34b49`.
Registry publication: 2026-10-04 05:41 UTC. This revision contains #57071 and
637 commits after our existing pin. Use the common Dockerfile's `BASE` argument
for a separate DeepSeek image; keep existing GLM/MiMo images and pins intact.
Build with `bash docker/build-deepseek.sh`; qualification results are recorded below.

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

**Upstream update:** [PR #57071](https://github.com/vllm-project/vllm/pull/57071),
merged 2026-09-26 as `5840d95284fe3ae9f56f2366074f58d277d69ad8`, fixes this
dispatch and adds Quark MXFP8 block-scale expansion, VL quant mappings and
Engram scale-name mapping. Verified present in main
`155488d853a0bc42df227dbfc74005b3fd488e94`, absent from v0.30.0 and our pin.
Prefer this upstream implementation (audited backport or separate pinned image)
over creating a parallel quantization adapter. The observations below describe
our currently installed version, not latest main.

The PR's end-to-end validation used MI355X/gfx950. Latest main selects a native
MXFP8 linear kernel only for gfx95x; other ROCm GPUs fall back to
`EmulationMxfp8LinearKernel`. That fallback dequantizes weights to BF16 and calls
`F.linear`; it does not itself quantize activations. Therefore block-layout
support is fixed upstream, but gfx1201 activation semantics, memory cost and
full-model correctness still require validation. No runtime upgrade was made.

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

On the old pin, merely passing `--quantization quark` does not prevent this override. The stock
Quark W8A8 per-block matcher also only accepts 128×128/group-128, so bypassing
the override is insufficient. The newer image supplies the complete upstream
fix; validate tiny layers against explicit QDQ before any full model load. Do
not silently reinterpret the checkpoint as W4A16 or the MiMo MXFP4×FP8 scheme.

## Stage-1 candidate

`serve/deepseek-v4.1-flash.sh` calls the shared launcher directly: TP8/C1/8K,
512-token prefill chunks, eager, no APC or speculation, stock RCCL, 10 GiB expert
offload per rank. This is an experimental correctness profile, not a qualified
production configuration. Set `MODELS_DIR` to the parent checkpoint directory
when using a different storage mount. Inspect without launching:

```bash
DRYRUN=1 bash serve/deepseek-v4.1-flash.sh
```

Only `r9700_deepseek` is enabled. `r9700_deepseek_quark` inherits upstream
Quark loading and MoE dispatch. On MXFP8 linears using the BF16 emulation backend,
it applies upstream group-32 activation QDQ first; native MXFP8 execution and
other quantization schemes are unchanged. Registration requires ROCm, gfx1201
and vLLM `18f8f960`. There are no new compute kernels.

Engram uses upstream `use_thp=true`, with private exact-size mapped host storage.
Huge-page coverage is best effort. A scoped guard aborts if host registration
falls back to the power-of-two pinned allocator. Expert UVA offload reuses that
same upstream allocator inside a DeepSeek-only initialization scope, replacing
only its pinning operation. The scope restores Torch even on failure. It does
not change placement or the forward path. GPU registration and view lifetime
passed the microtests below. Full-load memory observations and remaining runtime
blockers are recorded separately.

The backbone has 5.049 GiB of FP8 linear values outside the Engram tables. Default
load-time BF16 dequantization adds approximately that much persistent storage
(0.631 GiB/rank under ideal TP8 division), plus expanded scales and load-time
temporaries. Add this to the earlier storage-only estimate; it is not measured VRAM.

CPU checks in `tests/test_deepseek_quark.py` cover real-checkpoint dispatch when
`DEEPSEEK_MODEL_CONFIG` points to its config, scale expansion before TP8 slicing,
activation rounding, model isolation, exact-size expert copies and allocation
failure guards. GPU capability
checks are mocked in these CPU tests, so passing them cannot qualify GPU serving.

## Preparation results (2026-10-04)

* The separate image built successfully from the digest above, including the
  unchanged gfx1201 HIP library. Existing GLM/MiMo base pins are unchanged.
* Final image suite: **20/20 PASS** (8 audit + 12 Quark/allocation tests),
  run from files embedded in the image without a repository bind mount.
  The Quark tests used the local checkpoint
  configuration, without GPU devices. The initial test-harness failure (AMD
  imports probing GPU metadata) and subsequent attempts are retained locally.
* The MXFP8 numerical fixture exposes a difference in stock emulation with
  identity weights; the adapter matches explicit group-32 activation QDQ exactly,
  including zero inputs. This is not a full-model accuracy test.
* Dry-run: TP8, 8192 context, C1, 10 GiB/rank expert offload, eager, no draft,
  no EP, explicit private Engram allocation. No DeepSeek server was started.
* Evidence: ignored `bench/results/deepseek-stage1/`; image logs under
  `.runtime/deepseek/`. Do not interpret build success or mocked CPU checks as
  GPU qualification, memory-capacity validation or a speed result.

Reproduce the CPU suite inside the candidate image without exposing GPUs:

```bash
docker run --rm --entrypoint python3 \
  -e R9K_PLATFORM=0 -e VLLM_PLUGINS= -e PYTHONDONTWRITEBYTECODE=1 \
  r9700/vllm:deepseek-v41-18f8f960 \
  -m unittest discover -s /opt/r9700/tests -p 'test_deepseek_*.py'
```

The real-checkpoint override test additionally needs a read-only mount of its
`config.json` and `DEEPSEEK_MODEL_CONFIG` pointing to that container path.

## GPU checks and full-load attempts

`tests/test_deepseek_gpu.py` is opt-in (`R9700_DEEPSEEK_GPU_TEST=1`) and loads no
checkpoint. Six tests passed in the candidate image on gfx1201:

* MXFP8 group-32 activation QDQ vs an independent CPU reference for 1/8/16/512
  rows, width 5120, including zeros and different exponent groups: exact match.
* Emulated linear adapter vs FP32 reference using quantized activations
  (`atol=.02`, `rtol=.01`).
* Exact host allocation/UVA reads on all eight cards, retaining the GPU view
  after dropping explicit host references.
* The actual vLLM UVA expert offloader, including values, markers and byte count.
* Sparse attention prefill/decode with 448+64 dimensions, 8 local heads,
  1/2/8/16 rows, unequal lengths, attention sinks and SWA + compressed segments
  (`atol=.03`, `rtol=.02`). This does not qualify complete CSA2 model state.
* The model's actual C++ SWA writer crossing a 128-token page boundary, with
  identity RoPE, vs independently decoded FP8/E8M0/BF16 records: exact match.

Full-load attempt 1: TP8/C1/8K, NBT512, expert offload 10 GiB/rank, utilization
0.94. All eight workers loaded the checkpoint, reporting **26.86 GiB model
memory per rank**. Actual expert offload rounded up to **10.04 GiB/rank** at
parameter boundaries; Engram used two approximately 11.80 GiB host shards per
rank. Minimum sampled host MemAvailable was **25.42 GiB**; system swap filled
during loading, so this is not a claim of a swap-free host or a production-safe
memory margin. GPU usage is sampled every 5 seconds and can miss brief peaks.

The run failed before API readiness in
`OCP_MXQuantizationEmulationTritonExperts.apply -> _dequantize_weights(w2)`:
allocating **1.05 GiB** of BF16 expert values with only **472–498 MiB free**.
This is a real GPU OOM during profiling, not a retrieval/format failure, and no
text or speed result exists for this run. Reading all 48 shards did not mean
loading was complete; stack captures showed subsequent expert tensor copies.

Attempt 2 keeps arithmetic and 8K/C1 fixed, with 11.5 GiB/rank offload and 0.97
utilization. A monitor sends SIGINT if host MemAvailable drops below 8 GiB.
All eight ranks loaded at **25.18 GiB/rank**, down from 26.86 GiB/rank.
The run reached a different fatal error in `SparseAttnIndexer.forward_hip`:
`Sparse attention indexer ROCm path requires AITER or a supported native
architecture (gfx950/gfx11).` The native gate excludes gfx1201 with AITER disabled.
This is separate from the attention kernels exercised by the tiny tests. Do not
remove the gate without validating the downstream indexer path and cache layout.
There was no API readiness, generated answer or throughput measurement. Reaching
this gate does not prove the complete memory profile fits. Both attempts are
retained in ignored `bench/results/deepseek-stage1/` alongside every microtest
attempt. The original GLM container was restarted after the second failure.
`max_parallel_loading_workers` is ignored by this vLLM pin; it cannot be relied
on to serialize loading.

## Reuse and missing work

| Area | Reuse | Required gate/adaptation |
|---|---|---|
| Model, routing, mHC | Pinned vLLM DeepSeek V4.1 implementation | Check all ROCm dependencies on gfx1201; Qwen HC is not interchangeable |
| Expert correctness | Quark OCP MX emulation + existing QDQ helpers | Scoped quant dispatch, TP8 geometry, clamp/routing tests, temporary-memory budget |
| Expert optimization | `moe/w4a4.py`, packed GEMV and HIP entry points | Validate hidden 5120, I=288, E=384, top-6; retain strict GLM gates |
| Dense attention FP8 | Existing upstream quantization building blocks | Preserve block-32/E8M0 weights AND activation scales; no silent requantization |
| Host allocation | Upstream exact mmap/registration allocator; stack construction-scope pattern | Engram/expert guards and HIP/UVA microtests passed; full serving remains blocked |
| Offload/cache | Existing vLLM expert offloader and `moe/cache.py` concepts | Prove Quark layout compatibility, single-copy ownership, per-rank byte budget before LRU/hot-cold |
| Attention/indexer | Upstream ROCm sparse Triton fallback | Validate CSA2/cache sharing and 448+64 geometry; inspect gfx950-only/AITER gates and `fp8_ds_mla` cache assumptions |
| Collectives | RCCL first; existing custom AR later | Isolated A/B only after baseline |
| Serving/bench | Shared `serve.sh`, existing benchmark protocol | Explicit settings; its generic 34 GiB/rank offload default is not this model's budget |

No existing kernels or model registrations are changed by stage 0.
Dense activation rounding and small GPU dense/sparse-attention cases have
reference coverage. Full-model execution and the sparse indexer remain unqualified.

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

1. **Correctness prerequisites:** adopt upstream #57071 and validate its gfx1201
   fallback, adding only missing scoped adaptations; exact Engram allocation;
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

The profile is a launch candidate, not a recommendation. There is no throughput
result yet. Full runtime footprint and the final offload amount remain
measurements to obtain in stage 1.
