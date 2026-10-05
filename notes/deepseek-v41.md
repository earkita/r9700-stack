# DeepSeek V4.1 Flash: audit and correctness candidate

Current status (2026-10-05): **experimental, not qualified for agent workloads**.
Preserved on `experimental/deepseek-v4.1-flash`; the pre-experiment tree is
`feature/mimo-v2.6-flash` at `fcee46a`. EP8, reserved prefill staging and calibrated
HIP VMM placement passed bounded checks, but thinking/tool generation remains
unqualified and DSpark did not pass startup qualification. There is no matched
end-to-end performance result. The dated entries below retain the earlier
attempts and their limitations. Runtime artifacts stay in ignored `bench/results/`.

Status (2026-10-04): **not yet qualified to serve on gfx1201**. GPU microtests
have passed. The first full TP8 load reached memory profiling but failed on a
BF16 MoE workspace allocation. A second attempt loaded all eight ranks but
failed at the sparse indexer architecture gate. Attempt 4 reached memory profiling
and exposed a configuration-context lifetime bug in our indexer adapter. That bug
is now fixed and covered by CPU/GPU tests. Stage-1 serving is not qualified;
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
512-token prefill chunks, eager, no APC or speculation, stock RCCL, 11 GiB expert
offload per rank, GPU utilization 0.975. This is an experimental correctness profile, not a qualified
production configuration. Set `MODELS_DIR` to the parent checkpoint directory
when using a different storage mount. Inspect without launching:

```bash
DRYRUN=1 bash serve/deepseek-v4.1-flash.sh
```

Only `r9700_deepseek` is enabled. `r9700_deepseek_quark` inherits upstream
Quark loading and MoE dispatch. On MXFP8 linears using the BF16 emulation backend,
it applies upstream group-32 activation QDQ first; native MXFP8 execution and
other quantization schemes are unchanged. Registration requires ROCm, gfx1201
and vLLM `18f8f960`. The initial candidate used stock MoE emulation; the
opt-in packed decode adapter and indexer additions are described below.

Engram uses upstream `use_thp=true`, with private exact-size mapped host storage.
Huge-page coverage is best effort. A scoped guard aborts if host registration
falls back to the power-of-two pinned allocator. Expert UVA offload reuses that
same upstream allocator in the DeepSeek-only offloader. It copies directly
from the original parameter into one exact-size registered buffer, without a
pageable intermediate or a global Torch patch. Selection, offload accounting
and placement are preserved. GPU registration and view lifetime
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

## gfx1201 sparse indexer candidate

`compat/deepseek_indexer.py` enables only DeepSeek V4.1 on the separate
`18f8f960` / gfx1201 image, in eager mode with AITER disabled. It rejects FP4
indexer cache, context parallelism and unaudited compression/geometry. Other
models retain their original `SparseAttnIndexer.forward_hip` behavior.

The upstream ROCm indexer function is reused with a private globals dictionary
binding its paged-logits reader to `attn/deepseek_indexer.py`. This avoids both
pretending the GPU is gfx950 and changing a shared upstream function binding.
Insertion, prefill gathering, candidate selection/masking and top-k stay upstream.
The separate reader follows the existing stack tiled-reader algorithm without
modifying GLM's kernel: H32/D128, OCP FP8 values in 16x16 tiles, followed by FP32
scales; compression 1/2 and pages 128/64. The stock Torch decode fallback reads
values as token-major and is unsuitable for this cache layout.

Validation: 6 CPU contract/isolation tests; 4 GPU tests, also repeated together
with the launcher's `VLLM_USE_BREAKABLE_CUDAGRAPH=1`:
writer/gather exact match; paged logits for 1/2/8/16 rows, unequal lengths and
per-row/per-request context lengths against FP32 reference; full prefill/decode
indexer including candidate write/read and top-512 selection; actual model
normalization/RoPE/cache writer with skipped slots and compression boundaries
against independent byte packing. The final CPU image suite passed 26 checks
with 2 GPU-only modules skipped. These checks do not qualify full-model serving,
context parallelism, graph capture, speculation or throughput.

Attempt 3 stopped before weight loading: the launcher auto-enables a graph
wrapper around the indexer even in eager mode, which the adapter's interface
guard rejected. The adapter now unwraps the eager body (graph mode remains
explicitly rejected); the CPU regression and all four GPU tests passed with
that flag enabled. Attempt 4 uses the same 8K/C1, NBT512, 11.5 GiB/rank offload
and 0.97 utilization as attempt 2, with the sparse-indexer adaptation added.
Attempt 4 loaded all eight ranks at 25.18 GiB each, but failed before API
readiness during memory profiling: `compat/deepseek_indexer.py:forward`
called `_is_deepseek()` / `get_current_vllm_config()` outside the construction
configuration context. This is a plugin lifecycle bug, not a numerical
indexer failure. Capture model identity and eager eligibility during operator
construction and add a regression executing forward after that context exits;
the earlier context-wrapped microtests did not cover this lifecycle.

Loading was severely uneven: first ranks finished in 975 s, TP5 in 3783 s and
TP4 in 3843 s. Repeated snapshots showed expert `copy_` calls and continued
progress, not a demonstrated deadlock. Layer numbers are not completion
percentages: TP4 visited layer 8 after layer 39. A 15-second sample recorded
1296 MiB of storage reads on TP5 while TP4 had no reads or major faults.
Per-thread sampling found one busy native thread per rank; a native backtrace
on TP0 places that thread in HSA/HIP, while its main thread waited in ZMQ.
The container had no CPU quota/throttling in the inspected counters. These
observations do not establish the root cause of the loading delay or prove
that OpenMP, disk, or swap alone caused it. Short debugger attachments can
perturb timings; this attempt is diagnostic, not a performance baseline.

Evidence is in ignored `bench/results/deepseek-indexer/`. GLM was stopped for this continuation and
must not be automatically restarted while DeepSeek bring-up is ongoing.

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

### Planned W4A4 kernel adaptation

Reuse the existing gfx1201 `r9k_moe_mxfp4a4_c1` / batch HIP entry points and
MXFP4-QDQ encoding as the starting point for a separate DeepSeek adapter.
Keep GLM/Qwen dispatch and validated configurations unchanged. Target TP8:
hidden=5120, intermediate=288, experts=384, top-k=6. Preserve original packed
MXFP4 weights, E8M0 group-32 scales, activation QDQ, clamp=10 and upstream
DeepSeek routing/weight application. FP8 transport of QDQ values must remain
lossless; this is not a change to W4A8 model arithmetic.

The existing Python launch heuristic chooses Split-K=2 for down K=288, which
violates the HIP entry's `K % (Split-K * 32) == 0` requirement. Start with
Split-K=1 for this shape; satisfying the launch guard is not numerical or
performance qualification. Check the actual loaded strides and padding too.

Validate one expert and the routed MoE against Quark emulation, including
scale extremes, zeros, clamp boundaries, repeated experts, C1 and multiple
rows. Separately test resident and host/UVA expert storage and lifetime.
Keep emulation as a fallback for unqualified shapes, including prefill until
covered. The intended benefit is selected-expert fragment decoding in
registers without a full-layer BF16 weight allocation. Measure peak memory
and latency with the frozen baseline protocol after correctness; no speedup
is assumed. This does not implement hot/cold placement or fix weight loading.

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
4. **Stages 3–4:** qualify the W4A4 adaptation above and other existing
   optimizations one at a time, then qualify deterministic
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

## Four-fix candidate (2026-10-04)

1. Expert offload now copies directly into exact-size registered host storage.
   CPU tests reject use of an intermediate `Tensor.to` or `pin_memory`; the
   real GPU offloader test checks values, markers and accounting.
2. Model-scoped loading diagnostics distinguish iterator time from consumers
   and record actual expert-copy durations. The ignored loader-parallelism flag
   is removed; Torch kernel cache uses writable repository `.runtime` storage.
   These changes expose the slow-load cause; they do not yet prove a speed fix.
3. Sparse indexer captures model identity and eager mode during construction.
   Forward no longer reads the expired vLLM configuration context. Its GPU test
   explicitly executes with no current configuration.
4. `R9K_DEEPSEEK_MOE=w4a4` enables packed expert decode for TP8, BF16 inputs,
   1–16 rows, top-6, 384 experts, hidden 5120, intermediate 288 and clamp 10.
   Other calls, including prefill, retain upstream Quark emulation. The separate
   Python adapter reuses existing gfx1201 HIP exports; GLM/Qwen source is unchanged.
   K=288 uses split-K=1, while K=5120 uses split-K=4. No hot/cold cache is added.

Runtime attempt 5 uses image `bafec6041872` (same pinned vLLM). The subsequent
image `7b2fb83e0f38` only corrects diagnostics so source/target/UVA fields belong
to the slowest reported copy, rather than the last copy in that interval.
Embedded CPU suite: 30 PASS,
3 GPU modules skipped. GPU suite: 12/12 PASS, including sparse indexer and
MoE fixtures at 1/2/8/16 rows with both resident and host-mapped weights.
A follow-up 2/2 MoE run also exercised the actual scoped dispatcher and rejected
accidental stock dequantization. Packed output matched Quark W4A4 within
`atol=.02, rtol=.01`; these are synthetic fixtures, not model-quality results.
Evidence is in ignored `bench/results/deepseek-fixes/`. Full startup and response
validation remain required; the default profile keeps the new MoE path opt-in.

### Attempt 5: host OOM and desktop-session loss

All eight ranks loaded (six in approximately 594 s, TP7 in 627 s, TP5 in
799 s), reporting 25.18 GiB/rank. These are diagnostic observations, not a
controlled speed comparison with earlier attempts. TP5 read 14.5 GiB from
storage in a 15-second late-copy sample.

During profiling, host MemAvailable fell below 8 GiB. The monitor sent SIGINT
at 23:37:14 Europe/Warsaw, but startup did not stop promptly. An explicit
SIGTERM to EngineCore followed. At approximately 23:39:10–17, the kernel
reported global host OOM, killed a VS Code process, and systemd terminated
the GNOME session. The OS did not reboot (boot time 2026-09-30 19:17 CEST).
The container also logged RCCL allocation failure for 2 MiB and HIP OOM, and
exited at 23:39:38. Docker OOMKilled=false does not rule out this host OOM.

The SIGINT-only memory guard was insufficient. No API readiness or answer
validation was achieved. Do not repeat this launch unchanged: reduce the
profiling/prefill memory peak and provide a bounded shutdown targeting the
engine/workers before another full load. W4A4 remains qualified only by
microtests for 1–16 rows; prefill still uses full-weight Quark emulation.
Raw runtime, memory samples and host journal are retained in ignored
`bench/results/deepseek-fixes/`.

### Packed prefill and bounded host-memory protection

The W4A4 candidate now covers 1–512 rows, including prefill. Numerical checks
passed for 1/2/8/16/17/32/128/512 rows with resident and host-mapped weights;
the 512-row case routes all rows to the same six experts to exercise multiple
alignment tiles per expert. Outputs match stock Quark within `atol=.02, rtol=.01`.
The real dispatcher is tested with stock dequantization disabled. Unsupported
DeepSeek calls now fail clearly instead of silently expanding full BF16 expert
weights. Other models keep upstream behavior.

Attempt 6 used packed W4A4, 9 GiB/rank expert offload and
utilization 0.94. Docker caps host memory at 296 GiB, disables container swap
and assigns OOM score adjustment 500. This is a new configuration, not an A/B
performance result. Scope remains TP8/C1/8K/NBT512, eager, no draft/APC.

`serve/runtime_memory_guard.py CONTAINER --log PATH` separately watches host
headroom (current default 4 GiB, checked every second) and binds the immutable container
ID. It requires `--oom-score-adj >= 500`; on low headroom it sends TERM, then
KILL after at most two seconds. Run it as a persistent user service while
qualifying a full load. A real disposable container that ignored TERM was
successfully killed in approximately 1.2 seconds using a one-second test grace.
No host memory exhaustion was needed for that forced-threshold test.

CPU suite: 30 PASS plus 3 skipped GPU modules. Expanded MoE GPU suite: 2/2 PASS
with all row/storage subcases. Evidence: ignored `bench/results/deepseek-prefill/`.

The 9 GiB offload trial (rather than 10) reserves additional host headroom
for the 24 GiB guard, based on attempt 5's measured loading footprint. Whether
its larger GPU-resident weight set fits profiling/KV was tested in attempt 6.
A second disposable test confirmed the Docker memory cap is enforced: requesting
128 MiB under a 64 MiB cap produced container-only OOMKilled=true / exit 137.

Attempt 6 loaded all eight ranks (27.7 GiB model memory per GPU), but profiling
reported -0.88 GiB available for KV cache and the server exited before API
readiness. The current candidate restores 11.5 GiB/rank expert offload at the
user's request; NBT512, utilization 0.94 and Docker RAM limits remain unchanged.
This revised candidate was tested in attempt 7 below.

### Attempt 7: 11.5 GiB/rank reaches the host-memory guard

On 2026-10-05 at 00:17–00:19 Europe/Warsaw, the same packed-W4A4 image was
started with only expert offload changed from 9 to 11.5 GiB/rank. NBT512,
utilization 0.94, TP8/C1/8K and the 296 GiB container cap were preserved.
The runtime confirmed `cpu_offload_gb=11.5`; ranks that completed expert
allocation reported 11.72 GiB because allocation operates on whole tensors.
The structural preflight's printed memory estimate still assumes its default
80 GiB total expert offload; that estimate is not the actual launch setting.

The external guard observed 23.860 GiB host MemAvailable (below its 24 GiB
floor), sent TERM, then KILL after two seconds. The container exited 137 with
OOMKilled=false. No OOM events were present in the last cgroup sample or in
the inspected kernel journal interval. This was a protective stop, not an
inference/runtime correctness failure. API readiness was never reached; no
smoke requests were sent, and no performance result is available.

The highest sampled container usage was 295.951 GiB: anonymous memory
292.884 GiB, file memory 2.358 GiB (including 1.784 GiB shmem), and kernel
memory 0.704 GiB. Thus file-cache reclamation alone is not a demonstrated
solution. Anonymous huge pages accounted for 281.502 GiB, consistent with the
large exact host allocations, but these counters do not attribute all remaining
overhead to individual allocations. The container released its memory after
stopping. At the end of attempt 7 the profile retained 11.5 GiB offload and NBT512; no further launch or
lowering of the guard was performed. Evidence and full logs are in ignored
`bench/results/deepseek-attempt7/`.

After attempt 7, the user requested 10.5 GiB/rank expert offload. The current
candidate uses that budget with NBT512, utilization 0.94 and memory protections
unchanged. It passed launcher dry-run validation and was tested in attempt 8.

### Attempt 8: 10.5 GiB/rank reaches the guard at checkpoint-load start

On 2026-10-05 at 00:23–00:25 Europe/Warsaw, the unchanged image and smoke
protocol were launched with 10.5 GiB/rank offload. All eight ranks reported
10.61 GiB actual expert offload. TP0 finished model initialization and began
the safetensors loader, but host MemAvailable fell to 21.223 GiB. The unchanged
24 GiB guard sent TERM and then KILL; exit 137 was a protective stop,
OOMKilled=false. No OOM was found in the inspected kernel interval or the last
cgroup sample. API readiness and inference tests were not reached.

The highest 5-second cgroup sample was 288.013 GiB (284.999 GiB anonymous,
2.324 GiB file including 1.784 GiB shmem, 0.686 GiB kernel). That sample is not
the exact peak at shutdown; the 1-second host samples show a further drop
while GPU allocations ramped up and checkpoint loading began. Thus the lower
minimum host headroom than attempt 7 does not imply that reducing offload
increased its allocation: the attempts stopped at different startup stages.
The profile remains at 10.5 GiB/rank and NBT512, and the server is stopped.
Evidence: ignored `bench/results/deepseek-attempt8/`.

After attempt 8, the user explicitly requested lowering the host MemAvailable
guard threshold to 8 GiB. This is now the guard's default; the previous attempts
used 24 GiB as recorded above. The 296 GiB container cap, disabled container
swap, 10.5 GiB/rank expert offload and NBT512 are unchanged. No new runtime
attempt had been launched at the time of that change; attempt 9 follows.

### Attempt 9: loaded weights, then container-scoped RAM OOM

On 2026-10-05 at 00:28–00:41 Europe/Warsaw, the same 10.5 GiB/rank profile
and image ran with the explicitly requested 8 GiB host guard. The 296 GiB
container cap remained unchanged. All eight ranks finished model loading in
629.22–629.35 seconds, reporting 26.28 GiB model memory per GPU.

During startup memory profiling, the container reached its RAM limit. The
kernel journal records `CONSTRAINT_MEMCG` OOM kills of TP6 and TP0; cgroup
events report `oom=2, oom_kill=2`. The remaining workers shut down and the
container exited 1 with OOMKilled=true. This was a container-limit OOM, not a
host-global OOM or intervention by the 8 GiB guard. Minimum sampled host
MemAvailable was 22.564 GiB, and the guard never sent a stop signal.

No KV capacity result or API readiness was reached, and no smoke requests
were sent. The server is stopped; offload 10.5 GiB/rank, NBT512, guard 8 GiB
and cap 296 GiB remain configured. Logs, kernel OOM evidence, frozen protocol
and result summary are in ignored `bench/results/deepseek-attempt9/`.

After attempt 9, the user explicitly requested removing Docker's RAM cap.
The current profile omits both `--memory` and the dependent `--memory-swap`
option. The external guard now accepts containers without those limits and
retains the 8 GiB host MemAvailable threshold, one-second polling and bounded
TERM/KILL shutdown. It must still be launched separately; it is not a hard
kernel-enforced memory reservation. OOM score adjustment remains 500.
Offload stays at 10.5 GiB/rank and NBT512. This change has not been relaunched.


### Attempt 10: KV allocated, first request exposed MXFP4 alignment

The uncapped-container run (offload 10.5 GiB/rank, utilization 0.94, NBT512)
reached API readiness on 2026-10-05 at 01:01 Europe/Warsaw. Seven ranks loaded
in about 602 seconds and TP5 in 634 seconds, each reporting 26.28 GiB model
memory. Profiling reported 0.60 GiB available KV per GPU and a cache capacity
of 143,552 tokens; configured request context remained 8192/C1. This capacity
estimate is not long-context qualification. Minimum sampled host MemAvailable
was 18.395 GiB, peak sampled container usage 317.443 GiB, with no OOM events
or guard intervention.

The first arithmetic smoke prompt had 17 input tokens and failed HTTP 500 in
Quark HIP activation QDQ of the shared-expert down projection. Its TP8 K=288
means an odd number of rows has an element count divisible by 32 but not 64;
Quark's HIP launcher requires 64. The 512-row startup profile did not expose
this. The server shut down; no completed answers or throughput measurements
were obtained. Evidence: ignored `bench/results/deepseek-attempt10/`.

A scoped linear adapter candidate pads odd-row MXFP4 emulation calls with one
zero row and removes that output row. MXFP4 group boundaries and original
weights are preserved. CPU dispatch checks passed. The first numerical test
against Quark Triton failed; a separate probe reproduced HIP/Triton differences
also on unmodified, even-row inputs. This is being investigated before a full
rerun; do not treat the candidate as numerically validated yet. All attempts
are retained in ignored `bench/results/deepseek-linear-padding/`.

The user subsequently requested utilization 0.975 and offload 11 GiB/rank.
Those are now configured for the next run; NBT512, host guard 8 GiB and no
Docker RAM cap remain. They were not the settings measured in attempt 10.


The MXFP4 padding candidate subsequently passed all 18 row/bias cases
(1/2/3/8/17/31/128/511/512 rows, K=288, N=5120) against a CPU reference of
Quark HIP's BF16 scale rounding and FP4 ties-to-even, with an independent
packed-weight LUT and FP32 linear reference (`atol=.03, rtol=.01`). Activation
QDQ also matched that reference exactly. The test first reproduces the original
17-row launcher error. The original HIP/Triton mismatches were retained, not
hidden by relaxing tolerances. CPU scheme-selection tests: 13 PASS.
Evidence: `bench/results/deepseek-linear-padding/gpu-reference.log` and `cpu.log`.
A rebuilt image and the requested offload11/util0.975 configuration are under
qualification in attempt 11; no performance result is claimed from unit tests.

### Attempt 11: loading-time diagnosis

All eight ranks finished loading: 623.76–692.62 seconds, 25.44 GiB model
memory per GPU. The weight iterator yielded 96,085 tensors per rank in
148.9–153.1 seconds. Its exhaustion is not completion of physical disk reads:
buffered mmap tensors can fault in file pages later, when copied.

Logged expert-loader windows account for at least 384.2–480.8 seconds per
rank and 87,296–91,906 calls; the final partial window is not reported.
These are inclusive wall times, not pure PCIe transfer time. TP2/4/5 finished
roughly 64–69 seconds after the first ranks. A five-second sample near the
end observed 5.55–6.66 CPU cores per remaining worker and 393–466 major page
faults/second. Concurrent system-wide samples showed 2.19–2.37 GiB/s reads
on the model volume, `/dev/md127`. This supports investigating expert
sharding/copies and mmap page traffic; it does not isolate a single cause
or prove storage saturation. `OMP_NUM_THREADS=8` per worker is another
candidate for a controlled loading-only experiment, not a measured fix.

After all ranks finished, nonblocking stack samples showed TileLang mHC
initialization followed by Triton compilation in the scoped W4A4 activation
encoder. These are startup/profile costs separate from checkpoint loading.
No settings were changed during these observations. Evidence is retained in
ignored `bench/results/deepseek-attempt11/loading-breakdown.json`,
`loading-pidstat.txt`, `loading-iostat.txt`, and the two TP4 stack captures.
The file named `loading-stack-tp4.txt` was captured just after loading ended
and therefore shows the first forward, not an expert copy.

Attempt 11 reached readiness and passed three short, completed non-thinking
responses: arithmetic, Polish text and exact-key retrieval. TP0 reported
2.44 GiB available GPU KV and the engine reported 624,956 cache tokens; the
configured context remained 8192/C1 and this is not long-context validation.
During the 2910-token active-indexer probe, host MemAvailable fell to
7.924 GiB and the 8 GiB guard sent TERM then KILL. Docker reports exit 137,
OOMKilled=false; sampled cgroup OOM counters stayed zero and the captured
kernel journal contains no OOM kill. The server is stopped.

The interrupted probe has unknown strict/retrieval/format results and failed
completion, with a known operational interruption. It is not evidence of
incorrect retrieval. The following thinking request found the server already
closed; streaming/tools and performance were not reached. Peak recorded
cgroup usage was 318.298 GiB. In the final samples, cgroup anonymous usage
stayed near 301 GiB despite falling host headroom, so the host-memory loss
cannot yet be attributed to a growing Python/model buffer. GPU memory use
was near physical capacity; driver allocations/spill and other host users
require measurement before assigning a cause. No automatic retry or parameter
reduction followed. Full results are in `bench/results/deepseek-attempt11/`.

### Streaming checkpoint and direct host expert copies

`R9K_DEEPSEEK_LOAD=stream` enables a pinned, DeepSeek-only loader adapter.
When vision/aligner are both `StageMissingLayer`, it streams the existing
weight mapper into a single contiguous language-model group instead of
sorting and retaining all checkpoint tensors. Vision weights are skipped
only in this text-only case; unknown mapped roots fail explicitly. The
language model still finalizes once, after all its tensors have loaded.
The initial text-only adapter kept vision-enabled loading on the upstream path;
the vision extension below removes that full-checkpoint sort as well.

The exact offloader retains a CPU alias of its existing allocation (no
second copy). For packed expert weights and raw E8M0 bytes, the loader uses
that alias with upstream TP slicing/copy logic, then restores the accelerator
alias in `finally`. This avoids sending host-resident expert loads through
the GPU view. Dense linears, resident experts and non-DeepSeek models keep
their prior paths. `R9K_DEEPSEEK_LOAD=stock` allows controlled fallback.

Validation: 16 CPU tests passed, one optional checkpoint test skipped;
32 GPU subcases compare exact bytes against upstream TP8 loading for every
rank, both gate/up and down weights/scales, including an untouched expert.
CPU tests also cover streaming consumption, single finalization, vision
fallback, unknown roots and alias restoration on exceptions. Shell syntax,
profile dry-run and the rebuilt image passed. Evidence is in ignored
`bench/results/deepseek-loading-fix/`.

Attempt 12 uses these changes and an explicit **1 GiB/GPU KV override** for
one bounded repeat of the same three short prompts and 2910-token retrieval.
Offload11, utilization0.975, NBT512, 8K/C1 and host guard8 remain. This reduces
the KV allocation to leave GPU working memory without reducing the prefill
chunk. It is a correctness/memory investigation, not a matched throughput
or loading-speed comparison; the profile's automatic KV default has not yet
been changed. The guard now records per-device GTT/VRAM and host page/swap
counters. GTT growth alone does not prove spill, since ordinary host-visible
allocations also contribute. Results: `bench/results/deepseek-attempt12/`.

Attempt 12 reached API readiness and completed the three short probes
(arithmetic, Polish response, exact-key retrieval). The 2910-token probe then
failed on its first decode step: `paged_logits` rejected
`cache.shape[1] in (64, 128) and cache.is_contiguous()` at
`r9700_vllm/attn/deepseek_indexer.py:57`. The traceback does not identify which
part failed; actual cache geometry must be established before relaxing the
guard. Retrieval/strict/format are unknown, completion failed and runtime
error is confirmed. The shared-memory cleanup warning follows this failure.

The host guard did not intervene; minimum sampled MemAvailable was 10.110 GiB,
Docker OOMKilled was false. The API process exited with code 0 despite the
engine failure, so that exit code is not a success criterion. Reported KV
capacity was 256,206 tokens with the explicit 1 GiB/GPU override; only the
configured 8K/C1 smoke was attempted, not capacity qualification. Loading
times ranged from 457.38 to 1315.85 seconds across ranks; no loading-speed
improvement is established. A native stack sample found HIP/HSA inside a
tensor copy, and the kernel logged correctable PCIe errors and an amdgpu SVM
workqueue warning. These are diagnostic leads, not proven causes. Full
transcripts and the result summary remain in the ignored attempt directory.


### Indexer physical page strides and next smoke

A GPU fixture built by the pinned vLLM `create_kv_cache_views` reproduces
attempt 12's assertion: per-layer cache views can have padding between pages
and interleaved pages from other layers. The reader now validates dense page
contents, nonoverlapping page strides and FP32-scale alignment, preserving
the physical page stride and storage offset. It never compacts/copies cache.
The standalone regression reproduces the old assertion before the fix;
validation includes padded/packed views, compression 1/2, C1/C2, 1/8 query
rows, prefill/decode top-k and candidate selection against references.
Evidence: `bench/results/deepseek-indexer-strides/`.

The user requested a 4 GiB host MemAvailable guard (previously 8 GiB) and
explicitly requested **automatic KV allocation**, without a fixed KV override.
The next bounded smoke therefore keeps utilization0.975, offload11, NBT512,
8K/C1 and disables any manual KV budget. Capacity will be taken from startup
logs. These changes invalidate timing comparisons with attempt 12; the run
qualifies correctness/memory only. Historical attempts retain their original
recorded thresholds and budgets.

Validation completed: all 12 indexer CPU/GPU tests passed. Attempt 13 then
started with automatic KV and guard4, completed the same three short probes,
and returned exactly `COBALT-7319` for the 2910-token active-indexer prompt.
Strict exact-match, semantic retrieval, output compliance and completion all
passed; runtime error: NO. The formerly failing decode path now executes.
The server remains running and its health endpoint returns HTTP 200.

vLLM reported 631,938 shared cache tokens (TP0 available KV 2.48 GiB).
The context limit remains 8192 and concurrency remains one; this does not
qualify long contexts, C2, thinking, tools or performance. Minimum sampled
host MemAvailable was approximately 4.19 GiB, close to the requested 4 GiB
guard, without intervention. This is a bounded correctness result, not a
claim of adequate memory headroom under larger workloads. The long probe
also spent time in a prefill synchronization point; stack samples, PCIe
correctable-error messages and an externally issued drop_caches event are
preserved in `bench/results/deepseek-attempt13/`. No causal performance claim
or speed comparison follows from this run. No fixed KV cap was introduced.

After attempt 13, the user requested automatic context sizing too. The profile
now passes `MAXLEN=-1` (vLLM auto-fit), avoiding the shared launcher's implicit
32768 default when MAXLEN is omitted. Per-client limits belong in the future
DeepSeek LiteLLM configuration. Shell syntax and dry-runs for auto and an
explicit diagnostic override passed. The running attempt-13 process retains
8192 until restarted; its reported cache capacity is not long-context validation.

### Vision bring-up (2026-10-05)

The profile now loads the checkpoint's BF16 vision tower and aligner, using
TP8 weight sharding and the TRITON_ATTN encoder backend. It accepts at most
four images per request, with native processor sizing (checkpoint image-token
budget 1024); image spans consume the same context budget as text. Concurrency
is now four at the user’s request. Context and KV allocation remain automatic, with offload10.5,
utilization0.975 and NBT512. These settings are a bring-up candidate, not a
long-context or concurrency qualification.

The pinned vision modules have ordinary per-parameter loaders. The streaming
adapter copies vision/aligner tensors and image delimiters immediately through
those loaders while yielding only the contiguous language-model group. The
language model still finalizes exactly once after EOF. This avoids sorting and
retaining the full checkpoint and preserves the upstream parameter/TP loading
rules. Interleaved vision/text fixtures check exact loaded values, incremental
consumption and single finalization; the CPU suite passed 17 tests with one
optional checkpoint test skipped.

The previous text server was stopped explicitly. Vision startup and bounded
OCR/color, four-image, limit-rejection, local counting and proxy-stream checks
are recorded under ignored `bench/results/deepseek-vision/`. The external host
guard retains the requested 4 GiB available-memory threshold. Live validation
is pending; startup capacity alone will not qualify that context length.

The first vision start loaded all weights, then failed in encoder profiling:
FlashAttention's CK kernel returned `device kernel image is invalid` on gfx1201.
There was no OOM or guard intervention. The profile now selects upstream
`TRITON_ATTN`. Before reloading, its BF16 attention was checked on one R9700
against chunked FP32 softmax/matmul references at 576, 2304 and 9216 patches,
2 heads/rank and head size64. All passed (maximum absolute errors .001953125,
.0009765625 and .00048828125). Attempt-one evidence is retained; the next
startup and API checks use `bench/results/deepseek-vision2/`.

The second start used Triton and C4. All eight ranks loaded; the encoder probe
passed and the packed W4A4 execution marker appeared. It then failed in the
indexer's dummy-run reservation: generic RDNA accounting allocated a
per-head logits tensor and lacked the construction-time scheduler context,
requesting 32 GiB. This was a GPU allocation failure, not a host-guard kill.

The scoped indexer now records maximum decode rows during construction and
profiles the reader's actual `(rows, compressed_max_len)` layout. C4 at 1M
requires 16 MiB for uncompressed decode logits. Gathered keys are bounded by
four compressed request histories, rather than the upstream factor40: at
ratio1/1M this is 528 MiB. The transient prefill logits allowance remains.
The upstream prefill chunk metadata and actual gather allocations use the
same bounded capacity; no PCP/DCP is allowed by this adapter.

`R9K_DEEPSEEK_PREFILL_SEQS=1` also bounds the main MLA gather workspace to one
maximum-length history (about 1 GiB at 1M), using upstream chunk planning.
Shorter requests can still share a chunk. Scheduler concurrency remains four,
NBT remains512 and decode is unchanged. No context or KV budget is hardcoded.
Attempt-three evidence and allocation tests: `bench/results/deepseek-vision3/`.

### Bounded checkpoint staging after the full-eager attempt

Vision attempt three reached the API with auto context 581760 and passed the
text and direct single-image probes, but later hit an `execute_model` RPC
timeout during mixed prefill/decode (511+1 rows, two requests, 2.6% KV usage).
It does not qualify C4, full context, or the complete vision/proxy path.

The requested full `--safetensors-load-strategy eager` retry was stopped by
the existing 4 GiB host guard at 2.1 GiB MemAvailable with no free swap.
Docker reported OOMKilled=false; the guard escalated TERM to KILL. The
checkpoint includes two 94.56 GiB Engram shards, so whole-shard allocation
per worker is not viable at the current memory budget. Evidence remains in
`bench/results/deepseek-vision4-eager/`.

The profile now uses lazy safetensors plus `R9K_DEEPSEEK_LOAD_BUFFER_MIB=64`.
A thread-local dispatch scope exists only during DeepSeek `load_weights`:
CPU-to-GPU copy/to operations stage already-sliced data through one ordinary
64 MiB host buffer per rank, split larger copies into bounded views, and wait
for each transfer before buffer reuse. CPU-to-CPU offload copies and GPU-to-GPU
operations keep their original paths. The staging buffer is released on both
success and exception; full-eager loading is rejected when staging is enabled.
This bounds additional staging storage to 512 MiB across TP8, not total loader
RAM or checkpoint page cache. Setting the buffer size to zero disables staging.
Byte-exact copy tests include mmap, non-contiguous TP slices, broadcasting,
dtype conversion and raw FP8/E8M0 codes. Results are in
`bench/results/deepseek-buffered-load/`; no loading-speed improvement or fix for
the separate mixed-batch timeout is claimed from these tests.

The buffered attempt reached API readiness on all eight ranks. Each rank
reported 65424 staged copies, 65428 tiles and a largest tile of 67102720
bytes, then released its buffer. Model loading reported 854–874 seconds;
these startup observations are not a controlled speed comparison. The host
had about 19.6 GiB MemAvailable after startup, without guard intervention.
Auto-fit selected 570112 tokens per request and reported 570237 KV tokens;
this is a shared pool, not four full-length reservations. A bounded direct
API smoke (temperature1, top_p0.95, thinking off, maximum32 output tokens)
returned HTTP200, `42` for `17 + 25`, and finish_reason=stop. Evidence:
`bench/results/deepseek-vision5-buffered/`. This validates startup and one
short text request; the earlier mixed-batch RPC timeout remains unresolved.

### Prefill indexer OOM and APC bring-up

The buffered server subsequently failed during an Anthropic request at
02:44:05 UTC: `fp8_mqa_logits_torch` could not allocate another 192 MiB on
multiple ranks. APC was off. The scoped adapter replaces the paged decode
reader, but prefill still dispatches to the upstream PyTorch reference when
AITER is disabled. In pinned `rocm_aiter_mla_sparse.py:862–864`, this creates
`[H,M,N]` score tensors, casts/scales them, applies ReLU and head weights,
then reduces H. Here H=32. The metadata splitter budgets only the final
`M*N*4` bytes, and profiling reserves the configured 512 MiB transient logits
budget. Neither accounts for the reference path's per-head intermediates.
For illustration, M=512/N=3072 needs 6 MiB of final logits but 192 MiB for
each FP32 per-head tensor; these are not measured dimensions of the failed
request. The trace establishes the allocation site, not fragmentation as
the cause.

The user requested utilization0.95 instead of0.975, retaining offload10.5,
NBT512 and C4. This gives approximately 0.8 GiB extra nominal headroom per
GPU, but is not a general fix for the intermediates' growth with history.
The proposed scoped fix is tiled prefill logits with fused head reduction,
checked against a numerical reference including masks, scaling and top-k;
the scheduler's prefill token budget need not change.

APC is now enabled in the profile using the pinned vLLM default. Its Engram
implementation supports prefix reuse; ROCm explicitly disables SWA bounded
replay and retains prefix-cacheable SWA blocks instead. The startup and
bounded cold/warm/appended-turn smoke are recorded separately in
`bench/results/deepseek-apc-smoke/`. Changing both utilization and APC means
this attempt is not a performance A/B comparison.

The subsequent fix binds AITER's existing portable Triton `fp8_mqa_logits`
directly in the DeepSeek indexer's private function namespace. It leaves
global AITER flags, GLM/Qwen dispatch and the custom paged decode reader
unchanged. Inputs are checked for H32/D128, E4M3FN Q/K, FP32 scales/weights,
and contiguous int32 interval bounds; the pinned AITER interface is checked
at installation. `clean_logits=True` preserves -inf outside each query's
valid key interval. The kernel reduces heads internally and allocates only
FP32 `[M, ceil(N/256)*256]` logits. Existing gathering, compression1/2,
candidate filtering and top-k remain upstream.

On one gfx1201 GPU, the frozen M512/H32/D128/N3072 allocation test measured
614727680 bytes (586.25 MiB) for the stock reference and 6291456 bytes (6 MiB)
for the fused path, after one warmup each. M512/N65536 allocated 128 MiB.
Four GPU tests passed for FP32-reference logits, per-query masks (including
empty and disjoint ranges), scale broadcasting, strided inputs, top-k and
memory growth. This is kernel allocation evidence, not end-to-end throughput
or long-context serving qualification. Protocol and logs:
`bench/results/deepseek-indexer-prefill/`.

The rebuilt image (`efc72ba0c9fd`) started as
`deepseek41-flash-vision7-apc-fused`. All 21 targeted tests passed: 11 CPU
adapter contracts, 4 fused-prefill GPU tests and 6 integrated indexer GPU
tests (including decode, compression1/2, packed pages and candidate top-k).
The integration test makes any PyTorch prefill-reference call fail, while
checking that the global upstream dispatchers remain unchanged.

At utilization0.95 the server auto-fit to max_model_len106624 and reported
106740 shared KV tokens. Four C1 API probes all retrieved the exact key,
complied with the output format and completed with stop, without observed
runtime errors. Cache hit/query deltas were 0/2630 (cold), 2560/2630 (warm),
2560/2649 (appended turn), and 0/2634 (independent prefix/salt control).
Evidence is in `bench/results/deepseek-apc-fused/`. This qualifies this
bounded text APC smoke only, not C4, vision, full context or decode speed.
The proxy/Claude template's earlier 524288-token declarations exceed this
run's auto-fit limit; they do not expand the backend's capacity.

### Planned runtime expert staging and hot/cold placement (2026-10-05)

Status: design only; no service restart, runtime change or performance claim.
Reference inspected at `4a4e88ec2cdcf2fc7c802e1f6b7c943d897770c8`:

- [staging.py](https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel/blob/4a4e88ec2cdcf2fc7c802e1f6b7c943d897770c8/vllm_dsv41_opt/vllm_dsv41_opt/staging.py):
  clone one UVA packed matrix before a large eager Marlin GEMM; leave small
  batches/capture alone; allocation failure retains the original UVA call.
- [placement.py](https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel/blob/4a4e88ec2cdcf2fc7c802e1f6b7c943d897770c8/vllm_dsv41_opt/vllm_dsv41_opt/placement.py):
  static calibrated page placement across layers within the offload budget;
  combine corpora conservatively so offloaded experts are cold across domains.
- [partial_residency.cpp](https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel/blob/4a4e88ec2cdcf2fc7c802e1f6b7c943d897770c8/vllm_dsv41_opt/vllm_dsv41_opt/csrc/partial_residency.cpp):
  CUDA VMM maps host/device allocations into one unchanged tensor address range.

User revision: follow the reference's EP8 and whole-projection staging;
the proposed 128 MiB expert-group staging is superseded, not the implementation
target. Port the memory-access strategy, not Marlin or its W4A16 arithmetic.
Retain our Quark W4A4, original E8M0 scales and activation QDQ/clamp10. Target
TP8 for non-expert layers plus EP8 for routed experts, eager execution, NBT512
and C4. This is runtime prefill staging, independent
of the existing 64 MiB/rank checkpoint-load buffer, which is freed after load.
It is not an optimization of model startup or Engram host lookup.

1. **Establish the bottleneck and memory budget.** Freeze the protocol before
   measurement; capture bounded MoE/Engram/indexer/communication timings and
   host transfers. Profiling runs are not throughput results. The saved profile
   asks for utilization0.97/offload11 GiB; the last observed running server used
   0.95/10.5. Record actual runtime settings and rerun both variants if changed.
   Do not attribute a gain from switching these settings to staging.

2. **Enable EP8 through the packed adapter first.** The pinned vLLM already
   has EP configuration and expert-map-aware Triton alignment/reduction. Our
   adapter currently requires `expert_map is None` and TP-sharded shapes, so
   adding the flag alone fails its geometry gate. Support 48 complete local
   experts instead of 384 TP fragments: expected packed w1 `[48,4608,2560]`,
   w2 `[48,5120,1152]`, scales `[48,4608,160]` / `[48,5120,72]`.
   Verify these against actual loader/config output before accepting them.
   Map global routing IDs to local experts; mask nonlocal routes in GEMM and
   final reduction so uninitialized slots can never contribute. Validate a
   rank with no selected local experts. Inspect HIP entry-point support for
   negative/nonlocal IDs before reuse; put any necessary extension behind
   DeepSeek-specific entry points. Retain vLLM's distributed combination and
   shared-expert semantics, auditing offload and all eight ranks' ownership.
   Test EP numerical outputs including cross-rank reduction against the TP
   baseline (do not require bitwise equality across different reductions).
   After isolated tests, enable `--enable-expert-parallel` in the existing
   profile and qualify startup/C1/C4 with staging off. Do not silently fall
   back to full-weight emulation. Keep TP8 as the rollback option.

3. **Stage whole projections as in the reference.** Add a DeepSeek-only module,
   proposed `r9700_vllm/moe/deepseek_staging.py`, around the two GEMMs in
   `deepseek_w4a4.py`. EP8 has 540 MiB packed gate/up weights and 270 MiB down
   weights per rank, just like the reference geometry. Scales add 33.75 and
   16.875 MiB respectively; stage them only if host-backed (the reference
   wrapper clones weights only). Copy a complete local packed projection,
   run GEMM, release/reuse storage before the next projection/layer. Preserve
   bytes, layout, scales and arithmetic. Account for the actual peak, including
   up to 573.75 MiB staging per GPU plus other workspace, before KV sizing;
   do not assume utilization0.97 leaves that amount free after KV allocation.
   Start with the reference threshold of 256 query tokens, bypass small batches
   and graph capture, and log activation/fallback per rank. Validate stream
   ownership/lifetime. If the staging allocation fails before GEMM, retain the
   original packed-UVA call as the reference does; repeated bypass means this
   configuration has not qualified staging. No expert-group tiling, extra
   quantization, persistent full host duplicate or change to NBT in this stage.

4. **Calibrate routing for hot/cold placement.** Add scoped GPU counters for
   per-layer/expert selections and calls-hit, with snapshots outside the hot
   path. Use scheduler metadata to distinguish prefill/decode, including mixed
   batches. The reference's `<512 rows` decode heuristic is wrong for our
   NBT512 partial prefills. Cover code, prose, reasoning and tools; validate
   on held-out prompts. Save checkpoint/config/layout identity with each
   profile and reject mismatches. Timing with counters enabled is diagnostic.

5. **Qualify HIP partial residency, then integrate placement.** First probe
   mixed host/device VMM with small owned allocations on the pinned ROCm/gfx1201:
   query support/granularity, map, run our reader, check bytes and lifetime,
   and exercise cleanup after partial failure. HIP documents host/device
   `hipMemCreate`, but documentation is not proof that our installed runtime
   supports this combination: [HIP VMM reference](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___virtual.html).
   If qualified, retain tensor layout and use the reference's page-ranking
   approach with measured HIP granularity. Account for actual offloaded bytes,
   scale storage, page rounding and temporary migration peaks; migrate bounded
   pieces without a second full model in RAM. Verify complete placement on all
   ranks, mapping the global calibration profile through each rank's EP map.
   If mixed VMM is unsupported, keep EP/staging independently usable and report
   placement blocked on that capability; propose a separate design before
   substituting a different residency strategy. No assumption that CUDA VMM is
   a direct HIP port, and no online eviction in this phase.

6. **Correctness and comparable A/B gates.** First check exact copied bytes and
   scales, unchanged routing and numerical outputs against the current packed
   path for rows1/2/4/32/128/256/512. Include repeated experts, rank boundaries,
   mixed storage, scratch reuse across layers and failure cleanup. Then freeze
   C1/C2/C4 API tests with identical prompts, sampling/thinking, token budgets,
   APC preparation, warmups and repeats. First compare TP8/EP8 with staging off;
   then freeze EP8 and compare staging off/on. Compare placement off/on
   separately, then their combination. Reserve identical
   scratch/KV geometry for kernel-isolation comparisons and separately report
   production KV capacity cost. Report TTFT, effective prefill, per-request
   decode, aggregate throughput, RAM/VRAM peaks, preemption and measured PCIe
   traffic separately from estimated offloaded bytes. Check text, tools, vision
   and APC; NIAH retains separate strict/retrieval/format/completion/runtime
   fields. Keep protocols and every attempt in ignored `bench/results/`.

Deliver as scoped EP8 support plus tests, whole-projection staging plus tests,
routing calibration plus tests, and
placement plus tests/documentation. Check reference licensing/attribution before
copying code. Build through the existing DeepSeek image/profile; do not modify
shared GLM/Qwen/MiMo kernels. Retain the existing packed-UVA path for rollback.
Promote only measured improvements with correctness intact and the KV capacity
tradeoff explicitly reported; RTX5090 results are not R9700 speed targets.

### Reference parity audit: additional requirements (2026-10-05)

Read-only audit against reference commit `4a4e88ec2cdcf2fc7c802e1f6b7c943d897770c8`
(remote HEAD reconfirmed), including plugin code/tests, the full inherited
preset chain, launcher, vLLM patch script, fault reports and benchmark methods.
Files absent from the sparse reference checkout were read through `git show`.
Compared with our profile/adapters and installed vLLM `18f8f960`. This audit
extends the preceding plan: eager/NBT512/no speculation are initial test
conditions, not the final reference-equivalent performance configuration.

| Reference feature | Our status and required action |
| --- | --- |
| TP8 + EP8 | Missing in the packed adapter; implement as above. Our original TP intermediate288 is already unpadded, so the reference's Marlin padding saving (288→384) is not a promised R9700 saving. |
| Whole-projection staging, query threshold256 | Planned above; account for staging during startup memory profiling and distinguish enabled/actually used/allocation fallback. |
| Offload only `w13_weight` / `w2_weight`; scales stay resident | Our shared launcher selects the broader `experts` name segment, including eligible scale parameters. Add a DeepSeek-specific exact matrix selector using actual Quark parameter names; keep scales, routers, shared experts, attention and heads resident. Report actual bytes, since whole-tensor granularity can overshoot requested GiB. |
| Initial layer eligibility20–39, then calibrated placement across40 layers | Our offloader has no layer filter. Reproduce their initial placement as a separate control; once hot/cold is active, do not restrict its planner to20–39. All backbone layers still run in prefill; their older contrary comments are superseded by traces. |
| Single-copy exact host allocation, Engram CPU | Already implemented in our scoped allocator and CPU aliases; verify this remains true after EP loading/placement. Do not add another full host model copy. |
| GPU graphs and bounded capture ladder | Missing: both our MoE initialization and indexer guard require eager. Qualify HIP capture/replay with persistent workspaces, live route IDs and stable pointers; do not merely remove the guards. Staging must bypass capture/replay. |
| Null-block / masked-index correctness | Reference fixes both dummy state writes during capture and masked FlashInfer gathers reading NaNs from block0. Our capture entry point has no identical post-capture scrub, but this is not proof of the same bug on ROCm. Audit actual compressor/indexer/SWA writers and Triton readers; add synthetic poisoned-null-block tests, padded batches and first real requests after capture before enabling graphs. Do not copy FlashInfer-specific fixes into unrelated kernels. |
| Static DSpark5, probabilistic draft, adaptive verification disabled | Currently disabled. Local checkpoint declares three next-token predictor layers and128 routed DSpark experts versus384 in the target. Audit its quantization, loader, EP ownership and expert shapes separately; the target-only adapter must not reject or incorrectly process the drafter. Verify draft/target KV rollback, positions, acceptance by position and unequal-length C1/C4 requests. |
| Prefill budget2048 | We use512 and the adapter rejects larger row counts. Extend numerical/memory tests and dispatch to1024/2048 before raising NBT. Compare512/2048 separately with fixed other settings; do not jump to8192 (their attempt also hit OOM). |
| Small graph ladder for actual verification batch | Reference fast captures through48 (8×6). Our C4 DSpark5 needs up to24 verification rows, with the exact ladder determined by vLLM padding/dispatch; measure graph memory rather than importing48/96 blindly. |
| Small fixed KV allocation for fast profile | Reference fast uses256 MiB/rank, maxlen32768, offload9.75 GiB and text-only serving. We use automatic KV/context, vision4 and a different memory budget. Keep user-requested automatic sizing; report the speed/capacity tradeoff instead of silently copying the fixed KV limit or disabling vision. |
| NUMA-local pinned buffers under8 simultaneous readers | Their topology has two sockets and no GPU P2P. Measure our actual topology/NUMA placement and concurrent host bandwidth; do not infer it from a one-GPU copy or apply strict binding without per-node RAM headroom. |
| HostAR small-message experiment | Optional unpublished dependency, not part of their shipped fast preset. Their large-message variant hurt prefill. Keep RCCL as the baseline; custom all-reduce is a separate measured experiment, not a missing prerequisite. |

The reference CUDA-specific fixes cover FlashInfer decode/prefill dispatch,
dual-cache vision support, block64 and compression-aware block negotiation,
JIT-cache shadowing and disabling FlashInfer autotuning. Our ROCm path uses
different readers, so none is an instruction to change our block size to64.
Retain ratio1/2, cache-layout, padding, vision and prefill/decode tests; establish
whether each fault applies to the installed source before adapting a fix.
The reference's text-only image-buffer optimization must never suppress actual
image visibility in our vision-enabled profile. Its CUDA graphs are distinct
from `torch.compile`; unsupported compilation does not by itself prohibit HIP
graph execution.

Implementation order after this audit:

1. Freeze runtime/checkpoint/protocol identity; audit exact offload selection
   and budget, implement EP8 and its correctness tests in eager mode.
2. Reproduce whole-projection staging with resident scales; then extend/test
   NBT2048 and compare it independently. Preserve the automatic KV calculation.
3. Qualify graph-safe execution, null-block/padding invariants and a small
   capture ladder; measure against eager before adding speculation.
4. Integrate static DSpark5 (adaptive disabled, probabilistic draft), validating
   its separate128-expert architecture and rejection/rollback paths. Compare
   no speculation/DSpark at otherwise identical settings, recording acceptance.
5. Calibrate routing in the final serving regime, reset warmup counters, then
   validate HIP partial residency and hot/cold on held-out traffic. A profile
   fitted before adding speculation is not assumed representative afterwards.
6. Final C1/C2/C4 text/tools/vision/APC and context-capacity qualification. Report
   pure decode separately from request throughput and prefill. Add finite-logit,
   repeated-token and teacher-forced log-probability regression probes to the
   existing numerical/NIAH checks; deterministic probes are correctness checks,
   not model-quality claims. Sampling/thinking quality tests remain separate.

Reference preflight checks patch/backend availability before loading the large
checkpoint. Extend our preflight/contract tests similarly for EP geometry,
staging, graphs, DSpark, profile identity and actual placement completeness.
Placement code in the reference can catch a failure after earlier layers were
already changed: require an explicit completed/partial/inactive status across
all ranks and never label a partial run a qualified placed baseline.

Relevant evidence:
[preset chain](https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel/blob/4a4e88ec2cdcf2fc7c802e1f6b7c943d897770c8/deploy/presets/v41-flash-latency.env),
[graph failure mechanism](https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel/blob/4a4e88ec2cdcf2fc7c802e1f6b7c943d897770c8/upstream/vllm/ISSUE-v41-cudagraphs.md),
[patch inventory](https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel/blob/4a4e88ec2cdcf2fc7c802e1f6b7c943d897770c8/upstream/README.md),
[staging and fixed-KV results](https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel/blob/4a4e88ec2cdcf2fc7c802e1f6b7c943d897770c8/benchmarks/results/2026-09-15-staging-and-batching.md),
[placement results and regressions](https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel/blob/4a4e88ec2cdcf2fc7c802e1f6b7c943d897770c8/benchmarks/results/2026-09-16-expert-placement-and-wide-batches.md).

The reference's headline97.23 tok/s already includes its graphs/DSpark/EP and
memory choices; it is not a pure-decode prediction for our hardware. Placement
also regressed on random-token traffic, so record adverse results alongside
code/prose gains. No service changes or new benchmarks were run for this audit.

### Implementation and qualification in progress (2026-10-05)

The EP/staging implementation now exists; the design-only status above describes
the earlier audit. No performance improvement is claimed from the following
bounded correctness smoke tests. Their protocols, every attempt, transcripts,
startup logs and RAM guard output are in ignored
`bench/results/deepseek-ep-staging/`.

- **EP8:** 48 full local experts, global routing mapped through vLLM's expert
  map, nonlocal slots masked before QDQ/reduction. GPU tests cover resident,
  UVA and mixed HIP VMM weights, idle ranks, rows1..2048, changed routes under
  graph replay, and summed partial expert outputs against the Quark reference.
  The target still uses original Quark W4A4 scales, clamp10 and activation QDQ.
- **Exact matrix offload:** the profile selects routed w13/w2 matrices only;
  scales stay resident. `R9K_DEEPSEEK_OFFLOAD_FIRST_LAYER=20` is an explicit
  reference-placement control (20..39); the default remains0 during staging
  isolation. Existing single-copy Engram/expert loading is retained.
- **Whole-projection staging:** threshold256; small batches and graph capture
  bypass staging. Reference-style temporary `clone()` worked in startup
  profiling but all eight ranks subsequently hit staging allocation fallback.
  That attempt later failed at candidate-block `torch.topk` with
  `hipErrorNoBinaryForGpu`; its causal relationship to allocation pressure has
  not been established. It is retained as a failed runtime attempt.
- **Reserved staging:** `R9K_DEEPSEEK_STAGE_BUFFER=1` copies the same packed
  projection into a persistent, reusable GPU arena. This deliberate ROCm/runtime
  adaptation makes the staging allocation coexist with attention workspaces in
  automatic KV profiling. Resident scales require540 MiB/rank for the largest
  matrix; there is no second persistent host model. Stream events protect arena
  reuse. GPU tests match the temporary-copy result exactly. The repeated C1/C4
  smoke completed without staging fallback, preemption or runtime error.
- **Graphs:** explicit `R9K_DEEPSEEK_GRAPHS=1` permits qualification of
  `FULL_DECODE_ONLY`, sizes1,2,4. Kernel tests include a poisoned unused indexer
  cache block, dynamic lengths/padding and live MoE inputs/routes. Full-model
  capture and the C1/C4 smoke passed. This does not yet qualify DSpark rollback,
  all vision cases, long context, or a speed improvement.

The six short key probes (C1 cold, C4, C1 warm) had the same result in successful
EP eager/staging/graph attempts: retrieval6/6, strict4/6, format4/6, completion6/6,
runtime errors0. The C1 prompt reproducibly returned `SINGLE_SINGLE_419` instead
of `SINGLE_419`; no attribution to EP, staging or graphs is justified. These
~1.1K-token probes are not a long-context NIAH qualification. Warm APC increased
actual hit counters; preemptions remained zero.

Observed automatic capacities with utilization0.97/offload11 GiB, C4/NBT512,
vision4 and no draft were446,076 KV tokens (445,952 context limit) with reserved
staging/eager, and229,370 KV tokens (229,248 context limit) with graphs. Graph
pools measured0.15–0.17 GiB/rank versus the startup estimate0.38 GiB. These are
startup observations, not tested full-context capacities. No manual KV limit or
utilization increase was used to reclaim graph reservations.

Hot/cold support is opt-in: `R9K_DEEPSEEK_EXPERT_PROFILE` records global routes on
rank0; reset/snapshot commands exclude warmups, and scheduler metadata plus live
SWA slot mappings distinguish phases and graph padding. Use
`bench/deepseek_expert_calibrate.py` with frozen payloads and an idle backend.
Calibration timings are invalid for performance comparisons.
`R9K_DEEPSEEK_EXPERT_PLACEMENT` accepts measured profiles matching the checkpoint
config/index hashes and execution regime. Its page planner combines normalized
worst-case hotness across corpora, validates all40 layers and EP ownership,
keeps scale tensors resident, migrates one projection at a time and aborts
startup on partial placement. HIP mixed host/device VMM transport passed small
byte/lifetime/reader and MoE numerical tests; full-model placement and held-out
traffic remain to be qualified.

DSpark support is also opt-in (`R9K_DEEPSEEK_DSPARK=1`). Its MoE contract separately
checks128 global experts/top3 and16 complete local EP experts. A draft-only
Quark name mapper preserves FP8 blocks32×32 and exclusions when checkpoint
`mtp.0..2` names meet constructor prefixes `layers.40..42`. CPU mapping tests
passed; subsequent full-draft loading is recorded below, but serving does not
yet fit the remaining VRAM. NBT2048 requests, complete DSpark execution and
comparable performance measurements are still pending runtime qualification.
Completed routing calibration and bounded placement validation are recorded below.

The NBT2048 + graphs + routing-instrumentation attempt was interrupted before
API readiness. Multi-rank stacks initially remained in the sampler logits
all-gather with nearly full VRAM; later logs showed that profiling had progressed,
so this is **not evidence of a hang or OOM**. The recorded automatic KV capacity
was43,974 tokens (context43,904), graph profiling0.36–0.37 GiB/rank. No requests
were sent. This instrumented startup is not a performance comparison.

The host has one socket and one NUMA node (CPUs0–31); GPU sysfs NUMA nodes report
-1. No two-socket placement policy from the reference was copied. A wheel build
confirmed that the HIP extension source is packaged, and CPU preflight now checks
the exact packed target/draft geometry before GPU allocation.

The complete DSpark5 eager/NBT512 load and dummy forward subsequently passed
through the draft model, including its16/128 EP ownership. Loaded memory rose
from25.96 to27.25 GiB/rank. Startup then **failed memory qualification**: available
KV was-0.60 GiB and automatic context fitting refused to serve even one token.
This was with C4, vision4, utilization0.97, offload11 GiB and reserved staging;
no API requests were possible. Static5/probabilistic/adaptive-off settings are
now checked at construction, separately from the CPU checkpoint geometry audit.

A subsequent diagnostic retains dense MXFP8 weights instead of load-time BF16
expansion (`VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD=0`). This uses the same upstream
BF16 dequantization/GEMM per call; it is not a new native MXFP8 kernel. GPU tests
compared both storage policies for640×5120 and5120×15360 matrices and rows1,4,20,
24,512: outputs matched exactly and were finite. The memory/performance tradeoff
remains subject to full-model qualification; the profile default is unchanged.
Routing identities record this storage policy and staging settings so calibration
from a different execution regime is not silently reused.

The retained-MXFP8 DSpark attempt loaded26.11 GiB/rank, then was interrupted
before API readiness after prolonged device synchronization. All eight sampled
Python stacks were in `torch.cuda.synchronize` at the start of Inductor's
benchmarking wrapper, called by the draft embedding's `get_masked_input_and_mask`.
Physical VRAM was nearly full. No explicit OOM was emitted and the cause remains
unresolved; this is not evidence that disabling compilation fixes it. No requests
or acceptance measurements were possible. The default remains the qualified
bounded EP8/reserved-staging eager configuration without DSpark.

The host guard also exposed an independent failure: a five-second Docker inspect
timeout terminated it under load. Host headroom is now checked before inspection,
without potentially blocking GPU sysfs reads; inspect timeouts are logged and
retried, and shutdown tolerates timeouts while escalating TERM to KILL. Two CPU
tests cover timeout recovery and stopping by immutable container ID. This does
not introduce a Docker RAM limit or change the4 GiB host-headroom threshold.

#### Reproducing routing calibration

Use the same profile settings for calibration and placement; changing eager,
batch size, speculation or dense storage policy invalidates the profile identity.
Start with `R9K_DEEPSEEK_EXPERT_PROFILE` pointing to a file under the mounted
`/opt/r9700/bench/results/` directory. On the host, run the bounded calibration
client once per frozen corpus:

```bash
python3 bench/deepseek_expert_calibrate.py \
  --snapshot bench/results/routing-live.json \
  --prompts bench/results/calibration-code.json \
  --output bench/results/routes-code --concurrency 4
```

`--snapshot` must refer to the same mounted file as the server setting. The input
is a JSON list of OpenAI chat request bodies with explicit sampling/thinking and
at most256 output tokens each. Each invocation requires an idle backend, resets
startup counters, saves every response and writes `routing.json`. A successful
HTTP response is sufficient for routing collection; truncated generation is not
a completed-answer quality pass. Keep held-out correctness prompts separate.

For placement, restart with the recorder disabled and
`R9K_DEEPSEEK_EXPERT_PLACEMENT` set to comma-separated **container paths** to the
completed `routing.json` files. The planner requires all40 backbone layers and
valid decode observations. Every rank must log `placement COMPLETE` for80
projections; a partial migration aborts startup. Generated routing plans and
transcripts remain ignored in `bench/results/`, never committed as baselines.

The first live reset exposed an inference-mode boundary in the recorder's
background thread. Reset/snapshot now enters `torch.inference_mode()` while
holding the execution lock; a regression test exercises resetting real inference
tensors. The failed reset attempt is preserved alongside the corrected run.

The corrected live calibration completed12/12 HTTP requests across code,
prose/reasoning and tools. Every corpus covered40/40 layers with128–129 decode
calls per layer. HTTP completion here includes length-limited generations and
does not establish model quality. The combined profile passed identity and
ownership validation. Whole-tensor initial allocation consumed11.07421875 GiB
per rank for the requested11 GiB; the page planner preserves that actual budget.

Held-out probes **before** hot/cold placement passed six independent key checks
(C4 then C2), arithmetic without thinking (`51`), and image OCR (`LIME 2749`).
Teacher-forced prompt log-probabilities were finite. However, thinking repeated
the question until the256-token cap without a final answer, and tool generation
produced repeated malformed markup with no parsed call, also reaching256 tokens.
The tool-result turn was therefore not run. Neither result is a quality pass;
their cause is not established and they predate calibrated placement. The
profile is not qualified for agent workloads. Runtime errors and preemptions
were zero in these bounded probes. The current CPU suite reports72 passed,
3 skipped (plus5 passed subtests); GPU kernel checks are recorded separately.

The first full-model placement attempt was stopped by the host guard before
readiness: MemAvailable fell below4 GiB, with no completed-placement report.
Inspection found that sorting all VRAM-freeing jobs first temporarily added new
host pages before releasing old host-backed matrices. Migrating one projection
at a time alone did not bound the cumulative host allocation. The corrected
scheduler alternates RAM/VRAM changes, checks equal actual host budgets, and
keeps cumulative displacement within one projection (plus one transient copy).
A regression test covers the adversarial full-budget duplication order; the
placement/guard CPU tests report8 passed. The failed attempt and guard records
are retained; the4 GiB threshold is unchanged.

Balanced ordering alone was insufficient: the second attempt reached roughly
40–50/80 projections per rank before the host guard stopped it. The additional
retention was reproduced independently: vLLM's reload metadata copies parameter
attributes into meta tensors, including the real `_r9700_host_view` CPU alias.
Those snapshots kept old pinned allocations alive after `.data` replacement.
A512 MiB single-GPU reproduction released the full512 MiB RSS only after removing
the copied alias. Placement now removes these private host aliases/markers from
reload metadata before migrating; live parameters keep their storage until their
individual replacement. Layerwise in-place reload is explicitly rejected for a
placed model: checkpoint changes require a server restart. The CPU regression
uses actual vLLM reload metadata and checks both lifetime release and rejection.

The third full-model attempt completed **80/80 projections on all eight ranks**.
Balanced migration plus removal of retained metadata aliases kept at least
18.5 GiB of host MemAvailable through the recorded held-out checks
(approximately19 GiB during migration/profile startup).
The actual expert offload budget remains11.07421875 GiB/rank; utilization0.97,
C4, vision4, NBT512, eager execution and reserved540 MiB staging are unchanged.
Automatic sizing reported446,076 KV tokens and max model length445,952. This is
an allocation result, not validation of a request at that context length. The
Torch-only model-loading counter excludes externally allocated HIP VMM storage
and must not be interpreted as the complete weight footprint.

After placement, the same frozen held-out requests produced:

- C4 then C2 key checks: strict6/6, retrieval6/6, format6/6, completion6/6.
- Arithmetic without thinking and image OCR: both passed.
- Prescribed-prompt numerical probe: identical token and top-token
  log-probabilities before/after placement; maximum absolute difference0.
  This small probe does not establish general model quality.
- Thinking: repeated the question until256 tokens with no final answer, exactly
  as before placement. Tools: malformed repetitive markup, no parsed call,
  reaching256 tokens; the same failure class occurred before placement, although
  generated text differed. The tool-result turn was not run.
- Runtime error lines0, preemptions0, APC hit counter2,176 tokens.

The backend is left running with calibrated placement, recorder disabled, and
without DSpark or graphs. Placement remains opt-in via the measured profile paths;
the portable launcher does not depend on locally generated routing files. Full
transcripts, guard records, failed attempts and the final summary are ignored
under `bench/results/deepseek-ep-staging/`. No speedup or absence of performance
regression is claimed: these were correctness/memory checks, not matched timing
benchmarks. Agent-quality failures, DSpark startup, NBT2048 requests and long-context
qualification remain open. No cause is assigned to EP8, placement or the kernels
from these observations alone.
