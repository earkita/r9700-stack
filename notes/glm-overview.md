# GLM: current profiles and validation

Maintained configuration reference, updated 2026-10-03. Historical measurements
record their own settings; check the running API for live service status.

## Profiles

These four GLM launchers are maintained. Each invokes the shared
[`serve.sh`](../serve/serve.sh) directly.

| Profile | Role | Active sequences | Context limit | Speculation |
|---|---|---:|---|---|
| [glm-5.3-flash-c2.sh](../serve/baseline-earkita/glm-5.3-flash-c2.sh) | Two concurrent requests | 2 | auto | DFlash K4 |
| [glm-5.3-flash-c4.sh](../serve/baseline-earkita/glm-5.3-flash-c4.sh) | Four concurrent requests | 4 | auto | DFlash K4 |
| [glm-5.3-flash-c4-lmcache.sh](../serve/baseline-earkita/glm-5.3-flash-c4-lmcache.sh) | Experimental C4 with RAM/NVMe prefix cache | 4 | auto | DFlash K4 |
| [glm-5.3-flash.sh](../serve/baseline-earkita/glm-5.3-flash.sh) | Conservative fallback | 1 | 65536 | Off |

C2/C4 share these defaults:

- Eight R9700 GPUs, TP8 target and drafter. C2/C4 enable text and images;
  the conservative fallback remains text-only.
- Image `r9700/vllm:glm53-plugin-e97573215`, pinned vLLM `e97573215`.
- Quark MXFP4 target with packed W4A4, up to 32 verification rows.
- DFlash2 BF16 **weights**, K4; target and draft **KV** both FP8 E4M3.
- Shared KV pool: 4.125 GiB per GPU, APC enabled, corrected APC boundary.
- NBT2048 for vision (the earlier text-only measurements used NBT1024).
  This also covers the encoder cache's full 2048-token image limit; the
  default square profiling canvas reports only 2025 tokens.
  FULL_DECODE_ONLY graphs: 5/10 for C2, 5/10/15/20 for C4.
- Bounded indexer workspace follows NSEQ and index_kpool=4.
- No weight or KV offload configured.

The fallback retains stock model/quantization paths, eager execution and APC
off. Its default image is `r9700/vllm:dev` and model is a Hub ID; it does not
inherit the C2/C4 image or local model paths.

The separate `c4-lmcache` profile uses the C4 settings with
`KV_OFFLOAD=lmcache`, image `r9700/vllm:glm53-lmcache055`, and container name
`glm53-flash-c4-lmcache`. It starts the separate cache service first (container
`glm53-kv-cache-c4`, override with `KV_CACHE_NAME`), waits for its transfer server
to be ready, then calls `serve.sh` directly. Its offload validation covers short C1/C4
prompts; the `auto` context limit is not a long-context qualification.

### Vision qualification (2026-10-03)

C2/C4 now allow 100 images across the complete request history, with up to
2048 embedding tokens per image after resizing; video remains disabled.
Vision uses the image's existing Flash Attention Triton implementation on
RDNA4 (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, explicit encoder `FLASH_ATTN`).
The initial Torch SDPA attempt failed with HIP `invalid argument` during
encoder profiling. No kernel or checkpoint changes were needed.

All eleven live C4 checks passed: direct/proxied OpenAI, Anthropic
streaming/nonstreaming, image token counts, Claude-style `Read` tool results,
four concurrent image requests, four images in one request, 100 distinct
224x112 images, rejection of 101 images, and text afterward. The 100-image
probe retrieved the correct codes from images 1, 50 and 100. Fixtures also
cover an image with exactly 2048 embedding tokens; NBT2048 fixes the earlier
2025-token encoder-cache boundary. C2 was checked by CPU tests/dry-run;
live vision qualification used C4.

There were no preemptions. Driver-visible free VRAM was only 135–171 MB/GPU
after the 100-image smoke; this includes allocator reservations and does not
measure PyTorch-reusable memory. The checks qualify the count limit with small
images, not accuracy on all 100 images, 100 full-resolution images, C4 times
100 images, or long-context vision.

Reproduce with `bench/glm_vision_smoke.py`. All attempts, including the initial
four-image and encoder-boundary checks, remain in ignored
`bench/results/glm-vision-20261003/` and
`bench/results/glm-vision-100-20261003/`. See the
[proxy guide](../proxy/README.md) for the authenticated invocation.

## Dense block-FP8 tuning

`tuning/tune_glm_fp8_block.py` tunes launch parameters for the two GLM
dense W8A8 shapes `(N,K)=(3072,4096)` and `(4096,1536)` on gfx1201.
It preserves the checkpoint's FP8 values and block scales; it does not alter
W4A4 experts or KV precision. Compiled kernels contain the native
`v_wmma_f32_16x16x16_fp8_fp8` instruction with FP32 accumulation.

Run the tuner only on an idle GPU in the pinned GLM image. For example,
inside a container with the checkout at `/opt/r9700` and one GPU visible:

```bash
python3 /opt/r9700/tuning/tune_glm_fp8_block.py \
  --shape 3072,4096 --rows 1,5,10,15,20,64,128,256,512,1024,2048 \
  --out /opt/r9700/bench/results/glm-fp8-new-run
```

Repeat for `4096,1536` with a different output directory. Each candidate is
checked against FP32 dequantized matmul; finalists use two additional seeds
and five alternating default/candidate timing pairs. A gain below 5% retains
the default. The generated JSONs in `tuning/configs/glm/` are runtime launch
configurations, not benchmark dumps. Raw attempts stay in ignored
`bench/results/glm-fp8-tuning-20261003/`.

Opt in using `FP8_CONFIG_DIR="$PWD/tuning/configs/glm"` as an argument to a
GLM profile. The common launcher mounts each JSON read-only over the
corresponding vLLM config file, preserves other shapes, and includes config
contents in the compilation-cache key. `FP8_CONFIG_DIR=` selects the image's
defaults. These files target the pinned Python 3.12 runtime and must be
revalidated after changing vLLM, Triton, hardware or power limits.

The 2026-10-03 run passed all 1144 FP32-reference numerical checks
(maximum relative L2 0.001680), both baseline/candidate API contract gates,
13 GLM profile tests and six MiMo profile tests. Small-row isolated GEMMs
improved approximately 1.19–1.99x; both shapes retained the default at M=2048.
These are hot-weight microbenchmarks, not whole-model TFLOPS or tok/s.

Matched serving comparison: the C4 vision profile, **one active request**,
K4, 225 W/GPU, BetterBench 0.6.0, exact 8192/32768-token prompts, 256 forced
output tokens, temperature 1/top-p .95/top-k disabled, seed 1234, effort high,
one excluded warmup and three measured repetitions per context. APC stayed
enabled with unique salts and zero cached tokens in every measured request.
All runtime arguments and relevant environment variables matched; only the
two launch-configuration files changed. Medians below exclude warmups:

| Input | Default decode tok/s (range) | Tuned decode tok/s (range) | Default / tuned TTFT s | Default / tuned effective prefill tok/s | Default / tuned DFlash acceptance |
|---|---|---|---|---|---|
| 8K | 50.09 (49.09–55.59) | 50.70 (47.38–53.32) | 6.161 / 6.204 | 1329.75 / 1320.52 | 40.17% / 38.58% |
| 32K | 48.16 (48.15–54.38) | 54.51 (49.65–56.97) | 23.823 / 23.921 | 1375.49 / 1369.84 | 38.37% / 42.69% |

Acceptance is accepted/proposed tokens aggregated across the three measured
requests, separate from decode. The larger 32K decode result coincided with
greater DFlash acceptance. With overlapping sample ranges, three repetitions
and no prefill improvement, this does **not** establish a repeatable whole-model
speedup from kernel tuning. Fixed-output runs are not completed-answer quality
tests. No runtime errors or preemptions were observed; multi-request throughput
and long-context/vision quality were not requalified in this experiment.

The tuning remains opt-in; C2/C4 profile defaults are unchanged. Full protocol,
runtime identities, all attempts, acceptance by position and comparison
results are stored in the ignored evidence directory above.

## Launch

Run from the repository root. Preview without starting a service:

```bash
bash serve/baseline-earkita/glm-5.3-flash-c2.sh DRYRUN=1
bash serve/baseline-earkita/glm-5.3-flash-c4.sh DRYRUN=1
```

C2/C4 expect local target/draft checkpoints under `/mnt/ai/models/glm` by
default; override `MODELS_DIR`, `MODEL` and `DRAFT` for another installation.
The image and a compatible library in the mounted checkout must exist.
For a checkout without that library, use `BUILD_KERNELS=1` on first launch.
Build the image from the repository root with
`docker build -f docker/Dockerfile -t r9700/vllm:glm53-plugin-e97573215 .`.

On idle GPUs and a free API port:

```bash
bash serve/baseline-earkita/glm-5.3-flash-c4.sh
# Alternatively, the qualified fixed context settings:
# bash serve/baseline-earkita/glm-5.3-flash-c2.sh MAXLEN=262144
# bash serve/baseline-earkita/glm-5.3-flash-c4.sh MAXLEN=131072
```

These are alternatives, not services to run together on the same GPUs/port.
When switching, gracefully stop the verified active container with
`docker stop --signal SIGINT --timeout -1 NAME` and await completion.
The profiles use REPLACE=0 and do not stop another service automatically.

## Capacity and evidence

The current C4 vision profile (NBT2048) resolved auto to **700416 tokens for
one request**, with about **701510 group-aware KV capacity tokens**. Claude
Code's template declares **524288**. Text, image tokens, thinking and output
share that context. Earlier text-only NBT1024 starts reported 702720/703817;
those historical numbers and throughput measurements do not describe the
current vision configuration. C2's auto limit should be checked at startup.

The pool does not reserve two or four full-length contexts. Shared allocation
and recurrent/speculative state affect capacity; the full auto limit has not
been qualified by a full-length generation test.

Bounded long-context checks previously passed for C2 × 256K and C4 × 128K
with K4 at fixed limits on the text-only profile. The
[historical C4 report](glm-c4-profile.md) measures 1/2/4 concurrent 8K prompts
with NBT1024, with correctness reported separately.
Do not compare its warm-APC speed directly with earlier cold-cache results.

For all new measurements, follow [GLM testing](glm-testing.md): explicit
temperature/thinking, exact token counts, decode excluding TTFT, and separate
strict match, semantic retrieval, output compliance and completion results.

The [automated earkita baseline](../bench/baseline-earkita/README.md) fixes the
C1/C2/C4 × 8K/32K/128K × low/high/max matrix with cold, RAM and disk cache
proofs. It runs unattended on a dedicated benchmark deployment and preserves
all attempts; adding the runner does not qualify the untested full matrix.

W4A16 and FULL_AND_PIECEWISE remain experiments, not defaults. Their bounded
comparisons did not establish a consistent speed improvement.

## Experimental NVMe prefix cache

**Current status (2026-10-10): bounded functional checks passed; opt-in only.**
The scoped adapters fix registration/cache geometry and replace HIP event
handles with acknowledgements sent only after local GPU completion. Memory IPC
and native copy kernels are retained. Both endpoints must use this completion
protocol; it sacrifices transfer/compute overlap. Earlier trials caused memory
faults and left HIP devices 1/3 unresponsive. After the administrator reset PCI
e6:00.0/ec:00.0, both recovered and the corrected path passed the checks below.
The precise driver/event failure has not been established by these checks.

- Small native STORE probes passed on all eight GPUs; 14 CPU adapter tests
  passed inside the pinned image (host suite: 24 passed, 3 skipped).
- C1 cold/warm answers passed strict match, retrieval, format and completion.
  The first cold prompt produced 72 files (667,395,072 bytes).
- The model was stopped/restarted to discard GPU APC; LMCache L1 was explicitly
  cleared and its memory-usage counter verified at zero. Files remained intact.
  Repeating the 4246-token prompt restored **3456 tokens** and returned the
  correct key. `fs_native` completed a load and vLLM recorded 3456 external hits.
- Four distinct key prompts passed after restoration: strict 4/4, retrieval
  4/4, format 4/4, completion 4/4, runtime errors 0. The final isolated C4 run
  added 10,368 external hit tokens; no preemptions occurred. An earlier C4 run
  overlapped the proxy smoke and is retained separately as mixed-load evidence.
- LiteLLM OpenAI streaming and Anthropic streaming tool-call/image tool-result
  checks passed. These used short histories, not restoration of image prefixes.

Evidence and the frozen functional protocol are in ignored
`bench/results/glm-ipc-fix-20261010/post-reset/`. GLM, the cache service and
LiteLLM were left running. This is not a performance comparison, long-context
qualification, acceptance-regression test, cache-daemon restart test, or a
50-session capacity claim. The following paragraphs retain earlier failures.


Startup qualification on 2026-10-10 **failed** before serving requests. LMCache
0.5.3 calls the removed vLLM `get_kv_cache_layout()` helper; the pinned runtime
uses `CacheConfig.get_resolved_kv_cache_layout()` instead. LMCache catches the
import error and supplies no layout hint, causing cache registration to fail
with `Unsupported kv_layout: none. Only NHD and HND are supported.` The cache
service itself started, including its native filesystem adapter, but no disk
store/reload or model correctness test passed. A compatibility adapter and
full functional qualification were still required at that point. Evidence is
in ignored `bench/results/glm-kv-offload-20261010/`.

The original GLM container was restored with GPU APC; its health endpoint and
completed-answer smoke checks directly and through LiteLLM passed. These are
rollback checks, not offload qualification or a performance comparison. Offload is
opt-in (`KV_OFFLOAD=lmcache`) in the C2/C4 profiles; `0`
retains the existing launch. The shared launcher only forwards the optional
`KV_TRANSFER_CONFIG` JSON; it does not start or manage a cache service.

The subsequent 0.5.5 trial also **failed startup**, but passed the layout stage.
`docker/Dockerfile.glm-lmcache` builds source commit
`05a013b29da78cf2321b9b46ec5039dde2fb0bb0` with HIP code for gfx1201 on the
qualified GLM image. It preserves torch 2.12.0+rocm10.0.0, Triton and vLLM;
only LMCache and its build helper grpcio-tools are installed with `--no-deps`.
The extension imports, four real vLLM layout mappings and all 21 upstream
layout-discovery tests passed. At live registration, the cache server raised
`hipErrorInvalidDevicePointer` in `torch.UntypedStorage._new_shared_cuda`,
followed by a worker registration timeout. No inference or NVMe restoration
test ran. The initial cache service inherited `HSA_ENABLE_IPC_MODE_LEGACY=1`,
whereas the model used `=0`. The launcher now explicitly uses `=0` on both sides.
The initial standalone probes also failed with `=0`, but omitted host networking:
they did **not** reproduce the corrected service configuration. A 16-KiB tensor
transfer subsequently passed both within one container and between containers
sharing host network, IPC and PID namespaces. Adding ptrace/seccomp privileges
without sharing the network namespace did not fix the failing probe. The probe
interrupted by stopping its producer remains inconclusive.
Evidence and the fixed prompt corpus are in ignored
`bench/results/glm-kv-offload-055-20261010/`; corrected IPC probes are in
`bench/results/glm-ipc-fix-20261010/`. Keep host networking and the non-legacy
IPC setting aligned between the two services. A successful tensor probe alone
does not qualify model cache restoration.

The corrected IPC settings passed live TP8 registration. With the unmodified
0.5.5 connector, C1/C4 answers and proxy streaming/tools/vision passed, but no
cache objects were stored: the non-cacheable eight-token circular scratch group
prevented the minimum stored span from reaching one 1152-token chunk. After
adding the scoped geometry adapter, the first cold answer passed but the cache
process hit a GPU memory-access fault on STORE; no filesystem restore passed
in that attempt. CPU geometry tests and isolated eight-GPU transfers pass, including
padded hybrid groups, compressed indexer state and IPC offsets above 4 GiB.
These are diagnostic results, not qualification of GLM offload or a speed claim.
Evidence: `bench/results/glm-ipc-fix-20261010/geometry-adapter/` (ignored).

`serve/glm-kv-cache.sh` prepares a separate LMCache multiprocess service using
the experimental `glm53-lmcache055` image. It checks the pinned vLLM and LMCache
versions before initializing transfers. Use this same image for the candidate
model. No vLLM patch or model-weight offload is involved. The profile selects
`r9700_vllm.compat.glm_lmcache.GlmLMCacheMPConnector`, a version-guarded
adapter over the upstream multiprocess connector. It excludes non-cacheable
`KpoolTailSpec` scratch from external transfers and re-views MLA/indexer kernel
pages as logical scheduler pages without copying or changing model allocations.
The indexer retains its one-state-per-four-tokens compression.

| Launcher setting | Default | Meaning |
|---|---|---|
| `KV_CACHE_DIR` | `/mnt/ai/cache/` | Host storage, mounted at `/kv-cache` |
| `KV_CACHE_NAMESPACE` | `glm53-e975-fp8-dflash-k4-v2` | Disk subdirectory; choose a new one after changing weights, cache layout, kernels, or precision |
| `KV_CACHE_RAM_GB` | `16` | LMCache L1 budget in GiB, total, plus process/transfer overhead; lazy allocation starts at 1 GiB |
| `KV_CACHE_CHUNK` | `1152` | Token chunk span from the current GLM startup block-size log; must match runtime geometry |
| `KV_CACHE_PORT` | `5555` | Local ZMQ port; set the same value in service and model launchers |
| `KV_CACHE_METRICS_PORT` | `9091` | LMCache metrics port |
| `KV_CACHE_IO_WORKERS` | `4` | Native filesystem I/O workers |

The service uses separate object groups, RAM LRU, and the `fs_native` disk
adapter. Disk I/O initially uses the OS page cache (`use_odirect=false`), so
measurements must not label every external-cache hit a physical NVMe read.
RAM LRU is not a disk quota: this script does **not** impose a total disk-space
limit or promise retention of all sessions. Decide the disk budget/eviction
policy before a long-running rollout. All cache files stay below the selected
namespace; nothing is automatically deleted.

Experimental commands for the tested configuration,
on idle GPUs and with the existing model stopped:

```bash
# Build once if the candidate image is not already available.
docker build -f docker/Dockerfile.glm-lmcache -t r9700/vllm:glm53-lmcache055 .

# One command starts cache, waits for it, then launches the model.
bash serve/baseline-earkita/glm-5.3-flash-c4-lmcache.sh KV_CACHE_DIR=/mnt/ai/cache/ KV_CACHE_RAM_GB=16
# Preview both commands without starting anything:
bash serve/baseline-earkita/glm-5.3-flash-c4-lmcache.sh DRYRUN=1
```

The combined profile does not replace existing containers. If starting GLM's
container fails, it stops only the cache container created by that invocation.
Model loading/warmup is asynchronous after launch; check the API health endpoint
for model readiness. Existing services and stored files are not removed.
For separate lifecycle management (including C2), `serve/glm-kv-cache.sh` remains
available alongside the original profile's `KV_OFFLOAD=lmcache` switch and
candidate `IMG` setting.

Both launchers support `DRYRUN=1` without starting containers or creating cache
directories. Stopping/removing the cache service is a separate, explicit action;
it must not be stopped while a model using its connector is serving. Returning
to `KV_OFFLOAD=0` requires relaunching the model and leaves stored files intact.

Broader qualification is still required for long contexts, image-prefix
restoration and long-running cache eviction. The 0.5.3
connector required `block_size <= NBT < 2 * block_size` for Mamba `align` mode;
0.5.5 retains the lower bound. The observed block size 1152 and current
NBT2048 satisfy both checks. Do not
disable the hybrid cache manager or remove geometry checks to force startup.
On upgrades, repeat the disk-backed hit after eviction from GPU and RAM and
the C1/C4 and proxy checks. DFlash acceptance/performance comparisons require
the matched protocol in `glm-testing.md`; this functional smoke does not
establish absence of a performance regression.

References: [LMCache hybrid groups](https://docs.lmcache.ai/mp/hybrid_models.html),
[native filesystem adapter](https://docs.lmcache.ai/mp/l2_storage/fs_native.html),
[GLM recipe](https://docs.lmcache.ai/recipes/glm5_3.html). The GLM recipe explicitly
does not validate the Flash variant or our DFlash combination.
