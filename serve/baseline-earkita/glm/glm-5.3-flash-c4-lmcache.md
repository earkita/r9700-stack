# GLM C4 + LMCache

[Profile](glm-5.3-flash-c4-lmcache.sh) · [Shared configuration](README.md) · [Testing](testing.md)

## Recorded baseline — 2026-10-11

**Functional diagnostic baseline, not the complete earkita performance benchmark.**
Configuration: 8× R9700, TP8 without EP, NSEQ4, Quark MXFP4 packed W4A4, DFlash
K4, FP8 E4M3 target/draft KV, NBT2048, FULL_DECODE_ONLY, 4.125 GiB KV per GPU.
Image `r9700/vllm:glm53-lmcache055-lru`, vLLM `e97573215`, LMCache 0.5.5.
External cache: **128 GiB RAM / 512 GiB disk**, `pinned-gtt`, filesystem L2 with LRU.

All concurrency counts below are clients on this same NSEQ4 profile. Each case
ran one fixed-256-token cycle and one separate natural-EOS quality cycle with
8192 output-token budget. Sampling: temperature 1.0, top_p 0.95, top_k -1,
seed 1234, GLM reasoning effort **low**. Timed output is forced with ignore_eos;
it is not a completed-answer quality test. GPU APC was reset before warm-tier
checks, and L1 cleared before filesystem L2 restore. This is one diagnostic
observation per cell, not a three-repeat average or a comparison to older runs.

### Decode per request (tok/s, TTFT excluded)

Ranges span the sessions in one batch. These are not aggregate throughput.

| Input tokens | Clients | Warm RAM | Warm filesystem L2 |
|---|---:|---:|---:|
| 32768 | 1 | 48.88 | 44.92 |
| 32768 | 2 | 28.33–31.97 | 16.85–17.22 |
| 32768 | 4 | 17.38–26.92 | 19.62–28.44 |
| 131072 | 1 | 29.31 | 43.58 |
| 131072 | 2 | 29.47–35.68 | 23.98–29.82 |
| 131072 | 4 | 17.64–25.08 | 20.78–24.92 |

### TTFT (seconds per request)

First token can be reasoning. Cold TTFT includes prefill and scheduling;
warm TTFT includes cache retrieval and the remaining uncached tail.

| Input tokens | Clients | Cold | Warm RAM | Warm filesystem L2 |
|---|---:|---:|---:|---:|
| 32768 | 1 | 33.69 | 0.71 | 0.83 |
| 32768 | 2 | 28.59–49.83 | 0.83–1.43 | 0.85–1.48 |
| 32768 | 4 | 35.60–106.02 | 0.94–1.98 | 1.05–2.09 |
| 131072 | 1 | 78.54 | 1.51 | 1.76 |
| 131072 | 2 | 97.15–191.71 | 1.73–2.55 | 1.81–3.41 |
| 131072 | 4 | 80.58–323.46 | 1.87–5.50 | 2.57–6.34 |

C1/128K cold effective prefill was about **1670 tok/s** (`131072 / TTFT`),
including client-visible overhead. C1/32K included a 30-second BPF trace and is
not an uninstrumented performance baseline. Concurrent cold prefill also delays
ongoing decode; the C4/128K cold batch had no common decode overlap and is not
qualified as simultaneous C4 decode. Warm C2/C4 batches passed that check.
Filesystem L2 may hit the OS page cache: these are not physical-NVMe speed claims.
RAM/disk decode differences are observations, not proof of a storage effect.

### Aggregate output and speculative acceptance

Aggregate output is total generated tokens / batch wall time, **including TTFT**.
Acceptance is accepted speculative tokens / proposed speculative tokens; it is
separate from decode speed. Columns pair RAM / filesystem L2.

| Input tokens | Clients | Aggregate output (tok/s), RAM / L2 | Acceptance (%), RAM / L2 |
|---|---:|---:|---:|
| 32768 | 1 | 43.16 / 39.34 | 36.3 / 31.8 |
| 32768 | 2 | 52.07 / 31.44 | 34.4 / 33.5 |
| 32768 | 4 | 65.58 / 67.89 | 32.5 / 33.9 |
| 131072 | 1 | 25.08 / 33.62 | 16.7 / 29.9 |
| 131072 | 2 | 49.30 / 41.14 | 38.7 / 27.8 |
| 131072 | 4 | 58.30 / 58.93 | 34.3 / 37.6 |

### Correctness and evidence

Across C1/C2/C4 × 32K/128K × cold/RAM/L2: **84 qualified functional requests**,
including 42 natural-EOS quality responses:

| Dimension | Result |
|---|---:|
| Strict exact-match | 42/42 |
| Semantic needle retrieval | 42/42 |
| Output-format compliance | 42/42 |
| Generation completed | 42/42 |
| Backend/runtime errors | 0 |
| Foreign keys / preemptions | 0 / 0 |

The preceding two C2/128K cold requests completed but failed the original
concurrent-decode measurement gate; they remain in the original attempt.
The follow-up explicitly permits diagnostic-only serial cold decode. No failed
attempt was rewritten into a pass. Full low/high/max repeated timing and
full-capacity disk eviction remain unqualified. PCIe checks covered all eight
GPUs and shared uplinks before/during each run. Power caps were unchanged but
not uniformly 225 W; these timings are not comparable with historical 225 W runs.

Raw protocols, requests, acceptance-by-position, timings and all failed attempts
remain in ignored `bench/results/`, not as generated tracked baseline dumps:

- `bench/results/earkita-glm-gtt128-32k-85485c31/`: C1/C2/C4 32K and C1 128K.
- `bench/results/earkita-glm-gtt128-c24-a2c39a4a/`: C2/C4 128K continuation,
  direct/proxy deployment checks and cleanup record.

## Configuration, diagnosis and validation history

**Initial qualification (2026-10-10): bounded functional checks passed; opt-in only.**
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
| `KV_CACHE_RAM_GB` | `16` | LMCache L1 budget in GiB, total, plus process/transfer overhead |
| `KV_CACHE_ALLOCATOR` | `pinned-gtt` | Native HIP host allocation without pageable USERPTR; `lazy` and `pinned` are diagnostic alternatives |
| `KV_CACHE_DISK_GB` | `0` | L2 accounting budget in GiB per namespace; 0 disables disk eviction |
| `KV_CACHE_DISK_WATERMARK` | `0.75` | Start LRU eviction at this fraction of the disk budget |
| `KV_CACHE_DISK_EVICTION_RATIO` | `0.2` | Fraction of tracked objects selected per eviction pass |
| `KV_CACHE_CHUNK` | `1152` | Token chunk span from the current GLM startup block-size log; must match runtime geometry |
| `KV_CACHE_PORT` | `5555` | Local ZMQ port; set the same value in service and model launchers |
| `KV_CACHE_METRICS_PORT` | `9091` | LMCache metrics port |
| `KV_CACHE_IO_WORKERS` | `4` | Native filesystem I/O workers |

The service uses separate object groups, RAM LRU, and the `fs_native` disk
adapter. Disk I/O initially uses the OS page cache (`use_odirect=false`), so
measurements must not label every external-cache hit a physical NVMe read.
RAM LRU is independent of disk LRU. Setting `KV_CACHE_DISK_GB` enables native L2
LRU only within the selected namespace. This asynchronous budget is not a hard
filesystem quota. A fresh empty namespace is required because the pinned adapter
does not inventory pre-existing files after a daemon restart. The default of 0
preserves existing behavior (no automatic disk deletion).

The LMCache image build applies
`docker/patches/lmcache-0.5.5-fs-object-groups.patch`: upstream 0.5.5's native FS
parser rejected the five-field keys emitted for salted object groups. Existing
three/four-field filenames remain unchanged. CPU integration tests cover salted
group/rank isolation, reopening, readback and deletion
(`tests/test_lmcache_native_fs.py`). Rebuild the image to include this fix;
changing the mounted plugin alone does not update the native extension.
The build also patches native deletion to protect locked keys and pending
lookups/stores from eviction. CPU tests exercise LRU ordering and lock protection.
These CPU tests do not qualify long-context GPU cache transfers.

The 2026-10-11 full-suite attempt with 128 GiB L1 and 512 GiB L2 was interrupted
in the first C1/8K cold warmup: disk stores succeeded, but workers stalled in GPU
output synchronization before generation completed. That attempt did not qualify
disk restore or the larger matrix. The host had available RAM; amdgpu logged SVM/userptr
restore work warnings, which do not establish the cause. Note that LMCache's
"lazy" allocator pins the entire configured pool in a background thread, rather
than growing it only on demand. Evidence: ignored
`bench/results/earkita-glm-salted-5ae5e911/`.

The subsequent bounded diagnostic on the same patched image with **16 GiB L1**
and 512 GiB L2 passed all six C1/8K/low requests: one fixed-output cycle and
one natural-EOS quality cycle, each cold → RAM → disk. RAM and filesystem L2
each restored 8064 tokens after the required GPU/L1 resets, with no GPU APC
hits or preemptions. NIAH strict, retrieval, format and completion were all
3/3; no request runtime error was observed. PCIe checks passed for all eight
GPUs, including their shared upstream links. Filesystem L2 may still use the
OS page cache; this is not a physical-NVMe latency measurement. Evidence:
ignored `bench/results/earkita-glm-lru16-dbd84cb6/`.
This establishes bounded functionality of the patched image at 16 GiB, not
full-matrix performance, 128 GiB stability or a proven cause of the earlier
stall. Production settings were not promoted to the candidate image at that stage.

The same 16 GiB configuration subsequently passed bounded **C2 and C4 / 8K /
low** diagnostics: all 12 cold/RAM/disk batches (36 requests), including
simultaneous decode of all clients. Quality strict, retrieval, format and
completion were 6/6 for C2 and 12/12 for C4; no foreign keys, request runtime
errors or preemptions were observed. Each warm tier restored 8064 tokens per
request (16128 for C2, 32256 for C4), with GPU APC excluded. PCIe checks passed
for all eight GPUs. Evidence: ignored
`bench/results/earkita-glm-lru16-c24-040d8b4d/`. These remain functional smoke
results, not full-matrix performance or 128 GiB qualification.

The requested follow-up at **128 GiB L1 / 512 GiB L2**, C1/C2/C4 × 32K/128K
with the same low-effort smoke protocol, failed on its first C1/32K cold
request. All eight worker MainThreads were waiting in local GPU event
`synchronize → export_event → submit_store`, before any cache store was
submitted. BetterBench received no output chunks for 300 seconds; no TTFT
or decode result exists. The subsequent `idle()` "server busy" error refers
to that still-active request, not evidence of another client. The remaining
35 batches were not run. Original production with 16 GiB L1 was restored
and its health endpoint returned HTTP 200; LiteLLM remained running.
At least 112.5 GiB host RAM remained available; no request OOM was observed.
PCIe topology/width/speed checks passed, but the kernel recorded correctable
BadTLP AER events on upstream bridges; their contribution is unknown.
Both context length and L1 size changed from the passing 16 GiB/8K tests,
so this does not isolate the cause to RAM size. Evidence: ignored
`bench/results/earkita-glm-lru128-long-d6beab46/`, including all worker stacks.

A controlled **C1/32K, 16 GiB L1** follow-up on the same patched image passed
all six cold/RAM/disk requests, including three completed exact NIAH answers.
Both warm tiers restored 32256 tokens per request. Evidence: ignored
`bench/results/earkita-glm-diagnose16-32k-e2dbe93d/`. This rules out prompt
length alone as a sufficient explanation for the observed 128 GiB failure.

The next 128 GiB diagnostic used native HIP host allocation instead of lazy
registration. Its first three cache probes completed, but cold TTFT was 79 s,
with a 33 s inter-update stall; the quality cycle was stopped as incomplete.
A 30-second BPF trace observed **5516** `amdgpu_amdkfd_evict_userptr` calls from
`kcompactd0`, while cache/worker threads waited for GPU completion. Native HIP
allocation alone therefore does not resolve the stalls. Evidence: ignored
`bench/results/earkita-glm-pinned128-32k-fecc9b8c/`. Timing collected during
diagnostic tracing is not a performance baseline.

The cache launcher exposes `KV_CACHE_ALLOCATOR=lazy|pinned|pinned-gtt`.
`pinned-gtt` combines native HIP host allocation (`--no-l1-use-lazy`, no SHM)
with `HSA_USERPTR_FOR_PAGED_MEM=0` in the cache container only. Explicit lazy
host registration would not be redirected by that ROCr setting. This targets
the observed USERPTR eviction path without changing system VM policy, model
kernels or completion ordering; runtime qualification is recorded separately.
The allocation must fit host RAM and the driver's GTT limits. The compound
launcher allows `KV_CACHE_START_TIMEOUT` seconds (default 600) for allocating
the cache pool before starting the model.

The GTT-backed **128 GiB L1 / 512 GiB L2** follow-up passed C1/C2/C4 at 32K
and C1 at 128K: 24 cache/quality batches, 48 requests, NIAH strict/retrieval/
format/completion 24/24. The next two C2/128K cold requests both completed,
with full cache stores and no preemption, but had zero simultaneous decode
overlap: the first 256-token output ended before the second prompt finished
prefill. The strict benchmark correctly stopped at that measurement gate;
this is not a backend/cache failure. Evidence: ignored
`bench/results/earkita-glm-gtt128-32k-85485c31/`. Earlier failed attempts remain
unchanged. These low-effort functional diagnostics do not qualify the complete
low/high/max performance baseline.

The continuation at C2/C4 × 128K passed all 12 batches / 36 requests, including
18/18 completed NIAH answers in each of strict, retrieval and format dimensions.
Together the two GTT runs cover C1/C2/C4 × 32K/128K × cold/RAM/filesystem L2:
84 qualified functional requests, including 42/42 completed quality answers,
with no observed runtime errors, preemption or foreign keys. Raw evidence is in
ignored `bench/results/earkita-glm-gtt128-c24-a2c39a4a/`. The earlier two completed
but measurement-invalid C2 cold requests remain in their original attempt.

Warm C2/C4 decode overlapped and passed the strict concurrency check. C4/128K
cold did not: later prefills outlasted earlier short outputs. Its cache/answer
checks pass, but `decode_concurrency_qualified=false` explicitly excludes it
from simultaneous-decode performance claims. The continuation froze
`allow_serial_cold_prefill_diagnostic=true`; the standard benchmark still rejects
that condition by default. These are low-effort functional checks, not the full
three-repetition low/high/max baseline. Filesystem L2 reads can use the OS page
cache; they do not establish physical NVMe throughput. A 512 GiB L2 budget was
configured, but this run did not validate eviction at a completely full budget.

The deployed 2026-10-11 configuration now uses **128 GiB RAM / 512 GiB disk**,
allocator `pinned-gtt`, image `r9700/vllm:glm53-lmcache055-lru`, model container
`glm53-flash-c4-nvme-gtt128` (8080), and cache `glm53-kv-cache-gtt128` (5555/9091).
The existing LiteLLM on 4000 remains running. Both direct and proxy checks
returned `CACHE_READY` with normal `stop`; health returned 200. The initial
proxy check exhausted its 128-token thinking budget and triggered rollback;
that incomplete response is preserved. A separately recorded retry with an
8192-token budget completed on both paths. Evidence:
`bench/results/earkita-glm-gtt128-c24-a2c39a4a/production-retry8192/`.
Old production containers remain stopped for rollback. Only four completed
diagnostic cache namespaces were deleted (~476 GiB); all benchmark transcripts
and reports remain in `bench/results/`.

To reproduce these sizes on free GPUs, use a **fresh** namespace and unused
container names (this is a new launch, not a command to replace the live server):

```bash
bash serve/baseline-earkita/glm/glm-5.3-flash-c4-lmcache.sh \
  IMG=r9700/vllm:glm53-lmcache055-lru \
  NAME=glm53-flash-c4-nvme-new KV_CACHE_NAME=glm53-kv-cache-new \
  KV_CACHE_NAMESPACE=glm53-gtt128-new \
  KV_CACHE_RAM_GB=128 KV_CACHE_DISK_GB=512 KV_CACHE_ALLOCATOR=pinned-gtt
```

The reusable launcher still defaults to 16 GiB RAM and uncapped disk when sizes
are omitted; the deployed 128/512 budgets above are explicit launch parameters.

Experimental commands for the tested configuration,
on idle GPUs and with the existing model stopped:

```bash
# Build once if the candidate image is not already available.
docker build -f docker/Dockerfile.glm-lmcache -t r9700/vllm:glm53-lmcache055 .

# One command starts cache, waits for it, then launches the model.
bash serve/baseline-earkita/glm/glm-5.3-flash-c4-lmcache.sh KV_CACHE_DIR=/mnt/ai/cache/ KV_CACHE_RAM_GB=16
# Preview both commands without starting anything:
bash serve/baseline-earkita/glm/glm-5.3-flash-c4-lmcache.sh DRYRUN=1
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
the matched protocol in `testing.md`; this functional smoke does not
establish absence of a performance regression.

References: [LMCache hybrid groups](https://docs.lmcache.ai/mp/hybrid_models.html),
[native filesystem adapter](https://docs.lmcache.ai/mp/l2_storage/fs_native.html),
[GLM recipe](https://docs.lmcache.ai/recipes/glm5_3.html). The GLM recipe explicitly
does not validate the Flash variant or our DFlash combination.

## Upstream v0.3.2 review (2026-10-11)

Merged r9700-stack upstream through `02bab38`; this does not upgrade the
GLM image's vLLM `e97573215` or LMCache 0.5.5. GPU performance and cache
roundtrip results above predate this merge and do not qualify the new tree.

- **Communication:** upstream enables its small-message N-rank all-gather by
  default. C2/C4 profiles explicitly keep `R9K_AG=0` (RCCL all-gather),
  preserving the prior baseline. `R9K_AG=1` is the first bounded experiment:
  rebuild the library in an isolated image, pass numerical TP8 eager/graph
  communication checks, then compare C1/C2/C4 under the frozen protocol.
  All-reduce settings are unchanged; all-gather is a separate collective.
- **Dense FP8 decode:** the new `gemm_fp8_block_qa` fuses activation quantization
  with block-scaled GEMM. It is a candidate for GLM's 3072x4096 and 4096x1536
  projections, not an automatic Quark optimization: generic quant overrides
  are disabled in these profiles. A scoped Quark adapter must preserve scales
  and rounding and pass reference tests first. Its M<=16 bound covers C1/C2
  K4 verification (5/10 rows), not full C4 K4 (20 rows); retain a correct
  fallback or validate an extended kernel. Do not enable global quant hooks.
- **VRAM accounting:** upstream reduced Qwen compile transients and graph
  reservations. Our GLM already has an explicit 4.125 GiB KV budget and small
  decode graphs (5/10/15/20). Qwen's 0.98 utilization and reported KV gains
  cannot be transferred. `compat/memsnap.py` hooks Qwen decoder classes only;
  adapt diagnostics to GLM and measure all ranks before enlarging its pool.
- **Not applicable directly:** Qwen GDN single-page state, hyper-connection FP8
  copies and their BF16 release, and DFlash2 FP8 exclusion aliases. Our drafter
  weights are BF16 (FP8 KV does not mean FP8 weights). No reason to apply the
  exclusion workaround to this checkpoint or replace GLM's cache geometry.
- **Prefill:** upstream reports its exact block-scaled tiled FP8 path slower
  than stock Triton in its workloads; it is not a justified GLM default.
  The shared launcher now accepts `PROFACT=CUDA` for GPU-only profiling;
  verify profiler configuration against the pinned image before use.
- **LMCache:** upstream added no GLM LMCache fix. Keep the existing geometry,
  synchronized completion, filesystem group keys and eviction-lock patches.

Next performance step is a single-variable all-gather comparison, followed
by the scoped dense-FP8 kernel experiment. Follow `testing.md`, including
PCIe verification and cold/RAM/disk cache preparation. No new throughput or
memory-capacity claim follows from this source review.

Merge validation: host profile/benchmark/proxy selection 70 passed, 6 skipped
(13 subtests); final profile/launcher checks 30 passed (3 subtests).
Pinned-image CPU GLM DFlash/APC/indexer/LMCache checks: 68 passed. The initial
selection also included two GPU Mamba cases, which failed because the isolated
container intentionally had no GPUs; the CPU-only rerun deselected those two.
The merged HIP library builds with the pinned image for gfx1201, passes the
build's export/hostcall checks and loads with both GLM W4A4 and new candidate
symbols. These checks do not replace GPU numerical or serving benchmarks.
The production library was not replaced and services were not restarted.
