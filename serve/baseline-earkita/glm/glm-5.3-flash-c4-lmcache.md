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
During merge validation the production library was not replaced and services
were not restarted. The later candidate experiment below temporarily stops
and restores the model.

## Opt-in upstream candidates — 2026-10-11

Both candidates remain **off by default**, pending a controlled whole-model
performance comparison. Build the library from this tree in the pinned image
before enabling them; do not replace a library mapped into a running server.
The production library is unchanged.

- `R9K_AG=1 R9K_AG_MAX_KB=16`: small-message TP8 all-gather; RCCL handles
  larger messages. Initial kernel diagnostics favor this candidate for
  1–10 KiB, but RCCL for 38–82 KiB, on this host's Gen5 x16 endpoints / x4
  root-port uplinks. The upstream 128 KiB threshold is too broad here.
- `R9K_GLM_DENSE_FP8=1`: GLM TP8 / gfx1201 / vLLM e97573215 adapter for
  Quark block-128 shapes 3072x4096 and 4096x1536. It fuses BF16 activation
  quantization with FP8 GEMM for 1–32 rows, including C4/K4's 20 rows.
  It preserves the original FP8 bytes and FP32 scales, handles ROCm's padded
  weight rows, and keeps the stock path for larger batches and incompatible
  layouts. Very short prefill batches also qualify for the fused path.
  Vision, drafters and other model families are excluded. Quark's epsilon
  remains 1e-10; the generic upstream entry retains 1e-12.

The extra packed weights increased reported model loading memory from
22.19 to 22.24 GiB/GPU in the observed starts. KV stayed at 4.125 GiB/GPU;
this change does not increase KV capacity. CPU/RAM/disk cache limits were
not changed for this experiment.

Validation:

- Dense GPU: **18/18** shape/row cases, strided activations, padded weights,
  zero/tiny inputs and changed-input graph replay. Normal cases exactly
  matched unfused HIP; maximum relative L2 against stock Triton was 0.00224,
  below the predeclared 0.01 limit. No tolerance was loosened.
- TP8 all-gather with the 16 KiB cap: exactness, repeated calls and graph
  replay **PASS**. Tests follow the runtime's cap guards and RCCL fallback.
- Adapter CPU tests: **15 passed**. Profile tests: **15 passed + 3 subtests**.
- Final C1/C2/C4 echo smoke, both candidates active: correct key **7/7**,
  strict/format **6/7**, completed **7/7**, runtime errors **0**. C1 added
  quotation marks around the correct key; C2 and C4 were strict PASS.
  Keep the C1 strict failure; it is not a retrieval or backend failure.
- An earlier candidate C2 attempt refused one key. Stock GLM also refused
  that same C2 prompt, so this does not establish a candidate regression.
  Every attempt and full response is retained.

Echo sampling: temperature 1.0, top-p 0.95, top-k -1, seed 1234, effort low,
512 output tokens, EOS active. This is an API/scheduling smoke, not NIAH or
a model-quality benchmark. Sampling outputs need not be identical.

The paired HIP-graph microbenchmark measured roughly **8–27% shorter kernel
times** for rows 1/5/10/20, using equal inputs and layouts in both paths.
Production coexisted during microtests. This is **not** a measured improvement
in model decode tok/s, TTFT, or cold/RAM/disk prefill. PCIe topology and loaded
link checks are saved with the protocols and results.

```bash
R9K_AG=1 R9K_AG_MAX_KB=16 AG_MODEL=glm BENCH=0 torchrun --standalone --nproc-per-node=8 tests/test_ag_nrank.py
python3 tests/test_glm_fp8_qa_gpu.py
python3 -m pytest -q tests/test_glm_fp8_dense.py tests/test_glm_profile.py
```

Earlier integration attempts caught incorrect runtime-prefix filtering and
ROCm padding rejection; these now have regression tests. Stopped test workers
also left IPC mappings in LMCache, blocking model startup. Unregistering only
those stopped instance IDs released GPU memory without restarting cache or
deleting prefixes. Do not work around stale IPC by lowering the memory guard.
Raw evidence: ignored `bench/results/glm-upstream-candidates-20261011/`.

### Whole-model comparison attempt — 2026-10-11

**Incomplete; neither candidate is qualified as a default.** The frozen order
was baseline → dense FP8 → dense FP8 plus all-gather capped at 16 KiB,
with C1/C2/C4, 8192 input tokens, high effort and the same checkpoint sampling.
Synthetic NIAH used 256 forced output tokens after natural-EOS preparation,
one excluded warmup and three measured batches. A separate long-generation
control kept EOS active, with two excluded warmups and five measured batches.
BetterBench 0.6.0 counted reasoning and answer tokens; decode excluded TTFT.

Only baseline C1/C2 completed: synthetic mean decode was 45.86/27.93 tok/s
per request; the EOS-active control was 50.01/32.53 tok/s. These are partial
observations, **not an A/B result**. Power caps were the observed six × 210 W
and two × 300 W, not the historical uniform 225 W.

Baseline C4 failed during natural-EOS preparation at 21:30:07 UTC with
`HSA_STATUS_ERROR_MEMORY_FAULT`, Worker TP3, PCI `0000:ec:00.0`. Both candidate
flags were zero, but this control used the candidate library with those paths
disabled, not the byte-identical production library. The cause is not isolated.
A roughly 14 GiB GPU dump subsequently exhausted the root filesystem and
telemetry failed with ENOSPC. A corrected PCIe timeout was also recorded on
root port `0000:c0:01.2`; its relationship to the GPU fault is unestablished.

The series was stopped before either candidate leg. Preserve the failed C4
attempt; do not substitute an echo smoke or infer a performance gain from
the completed C1/C2 observations. Before retrying, isolate the C4 memory fault,
provide dump space on NVMe, and rerun all variants under one frozen protocol.
Evidence: ignored `bench/results/glm-dense-ag-abc-20261011-v2/` (`status.json`,
protocol, partial results and runtime/kernel logs). The preceding v1 stopped
at power-cap preflight before any requests. The possibly incomplete GPU dump
is retained on NVMe outside Git.

The original model/container and production library were restored. Health,
direct API and LiteLLM short arithmetic responses passed with natural EOS;
these are restoration checks, not C4 requalification. The proxy's Anthropic
route rejected the probe's `seed` parameter; the supported request without
`seed` passed. No proxy setting was changed. External LMCache stayed running
and retained its prefix objects. Both optimization flags remain off by default.

### C4 LMCache qualification attempt — 2026-10-11/12

**Full qualification failed at the first warmup; no measured throughput result.**
This run used the production `libr9k.so` (SHA256
`086c04a9dc9b870d881c2656175c11f9d346d29a51fc99769304b372d3574d95`),
with both `R9K_GLM_DENSE_FP8=0` and `R9K_AG=0`.

Before the matrix, a bounded C4/8K/high control passed 12 requests: four cold
and four GPU-APC-warm natural-EOS NIAH answers, then four warm forced-output
generations of 64/128/192/256 tokens. The eight completed answers passed
strict/retrieval/format/completion, with no observed runtime error or preemption.
The forced-output cases check execution, not completed-answer quality. This
control does not validate external RAM/disk restore. The unchanged LiteLLM
contract suite passed all 12 checks (OpenAI/Anthropic, streaming, tools and local
token counting); all-GPU PCIe evidence was captured. The runner's 22 CPU unit
tests also passed.

The frozen full protocol selected C4 × 8K/32K/128K × low/high/max × cold/RAM/disk,
one excluded warmup and three measured repeats, 256 forced output tokens, and
separate natural-EOS NIAH with an 8192-token budget: 540 requests in total.
It used a dedicated 128 GiB pinned-GTT RAM / 512 GiB filesystem-LRU cache.
The explicitly declared cold-prefill diagnostic exception permits cold batches
without overlapping decode, marking those decode results unqualified; warm
RAM/disk still require C4 overlap. This is not an unchanged full C1/C2/C4
standard or an A/B comparison.

At **22:18:42 UTC on October 11** (00:18:42 local on October 12), the first
excluded warmup, **8K/low/C4/cold**, produced `HSA_STATUS_ERROR_MEMORY_FAULT`:
Worker TP5, PID 929330, PCI `0000:c6:00.0`, address zero. The kernel logged
`SQC (inst)`. Corrected PCIe `BadTLP` errors also appeared on bridge
`0000:ca:00.0`; causality is unestablished. This is an invalid GPU memory
access, not an observed allocation OOM. It occurred with the production
library, so the earlier candidate-library-only explanation is insufficient;
the failing kernel or component has not been isolated.

The external guard stopped the matrix and initiated restoration of the original
model/cache. No RAM/disk warm batch, 32K/128K, high/max matrix cell or measured
repeat was reached. Do not promote the short gate's PASS to full C4 stability.
Production cache files were retained; restarting LMCache loses the in-memory
index because 0.5.5 does not inventory old disk objects. Runtime settings and
optimization defaults were not changed.

Evidence: ignored `bench/results/glm-c4-qualification-20261011/`, including the
frozen protocol, gate transcripts, proxy results, PCIe captures, kernel/runtime
logs and partial matrix report. The controller-aborted runner's last status is
preserved separately. GPU dump (possibly incomplete after bounded shutdown):
`/mnt/ai/cache/glm-c4-qualification-dumps/gpucore.929330`. Preserve this failure
before any further isolated diagnosis; do not retry the full matrix unchanged.

Restoration completed at 22:23:54 UTC. The original GLM/cache and LiteLLM are
running; two short sequential direct/proxy arithmetic checks returned the
correct answer with natural EOS. The production library hash is unchanged,
and the restoration PCIe check passed. These checks restore service availability;
they do not qualify C4 stability or explain the GPU fault.

### Offline memory-fault diagnosis — 2026-10-12

The failing kernel remains **unidentified**. The host reports an instruction
fetch fault (`SQC (inst)`) at address zero on TP5 / `0000:c6:00.0`, not an
allocation OOM. The last asynchronously logged/JIT-compiled kernel does not
identify the faulting dispatch.

The last telemetry reports two running requests and two waiting, three draft
iterations (12 draft tokens, eight accepted), no preemption and no completed
requests. L1 held 2,447,441,920 bytes (2.28 GiB); host available RAM was about
154 GiB. GPU stores reported 88 sent / 88 completed. External hits and L2 loads
were zero: disk restore was not exercised. Concurrent GPU-to-cache stores were
active, so this does not exclude the connector/store path. KV occupancy in the
server log was 11.3%; occupancy is not a measurement of free VRAM.

The passing gate and failing matrix used the same model image, environment,
mounts and production library. Their launch arguments differ only in the
LMCache server port (5555 versus 5556). Requests, sampling, cache state and
timing differ; the gate is not an A/B control that isolates any one of them.

ROCgdb reads the GPU agents from both retained dumps but exposes no active
queues, dispatches or wave backtrace. The new dump is truncated: 1,208,010,736
bytes versus an ELF-declared end of 33,737,709,552 bytes. The earlier dump is
also truncated. These dumps cannot establish the kernel/program counter; do
not infer a null kernel argument merely from the fault address.

Static review found no upper-bound tail-block check in the local
`r9700_vllm/attn/glm_tail.py` prefill seed or decode update kernels. This is a
candidate for diagnostic validation, not evidence that an invalid block was
passed in this run. Upstream [PR #56059](https://github.com/vllm-project/vllm/pull/56059)
proposes a prefill guard for an analogous reported fault; its review also
flags decode accesses. Our pinned allocator aliases tail storage into indexer
pages across the shared physical block pool, and the metadata builder computes
`own_block * ring + position % ring`. The reported small-tail allocation scenario
therefore cannot be assumed here. Capture actual tensor shapes, strides and
slot bounds before choosing a fix; silently dropping invalid tail writes could
hide a mapping error and damage generation correctness.

The corrected `BadTLP` bridge `0000:ca:00.0` is below root port `c0:01.4`,
whereas the faulting GPU c6 is below `c0:01.2`. Corrected errors also precede
this warmup. They warrant hardware follow-up but do not identify this fault's
cause; cross-GPU traffic still prevents excluding PCIe entirely.

Next diagnostic run should be one bounded reproduction of the saved failing
batch, after the mandatory PCIe check, with kernel/wave fault capture and
explicit slot-bound diagnostics. Preserve a complete dump on NVMe (allow at
least the observed 33.7 GB plus headroom). Serialization/debug instrumentation
changes timing and cannot qualify performance or rule out races on a PASS.
Only then isolate graph replay, LMCache stores and speculation one variable at
a time under the same saved request protocol. Do not restart the full matrix
as a diagnostic substitute.

Offline evidence and copied sources from the failed image:
`bench/results/glm-memory-fault-analysis-20261012/`. This investigation did not
restart services, launch GPU tests or modify runtime kernels.

### Bounded fault reproductions — 2026-10-12

**Unresolved; not a stability qualification or a confirmed fix.** Evidence is
in ignored `bench/results/glm-fault-fix-20261012/`. These runs used the production
library with the dense-FP8 and all-gather candidates disabled. PCIe evidence,
request bodies, streams and telemetry were retained. Production was stopped
for isolated diagnostic containers; no full matrix was retried.

- `debug1`: four concurrent 8K/low requests completed, each with 256 output
  tokens. The debug agent was not loaded because the image disables legacy
  HSA tools; this run cannot provide debugger fault coverage.
- `debug2`: three cold batches of four 8K/low requests completed (12 requests).
  The agent was verified loaded in all eight workers. Temporary, untracked
  kernel overlays asserted tail-block and routed-expert bounds; neither
  assertion fired. Instrumentation changes timing, so this does not exclude
  a race. Precise-memory reporting was not supported on all agents.
- `debug3`: preflight failed before model launch; retained as a failed attempt.
- `debug4`: the natural-EOS 8K/high C4 series was interrupted after requests
  started, because new driver faults made further validation unsafe. Do not
  count this attempt as a passing completed series.

The kernel timeline distinguishes two fault classes. The original worker
faults were VMID 8 (TCP data on ec at 21:30:07 UTC, then SQC instruction fetch
on c6 at 22:18:42 UTC on October 11). A new CPC fault series in VMID 0 / PASID 0
started at 22:58:39 UTC, during shutdown of `debug2`, after its requests passed.
At 23:07:02 UTC e6 also reported `MES(1) failed to respond to
msg=INVALIDATE_TLBS`. By 23:08:51 UTC there were 114 VMID-0 page-fault log entries.
The temporal association with debugger lifecycle is evidence to investigate,
not proof that the debugger caused either the original or new faults.

After stopping the diagnostic model and cache, a bounded allocation/addition/
synchronization check **without the debug agent** returned correct results on
all eight GPUs, but initialization still produced CPC/VMID-0 faults. Successful
arithmetic alone therefore does not establish a clean driver state. GPU reset
was requested from the operator because noninteractive sudo authentication is
unavailable. At this checkpoint production GLM/cache remain stopped, LiteLLM
is running without its backend, and production restoration is pending reset
and a clean health check. No runtime fix has been promoted.

CPU checks for cache geometry, completion-event ordering and the launcher:
19 passed, 3 skipped. These cannot validate GPU memory safety. Next steps after
reset: capture PCIe mapping again, verify clean kernel logs without the debug
agent, then isolate graph replay, cache stores and speculation one variable at
a time using the saved reproduction protocol. Do not resume the full matrix
or claim a fix from a non-reproducing smoke test.

### Post-reset reproduction: fatal PCIe link loss — 2026-10-12

The operator reset all eight GPUs successfully at 23:10 UTC on October 11
(01:10 local on October 12). Fresh PCIe evidence and a no-debugger arithmetic
check passed on all eight devices with no new GPU faults. `control5` then
started the unchanged C4/LMCache runtime, without debug-agent or assertion
overlays, with 128 GiB RAM / 512 GiB disk cache. Before any test request,
during model initialization, the following sequence occurred:

| Local time | Observation |
|---|---|
| 01:16:08 | Bridge ca:00.0 reported correctable Replay Rollover / Timeout |
| 01:16:10 | Root c0:01.4 reported DPC containment and **Uncorrectable (Fatal), SDES** |
| 01:16:10 | amdgpu cc:00.0 reported **device lost from bus** |
| 01:16:14 | Cache process reported GPU Hang on node 8; model workers failed |
| 01:16:15 | Link retraining at 2.5 GT/s failed; AER reported device recovery failed |

The affected path is `c0:01.4 -> ca:00.0 -> cb:00.0 -> cc:00.0`.
The GPU is **HIP/TP rank 7, AMD-SMI ID 3**, UUID
`62007551-0000-1000-80a7-8b0520f58a9b`; ordinals are not interchangeable.
Before failure the CPU-facing link was Gen5 x4 and the GPU-facing link Gen5
x16. After failure the downstream configuration space became unreadable
(`Unknown header type 7f`, link width 63 is invalid data, not a real x63 link).
The root fatal counter records one SDES event.

This establishes a PCIe-link failure for **this attempt**, rather than a
model allocation OOM or a failing inference request. It does not establish
which component failed (link/riser, power, card, port, firmware), nor prove
that the earlier TCP/SQC faults had the same cause. Linux's
[AER documentation](https://docs.kernel.org/PCI/pcieaer-howto.html) distinguishes
fatal link errors from corrected errors; the earlier BadTLP counts alone were
insufficient for this conclusion.

Both diagnostic services were stopped; production GLM/cache remain stopped.
No kernel workaround was promoted and the eight planned requests did not run.
Restore PCIe connectivity before more GPU work. An operator-controlled power
cycle and inspection of this card's connection/power are appropriate after
failed bus recovery. A subsequent Gen4-only test on the affected link can
isolate link-speed sensitivity, but is a hardware experiment and cannot be
compared to Gen5 performance as an unchanged baseline. Do not hide AER errors.

Evidence: `bench/results/glm-fault-fix-20261012/after-reset/` contains the reset,
health check, kernel timeline, before/after topology and link counters;
`control5/` contains the frozen protocol and failed startup. Restart and
hardware inspection require operator action; the original fault is not yet
closed as fixed.

### Recovery after connection/power repair and reboot — 2026-10-12

The operator confirmed repairing connections/power and rebooted the host at
01:59:44 local. Fresh mapping, topology, idle and loaded PCIe evidence was
captured for all eight GPUs. All eight passed a bounded arithmetic check.
The expected CPU-facing Gen5 x4 and GPU-facing Gen5 x16 links were observed
under load. All eight cards now have a 210 W power limit; earlier runs had
two cards at 300 W, so these results are **not a performance comparison**.

The original production model/cache containers were restarted without a debug
agent, assertion overlays or a promoted kernel change. GLM C4/DFlash K4 and
LMCache 128 GiB RAM / 512 GiB disk remain running, with LiteLLM on port 4000.

The bounded `recovery6` diagnostic sent four independent 8192-token NIAH
requests concurrently, then repeated those four prefixes. Sampling was
temperature 1.0, top_p 0.95, high effort, seed 1234, max_tokens 1024, EOS enabled.
All responses stopped naturally after 16–17 generated tokens; this does not
qualify long decode or long thinking.

| Check | Result |
|---|---|
| Strict exact-match | 8/8 |
| Semantic needle retrieval | 8/8 |
| Output-format compliance | 8/8 |
| Generation completed | 8/8 |
| Backend/runtime error | None in these requests |
| Preemptions | 0 |
| Cold GPU/external prefix hits | 0 / 0 |
| Repeat GPU prefix hits | 32256 / 32768 queried tokens (98.4375%) |
| LMCache cold stores | 224 submitted and completed GPU stores; 224 completed disk-store requests, 672 objects |

The repeated batch also reported L1 lookup hits, but external restored tokens
and L2 loads were zero: this validates GPU APC reuse and cache writes, **not
RAM/disk restore**. Two additional small streaming requests through LiteLLM
returned the correct arithmetic answer using OpenAI Chat Completions and
Anthropic Messages, with normal completion on both paths.

No new GPU page fault, fatal AER, device loss or MES timeout appeared in the
captured boot log. Correctable link errors remain: c4 reports nine BadTLP and
one Timeout; c0:01.2 one Timeout; ca, e1 and e4 one BadTLP each; c0:01.4 two
BadTLP. These are cumulative counters at the final capture, not all attributed
to an individual request. Restoration is successful, but the original
TCP/SQC memory-fault investigation remains open; a short non-reproducing test
does not establish that every fault was fixed. Full 32K/128K qualification and
RAM/disk restore tests have not been rerun after this hardware change.

Evidence: `bench/results/glm-fault-fix-20261012/post-reboot/` and `recovery6/`
contain the frozen protocol, PCIe samples, requests, streams, separate NIAH
scores, metric deltas, proxy checks and final kernel/AER logs.

### C4 × 128K after hardware repair — 2026-10-12

The requested bounded C4 test passed on the running production configuration
above: four independent inputs of **131072 tokens each**, effort high,
temperature 1.0, top_p 0.95, top_k -1 and seed 1234. The saved custom scenario
uses the standard corpus and BetterBench 0.6.0 for timing, but **is not the full
RAM/disk matrix or a matched comparison to historical measurements**.

Protocol: one cold NIAH quality batch with fresh per-client salts; then the
same GPU-cached prefixes with one excluded performance warmup and three
measured batches of 256 forced output tokens; finally one natural-EOS warm
quality batch. Quality allowed 8192 output tokens and completed in 24–40.
No cache was cleared and no service configuration changed. All 24 requests
completed without runtime errors or preemption. Loaded PCIe mapping/width/speed
checks passed for all eight GPUs, at 210 W per card.

| Measurement | Result |
|---|---|
| Cold quality: strict / retrieval / format / completion | 4/4 each |
| GPU-warm quality: strict / retrieval / format / completion | 4/4 each |
| Cold TTFT, in completion order | 93.44 / 187.81 / 289.86 / 376.65 s |
| Cold effective prefill from server phase counters | 1386 tokens/s: 524288 computed tokens / 378.22 summed prefill seconds |
| GPU-warm decode, 12 measured requests | Median 22.57 tok/s per request; range 19.24–30.77 |
| Per-client median decode over three repeats | 20.61 / 21.62 / 23.06 / 26.09 tok/s |
| GPU-warm TTFT, measured requests | Median 2.76 s; range 0.95–5.03 s |
| Aggregate output / whole-batch wall time | 58.82 / 69.63 / 65.59 tok/s; includes TTFT |
| All-four decode overlap, measured batches | 7.06 / 7.60 / 9.05 s |
| Speculative acceptance, measured batches | 38.07%; separate from decode rate |
| GPU prefix reuse in each warm batch | 520704 / 524288 tokens (99.3164%) |

Cold prefill was scheduled sequentially; the cold TTFTs include queue time
and do not represent four simultaneous prefill computations. Server prefill
phase time is not isolated kernel time. Simultaneous C4 decode was established
in every warm measured batch. Fixed-output NIAH continuation is synthetic
throughput, not a completed-answer quality test; only the two natural-EOS
batches contribute quality scores.

LMCache completed 3616 GPU stores and 3616 disk-store requests after cold
population. RAM cache reached approximately 99.46 GiB, including previous
cached content. Warm requests used GPU APC; external restored tokens and L2
loads were zero, so RAM/disk restore remains untested in this series.

No new GPU page fault, fatal link loss or OOM appeared. Correctable PCIe
BadTLP/Timeout errors continued, so this pass does not close the link-stability
investigation. GLM, LMCache and LiteLLM were left running.

Evidence: `bench/results/glm-c4-128k-20261012/` contains the frozen protocol,
corpus/hash, runtime identity, PCIe telemetry, all 24 requests/responses,
per-batch metrics, BetterBench timing, `summary.json` and kernel logs.
