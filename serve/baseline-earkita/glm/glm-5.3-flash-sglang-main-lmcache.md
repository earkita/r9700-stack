# GLM-5.3-Flash: SGLang main + LMCache candidate

This is an **unqualified** branch of the SGLang experiment. It does not replace
the v0.5.20/MTP3 profile or the vLLM K4 service. The image pins SGLang
`aaf5f397a5a6565ad2cbcfe64bab56bdebbe4b75` and LMCache
`32cbdcb7a94bfb07fe25dad104f95e11b1f472b1`; the ROCm base image is
digest-pinned in `docker/Dockerfile.sglang-glm-main-lmcache`.

The new SGLang cache is `LMCacheUnifiedRadixCache`, which registers full KV,
KDA/Mamba state, and DSA indexer buffers through LMCache's unified connector.
The old v0.5.20 LMCache path was KV-only and must not be enabled for this model.
Our port is the isolated `sglang-aaf5f397-gfx1201-glm.patch`. Upstream now
provides the ROCm KPool MQA path and HIP verify/plan support, so those older
patches were not copied. The remaining gfx1201 fixes include the Quark FP8
dense loader, packed MXFP4 MoE fallback, Hadamard fallback, Triton DSA tail,
and HIP pooled top-k. No arithmetic format was changed deliberately.
The separate GLM47 streaming buffer-drain patch is included. The older
structured-JSON/tool rewrite and stable-prefix experiments are left disabled;
their behavior must be requalified against the new upstream parser.

The first 8-GPU startup reached LMCache registration, then rejected GLM's
contiguous KDA conv state with `slice_axis=1`. In SGLang this field identifies
the TP slice axis *inside* each slot; the slot axis stays at dimension 0.
The separate pinned LMCache patch removes that mistaken restriction while
retaining the contiguity check. This mapping still needs a completed GPU
store/restore test before it can be considered qualified.
The next launch exposed `hipErrorInvalidDevicePointer` during LMCache IPC
registration: the model container used `HSA_ENABLE_IPC_MODE_LEGACY=0`, but the
separate cache container did not. Both containers now select the same HIP IPC
mode, as in the working vLLM LMCache deployment. This also needs a completed
store/restore check; startup alone cannot prove cache correctness.
With IPC fixed, the next first-token warmup reached AITER paged MQA but the
base image selected its Triton kernel while SGLang had enabled the
Gluon-only `Preshuffle=True` layout. The upstream
`SGLANG_DSA_HIP_DISABLE_PRESHUFFLE=1` switch was tested and rejected:
it sets KPool pages to one token, whereas this checkpoint requires
`index_kpool=4`. The candidate therefore keeps 64-token KPool pages and
adds a gfx1201-only Triton reader for the preshuffled layout in the pinned
AITER image. A one-GPU diagnostic matched the original row-major reader's
logits exactly for 64, 96 and 128 tokens. This checks the reader, not the
whole model or cache restoration. With that patch the TP8 server reached
`/ready`, but emitted an LMCache warning: GLM's KV, KDA/Mamba and indexer
objects need `--separate-object-groups`. The launcher now sets that flag;
the first ready instance was stopped before accepting a user request.

### ROCm memory ordering review (2026-10-17)

The next bounded target-only probe completed a **cold** 3503-token request and
stored 120 chunks. After SGLang's GPU prefix flush, LMCache reported a 3328-token
RAM restore, but the subsequent request ended in simultaneous GPU page faults on
seven TP workers. It yielded no valid warm answer or performance number. Raw
model/cache/kernel logs and the request are under ignored
`bench/results/sglang-main-upgrade-20261017/lmcache-probe/`.

The working vLLM GLM cache path has several distinct patches and settings:

| Existing vLLM measure | SGLang candidate status after review |
|---|---|
| `glm_lmcache_events.py` acknowledges an event only after local `event.synchronize()`; both endpoints install the same protocol | New opt-in `r9700_sglang/lmcache_events.py` and pinned LMCache patch select the same completion protocol in model and daemon. HIP memory IPC remains unchanged. |
| Matching `HSA_ENABLE_IPC_MODE_LEGACY=0` and host IPC/network namespaces | Already present on both SGLang containers after the earlier invalid-pointer startup failure. |
| `KV_CACHE_ALLOCATOR=pinned-gtt`: native HIP host allocation, `HSA_USERPTR_FOR_PAGED_MEM=0` on daemon only | The candidate now defaults to `--no-l1-use-lazy` with that ROCr setting and a new disk namespace. `lazy` remains an explicit diagnostic mode. |
| GLM logical MLA/cache geometry, KPool alignment and exclusion of DFlash scratch | The SGLang unified cache has a different layout and lifecycle. Its own KDA/DSA mapping and restore still require model-level validation; the vLLM adapter must not be copied into it. |
| Native filesystem object-group key and eviction-lock patches for LMCache 0.5.5 | The pinned LMCache dev commit already parses four/five-field group keys in its native FS backend. Its delete path lacked the old protection for locked/pending lookup and store keys, so `lmcache-32cbdcb-native-eviction-locks.patch` ports that guard against its current interface. Disk restore/eviction still needs a separate functional check. |

In pinned LMCache dev, the model records an event before store/retrieve, the
daemon waits on that producer event, copies KV/KDA/indexer objects on its
stream and returns a completion event; SGLang then waits before consuming the
restored state. The default HIP event IPC exported pending work. The new
adapter synchronizes locally **before** exporting each acknowledgement in
both directions, so this review covers all object groups using that transfer
context, at the cost of losing copy/compute overlap. CPU tests verify the
ordering and that a foreign handle or GPU synchronization error fails closed.

This is a **candidate mitigation**, not a diagnosed root cause or a passing GPU
restore. The prior run also used a 128 GiB lazy L1 pool, which could exercise
USERPTR behavior; neither that nor event IPC has been isolated experimentally.
The previous namespace remains untouched. After GPU recovery, first run one
bounded cold→RAM restore with complete-answer and KDA/DSA checks, then a
separate disk restore. Repeat PCIe preflight before the GPU series and save
raw evidence. Do not report comparative performance from this smoke test.

The patched image built successfully from the pinned commits. Host-only event
and launcher tests passed (7/7), and image-local CPU tests confirmed the native
eviction guard (3/3). The image also selected the synchronized event backend
with the opt-in flag and exposed both patch labels. These checks do not run
the HIP transfer kernels or qualify an actual cache hit on the model.

Build and inspect without starting a GPU process:

```bash
docker build --build-arg MAX_JOBS=4 \
  -f docker/Dockerfile.sglang-glm-main-lmcache \
  -t r9700/sglang:glm53-aaf5f397-lmcache-32cbdcb-syncgtt .
DRYRUN=1 bash serve/baseline-earkita/glm/glm-5.3-flash-sglang-main-lmcache.sh
```

The candidate script starts one LMCache daemon and one SGLang container using
the same image. Defaults are 128 GB host cache and a separate 512 GB LRU disk
namespace at `/mnt/ai/cache/sglang-aaf5f397-32cbdcb-syncgtt`; override with
`LMCACHE_RAM_GB`, `LMCACHE_DISK_GB`, `LMCACHE_ROOT`, and `LMCACHE_ALLOCATOR`.
The daemon listens on
host loopback port 5556 and SGLang on 8081. The daemon uses separate object
groups for the hybrid GLM states. The script refuses a stale
container name or an occupied cache port. `MTP_STEPS=0` is the first
correctness profile. Current SGLang main explicitly rejects speculative
decoding with this LMCache connector, so this profile rejects `MTP_STEPS>0`
before starting a container. MTP3 remains available in the unchanged
v0.5.20 profile; combining it with this cache needs upstream support and
separate qualification.

Before the first GPU run, recover from the recorded MES fault and perform the
mandatory PCIe preflight in `testing.md`. Save raw BDF/HIP/worker mappings,
topology, idle and loaded link state under `bench/results/`. Then compare
target-only 8K/32K responses with and without LMCache using frozen prompts,
verify cache hits and complete answers, and test KDA/DSA state restoration
across multiple turns before long-context cached prefill. A
successful import or server start alone does not qualify correctness or speed.

Rollback: stop only the two candidate containers, then use the unchanged
v0.5.20/MTP3 profile and image. No existing cache namespace is deleted.
