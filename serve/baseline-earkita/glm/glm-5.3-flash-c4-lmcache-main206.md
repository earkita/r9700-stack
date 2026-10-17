# GLM C4 K4 + LMCache on vLLM 2067297b (candidate)

This profile is isolated from the qualified e97573215 K4 service. It pins
vLLM source commit `2067297b2f15f0776be6d2a7e1b435bd20af60c1` and the
published ROCm 10 base digest `sha256:fe8256813e979a16e7d4528204b93b6523d9286bff420427118a4943628c3582`.
The base image corresponds to upstream `39983a5e`; the final source build
compiles the exact newer commit. LMCache remains the patched 0.5.5 source.

Build independently of the running service:

```bash
docker build -f docker/Dockerfile.glm-vllm-main206 -t r9700/vllm:upstream-2067297 .
docker build -f docker/Dockerfile --build-arg BASE=r9700/vllm:upstream-2067297 \
  -t r9700/vllm:glm53-plugin-2067297 .
docker build -f docker/Dockerfile.glm-lmcache \
  --build-arg BASE=r9700/vllm:glm53-plugin-2067297 \
  -t r9700/vllm:glm53-lmcache055-main206 .
docker build -f docker/Dockerfile.glm-lmcache-main206 \
  -t r9700/vllm:glm53-lmcache055-main206-blhnc .
DRYRUN=1 bash serve/baseline-earkita/glm/glm-5.3-flash-c4-lmcache-main206.sh
```

The candidate uses model port 8086, cache port 5655, metrics port 9191,
unique container names and NVMe namespace. Default cache limits are 128 GiB
RAM and 512 GiB disk; do not allocate this pool alongside the production
cache. The launchers refuse name collisions and never replace containers.
The candidate binds its API to `127.0.0.1` and enables the vLLM development
endpoint required to reset GPU APC between cache-tier tests. Do not expose
this profile to the LAN while development endpoints are enabled.
The candidate's GLM cache group uses 1280-token pages (the older pin used
1152). vLLM main206 requires physical `BLHNC`; after selecting a layer, its
pages are in `HND` order. The last image layer applies a narrowly scoped
LMCache 0.5.5 Mamba-page compatibility edit. It does not change model math
or the qualified K4 image. This interface requires a separate cold/RAM/disk
smoke before any cache behavior can be claimed.
`KVMEM_GB=3.25` is the per-GPU KV budget; this profile translates it to the
new `--kv-cache-memory-bytes` option while leaving the shared launcher and
qualified K4 profiles unchanged. Pin updates belong in this candidate profile,
its Dockerfile and its compatibility checks, so it can be synced independently.
At 4.125 GiB, a C2 8K cold prefill exhausted VRAM in upstream MXFP4 MoE
dequantization (576 MiB allocation); the later shared-experts assertion was a
consequence. The 3.25 GiB smoke completed C2 and gave a reported per-request
max length of 538880 tokens. This does **not** qualify C4 at 128K or any
256K workload; those require their own capacity and correctness checks.

Before a GPU trial, follow `testing.md`: capture PCIe mapping/topology and
idle/loaded links for all eight cards, freeze the quality and performance
protocol, and arrange exclusive access. First qualify an isolated K4 smoke
with `KV_OFFLOAD=0`, then the matched RAM/NVMe cache trial. Check cold/warm
hits, KDA and k-pool continuation, 8K/128K answers, tools, OOMs and K4
acceptance before comparing decode or switching service traffic. Record every
attempt in ignored `bench/results/`. Results from this candidate are currently
**unqualified**; the existing e97573215 K4 profile remains the rollback.

## Bounded diagnostic smoke, 2026-10-17

Raw evidence: ignored `bench/results/glm-main206-smoke-20261017T093052Z/`.
The final candidate run is `cache-kvmem-3p25/`; every failed attempt and its
log is retained alongside it. Same 8192-token prompt, temperature 0, top-p 1,
top-k disabled, effort high, natural EOS and a maximum of 256 output tokens.
These output-limited samples are synthetic diagnostics, not completed-answer
quality or a matched performance comparison with the qualified K4.

| C1 cache source | TTFT | End-to-end | SSE decode estimate | Verified hit |
|---|---:|---:|---:|---|
| Cold | 5.646 s | 10.264 s | 55.23 tok/s | external 0, GPU APC 0 |
| RAM | 0.576 s | 5.146 s | 55.80 tok/s | L1 7680 tokens, GPU APC 0 |
| NVMe | 0.593 s | 4.872 s | 59.60 tok/s | L2 7680 tokens, 72 L2 loads, L1/GPU APC 0 |

C2 with two independent cold 8K requests completed 256 tokens each, without
preemption: SSE decode estimates 18.12 and 34.27 tok/s per request. Their
prefills overlap decode, so these numbers are not isolated decode rates.
A separate arithmetic check ended naturally (`stop`) with the correct answer
`4`. The PCIe mapping and loaded link preflight passed for all eight GPUs.
One attempt per cache tier and one C2 batch establish only a smoke result;
spread, long-context stability, tool calling and comparative performance are
not established here.
