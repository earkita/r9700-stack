# Earkita baseline standard

Executable specification: [`glm.json`](glm.json). GLM profiles and their
per-script notes live in [`serve/baseline-earkita/glm`](../../serve/baseline-earkita/glm/README.md). This standard follows
[`serve/baseline-earkita/glm/testing.md`](../../serve/baseline-earkita/glm/testing.md). It is a **new protocol**;
results from older, differently configured runs are not comparable.

## GLM matrix

| Dimension | Frozen values |
|---|---|
| Input, after chat template | 8192 / 32768 / 131072 tokens |
| Concurrent independent clients | C1 / C2 / C4 on the same C4 server |
| Thinking | low / high / max, explicitly in `chat_template_kwargs` |
| Cache | cold / warm RAM (L1) / warm filesystem disk (L2) |
| Performance output | 256 tokens including reasoning; `ignore_eos=true` |
| Sampling | temperature 1, top-p 0.95, top-k -1, seed 1234 |
| Repetitions | 1 excluded warmup + 3 measured batches per cell |
| Separate NIAH diagnostic | One batch per cell; natural EOS, budget 8192 |
| Client | BetterBench 0.6.0, pinned Git revision in `glm.json` |
| Runtime | vLLM e97573215, LMCache 0.5.5, TP8/no EP, DFlash K4, FP8 target/draft KV, graphs |

The matrix contains **324 performance/warmup batches (756 requests)** plus
**81 quality batches (189 requests)**. `max` means reasoning effort, not an
additional maximum-context size. The longest tested input is 128 Ki tokens;
output needs additional context capacity. This is a substantial run, not a
smoke test; no duration or long-context qualification is claimed in advance.

The standard requires simultaneous decode overlap for C2/C4 performance
batches. A short 256-token output can finish before another long cold prompt
completes prefill; this fails measurement qualification, not necessarily the
backend. For explicitly frozen **cache-only diagnostics**, a custom protocol
may set `allow_serial_cold_prefill_diagnostic=true`. Only cold batches may then
continue without overlap, with `decode_concurrency_qualified=false` recorded
in JSON and reports; they must not be presented as simultaneous C2/C4 decode.
Warm RAM/disk batches still require overlap. This option is absent from the
standard protocol and must never silently change a comparison.

Each cycle uses fresh deterministic salts derived from its frozen run ID,
and executes cold → RAM → disk with the **same prompts and salts**. Distinct
clients use distinct salts/needles. Cold populates RAM and disk. Before every
batch, GPU APC is reset with a verified successful response. Before cold/disk,
the cache service's **L1 only** is cleared after pending disk stores complete.
Disk files are retained. All expected full prompt chunks must hit the declared
external tier; GPU hits, partial/mixed-tier hits or missing metrics invalidate
the batch. Counter publication is awaited, rather than interpreted as zero.
LMCache 0.5.5 OTel event counters are absent until their first observation;
only these known, pinned counters may start at zero. Once published, a
disappearing counter fails. Cold population must show completed GPU→RAM and
RAM→disk stores, including successfully stored L2 objects (completed requests
alone also include failures); warm states still require full, positive tier-hit evidence.

Disk-warm means filesystem/L2 restore with buffered I/O. The OS page cache is
**not** flushed, so this does not prove physical NVMe throughput. Repeated
prompt requests without a GPU reset would measure GPU APC instead of offload.

## One-time setup and dedicated serving session

```bash
bash bench/baseline-earkita/setup.sh
python3 bench/baseline-earkita/run.py plan
```

`setup.sh` installs [BetterBench](https://github.com/GGZ14/BetterBench) at the
pinned revision into `.runtime/baseline-earkita/venv`. No global installation.

Schedule an exclusive GPU session first: no Claude/Hermes/LiteLLM traffic to
this backend. The runner does **not** stop production or start containers.
When the GPUs and ports are free, launch the existing combined profile with
benchmark-only overrides:

```bash
VLLM_SERVER_DEV_MODE=1 HOST=127.0.0.1 \
KV_CACHE_RAM_GB=128 KV_CACHE_DISK_GB=512 KV_CACHE_ALLOCATOR=pinned-gtt \
KV_CACHE_NAMESPACE=earkita-bench-v1 \
bash serve/baseline-earkita/glm/glm-5.3-flash-c4-lmcache.sh \
  NAME=earkita-glm-bench KV_CACHE_NAME=earkita-glm-cache-bench
```

Wait for model readiness before starting the suite. The dev-mode flag exposes
the APC reset endpoint; it is required only for this dedicated **loopback**
benchmark service. A production endpoint without it fails preflight; no restart
or change is attempted. Never expose this benchmark API to LAN clients.

The full-suite example uses the approved **128 GiB L1 budget across all eight
GPUs** and **512 GiB L2 budget**, with native HIP/GTT-backed RAM allocation.
This avoids the cache pool's pageable USERPTR mapping and its observed
compaction-related stalls. GTT is host RAM, not GPU VRAM; its driver limit must
also fit the allocation. Cache-size production defaults
remain unchanged. This is not a claim that C4 × 128K has passed validation.
The actual configured budget is frozen with the protocol, and the runner
requires full-prefix L1 hits for every RAM-warm batch. Insufficient L1 capacity
fails that gate; the runner does not raise the allocation automatically or
silently count disk restores as RAM hits. Size changes require a fresh protocol
and both comparison variants to be rerun. Fresh salts accumulate disk
data throughout the matrix. With `KV_CACHE_DISK_GB` enabled, LMCache's L2 LRU
evicts least-recently-used objects in the dedicated benchmark namespace, starting
at 75% occupancy and removing 20% of tracked objects per pass. The 512 GiB value
is an asynchronous accounting budget, not a filesystem quota. Locked and pending
lookup/store objects are protected by the pinned image patch. All cache-hit gates
remain mandatory: eviction must not turn a RAM/disk measurement into recomputation.
Use a fresh empty namespace: this version does not inventory existing files after
a cache-daemon restart. No production cache or result/transcript is deleted.
The runner stops below 64 GiB free. A failed capacity gate is
not a benchmark result. Changing budgets/protocol requires a fresh frozen run.

## Unattended execution

```bash
bash bench/baseline-earkita/glm.sh
```

This generates and hashes exact-length corpora with the **serving container's
local tokenizer**, records runtime/image versions, plugin/client code hashes,
checkpoint metadata hashes, device mapping and PCIe topology, freezes
`protocol.json`, then runs the entire matrix without agent supervision.
`prepare` does no inference or cache clearing. Results go to a fresh ignored
`bench/results/earkita-glm-<UTC timestamp>/` directory.

To detach from the terminal and choose the output directory:

```bash
RUN_DIR="$PWD/bench/results/earkita-glm-$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_DIR
mkdir -p bench/results
nohup bash bench/baseline-earkita/glm.sh > "${RUN_DIR}.log" 2>&1 &
```

Overrides: `MODEL_CONTAINER`, `CACHE_CONTAINER`, `BASE_URL`,
`CACHE_METRICS_URL`, `CACHE_PORT`, `RUN_DIR`, `PCIE_POLICY`.
Do not point the runner at a shared production cache: its cache preparation
intentionally clears GPU APC and LMCache L1. Concurrent suite invocations in
this checkout are locked out. Unexpected requests also fail the counter gate.

PCIe evidence covers all HIP ordinals/TP ranks and every upstream bridge,
including shared uplinks and NUMA nodes. Default expected speed/width is each
link's advertised maximum. For intentionally narrower wiring, pass a JSON
policy with `reason` and `links`, mapping **every** BDF to
`{"speed": 32, "width": 16}` (GT/s and lane count). Obtain BDFs from
`hardware-before/links.json`; prepare a new run after changing policy.
Unexplained downtraining or an unobserved full-speed link during the planned
load fails the hardware gate; low speed at idle alone does not.

## Artifacts and interpretation

- `status.json`: running / complete / failed, current batch or error.
- `protocol.json` + SHA256, frozen corpora and hardware evidence.
- `batch-*/request-*.json`, `response-*.json`, `result.json`: every attempt,
  including warmups and failures. Quality responses retain final text and
  reasoning; performance records retain BetterBench streaming timing samples.
- Before/after raw Prometheus counters and per-second GPU/PCIe telemetry.
- `report.json`, `report.md`: partial reports updated after each batch and on
  failure; warmups excluded from performance aggregates.

Report TTFT and first answer latency separately; absence of a visible answer
within the synthetic performance budget is not a quality failure. Effective
prompt rate = prompt tokens / TTFT; for warm cache this includes restoration
and residual prefill and **is not pure prefill compute throughput**.
Decode includes reasoning and excludes TTFT. BetterBench uses
`(completion_tokens-1)/sum(stream_update_gaps)`; batched SSE updates are not
per-token ITL. Per-request rates are distinct from aggregate output/whole-batch
wall time, which includes prefill. C2/C4 require overlapping decode intervals.
DFlash acceptance and position counts are reported separately.

NIAH reports **strict exact-match / semantic needle retrieval / output-format
compliance / generation completed / backend-runtime error** independently.
Here retrieval means finding the intact synthetic key, with channel evidence,
not an LLM judge. An extra sentence can fail strict/format while retrieval and
completion pass. Format-only failures remain in reports and do not stop the
suite or implicate DFlash/kernels. Foreign client keys, retrieval failure,
truncation or runtime errors stop it. NIAH alone is not general model quality.
No selective reruns: a subsequent full run uses a new directory and preserves
the original failure; non-reproducibility must be established across full runs.

Errors cause a nonzero exit and a partial report. No automatic retries of
inference, server restarts, dependency updates, parameter tuning or pushes.
After a failure, correct the cause and prepare a **new** run. A protocol change
requires both comparison variants to be rerun; this script does not claim
regression/improvement from historical mismatched results.

Offline verification (no GPU work):

```bash
python3 -m unittest discover -s tests -p 'test_earkita_benchmark.py'
```
