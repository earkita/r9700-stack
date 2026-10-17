# GLM C4: checkpoint MTP3 with LMCache (experimental)

This is an opt-in candidate for a **matched** comparison against the working
[DFlash K4 + LMCache profile](glm-5.3-flash-c4-lmcache.md). It is not a
replacement for K4. The original MTP3 + LMCache image failed its 128K
correctness gate. The separate LMCache #5004 backport passed the focused
protocol, but did not beat K4 consistently. The launcher
does not replace containers and refuses to
start while a model answers on the K4 port 8080.

`glm-5.3-flash-c4-mtp3-lmcache.sh` uses the same pinned e97573215 / LMCache
0.5.5 source and GLM checkpoint as K4, with only the LMCache #5004 block
relocation backport in its separate image. It uses TP8, C4, FP8 KV, 4.125 GiB/GPU KV and
packed target W4A4. The checkpoint has **one** NextN layer, recycled for three
MTP proposals; its block-FP8 experts retain the upstream arithmetic. There is
no separate DFlash2 checkpoint and no DFlash cache adapter. MTP3 graph sizes
are 4/8/12/16 for C1/C2/C3/C4 verify batches. Vision remains unqualified for
the pinned built-in MTP implementation.

The cache service uses separate defaults: model port 8081, transfer port 5556,
metrics port 9092 and namespace `glm53-e975-fp8-mtp3-v1`. Its 128 GiB RAM and
512 GiB filesystem budgets are total cache budgets, not allocations per GPU.
Do not reuse the DFlash namespace: the draft attention/indexer state differs.
The pinned vLLM's GLM group builder puts the NextN MLA and compressed indexer
in the attention group. A CPU test of its **actual group builder** confirms
that `GlmLMCacheMPConnector` selects those two NextN layers and excludes only
the non-cacheable circular k-pool tail. This verifies the metadata mapping,
not restore correctness, cache hit rates, speculative rollback or model output
on GPU. The tail must be rebuilt from an aligned suffix after an external hit.

Inspect without using the GPU:

```bash
bash serve/baseline-earkita/glm/glm-5.3-flash-c4-mtp3-lmcache.sh DRYRUN=1 MAXLEN=131072
```

Before any model trial, follow [testing.md](testing.md): freeze the corpus and
protocol, record all eight HIP/BDF/worker mappings and PCIe links before and
during the planned run, and obtain exclusive access to the eight R9700. Keep
the running K4 service and its cache intact until that session is deliberately
scheduled. Starting a second model on the same eight cards is not a test.

Qualification sequence for the separate candidate:

1. Confirm the exact image, checkpoint and tokenizer hashes; capture resolved
   MTP3, graph and cache-group logs. Run bounded 8K/32K natural-EOS answer,
   streaming and two-turn tool checks with checkpoint sampling. Save every
   response, reasoning, finish reason and error.
2. Prove cold store, then RAM and filesystem-L2 restore of the **same** prompt,
   with GPU APC cleared before external-hit checks. Verify NextN attention and
   indexer persistence, reconstructed tail, prefix-hit counters and subsequent
   accepted-token continuation. A filesystem-L2 hit may be served by the OS
   page cache; do not call it a physical NVMe read.
3. Use the frozen new C1/C4 × 130816-input/256-output comparison protocol for
   both MTP3 and K4, with `MAXLEN=262144`: identical prompts, effort `high`, temperature 1, top-p 0.95,
   top-k -1, seed 1234, cache phase, one excluded warmup, three measured
   repetitions, BetterBench 0.6.0, power caps and 210 W policy only if actually
   observed. Require four overlapping decode intervals for C4. Save TTFT,
   effective prefill, each request's decode, aggregate output, speculative
   acceptance by position, full wall time, preemption and PCIe evidence.
4. Separately run the same natural-EOS completed-answer and tool corpus on both
   variants. Do not infer quality from forced-output benchmark text. Promote
   nothing unless correctness, stability and full-task latency pass together.

The existing `bench/sglang_glm_contract.py` exercises a generic direct
OpenAI-compatible endpoint despite its historical filename. In the exclusive
MTP3 session, run it with `--base http://127.0.0.1:8081 --model glm-5.3-flash`
and a fresh `--out` under `bench/results/`; repeat with the K4 endpoint and
the same prompt contract. It checks completed answers, streaming, and both
streamed and non-streamed two-turn tool results. This is a bounded contract
smoke at temperature 0, not a matched long-context performance or
checkpoint-sampling quality score.

## Correctness gate before speed claims

Use the same checkpoint, tokenizer, chat template, TP8 and target kernels for a
short **target-only (speculation disabled) reference** in the exclusive test
session. This is a correctness oracle, not the K4 speed baseline. Preserve its
resolved server arguments and every request/response. In particular:

```bash
# Inspect the target-only command without occupying GPU or changing K4:
bash serve/baseline-earkita/glm/glm-5.3-flash-c4-mtp3-lmcache.sh \
  REFERENCE=1 DRYRUN=1 MAXLEN=131072
```

`REFERENCE=1` removes speculative decoding while retaining the target model,
image, TP8, FP8 KV and target kernels. It uses a separate model name/port
(8082), cache name/port (5557), metrics port (9093) and target-only cache
namespace. `KV_OFFLOAD=0` gives an uncached target control; the default
`KV_OFFLOAD=lmcache` exercises target-only external restore. These modes are
for a separately scheduled exclusive session; the launcher refuses to start
while K4 answers on port 8080. A target-only control does not substitute for
the matched K4/MTP3 performance legs.

1. Compare target-only, K4 and MTP3 on the same frozen greedy prompts and
   token budgets at 8K/32K and at the intended 128K context, for C1 and C4.
   Compare generated **token IDs** where exposed, including after a rejected
   draft and across multiple turns. A mismatch needs investigation; floating
   point/batching effects mean a one-off text mismatch is not proof of a
   drafter bug. Repeat with identical runtime conditions and inspect logits or
   the first divergent token if available.
2. Separately test checkpoint sampling (`temperature=1`, `top_p=0.95`,
   `top_k=-1`, fixed prompt/seed pairs, natural EOS). Do not require identical
   sampled text. Require completed, correct final answers with reasoning
   recorded separately, valid JSON without duplicate keys, and valid streamed
   and non-streamed two-turn tool-call/tool-result cycles. Keep incorrect,
   truncated and runtime-error attempts in the denominator.
3. For the same prompt, exercise cold computation, confirmed RAM restore and
   confirmed filesystem-L2 restore after clearing GPU APC. Check cache-hit
   counters and that the answer still completes. Repeat with an appended user
   turn, an altered-prefix miss control, and C4 independent prompts to detect
   cross-request state mixing. Check NextN MLA/indexer restore and rebuilt
   k-pool tail through continued generation, not merely a successful lookup.
4. For MTP3, record drafts, accepted tokens and acceptance by position alongside
   answers. Cover rejection at early and late draft positions, short/unequal
   request lengths, EOS and context-boundary continuations. Verify that rejected
   proposals do not survive in the next turn's KV/indexer state. An aggregate
   acceptance percentage by itself is not a correctness result.

The focused unattended matrix proves only its declared exact-key NIAH and cache
gates. `bench/glm_quality.py` supports exact/JSON completed-answer probes;
`bench/sglang_glm_contract.py` covers the bounded tool/stream contract. Broader
coding-task correctness and target-only greedy equivalence still require their
own frozen corpus/transcripts. The known SGLang sampling/LMCache behavior is not
evidence that this pinned vLLM MTP3 combination is correct. The bounded 8K/32K
gate passed on the original image; the 128K external-restore gate failed.

### 2026-10-17 diagnostic status

The original pinned image passed bounded 8K/32K MTP3 + LMCache quality probes,
then failed the frozen 128K warm NIAH gate: 130176 external cache-hit tokens,
but 8192 generated tokens exhausted the budget without a correct final answer.
Keep that failed attempt in `bench/results/glm-mtp3-matched-20261017T0126/`;
its throughput is not a qualified MTP3 baseline. MTP3 with GPU APC only passed
the same 128K NIAH cold/warm check. Target-only with LMCache also passed cold
128K and a forced external restore after clearing GPU APC: 130176 external hits,
zero GPU-prefix hits and the exact final key. Evidence is in
`bench/results/glm-mtp3-gpu-20261017T010752Z/` (ignored by Git).

This isolates the reproducible failure to the MTP/external-cache combination,
without proving its internal cause. LMCache's upstream fix
[#5004](https://github.com/LMCache/LMCache/pull/5004) addresses an align-mode
Mamba speculative-block relocation omitted from this pinned LMCache 0.5.5.
`docker/patches/lmcache-0.5.5-mtp-relocated-blocks.patch` backports only that
tracker update into a separately tagged image built with
`docker/Dockerfile.glm-lmcache-mtp`. The original K4 Dockerfile is unchanged.
The patched image passed five consecutive 128K cold/warm NIAH pairs (10/10
strict, completed answers); every warm request had 129024 external-hit tokens
and zero GPU-prefix hits. The bounded answer/stream/two-turn tool contract,
forced JSON, C4 independent answers and filesystem-only restore also passed.
The K4 image and cache namespace remain the rollback.

### 2026-10-17 focused comparison

Both legs completed the frozen `glm-mtp3-k4.json` protocol: 130816 actual input
tokens, C1/C4, high thinking, checkpoint sampling, 256 fixed output tokens,
one excluded warmup and three measured batches per cold/RAM/filesystem state.
Each also passed 15/15 strict completed-answer NIAH requests. All warm C4
batches had overlapping decode. Cold C4 did not, so its decode rate is not a
qualified C4 decode comparison. Loaded PCIe links and power limits passed the
preflight in both runs. The raw attempts and diagnostic comparison are in
`bench/results/glm-mtp3-patched-matched-20261017T024626Z/`,
`bench/results/glm-k4-matched-20261017T033614Z/`, and
`bench/results/glm-mtp3-k4-20261017-comparison.json` (all Git-ignored).

| State | K4 decode median tok/s | MTP3 decode median tok/s | K4 TTFT median | MTP3 TTFT median |
|---|---:|---:|---:|---:|
| C1 cold | 61.1 | 49.1 | 79.2 s | 90.1 s |
| C1 RAM | 49.6 | 45.1 | 1.48 s | 1.93 s |
| C1 filesystem | 49.6 | 60.5 | 1.89 s | 2.22 s |
| C4 RAM | 24.3 | 22.8 | 3.24 s | 4.40 s |
| C4 filesystem | 25.2 | 22.0 | 4.51 s | 4.72 s |

C1 filesystem is the sole decode cell favoring MTP3; its three requests ranged
from 50.4 to 71.2 tok/s, and TTFT was worse. The fixed-output test does not
measure a completed coding/tool task. K4 remains the qualified default; this
comparison does not justify promoting MTP3. The MTP3 image differs from K4 by
the LMCache speculative-block relocation backport. The comparator explicitly
labels this a **matched-protocol image-variant diagnostic**: observed differences
apply to the deployed configurations, not to the drafter alone.

The existing `bench/baseline-earkita/glm.json` specifies a much larger matrix.
This focused comparison is a **new protocol**; it cannot be compared with old
K4 logs or presented as a run of that full baseline. Both variants must be
remeasured after the protocol is frozen. Rollback is the unchanged K4 launcher,
image, cache container and namespace; never clear or reuse its persisted data.

The unattended focused matrix is
[`glm-mtp3-k4.json`](../../../bench/baseline-earkita/glm-mtp3-k4.json):
C1/C4 × 130816 tokens × high × cold/RAM/filesystem, with one excluded warmup,
three measured batches and a separate natural-EOS NIAH batch per cell. That is
24 performance and 6 quality batches **per variant**. The tool/JSON/streaming
checks remain separate gates; this matrix alone cannot qualify them. It accepts
serial cold C4 prefill as a cache diagnostic but marks its decode as
unqualified if four requests do not decode simultaneously. Warm C4 requires
overlap. The runner verifies the exact model length, GPU KV and cache budgets,
speculative configuration, tokenizer output, full external cache hits and
loaded PCIe links. For NIAH, the focused variant requires the exact completed
final key; finding it only in reasoning or in a longer explanation is recorded
but does not pass quality. It never starts or replaces a service.

For each **exclusive** serving session, supply the same frozen corpus to the
unattended runner. The K4 leg uses `SPECULATIVE_PROFILE=dflash-k4` and the
candidate uses `SPECULATIVE_PROFILE=mtp3`; set `RUN_DIR`, container names,
`BASE_URL`, `CACHE_METRICS_URL` and `CACHE_PORT` to that leg's dedicated
loopback deployment. Both model launches need `MAXLEN=262144`,
`VLLM_SERVER_DEV_MODE=1`, 4.125 GiB/GPU KV and 128/512 GiB cache. For example,
after the MTP3 service has passed its bounded correctness smoke test:

```bash
PROTOCOL_TEMPLATE=glm-mtp3-k4.json SPECULATIVE_PROFILE=mtp3 \
FROZEN_CORPUS="$PWD/bench/results/glm-mtp3-k4-20261017-prep/corpus/corpus.json" \
MODEL_CONTAINER=glm53-mtp3-bench CACHE_CONTAINER=glm53-mtp3-cache-bench \
BASE_URL=http://127.0.0.1:8081 CACHE_METRICS_URL=http://127.0.0.1:9092 \
CACHE_PORT=5556 RUN_DIR="$PWD/bench/results/glm-mtp3-matched-$(date -u +%Y%m%dT%H%M%SZ)" \
bash bench/baseline-earkita/glm.sh
```

Do not start this while K4 occupies the eight GPUs. Use the identical template,
corpus and hardware conditions for the K4 leg in its own fresh results folder;
only the named speculative profile and its necessary drafter/cache integration
may differ. Preserve failed and warmup attempts. The original-image MTP3 leg
failed its 128K quality gate before C4; the patched-image pair above completed.

After **both** legs finish, compare their completed result directories offline:

```bash
python3 bench/baseline-earkita/compare.py \
  --baseline bench/results/<k4-run> --candidate bench/results/<mtp3-run> \
  --image-diff-reason 'MTP3 needs the LMCache #5004 relocation backport' \
  --out bench/results/<comparison>/matched-core.json
```

The comparator requires matching frozen protocol, corpus, source,
checkpoint mount, PCIe topology, power caps, all measured repeats, exact final
NIAH answers and warm-C4 decode overlap. It reports per-request decode and
TTFT, aggregate output rates and acceptance separately. Cold C4 without decode
overlap is retained as a cache/TTFT diagnostic with no qualified decode change.
Different images are rejected unless the operator supplies an explicit reason;
such a report remains diagnostic and records both image IDs.
Even a passing comparison is labelled **diagnostic** until matched completed
coding/tool tasks and their outputs establish end-to-end correctness and time.

The pre-inference frozen corpus, tokenizer hashes, profile/client hashes and
comparison settings are in ignored
`bench/results/glm-mtp3-k4-20261017-prep/protocol.json` and `corpus/`.
The earlier preparation records are retained; the focused runner/template
source hashes and the current comparison contract are recorded separately in
`protocol-v3.json` in the same ignored directory. A read-only K4 HIP/BDF and
idle topology snapshot is there too; it is **not** loaded-link proof for a
comparison leg. Each executed leg also
freezes its own full runtime identity and code hashes at `prepare` time. Any
change to a hashed script or checkpoint requires a fresh preparation and both
variants must be rerun; it does not retroactively change earlier records.
