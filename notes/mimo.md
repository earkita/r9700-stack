# MiMo-V2.6-Flash-MOPD on 8× R9700

The opt-in MiMo integration uses the original Xiaomi MXFP4 checkpoint and its
BF16 DFlash drafter. The plugin routes native MXFP4 experts to existing
MXFP4×FP8 kernels; it does not convert the model to GLM W4A4. Dense FP8 remains
handled by vLLM. No target or draft weights are offloaded.

## Build and launch

All three profiles call the common `serve.sh` directly:

| Profile | Active requests | Single-request limit | Shared KV/GPU | Communication |
|---|---:|---|---|---|
| `mimo-v2.6-flash-c2.sh` | 2 | auto | 7.625 GiB | Exact AR8 + RCCL |
| `mimo-v2.6-flash-c4.sh` | 4 | auto | 7.625 GiB | Exact AR8 + RCCL |
| `mimo-v2.6-flash.sh` | 2 | 131072 | Automatic | RCCL |

```bash
docker build -f docker/Dockerfile -t r9700/vllm:mimo-e97573215 .
bash serve/mimo-v2.6-flash-c4.sh DRYRUN=1
# After gracefully stopping the previous GPU server:
bash serve/mimo-v2.6-flash-c4.sh
```

The Dockerfile pins vLLM `e9757321527ca1ecd514c07c1418dd2c53da3d19` on ROCm 10.
The profile expects `$HOME/models/mimo/MiMo-V2.6-Flash-MOPD`; override
`MODELS_DIR` for another host. The `dflash` subdirectory must be present.
A CPU preflight validates the index, tensor headers, QKV/expert shapes and draft
configuration before launching. Stop the previous GPU server gracefully first;
the profile does not replace containers or stop another model automatically.

The fallback `mimo-v2.6-flash.sh` delegates to the common `serve.sh`. Defaults: TP8, C2, 131072 total
tokens per request, NBT 8192, utilization 0.94, BF16 target/draft KV, APC, eager,
RCCL, original scales (`R9K_FOLD=0`) and DFlash K7. `SPEC=0` selects the control;
`MAXLEN=32768` is the initial bring-up limit. Text and tools only. Configured
context and actual concurrent capacity are separate claims.

For a shared, explicitly sized cache, use `serve/mimo-v2.6-flash-c2.sh` or
`serve/mimo-v2.6-flash-c4.sh`. Each independently calls `serve.sh`; neither
chains through another model profile. They default to two/four active requests,
`MAXLEN=auto`, 7.625 GiB KV per GPU and exact AR8. DFlash K7, BF16 KV and eager
execution remain enabled. `auto` determines the maximum **single** request;
all active requests share the pool, so that limit is not reserved for each
session. `KVMEM=` restores vLLM's automatic sizing. The Claude template and main
proxy alias declare 512K; the fast alias continues to declare 128K.

The adapter uses the unique `r9700_mimo_mxfp4` quantization registration. It
requires explicit `R9K_MIMO=1`, MiMo, gfx1201 and the audited vLLM version.
It leaves the standard `fp8` config and GLM/Qwen kernels unchanged. MiMo cache
handling preserves nine global and 39 sliding-window layers. The isolated
BF16 DiffKV backend adds split-KV speculative verification; FP8 KV, graphs,
EP, multimodal support and 1M qualification are outside this profile's scope.

`AR8=1` enables the existing exact TP8 P2P all-reduce: one-shot through 16 KiB,
two-shot through 256 KiB, and RCCL for larger messages. `AR8=0` (the fallback
profile's default) keeps RCCL throughout. This switch uses the plugin
communicator; vLLM's own custom all-reduce remains disabled. Compressed AR4
is disabled in both variants.
Confirm all eight ranks log `8-rank P2P all-reduce installed` before measuring;
the communicator can fall back to RCCL if initialization fails.

```bash
bash serve/mimo-v2.6-flash.sh AR8=1
```

The pinned vLLM already supplies TP4→TP8 QKV sharding and the MiMo-compatible
`DFlashDraftModel` (sink bias, value scaling and sliding attention). Those paths
are tested and reused rather than copied into another model implementation.

## Gateway and Claude Code

With the model API ready, stop the previous proxy and start the MiMo config:

```bash
bash serve/litellm.sh HOST=0.0.0.0 CONFIG="$PWD/proxy/mimo.yaml" \
  BACKEND_MODEL=mimo-v2.6-flash-mopd
claude --settings serve/templates/mimo-v2.6-flash.settings.local.json
```

The MiMo template's key helper uses `LITELLM_MASTER_KEY` when supplied, otherwise
reads `secrets/litellm.env` relative to the helper's checkout. No `source` step
is needed, including when Claude runs in another project. The template locates
the helper under `$HOME/ai/r9700-stack`; set `R9700_STACK_ROOT` for another checkout
location, or `R9700_LITELLM_ENV_FILE` for a non-default credentials file. Restart
Claude with the updated template; copied project settings need the updated
`apiKeyHelper` value too. A remote client needs its own helper checkout/key or
an environment-based helper; the server's credentials file is not available remotely.

The aliases `mimo-v2.6-flash` and `mimo-v2.6-flash-fast` target the same GPU
server with thinking enabled/disabled respectively. The Claude template routes
Haiku/Small Fast to the latter. The template's context and compaction window are
524288 tokens (512K), with compaction at 90%. The main proxy alias declares
524288 input tokens; the fast alias declares 131072. These metadata values do
not reserve KV or enforce a combined input-plus-output budget. Leave room for
thinking and output within each session's intended total context budget.
The template's context override is client-wide, not a separate per-subagent
policy. A helper using the main alias shares its declaration; use the fast
alias and a 128K harness/session budget for smaller helpers. This does not
automatically enforce one 512K agent plus three 128K agents.
Restart Claude Code with this template (or update the project's copied
`.claude/settings.local.json`) to pick up the new client setting. The 128K
fallback server needs matching smaller client/proxy declarations if restored.
Use the server's LAN address in the template when connecting from another host.
The native Anthropic route also accepts OpenAI clients through LiteLLM. A
MiMo-only adapter preserves the alias's thinking flag in native Messages and
token-count requests to vLLM; it never rewrites responses or tool streams. Never point GLM-named
aliases at MiMo. To restore GLM, start its saved container
and the GLM proxy config after stopping MiMo.

## Validation

Run the profile tests on CPU and `tests/test_mimo_gpu.py` in the built ROCm
image with one free GPU and the repository mounted at `/opt/r9700`. GPU gates
cover TP4→TP8 QKV, native MoE shapes, and BF16 DiffKV across global/SWA,
sinks, C1/C2/C4, K7 verification up to 32 rows and unequal/mixed query lengths.

`bench/mimo_validate.py --freeze --model-path MODEL --out DIR` creates one
exact-token corpus and protocol using the checkpoint tokenizer. Its `smoke`,
`bench`, `quality` and `apc` phases write every attempt to a new output directory.
The benchmark requires BetterBench 0.6.0 and a `--corpus DIR/corpus.json` path.
`--bench-lengths 1024 32768` selects a bounded comparison, and
`--bench-cache-namespace NAME` avoids prefixes left by an earlier run. Freeze
these options for both legs; do not compare a subset with a different protocol.
Use `--concurrency 1 2 4` both when freezing a C4 corpus and when measuring it;
`--quality-lengths 8192` selects a bounded, separately budgeted retrieval probe.
The server independently verifies token counts. The runner checks C1/C2 APC
cold/warm counters and independent C2 retrieval keys. Compare only matched runtime
settings and frozen workloads. Prefix salts distinguish cold repetitions.

Quality sampling is temperature 1/top-p 0.95, explicitly enabled thinking,
no top-k restriction, with EOS respected. NIAH reports strict answer, retrieval,
format, completion and runtime status separately. Speed samples have a 256-token
budget and do not qualify answer quality. Decode excludes TTFT, includes
reasoning tokens and uses actual completion usage; C2 also reports overlapping
decode intervals. Cached prefill is reported separately.

Results belong in ignored `bench/results/`, not tracked baseline dumps.

## Local qualification, 2026-10-03

8× R9700 at 225 W, TP8/C2, eager, BF16 KV, identical cold prompts and
BetterBench 0.6.0 protocol. Each cell has one excluded warmup and three measured
rounds (36 requests per variant including warmups). Thinking was enabled at
T=1, top-p=0.95, top-k disabled, seed 1234. All speed responses reached the
256-token limit with EOS enabled; they are **truncated performance samples**.

| Input tokens | Active requests | Control decode tok/s | K7 decode tok/s | K7 TTFT s | K7 effective prefill tok/s | K7 aggregate E2E tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 14.3–16.5 | 37.2–39.3 | 0.47–0.49 | 2081–2175 | 36.03 |
| 1024 | 2 | 13.7–14.7 | 32.5–48.2 | 0.82–0.89 | 1152–1255 | 61.24 |
| 32768 | 1 | 15.0–15.4 | 35.4–40.1 | 14.5–14.7 | 2234–2258 | 12.05 |
| 32768 | 2 | 8.8–15.3 | 18.4–51.0 | 21.2–28.9 | 1132–1546 | 14.82 |
| 130816 | 1 | 13.9–15.5 | 47.3–50.3 | 74.1 | 1764–1765 | 3.23 |
| 130816 | 2 | 2.9–14.8 | 3.4–55.5 | 77.4–148.6 | 880–1690 | 3.34 |

Decode, TTFT and prefill columns show per-request ranges; aggregate E2E is the
mean batch output divided by batch wall time, including prefill. Decode uses
actual completion usage and excludes TTFT. K7 SSE chunks contain multiple
tokens; stream gaps are not individual GPU step times. At C2, decoding the
first request stalls during the other's cold prefill: the 128K range does
**not** mean both sessions continuously generate at 55 tok/s. Every measured
C2 batch had overlapping decode intervals. Neither variant had preemption or
APC hits in these deliberately cold speed samples.

At 128K/C1, K7 acceptance by position was 87.5%, 62.5%, 33.3%, 23.6%, 19.4%,
15.3%, 12.5%; mean acceptance length including the final target token was 3.54.
These are workload-specific acceptance counters, not a model-quality score.
All per-session timings, per-position counters, hardware samples and attempts
are retained in ignored `bench/results/mimo-integration-20261003/`.

Bounded NIAH results (T=1, thinking, output budget 4096):

| Runtime / input / requests | Strict | Retrieval | Format | Completed | Runtime errors |
|---|---:|---:|---:|---:|---:|
| K7, 8192 / C1+C2 | 3/3 | 3/3 | 3/3 | 3/3 | 0 |
| K7, 32768 / C1+C2 | 3/3 | 3/3 | 3/3 | 3/3 | 0 |
| K7, 126976 / C1+C2 | 3/3 | 3/3 | 3/3 | 3/3 | 0 |
| Control, complete matched rerun | 8/9 | 8/9 | 8/9 | 9/9 | 0 |

The control refused the 8192/C1 request, interpreting the synthetic secret-code
instruction as injection. The initial failed attempt is retained separately
(0/1); it reproduced in the full nine-request cold rerun, using fresh salts.
There was no cross-request key contamination. These are single-seed diagnostic
probes, not a broad accuracy benchmark or evidence that DFlash improves quality.

The first K7 API tool fixture also elicited a completed refusal: its tool result
contained an instruction. The revised fixture moves the instruction to the user
message and makes the tool result factual. Both variants passed all eight
revised API checks. The original failure remains recorded and is not labelled
a backend fault or a matched non-reproduction.

K7 APC at 32768 input tokens: C1 cold/warm hits **0 / 32736**; C2 cold/warm hits
**0 / 65472** across two independent requests. All six APC answers completed
with the expected key. This validates reuse mechanics, not a warm-throughput
benchmark. Both variants also completed two concurrent 130816+256 speed
requests and two concurrent 126976-token retrieval prompts without OOM,
preemption or observed answer mixing.

Final integration: 12/12 proxy contract checks passed after the fix, plus
MiMo-only model listing, exact token counts for both thinking prefixes,
Anthropic fast on/off-stream and OpenAI thinking on/off-stream. LiteLLM 1.103.0
was observed dropping `chat_template_kwargs` from native Messages requests;
`proxy/mimo_hooks.py` narrowly preserves this field for the configured MiMo
backend. The initial failed fast-alias attempt is retained. No response or
tool-stream rewriting was added.

Claude Code 2.1.197 passed a real request through the template and reported a
131072-token context. That client smoke used isolated settings, disabled tools
and a 4096-token output cap; tool-call/result contracts were tested separately
through both APIs. CPU profile/preflight/proxy gates and numerical QKV, MoE and
DiffKV gates passed. The GLM profile regression suite remained green.

The initial qualified local services were `mimo26-flash-c2` on 8080 and `r9700-litellm` on
LAN port 4000. The stopped GLM container/image and its original proxy container
were retained for rollback. No 1M, multimodal, FP8 KV, graph or general accuracy
qualification is implied by these results.

## Exact AR8 experiment (2026-10-03)

A new paired comparison used the same image, checkpoint, vLLM arguments,
DFlash K7, BF16 KV, eager execution and 225 W limits. Only plugin AR8 was
enabled with the thresholds above. The existing frozen prompts were reused
at 1024 and 32768 input tokens, C1/C2, temperature 1, top-p 0.95, thinking on,
EOS enabled, 256 output tokens, one excluded warmup and three measured batches
per cell. Both variants used a fresh cold-cache namespace and BetterBench 0.6.0.
All speed responses reached the output limit; they are throughput samples,
not completed-answer quality scores.

| Input | Concurrency | RCCL mean decode | AR8 mean decode | Change | RCCL → AR8 aggregate E2E |
|---|---:|---:|---:|---:|---:|
| 1K | 1 | 36.53 | 43.05 | +17.8% | 34.35 → 40.00 |
| 1K | 2 | 38.31 | 40.15 | +4.8% | 58.99 → 68.76 |
| 32K | 1 | 38.34 | 39.47 | +3.0% | 12.10 → 12.18 |
| 32K | 2 | 31.59 | 36.03 | +14.0% | 14.67 → 15.09 |

Rates are tokens/s. Decode excludes TTFT and averages individual requests
(three C1 or six C2 samples); aggregate E2E includes both prefill and decode.
At 32K/C2, AR8 sessions ranged from 20.6 to 52.8 tok/s because one request's
decode overlaps another's prefill. The mean is not a guaranteed rate for each
agent. At 32K/C1, ranges overlap (RCCL 37.2–40.2, AR8 37.0–41.7); the small
difference is inconclusive beyond this sample.

Prefill was effectively unchanged: C1 TTFT about 0.47–0.48 s at 1K and
14.5–14.6 s at 32K; C2 about 0.82–0.89 s and 21.2–29.0 s respectively.
Mean DFlash acceptance length, including the bonus token, changed as follows:
1K/C1 2.742→2.954; 1K/C2 2.931→2.872; 32K/C1 2.753→2.802;
32K/C2 3.318→3.109. Per-position acceptance is retained in the report.
These end-to-end observations do not isolate collective kernel latency from
sampling/acceptance variation. Before/after hardware snapshots are retained;
power caps matched, but temperature and clocks were not held constant.

Numerical reduction tests passed on eight ranks for BF16/FP16/FP32, including
8/16-row messages, varying sizes, bursts and graph replay against the FP32
rank-ordered reference. Both servers passed 8/8 answer/API smoke checks
(thinking on/off, streaming and two-turn tools). All benchmark batches had
zero prefix hits and preemptions, with overlapping C2 decode and no observed
runtime errors. The profile's CPU tests passed 5/5.

Evidence is ignored under `bench/results/mimo-ar8-20261003/`: frozen protocol,
corpus, runtime identities, all attempts, numerical logs, hardware snapshots
and `performance-summary.json`. This experiment used `mimo26-flash-ar8-c2`
on 8080; the LiteLLM proxy uses that port independently of container names.
The fallback profile still defaults to `AR8=0`; the shared C2/C4 profiles enable
AR8 explicitly. This comparison did not repeat the full 128K/C2 quality
qualification performed with RCCL.

## KV allocation search (2026-10-03)

The allocation search established an explicit `KVMEM=7.625` GiB per GPU
(61 GiB across eight cards), compared with 6.596 GiB from automatic sizing.
The fallback profile's default remains automatic; reproduce the tested budget with
`bash serve/mimo-v2.6-flash.sh AR8=1 KVMEM=7.625` when replacing the server.
Explicit KV bytes override utilization-based cache sizing. This search kept
MAXLEN=131072 and NSEQ=2; client/proxy declarations were unchanged during that search.
The recreated runtime uses the repository's `.runtime/cache/` mounts.

| KV GiB/GPU | Minimum observed free VRAM/GPU | C2 load requests | Preemptions |
|---:|---:|---:|---:|
| 7.5 | 698 MiB | 4/4 successful | 0 |
| 7.625 | 578 MiB | 4/4 successful | 0 |

Each variant passed the eight API smoke checks and two rounds of independent,
concurrent 130816-token inputs with 256 output tokens. Requests used thinking,
temperature 1, top-p 0.95, seed 1234, EOS enabled and unique cold cache salts;
prefix hits were zero. Both requests had overlapping decode. All load responses
hit the output cap; these test memory/runtime behavior, not completed-answer
quality. No OOM or runtime errors were observed. GPU memory was sampled every
second. This is a capacity probe, not a performance comparison.

The search used a 512 MiB minimum free-VRAM floor and refined the budget in
64 MiB steps. Extrapolating the largest observed non-KV footprint to 7.6875 GiB
leaves only 506 MiB; that step was not run. Thus 7.625 GiB is the tested operating
budget with this reserve and workload, not an absolute hardware maximum or a
guarantee for longer contexts/more active requests.

The pinned allocator reproduces the live baseline exactly: 50 KiB per pool
block, two global-attention groups, eight target SWA groups and one draft SWA
group. A 128K request reserves 21129 blocks at NBT=8192. The new budget gives
159907 blocks and a reported 991969-token capacity (7.568 × 128K). This reported
token count depends on request length; SWA contributes a per-request overhead.

For long requests, the approximate admission cost is
`2 * ceil(tokens / 16) + 4745` pool blocks per session. The following fit the
KV arithmetic; the later shared-profile tests below qualify two of them:

| Possible allocation | Status |
|---|---|
| 1 × 1048576 tokens | Within the checkpoint's 1M positional limit; estimate only |
| 2 × 524288 tokens | Runtime capacity passed with the C2 shared profile below |
| 4 × 262144 tokens | Runtime capacity passed with the C4 shared profile below |

Changing MAXLEN or NSEQ may change non-KV memory requirements and require
reducing the fixed budget. Evidence, allocator reproduction, all attempts and
memory samples are ignored under `bench/results/mimo-kv-capacity-20261003/`.

## Shared C2/C4 profiles (2026-10-03)

Both shared profiles use MAXLEN=auto, resolved to 1048576, and the same fixed
7.625 GiB/GPU KV pool (159907 blocks). Only NSEQ changes between matched C1/C2
cells. TP8, exact AR8, DFlash K7, BF16 KV, eager, NBT8192 and 225 W match.
The single-request limit does not reserve that context for each active request.

Cold performance: exact 1024/32768 input tokens, at most 256 output tokens,
T=1/top-p=0.95/top-k=-1/seed=1234, thinking and EOS enabled. Each cell has one
excluded warmup and three measured rounds. Every response reached the output
cap; these are truncated speed samples, not completed-answer quality tests.

| Profile | Input | Concurrent | Decode mean (range), tok/s/request | TTFT mean, s | Effective prefill, tok/s/request | Aggregate E2E, tok/s | Acceptance length |
|---|---:|---:|---:|---:|---:|---:|---:|
| C2 | 1024 | 1 | 44.18 (41.83–46.36) | 0.47 | 2157 | 40.97 | 2.95 |
| C2 | 1024 | 2 | 43.96 (39.37–48.87) | 0.85 | 1208 | 73.98 | 2.92 |
| C2 | 32768 | 1 | 44.95 (40.43–48.19) | 14.55 | 2253 | 12.65 | 2.80 |
| C2 | 32768 | 2 | 34.92 (18.67–54.35) | 25.10 | 1337 | 14.90 | 3.01 |
| C4 | 1024 | 1 | 49.38 (46.82–53.77) | 0.47 | 2188 | 45.44 | 2.95 |
| C4 | 1024 | 2 | 44.51 (35.98–50.14) | 0.85 | 1211 | 75.18 | 2.87 |
| C4 | 1024 | 4 | 44.16 (40.16–49.07) | 1.62 | 634 | 131.69 | 2.70 |
| C4 | 32768 | 1 | 41.28 (37.19–49.02) | 14.53 | 2255 | 12.32 | 2.80 |
| C4 | 32768 | 2 | 33.10 (18.07–50.74) | 25.09 | 1338 | 14.76 | 3.01 |
| C4 | 32768 | 4 | 21.11 (5.99–55.94) | 41.08 | 923 | 16.25 | 3.09 |

Decode excludes TTFT and includes thinking; it still includes stalls caused by
other requests' cold prefills. Effective prefill is input/TTFT, not kernel-only
throughput. Aggregate E2E includes prefill. Acceptance length includes the final
target token; per-position rates and every request are in the ignored report.
All concurrent batches had overlapping decode, zero preemptions and zero APC
hits. Three rounds do not establish a lasting performance improvement between
profiles; GPU temperatures/clocks and stochastic acceptance were not held constant.

API/thinking/streaming/two-turn tools passed 8/8 on each profile. At 8192 input
tokens, C2 NIAH passed strict/retrieval/format 2/3, completion 3/3, runtime errors
0; a full matched rerun reproduced the same C2 request refusal (2/3 again).
C4 passed strict/retrieval/format 6/7, completion 7/7, runtime errors 0. Its four
concurrent requests passed all four answer checks; the same C2 refusal accounts
for the failure. No cross-request key contamination was observed. The model
explicitly treated the synthetic secret-code instruction as prompt injection.
The original failures are preserved; this is not attributed to cache, kernels
or DFlash without additional causal evidence.

CPU profile checks: 6 MiMo and 10 GLM tests passed. GPU QKV, DiffKV and native
MoE tests passed, including C4 unequal query lengths and 32-row verification.
Evidence: `bench/results/mimo-c2-c4-20261003/` contains the frozen protocol and
corpora, runtime identities, all attempts, `benchmark-summary.json` (including
acceptance by position), and capacity telemetry.

Capacity was separately tested with one cold batch per profile, the same
sampling/thinking settings and an output cap of 2048 (EOS enabled). Each C2
input had exactly 522240 tokens; each C4 input had 260096. Every response used
2048 output tokens and stopped at the cap. This proves runtime/memory capacity
for these loads, not completed-answer quality at these long contexts.

| Profile | Tested input + output per request | Requests | Shared decode interval | Minimum free VRAM/GPU | TTFT range |
|---|---:|---:|---:|---:|---:|
| C2 | 522240 + 2048 | 2/2 | 23.37 s | 566 MiB | 9.82–19.53 min |
| C4 | 260096 + 2048 | 4/4 | 20.82 s | 556 MiB | 3.28–12.87 min |

Both loads had zero preemptions, zero prefix hits and no observed OOM/runtime
errors. The same 7.625 GiB/GPU budget remained usable at MAXLEN=auto and NSEQ=4.
Full cache allocation does not guarantee low cold-start latency: successive
long prefills delay existing decodes. Long-load per-request rates, timing and
acceptance counters are in `capacity-summary.json`; they are not comparable to
the repeated 256-output-token speed series above. A single 1M request and
long-context answer quality remain unqualified by these capacity tests.

This validation used `mimo26-flash-c4` on 8080 and LiteLLM on LAN port 4000;
it is a historical result, not live service status. C2 and the original
128K/RCCL fallback remain available. The client/proxy declarations are 512K
for the main alias and 128K for fast; per-session policy is separate from
this shared-pool measurement.
