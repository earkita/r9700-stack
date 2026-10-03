# MiMo-V2.6-Flash-MOPD on 8× R9700

The opt-in MiMo integration uses the original Xiaomi MXFP4 checkpoint and its
BF16 DFlash drafter. The plugin routes native MXFP4 experts to existing
MXFP4×FP8 kernels; it does not convert the model to GLM W4A4. Dense FP8 remains
handled by vLLM. No target or draft weights are offloaded.

## Build and launch

```bash
docker build -f docker/Dockerfile -t r9700/vllm:mimo-e97573215 .
bash serve/mimo-v2.6-flash.sh
```

The Dockerfile pins vLLM `e9757321527ca1ecd514c07c1418dd2c53da3d19` on ROCm 10.
The profile expects `$HOME/models/mimo/MiMo-V2.6-Flash-MOPD`; override
`MODELS_DIR` for another host. The `dflash` subdirectory must be present.
A CPU preflight validates the index, tensor headers, QKV/expert shapes and draft
configuration before launching. Stop the previous GPU server gracefully first;
the profile does not replace containers or stop another model automatically.

One profile delegates to the common `serve.sh`. Defaults: TP8, C2, 131072 total
tokens per request, NBT 8192, utilization 0.94, BF16 target/draft KV, APC, eager,
RCCL, original scales (`R9K_FOLD=0`) and DFlash K7. `SPEC=0` selects the control;
`MAXLEN=32768` is the initial bring-up limit. Text and tools only. Configured
context and actual concurrent capacity are separate claims.

The adapter uses the unique `r9700_mimo_mxfp4` quantization registration. It
requires explicit `R9K_MIMO=1`, MiMo, gfx1201 and the audited vLLM version.
It leaves the standard `fp8` config and GLM/Qwen kernels unchanged. MiMo cache
handling preserves nine global and 39 sliding-window layers. The isolated
BF16 DiffKV backend adds split-KV speculative verification; FP8 KV, graphs,
EP, multimodal support and 1M qualification are outside this profile's scope.

`AR8=1` enables the existing exact TP8 P2P all-reduce: one-shot through 16 KiB,
two-shot through 256 KiB, and RCCL for larger messages. `AR8=0` (default) keeps
RCCL throughout. This switch uses the plugin communicator; vLLM's own custom
all-reduce remains disabled. Compressed AR4 is disabled in both variants.
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
set -a
. ./secrets/litellm.env
set +a
claude --settings serve/templates/mimo-v2.6-flash.settings.local.json
```

The aliases `mimo-v2.6-flash` and `mimo-v2.6-flash-fast` target the same GPU
server with thinking enabled/disabled respectively. The Claude template routes
Haiku/Small Fast to the latter. Its context is 128K with compaction at 90%.
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
sinks, C1/C2, K7 verification and unequal/mixed query lengths.

`bench/mimo_validate.py --freeze --model-path MODEL --out DIR` creates one
exact-token corpus and protocol using the checkpoint tokenizer. Its `smoke`,
`bench`, `quality` and `apc` phases write every attempt to a new output directory.
The benchmark requires BetterBench 0.6.0 and a `--corpus DIR/corpus.json` path.
`--bench-lengths 1024 32768` selects a bounded comparison, and
`--bench-cache-namespace NAME` avoids prefixes left by an earlier run. Freeze
these options for both legs; do not compare a subset with a different protocol.
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
and `performance-summary.json`. The AR8 run remains active as
`mimo26-flash-ar8-c2` on 8080; the unchanged LiteLLM proxy uses it on port 4000.
The stopped `mimo26-flash-c2` container preserves the RCCL configuration.
The profile still defaults to `AR8=0`; opt in explicitly for this experiment.
AR8 has not repeated the full 128K/C2 qualification performed with RCCL.
