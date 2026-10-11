# MiMo-V2.6-Flash 128K/RCCL fallback

[Profile](mimo-v2.6-flash.sh) · [Shared configuration](README.md) · [Testing](testing.md)

Defaults: TP8, two active requests, 131072-token total request limit, automatic
KV sizing at utilization 0.94, NBT8192, DFlash K7, BF16 target/draft KV, eager,
APC and RCCL. Text/tools only; no EP, vision or weight offload. `SPEC=0` selects
the no-drafter control; `AR8=1` enables the plugin communicator.

```bash
bash serve/baseline-earkita/mimo/mimo-v2.6-flash.sh DRYRUN=1
bash serve/baseline-earkita/mimo/mimo-v2.6-flash.sh
```

This directly calls `serve/serve.sh`. The 512K client template and main proxy
alias require smaller matching declarations when using this 128K fallback.
The following measurements retain their original runtime overrides; they are
historical evidence, not current service status.

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
enabled with the thresholds documented in [the shared configuration](README.md). The existing frozen prompts were reused
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
`bash serve/baseline-earkita/mimo/mimo-v2.6-flash.sh AR8=1 KVMEM=7.625` when replacing the server.
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
KV arithmetic; the [shared-profile tests](mimo-v2.6-flash-c4.md) qualify two of them:

| Possible allocation | Status |
|---|---|
| 1 × 1048576 tokens | Within the checkpoint's 1M positional limit; estimate only |
| 2 × 524288 tokens | Runtime capacity passed with the [C2 shared profile](mimo-v2.6-flash-c2.md) |
| 4 × 262144 tokens | Runtime capacity passed with the [C4 shared profile](mimo-v2.6-flash-c4.md) |

Changing MAXLEN or NSEQ may change non-KV memory requirements and require
reducing the fixed budget. Evidence, allocator reproduction, all attempts and
memory samples are ignored under `bench/results/mimo-kv-capacity-20261003/`.
