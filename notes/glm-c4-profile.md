# GLM C4 profile: C1/C2/C4 decode — 2026-10-02

Historical **text-only, NBT1024** measurements. The maintained C2/C4 profiles
now enable vision with NBT2048; their current settings and separate validation
are in the [profile overview](glm-overview.md). These results are not a matched
performance comparison with the current vision configuration.

## Configuration at measurement time

Container `glm53-flash-c4`, port 8080, NSEQ=4. All eight GPUs were checked at
225 W. Image
`r9700/vllm:glm53-plugin-e97573215`, Quark MXFP4 target with packed W4A4,
BF16 DFlash2 weights, K4, TP8 for target/drafter, NBT1024,
FULL_DECODE_ONLY, FP8 E4M3 shared target/draft KV, APC enabled.

KV budget remains 4.125 GiB per GPU. `MAXLEN=auto` resolves to **702720**;
startup reports **703817 KV tokens**. This is a shared capacity, not four
reserved full-length contexts. The indexer log confirms max_num_seqs=4 and
index_kpool=4, with 1048576 workspace entries allocated against the initial
1M model limit instead of the upstream 41943040 entries.

## Benchmark

All concurrency levels were measured on this same NSEQ=4 server. BetterBench
0.6.0 through `bench/glm_concurrent.py`, frozen distinct NIAH cases at depths
5/35/65/95 as concurrency increases. Each request has exactly **8192 input
tokens and 256 output tokens**, including thinking, temperature 1, top_p .95,
top_k -1, seed 1234, reasoning_effort high. Forced output uses ignore_eos.

Each concurrency leg first runs natural-EOS correctness checks with a
1024-token output budget, then reuses those prefixes for performance. Salts
are separate across cases and concurrency legs. One warmup batch is excluded;
three batches are measured (3/6/12 requests for C1/C2/C4).

| Active requests | Mean decode per request | Minimum decode | Maximum decode | TTFT mean / p95 | End-to-end p95 | Aggregate output including TTFT |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 56.84 tok/s | 49.92 | 69.14 | 0.435 / 0.440 s | 5.540 s | 51.91 tok/s |
| 2 | 29.77 tok/s | 27.81 | 32.21 | 0.661 / 0.890 s | 10.051 s | 53.48 tok/s |
| 4 | 19.67 tok/s | 17.17 | 24.78 | 0.908 / 1.068 s | 15.903 s | 68.92 tok/s |

Decode is `(completion_tokens-1)/(last SSE update-first SSE update)` and
excludes TTFT. The first speculative update is approximated as one token.
Aggregate output is total completion tokens divided by batch wall time,
including TTFT; it is not the sum of per-request decode rates. Small-sample
p95 uses nearest rank. No uncached prefill rate is inferred from warm APC.

Each measured request hit **8064 cached tokens**. Peak active request counts
were 1/2/4 in every corresponding measured batch. All decode intervals overlap:
minimum intersection 7.46 s for C2 and 10.29 s for C4. The small uncached tail
can still cause brief prefill/decode interference. Acceptance was 44.6%,
33.5% and 28.7%, respectively; generated content and the expanded prompt set
also affect these rates. This is a bounded synthetic workload, not a universal
agent speed or a direct comparison with earlier cold-cache measurements.

## Correctness and checks

| Check | Result |
|---|---:|
| Strict exact-match | 7/7 |
| Semantic needle retrieval | 7/7 |
| Output-format compliance | 7/7 |
| Generation completed | 7/7 |
| Cross-request foreign keys | 0/7 |
| Backend/runtime errors | 0 |

The separate T=0 API gate passed streamed/non-streamed responses, arithmetic
answer 23, tool calls and exact 128-token generation. All performance requests,
including warmups, completed the exact 256-token budget with finish_reason
length. No preemptions, OOM or traceback. Minimum sampled free VRAM was 748
in amd-smi's MB units. Final health returned 200. This validates the short
workload, not the full auto context limit. Shell syntax checks and C2/C4 dry
runs passed; the container argument diff contains only NSEQ and graph sizes.

Evidence: `bench/results/glm-c4-profile/`: `run.py`, `run.log`, dry runs,
power/source/corpus hashes, before/candidate/final container snapshots,
`models.json`, API transcript, all quality/performance rounds and telemetry,
`runtime.log`, `report.py`, `summary.json`, and final metrics.

## Upstream decode investigation — 2026-10-03

Compared pre-sync `f7187798` with post-sync `8238b5b3`, using the same image,
weights, C4 arguments/environment and 225 W caps. The active upstream change
identified for this GLM configuration is the fixed-grid native all-reduce;
the GLM kernels/adapters/profiles are unchanged. GDN, PLE and offloaded-expert
changes are not active here. No production tuning or rollback was retained.

For C1, each version ran three independent blocks of the historical NIAH
protocol above (quality preparation, one excluded warmup, three measured
requests): nine timed requests per version/leg. All natural-EOS quality probes
passed strict/retrieval/format/completion (9/9 across the three legs), with no
backend errors. Timed requests hit 8064 cached tokens with no preemptions.

| Execution order | Mean decode | Range | DFlash acceptance | Approx. ms/draft step |
|---|---:|---:|---:|---:|
| New A1 | 44.20 tok/s | 34.57–50.41 | 29.89% | 50.16 |
| Old B | 47.23 tok/s | 37.38–56.72 | 33.39% | 50.06 |
| New A2 | 41.64 tok/s | 34.54–49.43 | 26.80% | 50.16 |

The old version's higher point estimate remains an observation, not a proven
cause. Captured transcripts expose a limitation: natural NIAH answers use just
16 tokens, while the speed test forces 256 with `ignore_eos=true`. Most timed
generation is artificial continuation; all nine timed texts differ within each
leg despite a fixed seed. Historical raw results did not retain these texts.
Acceptance differs, but these measurements do not isolate why it differs.

A separately frozen long-task control respects EOS and measures the first
256 tokens at the same 8192-token input, sampling and effort. Each leg has two
excluded warmups and five measured requests. All responses reached the token
limit while thinking; this is throughput evidence, not completed-code quality.

| Execution order | Mean decode | Range | DFlash acceptance | Approx. ms/draft step |
|---|---:|---:|---:|---:|
| New, initial observation | 50.03 tok/s | 48.06–52.07 | 38.27% | 50.51 |
| Old, after restart | 51.86 tok/s | 48.52–57.36 | 40.18% | 50.51 |
| New, after restart | 59.51 tok/s | 47.64–73.25 | 48.34% | 50.49 |

The primary long-task comparison is the last two legs: both after restart,
with the same hardware observer. All attempts and exclusions are retained;
the return-to-new leg was declared before observing old results. APC hits and
token counts matched, with no preemptions. Sampled hotspot peaked at 55 C;
AMD's nonspecific throttle flag was often set under the 225 W power cap, so
this is not evidence of unthrottled operation.

An isolated TP8 native all-reduce comparison used identical inputs for the
old implementation and new fixed grids 16/2: BF16/FP32, 8/16 KiB, 20 seeded
input sets per case, including graph replay. All outputs matched the FP32
rank-order reference bit for bit. Forward/reverse timing passes found similar
latencies (roughly 20–21, 36, 30–31 and 58–59 microseconds respectively).
This does not prove equivalence for every runtime input or message pattern.

These bounded tests do not establish a persistent upstream-caused 25% decode
regression. Throughput and acceptance vary substantially even within the new
version, while counter-derived time per draft step stays similar. That estimate
is not a direct kernel measurement. A smaller or workload-specific regression
is still possible; the source of generation/acceptance variability remains
unresolved. No upstream change is justified for rollback by this evidence.

Evidence: `bench/results/glm-decode-upstream-aba-20261003/`, including frozen
`plan.json`, all transcripts, `summary.json`, `ar-comparison.json`, and
`long-answer-control/{plan,summary}.json`. Current C4 was restored and its
health check passed. Temporary control runtime/worktree were removed.
