# GLM C4 profile: C1/C2/C4 decode — 2026-10-02

`serve/glm-5.3-flash-max.sh` was renamed to `serve/glm-5.3-flash-c2.sh`.
`serve/glm-5.3-flash-c4.sh` is a standalone copy with default NSEQ=4 and
container name `glm53-flash-c4`. Both call the shared `serve.sh` directly.
C2 retains its previous settings; C4 changes only NSEQ and the corresponding
graph sizes from 5/10 to 5/10/15/20. The bounded indexer workspace follows NSEQ
and index_kpool=4 automatically. README and runnable examples were updated.

## Configuration at measurement time

At completion on 2026-10-02, container `glm53-flash-c4` was left running on
port 8080. This report is not a live service-status record; see the
[current profile overview](glm-overview.md) for maintained settings. Previous
`glm53-flash-max` was stopped with SIGINT and preserved as a fallback.
All eight GPUs were checked at 225 W. Image
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
