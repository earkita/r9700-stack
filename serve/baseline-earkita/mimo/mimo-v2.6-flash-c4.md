# MiMo-V2.6-Flash C4

[Profile](mimo-v2.6-flash-c4.sh) · [Shared configuration](README.md) · [Testing](testing.md)

TP8, 4 active requests, `MAXLEN=auto`, shared **7.625 GiB KV per GPU**,
DFlash K7, BF16 target/draft KV, eager, APC, NBT8192 and plugin exact AR8
(up to 256 KiB, RCCL above). Text/tools only; no EP, vision or weight offload.
`KVMEM=` restores automatic sizing. The per-request context limit does not
reserve that many tokens for each session.

```bash
bash serve/baseline-earkita/mimo/mimo-v2.6-flash-c4.sh DRYRUN=1
bash serve/baseline-earkita/mimo/mimo-v2.6-flash-c4.sh
```

This independently calls `serve/serve.sh`, without chaining model profiles.

## Historical shared C2/C4 qualification (2026-10-03)

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
