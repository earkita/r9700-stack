# MiMo-V2.6-Flash C2

[Profile](mimo-v2.6-flash-c2.sh) · [Shared configuration](README.md) · [Testing](testing.md)

TP8, 2 active requests, `MAXLEN=auto`, shared **7.625 GiB KV per GPU**,
DFlash K7, BF16 target/draft KV, eager, APC, NBT8192 and plugin exact AR8
(up to 256 KiB, RCCL above). Text/tools only; no EP, vision or weight offload.
`KVMEM=` restores automatic sizing. The per-request context limit does not
reserve that many tokens for each session.

```bash
bash serve/baseline-earkita/mimo/mimo-v2.6-flash-c2.sh DRYRUN=1
bash serve/baseline-earkita/mimo/mimo-v2.6-flash-c2.sh
```

This independently calls `serve/serve.sh`, without chaining model profiles.

## Historical C2 measurements (2026-10-03)

These rows come from the [complete C2/C4 report](mimo-v2.6-flash-c4.md#historical-shared-c2c4-qualification-2026-10-03).
Cold prompts; thinking enabled, temperature 1, top_p .95, top_k -1, seed 1234,
EOS enabled, output cap 256. One excluded warmup and three measured rounds;
all speed samples reached the cap. Hardware: eight R9700 at 225 W.

| Profile | Input | Concurrent | Decode mean (range), tok/s/request | TTFT mean, s | Effective prefill, tok/s/request | Aggregate E2E, tok/s | Acceptance length |
|---|---:|---:|---:|---:|---:|---:|---:|
| C2 | 1024 | 1 | 44.18 (41.83–46.36) | 0.47 | 2157 | 40.97 | 2.95 |
| C2 | 1024 | 2 | 43.96 (39.37–48.87) | 0.85 | 1208 | 73.98 | 2.92 |
| C2 | 32768 | 1 | 44.95 (40.43–48.19) | 14.55 | 2253 | 12.65 | 2.80 |
| C2 | 32768 | 2 | 34.92 (18.67–54.35) | 25.10 | 1337 | 14.90 | 3.01 |

Decode excludes TTFT, includes thinking and interference from other cold
prefills. Effective prefill is input/TTFT; aggregate E2E includes prefill.
Acceptance length includes the final target token. These are historical
observations, not a new performance comparison.

- API/thinking/streaming/two-turn tools: 8/8.
- 8K NIAH: strict/retrieval/format 2/3, completion 3/3, runtime errors 0.
  The refusal reproduced in a full matched rerun; original attempts remain.
- Capacity: two concurrent 522240-token inputs + 2048 output tokens completed
  at the output cap. Decode overlap 23.37 s; minimum free VRAM 566 MiB/GPU;
  TTFT 9.82–19.53 min. No preemption or OOM. This qualifies runtime capacity,
  not completed-answer quality at 512K.

Evidence: ignored `bench/results/mimo-c2-c4-20261003/`. The full report retains
protocols, all attempts, cross-profile limitations and capacity qualifications.
