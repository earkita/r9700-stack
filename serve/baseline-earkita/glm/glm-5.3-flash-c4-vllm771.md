# GLM C4 on vLLM 77154ca6 (experimental)

This standalone profile keeps the C4 runtime settings of
`glm-5.3-flash-c4.sh` and changes only the vLLM image. It does not enable
LMCache. Do not promote it over the established profile until matched
long-context speed and correctness checks are complete.

The image uses the ROCm 10 nightly
`vllm/vllm-openai-rocm:nightly-rocm100-77154ca6116ae34cbfe314beef1016bde9b63989`
(digest `sha256:7164f874815b44cfd639126cee25d80a5c4872a6f8e59c4495a2b172cc7e5d15`,
vLLM `0.32.0rc1.dev100+g77154ca61`). Build it from this checkout:

```bash
docker build -f docker/Dockerfile \
  --build-arg BASE=vllm/vllm-openai-rocm:nightly-rocm100-77154ca6116ae34cbfe314beef1016bde9b63989 \
  -t r9700/vllm:glm53-plugin-77154ca6 .
```

Launch with `bash serve/baseline-earkita/glm/glm-5.3-flash-c4-vllm771.sh`.
For the controlled 4×128K test use `MAXLEN=131072 NSEQ=4 KVMEM=4.125`.
Follow `testing.md` for hardware preflight, frozen protocol and result
classification. The audited adapter path lives in
`r9700_vllm/compat/glm_latest.py`; it retains only the GLM-specific patches
still needed after upstream incorporated DFlash state, Mamba APC and KV
packing changes. This adapter is pinned to vLLM 77154ca6 and gfx1201 and
rejects incompatible versions.

As checked on 2026-10-14, `77154ca6` is the newest published ROCm 10
nightly image; upstream `main` is newer but has no matching ROCm 10 image
yet. Keep the exact image digest above for reproducibility. A newer nightly
requires a fresh adapter audit and the same correctness/API qualification;
the version gate must not be loosened merely to make it start.

Upstream PR [#60753](https://github.com/vllm-project/vllm/pull/60753)
switches the GLM **prefill** indexer top-k to AITER when supported. Its vLLM
gate allows only gfx950. The AITER top-k library shipped in this image has
gfx942/gfx950 code objects but no gfx1201 code object. Its FlyDSL one-block
prefill path explicitly allows only gfx950/gfx1250, and the PR's own
microbenchmark found the AITER HIP paths slower than the in-tree kernel on
gfx950. Copying the PR's dispatch into our plugin would be a no-op on R9700;
forcing it would select an unsupported or slower path. An R9700 experiment
would require a separately validated gfx1201 FlyDSL/top-k implementation,
matching output indices at uneven sequence lengths and kpool=4, followed by
a frozen cold-prefill A/B. This is not a decode optimization: the PR reports
unchanged ITL. `llvm-objdump --offloading` output is preserved in ignored
`bench/results/glm-k7-vllm771-20261014/pr60753-aiter-topk-offloading.txt`.
For the decode goal, upstream [#58167](https://github.com/vllm-project/vllm/pull/58167)
is the relevant GLM top-k change, but it is already present in 77154ca6
and its AITER dispatch is gated to gfx950. Its reported 1–5% ITL reduction
was measured on MI350, so it does not establish a gain on gfx1201. An R9700
decode candidate needs its own supported kernel and matched numerical and
end-to-end C1/C4 tests; changing the top-k backend flag alone cannot enable it.
Our diagnostic 128K DFlash trace instead attributes about 24.9% of GPU
kernel time to packed W4A4 MoE GEMV, 17.4% to RCCL collectives, 12.0% to
BF16 split-K GEMV and 11.3% to sparse verification, averaged across eight
ranks. The sampled window is not a precisely isolated decode interval, so
these shares prioritize investigation but do not predict end-to-end gains.
The raw trace summary is in ignored
`bench/results/glm-mtp-20261014/torchprof-dflash-diagnostic/trace-summary.json`.

Diagnostic results and raw transcripts: ignored
`bench/results/glm-vllm771-20261014/`. The latest-image C4 warm 128K run
completed all four 256-token requests with simultaneous decode. A separate
natural-EOS NIAH run passed strict, retrieval, format and completion for all
four requests. Text/stream/tool API smoke passed. This does not qualify vision
or LMCache on the new image.
The image warns that the model processor ignores `max_pixels`; the vision
limit in this experimental profile therefore remains unqualified.

## Matched old/new diagnostic, 2026-10-14

Both images used TP8, C4 K4 DFlash, 131072 max length, 4.125 GiB/GPU KV,
the same 130816-token prompts, 256 generated tokens, temperature 1.0,
top-p 0.95, top-k disabled, seed 1234, effort high, the same client,
210 W/GPU and three warm APC repetitions. Each image first primed the
four prompts with the same cache salts. All eight HIP-to-BDF mappings and
loaded PCIe links passed the recorded splitter-topology policy. Both cold
priming runs completed all four requests but lacked all-client decode overlap;
their decode values are not C4 measurements.

| Mode | vLLM | Median decode per request | Median TTFT | Median aggregate output/s including TTFT | Draft-token acceptance |
|---|---|---:|---:|---:|---:|
| C1 | e97573215 | 53.25 tok/s | 835 ms | 45.16 tok/s | 35.5% |
| C1 | 77154ca6 | 49.22 tok/s | 734 ms | 43.51 tok/s | 33.7% |
| C4 | e97573215 | 23.76 tok/s | 1746 ms | 76.53 tok/s | 35.8% |
| C4 | 77154ca6 | 24.09 tok/s | 1765 ms | 77.20 tok/s | 34.1% |

Every warm request completed the exact 256-token budget, had APC hits and
zero preemptions. C1 decode was 44.11–53.49 tok/s on the older image and
43.11–60.65 tok/s on the newer one; C4 per-request decode was 16.93–31.76
and 20.98–32.59 tok/s respectively. The medians do **not** establish a
decode-speed gain from the vLLM update. TTFT here is warm-prefix retrieval
plus first-token latency, not cold prefill speed.

The older image allocated 575459 KV tokens and used a 128-token indexer page
with 1152-token attention blocks; the newer image allocated 563413 KV tokens
and used 256/1280. Both exceeded the four-request 128K gate. The geometry
change is observed, not a proven cause of the small timing differences.

Raw per-attempt data, including PCIe evidence and acceptance counters, lives
under `matched-old/` and `audit3/matched-latest/` in the result directory.
The final launcher was also started from the current checkout with this image:
`/health` passed, HIP/BDF mapping and 2110 loaded-link observations passed,
four concurrent arithmetic answers were correct and completed, and the
bounded API text/stream/tool contract smoke passed. That final smoke does not
repeat the long-context speed comparison.

## Small all-gather candidate, 2026-10-14

On the pinned 77154ca6 image, an opt-in `R9K_AG=1` with a 16 KiB maximum
was compared with `R9K_AG=0` at C1/C4, 130816 input tokens and 256 generated
tokens. Both legs used DFlash K4, warm APC, explicit checkpoint sampling,
three repetitions and 210 W on each of eight R9700. All warm requests
completed, showed APC hits and zero preemptions; the C4 runs had real
four-request decode overlap. PCIe preflight and loaded-link checks passed
for both legs. The candidate also passed bounded text/stream/tool smoke.

| Setting | C1 median decode/request | C4 median decode/request | C4 median aggregate output, including TTFT | C4 mean draft acceptance |
|---|---:|---:|---:|---:|
| `R9K_AG=0` | 56.97 tok/s | 27.73 tok/s | 86.98 tok/s | 41.0% |
| `R9K_AG=1` | 39.48 tok/s | 22.95 tok/s | 79.49 tok/s | 32.5% |

There is no demonstrated speed gain. Acceptance differed substantially, so
the decode-rate difference is not an isolated estimate of all-gather cost.
The baseline harness stopped after cold priming because its cold-phase
concurrency assertion was too strict; its warm measurements resumed after
that assertion was corrected. The candidate ran the corrected harness from
cold onward, so the cold-to-warm delay differed. Treat this comparison as
diagnostic, not a regression claim. Keep `R9K_AG=0`. Exact protocol,
per-request observations, counters, PCIe records and report are in ignored
`bench/results/glm-ag-vllm771-20261014/`.

## Opt-in dense FP8 candidate on 77154ca6

The version-gated adapter now recognizes native `model.layers` on this vLLM
build while still excluding draft/vision prefixes. It remains **off by
default**. Its original FP8 bytes and FP32 scales are retained; an extra
packed copy adds about 0.06 GiB/GPU at model load. Enabling it requires a
library with `r9k_glm_gemm_fp8_block_qa` (the production library does not
export that symbol), so do not set `R9K_GLM_DENSE_FP8=1` with the normal
launcher/library.

An isolated GPU numerical test passed 18/18 shape/row cases, including
changed-input graph replay; four concurrent arithmetic answers and the
text/stream/tool API smoke passed. The runtime logged use of both targeted
Quark matrices. A matched, warm-APC 130816-input/256-output C1/C4 test on
the same 77154ca6 image and candidate library changed only the FP8 flag.
All four C4 requests overlapped in decode in every measured round; cache
hits and loaded PCIe checks passed and there were no preemptions.

| Dense FP8 | C1 median decode/request | C4 median decode/request | C4 aggregate output incl. TTFT | C4 draft acceptance |
|---|---:|---:|---:|---|
| off | 49.32 tok/s | 27.37 tok/s | 88.96 tok/s | 36.1%, 45.3%, 42.3% |
| on | 47.55 tok/s | 27.94 tok/s | 83.57 tok/s | 36.9%, 39.1%, 45.5% |

The fused micro-kernel is faster in isolation, but **there is no confirmed
whole-model decode gain**. Keep the flag off. Raw protocol, all attempts,
numerical results and PCIe evidence are in ignored
`bench/results/glm-dense-vllm771-20261014/`. This candidate was not tested
with LMCache, 256K input or broad answer-quality suites.

## 256K context qualification, 2026-10-14

The pinned 77154ca6 image with the stock kernel library also started with
`MAXLEN=262144 NSEQ=4 KVMEM=4.125` and reported 639809 KV tokens. This is a
shared KV pool: two distinct 261120-token prompts plus up to 256 generated
tokens completed together, as did four concurrent 130816-token prompts.
It does **not** establish capacity for four simultaneous 256K prompts.

For two 261120-token prompts, three warm APC repetitions gave median decode
of **52.04 tok/s** at C1 and **38.66 tok/s per request** at C2. C2 aggregate
output, including TTFT, was **57.33 tok/s**. Warm TTFT medians were 1.52 s
and 1.89 s respectively. Each request emitted exactly 256 tokens; all C2
rounds had simultaneous decode, APC hits and no preemptions. Cold prefill
partly serialized the two requests (TTFT 154/307 s), so its decode numbers
cannot be used as simultaneous C2 rates. Natural-EOS NIAH passed strict,
retrieval, format and completion **2/2** with no runtime error.

A bounded C4 128K smoke under the same 256K limit showed four-way decode
overlap, APC hits and no preemptions; NIAH passed all four dimensions
**4/4**. Its single warm decode observation was 27.89 tok/s per request,
which is not a comparable benchmark. The all-GPU PCIe preflight and loaded
checks passed for both series. Exact protocols, output transcripts and raw
records are in ignored `bench/results/glm-vllm771-256k-20261014/`.

This 256K result used no LMCache, dense FP8 adapter or custom all-gather.
Keep the versioned profile experimental; qualifying a later nightly needs
its own model/API/PCIe checks.

A separate bounded API smoke at this 256K setting passed non-stream and
streamed reasoning/content and tool calls, plus a 128-token forced-output
request. All eight GPUs passed 117 loaded PCIe observations. This is an API
contract check, not a throughput or completed-answer quality estimate. The
transcript and hardware evidence are in the same ignored result directory.

## Packed W4A4 down tile, matched C1/C4 diagnostic

The opt-in `R9K_GLM_PACKED_DOWN_BN=32` was tested against the default tile 8
on this pinned image with MAXLEN 262144, four 130816-token prompts, DFlash K4,
warm APC, three repetitions and the same checkpoint sampling. Each variant
used a fresh server and cold cache prime. Both passed four-way decode overlap,
zero preemptions, all-eight-GPU loaded PCIe checks and natural-EOS NIAH 4/4.

| Down tile | C1 median decode/request | C4 median decode/request | C4 median aggregate output incl. TTFT |
|---|---:|---:|---:|
| 8 | 61.86 tok/s | 24.37 tok/s | 85.10 tok/s |
| 32 | 53.57 tok/s | 24.15 tok/s | 83.36 tok/s |

Draft acceptance differed among attempts; BN32 shortened the derived C1
draft-round time by less than 1%, without a confirmed whole-model C4 gain.
Keep tile 8. Full per-attempt metrics, cold TTFT/prefill observations,
protocol and PCIe evidence are in ignored
`bench/results/glm-packed-vllm771-20261014/`.

## BF16 projection diagnostic

An isolated gfx1201 test compared the existing router GEMV with stock
`F.linear` for the TP8 BF16 Mamba Q/K/V and O projections and the sparse
KV-B projection. It used a frozen protocol, one exposed GPU, numerical
checks, eight timed graph replays per shape, and idle plus loaded PCIe
evidence. At five rows, only Q/K/V improved: 21.56 to 18.56 microseconds
per call (router split 1). The router was slower for the other projections;
at 20 rows it was substantially slower for every shape. These are isolated
kernel timings, not model decode rates. Keep the normal BF16 path and
consider a Q/K/V-only adapter only if a full-model A/B shows a benefit.
The ignored raw records and harness live under
`bench/results/glm-bf16-gemv-20261014/`; the reusable diagnostic is
`tuning/tune_glm_bf16_gemv.py`.

## DFlash K4/K7/K4 diagnostic, 2026-10-14

The newest published ROCm 10 image above was tested with a frozen A–B–A
protocol, changing only `num_speculative_tokens` and its required graph
capture sizes. All legs used TP8, C4 capacity, MAXLEN 262144, 4.125 GiB/GPU
KV, 130816 input tokens, 256 output tokens, checkpoint sampling, effort high,
210 W/GPU, cold cache priming followed by three warm APC repetitions, and the
same client. All eight GPU mappings and loaded PCIe links passed preflight in
each leg. Warm C4 had genuine four-request decode overlap; all requests
completed 256 tokens with APC hits and no preemptions. Natural-EOS NIAH
strict/retrieval/format/completion passed 4/4 in each leg; no backend error.

| Leg | C1 decode/request | C4 decode/request | C4 aggregate output incl. TTFT | C4 warm TTFT | C4 accepted/proposed draft tokens |
|---|---:|---:|---:|---:|---:|
| K4 first | 52.58 tok/s | 25.22 tok/s | 85.07 tok/s | 1351 ms | 37.6% |
| K7 | 47.97 tok/s | 22.24 tok/s | 74.29 tok/s | 1415 ms | 24.7% |
| K4 repeat | 51.41 tok/s | 26.96 tok/s | 91.92 tok/s | 1329 ms | 38.9% |

K7 accepted more tokens per draft round than K4, but its last three positions
were accepted in only 15.7%, 12.3% and 10.2% of C4 draft rounds. It did not
improve measured decode; keep K4 as the default. These are 128K-context
diagnostic results, not a 256K or LMCache qualification. Cold priming partly
serialized the C4 requests, so its elapsed time and TTFT are not pure prefill
or simultaneous C4 decode measurements. Exact protocol, every attempt,
position counters and PCIe evidence are in ignored
`bench/results/glm-k7-vllm771-20261014/`.

## HIP hardware queues, bounded C1 diagnostic, 2026-10-14

An upstream report about gfx1201 side queues suggested checking
`GPU_MAX_HW_QUEUES`. A frozen C1 screening changed only this value (2 versus
the launcher's default 1) on the same image, K4 drafter, 256K-capacity profile,
130816-token input, 256-token output, checkpoint sampling and warm APC. Each
leg had a separate cold prime and three warm repetitions. Idle and loaded
PCIe mapping/link checks passed on all eight GPUs for both legs, and
natural-EOS NIAH passed strict/retrieval/format/completion 1/1 without a
backend error. The two container configurations differ only in the queue
setting.

| HIP queues | C1 decode attempts | Median warm TTFT | Accepted/proposed draft tokens |
|---:|---:|---:|---:|
| 2 | 34.06, 32.61, 34.37 tok/s | 801 ms | 30.6% |
| 1 | 52.98, 77.18, 86.51 tok/s | 781 ms | 52.4% |

The response sampling and speculative acceptance varied substantially between
attempts even with the same seed. This screen therefore does not assign the
entire difference to HIP queue scheduling, but it gives **no reason to promote
HWQ=2**; keep `HWQ=1`. It does not establish a C4 or 256K-input result. The
exact protocol, each attempt, position counters, runtime diff and PCIe
evidence are in ignored `bench/results/glm-hwq-vllm771-20261014/`. Cold
effective input rates include first-token work and are not pure prefill rates.

## Exact TP8 all-reduce cap, A–B–A diagnostic, 2026-10-14

The 8-rank P2P all-reduce was already installed, but this GLM profile limits
it to 16 KiB; larger reductions use RCCL. A bounded A–B–A comparison tested
`R9K_ARN_MAX_KB=16 → 256 → 16`, keeping the same image, K4, 256K-capacity
profile, 130816-token inputs, 256-token outputs, checkpoint sampling, warm
APC, 210 W/GPU and three repetitions each at C1 and C4. The launcher uses
separate torch.compile cache directories for different `R9K_*` settings to
prevent stale graphs; startup and cold priming were excluded from warm decode.
Each leg passed idle and loaded PCIe checks on all eight GPUs, had genuine
four-request decode overlap, APC hits and no preemptions. Natural-EOS NIAH
strict/retrieval/format/completion passed 4/4 in each leg, without runtime
errors. An isolated eight-rank numerical test passed exact reductions,
including changing inputs under graph replay at the 160 KiB GLM C4 shape.

| AR8 cap | C1 decode/request | C4 decode/request | C4 aggregate incl. TTFT | C4 warm TTFT | C4 accepted/proposed drafts |
|---|---:|---:|---:|---:|---:|
| 16 KiB first | 54.62 tok/s | 25.92 tok/s | 83.49 tok/s | 1342 ms | 36.1% |
| 256 KiB | 57.44 tok/s | 24.93 tok/s | 84.79 tok/s | 1356 ms | 33.6% |
| 16 KiB repeat | 49.21 tok/s | 24.19 tok/s | 79.85 tok/s | 1353 ms | 34.2% |

The 256 KiB result lies within the C4 baseline spread, so **a C4 decode gain
is not established**. A C1 derived decode interval per draft round fell from
about 45.1/45.2 ms in the baseline legs to 40.9 ms with the 256 KiB cap,
but this includes more than the collective and does not control output or
acceptance. Keep 16 KiB as the profile default. This test does not qualify
256K input, LMCache or broader tool-use quality for the candidate. Frozen
protocol, all attempts, exactness and PCIe evidence are in ignored
`bench/results/glm-ar8-vllm771-20261014/`.

## DFlash K4/K3/K4 diagnostic, 2026-10-14

K3 tested whether omitting K4's often-rejected fourth draft token reduces
step time enough to improve whole-model decode. A frozen A–B–A run changed
only the speculative-token count and the corresponding required graph sizes;
image, target/draft weights, TP8, 4.125 GiB/GPU KV, 130816 actual input
tokens, 256 output tokens, checkpoint sampling, warm APC, power caps and
three C1/C4 repetitions per leg were fixed. Cold C4 priming was excluded.
All legs passed NIAH strict/retrieval/format/completion 4/4, with no runtime
errors or preemptions, genuine four-request decode overlap, APC hits and
loaded PCIe checks on all eight GPUs.

| DFlash | C1 decode/request | C4 decode/request | C4 aggregate incl. TTFT | C4 draft acceptance |
|---|---:|---:|---:|---:|
| K4 first | 48.36 tok/s | 25.46 tok/s | 83.86 tok/s | 35.8% |
| K3 | 50.11 tok/s | 25.25 tok/s | 80.62 tok/s | 41.7% |
| K4 repeat | 50.88 tok/s | 25.18 tok/s | 79.17 tok/s | 34.6% |

Higher acceptance percentage with K3 reflects its shorter proposal; it did
**not** produce a confirmed decode or aggregate-throughput gain. Keep K4 as
the default. The ignored `bench/results/glm-k3-vllm771-20261014/` contains
the frozen protocol, all attempts, position counters, cold effective prompt
rates (including queue/first-token work), runtime diff and PCIe evidence.
K3 has not been qualified at 256K actual input or with LMCache.

## Grouped WMMA W4A4 C4 diagnostic, 2026-10-14

The existing native grouped WMMA W4A4 kernel was temporarily enabled for up
to 32 verification rows and compared with the default packed GEMV in a
frozen `packed → grouped → packed` run. A one-GPU numerical gate first passed
7/7 selected cases, including 16/20/32 rows, uneven expert routes and
changed-input graph replay. The 128K model comparison then changed only
`R9K_GLM_W4A4_BATCH`: both packed legs had identical container commands,
image and environment, and the grouped leg differed by that one variable.
Every leg used DFlash K4, TP8, 130816 actual input tokens, 256 output tokens,
warm APC and three C1/C4 repetitions. All eight GPU PCIe mappings and loaded
links passed; C4 decode overlapped, with no preemptions. Natural-EOS NIAH
strict/retrieval/format/completion passed 4/4 in every leg.

| W4A4 implementation | C1 decode/request | C4 decode/request | C4 aggregate output incl. TTFT | C4 draft acceptance |
|---|---:|---:|---:|---:|
| packed first | 54.49 tok/s | 23.93 tok/s | 80.79 tok/s | 33.0% |
| grouped | 36.02 tok/s | 16.00 tok/s | 56.73 tok/s | 33.8% |
| packed repeat | 51.99 tok/s | 23.75 tok/s | 79.98 tok/s | 34.0% |

The grouped kernel is about one-third slower at C4 under this protocol;
similar speculative acceptance does not explain the gap. **Keep packed.**
The temporary 32-row grouped adapter gate and its tests were removed after
the run. Exact tested source snapshots, frozen protocol, all attempts,
per-position acceptance, numerical log, runtime diff and PCIe evidence remain
in ignored `bench/results/glm-grouped-vllm771-20261014/`. This rejected
variant was not qualified for 256K actual input, LMCache or tool calling.
