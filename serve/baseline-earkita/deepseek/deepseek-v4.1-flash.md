# DeepSeek V4.1 Flash: requalification on eight R9700 GPUs

## Decode critical path from the eight-rank trace (2026-10-14)

An offline alignment of the 8 traced decode steps across all 8 ranks identifies
the immediate bottleneck more precisely. Each rank launched 80 packed MoE
kernels and 86 NCCL kernels per step. Of 5120 MoE calls, 205 exceeded 1 ms;
the longest took 16.38 ms, while the typical call was about 0.03 ms. The slow
rank changes by layer and step, so this is not one persistently slow GPU.

At each of 320 layer/step positions, the spread between the fastest and
slowest rank's two MoE projections closely matches the spread in duration of
the **following** NCCL collective (Pearson r=0.995 in this trace). For example,
at step 3, layer 30, rank 1's projections took 21.126 ms versus 0.005 ms on
the fastest rank; the following collective's inter-rank duration spread was
21.140 ms. Across 688 corresponding collective positions, 308 had a duration
spread above 1 ms. This is direct evidence that intermittent MoE stragglers
make other ranks wait at synchronization. It does not establish that NCCL's
own data transfer is slow: summing the fastest observed rank at each
collective gives only 6.3–8.4 ms per step, whereas the eight-step rank
averages are 121–148 ms of collective duration per step, including waits.

The likely source of the MoE long tail is access to offloaded expert pages:
the server keeps 10.5 GiB/rank of expert matrices host-backed, and the
`R9K_DEEPSEEK_STAGE_MIN_TOKENS=256` threshold means C1 decode reads them
without whole-projection staging. However, the trace does **not** identify
the selected expert IDs or measure host-page traffic. The precise page-level
cause remains a hypothesis. Sparse attention was much steadier
(9.3–9.6 ms/step), and the
remaining GPU kernels took 17.5–18.8 ms/step. About 3255 kernel launches per
step and 9–21% gaps in the GPU event span are a secondary launch/scheduling
cost, subject to profiler overhead. No new runtime or comparison benchmark was
run for this analysis; the existing PCIe discrepancy still makes performance
observations diagnostic rather than comparable.

## Current decode trace (2026-10-14)

The placed, decode-weight-0.5 variant was restarted with the same vLLM
arguments plus a CUDA-activity Torch profiler. A single 8192-token, C1,
low-thinking diagnostic request traced eight decode steps after 20 untraced
iterations. Its 128 generated tokens all went to reasoning and ended at the
token limit; this is a kernel trace, **not** a completed-answer quality test or
a throughput comparison. Raw traces for all eight ranks, the frozen request,
and PCIe evidence are in `bench/results/deepseek-current-trace-20261014/`.
After profiling, the original decode-weight-0.5 DeepSeek container was
restarted and its `/health` endpoint passed; the profiler container is stopped.

Across the eight ranks, the 80 packed MoE kernels per token occupied
13.2–35.3 ms/token, with 10.4–27.4 ms in the up projection. Individual MoE
calls reached 14–16 ms on some ranks while the median call was roughly
0.03 ms. Sparse attention occupied 9.3–9.6 ms/token; other GPU kernels occupied
17.5–18.8 ms/token. The 86 NCCL kernels per token were recorded for
120.7–147.6 ms/token, but their durations include waiting for slower ranks:
the sum of the minimum duration at each matching collective position was only
7.1 ms/token. The profiler also observed roughly 3255 kernel launches per
token and 9–21% gaps inside the GPU event span. These categories overlap with
inter-rank waiting and profiler overhead; do not sum them into unprofiled
request latency.

The next focused optimization is to reduce long-tail expert accesses and rank
imbalance, for example by measuring selected offloaded expert-page traffic and
then testing selective prefetch/staging. Copying an entire projection on every
decode token would be too expensive. The existing offline held-out routing
estimate changes little from decode placement weight 0.5 to 0.75, so another
weight-only server restart is not justified by that estimate. The PCIe check
again found an unexpected negotiated width at bridge `0000:e1:00.0`; all
performance observations remain **diagnostic, not comparable**.

## Current diagnostic launch (2026-10-14)

`deepseek41-placed-decode05-20261014` is the currently running DeepSeek server
on port 8080. It uses the same calibrated routing corpora, TP8/EP8, 10.5 GiB
configured CPU offload per GPU, 65536 maximum model length, upstream reference
prefill and eager execution as the prefill-only placed variant. Its single
experimental change is `R9K_DEEPSEEK_PLACEMENT_DECODE_WEIGHT=0.5`, which ranks
the packed expert pages using an equal blend of prefill slots and decode call
hits. The default remains `0` (prefill-only). `/health` passed after loading.
The earlier prefill-only container `deepseek41-placed-20261014` is stopped,
available as the DeepSeek fallback; GLM was not restored.

The new candidate's frozen C1 8K, low-thinking, 128-output series yielded
decode 5.586/5.705/5.823 tok/s (median **5.705**), TTFT
50.146/50.499/49.917 s (median **50.146**) and effective prefill
163.4/162.2/164.1 tok/s (median **163.4**). All three runs reached the
128-token limit in reasoning; they measure throughput, not answer quality.
The separate exact-token retrieval checks at 8192 and 32768 input tokens both
returned exactly `PINE-4729`, ended with `finish_reason=stop` and had no
request-level runtime error. Their TTFT/decode were 46.440 s/5.595 tok/s and
185.187 s/5.749 tok/s, respectively. Strict exact match, retrieval, output
format and completion each passed 2/2. These two C1 completions do not qualify
C4, tools, or broader quality.

The previous prefill-only placed median was 3.482 tok/s decode and 53.391 s
TTFT under the same saved request protocol. This is an **observed diagnostic
difference**, not a certified speedup: the mandatory PCIe check reported an
unexpected negotiated width on bridge `0000:e1:00.0` in both variants. No
PCIe root-cause work was performed in this series. Offline cross-domain
expert-page traffic estimates predicted a decode benefit, but are not measured
PCIe traffic. Raw requests, stream timestamps, model identity, placement source
hashes, memory guard and PCIe captures are in
`bench/results/deepseek-placement-20261014/placed_decode05/`; the offline
estimates are in `bench/results/deepseek-decode-profile-20261014/`.
The placement unit tests passed 14/14 in the pinned DeepSeek image. This
candidate is opt-in: a replay needs the two saved calibration `routing.json`
files and must pass both `R9K_DEEPSEEK_EXPERT_PLACEMENT` and
`R9K_DEEPSEEK_PLACEMENT_DECODE_WEIGHT=0.5` to the launch profile. The launch
arguments are preserved in `placed_decode05/launch-dryrun.txt`; the measured
image and source identities are in `placed_decode05/source-manifest.json`.

Before this candidate, `deepseek41-placed-20261014` was restarted after the
three-variant study. Its `/health` check passed. A separate low-thinking smoke
request (`9 + 8`) completed naturally with answer `17`, 18 reasoning tokens
and `finish_reason=stop`; raw request, response and PCIe observations are in
`bench/results/deepseek-placement-20261014/placed/restart-smoke/`. This smoke
does not extend the frozen 8K performance series or establish broader quality.

The placed main-tree container `deepseek41-placed-20261014` and its matched
eager control, `deepseek41-eager-ab-20261014`, completed their diagnostic C1
series and were stopped cleanly between variants. Both use the pinned
5555a2d8 parser image, reference upstream prefill,
TP8/EP8, eager execution, text only, a **65536-token per-request ceiling**,
and 10.5 GiB/rank configured expert offload. Only the placed variant enables
calibrated hot/cold routed-expert page placement. Its placement profiles
were collected from six completed real-repository prompts (coding and prose),
with 40/40 router layers covered. The previous trace container was stopped
cleanly before the placed launch. Neither the full model context nor C4 is
qualified by this experiment. Earlier sections below
are historical attempts, not instructions to restore GLM.

The frozen C1 protocol and raw evidence are in
`bench/results/deepseek-placement-20261014/`; the routing profiles and decode
traces are in `bench/results/deepseek-decode-profile-20261014/`. The trace's
eight measured decode steps on rank 0 spent 104.1 ms/step in MoE kernels and
257.5 ms/step in NCCL kernels. The NCCL duration includes waiting for slower
ranks; it is not a measurement of pure collective latency. Offloading the
coldest pages while preserving the same host-byte budget is the current
experimental variable. Offline traffic estimates are in-sample predictions,
not observed speedups. The prescribed low-thinking C1 measurement records
TTFT/prefill and decode separately; hardware evidence currently permits only
diagnostic, not comparable, performance claims.

The placed run's frozen C1 8K/128-token series completed three measured
requests. Decode was 3.485/3.469/3.482 tok/s (median 3.482); TTFT was
53.264/53.391/53.602 s (median 53.391). The matched eager control gave
2.515/2.470/2.509 tok/s (median 2.509) and TTFT
28.850/28.875/28.853 s (median 28.853). A third diagnostic kept the same
placed pages but disabled whole-projection staging: decode was
3.461/3.461/3.450 tok/s (median 3.461), while TTFT rose to
97.813/97.910/98.442 s (median 97.910). This isolates the staging tradeoff:
it substantially helps prefill with placed pages and has little effect on
decode. It does not restore the eager control's prefill rate. All speed samples
reached the 128-token limit in reasoning; they are not quality checks. Some
responses combined a few tokens into SSE chunks; the script records this
mismatch. The independent
exact-token retrieval check finished naturally with strict, retrieval, format,
completion and runtime dimensions all passing at both 8192 and 32768 input
tokens. After correcting a parser that had ignored `delta.reasoning`, the raw
SSE timestamps give 8K TTFT 51.622 s / decode 3.375 tok/s and 32K TTFT
208.633 s / decode 3.308 tok/s. The original flawed derived fields and all
raw events are retained beside `*.verified.json`. These checks prove two C1
completions, not general model quality or C4 service capacity. The hardware
checker reported an unexpected negotiated width on bridge `0000:e1:00.0`;
formal A/B performance comparability is therefore unresolved. The no-staging
variant also recorded the same bridge-width warning. These are diagnostic
observations, not certified performance comparisons.

The launcher bind-mounts this repository over `/opt/r9700`. Consequently the
image's compiled library is hidden by the host checkout. Preserve a separate
library from the matching image instead of replacing the shared GLM library:

```bash
bash docker/build-deepseek.sh
mkdir -p .runtime/deepseek-latest
artifact_container=$(docker create r9700/vllm:deepseek-v41-5555a2d8-parser)
docker cp "$artifact_container:/opt/r9700/r9700_vllm/kernels/libr9k.so" \
  .runtime/deepseek-latest/libr9k.so
docker rm "$artifact_container"
```

After releasing the previous model's GPUs, reproduce the reference-path
compatibility diagnostic with explicit overrides (arguments after the profile
name override its defaults). Add the placement overrides from the preceding
paragraph only when deliberately replaying the placement candidate:

```bash
bash serve/baseline-earkita/deepseek/deepseek-v4.1-flash.sh \
  NAME=deepseek41-replay-fix-20261014 \
  MAXLEN=65536 LMONLY=--language-model-only \
  R9K_LIB=/opt/r9700/.runtime/deepseek-latest/libr9k.so
mkdir -p bench/results/deepseek-replay-fix-20261014
python3 serve/runtime_memory_guard.py deepseek41-replay-fix-20261014 \
  --minimum-available-gib 4 --maximum-full-stall-percent 50 \
  --stall-window-seconds 60 \
  --log bench/results/deepseek-replay-fix-20261014/memory-guard.jsonl
```

Keep the guard running in its own terminal throughout loading and inference.
These commands are for
a new launch, not a request to restart an active test. For inspection without
starting a container, prefix the profile command with `DRYRUN=1`.

The tested library SHA-256 is
`6ddc8d34daeb7b56c5effe7f9e3f3c2290bd66a01d662cc9beb890ac17fdac4a`;
the image ID and Python source hashes are saved in the run's
`source-manifest.json`. The default auto-context/vision profile is not the
same configuration and must not inherit this diagnostic's qualification.

## Latest attempt interrupted by host reboot (2026-10-13)

The 5555a2d8 parser image passed 87 CPU contract tests (one optional checkpoint
test skipped), 105 GPU numerical tests plus 19 subtests, and 49 prompt/parser
tests. Full adapter registration also passed. These are component checks only.

Container `deepseek41-latest-20261013` began loading with reference upstream
prefill, TP8/EP8, eager, text-only, utilization 0.95 and configured CPU offload
10.5 GiB/rank (whole tensors logged 10.81 GiB). The host rebooted before model
loading finished; neither the 8K nor the 32K inference test ran.

At 22:58:28 CEST the previous kernel logged a deferred UECC machine check.
On the next boot, BERT reported a DRAM storage error, locator `P0MH0`, physical
address `0x419188a340`, and an uncorrected processor data-read error. This is
hardware-error evidence, not a vLLM CUDA OOM. The precise reboot mechanism and
whether the DIMM, slot, controller or memory settings are responsible remain
unproven. Do not infer the physical DIMM from the locator without SMBIOS mapping.

The last valid guard sample had 27.99 GiB MemAvailable and roughly 0.001 GiB
SwapFree; minimum recorded MemAvailable was 25.61 GiB, above the existing 4 GiB
guard. Tail telemetry contained NUL bytes after the abrupt reset; preserve it
as raw evidence rather than claiming continuous monitoring through the crash.
Docker reports exit 255, OOMKilled=false. No automatic reload or GLM restoration.

Raw logs: `bench/results/deepseek-latest-20261013/reboot/`, `server.log`,
`memory-guard.jsonl`, and `startup-status.json`. Full loading should wait for
investigation of the recorded DRAM error. The operator subsequently confirmed
the reset was spontaneous, then explicitly requested deferring hardware
diagnostics and continuing DeepSeek qualification.

The launch audit found no overlapping old model and no increased host-memory
peak relative to the preceding failed pilots: sampled AnonPages peaked at
291.83 GiB versus 297.73/298.01 GiB previously. The old and new image have
identical Engram allocation-function ASTs and the same PyTorch/ROCm package
versions. This does not exclude a software/driver defect or establish a faulty
DIMM. The final previous-boot journal entries are the machine check; no recorded
panic or orderly reboot was found. The BMC event log could not be read without
administrator authentication.

The explicitly requested retry is recorded separately in
`bench/results/deepseek-latest-retry-20261013/`. It retains the reference profile
and C1 8K/32K protocol. The 4 GiB headroom guard is unchanged; this retry also
enables `--maximum-full-stall-percent 50 --stall-window-seconds 10` on
`serve/runtime_memory_guard.py`. This optional system-wide memory-pressure guard
stops only the named container after sustained stalls; it cannot prevent or fix
an uncorrected hardware error. Four CPU guard tests pass. No PCIe investigation
or GLM restoration is part of this retry.

Retry outcomes: the 10-second PSI guard stopped at 46/48 shards with 28.22 GiB
available; this is a guard action, not an OOM or a spontaneous reboot. A second
attempt with a 60-second sustained-pressure window loaded all weights in about
207 seconds, but reference memory profiling left **-0.05 GiB for KV**, so the
API did not start. Neither attempt produced an inference measurement.

The next diagnostic launch overrides only `MAXLEN=65536` to bound reference
prefill scratch while retaining room for 32768 input plus 1024 output tokens.
The profile default remains automatic. Results are in
`bench/results/deepseek-latest-64k-20261013/`; its frozen protocol records this
ceiling and retains the existing 8K/32K prompts and sampling. This is not a
capacity qualification of the model's full context window.

The 64K launch reached a healthy API with 0.87 GiB KV per GPU and a reported
431,198-token shared capacity. The excluded 46-input/64-output warmup observed
2.49 tok/s decode and hit its output limit. The subsequent 8K request failed
before output with `Current vLLM config is not set`: our disabled paged-prefill
constructor wrapper queried the global config when vLLM created replay metadata
builders during execution. This was an adapter bug, not a qualified 8K result.

The constructor now avoids the config lookup when disabled and uses the
builder's retained explicit config when enabled. Four replay-constructor cases
(on/off, positional/keyword config) and a disabled no-config case cover the
failure. Prefill plus memory-guard CPU tests: 14 passed. End-to-end retry uses
the same 64K ceiling and protocol, with results under
`bench/results/deepseek-replay-fix-20261014/`.

The fixed adapter passed the formerly failing 8K execution path. Diagnostic C1:
8192 input tokens, 1024 generated tokens, TTFT 30.109 s, effective input/TTFT
272.08 tok/s, decode 2.499 tok/s (actual output usage, separately timed).
There was no backend error, but **all output was reasoning and finish=length**;
no final answer, so completion/quality did not pass and 32K was not attempted in
that protocol. The visible reasoning retrieved the correct incident facts;
this must not be reported as a completed-answer success.

An explicitly separate completion protocol under the same results directory's
`completion/` subdirectory increases the generation budget to 4096, keeps EOS
active, uses fresh cold salts, and retains the prompts and sampling. It reuses
the existing warmup; it is not an A/B performance comparison with the 1024-token
attempt. C1 32K follows only after the 8K response finishes naturally.

The 4096-token attempt also reached its output limit: TTFT 28.692 s,
effective input/TTFT 285.51 tok/s, decode 2.500 tok/s, wall 1666.54 s.
It did transition to a meaningful final answer after extensive drafting and
word counting in reasoning (first final-answer content at 1518.47 s). The final
answer retrieved all required incident facts and separated proposals, but was
cut off in the last prevention paragraph, without `END OF REVIEW`. Thus
retrieval passed this manual review; completion, final format and the
correct-completed gate did not. No backend error or preemption was observed.
Reasoning/final token estimates are 3626/470 from BetterBench, not separate
authoritative server usage counters. Do not label this a successful quality run
or an endless reasoning loop. The harness correctly withheld the 32K stage.

The `completion6144/` attempt was deliberately interrupted when the user changed
priority to improving decode speed. Partial SSE and `interruption.json` are
preserved. This was not a backend failure or a completed quality test; its 32K
stage was not run.

### Decode optimization: HIP graphs (2026-10-14)

The diagnostic protocol is saved in
`bench/results/deepseek-decode-graphs-20261014/`: C1, exactly 8192 input tokens,
thinking low, temperature 1.0, top-p 0.95, seed 41012, EOS enabled, fresh cold
salt per request. One excluded 32-token warmup precedes three requests with
128-token output budgets. BetterBench 0.6.0 times decode separately from TTFT.

The eager leg observed 2.4823 / 2.4583 / 2.4805 tok/s decode (median 2.4805),
TTFT 28.71–28.82 s and effective input/TTFT 284.24–285.36 tok/s. All three
outputs hit their length limit; these are throughput samples, not completed
answers. Separate short chat arithmetic checks completed correctly with
thinking disabled and low. Raw continuation checks passed 3/4; the existing
water-freezing continuation failure is retained for the graph comparison.

The candidate enables `EAGER=0 CGMODE=FULL_DECODE_ONLY CGSIZES=1
R9K_DEEPSEEK_GRAPHS=1`, retaining eager prefill and the other launch settings.
The scoped MoE GPU test passed capture/replay for C1 and C4 with changed inputs,
routes, GPU/UVA weights and mixed page residency. The full model captured C1 on
all eight ranks, with an actual graph pool of 0.12 GiB/rank. Automatic KV capacity
fell to 240,298 tokens (rank zero reported 0.53 GiB available for KV), versus
431,198 in eager; the limit per request remains 65,536.

Measured decode was 2.2366 / 2.2456 / 2.2437 tok/s (median 2.2437), TTFT
27.025–27.041 s, and effective input/TTFT 302.94–303.13 tok/s. All measured
responses again hit the 128-token budget. Zero prefix hits and zero preemptions
were recorded. Both arithmetic chats completed correctly, and raw continuation
checks had the same 3/4 result with finite log-probabilities as eager. This small
check does not qualify general answer quality or long-context stability.

The candidate has fresh PCIe evidence including 59 samples during the measured
series, while the eager leg omitted it under the earlier user instruction:
this remains diagnostic, not a certified comparable A/B result. Graphs alone
have not delivered the desired decode speed; do not promote this candidate as
an optimization. A separate CPU stack sample during sanity checks found the
worker waiting for GPU completion, which does not isolate the expensive kernel.
Next is bounded GPU profiling and routing calibration before hot/cold placement.
The calibration identity now records the installed vLLM version and graph
configuration instead of hard-coding the old image; 13 CPU calibration/placement
tests passed, including rejection of mismatched profiles.

The next diagnostic launch is `deepseek41-profile-20261014`, eager with the
same 64K/TP8/EP8/offload settings, `PROF=1 PROFACT=CUDA`, and opt-in routing
counters. Its frozen plan and unattended runner live in
`bench/results/deepseek-decode-profile-20261014/`. The runner waits for that
exact container, then captures eight iterations after a 20-iteration delay in
a cold 8K/64-token request and records separate coding/prose routing corpora.
Profiler and routing instrumentation invalidate performance timing. Startup,
trace collection and placement qualification remain pending; no hot/cold
placement has been activated in this attempt.

## Prompt/protocol reference audit (2026-10-13)

The official [encoding reference](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/encoding/README.md)
defines V4.1 DSML tags (including the leading space), numeric effort 1–100
(`low=50`, `high=75`, `max=100`), and retention of earlier reasoning when tools
are defined. Clients/proxies must return reasoning and tool results without
silently discarding them. Missing history is a hypothesis to test, not an
established explanation for previous looping or the separate prefill timeout.

Our profile uses the vLLM `deepseek_v41` tokenizer and both V4.1 parsers.
The pinned 5555a2d8 build uses an adapted Python reference encoder, not the
[deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) Rust/Python
library. Recipe provides API conversion, prompt encoding and output parsing;
it does not supply inference or tool execution. Use it as an independent
protocol oracle before considering any replacement of serving components.

Validation: `tests/test_deepseek_parser_roundtrip.py` passed 49 cases on the
5555a2d8 parser image: 24 prompt parity cases against the local checkpoint
reference (thinking on/off, named/numeric effort, plain/tool conversations),
plus 25 existing parser/streaming cases. This is CPU protocol validation,
not proof that model-generated calls, proxy history or long-context inference
are correct. Raw output: `bench/results/deepseek-latest-20261013/prompt-reference-validation.log`.

Sampling remains temperature=1.0, top_p=0.95. The model card's large generation
budget recommendation does not turn a bounded diagnostic completion into a
full quality evaluation; a length stop must remain explicitly truncated.

Plan and observations, 2026-10-12. This profile remains experimental until the
completed-answer and tool round-trip gates pass. Main GLM checkout is preserved.

Integration update, 2026-10-13: the DeepSeek modules, parser image patch,
launcher, proxy configuration and tests now live in the main
`feature/arcon-stack` checkout. Commands below can run from that repository
root. Historical worktree paths identify the original validation runs.
The already running model/proxy still mount the original worktree; this source
transfer does not restart them or requalify a freshly built main-tree binary.
See the corrected-runtime results at the end for the passing short checks and
remaining limitations.

## Starting point

- Branch `integration/deepseek-v41-20261012`, based on experimental commit
  `308d59e`; isolated worktree under the main repository's `.runtime/worktrees/`.
- Checkpoint: `/mnt/ai/models/deepseek/DeepSeek-V4.1-Flash-Quark-MXFP4`.
  Header/index audit: all 48 shards, 96085 tensors, 474.585 GiB on disk.
- Existing image `r9700/vllm:deepseek-v41-18f8f960`; immutable image identity
  recorded with each run. Keep the scoped vLLM 18f8f960 adapters, rather than
  combining an unqualified runtime upgrade with this recovery.
- Earlier EP8/eager/staging runs served text, image OCR and concurrent short
  requests, including calibrated placement. Thinking and tools failed bounded
  quality probes even before placement. DSpark did not qualify for serving.

## Memory budget

The installed module capacities reported by the operator total 352 GiB
(3×96 + 4×16); Linux exposes 341.448 GiB. This is the usable host total, not
free RAM. Other server processes remain in place.

| Component | Payload / budget |
|---|---:|
| Engram tables, exact-size registered host storage | 188.833 GiB |
| Routed experts in checkpoint | 268.945 GiB |
| Expert offload trial | 11 GiB/rank, 88 GiB total |
| Minimum combined host payload | 276.833 GiB |
| Draft weights, excluded initially | 7.342 GiB checkpoint payload |
| Bounded loading buffer | 64 MiB/rank, 512 MiB total |
| Reserved prefill projection staging | 540 MiB/GPU |

Actual whole-tensor offload may exceed the requested budget slightly. These
numbers exclude process RSS, scratch, replication, page tables and load peaks.
Engram must not use the rounded pinned allocator (about 264 GiB for its tables
alone). No second complete CPU copy; no reliance on swap for registered weights.

Stop only GLM and its 128 GiB LMCache allocation before GPU bring-up; record
their identities and launch configuration for restoration. Re-measure actual
MemAvailable after release. Initial admission requires at least 300 GiB
available, covering the payload plus approximately 23 GiB loading/runtime
headroom. This is a diagnostic admission threshold, not a Docker memory cap.
The existing bounded host guard retains its 4 GiB emergency stop threshold.
Do not stop unrelated server services or change swap/kernel limits implicitly.

## Ordered implementation and validation

1. **Read-only preparation:** audit checkpoint geometry, inspect historical
   failures and reference changes, run existing CPU tests, verify image/kernel
   identity, and dry-run the single DeepSeek launcher. Preserve dirty GLM files.
2. **Controlled switch:** record rollback configuration; wait for GLM requests
   to finish, stop its model/cache, then recheck RAM and all eight PCIe paths.
   Save HIP/TP mapping, endpoint/bridge capabilities and idle/loaded readings.
   Correctable errors remain a hardware concern; new fatal AER, GPU page faults
   or device loss stop the diagnostic. Never attribute their timing to a kernel
   without evidence.
3. **Reuse the bounded baseline:** TP8+EP8, 48 complete experts per GPU; packed
   Quark W 4A 4, exact host offload, 64 MiB streaming loader, resident scales and
   reserved projection staging. C4/NBT512, eager, synchronous scheduler, RCCL,
   utilization 0.97, 11 GiB expert offload/rank; automatic KV/context fitting.
   No MTP/DSpark, graph capture or calibrated hot/cold placement initially.
   Preserve checkpoint quantization and dense FP8 32×32 scales. The larger
   chunk/staging optimization comes after correctness, not during this recovery.
4. **Validate the running model:** health/models/tokenization; C1 and C4
   independent keys and arithmetic; high/low/off thinking with natural EOS and
   sufficient, separately recorded output budgets; streaming; a real two-turn
   tool call/result. Record exact/retrieval/format/completion/runtime separately.
   An HTTP 200 or a token-limit stop does not qualify answer quality.
5. **If loops recur:** preserve raw completions and rendered token IDs; compare
   native DeepSeek encoding with the checkpoint reference and verify parser
   versus generated markup. Then isolate numerical paths using saved prompts
   and primitive QDQ references. Do not change multiple knobs or mask loops
   with penalties. Upstream fixes are backported only when the local code and
   reproduction establish relevance.
6. **Qualification expansion:** after correctness, bounded 8K/32K cold/warm APC
   and C4 decode, with pinned sampling and the stack measurement rules. Only
   then investigate graphs, larger prefill batches, calibrated expert placement
   and DSpark, one variable at a time. No claimed 1M capacity from startup alone.
7. **Proxy:** route LiteLLM to DeepSeek only after direct correctness passes;
   verify OpenAI/Anthropic streaming, thinking, tools and local token counts.
   Existing GLM containers/image/config remain available. Per the operator
   instruction on 2026-10-13, stay on DeepSeek during development: **do not
   restore GLM automatically** after failed qualification. Preserve the failed
   attempt and continue diagnosis; restore GLM only on explicit operator request.
   No automatic repeat load after a hardware fault.

## Reference findings and applicability

[DeepSeek-V4.1-Flash-Accel](https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel)
at `4a4e88ec2cdcf2fc7c802e1f6b7c943d897770c8` uses TP8/EP8, CPU Engram,
exact host registration and expert-only offload. Its 503 GiB machine and CUDA
kernels are not proof that our smaller ROCm host is qualified. The transport,
allocation and staging concepts are relevant; its measured speed is not a
prediction for R9700. Our old branch already implements most of these concepts.
Its Marlin TP padding issue does not automatically apply to our unpadded HIP
kernel. Graph/null-block masking fixes require checking the actual ROCm path.

[AMD checkpoint card](https://huggingface.co/amd/DeepSeek-V4.1-Flash-Quark-MXFP4)
and the local quantization config establish the Quark format; use this export's
activation and scale semantics, not assumptions from the original DeepSeek
checkpoint. [vLLM tracking](https://github.com/vllm-project/vllm/issues/56400)
documents broader model support, not a gfx 1201/offload qualification.

Raw audit, tests, memory samples and transcripts belong in ignored
`bench/results/deepseek-requalification-20261012/` in the main checkout.

## Requalification result (2026-10-12)

The server **started, but did not qualify for production**. The old branch is a
useful implementation baseline, not a verified agent-serving baseline.

- CPU tests: 73 passed, 2 skipped, 235 subtests passed on the pinned image.
- Eight GPUs passed a basic HIP operation. Loaded PCIe paths matched the
  recorded Gen 5 x 16 endpoint / Gen 5 x 4 upstream wiring. No new GPU memory fault,
  fatal AER or OOM was observed during this run. This is not a hardware soak test.
- Readiness took approximately 21 min 43 s. Per-worker model load times ranged from
  220.5 to 1094.2s; model allocation was 25.94 GiB/GPU. These startup observations
  are not prefill/decode throughput measurements or a comparison with GLM.
- Automatic KV allocation: 446076 shared tokens; max_model_len 445952. Four
  active sequences share that pool. Long-context capacity was **not** validated.
- Host RAM headroom after stopping GLM/LMCache was 312.6 GiB. During qualification
  it fell to approximately 4 GiB; swap reached 8 GiB and other services had swapped
  pages. Exact registered offload payload fits, but this leaves inadequate
  operating margin on this mixed-use host.

| Probe | Observation |
|---|---|
| Arithmetic, thinking off | Correct 51, natural EOS |
| C4 independent keys, 161/305/593/1169 input tokens | Strict 4/4, retrieval 4/4, format 4/4, completion 4/4; no observed backend error |
| Thinking low | Natural EOS;56 occurs, but final answer is duplicated and malformed. Exact FAIL; presence of 56 is not an answer-quality pass |
| Thinking high | Repetitive question marks; explicitly cancelled and retained as incomplete FAIL |
| Chat tool call | No complete parsed call after over five minutes; cancelled, no tool-result round trip executed |
| Raw completion without chat parser | Natural stop after 21 tokens, malformed DSML; input token IDs match chat request exactly |

Raw tool generation establishes that the chat parser/LiteLLM alone cannot
explain the failure. Source prompt encoding and actual `/tokenize` IDs match
the checkpoint reference. This does **not** identify which numerical/model
component is responsible. No sampling penalties or successful-only reruns were
used to hide failures. The high/tool cancellations are retained; no completed
quality pass is inferred from them.

DeepSeek was stopped after diagnostics; the original GLM/LMCache deployment was
restored. Unchanged LiteLLM returned the expected answer 4 with natural EOS through
`glm-5.3-flash-fast`; restoration health and loaded PCIe checks passed. Raw records are in the main checkout's
ignored `bench/results/deepseek-requalification-20261012/`, including the frozen
protocol, cancellation amendment, raw SSE, memory timeline and PCIe evidence.

## Remaining integration work, in order

1. **Resource margin:** qualify a text-only candidate with multimodal processor
   cache disabled and a smaller expert offload budget. For example, 10 GiB/rank
   nominally saves 8 GiB host payload against this run; actual whole-projection
   allocation must be measured. This adds roughly 1 GiB/GPU, so re-profile KV and
   workspace before declaring it feasible. Do not lower the host guard or rely
   on swap to pin weights. Vision and its additional budget are a separate gate.
2. **Numerical isolation:** validate actual checkpoint tensors and selected
   expert projections against Quark QDQ references, then dense attention/Engram
   and sparse-attention paths. Keep prompt token IDs fixed. Existing randomized
   primitive tests passing does not prove full-model quality. Avoid full-weight
   BF16 MoE materialization, which previously exhausted VRAM.
3. **Completed-answer gates:** repeat the entire frozen C1/C4/thinking/tool
   corpus after a justified fix; retain this failed attempt. Add nonstream and
   two-turn tool results only after calls are valid. Neither 21 raw tokens nor
   the incomplete stream establishes a throughput comparison.
4. **Then serving qualification:** bounded 8K/32K and APC checks, followed by
   LiteLLM/Claude/OpenCode tests. Correct metadata must reflect runtime capacity;
   the historical proxy declaration 524288 exceeds this run's 445952 limit.
   DSpark, HIP graphs, calibrated placement and larger context remain subsequent
   independently measured optimizations.

## Reference audit and opt-in WO_A candidate (2026-10-13)

Checks ran on CPU inside the pinned image, without exposing GPU devices or
restarting the GLM/LMCache service. Raw scripts, logs and results are in the
main checkout's ignored `bench/results/deepseek-reference-checks-20261013/`.

### Tool protocol correction

`<｜DSML｜ calls>` is the **correct V4.1** opening tag. The earlier interpretation
of that tag as malformed was wrong: `<｜DSML｜tool_calls>` belongs to the older
format. The malformed part of our recorded response is what follows the valid
opening: `>\n"name"`, without an invoke, arguments, or closing call block.

- The AMD and original DeepSeek `tokenizer.json`, `tokenizer_config.json` and
  reference `encoding.py` are byte-identical. Model config differs only in its
  quantization config. Earlier actual API token IDs also matched the reference.
- The pinned tool parser accepts reference-encoded V4.1 calls: two nonstream
  fixtures and six streaming fixtures (one token, seven tokens, whole output).
  Parsed function name and JSON arguments match. This is a parser contract
  check, not a model-generation or reasoning-quality test.
- The actual failed raw response produces no executable tool call, correctly.
- Spark commit `d45538f67366c62da668c35ec1afd4fcd0e4c8b4` records a successful
  tool round trip. Its script uses temperature 0; retain that as a separate
  API diagnostic, not a substitute for our temperature-1 quality protocol.
- Affinity `v41` commit `00f5a492f23b0758ede436b1400b38561b88f32a` describes
  the spaced tag itself as invalid. That interpretation conflicts with the
  authoritative encoder, so it is not evidence of our numerical defect.

### Numerical findings

The pinned ROCm source already disables V4's post-`wq_b` per-head RMS norm.
Engram q/k parameters are BF16 with dtype-aware Triton reads and explicit
strides. The lookup code uses each rank's vocabulary offset. These are source
checks against the reported Affinity/Spark bugs, not full Engram GPU validation.

A separate gap exists in **Quark WO_A activation handling**:

1. The checkpoint declares group-32 FP8 activations and 32x32 E8M0-scaled
   weights for `attn.wo_a`.
2. gfx950 uses quantized attention output and an MXFP8 grouped BMM. gfx1201
   instead takes `rocm_inv_rope_einsum`, which directly consumes BF16 inputs.
   This bypasses our ordinary MXFP8 linear activation-QDQ adapter.
3. On actual WO_A weights from layers 0, 20 and 39, with 1/4/32 synthetic
   activation rows, the BF16-input output differs from an independent group-32
   QDQ reference by 2.45–2.81% relative L2. Weight dequantization matches exactly.
   This is a contract discrepancy; it does **not** establish the cause of loops.
   The original non-Quark Python reference itself uses BF16 WO_A, so applying a
   generic fix to every model would be unjustified.
4. The scoped `deepseek_wo_a.py` candidate restores input QDQ only for Quark
   MXFP8 emulation on this pinned model class. Native quantized outputs and
   non-Quark schemes pass through. All nine checkpoint projection checks match
   the independent CPU QDQ reference exactly after the candidate transform.

The candidate is **off by default**, enabled with `R9K_DEEPSEEK_WO_A_QDQ=1`.
The launcher already forwards this variable; dry-run verified it. Registration
retains the existing ROCm/gfx1201/vLLM-18f8f960 gate and checks the `_o_proj`
interface. No common GLM/Qwen kernels or main-checkout files were changed.

Validation: **34 CPU tests and 230 subtests passed**, including nine checkpoint
projection cases, plus eight tool-parser fixtures. GPU and full-model validation
remain pending. The running GLM occupied nearly all VRAM (80–92 MiB free/card),
so no concurrent GPU test was attempted.

Next controlled window: record rollback and PCIe evidence, stop GLM/cache,
validate candidate QDQ on real GPU checkpoint projections, then repeat the
frozen completed-answer corpus with the candidate. Establish adequate RAM
headroom first as described above. If memory settings must change, record a
new protocol and do not attribute the outcome solely to WO_A. Keep the failed
attempt and restore GLM if the serving gates still fail.

Sources: [DeepSeek encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/encoding/encoding.py),
[AMD checkpoint](https://huggingface.co/amd/DeepSeek-V4.1-Flash-Quark-MXFP4),
[Spark tool evidence](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/blob/d45538f67366c62da668c35ec1afd4fcd0e4c8b4/results/boot10/vision-tools.txt),
[Affinity audit](https://github.com/neuromaniacMD/affinity/blob/00f5a492f23b0758ede436b1400b38561b88f32a/FORK.md).

### GPU and startup qualification result (2026-10-13)

The opt-in WO_A candidate passed **72/72 GPU primitive cases**: actual checkpoint
weights from layers 0/20/39, 1/4/32 rows, on all eight gfx1201 GPUs. GPU activation
QDQ matched an independent CPU group-32 reference exactly. Grouped projection
matched the same GPU operation using reference inputs exactly; relative L2
against CPU FP32 accumulation was 0.161–0.167%. These synthetic activations do
not establish full-model answer quality. PCIe capabilities/topology and loaded
links matched the recorded wiring; no new GPU memory fault was observed.

The subsequent full-model pilot uses text-only mode and 10 GiB expert offload
per rank, retaining TP8/EP8/C4/NBT512, eager and utilization 0.97. Host headroom
was 318 GiB after stopping only GLM/LMCache. A 4 GiB available-RAM guard is active.
This resource configuration differs from the previous run: **not comparable**
for throughput or attribution of changes solely to WO_A. Request bodies retain
the earlier arithmetic/C4/thinking/tool corpus. A new 180-second total request
deadline and a 256-identical-character abort bound pathological outputs;
interrupted responses remain incomplete, never a quality PASS.

All requests/protocols, candidate file hashes, image ID, rollback identities,
PCIe evidence and numerical results are preserved in the main checkout's
ignored `bench/results/deepseek-woa-gpu-20261013/`. Full-model startup failed as recorded below.


The text-only/offload-10 candidate loaded weights (26.89 GiB/GPU), but vLLM's
startup profiler reported **-0.15 GiB available KV memory** and refused to fit
even one token. No API quality requests ran. This is a VRAM-budget rejection,
not a demonstrated WO_A numerical failure. Minimum host MemAvailable was
20.21 GiB; the 4 GiB guard did not fire and Docker reported no OOM kill. Swap
nevertheless reached 8 GiB during loading. Saving host memory by reducing expert
offload did not leave a usable GPU KV budget at utilization 0.97.

During runtime initialization, the kernel log recorded 117 correctable AER
messages (log rate limiting means this is not a total hardware event count):
BadTLP on `c4:00.0` and replay timeout on `c0:01.2`, along the path to `c6:00.0`
(HIP5/EP5). No GPU page fault, fatal AER or device loss was observed. Link widths
and speeds alone do not clear those errors; resolve the PCIe issue before any
performance qualification. Do not attribute it to WO_A without evidence.

The failed DeepSeek container and transcripts are retained. The previous GLM/LMCache containers were restored and LiteLLM returned the
expected answer `4` with natural EOS. The same correctable BadTLP errors on
`c4:00.0` also appeared during GLM restoration, so they are not exclusive to
DeepSeek startup. The next full-model candidate
needs more expert offload (10.5–11 GiB/rank, measured rather than assumed), while
retaining text-only mode and the host guard. Do not solve the deficit by manually
forcing a KV allocation or raising utilization beyond this already tight value.
WO_A remains opt-in; full-model thinking and tool correctness are unverified.


### Active development policy (2026-10-13)

Remain on DeepSeek until the operator asks to switch models. Automatic GLM
restoration is disabled in the next supervisor. Next resource trial: text-only,
10.5 GiB expert offload/rank, utilization 0.97, automatic KV sizing; numerical
candidate and frozen quality requests unchanged. Evidence directory:
`bench/results/deepseek-woa-offload105-20261013/` in the main checkout.


### Offload 10.5 outcome: API starts; driver fault interrupts C4

Run `deepseek-woa-offload105-20261013` reached readiness with text-only mode,
WO_A QDQ enabled, expert offload 10.5 GiB/rank and utilization 0.97. Model
allocation was 26.10 GiB/GPU. Automatic KV budget was 0.62 GiB/GPU, with 356476
shared token capacity and `max_model_len=356352`. This is a startup estimate;
long-context requests were not qualified. Minimum host MemAvailable was
17.05 GiB; the RAM guard did not trigger.

The arithmetic request returned `51` with natural EOS (18 input, 2 output
tokens). All four subsequent key requests started at 11:11:03.338 UTC. The
kernel reported MES `INVALIDATE_TLBS` response failures on c6 at
11:11:03.363 UTC and c9 at 11:11:03.772 UTC. The diagnostic supervisor stopped
DeepSeek according to its predeclared driver-fault policy. Exit 137 resulted
from that shutdown; Docker reports `OOMKilled=false`. Do not describe this
as a vLLM OOM or a demonstrated C4/kernel numerical defect.

C4 requests remain monitor-aborted: completion NO, strict/retrieval/format
UNKNOWN. Thinking and tools were NOT RUN. Original partial JSON/SSE files
are retained unchanged; the derived summary records the interruption.

PCIe speed/width/topology checks passed, but device AER counters increased
by 359 BadTLP on bridge c4 (path to GPU c6) during this run. A root-port replay
timeout counter also increased. Host kernel is 7.0.0-38-generic with amdgpu
DKMS 7.1.3.31500000. The cause of the MES failures remains unresolved;
PCIe errors, driver/firmware behavior and the C4 allocation transition need
isolation before further model qualification. No automatic model reload or
GLM restoration was performed. Both model deployments remain stopped.

Reproduction settings (after GPU/driver health is cleared):

```bash
R9K_DEEPSEEK_WO_A_QDQ=1 NAME=<fresh-container-name> \
  bash serve/baseline-earkita/deepseek/deepseek-v4.1-flash.sh \
  OFFLOAD_GB=10.5 LMONLY=--language-model-only
```

A [ROCm issue with a similar MES signature](https://github.com/ROCm/legacy-rocm-build/issues/6630)
reports failures on a different R9700 host. It is a diagnostic lead, not proof
of an identical cause or a verified fix for this host. The Linux
[MES implementation](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdgpu/mes_v12_0.c)
identifies this message as a failure waiting for the TLB invalidation command.


### MES investigation: installed driver and competing explanations

Read-only investigation on 2026-10-13; no workload, reset, driver upgrade or
service restart. Evidence: `bench/results/deepseek-mes-investigation-20261013/`
in the main checkout (boot journal, packages, module parameters, loaded firmware,
source snapshots/hashes and memory samples).

- Host kernel `7.0.0-38-generic`; the loaded module comes from `updates/dkms`,
  amdgpu `7.1.3.31500000`, source package `7.1.3-2390945.26.04`.
  All eight cards report MES and MES-KIQ firmware `0x91`; `uni_mes=1`.
- Installed `mes_v12_0_submit_pkt_and_poll_completion` waits up to 2100 ms.
  The same error message covers an expired polling wait OR a failed firmware
  completion status; it does not distinguish them. The function returns
  `-ETIMEDOUT`. The installed `gmc_v12_0_flush_gpu_tlb_pasid` does not check
  that return or fall back to another invalidation method in this branch.
  This is a failure to confirm address-translation invalidation, not proof
  of an out-of-memory condition or a particular model kernel defect.
- The 25 ms proximity of the first MES log to C4 submission is insufficient
  attribution: the command could already have been pending during C1.
  Conversely, absence of a reset/page-fault log does not prove GPU health.
- Current-boot journal contains two matching MES response failures, both from
  this run, without a GPU page fault, ring timeout, reset or CPU soft lockup.
  Our supervisor terminated the run on these messages. A spontaneous engine
  crash or permanently wedged GPU was NOT demonstrated.
- The confirmed PCIe problem remains: 359 additional correctable BadTLP events
  on bridge c4, upstream of GPU c6, over the attempt. Similar errors existed
  before DeepSeek. Nominal link width/speed does not establish link integrity;
  the evidence does not isolate cable, bridge, endpoint, power or firmware.
- At the fault, MemAvailable was approximately 17 GiB. Sampled pgscan_direct,
  pgscan_kswapd and pswpout remained flat from 11:10:30 to 11:11:04 UTC; swap-in
  continued. Loading had caused substantial reclaim, but a new host-reclaim
  burst at the fault is not supported by these system-wide counters. Per-process
  KFD tracing would be needed to establish a USERPTR/queue-suspension cause.

The [R9700 MES report](https://github.com/ROCm/legacy-rocm-build/issues/6630)
reproduces related failures outside vLLM; it does not establish our cause or a
verified universal fix. A [separate kernel-upgrade report](https://github.com/hazyumps/flashnext-rocm10-r9700)
attributes success to a `gmc_v12_1` MMHUB fix. Our installed dispatch selects
`gmc_v12_0` for GC 12.0.1; `gmc_v12_1` is selected for GC 12.1.0. That particular
fix is not a demonstrated repair for this host. Replacing a container also does
not replace the loaded host DKMS module.

Next isolation steps: fresh PCIe preflight, then a bounded per-GPU compute and
host-transfer correctness probe with deadlines, AER deltas and kernel logs.
Compare affected c6/c9 to the remaining devices without loading the model.
If PCIe errors reproduce, isolate the c4/c5/c6 physical path before performance
qualification; any link-generation, cabling or driver change is a separate
controlled experiment. Do not silently ignore MES errors or modify the driver
invalidation path. Preserve the stopped model state until device health is
established; no automatic GLM restoration.

### PCIe correction recheck (2026-10-13)

Operator reported correcting connections. SMBIOS supplied by the operator maps
slot `PCIEx16(G5)_1` to e1 (e3/e6/e9/ec GPUs), and `PCIEX16(G5)_2` to c1
(c3/c6/c9/cc GPUs). Boot and HIP mapping were unchanged. No service was restarted.

Bounded diagnostic in the pinned DeepSeek image: each GPU separately, then all
eight concurrently; 3 seconds per GPU/phase, 64 MiB pinned host transfer buffers,
H2D/add/D2H exact checks and 512x512 FP32 GEMM checks. **16/16 passed**. No new
MES failure, GPU page fault or reset was logged. Correctable **BadTLP increased
by 6 on c4**, upstream of c6; corresponding root-port correctable count also
increased by 6 (the same reported errors, not twelve independent errors).
Four BadTLP events occurred during the isolated c6 probe. Other observed AER
counters did not change. Thus PCIe errors remain reproducible without loading
DeepSeek, while the MES failure did not reproduce in this small test.

Loaded links match the existing Gen5 x16 endpoints / Gen5 x4 bifurcated uplinks.
The generic maximum-width checker initially flagged upstream bridges supporting
x16 but intentionally connected at x4. Saved samples were re-evaluated against
the previous run's wiring; the original result was retained. No throughput or
full-model stability conclusion follows. Evidence and corrected summary:
`bench/results/pcie-recheck-20261013/`. Diagnostic container stopped; DeepSeek
and GLM remain stopped. Next physical isolation should target slot 2's c6 path,
changing only one cable/adapter/card placement at a time with power off.

Repeat requested by operator at 2026-10-13 12:13 UTC, same bounded smoke
protocol, fresh preflight and counters: **16/16 correctness PASS**, loaded
links PASS, zero new MES/page-fault/reset messages. PCIe errors remain on the
same slot-2 c6 path: c4 bridge +3 BadTLP and +1 BadDLLP; c0:01.2 root port
+5 Timeout, with +9 total correctable reports (not an additional nine errors).
Other captured counters unchanged. No inference of performance improvement
or deterioration from event counts in this short probe. Evidence:
`bench/results/pcie-recheck-20261013T121339Z/summary.json`. Test container stopped;
no model or unrelated service was restarted.

Repeat at 2026-10-13 12:32 UTC: same bounded protocol, fresh all-GPU PCIe
preflight; **16/16 correctness PASS**, loaded links PASS, no MES/page-fault/reset.
New errors remain isolated to slot-2 c6 path: c4 +9 BadTLP; c0:01.2 +3 Timeout
and +12 total correctable reports (the aggregate, not twelve additional events).
Other captured AER counters unchanged. Evidence:
`bench/results/pcie-recheck-20261013T123210Z/summary.json`. Diagnostic stopped;
model services remain stopped. This is not a full-model qualification.

### Post-correction and reboot recheck: clean bounded probe

Operator reported another connection correction and reboot. Boot time:
2026-10-13 15:17:10 Europe/Warsaw. Repeated the same bounded hardware smoke at
15:38 local time with fresh PCIe preflight. GPU identities, HIP/BDF mapping and
topology match the preceding run. **16/16 correctness checks PASS** (eight
individual and eight concurrent); loaded link speed/width checks PASS.
**No captured AER counters increased**, and no new MES failure, GPU page fault,
ring timeout or reset was logged. The c6 path that reproduced errors previously
was clean in this attempt. Both connection correction and reboot occurred, so
this does not isolate which action removed the observed symptom, nor establish
long-run or full-DeepSeek stability. No model service was restarted.
Evidence: `bench/results/pcie-recheck-20261013T133847Z/summary.json`.

### Full startup after connection correction and reboot (2026-10-13)

Run `deepseek-post-reboot-20261013T134234Z`, container
`deepseek41-reboot-20261013t134234z`, uses the same image and opt-in WO_A QDQ
candidate: TP8/EP8, C4, NBT512, expert offload 10.5 GiB/rank, utilization 0.97,
text-only, eager, automatic KV fitting. Fresh all-GPU PCIe evidence preceded
launch; loaded links retained the expected topology, Gen5 speeds and bifurcated
widths. Protocol and quality request bodies were saved before launch.

API became ready around 15:49 Europe/Warsaw. Model allocation: 26.10 GiB/GPU.
Minimum-rank KV budget: 0.58 GiB (rank 0 reports 0.62 GiB); pool 333051 tokens,
`max_model_len=332928`. These are startup capacities, not validated long context.
Minimum host MemAvailable: 16.82 GiB; swap reached approximately 8 GiB. Host
memory guard remained armed at the agreed 4 GiB emergency threshold.

| Bounded direct API check | Result |
|---|---|
| C1 arithmetic, thinking off | PASS: `51`, natural EOS |
| Four concurrent independent keys, unequal lengths | 4/4 strict, retrieval, format and completion PASS |
| Thinking low | INCOMPLETE: repetitive reasoning; 180-second client deadline |
| Thinking high | INCOMPLETE: repetitive reasoning; 180-second client deadline |
| Automatic tool call | FAIL task: natural EOS after announcing file read, zero tool calls |
| Tool-result round trip | NOT RUN: no valid initial call |
| Long context / performance / proxy | NOT RUN |

Sampling retained temperature 1.0, top_p 0.95, top_k=-1 and seed 41012.
Thinking failures are incomplete responses, not completed wrong answers or
proof of a runtime crash. The server continued streaming until the client
bound; the tool case completed normally but did not execute the requested tool.
The WO_A candidate therefore does not qualify this deployment for agent use.
No causal attribution of remaining output problems to a particular kernel,
parser or hardware fault follows from this pilot.

There were **no new MES failures, GPU page faults or resets**, and health remained
HTTP 200 after the suite. Correctable PCIe errors nevertheless recurred under
full-model load across slot 2: c1 +4 BadTLP, c4 +2, c7 +1, ca +61, and c0:01.2
itself +6. Root-port aggregate counters also reflect these reports; do not add
aggregates again. This disproves treating the clean short hardware test as full
link qualification. It does not prove that corrected PCIe errors caused the
reasoning/tool failures.

Monitor policy for this run: preserve isolated MES reports for bounded
observation; a burst of three newly observed reports within 60 seconds or a
hard GPU fault stops the diagnostic. Request deadlines and the host guard are
separate controls. No MES report occurred, so this policy difference did not
trigger. This was a quality pilot, not a matched throughput comparison.

DeepSeek remains running directly at `http://127.0.0.1:8080`, experimental and
not qualified, with its RAM guard. GLM remains stopped; LiteLLM routing was not
changed. Raw protocols, JSON/SSE, PCIe counters, logs, image/candidate identities
and the derived result are under
`bench/results/deepseek-post-reboot-20261013T134234Z/summary.json` in the main repo.

After the qualification supervisor exited, its child memory guard was no longer
running. Restarted the same 4 GiB guard as the independent transient user unit
`deepseek41-reboot-memory-guard.service`; verified active. It appends to the
same memory evidence file and targets the existing container ID.

## Quality investigation, 2026-10-13

This remains experimental: short arithmetic and C4 retrieval passed, but the
completed low/high-thinking and tool round-trip gates have not passed. A tool
parser fix alone does not establish model correctness.

- Reproduced conversation-history drift in the installed DeepSeek V4.1 parser:
  text before a tool call retained the template's two-newline separator; replay
  added another separator. Backport [vLLM #58295](https://github.com/vllm-project/vllm/pull/58295)
  in the separate `docker/Dockerfile.deepseek-parser` image layer. Adaptation:
  consume the separator only in `_events_to_delta`; consuming it again in
  `_strip_content_whitespace` removes intentional newlines in nonstreaming.
  The base runtime and GLM images remain unchanged. `docker/build-deepseek.sh`
  builds the base and then `r9700/vllm:deepseek-v41-18f8f960-parser`.
- CPU reference-encoder round-trip and plain-text streaming regression:
  **25 passed**. Run `tests/test_deepseek_parser_roundtrip.py` in that image with
  `DEEPSEEK_TEST_MODEL` pointing to the checkpoint. It reads tokenizer/config
  files only and does not load model weights or use GPUs.
- Actual checkpoint MoE weights from layers 0, 20, 39, six experts each,
  rows 1/4/32/512 and two input magnitudes: **24/24** match upstream emulation.
  This validates the sampled primitive, not the complete model or loader.
- mHC separate pre/post primitive checks with actual coefficient weights:
  **60/60** passed across eight GPUs. The active gfx1201/AITER-off dispatch
  calls the separate post/pre primitives; fused post/pre RMSNorm is not active.
  These checks do not establish correctness of inactive fused paths.
- A raw-completions numerical probe found a decode versus teacher-forced
  prefill log-probability discrepancy (32 identical continuation tokens;
  maximum absolute difference 14.79). This is a localization signal, not proof
  that either path is the reference. Layer-level tracing is required to
  distinguish token history, cache, causal masking and row-dependent math.
- Found a pinned-runtime gfx1201 correctness defect: the portable sparse
  decode kernel returns before inverse RoPE, but its caller reports all rows
  rotated. V4.1 therefore skips its standalone rotation. The model-scoped
  `deepseek_rope` adapter reports zero rotated rows and lets the existing
  caller rotate decode and prefill rows exactly once. Engram hash traces
  match; divergence is already visible in layer 0 attention with identical
  inputs. The regression suite reproduces the old defect and checks the
  fix against independent attention/rotation math, including mixed batches:
  **49 passed**, covering rows 1/4/16, SWA-only/compressed attention, all 8 GPUs.
  The bounded full-model results are recorded below.
- [#58782](https://github.com/vllm-project/vllm/pull/58782) does not fix the
  active route: this adapter uses QuarkConfig rather than DeepSeekV4FP8Config;
  fused shared experts are disabled with EP8; the Engram scale-name and bounded
  streaming-loader corrections are already present.
- [#60613](https://github.com/vllm-project/vllm/pull/60613) initializes a reused
  sparse-index output. Our installed ROCm combine creates an already -1-filled
  tensor, so that specific stale-tail fix does not apply here.
- [#59223](https://github.com/vllm-project/vllm/pull/59223) concerns split FP8 WK
  pairs in V3.2/GLM. V4.1's WK in this checkpoint is BF16 without scale pairs;
  the similar reported symptom is not evidence that this fix applies.

Raw attempts, image identities, PCIe evidence and protocols:
`bench/results/deepseek-quality-fix-20261013/` (ignored). Numerical/quality
probes are not performance benchmarks. No speedup or regression claim.

### Corrected decode runtime (2026-10-13)

Container `deepseek41-rope-20261013` uses the parser image above, the scoped
inverse-RoPE adapter, and WO_A QDQ. Launch the tested text-only configuration:

```bash
NAME=deepseek41-flash bash serve/baseline-earkita/deepseek/deepseek-v4.1-flash.sh \
  OFFLOAD_GB=10.5 LMONLY=--language-model-only
```

The launcher now defaults to the patched parser image and enables WO_A QDQ.
The vision/default-offload configuration remains outside this qualification.

- Direct arithmetic and four concurrent independent keys: 5/5 strict,
  retrieval, format and natural completion PASS.
- Thinking low/high: both completed with correct `56`, without the previous
  repetitive reasoning. Tool auto plus tool-result: valid call and `TOOL_427`.
- Additional low/high × streamed/nonstreamed tool-history matrix: first run
  **3/4 PASS**. High/nonstream produced a malformed argument containing DSML
  markers. The first FAIL is retained. Two raw-generation probes were valid;
  the complete rerun passed **4/4**, including exact result, natural completion
  and no extra tool calls. Classify the isolated failure as non-reproducible
  in that rerun, not as fixed by a new parser/filter. No argument sanitation.
- Greedy decode/teacher-forced diagnostic completed naturally after 20 tokens:
  mean/max absolute selected-token log-probability difference 0.048/0.423.
  Earlier faulty decoding generated a different continuation; these numbers
  are not a matched token-by-token improvement claim.
- Automatic startup limit is 351616 tokens, shared KV pool with C4. This is
  not a validated long-context capacity. No throughput benchmark was run.
- PCIe mapping/topology and loaded links were checked for every series.
  Correctable BadTLP recurred on bridge `ca:00.0`; no observed hard GPU fault,
  MES failure or OOM during these bounded tests. The link is not thereby
  certified stable. The 4 GiB host-memory guard remains active.

Evidence: `bench/results/deepseek-quality-fix-20261013/runtime-rope/` in the main
checkout. Preserve the original failed attempts beside reruns. Short smoke
passes do not qualify long-context, vision, broad agent accuracy or production
reliability. GLM was not restored; DeepSeek remains the active backend.

Gateway contract at `127.0.0.1:4001` (`deepseek-proxy-check-20261013`):
**12/12 PASS**: authentication, OpenAI answers and tools, Anthropic thinking,
streaming and real tool-result round trips, and local token counts with/without
tools equal to the backend. Temperature 1/top_p .95; 240-second bounded requests.
The first gateway attempt rejected `seed` as unsupported by the Anthropic-backed
LiteLLM route; preserved that failed attempt and repeated without `seed`.
This is an API contract check, not a matched sampling or performance comparison.
No global `drop_params` or tool-output rewrite was added. The existing gateway
on port 4000 keeps its GLM configuration; it was not silently switched. The test
DeepSeek proxy remains local-only on port 4001. Long-context and vision claims
in the historical proxy metadata are not qualified by this text-only run.

### Main-checkout 32K pilot (2026-10-13)

Ran the migrated code from the main checkout, with a fresh main-source HIP
library kept separately from GLM's library. Same text-only TP8/EP8, C4 server,
10.5 GiB/rank expert offload, utilization 0.97, eager and no speculation.
Startup completed; automatic max length was 344704 (capacity estimate only).

Frozen single-request pilot: 32768 tokenizer-verified input tokens, thinking
low, temperature 1/top_p .95/top_k -1, seed 41012, natural EOS, up to 1024 output
tokens. Requested a 300–350-word incident review using facts distributed through
the archive. Pinned BetterBench 0.6.0 timing with a custom SSE-capture driver;
one excluded short warmup, one measured attempt, fresh APC salts.

**32K did not produce a usable answer or decode measurement.** After 604 seconds
without a first token, stopped the diagnostic attempt early (original deadline
2400 seconds). All eight CPU stacks were in prefill
`combine_topk_swa_indices` / `torch.repeat_interleave`, with a 512-token chunk,
14336 compressed rows (approximately 28K context). This is a synchronization
point: it does not prove that repeat_interleave itself caused the delay.
VRAM was nearly full; swap I/O and AMD restore/eviction workqueue warnings were
observed. No hard GPU fault or OOM exception was observed. Exit 137 followed
explicit `docker stop`, and Docker reported `OOMKilled=false`.

The excluded 46-input/64-output warmup observed 2.48 tok/s decode and hit its
output limit. **That is not the 32K rate.** No quality verdict is possible for
the 32K request. CPU stack sampling after the stall also disqualifies this
attempt from performance comparison. Final metrics collection failed after
the intentional shutdown; the raw SSE (empty) and client result were retained.

All eight loaded PCIe speeds/widths matched expected wiring. Correctable BadTLP
on bridge ca:00.0 recurred; do not certify link stability or compare performance
with clean-link runs. DeepSeek was left stopped after this attempt; GLM was not
restored. Investigate prefill memory headroom/eviction and sparse gather/combine
before repeating the unchanged 32K workload.

Evidence, protocol, complete prompt, driver, stacks and telemetry:
`bench/results/deepseek-main-32k-20261013/` (ignored), especially `summary.json`.

### Paged prefill on gfx1201 (2026-10-13)

`R9K_DEEPSEEK_PAGED_PREFILL=1` selects the model-scoped adapter in
`r9700_vllm/compat/deepseek_prefill.py` (opt-in; the current reference profile defaults to `0`).
Set it to `0` to reproduce the former BF16 gather path. No AITER architecture
gate is bypassed: the two-cache AITER Gluon implementation requires gfx950.
Instead, upstream prefill slot mapping feeds vLLM's existing portable Triton
packed-cache reader, also used by the gfx1201 decode path.

The adapter removes the attention prefill's full-context BF16 gather allocation,
including its memory-profile reservation. SWA and compressed KV remain packed;
only selected tiles are dequantized inside the kernel. Decode dispatch, cache
format, inverse RoPE, expert offload, NBT and NSEQ are unchanged. The separate
sparse indexer still has its own FP8 gather/workspace; this change does not remove
all context-dependent temporary memory or host-memory pressure.

Scope: vLLM `18f8f960`, ROCm gfx1201, DeepSeek V4.1, eager, packed `fp8_ds_mla`,
BF16 attention output, compression 0/1/2, no KV connector. Unsupported interfaces
or output geometry fail explicitly. Nothing is installed in GLM/Qwen kernels.

Validation: 48/48 GPU cases with the actual TP8 head count passed across all eight
GPUs: C1/C2/C4 unequal query lengths, a leading decode row, SWA-only and compression
1/2, up to 512 prefill rows and 32K history. Outputs agree with independent FP32
attention over decoded cache bytes and with the original model gather path
(`atol=0.008`, `rtol=0.02`). An earlier 16-head diagnostic also passed 40/40.
Observed extra allocation for 512 rows was 1,052,672 bytes, about 1 MiB, excluding
resident inputs/output and the already-built SWA metadata. These are kernel
correctness/allocation tests, not end-to-end speed measurements. Loaded PCIe
links passed the wiring check; no new driver faults or BadTLP appeared in these
bounded kernel tests. Evidence: `bench/results/deepseek-paged-prefill-20261013/`.

A separate unchanged C1 32K prompt pilot is recorded in
`bench/results/deepseek-paged-32k-20261013/`. Its outcome must be read separately;
the earlier aborted run supplies no valid decode rate for a speedup comparison.

Pilot outcome: **FAIL, backend RPC timeout**, not a qualified 32K baseline.
Startup allocated 1.11 GiB KV per GPU and selected `max_model_len=634752`;
this is automatic capacity sizing, not a validated context length. The scheduler
reported 26,624 computed input tokens and another 512-token chunk when
`sample_tokens` timed out. The client received no content in 405.49 seconds;
TTFT, effective prefill and decode rates are unavailable. The short, excluded
warmup generated 64 tokens at 2.52 tok/s; it does not describe 32K performance.

CPU stack samples found TP0 waiting for a GPU copy event, with TP1–7 waiting for
commands. That event also depends on earlier GPU work, so this does not prove a
sampler fault. VRAM approached physical capacity, host swap traffic was active,
and correctable PCIe BadTLP reports recurred (primarily `0000:ca:00.0`, also
`c4:00.0` and upstream ports). No DeepSeek OOM exception/container OOM kill was
observed. The unrelated Tempo container's memcg OOM is recorded separately.
The failed DeepSeek container was stopped; GLM was not restored.

Further diagnosis must separate PCIe errors and memory residency from attention
correctness. Removing a workspace reservation lets automatic sizing spend the
savings on KV; it does not preserve that space as runtime headroom. A subsequent
lower-utilization experiment must retain the prompt, NBT and offload settings,
record fresh PCIe evidence and be labelled diagnostic. Do not increase the RPC
timeout or claim a speedup from these failed pilots.

### Latest upstream reference candidate (2026-10-13)

The current build script pins the latest published ROCm 10 nightly observed on
2026-10-13: vLLM `5555a2d8f7116b2b4877f66516f4128176cf722a`, image digest
`sha256:e7d4d8643501172acc5e9a6fcc3739680359613c2904ec977cde9e721e798e82`.
It already includes upstream PRs #56720 and #56638; no duplicate backports are
applied. The adapted #58295 parser patch is still needed (upstream unmerged).
The previous image tags remain available for reproduction.

The reference candidate defaults to upstream attention prefill
(`R9K_DEEPSEEK_PAGED_PREFILL=0`), utilization 0.95, 10.5 GiB expert CPU offload
per GPU, NBT 512 and a four-sequence server. Packed W4A4 experts, gfx1201 sparse
indexer, exact host allocations, Quark activation QDQ and inverse-RoPE correction
remain necessary compatibility paths. This is not an unmodified vLLM deployment
or a qualified stable baseline yet. Speculation and graphs remain disabled.

Qualification order is C1 8K then C1 32K, completed answers and separate timing.
PCIe diagnostics are omitted for this series at the user's explicit request;
results are diagnostic and must not claim comparable performance. Evidence:
`bench/results/deepseek-latest-20261013/`. The old-image backport kernel check
passed 48/48 after adding the upstream CPU sequence-length metadata to the test
fixture; it does not validate the new image.
