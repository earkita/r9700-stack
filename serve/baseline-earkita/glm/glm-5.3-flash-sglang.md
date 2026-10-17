# GLM-5.3-Flash on SGLang (experimental)

For new SGLang comparisons, the selected reference is the qualified MTP3
configuration with `PREFILL_CHUNK=2048` and `MAX_PREFILL_TOKENS=2048`. These
are already the launcher's default prefill values; freeze them explicitly in
each new protocol. Chunk 4096 remains a separate candidate: the completed
128K/256K pair found faster cold prefill but slower identical-prefix replay.
The earlier frozen rule selecting 4096 for a specific completed-agent
qualification is historical and does not change this new reference. This
decision does not replace production K4 or promote MTP4, MTP5 or DFlash2.

## C4 diagnostic with MTP3 and chunk 2048 — 2026-10-16

The first frozen C4 attempt was rejected in its excluded warm-up: SGLang
reserved four native-input framing tokens, so 131072 input + 4096 output did
not fit `context_length=135170`. Its evidence remains in
`bench/results/sglang-prefill-optimization-20261015/c4-2048-mtp3-v1/`.
The corrected, separately frozen `c4-2048-mtp3-v2/` used context limit 135232
without changing the 2048 prefill chunk, BF16 KV, MTP3 or corpus. One warm-up
was excluded; all five measured cohorts completed their four requests: three
8K sessions with 6144-token device-cache hits and one cold 128K session.

The code/format gate passed **19/20 measured answers**, or **4/5 complete C4
cohorts**. The only failure was a natural-EOS answer that appended prose after
the required single Python code fence; it is retained as `wrong_completed`.
The measured cold-prefill wall median was **99.61 s** (99.52–99.62). The three
short lanes each had a roughly **98.27 s** client-visible gap touching that
prefill; their per-request decode medians, which include that interruption,
were **2.76 / 3.16 / 3.01 tok/s**. The long lane's post-first-update decode
median was **17.24 tok/s**. C4 aggregate post-first-burst throughput was
**12.27 tok/s** (10.58–13.81), distinct from the per-request rates and from
pure C1 decode. All answers stopped naturally before the 4096-token budget.

Idle/loaded PCIe link checks passed on all eight GPUs and no GPU hang appeared,
but the kernel journal recorded **120 `BadTLP` warnings** during this series.
Treat the rates as **diagnostic, not comparable** with earlier runs. The result
does not qualify C4 for production because one cohort failed the frozen quality
gate and short-session progress stalled during cold prefill. Raw attempts,
responses, protocol, server logs, hardware captures and the derived
`summary.json` are in ignored `c4-2048-mtp3-v2/`. The owning controller stopped
only its experimental SGLang container; K4 was untouched.

## Fault gate and C4 qualification history

The overarching SGLang objective remains a correct, repeatable end-to-end
latency improvement for GLM at 128K–256K, including tool use, with MTP3 as
the preserved reference. Add a fault-triggered gate to that objective: if a
future run shows `GPU Hang`, `HSA_STATUS_ERROR_EXCEPTION`, a GPU page fault,
or an MES timeout, stop the owning experiment and diagnose the first loss of
progress **before** continuing the optimization series. Keep the raw request,
server and kernel timelines, one-second per-BDF IRQ samples, AMD-SMI samples,
HIP/worker mapping and PCIe preflight evidence together. Separate a stopped
user queue from a later MES recovery/discovery message; identify the stage and
rank if the evidence permits, otherwise record exactly what is missing. Do not
retry a faulting workload blindly or turn a diagnostic run into a performance
comparison. Preserve the MTP3 rollback; do not restart or replace K4 as a side
effect of diagnosis.

The completed long-context SGLang optimization comparisons before the C4
diagnostic above were **C1**: one active request at a time, even where a test
had multiple serial turns or seeds. C4 has now been measured but has **not**
passed the full-model four-independent-request quality and latency gate. The launcher
defaults to `NSEQ=1` and restricts full decode graphs to that setting; C4
still requires separate graph, state-isolation and capacity gates before
promotion. Do not present the diagnostic C4 result as a matched speedup claim.

Before the C4 model qualification, resolve the MTP4/MTP5 gate against the
preserved MTP3 reference. The existing matched C1 8K greedy-verify screen
completed at all three depths, but it is a different sampling mode. With the
corrected stochastic verifier, MTP3 passed 3/3 completed-code seeds and
MTP4/MTP5 each passed 2/3. Both failed outputs stopped normally and were
wrong answers, not GPU errors: MTP4 mutated its input; MTP5 used an undefined
name on an error path. The frozen quality-first protocol therefore skipped
their stochastic throughput legs. Do not select MTP4 from its faster greedy
8K result or infer a long-context gain. The next comparison must retain all
seeds, include the MTP3 control, prove effective sampling/acceptance and
rollback, and measure unprofiled 128K/256K end-to-end latency only after
completed-answer quality passes. Preserve every failed attempt; never retry
only the failing seed. A same-boot GPU fault blocks this GPU series until the
runtime health gate passes. The later C4 diagnostic ran separately while
MTP4/MTP5 promotion remained deferred.
If the three-seed gate is broadened to distinguish sampling variation from a
depth-specific regression, freeze the larger cohort first and rerun MTP3,
MTP4 and MTP5 under that same new protocol; the previous 2/3 results remain
failures of the original protocol, not erased by the new series.

The MTP4 follow-up is frozen at
`bench/results/sglang-prefill-optimization-20261015/mtp4-quality-followup-v1/`:
ten predeclared 8K code seeds plus the existing six API/tool checks per depth,
with corrected sampling, one unchanged image and serial MTP3/MTP4 legs. The
controller validates the frozen hashes and compares every launch argument
except the declared MTP depth, derived verify width and container name. On
2026-10-16 the kernel guard returned `blocked` (exit 86) **before any GPU
work**, because this boot contains the earlier c3 MES timeout and reset. No
MTP4 result or performance claim comes from this attempted follow-up.

After the host restart, the same frozen ten-seed MTP3/MTP4 screen ran on boot
`bb96f537-bf03-4891-83ff-5b2c07296399`. Both depths passed the six
answer/stream/two-turn-tool contract checks. MTP3 completed all ten coding
answers correctly; MTP4 completed nine correctly and one incorrectly, with no
truncation or request error. At seed 5678, MTP4 emitted an initial code fence,
then explanatory prose and a second code fence after a self-correction. The
task explicitly required one complete implementation without prose, so this
is a completed output-format failure. A separate CPU-only post-hoc check ran
each fenced implementation through the same 169-case harness using the pinned
image; both passed individually. This does not change the response-level FAIL.
Seed 1234, which failed in the earlier three-seed screen, passed here. This
small paired cohort does not prove a depth-specific quality regression, but
MTP4 did **not** pass the predeclared all-correct quality gate. Do not promote
it or use the older greedy 8K rate as its sampled or long-context decode rate.
The guard saw no fatal GPU fault, both variants retained eight-GPU PCIe
idle/loaded mapping and link evidence, and it logged BadTLP warnings during
the series. These warnings are retained without attributing the answer failure
to them. `comparison.json`, every generated response, server logs and hardware
evidence are in the run directory. No unprofiled 128K/256K speed comparison or
C4 model test was performed; both experiment containers stopped and K4 was
untouched.

#### ROCm/TheRock #9279 comparison — 2026-10-16 (no GPU workload)

The host runs ROCm 10.0.0 with amdgpu `7.1.3.31500000` on kernel
`7.0.0-38-generic`. The independent [R9700 report #9279](https://github.com/ROCm/TheRock/issues/9279)
reproduced an inference hang on the same ROCm/driver version in `llama.cpp`,
and also described failures with a second card and a minimal single-GPU vLLM
path. This makes a failure unique to our SGLang prefill patch less likely;
it does not prove identical root causes. Another October 16 reporter described
hard host reboots with ROCm inference. The issue is open/TODO, with no
confirmed fix; AMD reported that it could not reproduce the fault locally.

The pinned SGLang HTTP `/flush_cache?timeout=5` path always invokes
`Scheduler.flush_cache(empty_cache=True)`. After checking `is_fully_idle()`,
it resets the radix/hybrid-request/KV/draft pools and calls
`current_platform.empty_cache()`, which is `torch.cuda.empty_cache()` on this
platform. The successful HTTP 200 at 21:25:16 and logs from all ranks show
the flush returned; the c3 `MES ... INVALIDATE_TLBS` appeared at 21:25:17.
The engine then completed a full 256K cold prefill at 21:28:22 and several
cached/tool/decode turns through 21:29:07. Thus the MES line did **not** mark
an immediate, global loss of model progress. Multiple earlier flushes also
completed. A causal link from this flush to the later failure is unproven.
The installed amdgpu source routes compute-VM TLB flushes through
`amdgpu_vm_flush_compute_tlb()` -> `gmc_v12_1_flush_gpu_tlb_pasid()` -> MES;
the message can occur during ordinary mapping changes. In #9279 a CPC
`vmid:0` page fault preceded the same MES message, so MES may also be a
*secondary* failure. Our 21:25 journal has no prior GPU page fault; it does
have correctable PCIe errors. The AER on c4:00.0 immediately before MES is on
a different c0:01.2 branch than the failing c3:00.0 (c0:01.1); a c1:00.0
BadTLP appears only after MES. This does not exclude a hardware interaction.

The missing observation is user-queue progress at the first stall: our saved
samples did not include per-BDF GPU interrupt counts or MES/HQD state. The
read-only `bench/gpu_telemetry.py` now writes one-second `/proc/interrupts`
MSI totals to `irq-samples.jsonl` by the verified BDF mapping, in a separate
thread from AMD-SMI. Sampling continues if an AMD-SMI query waits its full
15-second timeout. Missing or malformed IRQ data is marked explicitly. Flat
IRQ plus busy GPU is a symptom, not a root-cause verdict. The driver also
exposes root-only debugfs `kfd/hqds`,
but that read iterates hardware HQDs on **all** GPUs; do not invoke it
automatically against a wedged production GPU. `mes_log_enable` is currently
`0` and read-only at runtime, so MES event-log capture needs a separate boot
configuration and is not part of this run. No model restart or GPU benchmark
was performed for this comparison; the current boot already contains a MES
fault, and the existing kernel guard blocks an automatic repeat.

Timeline refinement (2026-10-16, read-only review): in the final candidate
run, all eight ranks logged `Reset HybridReqToTokenPool` and `/flush_cache`
returned HTTP 200 at 21:25:16 CEST. The first `MES(1) ... INVALIDATE_TLBS`
on `c3:00.0` followed at 21:25:17. The next 4096-token chunk of the 256K
cold prefill was logged only at 21:25:22. The pinned SGLang
`Scheduler.flush_cache()` resets the radix tree, hybrid request pool, KV
allocator and draft pool, then calls `current_platform.empty_cache()` by
default. A closer review of the saved server trace shows that the entire
256K prefill completed at 21:28:22, followed by cached prefill and speculative
decode requests completing through 21:29:07. Therefore the 21:25 MES failure
and the final 21:29 HSA exception must be reported as distinct observations;
they are not proven to be one continuous GPU stall. Around 21:29:11 all eight
GPU metrics changed together to ~97–99% GFX busy, ~1% UMC activity and only
~38–42 W, while the final request stopped making progress. This pattern is
consistent with a stalled user queue, not a kernel identification. In [the
independent R9700 investigation](https://github.com/ROCm/TheRock/issues/9279),
the GPU stopped completing work minutes before the kernel logged a MES failure
on a subsequent message. Without our own IRQ/HQD samples during this interval,
we cannot tell whether the earlier MES warning was transient or a precursor.
Earlier 8K candidate faults and the October 12/13 vLLM MES logs
also prevent attributing this failure class solely to the new prefill patches.
The matched 2048/4096 prefill comparison completed with the same grouped MoE,
indexer-row and mHC-post optimizations, so those features are neither proven
faulty nor proven safe for every long tool workflow. Raw timing is in
`bench/results/sglang-prefill-optimization-20261015/tool-prefix-model-pair-v3/`.
The compact, output-free telemetry/log extraction is preserved at
`bench/results/sglang-prefill-optimization-20261015/tool-prefix-model-pair-v3/kernel-investigation-v2/offline-summary.json`.

Kernel investigation of the 2026-10-16 21:29 candidate failure: the current
boot's journal records `amdgpu 0000:c3:00.0: MES(1) failed to respond to
msg=INVALIDATE_TLBS` at 21:25:17, about four minutes before the eight ROCr
`HSA_STATUS_ERROR_EXCEPTION` messages. BDF `c3:00.0` was HIP/worker 4. The
journal also contains 83 **correctable** `BadTLP` reports between 21:10 and
21:30, plus three PCIe `Timeout` AER reports on the `c0` root complex's GPU paths, before and during
this run. A previous failed candidate also had `BadTLP` before HSA. These are
observations, not proof that PCIe caused the MES stall or that the new tool
renderer caused either. The failing request used `tool_choice=auto` without
`response_format`, outside the experimental rendering branch. Docker reports
`OOMKilled=false`; the nearest VRAM sample was not exhausted. A transient
allocation fault is not ruled out by five-second telemetry.

The controller observed only the **container** log, so it could not stop at
the earlier kernel MES message. `bench/sglang_runtime_health.py` now classifies
kernel MES/page-fault/reset events separately from correctable PCIe warnings.
The new `bench/sglang_kernel_guard.py` checks the entire current-boot journal
before launching an exclusive controller, follows new messages during its
run, interrupts only that child on a fatal event, and saves the journal and
decision. Wrap the next frozen controller with
`python3 bench/sglang_kernel_guard.py --evidence bench/results/<run>/kernel-guard -- python3 <controller>`.
It blocks a repeat attempt in the present boot because the 21:25 MES timeout
is already recorded. CPU regression tests exercise both the prior-fault and
live-fault paths. The frozen v3 controller and protocol remain unchanged.
Correctable AER requires
preserved evidence and hardware-comparability review, not an automatic claim
of a hardware root cause. Raw journal evidence is in the ignored
`tool-prefix-model-pair-v3/kernel-investigation-v1/` directory. The current
v3 run remains failed/non-comparable; no service was restarted.

The separate public R9700 report includes the same MES invalidate-TLB message
**without AER**, another reason not to equate our `BadTLP` observations with
the cause of the GPU exception. References: [ROCr exception status](https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/api-reference/api.html),
[Linux PCIe AER recovery](https://docs.kernel.org/next/PCI/pcieaer-howto.html),
and a [separate R9700 MES timeout report](https://github.com/ROCm/legacy-rocm-build/issues/6630).

#### Fault attribution research — 2026-10-16 (no GPU tests)

This failure class predates the SGLang tool-prefix patch on this host. The
vLLM GLM diagnostic journal recorded `MES(1) ... INVALIDATE_TLBS` on
`e6:00.0` at 01:07 on October 12, after separate VMID-8 memory faults and
later VMID-0 CPC faults. A DeepSeek vLLM journal recorded the same MES message
on `c6:00.0` and `c9:00.0` on October 13. These are evidence of a recurring
driver-visible symptom across runtimes, **not** proof of one root cause. The
October 12 vLLM recovery followed GPU reset, connection/power repair and a
host reboot, then only a bounded successful test; no software fix was proven.
See the fault and recovery sections of `glm-5.3-flash-c4-lmcache.md` and raw
`bench/results/glm-fault-fix-20261012/kernel-timeline.log` and
`bench/results/deepseek-woa-offload105-20261013/quality-kernel.log`.

The interrupted 256K request used `tool_choice=auto` and no
`response_format`. The opt-in tool-prefix patch requires `tool_choice=none`
*and* a JSON `response_format`, so its new branch was not entered by that
request. This rules out direct execution of that branch as the immediate
failing operation, but does not exonerate the shared model/GPU path or prove
the patch has no indirect effect. No GPU wave/kernel PC was captured; the
last AITER BF16-GEMM config log line does not identify the failing kernel.

Upstream [ROCm/TheRock #9279](https://github.com/ROCm/TheRock/issues/9279)
remains open: an independent R9700 reproducer hits the same MES
`INVALIDATE_TLBS` symptom in `llama.cpp` without vLLM, SGLang or AITER, and
the reporter also saw it with zero AER. AMD had no confirmed fix as of its
September 29 comment. [TheRock #7700](https://github.com/ROCm/TheRock/issues/7700)
documents a **different, verified** R9700 SGLang AITER-attention failure and
suggests Triton attention. This profile already selects Triton DSA/linear
attention, while AITER remains needed for the selected MoE path; turning it
off globally is not a drop-in fix. A reported AITER custom-all-reduce graph
failure ([SGLang #23580](https://github.com/sgl-project/sglang/issues/23580))
also does not match this launcher's `--disable-custom-all-reduce`. Upstream
[SGLang #37648](https://github.com/sgl-project/sglang/issues/37648) records
GLM MXFP4/EAGLE ROCm memory faults under a different GPU and concurrency;
it has no demonstrated fix for this C1 R9700 case. The gfx1201 TensileLite
out-of-bounds fix ([ROCm #8909](https://github.com/ROCm/rocm-libraries/pull/8909))
addresses a specific transpose-load training GEMM, not this unlocalized
inference fault. Do not treat an arbitrary ROCm/vLLM version bump as a fix.

Next isolated qualification, **only after a clean boot**: preserve MTP3 and
the existing image/rollback; first run a bounded reference request with the
kernel guard and complete PCIe preflight. Then test `HWQ=1` against the
unchanged HWQ-unset reference, one variable at a time, because an independent
[vLLM R9700 report](https://github.com/vllm-project/vllm/issues/41964)
found `GPU_MAX_HW_QUEUES=1` useful for MTP queue/idle behavior. That report
does **not** claim to fix MES crashes; neither do we. Reproduce the saved
256K failure protocol only after the bounded check, with coredump/kernel
capture ready and the guard stopping on the first driver fault. A pass is
only a diagnostic observation; a stability fix requires repeated complete
control/candidate runs with the same image, inputs and hardware conditions.
The current boot already contains an MES timeout and is blocked by the guard.

Latest long-workflow diagnostic (2026-10-16, postboot): the original MTP3
reference completed a bounded 8K tool task correctly after the host restarted.
A frozen 128K/256K comparison then started with the isolated stable-tool-prefix
image and chunk 4096. The candidate completed **5/6** long dependency-repair
tasks correctly (all three 128K seeds and the first two 256K seeds), each with
three tool calls and a tested final file. The final JSON turn reused at least
the whole base context minus 64 tokens according to exact-request server logs.
The sixth candidate attempt emitted `HSA_STATUS_ERROR_EXCEPTION` while its
second turn used `tool_choice=auto` without `response_format`, outside the
new rendering branch. The controller stopped its container, retained the
partial attempt and blocked automatic GPU reuse. The control leg did not run,
so this is **not a comparable performance
result** and does not establish a speedup or a cause of the GPU exception. K4,
the proxy and defaults remain unchanged; the reference is stopped. The complete
attempts passed an independent CPU-only transcript/state replay. Raw evidence,
frozen protocol, all-eight-GPU PCIe/HIP checks and partial audit are in
`bench/results/sglang-prefill-optimization-20261015/tool-prefix-model-pair-v3/`.

Latest matched postboot comparison (2026-10-16): the complete **2048/4096
prefill-chunk pair passed**, including independent raw-evidence audit,
all-eight-GPU idle/loaded checks and restoration of the reference MTP3 with
six API checks. Both legs use the same corrected-sampling MTP3 image, TP8/EP1,
BF16 KV, FP32 KDA, grouped prefill MoE, indexer rows 1024 and fused prefill mHC
post. Only chunk size and the coupled maximum prefill-token budget change.

One-output prefill mechanics: medians of five measured cycles after one full
excluded cycle per context. Times include HTTP/scheduler overhead, not just
GPU processing:

| Actual base tokens | Chunk | Cold TTFT | Identical replay | New 128 / 1024 / 8192-token tail |
|---|---:|---:|---:|---:|
| 131072 | 2048 | 100.099 s | 1.678 s | 0.522 / 1.078 / 6.419 s |
| 131072 | 4096 | 90.890 s | 2.993 s | 0.531 / 1.082 / 5.906 s |
| 262144 | 2048 | 208.050 s | 1.822 s | 0.559 / 1.164 / 6.903 s |
| 262144 | 4096 | 187.414 s | 3.269 s | 0.550 / 1.139 / 6.362 s |

Chunk 4096 reduces cold TTFT by **9.20% / 9.92%** at 128K/256K; every cold
candidate sample is below every corresponding control sample. Fresh 8192-token
tails take about 8% less time. All tails reuse the complete base in GPU, with
zero host hits and no base recomputation. The tradeoff is identical replay:
the first replay recomputes the last 4096 tokens instead of 2048 and takes
about 78–79% longer in this one-output cache-mechanics protocol. This must
not be described as an across-the-board cached-prefill improvement.

Warm native decode, five measured requests after separate cold population and
excluded warmup; TTFT and the actual first speculative burst are excluded:

| Actual input tokens | Chunk 2048 median (range), tok/s | Chunk 4096 median (range), tok/s |
|---|---:|---:|
| 131072 | 56.014 (55.424–66.170) | 55.738 (55.124–61.907) |
| 262144 | 57.797 (50.371–67.683) | 59.679 (58.985–63.522) |

The ranges overlap; these observations do not establish a decode gain or
regression. Every measured throughput response reached its 256-token cap
with EOS enabled, so these rates are not completed-task speeds. All measured
decode requests in both legs reuse the base except its final 64 tokens;
chunk-dependent replay warmup is excluded. API ITL p50/p95/p99 and update-gap
distributions are retained separately, including zero within-burst arrival
gaps; individual token GPU timing remains unknown.

Separate natural-EOS code responses pass **3/3 seeds at each length in each
leg** (12/12 overall), each with 169 independent execution cases. Six bounded
answer/stream/tool API checks pass per leg. This is one code task, not broad
quality qualification. Completed outputs differ in length; these mixed
cold/warm attempts do not establish repeatable long-agent end-to-end gains.

The frozen selection rule chooses 4096 **only for subsequent completed-agent
qualification**, accounting for the replay tradeoff. No default, K4 or proxy
promotion occurs. Full protocol, all attempts, source/image checks, spread,
PCIe evidence and rollback are in ignored
`bench/results/sglang-prefill-optimization-20261015/long-mtp3-pair-postboot-v3/`;
the independent report is `independent-audit-v1/audit.json` and API latency
addendum is `independent-audit-v1/latency-addendum.json`. K4 and the original
MTP3 images remain preserved. The owning controller restores
`glm53-sglang-agent-guided-final-v3-20261016`; do not manually start a second
all-GPU runtime while another exclusive controller is active.

The separate **DFlash2 full-model qualification** has now completed. At 8K,
3/3 natural-EOS code answers pass all 169 execution cases, plus six API/tool
checks. At 32K, 2/3 answers pass the original strict gate; seed 9012 emits two
implementations separated by prose self-correction. Both isolated code blocks
pass 169 cases in a separate CPU diagnostic, but the original
`wrong_completed` result remains unchanged. All six answers finish normally;
this is a format failure, not an observed timeout, infinite generation or GPU
fault. The 32K API leg was not run after the strict code gate failed.

DFlash2 is therefore **not promoted**. The independent audit verifies block 8,
the actual BF16 drafter, Triton attention, TP8, selector graph execution,
sampling/stream counters and all-eight-GPU PCIe evidence. Reference MTP3 was
restored with six API checks. Raw natural-length decode observations are
68–79 tok/s across these 8K/32K cases; they are **not a matched MTP comparison**.
The reported shared target cache has 237504 tokens at this configuration,
insufficient for a full 256K request plus output. No FP8 KV, HiCache or manual
pool override was introduced. The selector uses local top-16 followed by TP
candidate gathers; its generic FlashInfer fallback warning is not evidence
of a measured 2x R9700 loss. Full results are in
`bench/results/sglang-prefill-optimization-20261015/dflash2-model-qualification-postboot-v2/independent-audit-v1/audit.json`.
Completed long-agent workflow comparisons and DFlash2 long-context memory
qualification remain separate, unfinished stages.

The corrected MTP3 reference also completes the whole **three-task, three-seed
8K HTTP workflow pilot: 9/9 tasks**, spanning 45 model turns and 36 actual tool
calls. Tasks repair dependency waves, native cache-source evidence and decode
counter accounting. Each finished file passes respectively 169/19/11 independent
execution cases before the final JSON report. An offline transcript/state replay
checks that the latest written version was tested, all required files were read,
tool IDs/results and continuous counters agree, and JSON constraints appear only
after verified tests with tools disabled. No host repository files are modified
by these virtual workspaces.

Observed whole-task median times are 21.313 s, 41.962 s and 25.339 s respectively;
these three-seed observations include model and isolated tool execution, with
different cold/warm histories. They are not an A/B speed comparison or long-task
qualification. Exact CPU-prepared 128K/256K corpora now exist, but actual long
HTTP execution remains pending. Evidence:
`bench/results/sglang-prefill-optimization-20261015/repository-agent-pilot-v2/independent-audit-v1/audit.json`.

The isolated #33946 parser image also passes all six two-call model cases and
six basic API checks in each leg, with reference restoration and an independent
raw/source/hardware audit. Actual model chunking did not reproduce the rare
coalesced-final-delta bug; the separate 217 CPU parser cases reproduce and test
that exact defect. Evidence: `glm47-parser-qualification-v2/independent-audit-v3/`
under the same ignored results root. This image does not contain the tools+JSON
patch.

For #42269, the complete original 24-case control skips the tool in every case.
The first opt-in union candidate calls the tool in all 24, but only 12 final
answers pass: every explicit-thinking-False case leaves `content` empty and
puts JSON in reasoning (10 contain the right word, two echo the generic schema).
These FAILs remain unchanged. The real pinned GLM template always opens
`<think>`, whereas the disabled-thinking grammar assumed an already closed
prefix. A revised default-off candidate derives closure from the last 16 actual
rendered token IDs, emits `</think>` through the grammar when necessary, and
clarifies that answers must contain data rather than schema definitions.
It passes 23 host fixtures and 36 CPU checks with the real template, tokenizer,
154880-token vocabulary, EOS and reasoning parser. The revised full-model
matrix now passes **24/24 cases**, versus 24 missing tool calls in its paired
control. Both tool turns retain `response_format` and `tool_choice=auto`;
the final answer contains the secret obtained only from the tool result.
Streaming and non-streaming, JSON object/schema and thinking on/off cases
pass, with six basic API checks per leg and reference restoration. The
independent audit is `glm47-json-tools-qualification-v3/independent-audit-v1/`.
This qualifies the bounded non-strict GLM47 Chat Completions contract, not
Responses, every JSON Schema feature or performance. The fix remains opt-in
and separate from the #33946 image. K4, the reference MTP3 image and the proxy
remain unchanged.

An external power cut interrupted the subsequent long-agent comparison
during its excluded 8K warmup. Its partial files remain in
`long-agent-chunk-pair-v1/`. The fresh postboot attempt
`long-agent-chunk-pair-postboot-v2/` completed one correct 128K repair task in
225.096 s, including a failed implementation and successful repair. It exposed
a larger workflow cost: the initial TTFT was 99.790 s and the final JSON turn
needed another 100.815 s with **zero cached tokens**, despite GPU-prefix hits
on the intervening tool turns. These are client TTFTs, not pure kernel times.
The pair was intentionally stopped and the reference restored; its partial
results are **not a completed 2048/4096 performance comparison**.

Pinned `_process_messages` omits rendered tool definitions for
`tool_choice=none`, changing the beginning of the GLM prompt. An isolated,
default-off patch retains those definitions for text-only GLM47 structured
final answers, while retaining `tool_choice=none`, the JSON grammar and all
sampling settings. CPU replay of the exact serving/render methods with the
real saved 128K conversation confirms 10 shared prefix tokens originally
versus 131072 with preservation. It also covers the actual vision-capable
model's text path: `--language-only` does not make its model metadata
text-only. No multimodal-input support is claimed for this patch.

The separate GPU qualification `tool-prefix-model-pair-v1/` encountered
`HSA_STATUS_ERROR_EXCEPTION` and GPU coredumps during its excluded 8K task,
before any measured long task. It was stopped with no automatic reference
restart; cause and GPU correctness remain unqualified. The original model
capability guard also excluded the real serving path and has been corrected
to inspect message content. Image
`r9700/sglang:glm53-v0.5.20-gfx1201-stable-tool-prefix-v2` passes the revised
CPU replay, including unchanged sampling constraints and JSON/EOS rejection
of tool syntax. After the operator reported successful resets for all eight
GPUs, fresh PCIe capture and HIP mapping passed and the v2 API started. Its
first excluded 8K request then failed with `HW Exception ... GPU Hang`, before
the first model answer. That request used `tool_choice=auto` without a JSON
response format, so it did not enter the changed rendering branch. No operator
or hardware cause is established by this observation. Zero measured long
tasks completed; both failed attempts and raw logs remain preserved.

The owning controller was stopped, and the original MTP3 reference remains
stopped. The runtime fault filter now recognizes the actual GPU-hang signature.
The next controller also forbids automatic restoration after **any failed or
incomplete pair**, even if no known GPU-fault marker matched. Its 22 CPU
log/cleanup/restore checks pass, including the recorded HSA and GPU-hang lines,
unexpected EOF, observation/stop errors and incomplete-pair cases. These are
controller tests, not GPU recovery or model qualification. Further GPU work
requires recovery of the current runtime state first.

The preserved protocol in `tool-prefix-model-pair-v2/` compares only this
rendering flag, with both legs at chunk 4096. Later seed
requests are labelled **replays**, with observed hits rather than assumed hits.
There is no end-to-end speedup claim yet.

Experimental opt-in: `R9K_SGLANG_GLM_STABLE_TOOL_PREFIX=1` with the dedicated
v2 image. The launcher rejects an image lacking its capability label. K4,
the checkpoint, proxy and default SGLang settings remain preserved. Completed
parser and earlier prefill qualifications were not repeated after the cut.

Historical sampling qualification note (2026-10-16): the preserved image's HIP EAGLE
verify branch uses argmax even when requests specify temperature 1/top-p .95.
Historical MTP results therefore do **not** prove checkpoint-sampling quality
or sampling equivalence with target-only/DFlash. The unchanged MTP3/4/5 screen
records that effective override in `effective-sampling-audit-v1/`. A separate
image backports complete [upstream PR #37134](https://github.com/sgl-project/sglang/pull/37134),
including proposal graph lifetime and invalid-Q fixes. Nine upstream CPU gate tests and 1470 independent GPU p/q-chain controls pass;
whole-model stochastic qualification is still pending. K4 and the
original MTP3 image are unchanged. Build opt-in:
`--build-arg R9K_ROCM_EAGLE_SAMPLING_FIX=1`; default remains 0.

Earlier C1 result (2026-10-15): the workspace-v3 image completes 256K
prefill, exact-key retrieval and tool round trips. Five warm-prefix samples
measure 31.657 tok/s at 128K and 21.258 tok/s at 256K, excluding TTFT.
See [the final-image results](#final-image-c1-results-2026-10-15) for exact
lengths, ranges, cache preparation and limitations. Earlier sections retain
the sequence of rejected experiments and historical S0 configuration; they
are not descriptions of the currently running optimized container.

Launch: `bash serve/baseline-earkita/glm/glm-5.3-flash-sglang.sh`.
`DRYRUN=1` prints the container command. The profile uses a separate
container, port 8081, TP8 on eight R9700 GPUs, BF16 KV, Triton DSA,
eager execution and no draft model. It does not switch LiteLLM or stop vLLM.
`AITER_FORCE_A8W4=1` is required here for the separate dense MXFP4 linear
fallback; disabling it selected an unsupported gfx1201 AITER FP4 GEMM during
8K prefill. The routed MoE remains on the local packed GEMV adapter.
Build the pinned image with
`docker build -f docker/Dockerfile.sglang-glm -t r9700/sglang:glm53-v0.5.20-gfx1201 .`.
Compatibility patches and the packed-MXFP4 MoE adapter live in
[`r9700_sglang/`](../../../r9700_sglang/README.md).

The first full-model attempt repeated a prompt. A synthetic MoE probe found
that the pinned AITER FP4 path returned zero output on gfx1201; the isolated
packed GEMV adapter then passed that probe. With the adapter, a bounded TP8
smoke test on 2026-10-15 completed a chat answer (`12` for `7+5`), a streamed
answer (`OK`) and a two-turn `add(7,5)` tool call. PCIe idle and loaded
snapshots for all eight GPUs and raw responses are in ignored
`bench/results/glm-sglang-v0520-smoke-20261015/`.

These checks establish basic function only. Long-context quality, concurrent
requests, numerical regression against vLLM, stability and end-to-end
performance remain unqualified. Do not compare speeds until the protocol in
[`testing.md`](testing.md) is frozen and run for both variants.

## C1 8K diagnostic benchmark (2026-10-15)

With `MAXLEN=16384`, BetterBench 0.6.0 streamed the frozen 8192-token GLM
corpus at temperature 1.0, top-p 0.95, top-k disabled, seed 1234 and thinking
`high`. Each request had a unique cache salt and forced 256 output tokens.
One warmup was excluded; three requests were measured. All 16 prefill chunks
in the server log reported zero cached tokens. PCIe idle/loaded checks passed
for all eight GPUs. The raw protocol, four attempts, logs and PCIe evidence
are in ignored `bench/results/glm-sglang-v0520-bench-20261015-retry1/`.

| C1, 8K input | Median | Range (3 requests) |
|---|---:|---:|
| Decode | 4.413 tok/s | 4.411–4.413 tok/s |
| TTFT | 21.239 s | 21.227–21.303 s |
| Effective prefill (input/TTFT) | 385.7 tok/s | 384.5–385.9 tok/s |
| End-to-end request | 79.054 s | 79.004–79.088 s |

This is a **standalone diagnostic**, not a matched K4 comparison. The SGLang
profile has BF16 KV, no DFlash and eager execution; differences from K4 cannot
be assigned to one component. Forced output measures throughput, not completed
answer quality. An earlier attempt without `AITER_FORCE_A8W4=1` failed in the
8K warmup before its first token; its raw failure is preserved in
`bench/results/glm-sglang-v0520-bench-20261015/`.

## MXFP4 operator investigation (2026-10-15)

The checkpoint stores routed experts as packed E2M1 with E8M0 scales. On this
profile, the packed MoE adapter reads those weights directly and rounds
activations through Quark MXFP4 QDQ. Dense MXFP4 layers instead dequantize
**once at loading** to BF16 because the pinned AITER FP4 GEMM is unavailable
on gfx1201. Their decode path calls BF16 `linear`; it does not repeatedly
dequantize or transfer the weights from CPU. Checkpoint layer-22 gate/up/down
slices matched an independent CPU E2M1/E8M0 decoder in 9/9 checks. The CPU
oracle and tests are in `r9700_sglang/mxfp4_reference.py` and
`tests/test_sglang_mxfp4_reference.py`. Quark activation QDQ is shared with
the runtime in the GPU operator probe, not independently reimplemented.

The first MoE experiment grouped redundant gate/up scale reads without
changing arithmetic (`R9K_SGLANG_GATE_SCALE_GROUP=1`). Full routed operator
outputs were identical to S0 for synthetic 1/5/20 rows. At one row the
candidate took 0.193 ms versus S0's 0.182–0.187 ms, so it was rejected for
the C1 objective. It was about 6% and 4% faster at five and 20 rows; those
single-GPU operator results do not establish model decode speed. Raw protocol,
all timings, numeric checks and PCIe evidence are in ignored
`bench/results/sglang-gate-scale-20261015/`.

The next isolated candidate applies a Triton BF16 GEMV only to one-row dense
layers that SGLang already dequantized at load time. Its one-GPU synthetic
test passed all six representative shapes and five launch configurations,
with exact BF16 output against the existing linear in this sample. Best
operator times were 1.3–1.9× lower, depending on shape. This is **not** a
model-level speedup. The frozen operator protocol, raw A/B/A timings and
loaded PCIe checks are in `bench/results/sglang-bf16-dense-20261015/`.
`R9K_SGLANG_BF16_DENSE_GEMV=0` preserves S0; `=1` selects the experimental
decode path. A matched whole-model S0/S1 series is required before promoting
it. The original S0 container/image remain available for rollback.

A 16-step C1 8K Torch trace recorded a median host step interval around
226 ms and 313 `aiter::gemm_a16w16` CPU calls per step. Its GPU kernel events
were absent, so nested host annotation times cannot identify a GPU critical
path. `rocprofv3 --pid` attach also failed because the process did not expose
the required attach thread, including with `ROCP_TOOL_ATTACH=1`. These are
missing observations, not evidence that BF16 GEMM dominates total decode.
Raw traces and failures remain in ignored
`bench/results/sglang-step-trace-20261015/` and
`bench/results/sglang-rocprof-20261015/`.

## Matched dense decode experiment (2026-10-15)

S0 and S1 used the **same pinned SGLang image**, model, TP8 mapping, BF16 KV,
eager scheduler, no speculation, 8,192-token corpus, high thinking,
temperature 1.0, top-p 0.95, top-k disabled, seed 1234 and 256 forced output
tokens. Only `R9K_SGLANG_BF16_DENSE_GEMV` changed from `0` to `1`. Each leg
excluded one warmup and kept all five measured attempts. Fresh salts intended
cold APC; server prefill logs showed zero cached tokens. Both legs passed
eight-GPU PCIe checks under load with the same HIP mapping and power limits.

| C1 8K, five measured requests | S0 linear | S1 one-row BF16 GEMV |
|---|---:|---:|
| Median decode | 4.762 tok/s | 4.886 tok/s |
| Decode range | 4.752–4.769 | 4.859–4.906 tok/s |
| Median TTFT | 21.231 s | 21.102 s |
| Median whole request | 74.792 s | 73.297 s |

The decode median improved **2.6%** and whole-request median improved
**2.0%** in this matched, forced-output test. It is an observed end-to-end
gain for SGLang C1 8K only. Do not compare it with historical K4 or infer
128K/256K performance. Raw requests, protocols, server logs and PCIe evidence
are in ignored `bench/results/sglang-dense-ab-20261015/`. The new kernel
preserves the once-dequantized BF16 weights; it is not a native packed-MXFP4
dense GEMM and does not run on prefill. A separate bounded S1 API smoke test
passed arithmetic, streaming and two-turn `add(7,5)` tool calling with loaded
PCIe evidence in `bench/results/sglang-dense-ab-20261015/s1/contract/`.

The next independent MoE experiment is `R9K_SGLANG_DOWN_BN=64` versus the
default `8`, with dense GEMV held fixed. The change is prepared but remains
off until its numerical and full-model tests pass. Historical vLLM data for
this tile do not establish an SGLang gain.

## Long-context S1 diagnostic (2026-10-15)

The same BF16 dense candidate was started with a 131,072-token model limit
and no draft/graphs. This was a standalone diagnostic, **not** a matched S0
or K4 comparison. The 8K/32-token warmup completed (4.729 decode tok/s).
The 32K/128-token request completed with 84.659 s TTFT and 2.880 decode
tok/s. A 130,816-token request reached decode after 343.965 s and produced
38 tokens at an observed 3.005 tok/s, then returned `finish_reason=stop` even
though the client requested 128 output tokens with `ignore_eos=true`. It did
not meet the frozen output-length requirement, so this attempt does not
qualify as a 128K throughput result. No runtime exception or GPU OOM was
recorded. During its decode the server repeatedly warned that the indexer
tail kernel was device-loaded after startup with about 0.33–0.38 GiB free
per GPU. This is a memory-headroom observation, not an established cause of
the early stop. All eight PCIe links passed idle, warmup-loaded and 128K
prefill-loaded checks. Raw protocol, requests, logs and topology are in
ignored `bench/results/sglang-long-s1-20261015/`.

## Candidate operators already present in the vLLM plugin

The vLLM packed MXFP4 GEMV in `r9700_vllm/moe/packed_gemv.py` is already
adapted as `r9700_sglang/packed_moe.py`; it is not a new backend to port.
Its down-projection BN64 setting is prepared as an isolated SGLang experiment,
and now has an isolated SGLang operator result: original BN8 and BN64 had
bitwise-identical outputs for 1, 5 and 20 rows. A–B–A median full-MoE GPU
times were 0.187/0.178/0.184 ms at C1, 0.339/0.313/0.334 ms at five rows,
and 1.047/0.924/1.049 ms at 20 rows. This establishes an operator-level gain
only. Raw data and
one-GPU loaded PCIe evidence are in ignored
`bench/results/sglang-down-bn-20261015/`.

A separate, frozen whole-model BN8/BN64 comparison used one pinned candidate
image, the same dense GEMV setting, 8,192 input tokens, 256 forced output
tokens, one excluded warmup and five measured requests per variant. Both legs
passed eight-GPU PCIe checks and all measured requests completed. Only down BN
changed. The test was A then B; it has no return-to-A leg, so possible
time drift is a remaining limitation. This series is not comparable with the
earlier dense-GEMV A/B or K4.

| C1 8K, five requests | BN8 | BN64 |
|---|---:|---:|
| Median decode | 4.436 tok/s | 4.385 tok/s |
| Decode range | 4.424–4.442 | 4.373–4.401 tok/s |
| Median TTFT | 21.196 s | 19.930 s |
| Median whole request | 78.697 s | 78.061 s |

The wider tile improved the isolated MoE time and observed TTFT but **did not
improve model decode**: its median decode was 1.14% lower in this series.
Keep BN8 as the decode default; BN64 remains an opt-in diagnostic. Raw
protocols, all attempts, server logs and PCIe evidence are in ignored
`bench/results/sglang-down-bn-ab-20261015/`.

The larger candidate is the gfx1201 HIP WMMA W4A4 path in
`kernels/r9k_moe_mxfp4a8.hip` and `r9700_vllm/moe/w4a4.py`. It keeps E2M1
weights packed and transports Quark-MXFP4-QDQ activations through FP8 with
power-of-two group scales. The existing C1 entry point takes original Quark
weight and scale layouts, so this candidate needs **no weight repacking**.
Porting it requires a SGLang-specific route/alignment adapter and a build of
the existing HIP kernel in the pinned SGLang image. Keep `R9K_FOLD=0`; prove the
activation encoding, route/alignment, full routed operator and SGLang numerics
before any decode comparison. The first isolated C1 A–B–A test showed exact
full-operator output against packed GEMV, exact reconstruction of Quark-QDQ
activations from FP8 plus group scales, and a slower median GPU time:
0.190/0.256/0.190 ms. The independent dequantized oracle differed from both
by relative L2 0.0034 in this synthetic sample. The HIP candidate stays
disabled by default, without a full-model throughput claim. Raw data and
one-GPU loaded PCIe evidence are in ignored
`bench/results/sglang-hip-w4a4-c1-20261015/`.

`r9700_vllm/compat/glm_head_gemv.py` is another possible small-row operator
for the sparse-indexer head projection. It is **not** a direct adapter port:
this pinned SGLang HIP path invokes a separate `weights_proj` ReplicatedLinear
and returns BF16, whereas the vLLM hook replaces a fused `torch.mm` with
FP32 output. The shape, output semantics and numerical behavior must be
checked before using the router kernel here. The vLLM FP8 paged-logits path
targets FP8 KV and therefore does not directly apply to this BF16-KV profile.

## mHC reference-loop attribution (2026-10-15)

The earlier MoE/dense investigation missed a large reference path:
`sglang/kernels/ops/layernorm/mhc.py:_mhc_pre_torch`. With the launcher's
TileLang mHC flags disabled, every one of 45 layers calls this pre block
twice. Each call uses an FP32 projection `[M,16384] @ [16384,24]` and twenty
eager Sinkhorn iterations. The existing eight-rank CPU trace has 16 decode
steps per rank, approximately 29,463 CPU operator records per step, 3,523
`aten::div` and 3,737 `aten::sum`. The reference Sinkhorn loops structurally
account for 3,510 divisions and sums per step. These are operation counts,
not GPU critical-path percentages. The trace has no GPU kernel events.

The recorded vLLM trace actually contains
`mhc_pre_big_fuse_with_norm_tilelang_kernel`, `mhc_fused_tilelang_kernel`
and `mhc_post_tilelang_kernel`; mHC is therefore a concrete implementation
difference, in addition to its graphs, DFlash and FP8 KV. This does not
attribute the entire historical vLLM/SGLang throughput gap to mHC.

An isolated, eager one-GPU A/B/A measured the full mHC pre operator for
1/5/20 rows. For one row its median GPU event interval was
2.341/1.588/2.342 ms, and synchronized host wall time was
2.389/1.653/2.386 ms. The interval includes gaps between kernels and is not
the duration of one kernel. Only the Sinkhorn portion changed, reusing
the pinned SGLang Triton implementation; the FP32 projection and residual
mixing stayed on Torch. Nine numerical cases (three row counts and input
scales 0/0.1/2) passed predefined tolerances. The BF16 layer input was exact
at C1 in these samples; small FP32 mix differences remain. The test used
synthetic inputs, not a complete quality corpus. PCIe idle/loaded evidence,
source snapshots and results are in ignored
`bench/results/sglang-mhc-attribution-20261015/`.

`R9K_SGLANG_MHC_SINKHORN=1` opts into this decode-only experiment for
1..32 rows on gfx1201. Its default is `0`; prefill and other row counts
retain the reference path. A separate same-image C1 8K full-model A/B
protocol is frozen in `bench/results/sglang-mhc-ab-20261015/`, with five
measured 256-token requests and one excluded warmup per leg. Results from
that series must be inspected before any model-speed claim. The preserved
S1 container is its runtime rollback; production K4 is not changed.

### Historical vLLM evidence used to choose follow-up experiments

The following are separate historical protocols, not a single cumulative
benchmark and not matched comparisons with current SGLang:

| Evidence in ignored `bench/results/` | Observed decode | Scope / limitation |
|---|---|---|
| `glm-w4a4-candidate/short-comparison.json` | 5.067 → 9.099 tok/s | Initial W4A4 candidate, three short C1 chat prompts; sequential, not thermally paired |
| `glm-w4a4-graph-ab/comparison.json` | graph 26.701 → eager 9.825 → graph 26.795 tok/s | Same image/settings except execution mode, no speculation; historical short C1 A/B/A, TTFT excluded |
| `glm-w4a4-ar-tp8/comparison.json` | RCCL 26.796 → custom small-message AR 30.495 tok/s | Short sequential C1; uncompressed sum, not a blanket TP8 communication qualification |
| `glm-mtp-20261014/dflash-c1-8k32k/summary.json` | K4 median 58.103 / 58.691 tok/s at 8K / 32K | Three cold, fixed-output samples; separate protocol from the target-only chat runs |
| `glm-hwq-vllm771-20261014/report.md` | HWQ2 median 34.06; HWQ1 attempts 52.98 / 77.18 / 86.51 tok/s | C1 130816 actual input tokens; draft acceptance changed 30.6% → 52.4%, so not isolated proof of queue latency or a stable 80 tok/s baseline |

The graph A/B/A is concrete evidence for a graph experiment after eager
correctness. The current SGLang launcher explicitly disables graphs, uses
the reference Torch mHC path and does not set `GPU_MAX_HW_QUEUES`; the vLLM
profile sets it to 1. Qualify graphs and queue scheduling independently,
after the frozen mHC comparison; do not combine their changes or multiply
these historical speed ratios to predict SGLang throughput.

The first `sglang-mhc-ab-20261015` full-model series was interrupted by a
host reboot: only its warmup is complete and the next JSONL record is
incomplete. It yields no qualified measured result. The frozen rerun is
`sglang-mhc-ab-postboot-20261015-1426`, with fresh PCIe evidence and a saved
boot ID. Do not merge attempts across the two runs.

The post-boot full-model comparison completed with the same image, TP8,
BF16 KV, no graphs/draft, dense GEMV enabled, 8,192 actual input tokens,
high thinking, checkpoint sampling, fresh salts, 256 forced output tokens,
one excluded warmup and five measured requests per variant. Eight-GPU PCIe
checks passed for both legs. Only `R9K_SGLANG_MHC_SINKHORN` changed.

| C1 8K | Reference Torch | Triton Sinkhorn |
|---|---:|---:|
| Median decode | 4.755 tok/s | 6.033 tok/s |
| Decode range, five requests | 4.740–4.760 | 6.027–6.050 tok/s |
| Median TTFT | 21.239 s | 21.146 s |
| Median whole request | 74.864 s | 63.411 s |

This A→B series observed 26.9% higher median decode and 15.3% lower whole
request latency. It supports the eager Sinkhorn loop as a material decode
cost; it does not attribute the rest of the vLLM/SGLang gap. There is no
return-to-A leg, so possible time drift remains a limitation. This is an
8K C1 result, not a K4 or long-context comparison. Forced outputs do not
qualify completed-answer quality. A subsequent bounded arithmetic, streaming
and two-turn tool round trip passed all four requests, with separate loaded
PCIe evidence. Raw attempts, hashes, all settings and `comparison.json` are
in `bench/results/sglang-mhc-ab-postboot-20261015-1426/`.

### Full mHC pre operator from the pinned vLLM package

`r9700_sglang.mhc.pre_tilelang` reuses the runner-independent
`vllm.model_executor.kernels.mhc.tilelang.mhc_pre_tilelang` function already
present in the immutable base image. It does not instantiate a vLLM runner
or change shared plugin/kernel files. This is a separate operator from the
Sinkhorn-only variant: projection, normalization and residual mixing are
executed by the upstream fused implementation, with a different accumulation
order. It is not bitwise identical to Torch.

Its isolated post-boot one-GPU numerical test passed all nine predefined
cases at 1/5/20 rows. A/B/A one-row median GPU event intervals were
1.120/0.0668/1.120 ms, and synchronized wall medians were
1.155/0.101/1.157 ms. At the timed C1 input, BF16 layer-input relative L2
was 8.6e-6, with 99.902% exact elements. These are synthetic operator
measurements, not a model-throughput multiplier or broad quality proof.
The protocol, complete errors/timings and loaded PCIe evidence are in
`bench/results/sglang-mhc-vllm-op-20261015/`.

Select it with `R9K_SGLANG_MHC_BACKEND=vllm_tilelang`; `torch` and
`sinkhorn` select the two references. The old `R9K_SGLANG_MHC_SINKHORN=1`
spelling maps to `sinkhorn`; conflicting settings are rejected. All variants
retain the existing prefill path above 32 rows, and the default is `torch`.
The separate same-image full-model comparison is frozen in
`bench/results/sglang-mhc-full-ab-20261015/`; it changes only this backend
from `sinkhorn` to `vllm_tilelang`. Inspect its results before making any
whole-model performance claim for the fused operator.

`DECODE_GRAPHS=1` prepares a separate full decode graph for C1 only, with
prefill graphs disabled. `HWQ=1` explicitly sets `GPU_MAX_HW_QUEUES=1`;
leaving `HWQ` unset preserves the earlier environment. Both remain separate
experiments; prepared launch arguments are not successful runtime tests.

### Resumed full mHC comparison, 2026-10-15

The user stopped the earlier series after its baseline completed, before
candidate measurements. Both throughput legs were repeated in
`bench/results/sglang-mhc-full-ab-resume-20261015-155340/`; the original
attempts remain intact. This same-image eager C1 comparison changes only
`R9K_SGLANG_MHC_BACKEND=sinkhorn` to `vllm_tilelang`. The candidate's explicit
`SERVER_SEED` matches the baseline's recorded initialization seed; request
sampling seed remains 1234. An unused candidate startup with an automatically
chosen initialization seed was preserved separately before any timed requests.
Container environment/arguments and recorded initialization seeds were checked.
Eight-GPU PCIe idle/loaded checks passed for both legs on the same boot.

| C1, 8192 input / 256 forced output, five measured requests | Sinkhorn only | Full mHC pre |
|---|---:|---:|
| Median decode, TTFT excluded | 6.069 tok/s | 8.266 tok/s |
| Decode range | 6.052–6.087 | 8.158–8.367 tok/s |
| Median TTFT | 21.215 s | 21.175 s |
| Median whole request | 63.246 s | 52.010 s |

Observed median decode increased 36.2%; whole-request latency decreased
17.8%. This is an A→B result without a randomized or return-to-A leg, not
a K4 comparison or long-context result. No profiler ran in these requests.
The separate six-request answer, streaming and tool-result contract passed.
The first expanded contract exposed a client bug: explicit `tool_calls: null`
made `bench/api_smoke.py` iterate over `None`. The collector now accepts null;
a CPU regression test and a complete six-request rerun passed. Original failed
attempts are retained; no model/kernel change was needed for this client error.

Full decode graphs were qualified separately at C1 8K in
`bench/results/sglang-mhc-graphs-resume-20261015-155340/`. The protocol declared
dual use of exactly the full-mHC eager samples above before their execution;
the graph leg changes graph mode only. Runtime environments and image IDs
match; every recorded decode batch reports graph replay. The same six-request
API contract passed in eager and graph mode.

| C1 8K, same full mHC pre | Eager | Full decode graph |
|---|---:|---:|
| Median decode | 8.266 tok/s | 20.041 tok/s |
| Range, five measured requests | 8.158–8.367 | 19.912–20.075 tok/s |
| Median TTFT | 21.175 s | 21.095 s |
| Median whole request | 52.010 s | 33.822 s |

This separate A→B observed 142.5% higher median decode and 35.0% lower whole
request latency. Graph capture consumes about 1.41 GiB additional VRAM per
GPU. This does not establish a K4 gain, C2/C4 graph correctness, long-context
quality, or broad numerical equivalence. Prefix salts are fresh for every
throughput request; warmups are excluded by the predeclared rule. TTFT is
the first generated token, which may be reasoning, not time to final answer.
`SERVER_SEED` fixes initialization independently of each request's sampling
seed. Leave `HWQ` unset to reproduce this series.

Long-context natural-EOS quality diagnostics and separately frozen warm-prefix
throughput live in `bench/results/sglang-long-quality-resume-20261015/` and
`bench/results/sglang-long-warm-resume-20261015/`. They use full retained
transcripts for answer scoring; BetterBench's throughput records alone do not
contain generated text. These are distinct protocols from the cold 8K A/B.

The C1 128K runtime passed three bounded natural-EOS cases with checkpoint
sampling (temperature 1, top-p .95, seed 1234, effort high). The Python indexing
case returned `[2, 14]`; exact-key retrieval passed at 32,765 input tokens
(needle near the end) and 130,816 input tokens (needle at 35% depth). Both NIAH
cases passed strict answer, retrieval, output format and completion separately.
There were no observed request-level runtime errors. This is one seed and one
needle per length, not broad quality or equivalence qualification. The 128K
case's TTFT was 345.861 s and whole time 348.557 s; its short, 55-token natural
answer is not the five-repeat BetterBench throughput workload. Reasoning counts
are reported by SGLang's top-level `usage.reasoning_tokens`: 432/439 completion
tokens for Python, 17/34 for the shorter NIAH, and 39/55 for the longer NIAH.
Original rows and separate derived accounting are retained.

The first prepared warm-throughput corpus accidentally selected only the
system message. The runner's exact long-input guard rejected it before any
generation; the rejected corpus/protocol/log remain in
`bench/results/sglang-long-warm-corpus-rejected-20261015/`. The corrected
engineering-guide corpus has exactly 130,816 API-tokenized input tokens. It
uses a separate salt, one cold excluded population request, then five
identical-prefix timed requests with cache-hit evidence required. Do not
include the rejected preparation as a model/runtime failure or speed result.

`MAMBA_CACHE_SIZE` optionally limits KDA/Mamba state slots through upstream's
`--max-mamba-cache-size`. These are state slots, not tokens or sessions. Unset
preserves upstream's automatic memory split (113 slots, about 1.96 GiB per GPU
in this C1 runtime). A smaller state pool can leave more of the same static
budget for KV, but changes prefix-state retention and requires its own cache
and concurrency checks. The prepared 256K experiment uses 32 slots and keeps
the static fraction at .82, BF16 KV, default FP32 recurrent SSM state, and
the existing arithmetic. State-slot capacity does not change state precision.
Preparation alone does not qualify 256K or its throughput.

Pinned SGLang reserves two positions when clamping generation: its worker
reports `context_len-1` and its scheduler subtracts one more. The first
130,816-input/256-output population request at server context 131,072 therefore
produced only 254 tokens. It remains an excluded, rejected attempt in
`bench/results/sglang-long-warm-budget-edge-20261015/`, not a five-sample result.
The new run keeps the exact input and output budgets and uses server context
131,328; its corpus is unchanged and all attempts are retained. The prepared
256K run similarly reserves server context 262,400 for 261,888 input + 256
output. Client preflight now checks both reserved positions before generation.

The corrected 128K warm-prefix series completed all five measured requests:
median decode 19.935 tok/s (19.924–19.940), median warm TTFT 0.721 s
(0.671–5.121), and median whole request 13.515 s (13.459–17.911). The
excluded cold population request had TTFT 348.583 s. Observed prefix hits were
129,024 tokens on the first warm request and 130,752 on the remaining four.
Late tail-prefill kernel loading affects the first warm TTFT; all five samples
remain in the range and median. This is a standalone 130,816-input/256-output
measurement, not an improvement comparison with K4 or with the cold 8K series.

### 256K capacity and shared page-table mapping

With 32 KDA/Mamba state slots, static fraction .82 and unchanged state precision,
the runtime allocated 302,336 KV tokens (3.58 GiB per GPU), versus 184,832
tokens (2.19 GiB) with the automatic 113-slot split. State memory changed from
about 1.96 to 0.57 GiB per GPU. These are logical engine capacities, not token
counts to multiply by eight. Capture still uses about 1.40 GiB per GPU.

The first 256K test then failed during prefill on a separate allocation:
the ROCm fallback expanded the shared page table with
`page_table.index_select(0, page_table_row_index)`. For 2048 query tokens and
262,400 table columns, this materializes a 2 GiB table. The complete failed
series remains in `bench/results/sglang-long256-quality-resume-20261015/`:
two short cases passed, the long request failed with a runtime/transport error,
and no throughput series started. It is not a wrong-answer quality result.

`r9700_sglang/kpool.py` now performs the final integer group expansion, tail
and physical-token mapping through the original shared table inside Triton.
Score computation and `torch.topk` selection stay unchanged. A separate patch
selects this path for HIP queries with `page_table_row_index`; other paths
remain unchanged. No full-context table is copied per query row.

The single-GPU probe passed eight exact small/reference cases, eight graph
replays with changing metadata, and the full failed allocation shape. Its old
extra PyTorch peak was 2,272,264,704 bytes; the new extra peak was 16,801,792
bytes, exactly the required 2048×2051 int32 output. This proves integer
correctness/allocation behavior for the tested cases, not model speed.
Raw evidence and the frozen protocol are in
`bench/results/sglang-kpool-map-fix-20261015/`.

Image `r9700/sglang:glm53-v0.5.20-gfx1201-kpool-map`
(`sha256:d27bef0c72debdb9e7ce3d71a34bede89abc395011a1c70db05877b6fd148de3`)
is built by the existing Dockerfile, without live-container edits. The full
three-case rerun and subsequent API/throughput gates are in
`bench/results/sglang-long256-shared-map-20261015/`; its results must be
inspected before claiming 256K qualification.

That shared-map-only rerun passed the two short cases but hit a second OOM
later in the long prefill: `gather_positions.clamp(...)` requested 632 MiB.
It failed before a proposed controlled stop; classify it as a runtime error,
not as a manually interrupted passing test. The original ROCm score fallback
builds full query×history int64 index and clamp matrices, boolean masks, and
duplicate FP32 scores in addition to the useful logits.

`mask_pooled_logits` now performs the same score shift, validity mask and
short-history padding in one Triton pass. It retains a single FP32 output;
`torch.topk` itself is unchanged. Integer mapping specializes on the saturated
group budget rather than every growing history length, and query count is a
runtime scalar. The score kernel also keeps shape/stride scalars dynamic.
This removes the observed repeated device-loading of expansion variants.

The final operator probe in
`bench/results/sglang-kpool-workspace-v3-final-20261015/` passed exact integer
mapping, exact FP32 score bits (including NaN, ±Inf and signed zero), eight
replays of each operator with changing metadata, and four existing integrated
top-k cases in the built image. At 2048×40448 scores, extra peak allocation
changed from 1,657,090,048 to 331,350,016 bytes. At 2048×65536 it changed from
2,684,895,232 to 536,870,912 bytes. These allocation/correctness observations
are not a model-speed comparison.

The full rerun uses image
`r9700/sglang:glm53-v0.5.20-gfx1201-kpool-workspace-v3`
(`sha256:a1fe74d65ce961f445407b2e71190ed89b64bbc56ca87359a7c0cd5803907fb4`).
Evidence is in `bench/results/sglang-long256-workspace-v3-20261015/`.
After quality and API gates, the final-image C1 matrix records 8K, 32K, 128K
and 256K separately, with one cold population excluded and five timed
warm-prefix requests per context. Its protocols are frozen before timing in
`bench/results/sglang-c1-final-warm-matrix-20261015/`. No comparison with K4
or promotion to production is implied by these standalone measurements.

## Final-image C1 results (2026-10-15)

The current workspace-v3 image completed all four contexts. Each frozen case
uses its own fresh salt, one excluded cold population, then **five measured
warm-prefix requests**, 256 actual output tokens, temperature 1.0, top-p .95,
no top-k filter, request seed 1234 and thinking `high`. Throughput requests
force the output budget; they do not score completed-answer quality. BetterBench
0.6.0 counts actual tokens. Decode excludes TTFT; C1 has one active request,
so there is no independent concurrent aggregate rate or speculative acceptance.
No profiler was active.

| Actual input | Decode median [min–max], tok/s | Warm TTFT median [min–max], s | Warm whole request median [min–max], s |
|---:|---:|---:|---:|
| 8,192 | 31.914 [31.900–31.924] | 0.429 [0.428–5.296] | 8.420 [8.418–13.284] |
| 32,768 | 31.863 [31.831–31.873] | 0.495 [0.492–5.347] | 8.498 [8.495–13.358] |
| 130,816 | 31.657 [31.638–31.673] | 0.706 [0.679–4.994] | 8.757 [8.734–13.050] |
| 261,888 | 21.258 [20.387–21.269] | 0.938 [0.928–5.419] | 12.944 [12.933–17.412] |

The first warm request in each case still computes a larger uncached suffix;
its roughly 5-second TTFT remains included. Later warm requests reuse all
but 64 input tokens. None of the five measurements was discarded.

Cold population observations are separate, **one sample each**, not a
five-repetition prefill comparison:

| Actual input | Cold TTFT, s | Effective prefill (input/TTFT), tok/s |
|---:|---:|---:|
| 8,192 | 21.165 | 387.06 |
| 32,768 | 84.390 | 388.29 |
| 130,816 | 341.651 | 382.89 |
| 261,888 | 692.159 | 378.36 |

The ~21 tok/s at 256K is an observation for that workload, not a causal
comparison with 128K: prompt content, length and runtime history differ.
Likewise, do not divide these warm results by the earlier cold 8K rates or
claim a gain over production K4. The controlled mHC and graph comparisons
above remain the evidence for their respective whole-model improvements.
No current-image operator/critical-step timings were collected during this
matrix; do not substitute summed kernel timings for end-to-end decode.

All eight HIP/BDF mappings, bridge/root paths, supported links, NUMA nodes
and power-limit snapshots matched across the four cases. Loaded links passed
each case. The matrix retained all 24 requests (four excluded populations
and 20 measured requests), per-request server logs, protocols, corpus hashes
and hardware evidence in
`bench/results/sglang-c1-final-warm-matrix-20261015/`.
The checked consolidated report is
`bench/results/sglang-long256-workspace-v3-20261015/final-matrix-report.json`.
These are ignored local results, not tracked benchmark dumps.

### Correctness scope

On this same image, three natural-EOS quality tasks completed correctly:
a Python indexing exercise, exact-key retrieval at 32,765 input tokens, and
exact-key retrieval at 261,888 input tokens. The long key was placed at 35%
of the context. Strict answer, retrieval, format and completion passed;
no runtime error occurred. Six additional API requests passed normal answer,
streaming, and streamed/nonstreamed tool-call plus tool-result round trips.
The operator suite separately passed exact integer mapping, bit-exact FP32
score masking, changing-metadata graph replay and integrated pooled top-k.

After the final throughput matrix, a new retrieval instruction was appended
to its 256K prompt with the same salt. This **separate single-request warm
correctness diagnostic** completed with the exact expected key. Server logs
prove 261,824 cached tokens for the 261,914-token prompt. Evidence is in
`bench/results/sglang-long256-workspace-v3-20261015/apc-appended-niah/`.
It checks reuse after a changed suffix; it is not a cold/warm speed A/B,
a disk/RAM offload test, or broad quality equivalence.

Coverage is bounded: one quality seed, one needle per length, C1 only.
C2/C4 graphs, MTP/rollback, vLLM logit equivalence, broader quality and long
soak stability remain unqualified. K4 has not been replaced or reconfigured.
LiteLLM remains unchanged; this model is available separately on port 8081.

### Reproduce the tested C1 configuration

Build from this repository (no live-container modifications):

```bash
docker build -f docker/Dockerfile.sglang-glm \
  -t r9700/sglang:glm53-v0.5.20-gfx1201-kpool-workspace-v3 .
```

Start only in an exclusive GPU window, with the usual PCIe preflight:

```bash
IMG=r9700/sglang:glm53-v0.5.20-gfx1201-kpool-workspace-v3 \
MAXLEN=262400 NSEQ=1 MAMBA_CACHE_SIZE=32 SERVER_SEED=1049505620 \
R9K_SGLANG_BF16_DENSE_GEMV=1 \
R9K_SGLANG_MHC_BACKEND=vllm_tilelang DECODE_GRAPHS=1 \
  bash serve/baseline-earkita/glm/glm-5.3-flash-sglang.sh
```

The tested launcher defaults not shown above are TP8/EP1/PP1, prefill 2048,
static fraction .82, packed MoE BN8, RCCL and custom all-reduce disabled.
`HWQ` is unset. KV stays BF16 and recurrent SSM state stays FP32; no draft
model is loaded. Logical KV capacity is 302,336 tokens shared by this C1
engine, not eight times that value. Default launch settings remain S0;
this command explicitly opts into the tested experiments.

The previous qualified 128K container remains stopped and available locally.
To roll back in a controlled exclusive window:

```bash
docker stop --time 30 glm53-sglang-v0520
docker start glm53-sglang-128k-qualified-resume-20261015
```

This rollback restores the prior SGLang experiment, not production K4.
Starting production vLLM instead requires its own saved launch configuration
and an exclusive GPU window. Do not run either concurrently on these GPUs.

## Native MTP staged qualification (2026-10-15)

The next experiment is native checkpoint MTP, EAGLE chain depths 1/2/3
with top-k 1 and respectively 2/3/4 verify tokens. DFlash stays separate.
The target-only qualified container is retained locally as
`glm53-sglang-target256-qualified-20261015`; do not start it concurrently
with the MTP candidate. Neither this work nor the private SGLang endpoint
changes production K4 or LiteLLM.

The local Quark checkpoint has **original block-FP8** MTP experts and dense
projections in layer 45; its main routed experts remain MXFP4. It is unsafe
to infer draft precision from the global Quark format. The isolated adapter
maps `model.decoder`/`model.eh_proj` back to checkpoint metadata, requires
uniform original 128x128 FP8 expert metadata, preserves FP32 scales and BF16
exclusions, and runs those draft experts through upstream Triton MoE.
Strict NextN coverage rejects missing required weights. Target MXFP4
arithmetic and existing kernels remain unchanged.

The stock HIP MTP path first stopped at a CUDA-only activation selection,
then at the KPool target-verify CUDA guard. Both failed startups are retained
in `bench/results/sglang-mtp-integration-20261015/`. They are runtime failures,
not completed-answer or throughput results. The scoped activation preserves
GLM's clamp-before-SiLU formula; the upstream limit-only helper clamps after
SiLU and is not a valid substitute. KPool verify is explicitly gated on
our opt-in flag and gfx1201, retaining upstream write/commit algorithms.

Tests already performed:

- Five CPU metadata tests; actual built-image Quark exclusions and per-layer
  overrides passed after correcting the standalone test's missing packed map.
- Two upstream KDA verify/sequential tests passed on GPU0.
- A separate BF16-activation/FP32-state probe passed depths 1/2/3, noncontiguous
  state slots, every draft acceptance length, masked rows and subsequent decode.
  Verify did not mutate committed state; measured checkpoint errors were zero.
- Three upstream multi-pool KPool write/plan/effective-acceptance tests passed.
- The draft FP8 operator probe passed M=1/2/4 with unchanged tolerances
  (relative L2 <.02 and max absolute error <.05). Maximum observed relative L2
  was .000423 and max absolute error .000977. This is numerical qualification,
  not a model-speed measurement.

The first FP8 oracle failed at M=4 (relative L2 .0256). Diagnosis showed
native Triton approximate FP32 division versus Torch division changes scale
bits and some FP8 halfway rounding decisions. The revised oracle checks
native scales independently within two FP32 epsilons and checks every FP8
code with independent Torch RN-even conversion at that scale, before dense
reference arithmetic. It does **not** claim an independent bit-exact native
scale generator. Original failures and stage diagnostics remain retained;
no tolerance was relaxed to obtain the later pass.

Capacity preparation changes the experimental pool to eight KDA state slots
and static fraction .85: MTP's original draft adds about 1.22 GiB per rank.
The failed guard-stage startup allocated 281,728 logical KV tokens with these
settings. This is capacity evidence only, not a completed 256K MTP test.
Any throughput comparison must rerun target-only and MTP with the same pool,
context limit, graph configuration, native measurement client and sampling.

`bench/sglang_spec_decode.py` is prepared for such frozen comparisons. It
uses cumulative generated-token counters from native `/generate` streaming
and subtracts the actual first update's token count. This avoids assuming
one token per SSE burst. It is a new protocol: old BetterBench measurements
are not its comparison baseline. Three CPU tests check burst accounting.
The initial gates below have since passed. Qualification remains limited to
the measured contexts and tests; long-context MTP results are pending.

### MTP KPool history regression (2026-10-15)

The first full-model MTP1 tool smoke failed: the model answered directly
instead of returning the requested `add` call. The failed transcript remains
in `bench/results/sglang-mtp-integration-20261015/pilot3/`; it is not a pass.
An opt-in, synchronized layer trace localized a large divergence to the first
DSA layer. At logical position 186, verify supplied **3 KV positions instead
of 187**, missing all 184 historical positions. The attention result matched
an independent FP32 dense calculation over the three supplied keys (relative
L2 0.00181), isolating metadata selection rather than the sparse attention
arithmetic.

`kpool_plan.update_kpool_write_plan` returned immediately on HIP even though
its buffers had been allocated. Their zero request IDs, write locations and
history lengths then reached verify. The isolated, explicit gfx1201 MTP
adapter now enables the existing Triton plan update. It does not change
weights, KV precision, attention arithmetic or CUDA schedule generation.

The new `tests/test_sglang_mtp_kpool_plan.py` calls the **public wrapper**,
not just the underlying kernel. The previous image fails with unchanged
zero request IDs; the fixed image passes MTP depths 1/2/3, batch sizes 1/2,
six history/page-boundary cases and five graph replays with changed inputs.
All integer metadata and physical write locations match the independent
reference exactly. Raw evidence, both image IDs, frozen script hash and
single participating GPU's idle/loaded topology checks are retained in
`bench/results/sglang-mtp-integration-20261015/plan-wrapper-regression/`.

A prior diagnostic-only attention snapshot incorrectly indexed a `[1,T,H,D]`
output as `[T,H,D]`, terminating that trace with a Python `IndexError`. The
snapshot helper now normalizes the layout before selecting a query. This
instrumentation remains disabled by default and forbidden in throughput
measurements. Full-model validation after the metadata fix is still pending.

With the plan fix, position 186 now supplies all **187/187** expected KV
positions, with no missing/unexpected indices. The diagnostic greedy tool
sequence matches 17/17 tokens from repeated pure-prefill one-token reference
requests, whereas the previous image diverged at token 2. This is a bounded
structural regression check, not general logit equivalence. The six-request
answer/stream/two-turn tool smoke also passes with MTP1 in eager mode.
Evidence: `plan-fixed-attention-oracle.json`, `greedy-plan-fixed-trace/`, and
`pilot4-plan-fixed/` within the ignored integration result directory.
Graph-mode, longer contexts, MTP2/3 full-model tests and matched end-to-end
speed comparisons remain to be qualified.

### Matched C1 8K MTP screen (2026-10-15)

Image `r9700/sglang:glm53-v0.5.20-gfx1201-mtp-plan-v9`, immutable ID
`sha256:195ec4e9fc9896e0167a49815ab710a420048f8c1c8054316006f43a75843807`.
All four legs use the same image, BF16 KV, full decode/verify graphs,
context limit 262400, eight KDA slots, static fraction .85, seed 1049505620,
TP8/C1, mHC TileLang and one-row dense GEMV. Only native MTP changes.
Each leg passed the six-request answer/stream/two-turn tool contract.

Frozen input: 8192 tokens, high effort; temperature 1, top-p .95, top-k
unrestricted, sampling seed 1234. Throughput uses 256 forced output tokens
and is separate from natural-EOS answer checks. One excluded cold prefix
population precedes five measured warm-prefix requests. No profiler or
snapshot hooks were active. All eight GPU mappings, topology, idle/loaded
links and 210 W power caps match across the variants.

| Native MTP depth | Decode median, tok/s | Min–max, tok/s | Change vs target-only |
|---|---:|---:|---:|
| 0 | 30.25 | 30.21–30.32 | reference |
| 1 | 46.69 | 46.50–46.98 | +54.3% |
| 2 | 52.89 | 52.54–54.18 | +74.8% |
| 3 | 54.72 | 54.68–55.90 | +80.9% |

Decode excludes time before the first generated-token update, subtracts the
actual number of tokens in that update, and counts accepted output tokens.
It includes draft, verify, state updates and subsequent client-visible gaps.
It is not a GPU kernel/step measurement. Native acceptance counters and
histograms are retained separately from decode throughput.

Cold TTFT observations were 21.35/21.36/21.38/21.42 s for depths 0/1/2/3.
Measured warm TTFT medians were .436/.424/.430/.452 s, with respective
ranges .434–5.283/.419–5.279/.429–5.290/.447–5.299 s. The first warm request
reused 6144 input tokens; subsequent ones reused 8128. Those slower first
warm attempts are retained, not discarded. No claim of faster prefill is
made. Old BetterBench and production vLLM/K4 results are not matched legs.

Raw evidence: `bench/results/sglang-mtp-integration-20261015/graph-screen/`.
The orchestrator initially assumed a nested `server_args` response and
stopped before any benchmark request. It resumed the same live container
using this build's flat response; the original error/evidence and protocol
are retained. No model restart or protocol change was used for that retry.

A further GPU0 KPool ring test passed 174 accepted-prefix pairs across
MTP1/2/3 and six pool-boundary histories. After rejecting tentative suffixes,
committed compressed pools, live tails and subsequent writes exactly match
sequential one-token execution. Evidence: `long-comparison/kpool-rollback/`
inside the integration result directory. The forced KDA/conv commit tests,
plan graph replay test and full-model contract gates remain separate proofs.

The predefined highest-median rule selected depth 3 for matched C1 128K/256K
measurement and three-seed natural-EOS code/NIAH checks. Those long-context
gates are running; this 8K result does not qualify them or authorize K4
promotion. Production K4 and LiteLLM remain unchanged.

Acceptance counters use the pinned runtime's definitions: strict accepted
proposals / (`verify_ct` × depth), excluding bonus tokens. Its native
`spec_accept_length` is total completion tokens / verify calls and includes
the initial generated token as well as bonuses; it is not used to calculate
decode speed or GPU step time. Histograms retain partial and zero-accept
cycles instead of reporting only fully accepted proposals.

The first long-context throughput pilot used a short NIAH key request with
EOS ignored. Its 128K output repeated the completed answer and thinking;
this is not representative long-answer/coding decode. All six attempts
remain synthetic diagnostics in `long-comparison/`; no target-only leg or
speedup claim is attached. The same pilot's separate natural-EOS NIAH checks
passed strict retrieval, format and completion for seeds 1234/5678/9012.
The just-started 256K forced-NIAH request was explicitly cancelled by stopping
our client; the runtime stayed alive and its cache was flushed successfully.

A replacement protocol in `long-engineering-comparison/` reruns **both** legs
with exact-length prompts requesting a 2000-word engineering guide, visible
special tokens and a gate rejecting any generated end-of-turn marker.
The historical native client is preserved as `protocol-client-v1.py` in the
ignored integration evidence; its hash matches the completed 8K protocols.
NIAH remains a separate natural-EOS check and may reuse only the matching
archive prefix from the engineering guide. No quality-latency claim is made.

Output inspection of all twelve guide attempts found no within-answer repeated
12-word phrase or generated end-of-turn marker. Identical responses across
some fixed-seed repetitions are not within-answer loops. However, every sample
ends at its 256-token limit. At 256K four samples contain reasoning only and
treat the archive's "preserve this code" record as an instruction to include it
in the unrelated guide. These samples do **not** pass a completed-answer or
representative final-prose/coding quality gate. Their rates include reasoning
tokens and remain bounded observations for this exact synthetic workload;
they are not a general coding-speed claim. The paired target-only run was subsequently completed under the same bounded
protocol; it does not remove these quality limitations. Evidence:
`long-engineering-comparison/output-audit-128k256k.json` and full responses.

An additional natural-EOS coding diagnostic is prepared in
`bench/sglang_glm_completed_code.py`. It requests a complete dependency-wave
planner and checks normal, duplicate-dependency, cycle, unknown-dependency,
ordering and input-mutation cases with an independent repeated-round oracle.
Generated code runs in a read-only CPU container with no network, GPU devices,
host mounts or capabilities. A self-test passed 169 cases; the CPU tests also
reject incomplete reasoning and malformed answer format. This is preparation,
not evidence that the model passed. Its clean quoted source corpus excludes
the evaluator and its tests. Separate actual inputs 126976/258048 leave 4096
tokens for thinking plus a completed answer within 128K/256K total windows.
Both runtime variants must use the same saved corpus, seeds and budget; this
quality diagnostic does not replace the frozen performance comparison.

## Joint prefill/decode optimization (2026-10-15, active)

The goal now also covers cold/cached prefill, independent new tails and full
coding/tool task latency. Keep the qualified MTP3 image and stopped container
`glm53-sglang-engineering-mtp3-qualified-20261015` as rollback. K4/LiteLLM are
unchanged. The ongoing target-only control and natural-EOS coding gates run
serially before any new GPU operator test; do not overlap them.

The completed 128K guide pair has five warm samples each: target-only median
29.243 tok/s (29.223–29.275), MTP3 56.251 (56.216–56.875). Protocol, input,
image, cache preparation, all-eight GPU mappings/topology/caps and loaded link
checks match. This remains truncated synthetic-guide generation including
thinking, with the quality limitations above. It is not a completed coding
task or a comparison with K4. The completed 256K guide pair has target-only
median 29.159 tok/s (29.096–29.246), MTP3 61.987 (56.630–62.103), five warm
samples each. Both versions passed three-seed exact-key checks at both lengths
and their final six-request API/tool contracts. The protocol/hardware audit is
`long-engineering-comparison/paired-summary-audited.json`. These two contexts
use different prompts; the rates do not establish context-length scaling.

The profile exposes `PREFILL_CHUNK` (default 2048), `MAX_PREFILL_TOKENS`
(default the chosen chunk) and `OBSERVE_REQUESTS=1` (metrics and scheduler
request timing). Defaults retain the previous launch command. Test 4096 and
8192 separately; 16384 needs a measured workspace/headroom gate. MTP4/5 can
now be selected explicitly, but are **not qualified** by the earlier MTP1/2/3
tests. Full rollback/acceptance/answer/tool and matched speed tests remain
required. C2/C4 graphs are still gated off in this launcher pending their
own qualification and mixed-prefill/decode scheduling tests.

`bench/sglang_glm_prefill.py` implements frozen native one-output-token
cold/warm/tail cycles with one excluded complete warmup and retained measured
cycles. It checks exact input IDs, real GPU cache-source counters, recomputed
prefix tokens and rejects reuse of a previously computed "new" tail. Its warm
rate counts uncached work, not the whole cached context. Scheduler prefill
wall time is separate from TTFT and pure GPU time; absent lookup/restore and
KDA/k-pool subcomponent timings remain unknown until separately profiled.
Full 262144-token prefixes plus 8192 new tokens exceed the existing 262400
context limit. A separately qualified limit of at least 270337 positions is
needed; no benchmark may silently shorten that prefix to fit. The proposed
270592 limit must pass a memory gate before either compared variant runs.

An opt-in `R9K_SGLANG_PREFILL_MOE=hip_w4a4_grouped` candidate reuses the
existing gfx1201 multirow W4A4 HIP entry for >=128 rows; small decode/verify
keeps packed GEMV. It preserves original weights, E8M0 scales and Quark QDQ,
transporting the rounded values losslessly as group-scaled FP8. It does not
select the per-tensor W4A8 prefill entry. GPU arithmetic, full-operator timing
and full-model TTFT/quality qualification are still pending. The prepared
`tuning/sglang_prefill_moe.py` checks routing/padding and lossless transport,
an independent dense oracle and A–B–A operator timings at 128/512/2048 rows
with uniform/concentrated routes. CPU module/ABI import passed; the complete
SGLang alignment import requires GPU architecture discovery. An initial
vLLM alignment import pulled in an incompatible OpenAI parser; the candidate
now uses pinned SGLang alignment without changing installed dependencies.

Source inspection explains why this prefill experiment is justified: for a
2048-row TP8/topk8 chunk, the packed path's BN=8 gate/up and down GEMVs launch
1048576 and 8388608 grid workgroups per MoE layer. These are workgroups inside
two kernel calls, not millions of separate kernel launches. The prepared
16-row grouped entry allocates at most 103600 combined workgroups for the same
shapes, including padding/early-return blocks. Alignment, QDQ/transport and
reduction costs still apply, and fewer workgroups are not a speedup or
correctness proof. Source hashes and formulas are retained in
`static-moe-launch-analysis.json`; the queued operator/model tests decide.

Upstream [#43619](https://github.com/sgl-project/sglang/pull/43619) (head
`3f08bcaeeb112e59c8a30b8e3512308cb8076d71`) and
[#43655](https://github.com/sgl-project/sglang/pull/43655) (head
`cb75eee9610f676d52daca0fd99afb7e10ffce41`) were open when checked. Neither
applies directly to pinned 94602c9 plus our adapters. Row splitting needs
identical attention-TP inputs, no attention CP/capture, and adaptation from
DeepGEMM to the local AITER/indexer API. Microbatch overlap needs PP1/EP1,
full attention TP, independent child metadata/mHC state and no auxiliary
hidden capture or cross-layer top-k reuse. Our TP8/EP1/PP1 and full indexer
types satisfy only part of those gates; ROCm stream lifetime, KDA/k-pool and
actual dispatch still require tests. NVIDIA TP4 speed figures are not R9700
measurements. Profile the local prefill before choosing either experiment.

There is also a concrete memory prerequisite for the 8192-row/256K chunk test.
Pinned AITER allocates FP32 logits with the key stride aligned to 256 entries;
8192 queries against 65536 pooled keys need exactly 2 GiB before top-k scratch.
The ordinary upstream `Indexer` has a free-memory/ROCm addressing guard, but
`IndexerKPool._get_topk_ragged_kpool_plan` bypasses it. A separate opt-in
`R9K_SGLANG_INDEXER_PREFILL_ROWS=512|1024|2048` experiment processes query rows
in slices, gathers keys once and preserves each row's original causal lengths,
shared page-table index and top-k offsets. It releases each logits allocation
before allocating the next; decode/verify and the default value 0 are unchanged.
The hard bound uses the allocated aligned stride, not just the visible shape.
AITER changes its tile heuristic at M=1024, so bit equality was checked rather
than assumed. Isolated one-GPU tests passed PAGED C1/unequal C2/C4 at row
limits 512/1024/2048, plus RAGGED unequal C4 at 1024. All tested full-versus-sliced
logits and selected token indices were exactly equal, including padded query
rows. Sampled independent FP32 oracle error was below 2.4e-7 absolute. In the
65536-key test, peak extra allocation fell from 1097706496 to 564769280 bytes at
1024 rows (about 1047 to 539 MiB). This is an operator/workspace result, not a
model performance claim or full C4 runtime qualification.

Full MTP3 correctness reproduction started separately in
`indexer1024-model-quality/`, on pinned image
`r9700/sglang:glm53-v0.5.20-gfx1201-indexer-rows-v3`, with the original
4096/126976/258048-input cases, three seeds and natural EOS. Chunk 2048,
MAXLEN 262400, BF16 KV, original weights, MTP3 and other math settings stay
unchanged; grouped MoE and diagnostic annotations stay off. The launcher now
forwards both independent opt-in parameters and rejects an image without the
requested adapter label before starting it. The first guard reproduction keeps
MoE unchanged. A subsequent diagnostic MoE pair must keep the guard common in
both legs and retain its unresolved quality limitations. Such a screen does
not qualify the combined configuration or authorize baseline promotion.

Current qualification caveat: the guarded run passed 4096-input code 3/3.
At 126976 inputs it completed all three requests without a GPU/runtime error,
but seed 5678 added prose after its Python code fence, violating the frozen
code-only format rule. Original strict scoring stays 2/3, with the third
`wrong_completed` and `format_error` intact. An independent offline extraction
of that single code fence passed 169 executable cases, so semantic function
correctness and format compliance must be reported separately. This does not
rescore the strict failure as PASS or prove that row slicing caused it.

The original guard-quality controller and its queued MoE stage stopped at
that failed strict gate before further GPU experiments. The previously declared
258048-input case subsequently completed in `indexer1024-resume-258k/` on the
same guarded image, same three seeds/corpus/budget and fresh hardware evidence,
without retrying the failed 128K case. All three responses stopped normally,
passed the strict format rule and 169 executable cases each, without OOM.
Six API/tool contract requests passed. This addresses the observed 258K runtime
failure; the original 128K format failure and overall strict-gate failure remain.
Resolve format repeatability and remaining quality gates explicitly before
accepting a new baseline; keep all original attempts.

A further mixed-exponent FP8 indexer check used large encoded values and
independently varied feature scales, with C1 and unequal ragged C4 lengths.
Full versus 1024-row-sliced logits/top-k remained exactly equal; the sampled
FP32 oracle maximum error was below 3.6e-7. Evidence is in
`indexer-rows-operator-v5-high-fp8/`. This is an operator check, not full C4
serving qualification or an explanation of the natural-output format failure.

The first grouped-prefill MoE gate (`grouped-operator-v3-uniform/`) stopped
before timing. Its M128 candidate output was exactly equal to the packed
control, but both differed from the independent reference. Source inspection
found an extra BF16 rounding in that reference's down projection before
multiplication by the FP32 routing weight. Both serving kernels apply the
routing weight to the FP32 accumulator before their single BF16 store. The
reference and tiny CPU oracle now follow that ordering; a ties-to-even fixture
distinguishes the two results. The failed attempt remains intact, and neither
serving arithmetic nor the 0.25 absolute / 0.02 relative-L2 thresholds changed.

New frozen `grouped-operator-v4-uniform/` and
`grouped-operator-v4-concentrated/` tests cover M128/M512/M2048. Both candidates
and controls passed the corrected independent oracle, exact QDQ transport and
expert-alignment gates. Timings cover the full one-GPU routed operator with
three warmups and nine measurements in packed/grouped/packed order. For uniform
M2048 the packed medians were 91.05 and 91.35 ms, versus 10.24 ms grouped
(10.07–10.84 ms range). At uniform M128 the grouped path was slower: 7.42 ms
versus 6.15/6.17 ms packed. Thus the tested dispatch threshold is not uniformly
beneficial; do not infer a model-wide speedup or use it for decode/verify.
The independently frozen `operator-model-comparison-v2/` measures actual 32K
cold/warm prefill with the same row-guarded image in both legs, changing only
the MoE prefill flag, followed by completed-code and API/tool checks. It retains
the original 128K strict failure as a limitation and cannot replace long-context
quality, C4 or end-to-end qualification. K4 and LiteLLM remain unchanged.

The completed unprofiled C1 32768-input screen passed both measured legs,
with one excluded complete cold/warm cycle and five measured cycles each:

| Client-observed metric | Packed control | Grouped prefill |
|---|---:|---:|
| Cold TTFT median | 83.191 s | 28.177 s |
| Cold TTFT min–max | 83.097–83.255 s | 28.161–28.180 s |
| Cold input tokens / TTFT | 393.9 tok/s | 1162.9 tok/s |
| Warm TTFT median | 5.262 s | 1.823 s |
| Warm TTFT min–max | 5.258–5.268 s | 1.820–1.827 s |

The median cold/warm TTFT reductions were 66.13%/65.35%. Both used exactly
30720 device-cached tokens for the identical 32768-token repeat and recomputed
2048 tokens; no host/disk restore was involved. This warm time is not the
lookup/restore cost alone. The pinned request limits exact-repeat matching to
`input_len - 1`, while the hybrid radix cache needs a node with a saved KDA
state. A full-prefix leaf cannot supply the last prompt token's logits, and
a KV-only page split does not invent an intermediate KDA checkpoint. Fresh
appended turns can reuse a different boundary; their results remain separate.

Each leg also passed all three natural-EOS code cases (169 executable checks
per response) and six answer/stream/tool API requests. All eight GPU mappings,
topologies, capabilities and loaded links passed the paired hardware checks.
The same immutable image, MTP3, BF16 KV, chunk/budget 2048 and indexer guard
1024 were used; only the prefill-MoE flag changed. Full evidence and per-leg
source/protocol hashes are in `operator-model-comparison-v2/`, including
`measurement-and-correctness-audit.json`. This selects the grouped path for
further long-context qualification; it is not a 128K/256K, sustained-decode,
completed-task-latency or C4 result. The earlier 128K format failure remains.

DFlash2 remains a separate candidate. The pinned HIP **target-only** sampler
is unavailable, but that is not a blocker for our selector checkpoint:
`DFlashWorkerV2._selector_sampling_accept` uses realized selector Q and
DSPARK's existing Triton p/q sampler, with selector sampling enabled on HIP.
The earlier target-only finding was overbroad; the deeper branch audit is in
`dflash2-preflight-v1/selector-path-audit-v2.json`. The exact local revision
`bf582e4eacc1810f76656d1811693ff6c6737d2a` has selector rank 256, top-k 16 and
five BF16 layers; its 2,342,160,896 tensor bytes and the target's dense BF16
head were inspected from headers. Hub metadata matches that revision; this is
not a full weight checksum or GPU model qualification. Selector/worker-Q and
block-8 state tests precede draft-attention, TP8, memory and model checks.
HiCache and FP8 KV remain disabled; neither follows automatically from this plan.

`bench/sglang_latency_report.py` derives client-visible ITL p50/p95/p99 and
stream-update gaps from saved cumulative token counters, without new GPU
requests. Tokens in one speculative burst share an arrival timestamp; their
zero API-arrival gaps are not zero GPU execution times. End-to-end completed
coding/JSON/multistep tool tasks and C4 interference tests remain final gates.
Raw preparation, PR snapshots, failed imports/builds and pending operator
protocols are in `bench/results/sglang-prefill-optimization-20261015/`.

Completed-code control update: all three target-only responses at each actual
126976 and 258048 input length stopped naturally and each passed 169 independent
executable cases. MTP3 also passed all three 4096-input and all three
126976-input cases. Its 258048-input leg failed at runtime: an additional
420 MiB FP32 allocation in `mask_pooled_logits` exhausted GPU 7's VRAM. No
completed-answer pass is claimed for that request. The
original 4096-input format failures were an evaluator bug: native
`no_stop_trim=True` retains the normal terminal `<|user|>` marker. The parser
now removes exactly one terminal marker and still rejects embedded role tokens
or extra text. Offline rescoring of the three original responses passed 169
cases each; original failures and raw outputs remain intact. No model retry was
used to erase those attempts. See `completed-code-parser-rescore/report.json`
and `completed-code-comparison-v2/`. The native `/generate` reasoning counter
is not a validated split of the visible thinking text; do not interpret its
zero as proof that no reasoning tokens were generated.

The failed v9 container is retained, stopped, with exit 137 after the child
failure and the server's crash-diagnostic wait. This was a reported Torch OOM;
the diagnostic delay is not proof of a hardware address fault. Both queued MoE
stages stopped before GPU work because their correctness prerequisite failed.
All serving workers then exited and all eight cards returned to about 57 MiB
before an isolated indexer test. Original attempts, final logs, telemetry and
the diagnosis are in `mtp3-256k-stall-2144/` and the original quality directory.
The unmodified MTP3/K4 rollback containers remain preserved. No K4 or proxy
restart was performed.

In the completed MTP3 cold 126976-input coding example, client TTFT was 328.71 s
of 338.16 s total task latency. This single diagnostic observation motivates
prefill work; it is not a pure-GPU prefill duration, repeated throughput score
or operator attribution. Natural output lengths differ between variants, and
these quality timings are not used for a controlled speedup claim. The per-case
accounting is `completed-code-comparison-v2/prompt-share-diagnostic.json`.

The next frozen model experiment is a 32768-input prefill screen using the
same grouped-capable image in both legs and changing only
`R9K_SGLANG_PREFILL_MOE=packed` versus `hip_w4a4_grouped`. It waits for both
completed-code and exclusive operator gates. Each leg has one excluded complete
cold/warm cycle, five measured cycles, a separate three-seed 32K executable-code
gate and six API/tool requests. A candidate is selected for long qualification
only if its slowest cold TTFT beats the reference's fastest cold TTFT and its
worst warm TTFT stays within 10% of the reference's worst, with all correctness
and hardware gates passing. This is a screen; long contexts, tails, chunk
tuning and final task latency still require their own measurements. The exact
original MTP3 container is restored afterwards. Frozen plan and retained setup
failure are in `operator-model-comparison/` and `operator-model-driver*.log`.
That original queue is now halted after the preceding v9 MTP3 OOM. A future
MoE pair must freeze the qualified indexer guard as a common setting in both
legs; it must not silently reuse failed/pending v9 results as a new baseline.
Before either leg starts, a linked telemetry addendum fixes the same read-only
five-second observer across both variants. `bench/gpu_telemetry.py` preserves
VRAM, GFX/UMC utilization, power, clock and temperature readings with an explicit
SMI-ID-to-HIP/BDF mapping; SMI index 0 is not HIP ordinal 0 on this host. Unknown
metrics remain N/A, and sampled VRAM maxima are not exact allocator peaks.
It changes no power, clock or fan settings. The observer follows the verified
controller's process identity and exits when it becomes terminal.

The frozen prefill corpus contains actual 8192/32768/131072/262144 input IDs,
with five independent branches of each 128/1024/8192-token append after the
long prefixes. The complete base token sequence is preserved; no two tails
share a full first 64-token page. Local tokenizer/template reproduction matches
the previously served control exactly. This is corpus preparation, not a cache
hit or runtime-quality result.

Separate diagnostic image `r9700/sglang:glm53-v0.5.20-gfx1201-prefill-profile-v2`
adds `R9K_SGLANG_PROFILE_RANGES=1` host annotations for cache lookup/persistence,
KDA state copy/restore/forward, k-pool compression, indexer/top-k, mHC and MoE.
Disabled decorators return the original callable without a serving wrapper.
The patch applies to the qualified v9 image; CPU profiler semantics passed.
Scalar layer IDs and target/NextN identity distinguish corresponding calls
across ranks without inspecting a tensor or synchronizing a device. The v1
CPU-only prototype and source-equivalence evidence remain retained.
The first full-model capture exported CPU-only traces; the corrected capture
now includes device events on every TP rank. `profiling/sglang_prefill_trace.py` associates
HIP launches with asynchronous device events, reports host and GPU intervals
separately, and keeps missing/ambiguous correlations unknown. Nested costs are
inclusive, and a trace envelope is not a proved critical path. Profile bounded
windows separately from throughput; do not enable these ranges in the speed
comparison or sum their costs across operators/ranks.

`bench/sglang_glm_prefill_profile.py` collects bounded traces only on a separate,
explicitly annotated diagnostic image with an isolated writable trace mount.
Long prefixes are primed outside the profiler; only a cold 8K request or fresh
short tail is profiled. The trace bound includes prefix recomputation, and the
client rejects a shortened long prefix, missing GPU cache-source proof, an
aborted one-token request or missing TP-rank trace. A timeout observing profile
startup still triggers a stop attempt; both original and cleanup failures are
retained. These traces are diagnostic and are never unprofiled speed results.

The original `prefill-profile-stage-v2/` froze a separate annotated deployment
after the complete 32K pair terminates successfully. It profiles cold 8192
tokens and a fresh 128-token append after each complete 131072/262144-token
base; long bases are primed outside the profiler. The selected 32K prefill path,
MTP3, chunk 2048, row guard 1024 and BF16 KV stay explicit. Every TP-rank export,
PCIe evidence and read-only telemetry are retained. A profiler observation or
export timeout leaves the specific live diagnostic runtime for inspection;
it does not authorize restarting it. Shape and stack recording are explicitly
off: [PyTorch documents](https://docs.pytorch.org/docs/main/profiler) that
shape tracing retains tensor references and can
introduce additional copies; the observed long-context VRAM headroom was about
1.7 GB per participating GPU. CPU annotations and GPU launch correlation remain
enabled. The v1 queue was cancelled while waiting, before any GPU work; its
original protocol is preserved. That original stage completed its requests but captured **no GPU activity**.
Its raw traces and initial reports are retained; the host-only audits mark
GPU times unknown. Successful export is not GPU profiling success. It was
not a long-context answer-quality test.

`mtp45-state-preparation-v3/` completed after that diagnostic stage and
restored the original private reference. Its independent audit verifies the
image and PCIe evidence for every test. The KDA FP32 and BF16
width-4 convolution probes now accept C1/C2/C4, while preserving the prior C2
default. Depths 3/4/5 include a control plus both new draft depths. The new
`tests/test_sglang_mtp_kpool_isolation.py` compares batched tentative writes
against accepted-only sequential writes across unequal lengths, 64/256-token
boundaries, nonidentity physical pages and noncontiguous request slots. It
checks committed compressed keys/scales, live incomplete tails, changed
continuations, a masked request and reserved sentinels. GPU checks passed: 99 public-plan checks including graph replays, 462 original
C1 rollback prefix pairs, 693 C1/C2/C4 isolation pairs, 18 exact FP32 KDA
configurations and nine convolution configurations. Kernel/state checks do
not qualify the whole scheduler, attention-KV allocator, MTP4/5 answer quality,
C4 serving stability or end-to-end speed.


### Device tracing repair and next prefill experiment

The separate `prefill-profile-stage-v3/` diagnostic captured real GPU kernels
on TP0–TP7 for cold 8192 tokens and fresh 128-token tails after full 131072-
and 262144-token prefixes. Every rank has a common timestamp base and zero
missing/ambiguous launch correlations. The long tails hit the complete base
in **GPU cache**, with host hits and recomputed-prefix tokens both zero.
These are single profiled one-output-token requests, not throughput or
completed-answer quality measurements.

A minimal reproduction isolated missing GPU tracing to parent HIP initialization
before spawning a worker in this pinned PyTorch/ROCm image. Direct execution
and clean spawn each recorded 32 exact-control matrix kernels; parent-init
spawn recorded none. Clearing only the inherited HSA flag did not repair it;
clearing both native profiler registration variables around child startup did.
The opt-in `R9K_SGLANG_PROFILE_CHILD_ENV=both` preserves `spawn`, restores the
parent environment and changes no model arithmetic. Default serving remains
unchanged. The actual UnifiedRadixCache/MambaComponent paths now have opt-in
annotations; the earlier legacy-cache annotations did not cover that backend.
Reproductions, failed attempts, source snapshots and image hashes remain in
`bench/results/sglang-prefill-optimization-20261015/`.

The cold 8K diagnostic attributes substantial device time to the existing Torch
mHC post implementation. On TP0, its inclusive correlated GPU busy time was
about 1.299 seconds, versus a 7.112-second all-rank trace envelope. These are
**profiled observations, not critical-path fractions or a predicted speedup**.
The broadcast product materializes FP32 `[M,4,4,4096]`: 512 MiB at M=2048,
before the other outputs and intermediates. `r9700_sglang/mhc_post.py` prepares
an independent Triton candidate that removes that buffer, keeps FP32 products,
disables FMA and returns BF16. It has **no serving dispatch**. The exclusive
`tuning/sglang_mhc_post.py` gate requires bit-exact reference output before
operator timings and workspace measurements; any mismatch rejects it. Full
model correctness and paired unprofiled prefill remain separate later gates.

The fresh-tail 256K trace measured TP0 KDA restore at approximately 162 us of
correlated GPU work. `kda.state_copy` is nested in restore and must not be
added to it. The restore-planning range has no correlated GPU work: its device
cost is unknown, not measured zero. Cache lookup used about 2.394 ms of host
range time and 9.36 us of correlated GPU work on that rank. Host persistence
also contains runtime/synchronization costs; its host duration is not GPU work.
Detailed per-rank evidence is in the original trace reports.

`mtp345-model-screen-v1/` freezes a separate C1 8K comparison of MTP3/4/5,
using the same grouped prefill, chunk 2048, row guard 1024 and immutable image.
Each depth has six API/tool checks, three completed-code seeds with 169
execution checks each, then two excluded requests and five warm decode samples.
EOS stays active, actual counters exclude the first speculative burst, and
truncation is explicit. The offline ITL addendum propagates both excluded
requests without changing latency arithmetic or the measurement protocol.
This screen does not select the final winner at 128K/256K. The mHC operator
probe waits until that serial screen ends and restores the private reference;
it never overlaps a model benchmark on the same GPU.


The unchanged greedy-verify 8K screen completed all three depths. Every depth
passed three complete code responses (169 execution cases each), six API/tool
requests and five warm decode samples after two excluded requests. The
independent audit checks actual token-counter arithmetic, EOS markers, cache
source, both exclusions, source hashes and all eight PCIe mappings/links.

| Effective HIP argmax verify | Decode median, tok/s | Five-sample range | Median accepted length |
|---|---:|---:|---:|
| MTP3 | 63.802 | 63.018–67.995 | 3.122 |
| MTP4 | 70.756 | 67.781–70.788 | 3.765 |
| MTP5 | 67.117 | 63.320–67.252 | 3.875 |

TTFT is excluded. MTP5 stopped naturally at 248 tokens in three performance
requests; the other measured requests reached the 256-token budget. Both are
allowed by the frozen EOS-active protocol and use actual counts. These are
not fixed-length completed-answer scores, a 128K/256K comparison, or proof of
checkpoint-sampling quality. MTP4's highest median does not select the final
long-context configuration. Full raw outputs and the scope correction are in
`mtp345-model-screen-v1/independent-audit-v2/` and
`effective-sampling-audit-v1/`; an earlier offline auditor incorrectly rejected
normal terminal EOS markers and its failure is retained.

The first fused post-mixing operator passed **15 bit-exact cases**: five row
counts 128/512/2048/4096/8192 at three input scales. At M=2048, both Torch
control medians were about 3.63 ms and the candidate 1.04 ms, with 11 samples
per leg. Extra peak allocated memory measured 768 MiB versus 64 MiB. These
are single-GPU operator intervals and allocator peaks, not a model speedup.
`mhc-post-operator-v2/independent-audit-v1/` preserves all samples, numerical
results, sources and PCIe evidence. The serving candidate is gated by
`R9K_SGLANG_MHC_POST=triton` for at least 128 rows; the default is `torch`, and
small decode/verify calls keep the original path. Its exact disabled source
and separate image are recorded in `mhc-post-integration-v1/`.

The separate PR37134 sampler image passed 1470 forced accept/reject controls
at C1/C2/C4, widths 4/5/6/8 and vocabularies 17/4097/154880, including changed
coins under graph replay and zero/negative/nonfinite/out-of-range proposal
probabilities. These exercise the **classical p/q** contract. They do not add
DFlash's missing target-only sampler: passing its zero scratch Q to a classic
sampler is not equivalent. Our DFlash2 selector instead supplies real sparse Q
to that existing classical sampler; its distinct worker path needs its own gate.
The post-mixing comparison uses independently qualified stochastic MTP3;
rejected higher-depth model candidates do not qualify or block that common path.
Both post-mixing legs use the same corrected sampler image and MTP3, grouped
MoE, row guard and chunk 2048; only post-mixing dispatch changes. Long contexts,
chunk tuning, C4 interference and final completed-task latency remain pending.


Corrected stochastic screen update: MTP3 passed three complete code responses,
six API/tool requests and five warm decode samples at actual 8192 input tokens.
Its decode median was 62.896 tok/s (60.022–67.174); TTFT median 0.406 s is
reported separately. This is an 8K screen, not a new 128K/256K baseline.
Actual resolved `speculative_use_rejection_sampling=true` and the source/image
are retained. MTP4 and the separately resumed, previously unattempted MTP5 each
passed two of three completed-code seeds. MTP4 seed 1234 mutated input;
MTP5 seed 1234 referenced undefined `prq` on an error path. Neither response
was truncated or a runtime error. Both fail the frozen one-shot gate, so their
new throughput legs were not run. This is not proof that draft depth caused
the code errors. No failing seed was retried selectively or removed.

MTP3 remains the common configuration for the mHC post pair. Its v1 queue
required the entire three-depth screen and skipped before GPU work; v2 only
requires the independently passed MTP3 gate and waits for MTP5's verified
terminal state/reference restoration. Measurement inputs, sampling, warmups,
repetitions and one-variable post dispatch stay fixed. The rejected depths and
all prior evidence remain intact. `dflash2-state-selector-v1/` waits behind that
pair: actual upstream selector walk/realized Q versus a CPU oracle, graph
replay, worker Q scatter/cleanup and buffer reuse, then KPool plan/isolation,
FP32 KDA and BF16 convolution at block 8 / C1/C2/C4. Extending test depth limits
to 7 preserves prior defaults and does not rerun the completed depths 3/4/5.
A CPU-only worker import reached a device-properties query and failed without
GPU devices; the bounded exclusive GPU probe will exercise that import normally.
No DFlash model deployment or speed result is claimed yet.

### Qualified post-mixing screen, 2026-10-16

The full 32K Torch/Triton pair passed: same immutable image
`db8baa997cf1687c4486bce9259d2d3511385dcc6aeacd05dc9e412c96b26e97`,
correct stochastic MTP3, grouped MoE, chunk/max-prefill 2048, indexer rows 1024,
TP8/EP1, BF16 KV and FP32 KDA. Only `R9K_SGLANG_MHC_POST` changed.
One complete cold/warm cycle was excluded, then five cycles were measured.

| Actual 32768-token input | Torch TTFT median (range), s | Triton TTFT median (range), s |
|---|---:|---:|
| Cold, zero cache hits | 28.045 (28.035–28.082) | 24.329 (24.311–24.338) |
| GPU cache, 30720 hits, 2048 recomputed | 1.819 (1.816–1.822) | 1.587 (1.583–1.589) |

Median TTFT reductions are 13.25% cold and 12.77% cached. This one-output
test measures prefill mechanics; it does not measure sustained decode or
prove faster completed tasks. Each leg separately passed all three completed
code seeds (169 execution checks per answer) and six API/tool requests.
All eight GPU mappings, topology, idle/loaded links, power limits, sources,
actual sampler and image were independently checked. Raw evidence:
`bench/results/sglang-prefill-optimization-20261015/mhc-post-model-pair-v2/`,
including `independent-audit-v3/audit.json`. Earlier offline auditor failures
misread the server-info response layout/startup timings; their evidence remains.
The exact private MTP3 rollback container was restored and checked afterwards.

The launcher now exposes `R9K_SGLANG_MHC_POST=torch|triton` and
`EAGLE_REJECTION_SAMPLING=0|1`. Both preserve defaults; explicit enablement
requires image capability labels, and sampling requires `MTP_STEPS>0`.
Use an independently built/qualified image, not the old default image tag.
The frozen `chunk-screen-v2/` next compares 2048/4096/8192 with the fused post
and corrected MTP3, the same new context limit 270592 and actual 8K/32K inputs.
That context limit makes room for the entire 256K base plus an 8192-token tail;
it is a capacity setting, not proof that the pending long tests have passed.

The independent DFlash2 selector/state gate also passed: 720 selector/worker
checks, block-8 KPool metadata and request/page isolation at C1/C2/C4,
exact FP32 KDA outputs/state and BF16 convolution checkpoints/continuations.
`dflash2-state-selector-v1/independent-audit-v1/audit.json` validates the actual
image, source hashes, participating GPU and all-eight-GPU restoration checks.
These are operator/state tests on one GPU. Draft projection, TP8 attention,
full-model completion, long capacity and DFlash performance are still pending.

### Chunk and long-context qualification: interrupted on 2026-10-16

The host shut down spontaneously during the first long-context variant. All
controllers died; the waiting DFlash2 full-model and HTTP-agent stages never
started. The user confirmed an unplanned shutdown; its cause is not established.
Initial recovery left the model stopped. The user subsequently requested
continuing the objective and deferring further cause analysis. K4, the MTP3
rollback images and proxy configuration remain unchanged.

Raw evidence is in the ignored
`bench/results/sglang-prefill-optimization-20261015/host-interruption-20261016-0604/`:
55 complete request rows, container state, server/kernel logs, boot identities
and fresh idle PCIe evidence. There are five measured complete 128K cycles and
four of five 256K cycles; the fifth 256K cold request has no completed response.
Code, sustained decode and the 4096 comparison leg did not run. The final loaded
PCIe snapshot was lost with the host process. These partial observations do not
qualify the whole series or an A/B improvement. Do not append one new-boot sample
to the old four; a future comparison needs a fresh frozen, fully matched series.

The CPU-tested prefill client now distinguishes an explicitly allowed identical
replay miss (`allow_warm_cache_miss: true`) from a GPU hit. A zero-hit replay is
reported as `cache_source: none`, with all its prefix tokens recomputed; appended
turns and the declared prefix-recomputation bound remain strict. Nonzero hits
require reconciled source counters, and a GPU-only run rejects host/mixed hits.
Future protocols must freeze both `runner_sha256` and `cache_evidence_sha256`
(the helper at `bench/sglang_cache_evidence.py`). Old frozen runs remain untouched.
The client also persists request-start markers before timing/HTTP, records
the boot identity, flushes streamed evidence, and journals PCIe observations
incrementally. A killed process leaves an unfinished attempt rather than a
false success. Observer failures reject qualification even if earlier samples
exist; server logs are retained when that gate fails. Filesystem sync requests
cannot guarantee survival of faulty storage or power loss. Sixty-one CPU checks
passed, including killing an actual child mid-stream and injecting an observer
failure. These changes require a fresh protocol and both comparison legs; they
do not repair or qualify the interrupted old results.

Fresh post-boot work is separate:

- `agent-task-http-pilot-postboot-v2/`: actual 8K OpenAI tool conversation,
  one fixed seed, independent execution tests and final JSON, followed by six
  API contract checks. This is a bounded pilot, not a long-task comparison.
- `long-mtp3-pair-postboot-v2/`: waits for that live pilot to terminate and
  pass, then starts both long variants from the same frozen protocol. It uses
  whole 128K/256K bases, cold/replay/new tails, completed-code checks and separate
  decode measurements. No old sample is reused. Both variants use an explicit
  `runc` runtime and cache reporting. The corrected-MTP3 pilot configuration is
  the private rollback target; K4 and proxy configuration stay unchanged.

All eight HIP mappings and idle/loaded PCIe evidence are required anew. A queued
controller or a running container does not establish a passed test or a gain.

The post-boot v2 pilot finished: three actual tools, 169 execution cases,
the exact initial 8192-token HTTP count and six API contract requests passed.
Its final answer contained prose before the requested JSON and is retained as
`wrong_completed` (16.787 s total, 343 generated tokens), not rescored as a pass.
The dependent v2 long controller correctly skipped GPU work.

`agent-task-http-guided-final-v3/` is a new full three-seed pilot, not a retry
of only the failed seed. It explicitly enables a JSON schema and disables
further tools only after independent tests prove the latest file correct.
Failed tests remain unconstrained and repairable. This evaluates a guided
workflow, not unconstrained model JSON compliance. The fresh complete long
pair in `long-mtp3-pair-postboot-v3/` waits for this pilot's actual PASS.
The pilot subsequently passed all three seeds (1234/5678/9012), each with
three actual tool calls, 169 execution tests, the exact 8192-token initial
prompt and a normal-EOS final JSON. Whole-task walls were 25.409/25.372/19.379 s;
569/654/344 tokens were actually generated across the four model turns. These
are individual pilot observations with different cache histories, not an A/B
speed result. `independent-audit-v2/audit.json` independently checks frozen
source snapshots, image/runtime/effective sampling, both all-eight-GPU PCIe
gates and actual tool/response artifacts. The first audit's flat-versus-wrapped
server-info error is retained separately; it was not a model failure.

The full long-context pair has now started with chunk 2048. Independent
`dflash2-model-qualification-postboot-v2/` is queued behind its terminal state
and verified corrected-MTP3 restoration. It keeps the existing selector/state
unit gates, BF16 drafter, block 8 and Triton draft attention, and first qualifies
complete code plus API/tool behavior at 8K/32K. Neither a DFlash2 full-model
pass nor a long-context speedup is established by this queue.

Upstream parser review on 2026-10-16 found that merged
[PR #39136](https://github.com/sgl-project/sglang/pull/39136) is already an
ancestor of the pinned `94602c9` (111 commits ahead, none behind). Broader
streaming fixes [#35625](https://github.com/sgl-project/sglang/pull/35625) and
buffer-draining fix [#33946](https://github.com/sgl-project/sglang/pull/33946)
remain open. The tools-plus-response-format issue
[#42269](https://github.com/sgl-project/sglang/issues/42269) has an open partial
fix [#42906](https://github.com/sgl-project/sglang/pull/42906), which explicitly
excludes `glm47`'s full-assistant EBNF path. It is not a ready GLM fix to apply
blindly. The final-only schema workflow does not constrain tool-generation
turns and does not erase the original unconstrained format failure.

The conflict was also reproduced on CPU with the exact `db8baa99` runtime's
`FunctionCallParser` and `ChatCompletionRequest.to_sampling_params`. The parser
sets `use_full_assistant_constraint=True` for the non-strict tools we use;
its raw detector constructor's initial `False` does not describe effective
serving behavior. Tools alone retain the full-assistant/tool EBNF. Adding either
`json_object` or `json_schema` replaces it with a JSON-only constraint. The nine
CPU cases cover `auto`, `required` and `none` with both formats and no format;
in this pin even `required` plus a format does not reject that conflict on the
full-assistant path. Raw evidence is in
`json-final-source-analysis-v1/effective-constraints-cpu-report.json` under the
ignored results directory. These are request-conversion observations, not model
generation or a proof that every previous missing tool call had this cause.

The current task runner avoids the conflict by leaving `response_format` absent
while tools are selected or repairs are possible. It adds the final JSON schema
only after independently executed tests pass for the latest file version,
together with `tool_choice="none"`. This is a qualified workflow workaround;
it does not fix clients which attach an output schema to every tool-enabled
turn. Supporting that API combination requires a separate GLM grammar union
and end-to-end tool-result/termination qualification. The #33946 buffer-drain
patch does not address this constraint-selection bug.

A default-off GLM47-specific candidate for that combination is now prepared in
`r9700_sglang/glm_json_tools.py` and
`r9700_sglang/patches/sglang-v0.5.20-glm47-json-tools.patch`, with an isolated
`docker/Dockerfile.sglang-glm-json-tools`. It composes the existing thinking
prefix with a required-tool branch or the requested JSON Schema through
XGrammar's native `concat`/`union`, preserves the schema alias and injects a
final-format instruction. It keeps the resulting constraint as full-assistant
EBNF instead of installing a second JSON-only mask. Forced or named tool
choices do not acquire a final-JSON escape branch. The serving switch is
`R9K_SGLANG_GLM_JSON_TOOLS=1`; it is absent from every current model command.
No candidate image has been built or deployed.

The exact local XGrammar 0.2.8 passes 732 CPU byte-vocabulary grammar/EOS
checks, including thinking on/off, parallel calls, forced choices, local schema
references, nullable values, enums, arrays and rejection of off-schema answers.
There are also 13 host schema fixtures and 23 CPU request-integration checks
using the patched parser/protocol and exact serving-method AST with a renderer
stub. No-format behavior matches the original methods and the default-off
serving path preserves the original messages/constraint. The first integration
attempt's unvalidated named-tool fixture error is retained separately.

These gates do not prove real-template tokenization, model tool choice, loop
avoidance, speculative grammar state or completed answers. The initial schema
feature gate explicitly rejects unqualified assertions (for example `pattern`
and numeric bounds) instead of silently degrading to any object. Strict-tool
structural-tag requests and Responses API integration remain outside this
prototype. A real secret supplied only through a tool result must be tested
with the schema present on both model turns before claiming #42269 is fixed.
Raw evidence and the remaining implementation/qualification plan are in
`glm47-json-tools-fix-search-v1/` under the ignored results directory.

`bench/sglang_glm_json_tools.py` now retains that actual two-turn API contract,
including wrong answers, missing calls, repeated tools and truncation as
distinct failures. Sixteen CPU transport fixtures pass. A frozen complete
24-case control/candidate cohort (JSON object/schema, streaming on/off,
thinking on/off, three fixed seeds) is serially queued in
`glm47-json-tools-qualification-v1/`, after #33946's separate model
qualification and verified reference restoration. Secrets are frozen privately
in the client protocol, appear only in the executed tool result, and are not
encoded as schema enums. Both requests retain `response_format` and
`tool_choice="auto"`. The two fixes use separate images and comparisons;
neither is silently combined with the running prefill experiment.

The linked issue [#33901](https://github.com/sgl-project/sglang/issues/33901)
was closed automatically for inactivity, not by merging #33946. On 2026-10-16,
the exact GLM47 parser copied from the running `db8baa99` image reproduced
the bug both with and without arguments: only the first call was emitted,
and the complete second call remained buffered. A CPU-only container with no
GPU devices tested the upstream drain-loop change against this exact source:
217 chunking/partial-call checks passed, including every two-part split and
character-by-character streaming. The applied executable addition matches
PR head `8fafb2ac44e389647b395b6e5923304bf45f5a95`.

`r9700_sglang/patches/sglang-v0.5.20-glm47-buffer-drain.patch` preserves that
minimal candidate for the pinned source. It is **not enabled** in the image or
running service. Raw sources, PR metadata/diff, CPU harness/report and patch
provenance are retained in `glm47-buffer-drain-analysis-v1/` under the existing
ignored results directory. This proves a parser defect and a bounded CPU fix;
it does not establish that this caused earlier model loops or final-JSON
failures. Full-model streaming multi-call/tool-result qualification must use
a separate image after the frozen current comparisons, with the control and
rollback preserved.

`chunk-screen-v2/` completed the 2048 and 4096 legs, each with actual 8K/32K,
five measured cold/warm cycles, three complete code answers and six API/tool
checks. Warm identical-prompt reuse recomputed the last scheduler chunk:
2048 versus 4096 tokens. It must not be confused with a new turn appended
after a reusable full-prefix KDA checkpoint.

CPU-only agent-corpus preparation overlapped four measured 8K requests in the
4096 leg. That 8K performance comparison is **not comparable** under the
matched-background-load requirement. All attempts remain; no selected samples
were replaced. The exact intervals/request identities are in
`chunk-screen-v2/background-load-addendum-v1.json`. This qualification caveat
takes precedence over any automatic small-screen performance selection.

The independently frozen `long-mtp3-pair-v1/` started after that controller's
terminal state and verified reference restoration. Its protocol uses the same context
limit 270592 in both legs: entire 131072/262144-token bases, independent new
128/1024/8192-token tails, one excluded cycle and five measured cycles,
three strict completed-code seeds per context, and separate five-sample warm
decode/acceptance/ITL captures after two excluded requests. Only chunks
2048/4096 change; a leg requires its short correctness qualification. CPU-heavy
preparation finished before this long comparison. The host interrupted it;
no promotion follows from its partial results.

`bench/sglang_glm_agent_task.py` prepares the next bounded end-to-end check:
read a virtual Python file, write a repair, execute its independent 169-case
suite in a CPU-only isolated container, and return a final JSON answer. It
retains every streamed model turn and tool result, separates tool time from
model/total wall time, and uses actual continuous usage counters for decode
and burst-aware ITL. A self-reported pass, early evaluator exit, untested latest
file version or truncated answer cannot pass. Seven CPU fixture tests passed;
the full HTTP/model task, exact chat-template count calibration and repeated
128K/256K task comparisons remain pending. The runner edits no repository files.

The first long variant (chunk 2048) completed all five 128K mechanics cycles
after one excluded complete warmup cycle:

| Request at actual 131072-token base | TTFT median, s | Five-sample range, s |
|---|---:|---:|
| Cold | 100.213 | 100.190–100.235 |
| Identical replay, last 2048 recomputed | 1.688 | 1.686–1.690 |
| New 128-token tail | 0.535 | 0.532–0.539 |
| New 1024-token tail | 1.096 | 1.094–1.100 |
| New 8192-token tail | 6.438 | 6.431–6.452 |

Every appended request reused all 131072 base tokens from GPU, with zero host
hits and zero recomputed base tokens. The cold client-observed prefill rate is
about 1308 tok/s, including HTTP/scheduling overhead. These are one-output
mechanics observations of one variant, **not** completed-answer quality,
sustained decode, an audited whole-series result or a comparison win.
`long-mtp3-pair-v1/partial-128k-v1/` retains the immutable 30-request snapshot
and report; full 256K, code/decode and second-variant qualification is incomplete.

The fixed-seed 8K HTTP agent pilot was queued behind the independent DFlash2
qualification, with its own exclusive runtime and rollback. Its waiting process
died in the host reboot before launching a model. A future run must confirm
the real chat-template prompt count and continuous token counters, not merely
the CPU fixtures. It is still a single-contract pilot. Final task comparisons
must also include repository regression tasks and JSON/tool workflows at
128K/256K; passing the dependency-function fixture alone cannot satisfy them.

## FP8 KV experiment (prepared, not qualified)

The launcher keeps `bfloat16` KV by default. `KV_CACHE_DTYPE=fp8_e4m3`
selects an explicit, reversible SGLang/Triton DSA experiment on gfx1201;
unset it to return to the existing BF16 profile. This changes the DSA KV
storage format, not the MXFP4 weights or the FP32 KDA state. The pinned
SGLang source accepts BF16 queries with FP8 KV in its Triton sparse MLA
prefill/decode kernels. On HIP, the FP8 cache write uses the native FP8 dtype
and a direct cast without per-block scales, so correctness cannot be inferred
from successful startup. Total memory capacity need not double because KDA,
index/KPool, workspace and model weights still consume memory.

Before a performance comparison, check numerical DSA prefill/decode outputs
against the BF16 reference and run completed-response, JSON/tool, 128K/256K
prefix reuse, C4 isolation and MTP acceptance/rollback tests. Record the
effective dtype, KV capacity and cache-hit counters. Freeze a BF16/FP8 pair
with identical prompts, sampling, MTP depth, image, chunk size, concurrency,
warmups and repetitions; the dtype must be its only declared difference.
Perform the mandatory eight-GPU PCIe preflight before each leg. The current
GPU-fault guard blocks further GPU runs, so there is no FP8 KV quality or
performance result yet. The opt-in launcher and CPU dry-run tests prepare
the experiment without changing the qualified BF16 default.

## LMCache integration preflight (2026-10-17)

Do not add `--enable-lmcache` to the pinned v0.5.20 profile yet. Its
`mem_cache/registry.py` selects `LMCRadixCache(RadixCache)` in place of the
`UnifiedRadixCache` used for this hybrid GLM. The pinned LMCache connector
registers only the KV buffers; a reusable GLM prefix also needs the KDA/Mamba
checkpoint and DSA indexer buffers. The pinned image does not install LMCache.
Starting an API process with only KV offload would not establish correct
cached answers or tool calls.

Upstream SGLang `main` now has `LMCacheUnifiedRadixCache` and a matching
`UnifiedLMCacheMPConnector` in LMCache that register FULL, MAMBA and DSA
buffers. This is a different integration, not a flag-only change to our
current image. A read-only portability check against SGLang commit
`aaf5f397a5a6565ad2cbcfe64bab56bdebbe4b75` found that 10 of the 27
patches copied by `docker/Dockerfile.sglang-glm` apply cleanly; the other 17
need porting or replacement before an isolated image can be built. The raw
check is in ignored `bench/results/sglang-lmcache-preflight-20261017/`.

Qualification order: pin SGLang and LMCache revisions in a separate image;
port the required gfx1201/Quark/DSA patches; verify all three cache groups
and their geometry on CPU; then, after GPU recovery and the mandatory PCIe
preflight, compare completed cold and restored answers including tool calls,
MTP rollback, cache-source counters and TTFT. Keep the current profile as
rollback. No LMCache GPU run was made during this preflight: the MTP5 kernel
guard still reports `automatic_gpu_reuse_allowed=false` after the MES fault.
