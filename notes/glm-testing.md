# Testing GLM-5.3-Flash

These guidelines apply to the pinned GLM-5.3-Flash Quark MXFP4 checkpoint
and runtime described in [the current profile overview](glm-overview.md). Keep API
contract, throughput, numerical regression and completed-answer quality
results separate. Passing one does not qualify the others.

## Sampling and thinking

The local checkpoint at `/mnt/ai/models/glm/GLM-5.3-Flash-Quark-MXFP4`,
revision `b5688f25491202978c19c4d036eef579f61bbe07`, has
`temperature=1.0` and `top_p=0.95` in `generation_config.json`; it does not
specify `top_k`. These are checkpoint defaults, not a separately verified
vendor recommendation for every workload. Save the generation config and
chat-template hashes with results and recheck them when updating weights.

| Test | Sampling | Purpose |
|---|---|---|
| API contract (`bench/api_smoke.py`) | Existing temperature 0, effort `high` | Check response channels, streaming and tool-call structure; not an answer-quality score |
| Historical BetterBench comparison | Existing 0.7 / top-p 0.95 / top-k 20 | Preserve comparability with published stack measurements; these are benchmark defaults, not GLM defaults |
| Controlled regression | Temperature 0, top-p 1, no top-k restriction, fixed seed | Compare numerical/runtime variants and measure baseline repeatability |
| Quality with checkpoint sampling | Temperature 1.0, top-p 0.95, no top-k restriction | Evaluate completed answers with thinking across multiple prompts and seeds |

For new controlled or checkpoint-sampling requests, set sampling explicitly
instead of relying on inherited server/client defaults. In this vLLM API,
`top_k=-1` disables top-k filtering; this is our explicit test setting, not
a field supplied by the checkpoint. Record any penalties or other sampling
overrides. Do not silently change the historical BetterBench profile.

The pinned template accepts `low` and `high`; omitted effort, `medium` or
`max` renders **Max**. Qwen's `enable_thinking=False` does not disable GLM
thinking. For each comparison, choose and record one effective effort
(`high` for a bounded investigation, or `max` to exercise the template
default), identically in every leg. Sampling and effort are independent.
Requesting `high` does not guarantee nonzero reasoning tokens.

For example, the following is a **quality request body**, not a command to
run a benchmark. The 8192-token budget is an initial pilot choice, not a
guarantee that thinking will finish:

```json
{
  "model": "glm-5.3-flash",
  "messages": [{"role": "user", "content": "<fixed evaluation prompt>"}],
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": -1,
  "seed": 1234,
  "max_tokens": 8192,
  "ignore_eos": false,
  "chat_template_kwargs": {"reasoning_effort": "high"},
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

## Quality and completion budgets

Before comparing variants, fix the prompt set, scoring rule, effective
effort, sampling, seed list, repetitions and output budget. Include more
than one task; a single synthetic key-retrieval probe is a diagnostic, not
a model-quality benchmark. For stochastic tests, use the same predefined
prompt/seed pairs in each leg and report all attempts. For greedy tests,
repeat the baseline too: temperature 0 and a fixed seed do not guarantee
identical outputs on this stack.

Use a small, separately labelled pilot to choose a bounded budget that
allows completed answers. `max_tokens` covers reasoning **and** final
answer; ensure actual prompt tokens plus that budget fit the context.
If the pilot exhausts its limit, report it. A changed budget requires both
comparison legs to be rerun; do not keep retrying only failing examples
until they pass. HTTP timeout changes do not increase the token budget.

Quality requests must allow EOS (`ignore_eos=false`). Retain the full
request, final content, reasoning, usage, finish reason and errors. Report
correct completed answers, wrong completed answers, incomplete/truncated
responses and API failures separately, alongside completion rate and
correct-completed/total-attempts. An incomplete response is not a completed
wrong answer, but it must remain in the total and cannot pass the gate.
If reporting accuracy among completed answers, also show its denominator.

Score the final answer using the declared rule. For an exact-key task,
finding the expected key somewhere in reasoning or a self-correction is
not an exact-answer pass, but can be evidence of needle retrieval: use the
separate NIAH dimensions below. Report reasoning and final-answer token counts
separately when available; mark absent counters as unknown rather than
zero, and label any tokenizer-derived estimates.

`bench/eval.py` currently implements the historical greedy GSM8K protocol:
temperature 0, top-p 1, seed 1234, 4096 tokens with `EVAL_THINK=1`, and
`medium` which renders Max for GLM. It is not the checkpoint-sampling
quality runner and cannot select the example's budget/sampling through
its current environment options. Save `EVAL_TRANSCRIPT` and classify
truncation separately; its legacy aggregate score alone does not make that
distinction.

For bounded checkpoint-sampling probes, use `bench/glm_quality.py`. It
accepts a JSON list of cases with `id`, OpenAI `messages`, `expected`, and
`scoring` (`exact` or `json`). Exact scoring trims surrounding whitespace
only; JSON scoring compares the parsed object and rejects duplicate keys.
Prepare and freeze the corpus before the pilot; record its source and hash.
This is a small diagnostic runner, not a replacement for a broad quality
dataset. Example sequence, with the appropriate runtime already running:

```bash
python3 bench/glm_quality.py --corpus /path/to/corpus.json \
  --out /path/to/pilot --phase pilot --seeds 1234 --ids key_8k arithmetic_1k
# After inspecting pilot completion, fix the budget for every measured leg:
python3 bench/glm_quality.py --corpus /path/to/corpus.json \
  --out /path/to/off --phase off --max-tokens 8192 --seeds 1234 5678
# After explicitly switching to the matching APC-enabled runtime:
python3 bench/glm_quality.py --corpus /path/to/corpus.json \
  --out /path/to/on --phase on --max-tokens 8192 --seeds 1234 5678
```

Defaults are temperature 1.0, top-p 0.95, top-k disabled and effort `high`;
`--temperature`, `--top-p`, `--effort` and `--max-tokens` expose overrides.
Use a fresh output directory per invocation. The runner checks the observed
APC setting, context fit and API prompt counts. Each ON case/seed uses a
fresh salt for cold followed by the identical warm request. Manifests,
requests/responses, cache-counter deltas and summaries are saved. Missing
reasoning usage is `null`, not zero; the non-reasoning token count is an
explicitly labelled subtraction from completion usage. First generated
token, first content and end-to-end times are API observations, not a
BetterBench throughput score. A warm request without hits is recorded as
`warm_hit_observed=false` and does not qualify prefix reuse.

## NIAH: retrieval, format and completion are separate

Never interpret every strict exact-match failure as a retrieval failure.
Save the following independent fields for each NIAH request:

| Field | Rule |
|---|---|
| Strict exact-match | Completed final content equals the expected key after trimming surrounding whitespace; the original strict result is preserved |
| Semantic needle retrieval | The intact expected key occurs in generated content or reasoning; record `retrieval_channels` so reasoning-only evidence is explicit |
| Output-format compliance | Trimmed final content is exactly one key matching the corpus's declared key syntax, with no extra text; a wrong but well-formed key can pass format and fail retrieval |
| Generation completed | Normal `stop`, independently of answer correctness; token-limit termination fails completion |
| Backend/runtime error | `NO` means no observed request-level backend/runtime error, not a proof of kernel correctness; transport or measurement failures without attribution are `UNKNOWN` |

For this synthetic-key corpus, semantic retrieval is an intact-key presence
check, not an LLM semantic judge. Do not accept the expected key as a fragment
of a different longer identifier. A missing response gives unknown answer
dimensions; unknowns remain in the attempt denominator, not silently excluded.
The legacy `correct_completed`/`wrong_completed` status is the strict-answer
classification, not a retrieval verdict.

For a correct key followed by extra explanation, record:

```text
NIAH exact match: FAIL
Needle retrieved: PASS
Output compliance: FAIL
Generation completed: PASS
Backend/runtime error: NO
```

Reports must show **Strict exact-match**, **Semantic needle retrieval** and
**Output-format compliance** separately, with pass/attempt counts and unknown
counts if present. Split context length and cold/warm cache, retaining the
sampling/seed/effort identity of each series. Do not attribute an answer or
format failure to DFlash, a kernel or a backend without reproduction and
additional causal evidence.

If the failure does not recur in a complete request-matched rerun, annotate
the original attempt `non-reproducible` and link the rerun evidence. This
means not reproduced in that rerun, not impossible to reproduce. Keep the
original strict FAIL and all attempts. Selected passing retries are not a
full rerun; a different temperature, budget or prompt is not a matched one.

`bench/glm_quality.py` saves a `niah` object per request and `niah_groups`
by input length/cache phase. New corpora declare `task: niah` and
`key_pattern`; frozen `niah_<length>_depth<depth>` corpora are recognized
without rewriting their contents. Other exact/JSON tasks retain their
existing scoring. To rescore existing evidence offline into a new directory:

```bash
python3 bench/glm_niah_report.py --corpus /path/to/corpus.json \
  --requests /path/to/original/requests.jsonl \
  --full-rerun /path/to/matched-rerun/requests.jsonl \
  --out /path/to/new-dimensional-report
```

The optional full rerun must cover the same request identities and options
(cache salts may differ), finish normally and have no recorded request errors.
Original transcripts/status fields stay intact; derived reports retain source
hashes and reproduction annotations. No inference or server change occurs.

## Throughput measurements

Use the pinned BetterBench 0.6.0 client and retain its raw results. A custom
scenario driver must be labelled as such; manual semantic probes are not
BetterBench quality results. Record exact API prompt/completion counts,
sampling, effort, warmups, repetitions, concurrency and cache policy.
Keep image/commit, weights, TP/EP, graph mode, kernels, KV budget, power
caps and thermal observations matched. Profile separately from timing.

For the current single-token decode measurements, BetterBench's decode
rate is `(completion_tokens - 1) / sum(stream_update_gaps_seconds)`;
reasoning and answer tokens both count. TTFT is excluded and may end at
the first **reasoning** token. Report TTFT and end-to-end latency separately;
do not describe TTFT as time to a visible final answer. Stream updates can
contain multiple tokens, so their gaps are not necessarily GPU step times.
Prompt tokens divided by TTFT is client-observed prefill, including HTTP
and scheduling overhead; cached prefill must be labelled separately.

Fixed-output tests may use `ignore_eos=true`, for example 256 output tokens,
to measure throughput. They cannot establish completed-answer quality.
Do not merge results with different sampling/effort or compare historical
short-chat and long-context workloads as if only context length changed.

## APC comparisons and decisions

Compare APC OFF, APC ON/cold and APC ON/warm with identical prompts,
sampling, effective effort, budgets and predefined seeds. Repeat cold
controls as well as warm requests to estimate existing variability. Run
greedy diagnostics and checkpoint-sampling quality as separate experiments.
Different stochastic text is not by itself a cache-correctness failure.

Warm kernels separately, then use a new supported `cache_salt` for a cold
APC request and retain it for warm repeats. For quality comparisons, reset
the generation seed to the selected paired seed per request. Keep messages
identical; do not insert a changing nonce into a prefix intended for reuse.
Verify actual hits from usage or isolated per-request cache metric deltas.
Also check appended turns and a changed-start control. The current runtime
does not expose `/reset_prefix_cache`; HTTP 404 is not a successful reset.

Report cache mechanics, TTFT/decode and answer quality independently.
Failures or exhausted thinking budgets in both cache modes do not isolate
APC as the cause. Label such evidence **inconclusive for APC attribution**.
A conservative rollback is an operational choice, not proof of a fix;
likewise a speed improvement alone does not qualify correctness. Record
the active configuration separately from the experiment's conclusion.

These guidelines do not schedule tests or change runtime settings. Full
BetterBench, GSM8K and soak runs remain deferred until separately requested.
