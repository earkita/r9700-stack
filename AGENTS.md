# Benchmark requirements

- Before measuring model performance, read `notes/glm-testing.md`; its
  measurement-comparability rules apply to all models, with model-specific
  sampling and thinking settings recorded explicitly.
- Freeze and save the comparison protocol before running either variant.
  Change only the declared experimental variable. Match prompts, token budgets,
  sampling/thinking, cache preparation, warmups, repetitions, concurrency,
  measurement client and hardware conditions.
- Reuse the baseline protocol exactly. Do not silently shorten a comparison
  into a smoke test. If the protocol changes, rerun both variants under the new
  protocol or label the results **not comparable**; do not claim an improvement,
  regression or absence of regression from mismatched runs.
- Report decode separately from prefill/TTFT, per-request rates separately from
  aggregate throughput, and speculative acceptance separately from decode rate.
  Preserve every attempt and distinguish observations from causal explanations.
- These requirements do not authorize extra full benchmarks, service restarts
  or long soak tests. A bounded smoke test is valid as a smoke test only.
