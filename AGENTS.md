# Benchmark requirements

- Before measuring model performance, read `serve/baseline-earkita/glm/testing.md`; its
  measurement-comparability rules apply to all models, with model-specific
  sampling and thinking settings recorded explicitly.
- PCIe verification is mandatory before every GPU/runtime test or benchmark
  series, for every participating GPU. Save PCI BDF-to-worker/HIP mappings,
  topology (root ports, bridges/switches, shared uplinks and NUMA nodes), and
  supported versus negotiated link speed/width. Capture idle values and
  recheck during the already planned warmup/test; idle power-saving speed is
  not evidence of a degraded link. Record raw evidence in `bench/results/`.
  Recheck after a reboot, GPU reset, hardware/BIOS change or runtime device-map
  change, and before each comparison variant. Missing evidence or unexpected
  link/topology differences must be resolved before claiming comparable
  performance. See the PCIe preflight in `serve/baseline-earkita/glm/testing.md`.
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
