# Testing MiMo-V2.6-Flash-MOPD

[Profiles](README.md). Follow the repository-wide [measurement and PCIe rules](../glm/testing.md),
with the MiMo-specific sampling and thinking settings below. The GLM automated
matrix is not a MiMo preset. These instructions do not start services or tests.


Run the profile tests on CPU and `tests/test_mimo_gpu.py` in the built ROCm
image with one free GPU and the repository mounted at `/opt/r9700`. GPU gates
cover TP4→TP8 QKV, native MoE shapes, and BF16 DiffKV across global/SWA,
sinks, C1/C2/C4, K7 verification up to 32 rows and unequal/mixed query lengths.

`bench/mimo_validate.py --freeze --model-path MODEL --out DIR` creates one
exact-token corpus and protocol using the checkpoint tokenizer. Its `smoke`,
`bench`, `quality` and `apc` phases write every attempt to a new output directory.
The benchmark requires BetterBench 0.6.0 and a `--corpus DIR/corpus.json` path.
`--bench-lengths 1024 32768` selects a bounded comparison, and
`--bench-cache-namespace NAME` avoids prefixes left by an earlier run. Freeze
these options for both legs; do not compare a subset with a different protocol.
Use `--concurrency 1 2 4` both when freezing a C4 corpus and when measuring it;
`--quality-lengths 8192` selects a bounded, separately budgeted retrieval probe.
The server independently verifies token counts. The runner checks C1/C2 APC
cold/warm counters and independent C2 retrieval keys. Compare only matched runtime
settings and frozen workloads. Prefix salts distinguish cold repetitions.

Quality sampling is temperature 1/top-p 0.95, explicitly enabled thinking,
no top-k restriction, with EOS respected. NIAH reports strict answer, retrieval,
format, completion and runtime status separately. Speed samples have a 256-token
budget and do not qualify answer quality. Decode excludes TTFT, includes
reasoning tokens and uses actual completion usage; C2 also reports overlapping
decode intervals. Cached prefill is reported separately.

Results belong in ignored `bench/results/`, not tracked baseline dumps.
