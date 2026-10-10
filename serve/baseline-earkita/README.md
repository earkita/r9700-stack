# Earkita serving baselines

Model profiles in this directory use the common `../serve.sh` launcher.
Preview launch commands with `DRYRUN=1`.

The required measurement protocol and unattended runner live in
[`bench/baseline-earkita`](../../bench/baseline-earkita/README.md).
The current automated matrix covers GLM C4 + LMCache: C1/C2/C4 clients,
8K/32K/128K inputs, low/high/max reasoning, cold/RAM/disk prefix cache,
separate decode and NIAH correctness. MiMo needs its own frozen model-specific
preset before claiming compliance; do not reuse GLM thinking/template settings.

Adding or modifying a profile does not trigger a GPU benchmark or restart a
running service. Full measurements are launched explicitly, then run unattended.
