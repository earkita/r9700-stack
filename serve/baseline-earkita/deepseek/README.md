# DeepSeek V4.1 Flash on 8× R9700

The [experimental launch profile](deepseek-v4.1-flash.sh) uses the shared
`serve/serve.sh` launcher. Its [per-script notes](deepseek-v4.1-flash.md)
record the current validation, capacity limits, and diagnostic measurements.
This profile is not yet a qualified performance baseline.

Build the separate pinned image with `bash docker/build-deepseek.sh` and preview
the command with `DRYRUN=1 bash serve/baseline-earkita/deepseek/deepseek-v4.1-flash.sh`.
Launching the profile requires exclusive use of the participating GPUs.
