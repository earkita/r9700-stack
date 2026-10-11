# MiMo-V2.6-Flash-MOPD on 8× R9700

The opt-in MiMo integration uses the original Xiaomi MXFP4 checkpoint and its
BF16 DFlash drafter. The plugin routes native MXFP4 experts to existing
MXFP4×FP8 kernels; it does not convert the model to GLM W4A4. Dense FP8 remains
handled by vLLM. No target or draft weights are offloaded.

## Build and launch

All three profiles call the common `serve.sh` directly:

| Profile | Notes / results | Active requests | Single-request limit | Shared KV/GPU | Communication |
|---|---|---:|---|---|---|
| [mimo-v2.6-flash-c2.sh](mimo-v2.6-flash-c2.sh) | [C2](mimo-v2.6-flash-c2.md) | 2 | auto | 7.625 GiB | Exact AR8 + RCCL |
| [mimo-v2.6-flash-c4.sh](mimo-v2.6-flash-c4.sh) | [C4](mimo-v2.6-flash-c4.md) | 4 | auto | 7.625 GiB | Exact AR8 + RCCL |
| [mimo-v2.6-flash.sh](mimo-v2.6-flash.sh) | [Fallback](mimo-v2.6-flash.md) | 2 | 131072 | Automatic | RCCL |

```bash
docker build -f docker/Dockerfile -t r9700/vllm:mimo-e97573215 .
bash serve/baseline-earkita/mimo/mimo-v2.6-flash-c4.sh DRYRUN=1
# After gracefully stopping the previous GPU server:
bash serve/baseline-earkita/mimo/mimo-v2.6-flash-c4.sh
```

The Dockerfile pins vLLM `e9757321527ca1ecd514c07c1418dd2c53da3d19` on ROCm 10.
The profile expects `$HOME/models/mimo/MiMo-V2.6-Flash-MOPD`; override
`MODELS_DIR` for another host. The `dflash` subdirectory must be present.
A CPU preflight validates the index, tensor headers, QKV/expert shapes and draft
configuration before launching. Stop the previous GPU server gracefully first;
the profile does not replace containers or stop another model automatically.

The fallback `mimo-v2.6-flash.sh` delegates to the common `serve.sh`. Defaults: TP8, C2, 131072 total
tokens per request, NBT 8192, utilization 0.94, BF16 target/draft KV, APC, eager,
RCCL, original scales (`R9K_FOLD=0`) and DFlash K7. `SPEC=0` selects the control;
`MAXLEN=32768` is the initial bring-up limit. Text and tools only. Configured
context and actual concurrent capacity are separate claims.

For a shared, explicitly sized cache, use `serve/baseline-earkita/mimo/mimo-v2.6-flash-c2.sh` or
`serve/baseline-earkita/mimo/mimo-v2.6-flash-c4.sh`. Each independently calls `serve.sh`; neither
chains through another model profile. They default to two/four active requests,
`MAXLEN=auto`, 7.625 GiB KV per GPU and exact AR8. DFlash K7, BF16 KV and eager
execution remain enabled. `auto` determines the maximum **single** request;
all active requests share the pool, so that limit is not reserved for each
session. `KVMEM=` restores vLLM's automatic sizing. The Claude template and main
proxy alias declare 512K; the fast alias continues to declare 128K.

The adapter uses the unique `r9700_mimo_mxfp4` quantization registration. It
requires explicit `R9K_MIMO=1`, MiMo, gfx1201 and the audited vLLM version.
It leaves the standard `fp8` config and GLM/Qwen kernels unchanged. MiMo cache
handling preserves nine global and 39 sliding-window layers. The isolated
BF16 DiffKV backend adds split-KV speculative verification; FP8 KV, graphs,
EP, multimodal support and 1M qualification are outside this profile's scope.

`AR8=1` enables the existing exact TP8 P2P all-reduce: one-shot through 16 KiB,
two-shot through 256 KiB, and RCCL for larger messages. `AR8=0` (the fallback
profile's default) keeps RCCL throughout. This switch uses the plugin
communicator; vLLM's own custom all-reduce remains disabled. Compressed AR4
is disabled in both variants.
Confirm all eight ranks log `8-rank P2P all-reduce installed` before measuring;
the communicator can fall back to RCCL if initialization fails.

```bash
bash serve/baseline-earkita/mimo/mimo-v2.6-flash.sh AR8=1
```

The pinned vLLM already supplies TP4→TP8 QKV sharding and the MiMo-compatible
`DFlashDraftModel` (sink bias, value scaling and sliding attention). Those paths
are tested and reused rather than copied into another model implementation.

## Gateway and Claude Code

With the model API ready, stop the previous proxy and start the MiMo config:

```bash
bash serve/litellm.sh HOST=0.0.0.0 CONFIG="$PWD/proxy/mimo.yaml" \
  BACKEND_MODEL=mimo-v2.6-flash-mopd
claude --settings serve/templates/mimo-v2.6-flash.settings.local.json
```

The MiMo template's key helper uses `LITELLM_MASTER_KEY` when supplied, otherwise
reads `secrets/litellm.env` relative to the helper's checkout. No `source` step
is needed, including when Claude runs in another project. The template locates
the helper under `$HOME/ai/r9700-stack`; set `R9700_STACK_ROOT` for another checkout
location, or `R9700_LITELLM_ENV_FILE` for a non-default credentials file. Restart
Claude with the updated template; copied project settings need the updated
`apiKeyHelper` value too. A remote client needs its own helper checkout/key or
an environment-based helper; the server's credentials file is not available remotely.

The aliases `mimo-v2.6-flash` and `mimo-v2.6-flash-fast` target the same GPU
server with thinking enabled/disabled respectively. The Claude template routes
Haiku/Small Fast to the latter. The template's context and compaction window are
524288 tokens (512K), with compaction at 90%. The main proxy alias declares
524288 input tokens; the fast alias declares 131072. These metadata values do
not reserve KV or enforce a combined input-plus-output budget. Leave room for
thinking and output within each session's intended total context budget.
The template's context override is client-wide, not a separate per-subagent
policy. A helper using the main alias shares its declaration; use the fast
alias and a 128K harness/session budget for smaller helpers. This does not
automatically enforce one 512K agent plus three 128K agents.
Restart Claude Code with this template (or update the project's copied
`.claude/settings.local.json`) to pick up the new client setting. The 128K
fallback server needs matching smaller client/proxy declarations if restored.
Use the server's LAN address in the template when connecting from another host.
The native Anthropic route also accepts OpenAI clients through LiteLLM. A
MiMo-only adapter preserves the alias's thinking flag in native Messages and
token-count requests to vLLM; it never rewrites responses or tool streams. Never point GLM-named
aliases at MiMo. To restore GLM, start its saved container
and the GLM proxy config after stopping MiMo.


## Per-profile notes and results

- [C2 shared pool](mimo-v2.6-flash-c2.md): settings, C2 measurements and capacity.
- [C4 shared pool](mimo-v2.6-flash-c4.md): settings and the full matched C2/C4 report.
- [128K/RCCL fallback](mimo-v2.6-flash.md): original qualification, AR8 experiment and KV allocation search.
- [Testing](testing.md): frozen protocols, sampling and validation commands.

Keep notes alongside their matching scripts. Raw evidence remains in ignored
`bench/results/`; moving these files does not qualify a new runtime or benchmark.
