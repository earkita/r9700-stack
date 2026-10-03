# Optional LiteLLM gateway

An independent, CPU-only Docker service in front of an already running GLM
server. It does not install dependencies into vLLM, restart the model, or change
GPU/KV allocation. The launcher calls Docker directly, with no profile chain.

```text
OpenAI / Anthropic client -> 127.0.0.1:4000 -> vLLM :8080
```

## Start

Start a GLM profile first, for example `serve/glm-5.3-flash-c4.sh`. Create a
local credential file once, from the repository root:

```bash
python3 - <<'PY'
import os, secrets
from pathlib import Path
p = Path('secrets/litellm.env')
p.parent.mkdir(mode=0o700, exist_ok=True)
fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as f:
    f.write('LITELLM_MASTER_KEY=sk-' + secrets.token_urlsafe(32) + '\n')
    f.write('LITELLM_BACKEND_KEY=EMPTY\n')
PY
bash serve/litellm.sh
```

The file is excluded from Git and Docker build contexts. Set the backend key
there if vLLM requires authentication. The proxy requires its own key even if
the backend is unauthenticated. This simple single-user deployment uses a
master key, with no database, Redis, virtual-key management or response cache.

LiteLLM **1.103.0** is pinned by image digest. Options use `NAME=value`:

| Option | Default / meaning |
|---|---|
| `NAME` | `r9700-litellm`, independent of the model container |
| `HOST` | `127.0.0.1`; set `0.0.0.0` explicitly to accept LAN clients |
| `PORT` | `4000` |
| `BACKEND_BASE` | `http://127.0.0.1:8080`, root URL without `/v1` |
| `BACKEND_MODEL` | `glm-5.3-flash`, the backend's served model name |
| `ENV_FILE` | `secrets/litellm.env` in this checkout |
| `CONFIG` | YAML file; defaults to `proxy/litellm.yaml`, use `proxy/mimo.yaml` for MiMo |
| `IMG` | Override only when validating a new LiteLLM version |
| `DRYRUN` | `1` prints the launch command, never credential contents |

This launcher uses Linux host networking and `--restart unless-stopped`.
An existing container is never replaced automatically. To reload configuration,
stop/remove only the proxy and launch it again:

```bash
docker logs --tail 100 r9700-litellm
docker stop --timeout 60 r9700-litellm
docker rm r9700-litellm
bash serve/litellm.sh
```

## Clients

| Client | URL | Model alias |
|---|---|---|
| OpenAI chat completions | `http://127.0.0.1:4000/v1` | `glm-5.3-flash` |
| Anthropic Messages / harness | `http://127.0.0.1:4000` | `glm-5.3-flash-high` |

These aliases select API routes to the **same model**, not independent GPU
workers. Default sampling is temperature 1 / top-p 0.95, with high reasoning
effort. Requests can supply sampling overrides. The OpenAI alias sends GLM's
`chat_template_kwargs.reasoning_effort`; the Anthropic alias uses native vLLM
Messages handling. GLM's `high`/`low`/`max` semantics are described in
[the testing guide](../notes/glm-testing.md). Do not use Qwen's
`enable_thinking=false` to disable GLM thinking.

For a local Anthropic-compatible harness, load the generated file in its shell:

```bash
set -a
. ./secrets/litellm.env
set +a
export ANTHROPIC_BASE_URL=http://127.0.0.1:4000
export ANTHROPIC_AUTH_TOKEN="$LITELLM_MASTER_KEY"
export ANTHROPIC_MODEL=glm-5.3-flash-high
export ANTHROPIC_DEFAULT_OPUS_MODEL=glm-5.3-flash-high
export ANTHROPIC_DEFAULT_SONNET_MODEL=glm-5.3-flash-high
export ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-5.3-flash-high
export ANTHROPIC_SMALL_FAST_MODEL=glm-5.3-flash-high
```

Alternatively, the ready-made Claude Code template is
[`serve/templates/glm-5.3-flash.settings.local.json`](../serve/templates/glm-5.3-flash.settings.local.json),
adapted from the previous GLM deployment. From this checkout:

```bash
set -a
. ./secrets/litellm.env
set +a
claude --settings serve/templates/glm-5.3-flash.settings.local.json
```

The template's key helper reads `LITELLM_MASTER_KEY` from the environment;
no credential or path to the old repository is embedded. It preserves the
previous `acceptEdits` and `Read`/`Bash` permission rules, routes every model
alias to `glm-5.3-flash-high`, and sets high effort. Its 524288-token window
with 90% auto-compaction is a client budget, not a KV reservation or a promise
of four simultaneous 512K sessions. Check the backend's `/v1/models` limit
before using it with a smaller serving profile. `DISABLE_PROMPT_CACHING=1`
disables Claude's provider cache controls; it does not disable vLLM APC.
For a client on another LAN machine, copy the template and replace its
`ANTHROPIC_BASE_URL` with `http://SERVER_IP:4000`; provide the key on that
client via `LITELLM_MASTER_KEY`. The template is not installed into your
personal or project `.claude` settings automatically.

There is no proxy queue policy or automatic retry/fallback configured. vLLM
still schedules requests: C4 admits at most four active sequences, and more
requests can wait there. Proxy aliases do not reserve context, increase KV
capacity or persist APC to disk. No unvalidated context limit is advertised.

## Local compatibility and checks

`litellm_hooks.py` contains one narrowly scoped compatibility fix: LiteLLM
1.103.0's Anthropic token counter hardcodes the remote Anthropic endpoint.
The hook sends it to this deployment's local `/v1/messages/count_tokens`.
**This configuration is for one local backend.** Do not add remote Anthropic
models while that process-wide override is loaded. Recheck the fix on upgrades.

No historical tool-stream rewriting or forced `strict` schemas are installed.
Validate the native tool flow before adding compatibility code. With the key
loaded as above, run the live contract gate (records every response):

```bash
python3 bench/litellm_smoke.py --out bench/results/litellm-smoke-UNIQUE
```

It checks authentication, OpenAI and Anthropic streaming/nonstreaming,
thinking, tool-result round trips, and local token counts with/without tools.
It is a bounded integration check, not a model-quality score or throughput
benchmark. Measure serving performance directly against vLLM; label any proxy
comparison separately and hold request/cache settings constant.

CPU configuration/transport tests (the same pinned image as the launcher):

```bash
docker run --rm --network none -e LITELLM_LOCAL_MODEL_COST_MAP=True \
  --entrypoint python -v "$PWD:/repo:ro" -w /repo \
  ghcr.io/berriai/litellm:v1.103.0@sha256:bd089afdcd35b894b14a93f9743cdc8b591f82da1a38dd43a010a7b0c9de5fd7 \
  -m unittest discover -s tests -p test_litellm_proxy.py -v
```

Upstream references: [vLLM provider](https://docs.litellm.ai/docs/providers/vllm),
[Anthropic Messages](https://docs.litellm.ai/docs/anthropic_unified).

## Bounded validation (2026-10-03)

On the existing GLM C4 server: all 12 live contract checks and all six CPU
gates passed. Token counts matched the backend for both plain and tool-bearing
requests. A separate local capture backend received identical JSON request
bodies through the OpenAI proxy route and the direct client, including seed,
sampling, reasoning effort and stream options.

A short interleaved BetterBench 0.6.0 diagnostic used one excluded warmup per
route and three measured requests per route, the same short prompt, T=1,
top-p=.95, top-k=-1, seed=1234, high effort, 256 generated tokens, EOS respected
and zero APC hits. Direct/proxy means: TTFT 308/315 ms, decode 60.77/55.00 tok/s,
DFlash acceptance 45.18/38.24%. Counter-derived time per draft was about
45.78/45.77 ms, not a directly measured kernel latency. The sample is too small
to attribute the decode difference to gateway overhead or qualify performance;
these truncated generations do not validate completed-answer quality.

Local evidence is under `bench/results/litellm-integration-20261003/`, excluded
from Git: full transcripts, frozen comparison protocol, every timing attempt,
transport captures, source hashes and final service metrics. The proxy remains
running; the GPU server was not restarted during integration.

MiMo uses its own aliases and template; see [the MiMo profile](../notes/mimo.md).
