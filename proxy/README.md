# Optional LiteLLM gateway

An independent, CPU-only Docker service in front of an already running GLM or MiMo
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
| `CONFIG` | YAML file; defaults to `proxy/glm.yaml`, use `proxy/mimo.yaml` for MiMo |
| `IMG` | Override only when validating a new LiteLLM version |
| `DRYRUN` | `1` prints the launch command, never credential contents |

This launcher uses Linux host networking and `--restart unless-stopped`.
An existing container is never replaced automatically. To reload configuration,
stop/remove only the proxy and launch it again:

```bash
docker logs --tail 100 r9700-litellm
docker stop --signal SIGTERM --timeout -1 r9700-litellm
docker rm r9700-litellm
bash serve/litellm.sh
```

## Clients

| Client | URL | Model alias |
|---|---|---|
| OpenAI chat completions | `http://127.0.0.1:4000/v1` | `glm-5.3-flash` |
| Anthropic Messages / harness | `http://127.0.0.1:4000` | `glm-5.3-flash-high` |
| Claude Code Haiku / Small Fast | `http://127.0.0.1:4000` | `glm-5.3-flash-fast` |

These aliases select API routes to the **same model**, not independent GPU
workers. Default sampling is temperature 1 / top-p 0.95, with high reasoning
effort. Requests can supply sampling overrides. The OpenAI alias sends GLM's
`chat_template_kwargs.reasoning_effort`; the Anthropic alias uses native vLLM
Messages handling. GLM's `high`/`low`/`max` semantics are described in
[the testing guide](../notes/glm-testing.md). Do not use Qwen's
`enable_thinking=false` to disable GLM thinking.

The `fast` alias selects GLM's **Low** template effort on the same backend;
it is not a smaller model and does not guarantee a particular latency. The scoped
`glm_hooks` adapter preserves `chat_template_kwargs.reasoning_effort=low` through
the native Anthropic route, including when the client supplies high effort.
Local token counting uses the same Low template. Main roles keep their existing
`high` route. This role split
does not configure Claude Code's auto-mode classifier or its timeout.

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
export ANTHROPIC_DEFAULT_HAIKU_MODEL=glm-5.3-flash-fast
export ANTHROPIC_SMALL_FAST_MODEL=glm-5.3-flash-fast
```

Alternatively, the ready-made Claude Code template is
[`serve/templates/glm-5.3-flash.settings.local.json`](../serve/templates/glm-5.3-flash.settings.local.json),
adapted from the previous GLM deployment. From this checkout:

```bash
claude --settings serve/templates/glm-5.3-flash.settings.local.json
```

The template's key helper reads `LITELLM_MASTER_KEY` from the environment or
`secrets/litellm.env` in this checkout. It locates the helper under
`$HOME/ai/r9700-stack`; set `R9700_STACK_ROOT` for another checkout location.
No credential or path to the old repository is embedded. It preserves the
previous `acceptEdits` and `Read`/`Bash` permission rules, routes main roles to
`glm-5.3-flash-high` and Haiku/Small Fast to `glm-5.3-flash-fast`, and keeps high
client effort for main tasks. Its 524288-token window
with 90% auto-compaction is a client budget, not a KV reservation or a promise
of four simultaneous 512K sessions. Check the backend's `/v1/models` limit
before using it with a smaller serving profile. `DISABLE_PROMPT_CACHING=1`
disables Claude's provider cache controls; it does not disable vLLM APC.
GLM C2/C4 also accept images, including Claude Code `Read` image tool results.
The limit is 100 images across the entire request history, not 100 new images
per turn. The processor resizes each image to at most 2048 image tokens;
image tokens and delimiters share the context window with text and output.
Video is disabled. The text-only fallback cannot accept these image requests.
The proxy's `supports_vision` metadata must match the active backend.
For a client on another LAN machine, copy the template and replace its
`ANTHROPIC_BASE_URL` with `http://SERVER_IP:4000`; provide the key on that
client via `LITELLM_MASTER_KEY` with a local helper checkout, or use
`apiKeyHelper: "printenv LITELLM_MASTER_KEY"` there. The template is not installed into your
personal or project `.claude` settings automatically.

There is no proxy queue policy or automatic retry/fallback configured. vLLM
still schedules requests: C4 admits at most four active sequences, and more
requests can wait there. Proxy aliases do not reserve context, increase KV
capacity or persist APC to disk. Declared context windows are client settings,
not full-length or aggregate capacity qualifications; see each model's results.

## Comparing GLM and MiMo in Claude Code

Each model has its own proxy configuration and client template:

| Backend served name | Proxy config | Claude settings | Reasoning |
|---|---|---|---|
| `glm-5.3-flash` | [`glm.yaml`](glm.yaml) | [`glm-5.3-flash.settings.local.json`](../serve/templates/glm-5.3-flash.settings.local.json) | All roles use GLM `high` |
| `mimo-v2.6-flash-mopd` | [`mimo.yaml`](mimo.yaml) | [`mimo-v2.6-flash.settings.local.json`](../serve/templates/mimo-v2.6-flash.settings.local.json) | Main roles use thinking; Haiku/Small Fast uses fast |

`glm.yaml` replaces the former generic `litellm.yaml`; update any explicit
`CONFIG` path in local launch commands. The launcher still defaults to GLM.
The provided C2/C4 GPU profiles share GPUs 0–7 and port 8080: switch models
sequentially. Changing the Claude template alone does not switch GPU weights.
Only advertise the aliases of the active backend.

When intentionally switching, finish active requests, gracefully stop the
current GPU container with `docker stop --signal SIGINT --timeout -1 NAME`,
and launch the desired model profile on the freed GPUs. Profiles use REPLACE=0;
an existing stopped container with the same name must be removed before a fresh
launch. Check `http://127.0.0.1:8080/v1/models` for the expected served name.
Then stop/remove the proxy as above and run **one** matching command:

```bash
# GLM backend already ready on 8080:
bash serve/litellm.sh HOST=0.0.0.0 CONFIG="$PWD/proxy/glm.yaml" BACKEND_MODEL=glm-5.3-flash
claude --settings "$PWD/serve/templates/glm-5.3-flash.settings.local.json"

# Alternative, MiMo backend already ready on 8080:
bash serve/litellm.sh HOST=0.0.0.0 CONFIG="$PWD/proxy/mimo.yaml" BACKEND_MODEL=mimo-v2.6-flash-mopd
claude --settings "$PWD/serve/templates/mimo-v2.6-flash.settings.local.json"
```

The same local proxy key and port 4000 work for both. GPU launchers are
`serve/glm-5.3-flash-c2.sh` / `-c4.sh` and
`serve/mimo-v2.6-flash-c2.sh` / `-c4.sh`. Recreate older containers through these
profiles when their saved mounts point to retired cache/config paths.

For a manual task comparison, start fresh conversations from the same clean
project state, use the same prompt, tools, context budget and output budget,
and record the model and completion/tool errors. The templates retain their
model-specific reasoning behavior. Haiku/Small Fast differs between templates;
use main-role tasks for a narrower comparison or report helper usage explicitly.
Both main templates declare 512K with compaction at 90%; this does not qualify
512K sessions on both models or promise four full windows concurrently.

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

For GLM C4 vision, run the bounded image contract check (requires Pillow):

```bash
LITELLM_MASTER_KEY="$(python3 serve/claude-litellm-key.py)" \
  python3 bench/glm_vision_smoke.py --out bench/results/glm-vision-UNIQUE
```

This saves synthetic OCR fixtures and every API attempt, checks direct OpenAI,
proxied OpenAI/Anthropic streaming, local image token counts, a Claude-style
`Read` image result, four concurrent requests, 100 small images and rejection
of the 101st image. The 100-image gate does not qualify 100 full-resolution
images or four simultaneous 100-image sessions.
It is not a general visual-quality benchmark or a long-context qualification.

CPU configuration/transport tests (the same pinned image as the launcher):

```bash
docker run --rm --network none -e LITELLM_LOCAL_MODEL_COST_MAP=True \
  --entrypoint python -v "$PWD:/repo:ro" -w /repo \
  ghcr.io/berriai/litellm:v1.103.0@sha256:bd089afdcd35b894b14a93f9743cdc8b591f82da1a38dd43a010a7b0c9de5fd7 \
  -m unittest discover -s tests -p test_litellm_proxy.py -v
```

Upstream references: [vLLM provider](https://docs.litellm.ai/docs/providers/vllm),
[Anthropic Messages](https://docs.litellm.ai/docs/anthropic_unified).

## Initial text-only gateway validation (2026-10-03)

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
transport captures, source hashes and final service metrics. The GPU server
was not restarted during that integration test. These are historical results;
check `/v1/models` for the currently served backend.

MiMo uses its own aliases and template; see [the MiMo profile](../notes/mimo.md).
