# GLM: current profiles and validation

Configuration reference after profile cleanup, 2026-10-02. Start here for
current settings.

## Profiles

Only these three GLM launchers are maintained. Each invokes the shared
[`serve.sh`](../serve/serve.sh) directly.

| Profile | Role | Active sequences | Context limit | Speculation |
|---|---|---:|---|---|
| [glm-5.3-flash-c2.sh](../serve/glm-5.3-flash-c2.sh) | Two concurrent requests | 2 | auto | DFlash K4 |
| [glm-5.3-flash-c4.sh](../serve/glm-5.3-flash-c4.sh) | Four concurrent requests | 4 | auto | DFlash K4 |
| [glm-5.3-flash.sh](../serve/glm-5.3-flash.sh) | Conservative fallback | 1 | 65536 | Off |

C2/C4 share these defaults:

- Eight R9700 GPUs, TP8 target and drafter; text only.
- Image `r9700/vllm:glm53-plugin-e97573215`, pinned vLLM `e97573215`.
- Quark MXFP4 target with packed W4A4, up to 32 verification rows.
- DFlash2 BF16 **weights**, K4; target and draft **KV** both FP8 E4M3.
- Shared KV pool: 4.125 GiB per GPU, APC enabled, corrected APC boundary.
- NBT1024; FULL_DECODE_ONLY graphs: 5/10 for C2, 5/10/15/20 for C4.
- Bounded indexer workspace follows NSEQ and index_kpool=4.
- No weight or KV offload configured.

The fallback retains stock model/quantization paths, eager execution and APC
off. Its default image is `r9700/vllm:dev` and model is a Hub ID; it does not
inherit the C2/C4 image or local model paths. It is not the former BF16-K7
DFlash reference.

## Launch

Run from the repository root. Preview without starting a service:

```bash
bash serve/glm-5.3-flash-c2.sh DRYRUN=1
bash serve/glm-5.3-flash-c4.sh DRYRUN=1
```

C2/C4 expect local target/draft checkpoints under `/mnt/ai/models/glm` by
default; override `MODELS_DIR`, `MODEL` and `DRAFT` for another installation.
The image and a compatible library in the mounted checkout must exist.
For a checkout without that library, use `BUILD_KERNELS=1` on first launch.
Build the image from the repository root with
`docker build -f docker/Dockerfile -t r9700/vllm:glm53-plugin-e97573215 .`.

On idle GPUs and a free API port:

```bash
bash serve/glm-5.3-flash-c4.sh
# Alternatively, the qualified fixed context settings:
# bash serve/glm-5.3-flash-c2.sh MAXLEN=262144
# bash serve/glm-5.3-flash-c4.sh MAXLEN=131072
```

These are alternatives, not services to run together on the same GPUs/port.
When switching, gracefully stop the verified active container with
`docker stop --signal SIGINT --timeout -1 NAME` and await completion.
The profiles use REPLACE=0 and do not stop another service automatically.

## Capacity and evidence

The latest C2/C4 starts resolved auto to **702720 tokens for one request**,
with about **703817 group-aware KV capacity tokens**. This does not reserve
two or four full-length contexts. Context includes input, thinking and final
output. Shared allocation and recurrent/speculative state affect capacity;
the full auto limit has not been qualified by a full-length generation test.

Bounded long-context checks previously passed for C2 × 256K and C4 × 128K
with K4 at fixed limits. The [latest C4 report](glm-c4-profile.md) measures 1/2/4 concurrent
8K prompts on the auto-context profile, with correctness reported separately.
Do not compare its warm-APC speed directly with earlier cold-cache results.

For all new measurements, follow [GLM testing](glm-testing.md): explicit
temperature/thinking, exact token counts, decode excluding TTFT, and separate
strict match, semantic retrieval, output compliance and completion results.

W4A16 and FULL_AND_PIECEWISE remain experiments, not defaults. Their bounded
comparisons did not establish a consistent speed improvement.
