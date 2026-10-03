# GLM: current profiles and validation

Maintained configuration reference, updated 2026-10-03. Historical measurements
record their own settings; check the running API for live service status.

## Profiles

Only these three GLM launchers are maintained. Each invokes the shared
[`serve.sh`](../serve/serve.sh) directly.

| Profile | Role | Active sequences | Context limit | Speculation |
|---|---|---:|---|---|
| [glm-5.3-flash-c2.sh](../serve/glm-5.3-flash-c2.sh) | Two concurrent requests | 2 | auto | DFlash K4 |
| [glm-5.3-flash-c4.sh](../serve/glm-5.3-flash-c4.sh) | Four concurrent requests | 4 | auto | DFlash K4 |
| [glm-5.3-flash.sh](../serve/glm-5.3-flash.sh) | Conservative fallback | 1 | 65536 | Off |

C2/C4 share these defaults:

- Eight R9700 GPUs, TP8 target and drafter. C2/C4 enable text and images;
  the conservative fallback remains text-only.
- Image `r9700/vllm:glm53-plugin-e97573215`, pinned vLLM `e97573215`.
- Quark MXFP4 target with packed W4A4, up to 32 verification rows.
- DFlash2 BF16 **weights**, K4; target and draft **KV** both FP8 E4M3.
- Shared KV pool: 4.125 GiB per GPU, APC enabled, corrected APC boundary.
- NBT2048 for vision (the earlier text-only measurements used NBT1024).
  This also covers the encoder cache's full 2048-token image limit; the
  default square profiling canvas reports only 2025 tokens.
  FULL_DECODE_ONLY graphs: 5/10 for C2, 5/10/15/20 for C4.
- Bounded indexer workspace follows NSEQ and index_kpool=4.
- No weight or KV offload configured.

The fallback retains stock model/quantization paths, eager execution and APC
off. Its default image is `r9700/vllm:dev` and model is a Hub ID; it does not
inherit the C2/C4 image or local model paths.

### Vision qualification (2026-10-03)

C2/C4 now allow 100 images across the complete request history, with up to
2048 embedding tokens per image after resizing; video remains disabled.
Vision uses the image's existing Flash Attention Triton implementation on
RDNA4 (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, explicit encoder `FLASH_ATTN`).
The initial Torch SDPA attempt failed with HIP `invalid argument` during
encoder profiling. No kernel or checkpoint changes were needed.

All eleven live C4 checks passed: direct/proxied OpenAI, Anthropic
streaming/nonstreaming, image token counts, Claude-style `Read` tool results,
four concurrent image requests, four images in one request, 100 distinct
224x112 images, rejection of 101 images, and text afterward. The 100-image
probe retrieved the correct codes from images 1, 50 and 100. Fixtures also
cover an image with exactly 2048 embedding tokens; NBT2048 fixes the earlier
2025-token encoder-cache boundary. C2 was checked by CPU tests/dry-run;
live vision qualification used C4.

There were no preemptions. Driver-visible free VRAM was only 135–171 MB/GPU
after the 100-image smoke; this includes allocator reservations and does not
measure PyTorch-reusable memory. The checks qualify the count limit with small
images, not accuracy on all 100 images, 100 full-resolution images, C4 times
100 images, or long-context vision.

Reproduce with `bench/glm_vision_smoke.py`. All attempts, including the initial
four-image and encoder-boundary checks, remain in ignored
`bench/results/glm-vision-20261003/` and
`bench/results/glm-vision-100-20261003/`. See the
[proxy guide](../proxy/README.md) for the authenticated invocation.

## Dense block-FP8 tuning

`tuning/tune_glm_fp8_block.py` tunes launch parameters for the two GLM
dense W8A8 shapes `(N,K)=(3072,4096)` and `(4096,1536)` on gfx1201.
It preserves the checkpoint's FP8 values and block scales; it does not alter
W4A4 experts or KV precision. Compiled kernels contain the native
`v_wmma_f32_16x16x16_fp8_fp8` instruction with FP32 accumulation.

Run the tuner only on an idle GPU in the pinned GLM image. For example,
inside a container with the checkout at `/opt/r9700` and one GPU visible:

```bash
python3 /opt/r9700/tuning/tune_glm_fp8_block.py \
  --shape 3072,4096 --rows 1,5,10,15,20,64,128,256,512,1024,2048 \
  --out /opt/r9700/bench/results/glm-fp8-new-run
```

Repeat for `4096,1536` with a different output directory. Each candidate is
checked against FP32 dequantized matmul; finalists use two additional seeds
and five alternating default/candidate timing pairs. A gain below 5% retains
the default. The generated JSONs in `tuning/configs/glm/` are runtime launch
configurations, not benchmark dumps. Raw attempts stay in ignored
`bench/results/glm-fp8-tuning-20261003/`.

Opt in using `FP8_CONFIG_DIR="$PWD/tuning/configs/glm"` as an argument to a
GLM profile. The common launcher mounts each JSON read-only over the
corresponding vLLM config file, preserves other shapes, and includes config
contents in the compilation-cache key. `FP8_CONFIG_DIR=` selects the image's
defaults. These files target the pinned Python 3.12 runtime and must be
revalidated after changing vLLM, Triton, hardware or power limits.

The 2026-10-03 run passed all 1144 FP32-reference numerical checks
(maximum relative L2 0.001680), both baseline/candidate API contract gates,
13 GLM profile tests and six MiMo profile tests. Small-row isolated GEMMs
improved approximately 1.19–1.99x; both shapes retained the default at M=2048.
These are hot-weight microbenchmarks, not whole-model TFLOPS or tok/s.

Matched serving comparison: the C4 vision profile, **one active request**,
K4, 225 W/GPU, BetterBench 0.6.0, exact 8192/32768-token prompts, 256 forced
output tokens, temperature 1/top-p .95/top-k disabled, seed 1234, effort high,
one excluded warmup and three measured repetitions per context. APC stayed
enabled with unique salts and zero cached tokens in every measured request.
All runtime arguments and relevant environment variables matched; only the
two launch-configuration files changed. Medians below exclude warmups:

| Input | Default decode tok/s (range) | Tuned decode tok/s (range) | Default / tuned TTFT s | Default / tuned effective prefill tok/s | Default / tuned DFlash acceptance |
|---|---|---|---|---|---|
| 8K | 50.09 (49.09–55.59) | 50.70 (47.38–53.32) | 6.161 / 6.204 | 1329.75 / 1320.52 | 40.17% / 38.58% |
| 32K | 48.16 (48.15–54.38) | 54.51 (49.65–56.97) | 23.823 / 23.921 | 1375.49 / 1369.84 | 38.37% / 42.69% |

Acceptance is accepted/proposed tokens aggregated across the three measured
requests, separate from decode. The larger 32K decode result coincided with
greater DFlash acceptance. With overlapping sample ranges, three repetitions
and no prefill improvement, this does **not** establish a repeatable whole-model
speedup from kernel tuning. Fixed-output runs are not completed-answer quality
tests. No runtime errors or preemptions were observed; multi-request throughput
and long-context/vision quality were not requalified in this experiment.

The tuning remains opt-in; C2/C4 profile defaults are unchanged. Full protocol,
runtime identities, all attempts, acceptance by position and comparison
results are stored in the ignored evidence directory above.

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

The current C4 vision profile (NBT2048) resolved auto to **700416 tokens for
one request**, with about **701510 group-aware KV capacity tokens**. Claude
Code's template declares **524288**. Text, image tokens, thinking and output
share that context. Earlier text-only NBT1024 starts reported 702720/703817;
those historical numbers and throughput measurements do not describe the
current vision configuration. C2's auto limit should be checked at startup.

The pool does not reserve two or four full-length contexts. Shared allocation
and recurrent/speculative state affect capacity; the full auto limit has not
been qualified by a full-length generation test.

Bounded long-context checks previously passed for C2 × 256K and C4 × 128K
with K4 at fixed limits on the text-only profile. The
[historical C4 report](glm-c4-profile.md) measures 1/2/4 concurrent 8K prompts
with NBT1024, with correctness reported separately.
Do not compare its warm-APC speed directly with earlier cold-cache results.

For all new measurements, follow [GLM testing](glm-testing.md): explicit
temperature/thinking, exact token counts, decode excluding TTFT, and separate
strict match, semantic retrieval, output compliance and completion results.

W4A16 and FULL_AND_PIECEWISE remain experiments, not defaults. Their bounded
comparisons did not establish a consistent speed improvement.
