# GLM-5.3-Flash C4: built-in MTP candidate

**Status: experimental; text-only C1 diagnostics passed, no production qualification.**
This opt-in script is separate from the maintained C4/DFlash and C4/LMCache
profiles. The pinned vLLM build warns that its GLM MTP drafter does not accept
external multimodal embeddings, so vision has not been qualified for this
variant.

The local Quark checkpoint declares one `num_nextn_predict_layers` block and
contains layer 45, including its MTP projection and shared head. The pinned
vLLM image contains `Glm5NextMTP` and an MTP speculator that can reuse the
indexer's top-k selection on later proposal steps. This establishes a plausible
load path, not ROCm/TP8 correctness or performance. The local packed W4A4
adapter identifies the target model by `glm5_next`; the MTP draft config has
`glm5_next_mtp`, so draft experts may take the upstream emulation path. Check
the actual backend log before interpreting any speed result.

`glm-5.3-flash-c4-mtp.sh` keeps the C4 target weights, TP8, NSEQ, KV budget,
sampling interface and target kernels, but selects built-in MTP instead of the
separate DFlash2 checkpoint. It does not start LMCache. Its graph sizes follow
`(MTP+1) × 1..NSEQ`; verify actual graph capture and fallback in runtime logs.
The DFlash-only auxiliary-state, cache-sharing and APC-boundary adapters are
off. Recheck APC correctness and capacity before interpreting warm results.

CPU-only command preview:

```bash
bash serve/baseline-earkita/glm/glm-5.3-flash-c4-mtp.sh DRYRUN=1 MTP=1
```

On an explicitly prepared idle GPU deployment, first validate a completed
text response, tool-call round trip, C1/C4 correctness, no OOM/preemption, and
speculative acceptance by position for MTP=1. Then repeat for 2 and 3; 5 is
only a later candidate. If native MTP fails to load or uses an unsuitable
draft backend, diagnose that path before running a performance comparison.

Before any GPU test, follow [testing.md](testing.md): record all eight GPU
PCIe mappings/topology/link evidence and the frozen request protocol in ignored
`bench/results/`. For a performance comparison, rerun the DFlash and MTP
variants on identical prompts, sampling/thinking, actual input/output token
counts, cache preparation, client, repetitions, hardware conditions and
concurrency. Report TTFT, effective prefill, decode per request, aggregate
throughput and acceptance separately. The existing ~50 tok/s observations
cannot be compared to a newly run MTP smoke test.

## Initial measurements (2026-10-14)

The model loaded on 8× R9700 with TP8, FP8 KV, NSEQ4 and 131072 max model
length. Both target decode and the MTP speculator captured graphs. A bounded
temperature-0 arithmetic request completed with final answer `42`, natural
stop and 7 reasoning tokens. This is a smoke check, not a quality suite.
PCIe snapshots before startup and during JIT, a short request and 128K
requests matched the recorded x4 uplinks/x16 endpoint policy for all eight
cards. GPU power caps were 210 W, so historical 225 W results are not a
comparison.

BetterBench 0.6.0, C1, 256 forced output tokens, temperature 1.0, top-p 0.95,
top-k disabled, seed 1234, effort high, one excluded warmup and three measured
requests gave the following **diagnostic** results:

| Actual prompt tokens | Cache | Decode mean (range) | TTFT mean | Draft acceptance |
|---:|---|---:|---:|---:|
| 8192 | cold | 55.06 (54.93–55.31) tok/s | 4.85 s | 80–81% |
| 32768 | cold | 53.56 (52.73–54.88) tok/s | 19.51 s | 72–80% |
| 130816 | APC warm | 47.30 (42.83–54.11) tok/s | 1.28 s | 49–79% |

The 130816-token warmup was cold (TTFT 109.06 s); all three measured requests
hit 129024 cached tokens. The long-context spread tracks variable draft
acceptance but does not prove its cause. These fixed-length continuations are
not completed-answer quality evidence. Raw requests, timings, counters, PCIe
captures and frozen protocols are under ignored
`bench/results/glm-mtp-20261014/`. The matched DFlash K4 rerun measured
58.09 tok/s at 8K and 57.97 tok/s at 32K; MTP K1 did not beat it. MTP K2
measured 53.56 tok/s at 8K and 55.93 tok/s at 32K, also below K4. These
comparisons share the frozen short-context protocol, 210 W power cap, and
recorded PCIe preflight. MTP K2's per-position draft acceptance was about
57–62% at 8K and 63–69% at 32K; extra draft work did not pay for itself.

At 130816 actual input tokens with warm APC, DFlash K4 averaged 48.60
tok/s (39.90–56.28), MTP K1 47.30 (42.83–54.11), and the exploratory MTP
K2 run 49.28 (41.59–57.63). The wide spread and distinct K2 diagnostic
protocol preclude a speed claim there. The pinned vLLM logs that fused
multi-step drafting is unavailable for `DEEPSEEK_V32_INDEXER`, `KPOOL_TAIL`
and `ROCM_AITER_MLA_SPARSE`, so it rebuilds attention metadata between MTP
draft steps. This is an observed execution path, not a measured attribution
of the elapsed time.

The upstream stack's [2026-10-12 vLLM pin and MoE fusion experiment](https://github.com/bkvargyas/r9700-stack/commit/053b143)
are Qwen-specific evidence: the image reduced its TP4 decode step from 15.0
to 14.3 ms, whereas its bit-identical fused MoE path did not improve step
time and stayed off. GLM uses a separately pinned vLLM commit and MXFP4×MXFP4
with Quark clamp/QDQ, so neither the image change nor the Qwen fusion can be
claimed as a GLM gain without compatibility and numerical gates. A
[vLLM GLM/ROCm issue](https://github.com/vllm-project/vllm/issues/54369)
reports that opting sparse backends into fused drafting produced 2.7% at high
concurrency on MI350X, while deeper MTP still carried substantial per-step
cost. That result motivates profiling the C1 R9700 path, not copying an
untested empty metadata-update hook.

## DFlash C1/128K decode hotspot (diagnostic trace)

A separate DFlash K4 C4-profile launch captured a torch CUDA-activity trace
for one **warm** 130816-input/64-output request. The excluded cold warmup
completed; the profiled response reported 130816 prompt and 64 completion
tokens. All eight GPUs passed the same PCIe policy in 430 during-request
samples. Raw traces, protocol, response and per-rank summary are in ignored
`bench/results/glm-mtp-20261014/torchprof-dflash-diagnostic/`.

For a rank-local 600–1600 ms window from the first GPU event, the eight ranks
agree within about 2 ms on each category. Mean shares of **summed GPU kernel
time** are:

| Kernel family | Share | What it identifies |
|---|---:|---|
| `packed_gemv._gemv` | 24.9% | Our routed W4A4 expert GEMV |
| `ncclDevKernel` | 17.4% | TP8 collectives on RCCL |
| `wvSplitK_hf_sml` | 12.0% | BF16 split-K GEMV family |
| GLM sparse verification | 11.3% | `_sparse_attn_prefill_ragged_kernel` during multi-token verify |
| MXFP4 dequantization | 1.6% | Residual stock/emulation work; exact owner unverified |

This is the largest **observed time share**, not a measured loss against an
optimized implementation. Profiling stretches timing and the selected middle
window is not an exact decode-step boundary. The full rank-0 trace spans
1656 ms with 65514 kernel launches and 18.3% internal gaps; it also includes
the residual uncached prefill, so dividing those totals by output tokens
would misstate decode. The leading optimization hypothesis is the packed
expert GEMV itself, followed by TP8 communication. Upstream Qwen's fusion
result argues against assuming that removing MoE glue launches alone will
help. A proposed kernel or all-reduce change needs a frozen, matched A/B and
numerical/answer checks before any speed claim.

## Packed W4A4 down-tile experiment (2026-10-14)

`tuning/tune_glm_packed_gemv.py` compared the stock 8-column down tile with
16 and 32 columns on one isolated gfx1201. At 5/20 routed rows, tile 32 took
76.0/266.8 us versus stock 90.1/324.3 us for the down projection; the
gate/up projection did not show the same gain. Full routed-MoE tests at both
row counts passed against upstream emulation, including changing routes during
CUDA-graph replay. The source exposes `R9K_GLM_PACKED_DOWN_BN=32` as an
**opt-in only**; the default remains 8.

A matched C1, 130816-input/256-output, warm-APC DFlash K4 test used the same
checkpoint, image, profile, client, prompt, sampling and 210 W cap. Each
variant had one excluded cold warmup and three measured requests. All eight
PCIe paths passed the recorded loaded-link policy (489 candidate and 2282
baseline samples); each measured request hit 130176 cached tokens. Raw
protocol, requests, counters, PCIe captures and summaries are in ignored
`bench/results/glm-packed-gemv-20261014/`.

| Down tile | Decode tok/s, three attempts | Mean acceptance | Median derived ms/draft round | Mean warm TTFT |
|---|---|---:|---:|---:|
| 32 | 70.90, 58.26, 50.00 | 45.6% | 46.79 | 0.878 s |
| 8 | 56.96, 55.70, 44.78 | 38.2% | 46.72 | 0.830 s |

The observed token-rate gap tracks different speculative acceptance; the
round-time medians are effectively level. The derived round time divides the
BetterBench decode interval by the vLLM draft-round count and is diagnostic,
not an exact kernel-step trace. The cold warmups had ~79 s TTFT including
first-use JIT, so they do not establish a cold-prefill improvement. The warm
TTFT values include APC retrieval and cannot be read as raw prefill speed.
This experiment does **not** qualify tile 32 for the default profile, C4, or
LMCache. Keep tile 8 as the stable path; the next investigation should isolate
TP8 collective time and the BF16 GEMV family under the same correctness gates.
