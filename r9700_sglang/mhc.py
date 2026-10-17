# SPDX-License-Identifier: Apache-2.0
# Projection/mixing semantics: SGLang v0.5.20 layernorm/mhc.py (Apache-2.0).
"""Decode-only mHC using the pinned SGLang Triton Sinkhorn implementation.

Keep the Torch projection, residual mixing and BF16 rounding. Only replace
the twenty eager Sinkhorn iterations; prefill remains on its existing path.
"""

import torch
import torch.nn.functional as F


def _validate(residual, fn, hc_scale, hc_base, hc_pre_eps,
              hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat):
    if (residual.ndim != 3 or not 1 <= residual.shape[0] <= 32 or
            residual.shape[1:] != (4, 4096) or
            residual.dtype != torch.bfloat16 or
            fn.shape != (24, 16384) or fn.dtype != torch.float32 or
            hc_scale.shape != (3,) or hc_base.shape != (24,) or
            hc_scale.dtype != torch.float32 or hc_base.dtype != torch.float32 or
            any(t.device != residual.device or not t.is_contiguous()
                for t in (residual, fn, hc_scale, hc_base)) or
            not residual.is_cuda or hc_pre_eps != hc_sinkhorn_eps or
            hc_post_mult_value != 2.0 or sinkhorn_repeat != 20):
        raise ValueError("mHC decode requires contiguous GLM HC4/H4096, 1..32 rows, equal eps and 20 iterations")


def pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat):
    from sglang.kernels.ops.layernorm.mhc import _hc_split_sinkhorn_triton

    _validate(residual, fn, hc_scale, hc_base, hc_pre_eps,
              hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)
    rows = residual.shape[0]
    flat = residual.reshape(rows, 16384).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(flat, fn) * rsqrt
    pre_mix, post, comb = _hc_split_sinkhorn_triton(
        mixes.unsqueeze(1), hc_scale, hc_base, hc_mult=4,
        sinkhorn_iters=sinkhorn_repeat, eps=hc_sinkhorn_eps)
    layer_input = (pre_mix.squeeze(1).unsqueeze(-1) * residual.float()).sum(1)
    return post.squeeze(1).unsqueeze(-1), comb.squeeze(1), layer_input.to(residual.dtype)


def pre_tilelang(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                 hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat):
    """Separate operator probe: reuse vLLM's runner-independent ROCm mHC.

    Not dispatched by the server's Sinkhorn experiment. This also changes
    projection execution and needs its own numerical and model qualification.
    """
    from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_tilelang

    _validate(residual, fn, hc_scale, hc_base, hc_pre_eps,
              hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)
    return mhc_pre_tilelang(
        residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)
