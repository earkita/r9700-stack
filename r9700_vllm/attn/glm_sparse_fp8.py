# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scaled E4M3 sparse MLA reader for the pinned GLM RDNA4 adapter.

Adapted from vLLM e9757321527ca1ecd514c07c1418dd2c53da3d19 and the
validated patches/vllm/0001 and 0002. Arithmetic and launch configuration
are retained verbatim. No installed upstream source is modified.
"""
import torch
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _sparse_kv_row_offset, _as_int32_contiguous_1d, _validate_sparse_dims,
    build_ragged_indices_from_dense,
)

_ON_RDNA4 = current_platform.is_rocm() and torch.cuda.get_device_properties(
    torch.cuda.current_device()).gcnArchName.split(":")[0] == "gfx1201"

@triton.jit
def _sparse_attn_prefill_ragged_kernel(
    q_ptr,
    kv_ptr,
    kv_scale_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    num_heads,
    head_dim,
    num_kv,
    scale,
    FP8_KV: tl.constexpr,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim

    q = tl.load(
        q_ptr
        + query_idx * q_stride_t
        + head_offsets[:, None] * q_stride_h
        + dim_offsets[None, :] * q_stride_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    kv_start = tl.load(kv_indptr_ptr + query_idx)
    kv_end = tl.load(kv_indptr_ptr + query_idx + 1)
    kv_len = kv_end - kv_start

    k_offsets = tl.arange(0, BLOCK_K)
    slot = tl.load(
        kv_indices_ptr + kv_start + k_offsets, mask=k_offsets < kv_len, other=-1
    )
    for k_start in tl.range(0, kv_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < kv_len
        valid = in_range & (slot >= 0) & (slot < num_kv)
        safe_slot = tl.where(valid, slot, 0)

        kv = tl.load(
            kv_ptr
            + _sparse_kv_row_offset(safe_slot[:, None], kv_stride_n)
            + dim_offsets[None, :] * kv_stride_d,
            mask=valid[:, None] & dim_mask[None, :],
            other=0.0,
        )
        if FP8_KV:
            kv = (
                kv.to(tl.float32) * tl.load(kv_scale_ptr).to(tl.float32)
            ).to(tl.bfloat16)

        next_k_pos = k_start + BLOCK_K + k_offsets
        slot = tl.load(
            kv_indices_ptr + kv_start + next_k_pos, mask=next_k_pos < kv_len, other=-1
        )

        scores = tl.dot(q, tl.trans(kv)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_final)
        l_final = l_i * alpha + tl.exp(sink - m_final)
        denom = tl.maximum(l_final, 1.0e-30)
        out = tl.where(
            l_final[:, None] > 0.0,
            (acc * alpha[:, None]) / denom[:, None],
            0.0,
        )
    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)

    tl.store(
        out_ptr
        + query_idx * out_stride_t
        + head_offsets[:, None] * out_stride_h
        + dim_offsets[None, :] * out_stride_d,
        out,
        mask=head_mask[:, None] & dim_mask[None, :],
    )

def _rocm_sparse_attn_prefill_ragged_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    kv_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    assert q.ndim == 3, f"expected q=[sq,h,d], got {q.shape}"
    assert kv.ndim == 2, f"expected kv=[skv,d], got {kv.shape}"
    assert indices.ndim == 1, f"expected indices=[nnz], got {indices.shape}"
    assert indptr.ndim == 1, f"expected indptr=[sq+1], got {indptr.shape}"
    assert not q.is_cpu and not kv.is_cpu and not indices.is_cpu and not indptr.is_cpu

    indices = _as_int32_contiguous_1d(indices)
    indptr = _as_int32_contiguous_1d(indptr)
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    assert indptr.numel() == num_queries + 1, (
        f"expected indptr shape [{num_queries + 1}], got {indptr.shape}"
    )
    _validate_sparse_dims(
        head_dim,
        nope_head_dim,
        rope_head_dim,
        "_rocm_sparse_attn_prefill_ragged_triton",
    )

    fp8_kv_dtypes = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    fp8_kv = kv.dtype in fp8_kv_dtypes
    if fp8_kv:
        if kv.dtype != current_platform.fp8_dtype():
            raise ValueError("FP8 KV dtype must match the platform FP8 dtype")
        if q.dtype != torch.bfloat16:
            raise ValueError("FP8 sparse MLA requires a BF16 query")
        if kv_scale is None:
            raise ValueError("FP8 sparse MLA requires kv_scale")
        if kv_scale.dtype != torch.float32 or kv_scale.numel() != 1:
            raise ValueError("FP8 sparse MLA requires one float32 kv_scale value")
        if kv_scale.device != kv.device:
            raise ValueError("kv_scale must be on the KV cache device")
        kv_scale_arg = kv_scale
    else:
        if kv.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"Unsupported sparse MLA cache dtype: {kv.dtype}")
        kv_scale_arg = q  # Unused in the non-FP8 specialization.

    block_h = 16
    block_d = triton.next_power_of_2(head_dim)
    block_k = 16 if head_dim >= 256 else 32
    # The gfx1201 FP8 decoder with H=16/D=512 spills 127 registers at
    # four warps. Eight warps reduce that to 24 and preserve the reduction
    # order (BLOCK_K=16). Keep the measured choice local to short decode
    # batches; BF16, larger batches and other architectures keep defaults.
    fp8_rdna_decode = (
        _ON_RDNA4
        and fp8_kv
        and 0 < num_queries <= 8
        and num_heads == 16
        and head_dim == 512
        and rope_head_dim == 0
    )
    num_warps = 8 if fp8_rdna_decode else 4
    out = torch.empty_like(q)
    _sparse_attn_prefill_ragged_kernel[(num_queries, triton.cdiv(num_heads, block_h))](
        q,
        kv,
        kv_scale_arg,
        indices,
        indptr,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        head_dim,
        kv.shape[0],
        float(scale),
        FP8_KV=fp8_kv,
        HAS_ATTN_SINK=has_attn_sink,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return out

def _rocm_sparse_attn_prefill_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
    nope_head_dim: int,
    rope_head_dim: int,
    topk_length: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
        indices,
        topk_length
        if topk_length is not None
        else (indices >= 0).sum(dim=-1, dtype=torch.int32),
        num_rows=kv.shape[0],
    )
    return _rocm_sparse_attn_prefill_ragged_triton(
        q=q,
        kv=kv,
        indices=ragged_indices,
        indptr=ragged_indptr,
        scale=scale,
        attn_sink=attn_sink,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        kv_scale=kv_scale,
    )

def rocm_sparse_attn_prefill(q, kv, indices, scale, head_dim, nope_head_dim,
                           rope_head_dim, attn_sink, output, topk_length=None,
                           ragged_indices=None, ragged_indptr=None, kv_scale=None):
    if kv.ndim != 3 or kv.shape[1] != 1:
        raise ValueError("Expected KV [rows,1,head_dim]")
    if ragged_indices is not None and ragged_indptr is not None:
        out = _rocm_sparse_attn_prefill_ragged_triton(
            q, kv.squeeze(1), ragged_indices, ragged_indptr, scale,
            None if attn_sink is None else attn_sink[:q.shape[1]],
            nope_head_dim, rope_head_dim, kv_scale)
    else:
        if indices is None:
            raise ValueError("Dense or ragged indices required")
        out = _rocm_sparse_attn_prefill_triton(
            q, kv.squeeze(1), indices.reshape(indices.shape[0], -1), scale,
            None if attn_sink is None else attn_sink[:q.shape[1]],
            nope_head_dim, rope_head_dim, topk_length, kv_scale)
    output.copy_(out[..., :output.shape[-1]].to(output.dtype))
