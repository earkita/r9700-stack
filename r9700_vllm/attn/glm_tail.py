# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned AMD kpool kernels with a speculation-safe tail ring.

Adapted from vLLM e97573215 and my-llm's v0.32 ring fix. Pool compression
math/storage stay upstream; only tail addressing uses the allocated ring
capacity instead of the number of tokens per compressed pool. Keeping
committed keys plus every speculative row prevents rejection rollback from
reading keys overwritten by later rejected proposals.
"""
import torch
from vllm.triton_utils import triton, tl
from vllm.models.glm5next.amd.ops.kpool_compress import (
    INDEX_HEAD_DIM, FP8_DTYPE, FP8_MAX, _hadamard128, _cache_k_offset,
)

@triton.jit
def _kpool_tail_seed_kernel(
    key_ptr,
    score_ptr,
    tslot_ptr,
    tail_ptr,
    n_tokens,
    TAIL_BLOCK_ELEMS: tl.constexpr,
    KPOOL_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KPOOL: tl.constexpr,
    RING: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Copy token ``i``'s raw K + gate into its request's tail block.

    Token ``i`` is among its request's last KPOOL tokens iff the token KPOOL
    ahead belongs to a different tail block (or is past the batch / padding,
    slot < 0). ``tslot = block * RING + pos % RING``; the destination is
    ``tail[block, {0:K, 1:score}, pos % RING, :]``.
    """
    i = tl.program_id(0)
    t = tl.load(tslot_ptr + i).to(tl.int64)
    if t < 0:
        return
    blk = t // RING  # t >= 0 here, so trunc == floor
    ahead = tl.load(tslot_ptr + i + KPOOL, mask=i + KPOOL < n_tokens, other=-1).to(
        tl.int64
    )
    # Match the torch semantics exactly: a negative ahead slot floors to a
    # block id that differs from every real block -> token is in the tail.
    # Only divide non-negative slots (Triton int div truncates, torch floors).
    if ahead >= 0 and ahead // RING == blk:
        return
    offs = tl.arange(0, BLOCK_D)
    m = offs < HEAD_DIM
    block_base = blk * TAIL_BLOCK_ELEMS
    base = block_base + (t % RING) * HEAD_DIM
    k = tl.load(key_ptr + i * HEAD_DIM + offs, mask=m)
    s = tl.load(score_ptr + i * HEAD_DIM + offs, mask=m)
    tl.store(tail_ptr + base + offs, k, mask=m)
    tl.store(
        tail_ptr + block_base + KPOOL_HEAD + (t % RING) * HEAD_DIM + offs, s, mask=m
    )


def kpool_seed_tail_cache(
    tail_kv_cache: torch.Tensor,
    key: torch.Tensor,
    gate_score: torch.Tensor,
    tslot: torch.Tensor,
    kpool: int,
    head_dim: int = INDEX_HEAD_DIM,
) -> None:
    """Seed the paged tail cache from a prefill batch (see the kernel)."""
    assert tail_kv_cache.dtype == torch.bfloat16
    assert key.dtype == torch.bfloat16
    n = tslot.shape[0]
    if n == 0:
        return
    _kpool_tail_seed_kernel[(n,)](
        key,
        gate_score,
        tslot,
        tail_kv_cache,
        n,
        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),
        KPOOL_HEAD=tail_kv_cache.stride(1),
        HEAD_DIM=head_dim,
        KPOOL=kpool,
        RING=tail_kv_cache.shape[2],
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


# Update each request's tail during decode and write completed pools.


@triton.jit
def _kpool_decode_update_batched_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    tail_kv_ptr,
    tail_slot_mapping_ptr,  # [B, NEXT_N] int32
    key_ptr,  # [B, NEXT_N, HEAD_DIM] bf16
    key_stride_b,
    key_stride_t,
    slot_score_ptr,  # [B, NEXT_N, HEAD_DIM] bf16
    ss_stride_b,
    ss_stride_t,
    ape_ptr,
    ape_stride_0,
    slot_mapping_ptr,  # [B, NEXT_N] int32
    positions_ptr,  # [B, NEXT_N] int32
    NEXT_N,  # runtime token count per request (no .item() needed)
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    RING: tl.constexpr,
    TAIL_BLOCK_ELEMS: tl.constexpr,
    KPOOL_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    PRESHUFFLE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per request; iterates its NEXT_N verify tokens in order.

    Replaces the caller's per-token sequential launch loop. The intra-request
    iteration MUST stay in position order: a pool-completion at token t* reads
    the tail-ring slots that tokens t < t* (same request) just stashed in this
    same invocation. ``tl.range`` iterates sequentially within the program, so
    those stashes are visible to the later completion read. Cross-request
    programs are independent (distinct tail blocks). With NEXT_N < POOL_SIZE
    (the spec-verify case: NEXT_N ~= num_spec+1, POOL_SIZE=16) at most one
    completion can occur per request per call, but the ordered loop is correct
    for any NEXT_N.
    """
    req = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    dim_mask = offs < HEAD_DIM

    for t in tl.range(0, NEXT_N):
        idx = req * NEXT_N + t
        cache_loc = tl.load(slot_mapping_ptr + idx)
        pos = tl.load(positions_ptr + idx)
        safe_pos = tl.maximum(pos, 0)
        pos_valid = (cache_loc >= 0) & (pos >= 0)

        slot = safe_pos % POOL_SIZE
        phys_slot = safe_pos % RING

        # Derive the tail block from THIS token's tail_slot (the request's block
        # is constant across a pool, but a padded / invalid entry carries a
        # negative sentinel -- reading it from token 0 would poison every
        # token's base address). Clamp so an invalid entry can never form an
        # out-of-bounds base; the accesses below are gated on pos_valid anyway.
        tail_slot = tl.load(tail_slot_mapping_ptr + idx)
        block = tl.maximum(tail_slot, 0).to(tl.int64) // RING
        block_base = block * TAIL_BLOCK_ELEMS

        # The tail-ring stash must run for EVERY real token, so it is gated on
        # the token-granular tail slot -- not on `pos_valid`, which keys off the
        # POOL-granular `slot_mapping` and is therefore only true on the pool's
        # last token. Gating the stash on pos_valid dropped every intra-pool
        # token, so a decode-built pool compressed 3 stale ring entries (the
        # prefill-seeded prompt tail, frozen forever) plus the current token.
        stash_valid = (pos >= 0) & (tail_slot >= 0)

        key = tl.load(
            key_ptr + req * key_stride_b + t * key_stride_t + offs,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        score_current = tl.load(
            slot_score_ptr + req * ss_stride_b + t * ss_stride_t + offs,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)

        if pos_valid & (slot == POOL_SIZE - 1):
            pool_logical_start = safe_pos - slot

            max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
            for pool_slot in tl.static_range(0, POOL_SIZE):
                is_current = pool_slot == slot
                phys = (pool_logical_start + pool_slot) % RING
                score_buf = tl.load(
                    tail_kv_ptr + block_base + KPOOL_HEAD + phys * HEAD_DIM + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                score = tl.where(is_current, score_current, score_buf)
                score += tl.load(
                    ape_ptr + pool_slot * ape_stride_0 + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                max_score = tl.maximum(max_score, score)

            acc = tl.full((BLOCK_D,), 0.0, tl.float32)
            denom = tl.full((BLOCK_D,), 0.0, tl.float32)
            for pool_slot in tl.static_range(0, POOL_SIZE):
                is_current = pool_slot == slot
                phys = (pool_logical_start + pool_slot) % RING
                score_buf = tl.load(
                    tail_kv_ptr + block_base + KPOOL_HEAD + phys * HEAD_DIM + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                score = tl.where(is_current, score_current, score_buf)
                score += tl.load(
                    ape_ptr + pool_slot * ape_stride_0 + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                prob = tl.exp(score - max_score)
                denom += prob
                k_buf = tl.load(
                    tail_kv_ptr + block_base + phys * HEAD_DIM + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                k = tl.where(is_current, key, k_buf)
                acc += k * prob

            x = (acc / denom).to(tl.bfloat16).to(tl.float32)
            x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)

            fp8_max_inv = 1.0 / FP8_MAX
            absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
            if ROUND_SCALE:
                scale = tl.exp2(tl.ceil(tl.log2(absmax * fp8_max_inv)))
            else:
                scale = absmax * fp8_max_inv
            quantized = tl.minimum(tl.maximum(x / scale, -FP8_MAX), FP8_MAX)

            loc = cache_loc.to(tl.int64)
            loc_page_index = loc // PAGE_SIZE
            loc_token_offset_in_page = loc % PAGE_SIZE
            out_k_offsets = loc_page_index * BUF_NUMEL_PER_PAGE + _cache_k_offset(
                loc_token_offset_in_page,
                offs,
                HEAD_DIM,
                PRESHUFFLE,
            )
            out_s_offset = (
                loc_page_index * BUF_NUMEL_PER_PAGE // 4
                + S_OFFSET_NBYTES_IN_PAGE // 4
                + loc_token_offset_in_page
            )
            tl.store(buf_fp8_ptr + out_k_offsets, quantized, mask=dim_mask)
            tl.store(buf_fp32_ptr + out_s_offset, scale)

        # Stash the current token AFTER any completion read so the completion
        # uses prior stashes (and the current token's own key/score via
        # is_current), then leaves this token for future pools. Order matches
        # the per-token kernel: completion read first, stash second.
        update_mask = dim_mask & stash_valid
        tl.store(
            tail_kv_ptr + block_base + phys_slot * HEAD_DIM + offs,
            key,
            mask=update_mask,
        )
        tl.store(
            tail_kv_ptr + block_base + KPOOL_HEAD + phys_slot * HEAD_DIM + offs,
            score_current,
            mask=update_mask,
        )


def kpool_decode_update_and_maybe_write_cache_batched(
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    pool_size: int,
    head_dim: int = INDEX_HEAD_DIM,
    round_scale: bool = True,
) -> None:
    """Batched decode-step kpool update for spec verify (``next_n > 1``).

    One launch replaces the caller's per-token loop. Inputs are grouped per
    request: ``[num_requests, next_n, ...]``. Each program handles one
    request's ``next_n`` tokens in position order (see the kernel docstring for
    why ordering is required for pool-completion correctness).

    Plain decode (``next_n == 1``) is handled here too — the kernel collapses
    to a single-iteration loop.

    Args:
        kv_cache: indexer K cache ``[num_blocks, block_size, head_dim+4]`` uint8.
        tail_kv_cache: paged tail cache ``[num_blocks, 2, ring_size, head_dim]``
            bf16 (K at half 0, gate score at half 1).
        tail_slot_mapping: ``[num_requests, next_n]`` int32.
        key: ``[num_requests, next_n, head_dim]`` bf16.
        slot_score: ``[num_requests, next_n, head_dim]`` bf16.
        ape: ``[pool_size, head_dim]`` fp32.
        slot_mapping: ``[num_requests, next_n]`` int32.
        positions: ``[num_requests, next_n]`` int32.
        pool_size: Number of tokens compressed into one cache entry.
        head_dim: Indexer head dimension.
        round_scale: Round each fp8 scale down to a power of two.

    """
    num_requests, next_n = key.shape[0], key.shape[1]
    if num_requests == 0 or next_n == 0:
        return
    assert tail_kv_cache.ndim == 4
    assert tail_kv_cache.shape[1] == 2
    assert tail_kv_cache.shape[2] >= pool_size and tail_kv_cache.shape[2] % pool_size == 0
    assert tail_kv_cache.shape[3] == head_dim
    assert tail_kv_cache.dtype == torch.bfloat16
    assert key.ndim == 3 and key.shape[2] == head_dim
    assert slot_score.shape == key.shape
    assert ape.shape == (pool_size, head_dim)
    assert tail_slot_mapping.shape == (num_requests, next_n)
    assert slot_mapping.shape == (num_requests, next_n)
    assert positions.shape == (num_requests, next_n)
    assert key.dtype == torch.bfloat16
    assert slot_score.dtype == torch.bfloat16
    assert ape.dtype == torch.float32
    assert kv_cache.dtype == torch.uint8

    page_size = kv_cache.shape[1]
    buf = kv_cache
    buf_fp8 = buf.view(FP8_DTYPE)
    buf_fp32 = buf.view(torch.float32)

    # The kernel indexes the int tensors as ``req * next_n + t`` (row-major),
    # so they must be contiguous. Callers pass either a view of a contiguous
    # slice or a freshly scattered tensor, making these no-ops; the calls guard
    # against a future caller handing over a strided view.
    tail_slot_mapping = tail_slot_mapping.contiguous()
    slot_mapping = slot_mapping.contiguous()
    positions = positions.contiguous()

    if page_size > 1:
        assert page_size % 16 == 0, "ROCm preshuffle requires 16-token tiles"

    _kpool_decode_update_batched_kernel[(num_requests,)](
        buf_fp8,
        buf_fp32,
        tail_kv_cache,
        tail_slot_mapping,
        key,
        key.stride(0),
        key.stride(1),
        slot_score,
        slot_score.stride(0),
        slot_score.stride(1),
        ape,
        ape.stride(0),
        slot_mapping,
        positions,
        next_n,
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf.stride(0),
        POOL_SIZE=pool_size,
        RING=tail_kv_cache.shape[2],
        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),
        KPOOL_HEAD=tail_kv_cache.stride(1),
        HEAD_DIM=head_dim,
        S_OFFSET_NBYTES_IN_PAGE=page_size * head_dim,
        FP8_MAX=FP8_MAX,
        PRESHUFFLE=page_size > 1,
        ROUND_SCALE=round_scale,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


@triton.jit
def _expand_pools_and_append_tail_kernel(
    pool_ids_ptr,  # [rows, n_groups], int (any int dtype)
    seq_lens_ptr,  # [rows], int32 (token-granular seq_len)
    out_ptr,  # [rows, out_cols], int32
    topk,  # n_groups * pool_size
    out_cols,  # topk + pool_size - 1
    POOL_SIZE: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    pid_s0,
    out_s0,
):
    # Fuses expand_pools_to_tokens + append_tail_to_topk (identity path) into a
    # single kernel. Each program writes one (row, column-tile) of the output.
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = cols < out_cols

    seq_len = tl.load(seq_lens_ptr + row)
    pool_len = seq_len // POOL_SIZE
    tail_start = pool_len * POOL_SIZE
    tail_count = seq_len - tail_start  # in [0, POOL_SIZE)

    # History region [0, topk): expand selected pool g = cols // POOL_SIZE.
    is_history = cols < topk
    g = cols // POOL_SIZE
    o = cols % POOL_SIZE
    pid = tl.load(pool_ids_ptr + row * pid_s0 + g, mask=mask & is_history, other=-1)
    hist_val = (pid * POOL_SIZE + o).to(tl.int32)
    hist_out = tl.where((pid >= 0) & (pid < pool_len), hist_val, -1)

    # Tail region [topk, out_cols): the request's trailing incomplete pool.
    tail_off = cols - topk
    is_tail = (tail_off >= 0) & (tail_off < tail_count)
    tail_val = (tail_start + tail_off).to(tl.int32)
    tail_out = tl.where(is_tail, tail_val, -1)

    result = tl.where(is_history, hist_out, tail_out)
    tl.store(out_ptr + row * out_s0 + cols, result, mask=mask)


def expand_pools_and_append_tail(
    pool_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Fuse ``expand_pools_to_tokens`` + ``append_tail_to_topk`` (identity path).

    Produces the same ``[rows, topk + pool_size - 1]`` int32 output as calling
    the two functions in sequence when neither ``page_table`` nor
    ``topk_offsets`` is passed — the only path used by the GLM-5.3-Flash indexer.
    The kernel derives ``pool_len = seq_len // pool_size`` internally, so the
    caller no longer needs to precompute it. Replaces ~25 elementwise kernels
    with one Triton launch.
    """
    rows, n_groups = pool_ids.shape
    topk = n_groups * pool_size
    out_cols = topk + pool_size - 1
    out = torch.empty((rows, out_cols), dtype=torch.int32, device=pool_ids.device)
    BLOCK_COLS = 128
    n_tiles = triton.cdiv(out_cols, BLOCK_COLS)
    _expand_pools_and_append_tail_kernel[(rows, n_tiles)](
        pool_ids,
        seq_lens,
        out,
        topk,
        out_cols,
        POOL_SIZE=pool_size,
        BLOCK_COLS=BLOCK_COLS,
        pid_s0=pool_ids.stride(0),
        out_s0=out.stride(0),
    )
    return out
