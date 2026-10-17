# SPDX-License-Identifier: Apache-2.0
"""Expand selected KPool groups through a shared page table without copying it.

Semantics follow pinned SGLang kpool_fp8_index.py's group expansion and tail.
Only the final integer mapping changes; score computation/top-k stay upstream.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mask_scores(LOGITS, LENGTHS, STARTS, OUT, LS0, LS1, LENGTH_S, START_S,
                 COLS, OUT_COLS, HAS_STARTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(LENGTHS + row * LENGTH_S).to(tl.int64)
    start = tl.load(STARTS + row * START_S).to(tl.int64) if HAS_STARTS else 0
    source = start + col
    valid = (col < OUT_COLS) & (col < length) & (source >= 0) & (source < COLS)
    value = tl.load(LOGITS + row.to(tl.int64) * LS0 + source * LS1,
                    mask=valid, other=-float("inf"))
    tl.store(OUT + row.to(tl.int64) * OUT_COLS + col, value, mask=col < OUT_COLS)


def mask_pooled_logits(logits, lengths, row_starts=None, *, min_cols):
    """Exact gather/shift/mask/pad for Torch top-k, with one FP32 output only.

    Shape/stride scalars stay runtime parameters: growing prefill histories
    must not generate a new kernel variant at every step.
    """
    if (logits.ndim != 2 or logits.dtype != torch.float32 or not logits.is_cuda or
            lengths.ndim != 1 or lengths.numel() != logits.shape[0] or
            lengths.device != logits.device or lengths.dtype not in (torch.int32, torch.int64) or
            min_cols <= 0):
        raise ValueError("KPool score mask needs GPU FP32 logits and per-query integer lengths")
    if row_starts is not None and (row_starts.ndim != 1 or row_starts.numel() != logits.shape[0] or
            row_starts.device != logits.device or row_starts.dtype not in (torch.int32, torch.int64)):
        raise ValueError("KPool starts must contain one GPU integer offset per query")
    if torch.cuda.get_device_properties(logits.device).gcnArchName.split(":")[0] != "gfx1201":
        raise RuntimeError("KPool score mask is qualified for gfx1201 only")
    rows, cols = logits.shape
    out_cols = max(cols, min_cols)
    out = torch.empty((rows, out_cols), device=logits.device, dtype=torch.float32)
    if rows:
        starts = lengths if row_starts is None else row_starts
        _mask_scores[(rows, triton.cdiv(out_cols, 512))](
            logits, lengths, starts, out, logits.stride(0), logits.stride(1),
            lengths.stride(0), starts.stride(0), cols, out_cols,
            row_starts is not None, 512, num_warps=4)
    return out


@triton.jit
def _expand(G, LENGTHS, TABLE, ROW_INDEX, SEQ, OUT,
            GS0: tl.constexpr, GS1: tl.constexpr, LS: tl.constexpr,
            TS0: tl.constexpr, TS1: tl.constexpr, RS: tl.constexpr,
            SS: tl.constexpr, ROWS, GROUPS: tl.constexpr,
            LOGIT_COLS: tl.constexpr, TABLE_COLS: tl.constexpr,
            POOL: tl.constexpr, TOPK: tl.constexpr, HAS_TAIL: tl.constexpr,
            OUT_COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    if row < ROWS:
        length = tl.load(LENGTHS + row * LS).to(tl.int64)
        table_row = tl.load(ROW_INDEX + row * RS).to(tl.int64)
        group_col = col // POOL
        group = tl.load(G + row * GS0 + group_col * GS1,
                        mask=group_col < GROUPS, other=0).to(tl.int64)
        valid_groups = tl.minimum(length, min(LOGIT_COLS, GROUPS))
        history = (col < TOPK) & (group_col < valid_groups)
        raw = group * POOL + col % POOL
        if HAS_TAIL:
            history_len = tl.minimum(length * POOL, TOPK)
            history = history & (col < history_len)
            seq_len = tl.load(SEQ + row * SS).to(tl.int64)
            offset = col - history_len
            tail = (offset >= 0) & (offset < seq_len % POOL)
            raw = tl.where(tail, length * POOL + offset, raw)
            valid = history | tail
        else:
            valid = history
        safe = tl.minimum(tl.maximum(raw, 0), TABLE_COLS - 1)
        value = tl.load(TABLE + table_row * TS0 + safe * TS1,
                        mask=(col < OUT_COLS) & valid, other=-1).to(tl.int32)
    else:
        value = tl.full((BLOCK,), -1, tl.int32)
    tl.store(OUT + row * OUT_COLS + col, value, mask=col < OUT_COLS)


def expand_mapped_groups(groups, lengths, page_table, row_index, *, pool_size,
                         logit_cols, seq_lens=None, out_rows=None):
    """Bit-exact integer equivalent of expanded-table gather + optional tail.

    Workspace is O(query rows × output top-k), independent of context length.
    Row IDs are supplied by the caller's existing validated scheduling metadata.
    """
    if (groups.ndim != 2 or lengths.ndim != 1 or page_table.ndim != 2 or
            row_index.ndim != 1 or groups.shape[0] != lengths.numel() or
            row_index.numel() != groups.shape[0] or page_table.shape[1] == 0 or
            pool_size != 4 or logit_cols < 0 or
            any(t.dtype not in (torch.int32, torch.int64)
                for t in (groups, lengths, page_table, row_index)) or
            any(t.device != groups.device for t in (lengths, page_table, row_index)) or
            not groups.is_cuda):
        raise ValueError("KPool mapping needs GPU integer groups/lengths, shared table and per-query row IDs, pool_size=4")
    if torch.cuda.get_device_properties(groups.device).gcnArchName.split(":")[0] != "gfx1201":
        raise RuntimeError("Shared KPool mapping is qualified for gfx1201 only")
    rows, n_groups = groups.shape
    if seq_lens is not None and (seq_lens.ndim != 1 or seq_lens.numel() != rows or
            seq_lens.device != groups.device or seq_lens.dtype not in (torch.int32, torch.int64)):
        raise ValueError("KPool tail needs one integer sequence length per query row")
    out_rows = rows if out_rows is None else out_rows
    if out_rows < rows:
        raise ValueError("KPool output cannot have fewer rows than input")
    topk = n_groups * pool_size
    out_cols = topk + (pool_size - 1 if seq_lens is not None else 0)
    if topk == 0:
        raise ValueError("KPool group budget must be positive")
    result = torch.empty((out_rows, out_cols), device=groups.device, dtype=torch.int32)
    if out_rows:
        seq = lengths if seq_lens is None else seq_lens
        _expand[(out_rows,)](
            groups, lengths, page_table, row_index, seq, result,
            groups.stride(0), groups.stride(1), lengths.stride(0),
            page_table.stride(0), page_table.stride(1), row_index.stride(0),
            # The kernel only needs min(logit_cols, group budget). Canonicalize
            # here: otherwise every growing prefill history creates another
            # compiled/device-loaded variant even after the budget saturates.
            seq.stride(0), rows, n_groups, min(logit_cols, n_groups), page_table.shape[1],
            pool_size, topk, seq_lens is not None, out_cols,
            triton.next_power_of_2(out_cols), num_warps=4)
    return result
