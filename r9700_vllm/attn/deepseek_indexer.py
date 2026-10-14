"""DeepSeek V4.1 FP8 index-key reader for gfx1201.

Independent copy of the stack's GLM tiled-reader algorithm. DeepSeek uses
compression ratios 1/2 and 128/64-token pages, with the same 16x16 shuffle.
Keep GLM's kernel and dispatch unchanged. FP8 -> BF16 is exact; dot products
and head reduction accumulate in FP32. This is not the main MLA KV cache.
"""
import torch
from vllm.triton_utils import tl, triton
from vllm.v1.worker.workspace import current_workspace_manager


@triton.jit
def _paged_logits(Q, K, W, LENS, TABLE, OUT,
                  Q_BATCH: tl.constexpr, Q_NEXT: tl.constexpr, Q_HEAD: tl.constexpr,
                  W_ROW: tl.constexpr, K_PAGE: tl.constexpr, TABLE_ROW: tl.constexpr,
                  MAX_LEN: tl.constexpr, TABLE_WIDTH: tl.constexpr,
                  PAGE: tl.constexpr, NEXT: tl.constexpr, PER_ROW: tl.constexpr,
                  TILE: tl.constexpr = 64):
    row = tl.program_id(0)
    batch, step = row // NEXT, row % NEXT
    pos = tl.program_id(1) * TILE + tl.arange(0, TILE)
    dim = tl.arange(0, 128)
    head = tl.arange(0, 32)
    length = tl.load(LENS + (row if PER_ROW else batch))
    if not PER_ROW:
        length = length - NEXT + step + 1
    valid = (pos < length) & (pos < MAX_LEN) & (pos // PAGE < TABLE_WIDTH)
    physical = tl.load(TABLE + batch * TABLE_ROW + pos // PAGE, valid, 0)
    offset = pos % PAGE
    # [token tile, dim tile, token within tile, dim within tile].
    address = (physical.to(tl.int64)[None, :] * K_PAGE
               + (offset[None, :] // 16) * 128 * 16
               + (dim[:, None] // 16) * 256
               + (offset[None, :] % 16) * 16 + dim[:, None] % 16)
    keys = tl.load(K + address, valid[None, :], 0.0).to(tl.bfloat16)
    scales_ptr = (K + physical.to(tl.int64) * K_PAGE + PAGE * 128).to(tl.pointer_type(tl.float32))
    scales = tl.load(scales_ptr + offset, valid, 0)
    q = tl.load(Q + batch * Q_BATCH + step * Q_NEXT
                + head[:, None] * Q_HEAD + dim[None, :]).to(tl.bfloat16)
    weights = tl.load(W + row * W_ROW + head)
    dot = tl.dot(q, keys)
    logits = tl.sum(tl.maximum(dot * scales[None, :], 0.) * weights[:, None], axis=0)
    logits = tl.where(valid, logits, -float("inf"))
    tl.store(OUT + row * MAX_LEN + pos, logits, pos < MAX_LEN)


def cache_view(cache):
    """View dense page contents while preserving padding/interleaving between pages.

    vLLM packs layer pages into a common backing allocation. Only each page's
    contents must be contiguous; K_PAGE already addresses the physical stride.
    Never compact/copy this cache: other layers alias the backing allocation.
    """
    if (cache.dtype != torch.uint8 or cache.ndim != 4
            or cache.shape[1] not in (64, 128) or cache.shape[2:] != (1, 132)
            or cache.stride(1) != 132 or cache.stride(3) != 1
            or cache.stride(0) < cache.shape[1] * 132
            or cache.stride(0) % 4 or cache.storage_offset() % 4):
        raise ValueError(
            "Unsupported DeepSeek indexer cache layout: "
            f"shape={tuple(cache.shape)}, stride={cache.stride()}, "
            f"dtype={cache.dtype}, offset={cache.storage_offset()}")
    return cache.view(cache.shape[0], -1).view(torch.float8_e4m3fn)


def paged_logits(q, cache, weights, context_lens, table, schedule_metadata,
                 max_model_len, *, compress_ratio=1):
    """Reject unaudited layouts; never silently read shuffled values as flat."""
    batch, next_n, heads, dim = q.shape
    if (heads, dim) != (32, 128) or compress_ratio not in (1, 2):
        raise ValueError("DeepSeek indexer requires H32/D128 and compression 1 or 2")
    if q.dtype != torch.float8_e4m3fn:
        raise ValueError("DeepSeek indexer requires OCP FP8 queries")
    flat = cache_view(cache)
    assert q.stride(-1) == weights.stride(-1) == table.stride(-1) == 1
    lens = context_lens.reshape(-1)
    per_row = lens.numel() == batch * next_n
    assert per_row or lens.numel() == batch
    (out,) = current_workspace_manager().get_simultaneous(
        ((batch * next_n, max_model_len), torch.float32),
    )
    _paged_logits[(batch * next_n, triton.cdiv(max_model_len, 64))](
        q, flat, weights, lens, table, out,
        *q.stride()[:3], weights.stride(0), flat.stride(0), table.stride(0),
        max_model_len, table.shape[1], cache.shape[1], next_n, per_row,
        num_warps=4, num_stages=2)
    return out
