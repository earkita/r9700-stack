"""Connector-only GLM cache views; never change vLLM allocation or kernels."""
from copy import copy
from dataclasses import replace


def select_cache_groups(config):
    """Exclude only GLM's non-cacheable circular scratch, preserving group order."""
    kept, excluded = [], []
    for i, group in enumerate(config.kv_cache_groups):
        spec = group.kv_cache_spec
        leaves = getattr(spec, 'kv_cache_specs', None) or {n: spec for n in group.layer_names}
        # Native GLM replaced KpoolTailSpec with a per-request circular ring.
        # Both are non-cacheable scratch and must be reconstructed from suffix.
        tail = [type(s).__name__ in ('KpoolTailSpec', 'CircularBufferSpec')
                for s in leaves.values()]
        if any(tail):
            if not all(tail) or any(s.prefix_cacheable for s in leaves.values()):
                raise ValueError('Unsupported mixed/cacheable GLM tail group')
            excluded.extend(group.layer_names)
        else:
            if not spec.prefix_cacheable or not group.enable_kv_transfer:
                raise ValueError('Unsupported non-transferable GLM cache group')
            kept.append(i)
    if not excluded or not kept or kept[0] != 0:
        raise ValueError('Expected GLM MLA group followed by circular tail scratch')
    filtered = replace(config, kv_cache_groups=[config.kv_cache_groups[i] for i in kept])
    return filtered, tuple(kept), frozenset(excluded)


def select_block_ids(blocks, kept, original_count):
    if not blocks:
        return blocks
    if len(blocks) != original_count:
        raise ValueError(f'Expected {original_count} GLM block groups, got {len(blocks)}')
    return tuple(blocks[i] for i in kept)


def select_scheduler_output(output, kept, original_count):
    """Copy only containers we edit; the scheduler's allocation stays untouched."""
    result = copy(output)
    result.scheduled_new_reqs = []
    for request in output.scheduled_new_reqs:
        row = copy(request)
        row.block_ids = select_block_ids(row.block_ids, kept, original_count)
        result.scheduled_new_reqs.append(row)
    cached = result.scheduled_cached_reqs = copy(output.scheduled_cached_reqs)
    cached.new_block_ids = [select_block_ids(b, kept, original_count)
                            for b in cached.new_block_ids]
    return result


def logical_mla_view(tensor, spec, layout='NHD'):
    """Collapse kernel pages into logical pages, keeping real kpool compression.

    GLM groups contain both MLA (one state/token) and indexer (one state/four
    tokens). Their 128/32-state kernel pages must be grouped into the logical
    block span supplied by vLLM (1152 on e975, 1280 on main206) before
    LMCache uses scheduler block IDs. The view preserves every byte.
    """
    states = spec.block_size // spec.tokens_per_state
    if spec.block_size % spec.tokens_per_state or tensor.ndim not in (3, 4):
        raise ValueError('Unsupported GLM MLA state geometry')
    layout = {'BLHNC': 'HND', 'BLNHC': 'NHD'}.get(layout, layout)
    if layout not in ('NHD', 'HND'):
        raise ValueError(f'Unsupported GLM cache layout: {layout}')
    token_axis = 2 if tensor.ndim == 4 and layout == 'HND' else 1
    if tensor.ndim == 4 and tensor.shape[3-token_axis] != 1:
        raise ValueError('GLM MLA external cache requires one latent head')
    kernel_states = tensor.shape[token_axis]
    if states % kernel_states:
        raise ValueError('GLM MLA kernel pages must tile contiguous logical pages: '
                         f'shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}, '
                         f'block_size={spec.block_size}, tokens_per_state={spec.tokens_per_state}, '
                         f'layout={layout}, contiguous={tensor.is_contiguous()}')
    ratio = states // kernel_states
    if tensor.shape[0] % ratio:
        raise ValueError(f'GLM MLA kernel page count is not block aligned: '
                         f'shape={tuple(tensor.shape)}, states={states}, layout={layout}')
    page_bytes = tensor[0].numel() * tensor.element_size() * ratio
    if page_bytes != spec.page_size_bytes:
        raise ValueError('GLM MLA view would cross the declared layer page')
    # BLHNC interleaves complete layers between manager blocks. One layer's
    # block-to-block stride is therefore larger than its page, while each
    # page itself remains dense. LMCache understands this physical layout and
    # copies one page at a time. Preserve the original strided tensor when
    # vLLM already presents manager-sized pages; torch.view would reject it.
    if ratio == 1:
        if not tensor[0].is_contiguous() or tensor.stride(0) < tensor[0].numel():
            raise ValueError('GLM MLA page must be dense and non-overlapping')
        return tensor
    if not tensor.is_contiguous():
        raise ValueError('Split GLM MLA kernel pages require a contiguous layer view: '
                         f'shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}, '
                         f'block_size={spec.block_size}, layout={layout}')
    shape = list(tensor.shape)
    shape[0] //= ratio
    shape[token_axis] = states
    return tensor.view(*shape)
