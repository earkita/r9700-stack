"""GLM/DFlash views sharing the upstream global block pool.

Groups own distinct block IDs; their layer views can therefore alias the same
physical slots, as the upstream GLM MLA/Mamba groups already do. Allocation
and admission stay in upstream code. Never share IDs between live groups.
"""
from dataclasses import replace


def install_shared_cache_adapters(kv):
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, SlidingWindowSpec
    old_groups = kv._get_kv_cache_groups_glm5_next
    old_layout = kv._glm5_next_tensor_layout

    def layout(kv_cache_groups):
        draft = [g for g in kv_cache_groups if type(g.kv_cache_spec) is SlidingWindowSpec]
        target = [g for g in kv_cache_groups if type(g.kv_cache_spec) is not SlidingWindowSpec]
        base = old_layout(target)
        if base is None or not draft:
            return old_layout(kv_cache_groups)
        _, slots, mla_names, _, mla_page, *_ = base
        if any(len(g.layer_names) > len(mla_names)
               or g.kv_cache_spec.page_size_bytes != mla_page
               for g in draft):
            raise RuntimeError("Unsupported GLM shared draft cache geometry")
        return (base[0], slots + draft, *base[2:])

    def groups(vllm_config, kv_cache_spec):
        cfg = vllm_config
        spec = cfg.speculative_config
        if (getattr(cfg.model_config.hf_config, "model_type", None)
                not in ("glm5_next", "glm5_next_text")
                or spec is None or spec.method != "dflash"):
            return old_groups(cfg, kv_cache_spec)
        draft = {n: s for n, s in kv_cache_spec.items() if type(s) is SlidingWindowSpec}
        if not draft:
            return old_groups(cfg, kv_cache_spec)
        target = old_groups(cfg, {n: s for n, s in kv_cache_spec.items() if n not in draft})
        base = old_layout(target) if target is not None else None
        if base is None:
            raise RuntimeError("Unsupported GLM target cache geometry")
        attn, _, mla_names, _, mla_page, *_ = base
        block = attn.kv_cache_spec.block_size
        buckets = {}
        for name, draft_spec in draft.items():
            padded = replace(draft_spec, block_size=block, page_size_padded=None)
            if padded.page_size_bytes > mla_page:
                raise RuntimeError("GLM draft cache page exceeds shared MLA slot")
            padded = replace(padded, page_size_padded=mla_page)
            buckets.setdefault(padded, []).append(name)
        result = list(target)
        for padded, names in buckets.items():
            for start in range(0, len(names), len(mla_names)):
                result.append(KVCacheGroupSpec(names[start:start + len(mla_names)], padded,
                              is_eagle_group=spec.use_eagle_block_drop()))
        return result

    kv._get_kv_cache_groups_glm5_next = groups
    kv._glm5_next_tensor_layout = layout
