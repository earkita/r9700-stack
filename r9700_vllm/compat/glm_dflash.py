"""Opt-in GLM DFlash interfaces and independent draft KV allocation.

Adapted allocation semantics from vllm-project/vllm#55423/#56983. Keep
upstream GLM target aliases and add a separately accounted draft region.
"""
from dataclasses import replace
from functools import wraps


def install_draft_projection():
    """Shard only GLM's BF16 5-tap projection, before checkpoint loading."""
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
    from vllm.model_executor.layers.linear import RowParallelLinear
    original = DFlashQwen3Model.__init__

    @wraps(original)
    def init(self, *, vllm_config, start_layer_id=0, prefix=""):
        original(self, vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix)
        if (getattr(vllm_config.model_config.hf_config, "model_type", None) == "glm5_next"
                and self.use_aux_hidden_state and self.fc.weight.shape == (4096, 20480)
                and self.quant_config is None):
            self.fc = RowParallelLinear(
                input_size=20480, output_size=4096, bias=False, return_bias=False,
                input_is_parallel=False, reduce_results=True,
                params_dtype=vllm_config.model_config.dtype, quant_config=None,
                prefix=f"{prefix}.fc" if prefix else "fc")
            from vllm.logger import init_logger
            init_logger("vllm.r9700_vllm").info("r9700: GLM DFlash auxiliary FC is row-parallel: %s", self.fc.weight.shape)

    DFlashQwen3Model.__init__ = init


def install_tail_adapter():
    from vllm.models.glm5next.common.attention import Glm5NextTailCache
    from vllm.models.glm5next.amd.ops import kpool_compress as upstream
    from ..attn import glm_tail
    original = Glm5NextTailCache.get_kv_cache_spec

    def get_spec(self, vllm_config):
        spec = original(self, vllm_config)
        pool = self._index_kpool
        span = pool + vllm_config.num_speculative_tokens
        ring = ((span + pool - 1) // pool) * pool
        return replace(spec, block_size=ring, sliding_window=ring)

    Glm5NextTailCache.get_kv_cache_spec = get_spec
    upstream.kpool_seed_tail_cache = glm_tail.kpool_seed_tail_cache
    upstream.kpool_decode_update_and_maybe_write_cache_batched = (
        glm_tail.kpool_decode_update_and_maybe_write_cache_batched)
    upstream.expand_pools_and_append_tail = glm_tail.expand_pools_and_append_tail


def install_cache_adapters(kv):
    from vllm.v1.kv_cache_interface import (
        AttentionSpec, KVCacheGroupSpec, KVCacheTensor, KpoolTailSpec,
        MLAAttentionSpec, UniformTypeKVCacheSpecs,
    )
    old_groups = kv._get_kv_cache_groups_glm5_next
    old_bytes = kv._get_kv_cache_bytes_per_block
    old_config = kv.get_kv_cache_config_from_groups
    old_max = kv._max_memory_usage_bytes_from_groups

    def plain(spec):
        return isinstance(spec, AttentionSpec) and not isinstance(spec, (MLAAttentionSpec, KpoolTailSpec))

    def split(groups):
        foreign = [g for g in groups if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
                   and all(plain(s) for s in g.kv_cache_spec.kv_cache_specs.values())]
        target = [g for g in groups if not any(g is f for f in foreign)]
        if len(foreign) != 1 or kv._glm5_next_tensor_layout(target) is None:
            return None
        return target, foreign[0]

    def foreign_bytes(group):
        return sum(s.page_size_bytes for s in group.kv_cache_spec.kv_cache_specs.values())

    def groups(vllm_config, kv_cache_spec):
        config, specs = vllm_config, kv_cache_spec
        spec_cfg = config.speculative_config
        if (getattr(config.model_config.hf_config, "model_type", None) not in ("glm5_next", "glm5_next_text")
                or spec_cfg is None or spec_cfg.method != "dflash"):
            return old_groups(config, specs)
        foreign = {name: s for name, s in specs.items() if plain(s)}
        target = {name: s for name, s in specs.items() if name not in foreign}
        if not foreign:
            return old_groups(config, specs)
        target_groups = old_groups(config, target)
        uniform = UniformTypeKVCacheSpecs.from_specs(foreign)
        if target_groups is None or uniform is None:
            raise RuntimeError("Unsupported GLM/DFlash cache group geometry")
        return target_groups + [KVCacheGroupSpec(list(foreign), uniform,
                                is_eagle_group=spec_cfg.use_eagle_block_drop())]

    def bytes_per_block(kv_cache_groups):
        groups = kv_cache_groups
        parts = split(groups)
        return old_bytes(groups) if parts is None else old_bytes(parts[0]) + foreign_bytes(parts[1])

    def allocate(vllm_config, kv_cache_groups, available_memory):
        config, groups = vllm_config, kv_cache_groups
        parts = split(groups)
        if parts is None:
            return old_config(config, groups, available_memory)
        target, draft = parts
        target_bytes = old_bytes(target)
        total_bytes = target_bytes + foreign_bytes(draft)
        blocks = kv.may_override_num_blocks(config, available_memory // total_bytes)
        result = old_config(config, target, blocks * target_bytes)
        assert result.num_blocks == blocks
        size = blocks * total_bytes
        tensors = [replace(t, size=size) for t in result.kv_cache_tensors]
        offset = blocks * target_bytes
        for name in draft.layer_names:
            page = draft.kv_cache_spec.kv_cache_specs[name].page_size_bytes
            tensors.append(KVCacheTensor(size=size, layers=[name], layer_stride=page * blocks,
                                         block_stride=page, offset=offset))
            offset += page * blocks
        assert offset == size
        return replace(result, kv_cache_tensors=tensors, kv_cache_groups=groups)

    def max_memory(vllm_config, kv_cache_groups):
        config, groups = vllm_config, kv_cache_groups
        parts = split(groups)
        if parts is None:
            return old_max(config, groups)
        target, draft = parts
        target_bytes = old_bytes(target)
        target_required = old_max(config, target)
        assert target_required % target_bytes == 0
        blocks = target_required // target_bytes + draft.kv_cache_spec.max_memory_usage_pages(config)
        return blocks * (target_bytes + foreign_bytes(draft))

    kv._get_kv_cache_groups_glm5_next = groups
    kv._get_kv_cache_bytes_per_block = bytes_per_block
    kv.get_kv_cache_config_from_groups = allocate
    kv._max_memory_usage_bytes_from_groups = max_memory


def patch():
    import os
    if os.environ.get("R9K_GLM_DFLASH", "0") == "0":
        return False
    if os.environ["R9K_GLM_DFLASH"] != "1":
        raise ValueError("R9K_GLM_DFLASH must be 0 or 1")
    shard_fc = os.environ.get("R9K_GLM_DFLASH_SHARD_FC", "0")
    if shard_fc not in ("0", "1"):
        raise ValueError("R9K_GLM_DFLASH_SHARD_FC must be 0 or 1")
    import torch
    from vllm import ModelRegistry
    from vllm.platforms import current_platform
    from vllm.v1.core import kv_cache_utils as kv
    from .gate import check, vllm_commit
    if (not current_platform.is_rocm()
            or torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0] != "gfx1201"
            or not vllm_commit().startswith("e97573215")):
        raise RuntimeError("GLM DFlash adapter requires gfx1201 and audited vLLM e97573215")
    if getattr(kv, "_r9700_glm_dflash", False):
        return True
    check("glm_dflash")
    install_tail_adapter()
    if shard_fc == "1":
        install_draft_projection()
    if os.environ.get("R9K_GLM_DFLASH_AUDIT_DIR"):
        from .glm_dflash_audit import install_draft_capture
        install_draft_capture()
    install_cache_adapters(kv)
    for arch, cls in [("Glm5NextForCausalLM", "R9kGlmDFlashForCausalLM"),
                      ("Glm5NextForConditionalGeneration", "R9kGlmDFlashForConditionalGeneration")]:
        ModelRegistry.register_model(arch, f"r9700_vllm.models.glm_dflash:{cls}")
    kv._r9700_glm_dflash = True
    from vllm.logger import init_logger
    init_logger("vllm.r9700_vllm").info("r9700: experimental GLM DFlash aux-state and independent draft-cache adapters")
    return True
