"""Materialize the last reusable GLM/DFlash prefill boundary.

Installed only through the gfx1201/e97573215 gated GLM adapter. Keep the
upstream splitter in charge of all other stops, including resumed requests.
"""
from functools import wraps


def install_boundary_adapter(scheduler_cls):
    original = scheduler_cls._mamba_block_aligned_split
    if getattr(original, "_r9700_glm_boundary", False):
        return

    @wraps(original)
    def split(self, request, num_new_tokens, num_new_local_computed_tokens=0,
              num_external_computed_tokens=0):
        result = original(self, request, num_new_tokens,
                          num_new_local_computed_tokens, num_external_computed_tokens)
        cfg = self.vllm_config
        spec = cfg.speculative_config
        if (getattr(cfg.model_config.hf_config, "model_type", None)
                not in ("glm5_next", "glm5_next_text")
                or spec is None or spec.method != "dflash"
                or self.use_eagle_block_drop):
            return result
        start = (request.num_computed_tokens + num_new_local_computed_tokens
                 + num_external_computed_tokens)
        block = self.cache_config.block_size
        boundary = (request.num_tokens - 1) // block * block
        if start < boundary < start + result:
            return boundary - start
        return result

    split._r9700_glm_boundary = True
    scheduler_cls._mamba_block_aligned_split = split
