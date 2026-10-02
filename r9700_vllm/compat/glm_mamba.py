"""GLM V2 APC state seed, vllm-project/vllm#55601 (pinned backport).

Keep the upstream request initialization, then seed the state column in Mamba
block units. Attention and Mamba pages have different token spans on GLM.
"""
from functools import wraps


def _is_glm(state):
    return state.model_config.hf_config.model_type == "glm5_next"


def _wrap_add_request(original):
    @wraps(original)
    def add_request(self, req_index, new_req_data):
        original(self, req_index, new_req_data)
        if _is_glm(self) and self._align_mode:
            size = self.cache_config.mamba_block_size
            if not isinstance(size, int) or size <= 0:
                raise RuntimeError("GLM APC needs a positive Mamba block size")
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // size
            )
    add_request._r9700_glm_mamba = True
    return add_request


def _wrap_group_info(original, log):
    @wraps(original)
    def group_info(self, kv_cache_config):
        first = self._mamba_spec is None
        groups, spec = original(self, kv_cache_config)
        if _is_glm(self) and first:
            if spec.block_size != self.cache_config.mamba_block_size:
                raise RuntimeError("GLM Mamba seed and effective cache spec use different block sizes")
            log.info("r9700: GLM APC state seed: attention=%d, mamba=%d, effective_mamba=%d",
                     self.cache_config.block_size, self.cache_config.mamba_block_size,
                     spec.block_size)
        return groups, spec
    return group_info


def patch() -> bool:
    import inspect
    import os
    import torch
    from vllm.logger import init_logger
    from vllm.platforms import current_platform

    if os.environ.get("R9K_GLM_MAMBA", "aligned") == "stock" or not current_platform.is_rocm():
        return False
    if torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0] != "gfx1201":
        return False
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
    cls = MambaHybridModelState
    if getattr(cls.add_request, "_r9700_glm_mamba", False):
        return True
    from .gate import check, vllm_commit
    source = inspect.getsource(cls.add_request)
    if not vllm_commit().startswith("e97573215") or source.count("// self.cache_config.block_size") != 1:
        raise RuntimeError("GLM Mamba adapter requires audited vLLM e97573215; revalidate the new pin")
    check("glm_mamba_seed")
    log = init_logger("vllm.r9700_vllm")
    cls.add_request = _wrap_add_request(cls.add_request)
    cls._get_mamba_group_info = _wrap_group_info(cls._get_mamba_group_info, log)
    log.info("r9700: GLM V2 APC Mamba seed uses Mamba block units (#55601)")
    return True
